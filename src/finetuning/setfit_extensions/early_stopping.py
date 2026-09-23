import inspect
import numpy as np
from torch import nn

import torch
from setfit import SetFitModel, Trainer, TrainingArguments

from datasets import Dataset
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import set_seed, PredictionOutput
from transformers import EarlyStoppingCallback

# for subclassing of hyperparameter search logic
import optuna
from transformers.trainer_utils import HPSearchBackend, default_compute_objective, set_seed
from setfit.integrations import default_hp_search_backend, is_optuna_available, run_hp_search_optuna
from setfit.utils import BestRun, default_hp_space_optuna



from tqdm import tqdm, trange
import warnings

from copy import copy, deepcopy
import inspect
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Callable, Any, Tuple

from setfit import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

class SetFitModelWithEarlyStopping(SetFitModel):
    """
    A SetFit model with early stopping functionality.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_log_history: Optional[List[Dict[str, float]]] = None
    
    def fit(
        self,
        x_train: List[str],
        y_train: Union[List[int], List[List[int]]],
        x_eval: Optional[List[str]] = None,
        y_eval: Optional[Union[List[int], List[List[int]]]] = None,
        num_epochs: int = 1,
        batch_size: Optional[int] = None,
        body_learning_rate: Optional[float] = None,
        head_learning_rate: Optional[float] = None,
        end_to_end: bool = False,
        l2_weight: Optional[float] = None,
        max_length: Optional[int] = None,
        show_progress_bar: bool = True,
        # Early stopping parameters
        early_stopping_patience: Optional[int] = None,
        early_stopping_threshold: float = 0.0,
        metric_for_best_model: str = "loss",
        compute_metrics: Optional[callable] = None,
        greater_is_better: bool = False,
        load_best_model_at_end: bool = True,
    ) -> None:
        """Train the classifier head, only used if a differentiable PyTorch head is used.

        Args:
            x_train (`List[str]`): A list of training sentences.
            y_train (`Union[List[int], List[List[int]]]`): A list of labels corresponding to the training sentences.
            num_epochs (`int`): The number of epochs to train for.
            x_eval (`List[str]`, *optional*): A list of validation sentences.
            y_eval (`Union[List[int], List[List[int]]]`, *optional*): A list of labels corresponding to the validation sentences.
            batch_size (`int`, *optional*): The batch size to use.
            body_learning_rate (`float`, *optional*): The learning rate for the `SentenceTransformer` body
                in the `AdamW` optimizer. Disregarded if `end_to_end=False`.
            head_learning_rate (`float`, *optional*): The learning rate for the differentiable torch head
                in the `AdamW` optimizer.
            end_to_end (`bool`, defaults to `False`): If True, train the entire model end-to-end.
                Otherwise, freeze the `SentenceTransformer` body and only train the head.
            l2_weight (`float`, *optional*): The l2 weight for both the model body and head
                in the `AdamW` optimizer.
            max_length (`int`, *optional*): The maximum token length a tokenizer can generate. If not provided,
                the maximum length for the `SentenceTransformer` body is used.
            show_progress_bar (`bool`, defaults to `True`): Whether to display a progress bar for the training
                epochs and iterations.
            early_stopping_patience (`int`, *optional*): Number of epochs with no improvement 
                after which training will be stopped. Only used if validation set is provided.
            early_stopping_threshold (`float`, defaults to 0.0): Minimum change to qualify as improvement.
            metric_for_best_model (`str`, defaults to "loss"): Metric to monitor for early stopping.
            greater_is_better (`bool`, defaults to False): Whether higher values of the metric indicate better models.
        """
        if self.has_differentiable_head:  # train with pyTorch
            self.model_body.train()
            self.model_head.train()
            if not end_to_end:
                self.freeze("body")

            dataloader = self._prepare_dataloader(x_train, y_train, batch_size, max_length)
            if x_eval is not None and y_eval is not None:
                eval_dataloader = self._prepare_dataloader(x_eval, y_eval, batch_size, max_length)
            criterion = self.model_head.get_loss_fn()
            optimizer = self._prepare_optimizer(head_learning_rate, body_learning_rate, l2_weight)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

            # Early stopping setup
            best_metric = float("-inf") if greater_is_better else float("inf")
            best_model_state = {"epoch": None, "body": None, "head": None}
            no_improvement_count = 0
            self.head_log_history = self.head_log_history if self.head_log_history is not None else []

            for epoch_idx in trange(num_epochs, desc="Epoch", disable=not show_progress_bar):
                # Training loop - existing code
                epoch_loss = 0.0
                num_batches = 0
                
                for batch in tqdm(dataloader, desc="Iteration", disable=not show_progress_bar, leave=False):
                    features, labels = batch
                    optimizer.zero_grad()

                    # to model's device
                    features = {k: v.to(self.device) for k, v in features.items()}
                    labels = labels.to(self.device)

                    outputs = self.model_body(features)
                    if self.normalize_embeddings:
                        outputs["sentence_embedding"] = nn.functional.normalize(
                            outputs["sentence_embedding"], p=2, dim=1
                        )
                    outputs = self.model_head(outputs)
                    logits = outputs["logits"]

                    loss: torch.Tensor = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                # Calculate epoch average loss
                epoch_avg_loss = epoch_loss / max(1, num_batches)
                self.head_log_history.append({"epoch": epoch_idx + 1, "train_loss": epoch_avg_loss})

                # Validation and early stopping logic
                if x_eval is not None and y_eval is not None and early_stopping_patience is not None:
                    val_metric, greater_is_better, metrics = self._compute_eval_metric(eval_dataloader, metric_for_best_model, greater_is_better, compute_metrics, epoch_avg_loss, "eval", show_progress_bar)
                    self.head_log_history[-1].update(metrics)
                    print(self.head_log_history[-1])
                    
                    improved = \
                        (greater_is_better and val_metric > (best_metric + early_stopping_threshold)) or \
                        (not greater_is_better and val_metric < (best_metric - early_stopping_threshold))

                    if improved:
                        
                        best_metric = val_metric
                        # update best model state
                        best_model_state["epoch"] = epoch_idx + 1
                        if end_to_end:
                            best_model_state["body"] = {k: deepcopy(v.cpu()) for k, v in self.model_body.state_dict().items()}
                        best_model_state["head"] = {k: deepcopy(v.cpu()) for k, v in self.model_head.state_dict().items()}

                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    # Early stopping check
                    if no_improvement_count >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch_idx + 1} epochs")
                        break
                
                scheduler.step()
            
            # Load best model if early stopping occurred
            if load_best_model_at_end and best_model_state is not None and best_model_state["epoch"] is not None:
                print(f"Loading best model from epoch {best_model_state['epoch']}")
                if end_to_end and best_model_state["body"] is not None:
                    
                    self.model_body.load_state_dict(
                        {k: v.to(self.model_body.device) for k, v in best_model_state["body"].items()}
                    )
                if best_model_state["head"] is not None:
                    
                    self.model_head.load_state_dict(
                        {k: v.to(self.model_head.device) for k, v in best_model_state["head"].items()}
                    )

            if not end_to_end:
                self.unfreeze("body")

            self.model_body.eval()
            self.model_head.eval()

        else:  # train with sklearn
            if early_stopping_patience:
                warnings.warn(
                    "Early stopping not supported for non-differentiable head",
                    NotImplementedError,
                    stacklevel=2
                )
            
            embeddings = self.model_body.encode(x_train, normalize_embeddings=self.normalize_embeddings)
            self.model_head.fit(embeddings, y_train)
            if self.labels is None and self.multi_target_strategy is None:
                # Try to set the labels based on the head classes, if they exist
                # This can fail in various ways, so we catch all exceptions
                try:
                    classes = self.model_head.classes_
                    if classes.dtype.char == "U":
                        self.labels = classes.tolist()
                except Exception:
                    pass

    def _reset_head_log_history(self):
        self.head_log_history = None
    
    def _compute_eval_metric(self, eval_dataloader, metric_name, greater_is_better, compute_metrics, train_loss: float, metric_prefix="eval", show_progress_bar=True):
        """
        Compute the validation metrics for early stopping.

        Args:
            eval_dataloader: DataLoader for the evaluation dataset.
            metric_name: The name of the metric to compute.
            greater_is_better: Whether higher values of the metric indicate better models.
            compute_metrics: A callable to compute additional metrics.
            train_loss: The training loss to include in the metrics.
            metric_prefix: Prefix for the metric names.
            show_progress_bar: Whether to show a progress bar during evaluation.
        Returns:
            The computed validation metric and a dictionary of all metrics.
        """
        self.model_body.eval()
        self.model_head.eval()

        val_loss = 0.0
        num_batches = 0
        all_logits = []
        all_labels = []
        
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not show_progress_bar, leave=False):
            features, labels = batch
            all_labels.extend(self._output_type_conversion(labels, as_numpy=True))
            with torch.no_grad():
                features = {k: v.to(self.device) for k, v in features.items()}
                labels = labels.to(self.device)

                outputs = self.model_body(features)
                if self.normalize_embeddings:
                    outputs["sentence_embedding"] = nn.functional.normalize(
                        outputs["sentence_embedding"], p=2, dim=1
                    )
                outputs = self.model_head(outputs)
                logits = outputs["logits"]

                loss: torch.Tensor = self.model_head.get_loss_fn()(logits, labels)
                val_loss += loss.item()
                num_batches += 1
                logits = self._output_type_conversion(logits, as_numpy=True)
                all_logits.append(logits)
        val_avg_loss = val_loss / max(1, num_batches)
        
        val_metric = val_avg_loss
        metrics = {}
        if compute_metrics is not None:
            logits = np.concatenate(all_logits, axis=0)
            p = PredictionOutput(predictions=logits, label_ids=np.array(all_labels), metrics=None)
            metrics = compute_metrics(p) # NOTE: overwrite empty metrics dict
            if metric_name in metrics:
                val_metric = metrics[metric_name]
            else:
                warnings.warn(
                    f"metric \"{metric_name}\" not in output of `compute_metrics` function. "
                    f"Using validation loss as fallback",
                    stacklevel=2,
                )
                greater_is_better = False  # for loss
        metrics = {
            "train_loss": train_loss, 
            f"{metric_prefix}_loss" if metric_prefix else "validation_loss": val_avg_loss, 
            **{f"{metric_prefix}_{k}" if metric_prefix else k: v for k, v in metrics.items()}
        }
        return val_metric, greater_is_better, metrics
    
    # just for convenience
    def predict_logits(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 32,
        as_numpy: bool = False,
        show_progress_bar: Optional[bool] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Predict the logits of the various classes.

        Args:
            inputs (`Union[str, List[str]]`): The input sentences to predict class probabilities for.
            batch_size (`int`, defaults to `32`): The batch size to use in encoding the sentences to embeddings.
                Higher often means faster processing but higher memory usage.
            as_numpy (`bool`, defaults to `False`): Whether to output as numpy array instead.
            show_progress_bar (`Optional[bool]`, defaults to `None`): Whether to show a progress bar while encoding.

        Returns:
            `Union[torch.Tensor, np.ndarray]`: A matrix with shape [INPUT_LENGTH, NUM_CLASSES] denoting
            logits of predicting an input as a class. If the input is a string, then the output
            is a vector with shape [NUM_CLASSES,].
        """
        is_singular = isinstance(inputs, str)
        if is_singular:
            inputs = [inputs]
        embeddings = self.encode(inputs, batch_size=batch_size, show_progress_bar=show_progress_bar)
        # replicate logic from head's `predict_proba` but return logits instead of probabilities
        with torch.no_grad():
            logits = self.model_head(embeddings)[0]
        if isinstance(logits, list):
            if self.has_differentiable_head:
                logits = torch.stack(logits, axis=1)
            else:
                logits = np.stack(logits, axis=1)
        outputs = self._output_type_conversion(logits, as_numpy=as_numpy)
        return outputs[0] if is_singular else outputs
    

@dataclass
class EarlyStoppingTrainingArguments(TrainingArguments):
    metric_for_best_model: Optional[Tuple[str, ...]] = field(default=('embedding_loss', 'loss'), repr=True)
    greater_is_better: Tuple[bool, ...] = field(default=(False, False), repr=True)

    def __post_init__(self):
        super().__post_init__()
        
        if isinstance(self.metric_for_best_model, str):
            self.metric_for_best_model = (self.metric_for_best_model,)
        if isinstance(self.greater_is_better, bool):
            self.greater_is_better = (self.greater_is_better,)

    def update(self, arguments: Dict[str, Any], ignore_extra: bool = False) -> "EarlyStoppingTrainingArguments":
        return EarlyStoppingTrainingArguments.from_dict({**self.to_dict(), **arguments}, ignore_extra=ignore_extra)

    @classmethod
    def from_dict(cls, arguments: Dict[str, Any], ignore_extra: bool = False) -> "EarlyStoppingTrainingArguments":
        """Initialize a EarlyStoppingTrainingArguments instance from a dictionary.

        Args:
            arguments (`Dict[str, Any]`): A dictionary of arguments.
            ignore_extra (`bool`, *optional*): Whether to ignore arguments that do not occur in the
                EarlyStoppingTrainingArguments __init__ signature. Defaults to False.

        Returns:
            `EarlyStoppingTrainingArguments`: The instantiated EarlyStoppingTrainingArguments instance.
        """
        if ignore_extra:
            return cls(**{key: value for key, value in arguments.items() if key in inspect.signature(cls).parameters})
        return cls(**arguments)
    
    def copy(self) -> "EarlyStoppingTrainingArguments":
        """Create a shallow copy of this EarlyStoppingTrainingArguments instance."""
        return copy(self)


# define helper function that precomputes logits and label_ids to format (`y_pred`, `y_test`) required by setfit trainers' `metric` argument
def _prepare_singlelabel_metrics_fn(p: PredictionOutput):
    labels = p.label_ids
    logits = p.predictions
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    y_pred = logits.argmax(axis=-1)
    y_test = labels
    return y_pred, y_test

def _prepare_multilabel_metrics_fn(p: PredictionOutput):
    labels = p.label_ids
    logits = p.predictions
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    y_pred = (probs > 0.5).astype(int)
    y_test = labels
    return y_pred, y_test

class SetFitEarlyStoppingTrainer(Trainer):

    def __init__(
        self,
        model: Optional["SetFitModel"] = None,
        args: Optional[Union[TrainingArguments, EarlyStoppingTrainingArguments]] = None,
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
        model_init: Optional[Callable[[], "SetFitModel"]] = None,
        metric: Union[str, Callable[["Dataset", "Dataset"], Dict[str, float]]] = "accuracy",
        metric_kwargs: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        compute_metrics: Optional[Callable] = None,
    ) -> None:
        
        # parse callbacks to separate head early stopping callback (if any)
        new_callbacks, head_early_stopping_callback = self._parse_callbacks(callbacks)

        # initialize base class
        super().__init__(
            model=model,
            args=self._to_setfit_train_args(args),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            metric=metric,
            metric_kwargs=metric_kwargs,
            callbacks=new_callbacks,
            column_mapping=column_mapping,
        )
        
        # set args and model using the property setters to ensure consistency
        self.args = args
        
        # add an attribute to the model to store the head early stopping callback (if any)
        self.model.head_early_stopping_callback = head_early_stopping_callback
        
        # set compute metrics function
        self._set_compute_metrics(compute_metrics)

    @staticmethod
    def _parse_callbacks(callbacks: Optional[List[TrainerCallback]]):
        # remove duplicate early stopping callback, from sentence transformer trainer, if any
        # NOTE: this ensures that the callbacks provided to init the sentence transformer trainer inside setfit trainer do not contain multiple early stopping callbacks (which would lead to errors)
        
        early_stopping_callbacks = [c for c in callbacks if isinstance(c, EarlyStoppingCallback)] if callbacks is not None else []
        head_early_stopping_callback = None
        if len(early_stopping_callbacks) < 2:
            return callbacks, head_early_stopping_callback
        
        new_callbacks = []
        surplus_eac = False
        for cb in callbacks:
            # cb = callbacks.pop()
            if not isinstance(cb, EarlyStoppingCallback):
                new_callbacks.append(cb)
            elif not surplus_eac:
                new_callbacks.append(cb)
                surplus_eac = True
            elif not head_early_stopping_callback:
                head_early_stopping_callback = cb
            else:
                # skip surplus early stopping callbacks
                # TODO: warn?
                pass
        return new_callbacks, head_early_stopping_callback
    
    def _set_compute_metrics(self, compute_metrics: Optional[Callable]=None) -> None:
        # setup the metric computation function
        if isinstance(compute_metrics, Callable):
            self._compute_metrics = compute_metrics
        elif isinstance(self.metric, str):
            import evaluate
            metric_kwargs = self.metric_kwargs or {}
            metric_config = None
            if self.model.multi_target_strategy:
                metric_config = "multilabel"
                if "average" not in metric_kwargs or metric_kwargs["average"] == "binary":
                    # TODO: consider to instead raise an error/warning here
                    metric_kwargs["average"] = "macro"
            metric_fn = evaluate.load(self.metric, config_name=metric_config)
            def compute_metrics(p):
                y_pred, y_test = _prepare_multilabel_metrics_fn(p) if self.model.multi_target_strategy is not None else _prepare_singlelabel_metrics_fn(p)
                return metric_fn.compute(predictions=y_pred, references=y_test, **metric_kwargs)
            self._compute_metrics = compute_metrics
        elif callable(self.metric):
            metric_kwargs = self.metric_kwargs or {}
            def compute_metrics(p):
                y_pred, y_test = _prepare_multilabel_metrics_fn(p) if self.model.multi_target_strategy is not None else _prepare_singlelabel_metrics_fn(p)
                return self.metric(y_pred, y_test, **metric_kwargs)
            self._compute_metrics = compute_metrics
        else:
            self._compute_metrics = None

    @staticmethod
    def _to_setfit_train_args(args: Union[TrainingArguments, EarlyStoppingTrainingArguments]):
        sf_args = TrainingArguments(**(args.to_dict() if args else {}))
        if isinstance(sf_args.metric_for_best_model, tuple):
            sf_args.metric_for_best_model = sf_args.metric_for_best_model[0]
        if isinstance(sf_args.greater_is_better, tuple):
            sf_args.greater_is_better = sf_args.greater_is_better[0]
        return sf_args
    
    @property
    def args(self) -> Union[TrainingArguments, EarlyStoppingTrainingArguments]:
        return self._args

    @args.setter
    def args(self, args: Union[TrainingArguments, EarlyStoppingTrainingArguments]) -> None:
        self._args = args
        if hasattr(self, "st_trainer"):
            self.st_trainer.setfit_args = self._to_setfit_train_args(args)

    @property
    def model(self) -> Union["SetFitModel", "SetFitModelWithEarlyStopping"]:
        return self._model

    @model.setter
    def model(self, model: Union["SetFitModel", "SetFitModelWithEarlyStopping"]) -> None:
        self._model = model
        if hasattr(self, "st_trainer"):
            self.st_trainer.setfit_model = model

    
    def train(
        self,
        args: Optional[Union[TrainingArguments, EarlyStoppingTrainingArguments]] = None,
        trial: Optional[Union["optuna.Trial", Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """
        Main training entry point.

        Args:
            args (`TrainingArguments`, *optional*):
                Temporarily change the training arguments for this training call.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        if len(kwargs):
            warnings.warn(
                f"`{self.__class__.__name__}.train` does not accept keyword arguments anymore. "
                f"Please provide training arguments via a `TrainingArguments` or `EarlyStoppingTrainingArguments` instance to the `{self.__class__.__name__}` "
                f"initialisation or the `{self.__class__.__name__}.train` method.",
                DeprecationWarning,
                stacklevel=2,
            )

        if trial:  # Trial and model initialization
            self._hp_search_setup(trial)  # sets trainer parameters and initializes model

        args = args or self.args or EarlyStoppingTrainingArguments() # NOTE: using subclassed train args here 

        if self.train_dataset is None:
            raise ValueError(
                f"Training requires a `train_dataset` given to the `{self.__class__.__name__}` initialization."
            )

        train_parameters = self.dataset_to_parameters(self.train_dataset)
        full_parameters = (
            train_parameters + self.dataset_to_parameters(self.eval_dataset) if self.eval_dataset else train_parameters
        )

        # NOTE: allow no training of body when num_epochs[0] <=0 or max_steps <=0
        if args.num_epochs[0] <= 0:
            warnings.warn("Skipping body training step because `num_epochs[0]` is set to 0 or less.")
        elif args.max_steps <= 0:
            warnings.warn("Skipping body training step because `max_steps` is set to 0 or less.")
        else:
            self.train_embeddings(*full_parameters, args=args)
        
        # NOTE: also passing eval X and y to classifier for training (in vanilla setfit trainer these where not passed and only *train_parameters used)
        self.train_classifier(*full_parameters, args=args)
    
    def train_embeddings(
        self,
        x_train: List[str],
        y_train: Optional[Union[List[int], List[List[int]]]] = None,
        x_eval: Optional[List[str]] = None,
        y_eval: Optional[Union[List[int], List[List[int]]]] = None,
        args: Optional[Union[TrainingArguments, EarlyStoppingTrainingArguments]] = None,
    ):
        st_args = deepcopy(args)
        if st_args and hasattr(st_args, "metric_for_best_model") and isinstance(st_args.metric_for_best_model, tuple):
            st_args.metric_for_best_model = st_args.metric_for_best_model[0]
        if st_args and hasattr(st_args, "greater_is_better") and isinstance(st_args.greater_is_better, tuple):
            st_args.greater_is_better = st_args.greater_is_better[0]
        super().train_embeddings(x_train, y_train, x_eval, y_eval, st_args)
        
    def train_classifier(
        self,
        x_train: List[str],
        y_train: Optional[Union[List[int], List[List[int]]]] = None,
        x_eval: Optional[List[str]] = None,
        y_eval: Optional[Union[List[int], List[List[int]]]] = None,
        args: Optional[Union[TrainingArguments, EarlyStoppingTrainingArguments]] = None,
    ):
        shared_args = {
            "num_epochs": args.classifier_num_epochs,
            "batch_size": args.classifier_batch_size,
            "body_learning_rate": args.body_classifier_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "l2_weight": args.l2_weight,
            "max_length": args.max_length,
            "show_progress_bar": args.show_progress_bar,
            "end_to_end": args.end_to_end,
        }
        if isinstance(self.model, SetFitModelWithEarlyStopping):
            early_stopping_kwargs = {}
            
            if isinstance(args, EarlyStoppingTrainingArguments):
                early_stopping_kwargs["metric_for_best_model"] = args.metric_for_best_model[1] if isinstance(args.metric_for_best_model, tuple) else "loss"
                early_stopping_kwargs["greater_is_better"] = args.greater_is_better[1] if isinstance(args.greater_is_better, tuple) else False 
                if early_stopping_kwargs["metric_for_best_model"]=="loss" and early_stopping_kwargs["greater_is_better"]:
                    # TODO: warn that EarlyStoppingTrainingArguments.greater_is_better[1] should be false when using "loss"
                    pass
                early_stopping_kwargs["load_best_model_at_end"] = args.load_best_model_at_end
            else:
                early_stopping_kwargs["metric_for_best_model"] = "loss"
                early_stopping_kwargs["greater_is_better"] = False

            #if any(isinstance(c, EarlyStoppingCallback) for c in self._callbacks):
            #    cb = [c for c in self._callbacks if isinstance(c, EarlyStoppingCallback)][-1]
            if hasattr(self.model, "head_early_stopping_callback") and self.model.head_early_stopping_callback is not None:
                early_stopping_kwargs["early_stopping_patience"] = self.model.head_early_stopping_callback.early_stopping_patience
                early_stopping_kwargs["early_stopping_threshold"] = self.model.head_early_stopping_callback.early_stopping_threshold
            early_stopping_kwargs["compute_metrics"] = self._compute_metrics

            self.model.fit(
                x_train,
                y_train,
                x_eval,
                y_eval,
                # shared args
                **shared_args,
                # early stopping args
                **early_stopping_kwargs
            )
        elif isinstance(self.model, SetFitModel):
            self.model.fit(
                x_train,
                y_train,
                # shared args
                **shared_args
            )
        else:
            raise ValueError("Model must be an instance of SetFitModel or SetFitModelWithEarlyStopping")

    def _hp_search_setup(self, trial: Union["optuna.Trial", Dict[str, Any]]) -> None:
        """HP search setup code"""

        # Heavily inspired by transformers.Trainer._hp_search_setup
        if self.hp_search_backend is None or trial is None:
            return
        if isinstance(trial, Dict):  # For passing a Dict to train() -- mostly unused for now
            params = trial
        elif self.hp_search_backend == HPSearchBackend.OPTUNA:
            params = self.hp_space(trial)
        else:
            raise ValueError("Invalid trial parameter")

        logger.info(f"Trial: {params}")
        self.apply_hyperparameters(params, final_model=False)
        
        # NOTE the next two lines are the only ones that differs from transformers.Trainer._hp_search_setup
        if isinstance(self.model, SetFitModelWithEarlyStopping):
            self.model._reset_head_log_history()

    def apply_hyperparameters(self, params: Dict[str, Any], final_model: bool = False) -> None:
        """Applies a dictionary of hyperparameters to both the trainer and the model

        Args:
            params (`Dict[str, Any]`): The parameters, usually from `BestRun.hyperparameters`
            final_model (`bool`, *optional*, defaults to `False`): If `True`, replace the `model_init()` function with a fixed model based on the parameters.
        """
        
        if self.args is not None:
            self.args = self.args.update(params, ignore_extra=True)
        else:
            self.args = EarlyStoppingTrainingArguments.from_dict(params, ignore_extra=True)

        # capture head early stopping callback if any
        head_early_stopping_callback = self.model.head_early_stopping_callback or None
        
        # Seed must be set before instantiating the model when using model_init.
        set_seed(self.args.seed)
        self.model = self.model_init(params)
        
        # set head early stopping callback
        self.model.head_early_stopping_callback = head_early_stopping_callback

        # need to create a new sentence transformer trainer because of the following problem:
        # Problem:
        #   trainer.apply_hyperparameters() re-instantiates a new SetFitModel (by calling model_init function) and thus a new SentenceTransformer "body".
        #   However, the old internal Sentence-Transformers trainer (self.st_trainer) is kept and still bound to the previous model instance and its optimizer graph.
        #   As a result, gradients are computed for the wrong model (or detached entirely), so the sentence-embedding body stopped updating.
        # 
        # My fix (and why it works):
        #   By recreating a fresh BCSentenceTransformersTrainer after every call to apply_hyperparameters(),  ensure that the internal ST trainer, its loss, optimizer, and callbacks are all initialized with the current model body and arguments, keeping the computation graph and gradient flow consistent with the newly instantiated model.
        callbacks = copy(self.st_trainer.callback_handler.callbacks)
        from setfit.trainer import BCSentenceTransformersTrainer
        self.st_trainer = BCSentenceTransformersTrainer(
            setfit_model=self.model,
            setfit_args=self._to_setfit_train_args(self.args),
            callbacks=[]#,
        )
        self.st_trainer.callback_handler.callbacks = callbacks

        if final_model:
            self.model_init = None

    
    # def hyperparameter_search(
    #     self,
    #     hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
    #     compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
    #     n_trials: int = 10,
    #     direction: str = "maximize",
    #     backend: Optional[Union["str", HPSearchBackend]] = None,
    #     hp_name: Optional[Callable[["optuna.Trial"], str]] = None,
    #     **kwargs,
    # ) -> BestRun:
    #     """
    #     Launch a hyperparameter search using `optuna`. The optimized quantity is determined
    #     by `compute_objective`, which defaults to a function returning the evaluation loss when no metric is provided,
    #     the sum of all metrics otherwise.

    #     <Tip warning={true}>

    #     To use this method, you need to have provided a `model_init` when initializing your [`Trainer`]: we need to
    #     reinitialize the model at each new run.

    #     </Tip>

    #     Args:
    #         hp_space (`Callable[["optuna.Trial"], Dict[str, float]]`, *optional*):
    #             A function that defines the hyperparameter search space. Will default to
    #             [`~transformers.trainer_utils.default_hp_space_optuna`].
    #         compute_objective (`Callable[[Dict[str, float]], float]`, *optional*):
    #             A function computing the objective to minimize or maximize from the metrics returned by the `evaluate`
    #             method. Will default to [`~transformers.trainer_utils.default_compute_objective`] which uses the sum of metrics.
    #         n_trials (`int`, *optional*, defaults to 100):
    #             The number of trial runs to test.
    #         direction (`str`, *optional*, defaults to `"maximize"`):
    #             Whether to optimize greater or lower objects. Can be `"minimize"` or `"maximize"`, you should pick
    #             `"minimize"` when optimizing the validation loss, `"maximize"` when optimizing one or several metrics.
    #         backend (`str` or [`~transformers.training_utils.HPSearchBackend`], *optional*):
    #             The backend to use for hyperparameter search. Only optuna is supported for now.
    #             TODO: add support for ray and sigopt.
    #         hp_name (`Callable[["optuna.Trial"], str]]`, *optional*):
    #             A function that defines the trial/run name. Will default to None.
    #         kwargs (`Dict[str, Any]`, *optional*):
    #             Additional keyword arguments passed along to `optuna.create_study`. For more
    #             information see:

    #             - the documentation of
    #               [optuna.create_study](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html)

    #     Returns:
    #         [`trainer_utils.BestRun`]: All the information about the best run.
    #     """
    #     if backend is None:
    #         backend = default_hp_search_backend()
    #         if backend is None:
    #             raise RuntimeError("optuna should be installed. To install optuna run `pip install optuna`.")
    #     backend = HPSearchBackend(backend)
    #     if backend == HPSearchBackend.OPTUNA and not is_optuna_available():
    #         raise RuntimeError("You picked the optuna backend, but it is not installed. Use `pip install optuna`.")
    #     elif backend != HPSearchBackend.OPTUNA:
    #         raise RuntimeError("Only optuna backend is supported for hyperparameter search.")
    #     self.hp_search_backend = backend
    #     if self.model_init is None:
    #         raise RuntimeError(
    #             "To use hyperparameter search, you need to pass your model through a model_init function."
    #         )

    #     self.hp_space = default_hp_space_optuna if hp_space is None else hp_space
    #     self.hp_name = hp_name
    #     self.compute_objective = default_compute_objective if compute_objective is None else compute_objective

    #     backend_dict = {
    #         HPSearchBackend.OPTUNA: run_hp_search_optuna,
    #     }
    #     best_run = backend_dict[backend](self, n_trials, direction, **kwargs)

    #     # do _not_ delete hp_search_backend attribute to allow for inspecting it after HP search
    #     # self.hp_search_backend = None 
    #     # NOTE: this is the only change to base class's implementation (see https://github.com/huggingface/setfit/blob/f8b42a96ab8b9b504c879d62b6f5020cf4876ac8/src/setfit/trainer.py#L719C1-L793C24)
        
    #     return best_run
