import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_splits_path', type=str, required=True, help='Path to data splits directory. Should contain files "train.pkl", "val.pkl", and "test.pkl"')
    parser.add_argument('--label_cols', type=str, required=True, help='Comma-separated list of label column names')
    parser.add_argument('--exclude_label_cols', type=str, nargs='+', default=None, help='Comma-separated list of label column names to exclude from --label_cols')
    parser.add_argument('--id_col', type=str, default='mention_id', help='Column name for unique mention IDs')
    parser.add_argument('--text_col', type=str , default='text', help='Column name for mention context text')
    parser.add_argument('--mention_col', type=str , default='mention', help='Column name for mention text')
    parser.add_argument('--span_col', type=str , default='span', help='Column name for mention span (start, end)')

    parser.add_argument('--combine_splits', default=None, type=str, nargs='+', help='If specified, combine the given splits into a single training set (e.g., --combine_splits val test)')

    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use. Must be a sentence-transformers compatible model.')
    parser.add_argument('--use_span_embeddings', action='store_true', help='Whether to use custom SeFitForSpanClassification Trainer instead of mention and text concatenation or mention-only strategies')
    parser.add_argument('--concat_strategy', type=str, choices=[None, 'prefix', 'suffix'], default=None, help='If not None, concatenate the mention text as prefix or suffix to the context text using --concat_sep_token')
    parser.add_argument('--concat_sep_token', type=str, default=': ', help='Separator token to use when concatenating mention text to context text')

    parser.add_argument('--class_weighting_strategy', type=str, choices=[None, 'balanced', 'inverse_proportional'], default=None, help='Class weighting strategy to use during training')
    parser.add_argument('--class_weighting_smooth_exponent', type=float, default=None, help='Smoothing exponent to use when computing class weights (only relevant if --class_weighting_strategy is set to "inverse_proportional")')


    parser.add_argument('--num_epochs', type=int, nargs='+', default=[1, 15], help='Tuple of (min, max) number of epochs to use for embedding model and classifier training, respectively')
    parser.add_argument('--train_batch_sizes', type=int, nargs='+', default=[32, 8], help='Tuple of batch sizes to use for embedding model and classifier training, respectively')

    parser.add_argument('--body_train_max_steps', type=int, default=750, help='Maximum number of training steps for sentence transformer finetuning')
    parser.add_argument('--head_learning_rate', type=float, default=0.01, help='Learning rate to use for classifier head training')
    parser.add_argument('--l2_weight', type=float, default=0.01, help='L2 weight to use for classifier head training')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Warmup proportion to use for classifier head training')

    parser.add_argument('--body_early_stopping_patience', type=int, default=3, help='Early stopping patience for sentence transformer finetuning')
    parser.add_argument('--body_early_stopping_threshold', type=float, default=0.01, help='Early stopping threshold for sentence transformer finetuning')
    parser.add_argument('--head_early_stopping_patience', type=int, default=5, help='Early stopping patience for classifier head finetuning')
    parser.add_argument('--head_early_stopping_threshold', type=float, default=0.015, help='Early stopping threshold for classifier head finetuning')

    parser.add_argument('--save_eval_results_to', type=str, default=None, help='Directory to save evaluation results to')
    parser.add_argument('--overwrite_results', action='store_true', help='Whether to overwrite existing evaluation results')

    parser.add_argument('--do_eval', action='store_true', help='Whether to perform evaluation on the validation set')
    parser.add_argument('--save_eval_results', action='store_true', help='Whether to save evaluation results to disk')
    parser.add_argument('--save_eval_predictions', action='store_true', help='Whether to save evaluation predictions to disk')

    parser.add_argument('--do_test', action='store_true', help='Whether to perform testing on the test set')
    parser.add_argument('--save_test_results', action='store_true', help='Whether to save test results to disk')
    parser.add_argument('--save_test_predictions', action='store_true', help='Whether to save test predictions to disk')

    parser.add_argument('--save_model', action='store_true', help='Whether to save the trained model to disk')
    parser.add_argument('--save_model_to', type=str, help='Directory to save the trained model to')

    return parser.parse_args()


# ## Setup

import os
from pathlib import Path
import shutil
import warnings
import json

import numpy as np
np.set_printoptions(precision=4, suppress=True)
import pandas as pd
import regex

import torch
torch.set_float32_matmul_precision('high')
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, set_seed
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # to enable deterministic behavior with CuBLAS
SEED = 42
set_seed(SEED, deterministic=True) # for reproducibility

# default setfit body and head
from sentence_transformers import SentenceTransformer
from setfit.modeling import SetFitHead, SetFitModel

# class weight head
from finetuning.setfit_extensions.class_weights_head import (
    compute_class_weights,
    SetFitHeadWithClassWeights
)
# early stopping model, training args, and trainer
from finetuning.setfit_extensions.early_stopping import (
    SetFitModelWithEarlyStopping, 
    EarlyStoppingTrainingArguments,
    EarlyStoppingCallback,
    SetFitEarlyStoppingTrainer
)
# span embedding model, head, and trainer
from finetuning.setfit_extensions.span_embedding import (
    SentenceTransformerForSpanEmbedding,
    SetFitModelForSpanClassification,
    SetFitTrainerForSpanClassification,
)

from sklearn.metrics import classification_report
# from utils.metrics import *

def model_init(
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        id2label: dict = None, #num_classes: int = 2,
        class_weights: np._typing.NDArray = None,
        multilabel: bool = False,
        use_span_embedding: bool = False,
        body_kwargs: dict = {},
        head_kwargs: dict = {},
        model_kwargs: dict = {},
        enable_early_stopping: bool = True,
    ) -> SetFitModel | SetFitModelWithEarlyStopping | SetFitModelForSpanClassification:
    """
    Initialize a SetFit model with optional span embeddings and class weights.
    """
    
    body_class = SentenceTransformerForSpanEmbedding if use_span_embedding else SentenceTransformer
    body_kwargs={"device_map": "auto", **body_kwargs}
    body = body_class(model_name, model_kwargs=body_kwargs, trust_remote_code=True)
    
    
    head_class = SetFitHead
    head_kwargs = {
        "in_features": body.get_sentence_embedding_dimension(),
        "out_features": len(id2label), # num_classes,
        "device": body.device,
        "multitarget": multilabel,
        **head_kwargs
    }
    if class_weights is not None:
        head_class = SetFitHeadWithClassWeights
        head_kwargs["class_weights"] = class_weights
    head = head_class(**head_kwargs)
    

    if use_span_embedding:
        model_class = SetFitModelForSpanClassification
    elif enable_early_stopping:
        model_class = SetFitModelWithEarlyStopping
    else:
        model_class = SetFitModel
    if multilabel and "multi_target_strategy" not in model_kwargs:
        model_kwargs["multi_target_strategy"] = "one-vs-rest"
    return model_class(
        model_body=body,
        normalize_embeddings=True,
        model_head=head.to(body.device),
        labels=list(id2label.values()),
        **model_kwargs
    )

# ## Prepare the datasets

def prepare_dataset(args: argparse.Namespace) -> pd.DataFrame:
    """Load the splits, combine them if requested, and resolve label columns (may be glob patterns) into the 'labels' column."""

    # ### Load the splits

    df = pd.concat({split: pd.read_pickle(args.data_splits_path / f"{split}.pkl") for split in ['train', 'val', 'test']})
    df.reset_index(level=0, names='split', inplace=True)
    if args.combine_splits is not None:
        for split in args.combine_splits:
            df.loc[df.split == split, 'split'] = 'train'
    # df['split'] = pd.Categorical(df['split'], categories=['train', 'val', 'test'], ordered=True)

    # ### prepare the label column

    # consider that entries in args.label_cols may be glob patterns
    import fnmatch
    expanded_label_cols = []
    for lab in args.label_cols:
        matched = fnmatch.filter(df.columns, lab)
        if matched:
            expanded_label_cols.extend(matched)
        elif lab in df.columns:
            expanded_label_cols.append(lab)
        else:
            raise ValueError(f"Label column '{lab}' not found in dataframe columns and did not match any glob patterns.")
    args.label_cols = expanded_label_cols

    # print info about label columns
    print(f"Using label columns: {args.label_cols}")

    df['labels'] = df[args.label_cols].apply(list, axis=1)

    return df


def format_inputs(df: pd.DataFrame, args: argparse.Namespace, tokenizer) -> tuple[list[str], dict[str, str], int]:
    """Build the model input column (span, concatenated, or mention-only) and compute the max token length."""

    # ### format inputs

    if args.use_span_embeddings:
        if "span" not in df.columns:
        # using span embedding strategy
            df['span'] = df.apply(lambda x: regex.search(regex.escape(x[args.mention_col]), x[args.text_col]).span(), axis=1)
        max_length_ = max(tokenizer(df[args.text_col].to_list(), truncation=False, padding=False, return_length=True).length)
        cols = [args.text_col, 'span', 'labels']
        cols_mapping = {args.text_col: 'text', 'span': 'span', 'labels': 'label'}
    elif args.concat_strategy is None:
        # default: just the mention text
        max_length_ = max(tokenizer(df[args.mention_col].to_list(), truncation=False, padding=False, return_length=True).length)
        cols = [args.mention_col, 'labels']
        cols_mapping = {args.mention_col: 'text', 'labels': 'label'}
    else:
        # using concat strategy
        sep_tok = tokenizer.sep_token if args.concat_sep_token is None else args.concat_sep_token
        if args.concat_strategy == 'prefix':
            df['input'] = df[args.mention_col] + sep_tok + df[args.text_col]
        elif args.concat_strategy == 'suffix':
            df['input'] = df[args.text_col] + sep_tok + df[args.mention_col]
        else:
            raise ValueError(f"Unknown concat strategy: {args.concat_strategy}")
        max_length_ = max(tokenizer(df['input'].to_list(), truncation=False, padding=False, return_length=True).length)
        cols = ['input', 'labels']
        cols_mapping = {"input": "text", "labels": "label"}

    return cols, cols_mapping, max_length_


def split_data(df: pd.DataFrame, cols: list[str], cols_mapping: dict[str, str]) -> DatasetDict:
    """Convert the dataframe into a DatasetDict keyed by 'split', keeping only the model-input columns."""

    # ### split the data

    datasets = DatasetDict({
        s: Dataset.from_pandas(d, preserve_index=False)
        for s, d in df.groupby('split', observed=True)
    })

    datasets = datasets.remove_columns(set(df.columns)-set(cols))

    datasets = datasets.rename_columns(column_mapping=cols_mapping)

    return datasets


def prepare_finetuning(args: argparse.Namespace, datasets: DatasetDict, max_length_: int):
    """Compute class weights, resolve the model output dir, and build the (early-stopping) SetFit trainer."""

    # ### Prepare fine-tuning

    id2label = {i: l for i, l in enumerate(args.label_cols)}
    label2id = {l: i for i, l in id2label.items()}

    if args.class_weighting_strategy in ['inverse_proportional']:
        class_weighting_args = {
            "multitarget": len(args.label_cols) > 1,
            "smooth_weights": args.class_weighting_strategy is not None and args.class_weighting_strategy != 'balanced',
            "smooth_exponent": args.class_weighting_smooth_exponent if args.class_weighting_smooth_exponent is not None else 0.5
        }
        class_weights = compute_class_weights(datasets['train']['label'], **class_weighting_args)
        print(f"Class weights: {dict(zip(label2id.keys(), class_weights))}")
    else:
        class_weights = None

    from sentence_transformers.losses import ContrastiveLoss

    if args.save_model:
        if args.save_model_to is None:
            raise ValueError("Both 'save_model_to' and 'save_model_as' must be specified if 'save_model' is True.")
        model_dir = args.save_model_to
    else:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdirname:
            model_dir = tmpdirname

    eval_while_training = "val" in datasets.keys()
    if eval_while_training:
        early_stopping_train_args = dict(
            # early stopping config
            metric_for_best_model=("embedding_loss", "f1"),
            greater_is_better=(False, True),
            load_best_model_at_end=True,
            save_total_limit=2, # NOTE: currently no effect on (early stopping in) classification head training
        )
    else:
        early_stopping_train_args = dict()

    training_args = EarlyStoppingTrainingArguments(
        output_dir=model_dir,
        loss=ContrastiveLoss,
        
        num_epochs=tuple(args.num_epochs),
        batch_size=tuple(args.train_batch_sizes),
        end_to_end=True,

        head_learning_rate = args.head_learning_rate,
        l2_weight=args.l2_weight,
        warmup_proportion=args.warmup_proportion,
        
        # sentence transformer (embedding) finetuning args
        max_steps=args.body_train_max_steps,
        logging_first_step=False,
        eval_strategy="steps" if eval_while_training else "no",
        eval_steps=50 if eval_while_training else None,
        eval_max_steps=250 if eval_while_training else None,
        
        # early stopping args
        **early_stopping_train_args,
    )


    training_callbacks = []
    if eval_while_training:
        training_callbacks += [
            # for sentence transformer finetuning
            EarlyStoppingCallback(
                early_stopping_patience=args.body_early_stopping_patience,
                early_stopping_threshold=args.body_early_stopping_threshold,
            ), 
            # for classifier finetuning
            EarlyStoppingCallback(
                early_stopping_patience=args.head_early_stopping_patience,
                early_stopping_threshold=args.head_early_stopping_threshold,
            ), 
        ]

    trainer_class = SetFitTrainerForSpanClassification if args.use_span_embeddings else SetFitEarlyStoppingTrainer

    trainer = trainer_class(
        model_init=lambda: model_init(
            model_name=args.model_name,
            #num_classes=len(id2label),
            id2label=id2label,
            multilabel=True,
            class_weights=class_weights,
            use_span_embedding=args.use_span_embeddings,
        ),
        metric="f1" if eval_while_training else None,
        metric_kwargs={
            "average": "macro" if args.label_cols and len(args.label_cols) > 1 else "binary",
        },
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'] if eval_while_training else None,
        callbacks=training_callbacks,
    )
    # fix max_length issue
    trainer._args.max_length = min(trainer.st_trainer.model.tokenizer.model_max_length, int(max_length_*1.1))

    # set seeds for reproducibility
    trainer._args.seed = SEED
    trainer.st_trainer.args.seed = SEED
    trainer.st_trainer.args.data_seed = SEED
    trainer.st_trainer.args.full_determinism = True

    # don't report to wandb or other experiment trackers
    trainer._args.report_to = 'none'
    trainer.st_trainer.args.report_to = 'none'

    return trainer, model_dir


def finetune_and_evaluate(trainer, model_dir, df: pd.DataFrame, datasets: DatasetDict, args: argparse.Namespace) -> None:
    """Train the trainer, save the model if requested, and evaluate on the val/test splits."""

    # ### Fine-tune

    trainer.train()

    # clean up
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    # ## Save the model

    if args.save_model:
        trainer.model.save_pretrained(model_dir)


    # ## Evaluate

    def get_predictions_df(split: str = 'val') -> pd.DataFrame:
        preds_df = df.loc[df['split'] == split, [args.id_col, args.text_col, args.mention_col, args.span_col, *args.label_cols]].copy()
        
        inputs = trainer.model._normalize_inputs(texts=datasets[split]['text'], spans=datasets[split]['span']) if args.use_span_embeddings else datasets[split]['text']
        
        probs = trainer.model.predict_proba(inputs, as_numpy=True)
        prob_cols = [f"prob_{col}" for col in trainer.model.labels]
        preds_df[prob_cols] = probs
        
        preds = np.where(probs > 0.5, 1, 0)
        pred_cols = [f"pred_{col}" for col in trainer.model.labels]
        preds_df[pred_cols] = preds
        
        for lab in trainer.model.labels:
            preds_df[f"error_{lab}"] = preds_df[f"pred_{lab}"] != preds_df[lab]
        
        return preds_df

    # ### Validation set

    if args.do_eval:

        inputs = trainer.model._normalize_inputs(texts=datasets['val']['text'], spans=datasets['val']['span']) if args.use_span_embeddings else datasets['val']['text']
        preds = trainer.model.predict(inputs, as_numpy=True)
        print(classification_report(y_pred=preds, y_true=datasets['val']['label'], target_names=args.label_cols, zero_division=0))

        if args.save_eval_results:
            res = classification_report(y_pred=preds, y_true=datasets['val']['label'], target_names=args.label_cols, zero_division=0, output_dict=True)
            fp = args.save_eval_results_to / 'eval_results.json'
            with open(fp, 'w') as f:
                json.dump(res, f)

        if args.save_eval_predictions:
            preds_df = get_predictions_df(split='val')
            fp = args.save_eval_results_to / 'eval_predictions.pkl'
            preds_df.to_pickle(fp)

    # ### Test set

    if args.do_test:
        print(classification_report(y_pred=preds, y_true=datasets['test']['label'], target_names=args.label_cols, zero_division=0))

        inputs = trainer.model._normalize_inputs(texts=datasets['test']['text'], spans=datasets['test']['span']) if args.use_span_embeddings else datasets['test']['text']
        preds = trainer.model.predict(inputs, as_numpy=True)


if __name__ == "__main__":
    args = parse_args()

    args.data_splits_path = Path(args.data_splits_path)

    if isinstance(args.label_cols, str):
        args.label_cols = [col.strip() for col in args.label_cols.split(',')]
    if args.exclude_label_cols is not None:
        if isinstance(args.exclude_label_cols, str):
            args.exclude_label_cols = [col.strip() for col in args.exclude_label_cols.split(',')]
        args.label_cols = [col for col in args.label_cols if col not in args.exclude_label_cols]


    if args.save_eval_results_to is not None:
        args.save_eval_results_to = Path(args.save_eval_results_to)
        if not (args.do_eval or args.do_test):
            raise ValueError("'save_eval_results_to' is specified but neither 'do_eval' nor 'do_test' is set.")
        elif not any([args.save_eval_results, args.save_eval_predictions, args.save_test_results, args.save_test_predictions]):
            warnings.warn("'save_eval_results_to' is specified but none of 'save_eval_results', 'save_eval_predictions', 'save_test_results', or 'save_test_predictions' is set.")
        elif args.save_eval_results_to.exists() and not args.overwrite_results:
            raise ValueError(f"The directory '{args.save_eval_results_to}' already exists. To avoid overwriting, please specify a different path or set 'overwrite_results'")
        else:
            args.save_eval_results_to.mkdir(parents=True, exist_ok=True)

    if args.save_model:
        if args.save_model_to is None:
            raise ValueError("Both 'save_model_to' and 'save_model_as' must be specified if 'save_model' is True.")
        args.save_model_to = Path(args.save_model_to)

    df = prepare_dataset(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    cols, cols_mapping, max_length_ = format_inputs(df, args, tokenizer)

    datasets = split_data(df, cols, cols_mapping)

    trainer, model_dir = prepare_finetuning(args, datasets, max_length_)

    finetune_and_evaluate(trainer, model_dir, df, datasets, args)

