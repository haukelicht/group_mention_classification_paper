import os
import json
import math
import copy
import numpy as np
import regex

import torch
from torch import Tensor
import torch.nn as nn

from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling
from sentence_transformers.util import batch_to_device, truncate_embeddings
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sentence_transformers.quantization import quantize_embeddings
from sentence_transformers import losses

from transformers import PreTrainedTokenizerBase
from typing import Dict, List
TokenizerOutput = Dict[str, List[int]]

from typing import List, Tuple, Union, Optional, overload
Text = str
Span = Tuple[int, int]
TextSpanInputItem = Tuple[Text, Span]
TextPhraseInputItem = Tuple[Text, str]
InputItem = Union[TextSpanInputItem, TextPhraseInputItem]

from setfit import SetFitModel, Trainer, TrainingArguments
from setfit.losses import SupConLoss
from setfit.trainer import ColumnMappingMixin, BCSentenceTransformersTrainer
from setfit.sampler import ContrastiveDataset, shuffle_combinations
from setfit.model_card import ModelCardCallback

import evaluate

from setfit import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)
from tqdm.autonotebook import trange

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Literal, Optional, Union, Any, Iterable

class SpanEmbeddingPooling(Pooling):
    """
    Subclass of sentence_transformers.models.Pooling.Pooling Performs pooling (max or mean) on the token embeddings of
    tokens in the span by using the span_mask to mask out embeddings of tokens that are not part of the span.

    Using pooling, it generates from a variable sized sentence a fixed sized span embedding.

    Args:
        word_embedding_dimension: Dimensions for the word embeddings
        pooling_mode: Either "max", "mean",
            "mean_sqrt_len_tokens", or "weightedmean". If set,
            overwrites the other pooling_mode_* settings
        pooling_mode_cls_token: Use the first token (CLS token) as text
            representations
            IMPORTANT: not supported and only kept for consistency
        pooling_mode_max_tokens: Use max in each dimension over all
            tokens.
        pooling_mode_mean_tokens: Perform mean-pooling
        pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but
            divide by sqrt(input_length).
        pooling_mode_weightedmean_tokens: Perform (position) weighted
            mean pooling. See `SGPT: GPT Sentence Embeddings for
            Semantic Search <https://arxiv.org/abs/2202.08904>`_.
        pooling_mode_lasttoken: Perform last token pooling. See `SGPT:
            GPT Sentence Embeddings for Semantic Search
            <https://arxiv.org/abs/2202.08904>`_ and `Text and Code
            Embeddings by Contrastive Pre-Training
            <https://arxiv.org/abs/2201.10005>`_.
            IMPORTANT: not supported and only kept for consistency
    """

    POOLING_MODES = (
        "max",
        "mean",
        "mean_sqrt_len_tokens",
        "weightedmean",
    )

    def __init__(self, *args, **kwargs):
        if kwargs.get('pooling_mode_cls_token', False):
            raise NotImplementedError("pooling_mode_cls_token is not supported for SpanEmbeddingPooling")
        if kwargs.get('pooling_mode_last_token', False):
            raise NotImplementedError("pooling_mode_last_token is not supported for SpanEmbeddingPooling")
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return f"SpanEmbeddingPooling({self.get_config_dict()})"

    @staticmethod
    def load(input_path) -> Pooling:
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        return SpanEmbeddingPooling(**config)

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        features["attention_mask"] = features["span_mask"].to(features["attention_mask"].device)
        # del features["span_mask"]
        output = super().forward(features)
        return output



def _process_span_inputs(inputs: List[InputItem]):
    """
    Helper function to process the input texts and spans into a list of tuples of texts and span character positions.

    Args:
        inputs (List[InputItem]): A list of tuples of texts to be tokenized and the span (positions) they contain.

    Returns:
        Tuple[List[str], List[Tuple[int, int]]]: A tuple of lists of texts and span character positions
    """
    sentences, spans = [], []
    for i, (t, s) in enumerate(inputs):
        sentences.append(t)
        if isinstance(s, str):
            m = regex.search(regex.escape(s), t)
            if m is None:
                raise ValueError(f"Could not find '{s}' in '{t}'.")
            spans.append(m.span())
        elif isinstance(s, (tuple, list)) and len(s) == 2 and all(isinstance(pos, int) for pos in s):
            if s[0] < 0 or s[1] > len(t) or s[0] >= s[1]:
                raise ValueError(f"Invalid span positions at index {i}: {s} for text of length {len(t)}.")
            spans.append(s)
        else:
            raise ValueError(f"Invalid span format at index {i}: {s}. Must be either a string or a tuple or list of two integers.")
        if isinstance(s, list):
            s = tuple(s)
    return sentences, spans


class SentenceTransformerForSpanEmbedding(SentenceTransformer):
    """
    SentenceTransformer model for span embedding.

    Args: check ``?sentence_transformers.SentenceTransformer``

    Example:
        ::

            model = SentenceTransformerForSpanEmbedding('all-mpnet-base-v2')
            sentences = [
                ("The weather is lovely today.", "weather"),
                ("It's so sunny outside!", "outside"),
                ("He drove to the stadium.", "He drove")
            ]
            embeddings = model.encode(sentences)
            print(embeddings.shape)
            # (3, 768)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_module_class_from_ref(
            self,
            class_ref: str,
            *args,
            **kwargs
    ) -> nn.Module:
        # use SpanEmbeddingPooling instead of Pooling for pooling module
        if class_ref == "sentence_transformers.models.Pooling":
            return SpanEmbeddingPooling
        # otherwise, use the default implementation
        return super()._load_module_class_from_ref(class_ref, *args, **kwargs)

    def tokenize(
        self,
        # NOTE: keeping the identical to super's encode() and hence using "texts" here
        texts: List[InputItem]
    ) -> Dict[str, Tensor]:
        """
        Tokenizes the texts.

        Args:
            texts (List[InputItem]): A list of tuples of texts to be tokenized
            and the span (positions) they contain.

        Returns:
            Dict[str, Tensor]: A dictionary of tensors with the tokenized texts. Common keys are "input_ids",
            "attention_mask", "token_type_ids", and "span_mask".
        """
        sentences, spans = _process_span_inputs(texts)

        # NOTE: super's tokenize() usually calls self._first_module().tokenize(), which applies some extra preprocessing
        #        to allow for differnt input formats. We mimic this here but add `return_offsets_mapping=True` to get the
        #        character locations of the tokens.
        features = self._first_module().tokenizer(
            sentences,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self._first_module().tokenizer.max_len_single_sentence,
            return_offsets_mapping=True, # <== added to get tokens character locations
        )

        # iterate over tokenized inputs and flag locations of spans
        # NOTE: this is added logic relative to super's implementation
        span_mask = torch.zeros(features['input_ids'].shape, dtype=features["attention_mask"].dtype)
        for i, span in enumerate(spans):
            inside = False
            for t, (om, am) in enumerate(zip(features['offset_mapping'][i], features['attention_mask'][i])):
                if am == 0:
                    break
                if span[0] in range(*om) and not inside:
                    span_mask[i][t] = 1
                    inside = True
                elif span[1]-1 in range(*om) and inside:
                    span_mask[i][t] = 1
                    inside = False
                elif inside:
                    span_mask[i][t] = 1
        features.update({'span_mask': span_mask.to(features["attention_mask"].device)})
        del features['offset_mapping']

        return features

    def encode(
        self,
        # NOTE: keeping the identical to super's encode() and hence using "sentences" here
        sentences: InputItem | List[InputItem],
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor:
        """
        Computes sentence embeddings.

        Args:
            sentences (InputItem | List[InputItem]):
                The tuple(s) of span(s) in sentence(s) to embed.
            prompt_name (Optional[str], optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
                which is either set in the constructor or loaded from the model configuration. For example if
                ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
                is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
                is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
            prompt (Optional[str], optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
                sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.
            batch_size (int, optional): The batch size used for the computation. Defaults to 32.
            show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences. Defaults to None.
            output_value (Optional[Literal["sentence_embedding", "token_embeddings"]], optional): The type of embeddings to return:
                "sentence_embedding" to get sentence embeddings, "token_embeddings" to get wordpiece token embeddings, and `None`,
                to get all output values. Defaults to "sentence_embedding".
            precision (Literal["float32", "int8", "uint8", "binary", "ubinary"], optional): The precision to use for the embeddings.
                Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions are quantized embeddings.
                Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy. They are useful for
                reducing the size of the embeddings of a corpus for semantic search, among other tasks. Defaults to "float32".
            convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.
                Defaults to True.
            convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
                Defaults to False.
            device (str, optional): Which :class:`torch.device` to use for the computation. Defaults to None.
            normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.

        Returns:
            Union[List[Tensor], ndarray, Tensor]: By default, a 2d numpy array with shape [num_inputs, output_dimension] is returned.
            If only one string input is provided, then the output is a 1d array with shape [output_dimension]. If ``convert_to_tensor``,
            a torch Tensor is returned instead. If ``self.truncate_dim <= output_dimension`` then output_dimension is ``self.truncate_dim``.

        Example:
            ::

                model = SentenceTransformerForSpanEmbedding('all-mpnet-base-v2')

                sentences = [
                    ("The weather is lovely today.", "weather)
                    ("It's so sunny outside!", "outside")
                    ("He drove to the stadium.", "He drove")
                ]
                embeddings = model.encode(sentences)
                print(embeddings.shape)
                # (3, 768)
        """
        if self.device.type == "hpu" and not self.is_hpu_graph_enabled:
            # import habana_frameworks.torch as ht
            #
            # ht.hpu.wrap_in_hpu_graph(self, disable_tensor_cache=True)
            # self.is_hpu_graph_enabled = True
            NotImplementedError("HPU is not supported in this version")

        self.eval()
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        single_input = False
        if isinstance(sentences, Tuple) or not hasattr(
            sentences[0], "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            single_input = True

        if prompt is None:
            if prompt_name is not None:
                try:
                    prompt = self.prompts[prompt_name]
                except KeyError:
                    raise ValueError(
                        f"Prompt name '{prompt_name}' not found in the configured prompts dictionary with keys {list(self.prompts.keys())!r}."
                    )
            elif self.default_prompt_name is not None:
                prompt = self.prompts.get(self.default_prompt_name, None)
        else:
            if prompt_name is not None:
                logger.warning(
                    "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                    "Ignoring the `prompt_name` in favor of `prompt`."
                )

        extra_features = {}
        if prompt is not None:
            prompt_len = len(prompt)
            for i, input in enumerate(sentences):
                span = input[1]
                if isinstance(span, tuple):
                    span[0] += prompt_len
                    span[1] += prompt_len
                    input[1] = span
                sentences[i] = (prompt + input[0], span)

            # Some models (e.g. INSTRUCTOR, GRIT) require removing the prompt before pooling
            # Tracking the prompt length allow us to remove the prompt during pooling
            tokenized_prompt = self.tokenize([prompt])
            if "input_ids" in tokenized_prompt:
                extra_features["prompt_length"] = tokenized_prompt["input_ids"].shape[-1] - 1

        if device is None:
            device = self.device

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(input[0]) for input in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch)
            if self.device.type == "hpu":
                if "input_ids" in features:
                    curr_tokenize_len = features["input_ids"].shape
                    additional_pad_len = 2 ** math.ceil(math.log2(curr_tokenize_len[1])) - curr_tokenize_len[1]
                    features["input_ids"] = torch.cat(
                        (
                            features["input_ids"],
                            torch.ones((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                        ),
                        -1,
                    )
                    features["attention_mask"] = torch.cat(
                        (
                            features["attention_mask"],
                            torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                        ),
                        -1,
                    )
                    features["span_mask"] = torch.cat(
                        (
                            features["span_mask"],
                            torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                        ),
                        -1,
                    )
                    if "token_type_ids" in features:
                        features["token_type_ids"] = torch.cat(
                            (
                                features["token_type_ids"],
                                torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                            ),
                            -1,
                        )

            features = batch_to_device(features, device)
            features.update(extra_features)

            with torch.no_grad():
                out_features = self.forward(features, **kwargs)
                if self.device.type == "hpu":
                    out_features = copy.deepcopy(out_features)

                out_features["sentence_embedding"] = truncate_embeddings(
                    out_features["sentence_embedding"], self.truncate_dim
                )

                if output_value == "token_embeddings":
                    # TODO: maybe make NotImplementedError
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if precision and precision != "float32":
            all_embeddings = quantize_embeddings(all_embeddings, precision=precision)

        if convert_to_tensor:
            if len(all_embeddings):
                if isinstance(all_embeddings, np.ndarray):
                    all_embeddings = torch.from_numpy(all_embeddings)
                else:
                    all_embeddings = torch.stack(all_embeddings)
            else:
                all_embeddings = torch.Tensor()
        elif convert_to_numpy:
            if not isinstance(all_embeddings, np.ndarray):
                if all_embeddings and all_embeddings[0].dtype == torch.bfloat16:
                    all_embeddings = np.asarray([emb.float().numpy() for emb in all_embeddings])
                else:
                    all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        elif isinstance(all_embeddings, np.ndarray):
            all_embeddings = [torch.from_numpy(embedding) for embedding in all_embeddings]

        if single_input:
            all_embeddings = all_embeddings[0]

        return all_embeddings

class SpanColumnMappingMixin(ColumnMappingMixin):
    _REQUIRED_COLUMNS = {"text", "span", "label"}

@dataclass
class SentenceTransformerDataCollatorForSpanClassification(SentenceTransformerDataCollator):

    required_features: list[str] = field(default_factory=lambda: ["sentence_1", "span_1", "sentence_2", "span_2", "label"])

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        column_names = list(features[0].keys())

        # TODO: implement this sanity check
        # if tuple(column_names) not in self.required_features:
        #     raise ValueError(
        #         f"Column names must be {self.required_features}. Got {column_names}."
        #     )

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        # if tuple(column_names) not in self._warned_columns:
        #     self.maybe_warn_about_column_order(column_names)

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        # # Extract the feature columns
        # for column_name in column_names:
        #     tokenized = self.tokenize_fn([row[column_name] for row in features])
        #     for key, value in tokenized.items():
        #         batch[f"{column_name}_{key}"] = value

        for idx in [1, 2]:
            inputs = [(row[f'sentence_{idx}'], row[f'span_{idx}']) for row in features]
            tokenized = self.tokenize_fn(inputs)
            for key, value in tokenized.items():
                batch[f"sentence_{idx}_{key}"] = value

        return batch


class ContrastiveDatasetForSpanEmbedding(ContrastiveDataset):

    def __init__(
            self,
            # NOTE: keeping the identical to super's encode() and hence using "sentences" here
            sentences: List[TextSpanInputItem],
            labels: List[Union[int, float]],
            multilabel: bool,
            num_iterations: Optional[None] = None,
            sampling_strategy: str = "oversampling",
            max_pairs: int = -1,
        ):
        IterableDataset.__init__(self)
        self.pos_index = 0
        self.neg_index = 0
        self.pos_pairs = []
        self.neg_pairs = []
        self.sentences = [t for t, _ in sentences]
        self.spans = [s for _, s in sentences]
        self.labels = labels
        self.sentence_labels = list(zip(self.sentences, self.spans, self.labels))
        self.max_pos_or_neg = -1 if max_pairs == -1 else max_pairs // 2

        if multilabel:
            self.generate_multilabel_pairs()
        else:
            self.generate_pairs()

        if num_iterations is not None and num_iterations > 0:
            self.len_pos_pairs = num_iterations * len(self.sentences)
            self.len_neg_pairs = num_iterations * len(self.sentences)

        elif sampling_strategy == "unique":
            self.len_pos_pairs = len(self.pos_pairs)
            self.len_neg_pairs = len(self.neg_pairs)

        elif sampling_strategy == "undersampling":
            self.len_pos_pairs = min(len(self.pos_pairs), len(self.neg_pairs))
            self.len_neg_pairs = min(len(self.pos_pairs), len(self.neg_pairs))

        elif sampling_strategy == "oversampling":
            self.len_pos_pairs = max(len(self.pos_pairs), len(self.neg_pairs))
            self.len_neg_pairs = max(len(self.pos_pairs), len(self.neg_pairs))

        else:
            raise ValueError("Invalid sampling strategy. Must be one of 'unique', 'oversampling', or 'undersampling'.")


    def generate_pairs(self) -> None:
        for (_text, _span, _label), (text, span, label) in shuffle_combinations(self.sentence_labels):
            is_positive = _label == label
            is_positive_full = self.max_pos_or_neg != -1 and len(self.pos_pairs) >= self.max_pos_or_neg
            is_negative_full = self.max_pos_or_neg != -1 and len(self.neg_pairs) >= self.max_pos_or_neg

            if is_positive:
                if not is_positive_full:
                    self.pos_pairs.append({"sentence_1": _text, "span_1": _span, "sentence_2": text, "span_2": span, "label": 1.0})
            elif not is_negative_full:
                self.neg_pairs.append({"sentence_1": _text, "span_1": _span, "sentence_2": text, "span_2": span, "label": 0.0})

            if is_positive_full and is_negative_full:
                break

    def generate_multilabel_pairs(self) -> None:
        for (_text, _span, _label), (text, span, label) in shuffle_combinations(self.sentence_labels):
            # logical_and checks if labels are both set for each class
            is_positive = any(np.logical_and(_label, label))
            is_positive_full = self.max_pos_or_neg != -1 and len(self.pos_pairs) >= self.max_pos_or_neg
            is_negative_full = self.max_pos_or_neg != -1 and len(self.neg_pairs) >= self.max_pos_or_neg

            if is_positive:
                if not is_positive_full:
                    self.pos_pairs.append({"sentence_1": _text, "span_1": _span, "sentence_2": text, "span_2": span, "label": 1.0})
            elif not is_negative_full:
                self.neg_pairs.append({"sentence_1": _text, "span_1": _span, "sentence_2": text, "span_2": span, "label": 0.0})

            if is_positive_full and is_negative_full:
                break


class SetFitDatasetForSpanClassification(TorchDataset):
    """SetFitDatasetForSpanClassification

    A dataset for training the differentiable head on span classification.

    Args:
        x (`List[Tuple[str, Tuple[int, int]]]`):
            A list of input data as tuples of texts and span start and end character positions that will be fed into `SetFitModel`.
        y (`Union[List[int], List[List[int]]]`):
            A list of input data's labels. Can be a nested list for multi-label classification.
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer from `SetFitModel`'s body.
        max_length (`int`, defaults to `32`):
            The maximum token length a tokenizer can generate.
            Will pad or truncate tokens when the number of tokens for a text is either smaller or larger than this value.
    """

    def __init__(
        self,
        x: List[TextSpanInputItem],
        y: Union[List[int], List[List[int]]],
        tokenizer: "PreTrainedTokenizerBase",
        max_length: int = 32,
    ) -> None:
        assert len(x) == len(y)

        # TODO (maybe): use _process_inputs to extract text and span from input
        self.sentences, self.spans = _process_span_inputs(x)
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[TokenizerOutput, Union[int, List[int]]]:
        label = self.y[idx]

        feature = self.tokenizer(
            self.sentences[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask="attention_mask" in self.tokenizer.model_input_names,
            return_token_type_ids="token_type_ids" in self.tokenizer.model_input_names,
            return_offsets_mapping=True, # <== added to get tokens character locations
        )
        # iterate over tokenized inputs and flag locations of spans
        span = self.spans[idx]
        span_mask = [0]*len(feature['input_ids'])
        inside = False
        for t, (om, am) in enumerate(zip(feature['offset_mapping'], feature['attention_mask'])):
            if am == 0:
                break
            if span[0] in range(*om) and not inside:
                span_mask[t] = 1
                inside = True
            elif span[1]-1 in range(*om) and inside:
                span_mask[t] = 1
                inside = False
            elif inside:
                span_mask[t] = 1
        feature.update({'span_mask': span_mask})

        del feature['offset_mapping']

        return feature, label

    def collate_fn(self, batch):
        features = {input_name: [] for input_name in self.tokenizer.model_input_names + ['span_mask']}

        labels = []
        for feature, label in batch:
            features["input_ids"].append(feature["input_ids"])
            if "attention_mask" in features:
                features["attention_mask"].append(feature["attention_mask"])
            if "token_type_ids" in features:
                features["token_type_ids"].append(feature["token_type_ids"])
            if "span_mask" in features:
                features["span_mask"].append(feature["span_mask"])
            labels.append(label)

        # convert to tensors
        features = {k: torch.Tensor(v).int() for k, v in features.items()}
        labels = torch.Tensor(labels)
        labels = labels.long() if len(labels.size()) == 1 else labels.float()
        return features, labels

from src.finetuning.setfit_extensions.early_stopping import SetFitModelWithEarlyStopping
class SetFitModelForSpanClassification(SetFitModelWithEarlyStopping):
    def _prepare_dataloader(
            self,
            x_train: List[TextSpanInputItem],
            y_train: Union[List[int], List[List[int]]],
            batch_size: Optional[int] = None,
            max_length: Optional[int] = None,
            shuffle: bool = True,
        ) -> DataLoader:
            max_acceptable_length = self.model_body.get_max_seq_length()
            if max_length is None:
                max_length = max_acceptable_length
                logger.warning(
                    f"The `max_length` is `None`. Using the maximum acceptable length according to the current model body: {max_length}."
                )

            if max_length > max_acceptable_length:
                logger.warning(
                    (
                        f"The specified `max_length`: {max_length} is greater than the maximum length of the current model body: {max_acceptable_length}. "
                        f"Using {max_acceptable_length} instead."
                    )
                )
                max_length = max_acceptable_length

            dataset = SetFitDatasetForSpanClassification(
                x_train,
                y_train,
                tokenizer=self.model_body.tokenizer,
                max_length=max_length,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                pin_memory=True,
            )

            return dataloader
    
    @staticmethod
    def _normalize_inputs(
        inputs: Optional[List[InputItem]] = None,
        *,
        texts: Optional[List[Text]] = None,
        spans: Optional[List[Span]] = None,
        span_texts: Optional[List[Text]] = None,
    ) -> List[InputItem]:
        """Normalize all supported input styles to [(text, (start, end)), ...]."""
        modes = [
            inputs is not None,
            texts is not None and spans is not None,
            texts is not None and span_texts is not None,
        ]
        if sum(modes) != 1:
            raise ValueError(
                "Provide exactly one of: "
                "`inputs`, or `texts`+`spans`, or `texts`+`span_texts`."
            )

        if inputs is not None:
            return inputs

        if texts is None:
            raise ValueError("`texts` must be provided when `inputs` is None.")

        if spans is not None:
            if len(texts) != len(spans):
                raise ValueError("`texts` and `spans` must have the same length.")
            return list(zip(texts, spans))

        # texts + span_texts
        if span_texts is None:
            raise ValueError("`span_texts` must be provided with `texts`.")
        if len(texts) != len(span_texts):
            raise ValueError("`texts` and `span_texts` must have the same length.")

        norm_inputs: List[InputItem] = []
        for t, st in zip(texts, span_texts):
            start = t.find(st)
            if start < 0:
                raise ValueError(f"Span text {st!r} not found in text: {t[:80]!r}...")
            end = start + len(st)
            norm_inputs.append((t, (start, end)))
        return norm_inputs

    @overload
    def predict(self, inputs: List[InputItem], **kwargs) -> Union[List[int], List[List[int]], np.ndarray]: ...
    @overload
    def predict(self, inputs: None = ..., *, texts: List[Text], spans: List[Span], **kwargs) -> Union[List[int], List[List[int]], np.ndarray]: ...
    @overload
    def predict(self, inputs: None = ..., *, texts: List[Text], span_texts: List[str], **kwargs) -> Union[List[int], List[List[int]], np.ndarray]: ...
    def predict(
        self,
        inputs: Optional[List[InputItem]] = None,
        *,
        texts: Optional[List[Text]] = None,
        spans: Optional[List[Span]] = None,
        span_texts: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[List[int], List[List[int]], np.ndarray]:
        """
        Accepts one of:
          - inputs=[(text, (start, end)), ...]
          - texts=[text, ...], spans=[(start, end), ...]
          - texts=[text, ...], span_texts=[phrase, ...]
        """
        # Validate mutually exclusive interfaces
        norm_inputs = self._normalize_inputs(inputs, texts=texts, spans=spans, span_texts=span_texts)
        return super().predict(norm_inputs, **kwargs)
    
    @overload
    def predict_proba(self, inputs: List[InputItem], **kwargs) -> np.ndarray: ...
    @overload
    def predict_proba(self, inputs: None = ..., *, texts: List[Text], spans: List[Span], **kwargs) -> np.ndarray: ...
    @overload
    def predict_proba(self, inputs: None = ..., *, texts: List[Text], span_texts: List[str], **kwargs) -> np.ndarray: ...
    def predict_proba(
            self, 
            inputs: Optional[List[InputItem]] = None, 
            *, 
            texts: Optional[List[Text]] = None, 
            spans: Optional[List[Span]] = None,
            span_texts: Optional[List[str]] = None, 
            **kwargs
        ) -> np.ndarray:
        """
        Accepts one of:
          - inputs=[(text, (start, end)), ...]
          - texts=[text, ...], spans=[(start, end), ...]
          - texts=[text, ...], span_texts=[phrase, ...]
        """
        norm_inputs = self._normalize_inputs(inputs, texts=texts, spans=spans, span_texts=span_texts)
        return super().predict_proba(norm_inputs, **kwargs)


from src.finetuning.setfit_extensions.early_stopping import SetFitEarlyStoppingTrainer
class SetFitTrainerForSpanClassification(SetFitEarlyStoppingTrainer, SpanColumnMappingMixin):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # capture the callbacks
        # NOTE: mabye better to get callbacks from self.st_trainer (`trainer.st_trainer.callback_handler.callbacks`)
        callbacks = kwargs.get("callbacks", None)
        callbacks = callbacks + [ModelCardCallback(self)] if callbacks else [ModelCardCallback(self)]

        # re-init the sentence transformer trainer (used when calling `self.train_embeddings()``)
        self.st_trainer = BCSentenceTransformersTrainer(
            setfit_model=self.model,
            setfit_args=self.args,
            callbacks=callbacks,
            # NOTE: now using appropraite data collator
            data_collator=SentenceTransformerDataCollatorForSpanClassification(tokenize_fn=self.model.model_body.tokenize),
        )

    def _dataset_format_inputs(self, dataset: Dataset) -> List[Tuple[str, Tuple[int, int]]]:
        texts = dataset['text']
        spans = dataset['span']
        # spans = [tuple(span) if isinstance(span, list) else span for span in spans]
        return list(zip(texts, spans))

    def dataset_to_parameters(self, dataset: Dataset) -> List[Iterable]:
        return [ self._dataset_format_inputs(dataset), dataset['label'] ]

    def get_dataset(
        self, x: List[Tuple[str, Tuple[int, int]]], y: Union[List[int], List[List[int]]], args: TrainingArguments, max_pairs: int = -1
    ) -> Tuple[Dataset, nn.Module, int, int]:
        if args.loss in [
            losses.BatchAllTripletLoss,
            losses.BatchHardTripletLoss,
            losses.BatchSemiHardTripletLoss,
            losses.BatchHardSoftMarginTripletLoss,
            SupConLoss,
        ]:
            dataset = Dataset.from_dict({"sentence": [d[0] for d in x], "span": [d[1] for d in x], "label": y})

            if args.loss is losses.BatchHardSoftMarginTripletLoss:
                loss = args.loss(
                    model=self.model.model_body,
                    distance_metric=args.distance_metric,
                )
            elif args.loss is SupConLoss:
                loss = args.loss(model=self.model.model_body)
            else:
                loss = args.loss(
                    model=self.model.model_body,
                    distance_metric=args.distance_metric,
                    margin=args.margin,
                )
        else:
            data_sampler = ContrastiveDatasetForSpanEmbedding(
                x,
                y,
                self.model.multi_target_strategy,
                args.num_iterations,
                args.sampling_strategy,
                max_pairs=max_pairs,
            )
            dataset = Dataset.from_list(list(data_sampler))
            loss = args.loss(self.model.model_body)

        return dataset, loss

    def evaluate(self, dataset: Optional[Dataset] = None, metric_key_prefix: str = "test") -> Dict[str, float]:
        """
        Computes the metrics for a given classifier.

        Args:
            dataset (`Dataset`, *optional*):
                The dataset to compute the metrics on. If not provided, will use the evaluation dataset passed via
                the `eval_dataset` argument at `Trainer` initialization.

        Returns:
            `Dict[str, float]`: The evaluation metrics.
        """

        if dataset is not None:
            self._validate_column_mapping(dataset)
            if self.column_mapping is not None:
                logger.info("Applying column mapping to the evaluation dataset")
                eval_dataset = self._apply_column_mapping(dataset, self.column_mapping)
            else:
                eval_dataset = dataset
        else:
            eval_dataset = self.eval_dataset

        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided to `Trainer.evaluate` nor the `Trainer` initialzation.")

        # NOTE: Below is the _only_ line that differs from the parent class
        x_test = self._dataset_format_inputs(eval_dataset)
        y_test = eval_dataset["label"]

        logger.info("***** Running evaluation *****")
        y_pred = self.model.predict(x_test, use_labels=False)
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu()

        # Normalize string outputs
        if y_test and isinstance(y_test[0], str):
            encoder = LabelEncoder()
            encoder.fit(list(y_test) + list(y_pred))
            y_test = encoder.transform(y_test)
            y_pred = encoder.transform(y_pred)

        metric_kwargs = self.metric_kwargs or {}
        if isinstance(self.metric, str):
            metric_config = "multilabel" if self.model.multi_target_strategy is not None else None
            metric_fn = evaluate.load(self.metric, config_name=metric_config)

            results = metric_fn.compute(predictions=y_pred, references=y_test, **metric_kwargs)

        elif callable(self.metric):
            results = self.metric(y_pred, y_test, **metric_kwargs)

        else:
            raise ValueError("metric must be a string or a callable")

        if not isinstance(results, dict):
            results = {"metric": results}
        self.model.model_card_data.post_training_eval_results(
            {f"{metric_key_prefix}_{key}": value for key, value in results.items()}
        )
        return results

