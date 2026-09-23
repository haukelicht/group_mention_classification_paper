
import torch

from typing import Mapping, Optional, Union
from numpy._typing import NDArray

# default setfit body and head
from sentence_transformers import SentenceTransformer
from setfit.modeling import SetFitHead

# class weight head
from .class_weights_head import SetFitHeadWithClassWeights
from .early_stopping import SetFitModelWithEarlyStopping
from .span_embedding import (
    SentenceTransformerForSpanEmbedding,
    SetFitModelForSpanClassification,
)

def model_init(
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        id2label: Optional[Mapping[int, str]] = None,
        # num_classes: int = 2,
        class_weights: NDArray = None,
        multilabel: bool = False,
        use_span_embedding: bool = False,
        body_kwargs: dict = {},
        head_kwargs: dict = {},
        model_kwargs: dict = {},
        **kwargs
    ) -> SetFitModelWithEarlyStopping | SetFitModelForSpanClassification:
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
    

    model_class = SetFitModelForSpanClassification if use_span_embedding else SetFitModelWithEarlyStopping
    if multilabel and "multi_target_strategy" not in model_kwargs:
        model_kwargs["multi_target_strategy"] = "one-vs-rest"
    model = model_class(
        model_body=body,
        model_head=head.to(body.device),
        normalize_embeddings=True,
        labels=list(id2label.values()) if id2label is not None else None,
        **model_kwargs
    ).to(body.device)

    return model
