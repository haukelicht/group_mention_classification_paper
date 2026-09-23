"""A SetFit head that supports class-weights aware multi-class classification for end-to-end training."""
import numpy as np
import torch
from torch import nn
from setfit import SetFitHead
from typing import List, Dict, Optional, Union, Literal
from numpy._typing import NDArray

def compute_class_weights(
        x: Union[List[Union[int, str]], NDArray, torch.Tensor], 
        multitarget: bool=False,
        smooth_weights: bool=False,
        smooth_exponent: float=0.5,
    ) -> NDArray:
    """Compute class weights from target labels.

    The function computes class weights inversely proportional to the frequency of positive instances for each class. 
    For multitarget classification, it assumes that the target labels are binary indicators (0/1) for each class.

    Args:
        x (Union[List[Union[int, str]], NDArray, torch.Tensor]):
            The target labels. For multitarget, shape should be (num_samples, num_targets).
        multitarget (bool, defaults to `False`):
            Whether to compute class weights for multitarget classification.
        smooth_weights (bool, defaults to `False`):
            Whether to apply smoothing to the computed class weights.
            If set to `True`, the weights will be smoothed with the root function, i.e., using the formula:
            `w_smooth = w ** smooth_exponent`.
        smooth_exponent (float, defaults to `0.5`):
            The exponent used for smoothing the class weights if `smooth_weights` is `True`.
            A value must in [0.0, 1.0].
            A value of 1 will result in no smoothing (i.e., uniform weights across classes).
            A value of 0.5 will apply square root smoothing.
            A value of 0 will switch off smoothing and keep weights as-is.

    Returns:
        NDArray:
            The computed class weights.
    """
    # --- input to numpy ---
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.array(x)

    # --- basic checks ---
    if x.size == 0:
        raise ValueError("x is empty")
    if multitarget:
        if x.ndim != 2:
            raise ValueError("if multitarget=True, x.ndim must be 2")
        if not np.issubdtype(x.dtype, np.number):
            raise ValueError("multi-label x must be numeric (0/1)")
    else:
        if x.ndim != 1:
            raise ValueError("if multitarget=False, x.ndim must be 1")

    if smooth_weights and not (0.0 <= smooth_exponent <= 1.0):
        raise ValueError("smooth_exponent must be in [0.0, 1.0]")

    # helper: apply smoothing exponent; alpha=0 -> ones; alpha=1 -> identity
    def _temper(w: NDArray) -> NDArray:
        if not smooth_weights:
            return w
        if smooth_exponent == 0.0:
            return np.ones_like(w, dtype=float)
        return w.astype(float) ** float(smooth_exponent)

    # --- multi-label ---
    if multitarget:
        prevs = x.mean(axis=0).astype(float)  # p_k
        eps = 1e-12
        prevs = np.clip(prevs, eps, 1 - eps)
        w = (1.0 - prevs) / prevs              # (neg:pos)
        w = _temper(w)                         # ((1-p)/p)^alpha
        return w.astype(np.float64)

    # --- single-label ---
    classes, cnts = np.unique(x, return_counts=True)
    if len(classes)<2:
        raise ValueError("binary_bce requires exactly 2 classes in single-label targets.")

    freqs = cnts.astype(float) / cnts.sum()    # p_k
    eps = 1e-12
    freqs = np.clip(freqs, eps, 1.0)

    w = 1.0 / freqs
    w = _temper(w)
    w = w * (w.size / w.sum())
    return w.astype(np.float64)
    

class SetFitHeadWithClassWeights(SetFitHead):
    """
    A SetFit head that supports class-weights aware multi-class classification for end-to-end training.
    Binary classification is treated as 2-class classification.

    Args:
        in_features (`int`, *optional*):
            The embedding dimension from the output of the SetFit body. If `None`, defaults to `LazyLinear`.
        out_features (`int`, defaults to `2`):
            The number of targets. If set `out_features` to 1 for binary classification, it will be changed to 2 as 2-class classification.
        temperature (`float`, defaults to `1.0`):
            A logits' scaling factor. Higher values make the model less confident and lower values make
            it more confident.
        eps (`float`, defaults to `1e-5`):
            A value for numerical stability when scaling logits.
        bias (`bool`, *optional*, defaults to `True`):
            Whether to add bias to the head.
        device (`torch.device`, str, *optional*):
            The device the model will be sent to. If `None`, will check whether GPU is available.
        multitarget (`bool`, defaults to `False`):
            Enable multi-target classification by making `out_features` binary predictions instead
            of a single multinomial prediction.
        class_weights (`List[float]`, `numpy.typing.NDarray`, *optional*):
    """

    def __init__(
        self,
        class_weights: Optional[Union[List[float], NDArray]] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if class_weights is None:
            # No class weighting
            self.class_weights = None
            return

        # Convert safely to a FLOAT tensor on the correct device
        # (avoid passing a Python list/sequence to PyTorch loss)
        w = torch.as_tensor(class_weights, dtype=torch.float32)
        if w.ndim != 1:
            raise ValueError("`class_weights` must be 1D (length = out_features).")

        if w.numel() != self.out_features:
            raise ValueError(
                f"length of `class_weights` ({w.numel()}) must equal `out_features` ({self.out_features})"
            )

        # Move to same device/dtype as the linear layer
        w = w.to(device=self.linear.weight.device, dtype=self.linear.weight.dtype)

        # Register as a buffer so it follows .to(device), DDP, etc.
        self.register_buffer("class_weights", w, persistent=False)

    def get_loss_fn(self) -> nn.Module:
        if self.multitarget:  # if sigmoid output
            return nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        return nn.CrossEntropyLoss(weight=self.class_weights)

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the model is placed.

        Reference from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L869
        """
        return next(self.parameters()).device
    
    def to(self, device: Union[str, torch.device]) -> "SetFitHeadWithClassWeights":
        """Move this SetFitHeadWithClassWeights to `device`, and then return `self`. This method does not copy.

        Args:
            device (Union[str, torch.device]): The identifier of the device to move the model to.

        Returns:
            SetFitHeadWithClassWeights: Returns the original model, but now on the desired device.
        """
        self.linear = self.linear.to(device)
        if hasattr(self, "class_weights"):
            self.class_weights = self.class_weights.to(device)
        return self
    
    def predict_proba(self, x_test: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            probs = self(x_test)[1]
        return probs
    
    def get_config_dict(self) -> Dict[str, Optional[Union[int, float, bool, List[float]]]]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "temperature": self.temperature,
            "bias": self.bias,
            "device": self.device.type,
            "multitarget": self.multitarget,
            "class_weights": self.class_weights.cpu().numpy().round(3).tolist()
        }

    def __repr__(self) -> str:
        return "SetFitHeadWithClassWeights({})".format(self.get_config_dict())
