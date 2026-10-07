"""Masked binary cross entropy for partially annotated relations."""

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_binary_cross_entropy(
    logits: Tensor, targets: Tensor, masks: Tensor
) -> Tensor:
    if logits.shape != targets.shape or targets.shape != masks.shape:
        raise ValueError(
            f"Expected identical shapes, got logits={logits.shape}, "
            f"targets={targets.shape}, masks={masks.shape}"
        )
    denominator = masks.sum()
    if denominator.item() <= 0:
        raise ValueError("At least one label must be observed in each batch")
    elementwise = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (elementwise * masks).sum() / denominator
