import pytest
import torch

from spacebyte.losses import masked_binary_cross_entropy


def test_masked_bce_ignores_unobserved_entries():
    logits = torch.tensor([[0.0, 100.0]])
    targets = torch.tensor([[1.0, 0.0]])
    masks = torch.tensor([[1.0, 0.0]])
    loss = masked_binary_cross_entropy(logits, targets, masks)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_masked_bce_rejects_empty_mask():
    with pytest.raises(ValueError, match="At least one label"):
        masked_binary_cross_entropy(
            torch.zeros(1, 9), torch.zeros(1, 9), torch.zeros(1, 9)
        )
