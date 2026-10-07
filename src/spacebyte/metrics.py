"""Metrics restricted to observed SpatialSense labels."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)

from spacebyte.constants import RELATIONS


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    thresholds: torch.Tensor | None = None,
) -> dict[str, Any]:
    probabilities = torch.sigmoid(logits).cpu().numpy()
    target_array = targets.cpu().numpy().astype(int)
    mask_array = masks.cpu().numpy().astype(bool)
    if thresholds is None:
        threshold_array = np.full(len(RELATIONS), 0.5)
    else:
        threshold_array = thresholds.cpu().numpy()

    per_relation: dict[str, Any] = {}
    macro_values: dict[str, list[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "f1": [],
        "auroc": [],
    }
    for index, relation in enumerate(RELATIONS):
        observed = mask_array[:, index]
        if not observed.any():
            continue
        truth = target_array[observed, index]
        scores = probabilities[observed, index]
        predicted = (scores >= threshold_array[index]).astype(int)
        values = {
            "count": int(observed.sum()),
            "accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "f1": float(f1_score(truth, predicted, zero_division=0)),
            "auroc": (
                float(roc_auc_score(truth, scores))
                if len(np.unique(truth)) == 2
                else None
            ),
        }
        per_relation[relation] = values
        for name in macro_values:
            if values[name] is not None:
                macro_values[name].append(values[name])

    observed_all = mask_array
    truth_all = target_array[observed_all]
    score_matrix = probabilities >= threshold_array.reshape(1, -1)
    predicted_all = score_matrix[observed_all].astype(int)
    return {
        "observed_labels": int(observed_all.sum()),
        "micro_accuracy": float(accuracy_score(truth_all, predicted_all)),
        "micro_f1": float(f1_score(truth_all, predicted_all, zero_division=0)),
        "macro": {
            name: float(np.mean(values)) if values else None
            for name, values in macro_values.items()
        },
        "per_relation": per_relation,
    }
