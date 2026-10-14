"""Stopping, selection, and paired-bootstrap helpers for stability training."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def metric_row_is_stable(row: dict[str, Any]) -> bool:
    """Return whether one epoch satisfies every predeclared stability guard."""

    metrics = row["evaluations"]
    normal = metrics["normal"]["logit_yes_minus_no"]
    zero = metrics["zero_byte"]["logit_yes_minus_no"]
    shuffled = metrics["shuffled_byte"]["logit_yes_minus_no"]
    confusion = normal["confusion_matrix"]
    predicts_both = (confusion["tp"] + confusion["fp"] > 0) and (
        confusion["tn"] + confusion["fn"] > 0
    )
    finite = bool(row.get("all_finite", True))
    return (
        finite
        and predicts_both
        and normal["macro_f1"] >= 0.50
        and normal["auroc"] - max(zero["auroc"], shuffled["auroc"]) >= 0.02
    )


def best_epoch(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select by Normal AUROC, then Macro-F1, retaining the earlier exact tie."""

    if not rows:
        raise ValueError("Cannot select a best epoch from an empty sequence")
    selected = rows[0]
    for candidate in rows[1:]:
        current = selected["evaluations"]["normal"]["logit_yes_minus_no"]
        proposed = candidate["evaluations"]["normal"]["logit_yes_minus_no"]
        if (proposed["auroc"], proposed["macro_f1"]) > (
            current["auroc"],
            current["macro_f1"],
        ):
            selected = candidate
    return selected


def stopping_reason(
    history: Sequence[dict[str, Any]],
    *,
    first_new_epoch: int = 4,
    max_epoch: int = 10,
    plateau_delta: float = 0.002,
) -> str | None:
    """Apply stability, three-new-epoch plateau, then maximum-epoch stopping."""

    if len(history) >= 2 and all(metric_row_is_stable(row) for row in history[-2:]):
        return "stability_confirmed_two_consecutive_epochs"

    new_rows = [row for row in history if row["epoch"] >= first_new_epoch]
    if len(new_rows) >= 3:
        previous_rows = [row for row in history if row["epoch"] < new_rows[-3]["epoch"]]
        previous_best = max(
            (
                row["evaluations"]["normal"]["logit_yes_minus_no"]["auroc"]
                for row in previous_rows
            ),
            default=float("-inf"),
        )
        improvements = []
        running_best = previous_best
        for row in new_rows[-3:]:
            value = row["evaluations"]["normal"]["logit_yes_minus_no"]["auroc"]
            next_best = max(running_best, value)
            improvements.append(next_best - running_best)
            running_best = next_best
        if all(value < plateau_delta for value in improvements):
            return "plateau"

    if history and history[-1]["epoch"] >= max_epoch:
        return "max_epochs_without_stability"
    return None


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware rank AUROC without an sklearn runtime dependency."""

    labels = labels.astype(bool, copy=False)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def paired_bootstrap(
    labels: Sequence[bool],
    normal_scores: Sequence[float],
    zero_scores: Sequence[float],
    shuffled_scores: Sequence[Sequence[float]],
    *,
    iterations: int = 2000,
    seed: int = 20261013,
) -> dict[str, Any]:
    """Bootstrap Normal AUC and paired AUC differences over shared rows."""

    label_array = np.asarray(labels, dtype=bool)
    normal = np.asarray(normal_scores, dtype=np.float64)
    zero = np.asarray(zero_scores, dtype=np.float64)
    shuffled = [np.asarray(values, dtype=np.float64) for values in shuffled_scores]
    arrays = [normal, zero, *shuffled]
    if not shuffled:
        raise ValueError("At least one shuffled score vector is required")
    if label_array.size == 0 or any(values.shape != label_array.shape for values in arrays):
        raise ValueError("All paired bootstrap arrays must have equal nonzero length")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    rng = np.random.default_rng(seed)
    normal_values: list[float] = []
    normal_minus_zero: list[float] = []
    normal_minus_shuffled_mean: list[float] = []
    discarded = 0
    for _ in range(iterations):
        indices = rng.integers(0, label_array.size, size=label_array.size)
        sampled_labels = label_array[indices]
        normal_auc = _auc(sampled_labels, normal[indices])
        zero_auc = _auc(sampled_labels, zero[indices])
        shuffled_aucs = [_auc(sampled_labels, values[indices]) for values in shuffled]
        if not np.isfinite([normal_auc, zero_auc, *shuffled_aucs]).all():
            discarded += 1
            continue
        normal_values.append(normal_auc)
        normal_minus_zero.append(normal_auc - zero_auc)
        normal_minus_shuffled_mean.append(normal_auc - float(np.mean(shuffled_aucs)))

    if not normal_values:
        raise RuntimeError("Every bootstrap resample was invalid")

    def summarize(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        low, high = np.percentile(array, [2.5, 97.5])
        return {
            "mean": float(array.mean()),
            "lower_95": float(low),
            "upper_95": float(high),
        }

    return {
        "iterations_requested": iterations,
        "iterations_valid": len(normal_values),
        "iterations_discarded": discarded,
        "seed": seed,
        "normal_auroc": summarize(normal_values),
        "normal_minus_zero_auroc": summarize(normal_minus_zero),
        "normal_minus_mean_shuffled_auroc": summarize(normal_minus_shuffled_mean),
    }


def conclusion(
    normal_auroc: float,
    zero_auroc: float,
    mean_shuffled_auroc: float,
    bootstrap: dict[str, Any],
) -> str:
    """Classify the predeclared visual-evidence conclusion."""

    point_passes = (
        normal_auroc - zero_auroc >= 0.02
        and normal_auroc - mean_shuffled_auroc >= 0.02
    )
    if not point_passes:
        return "not_confirmed"
    lower_zero = bootstrap["normal_minus_zero_auroc"]["lower_95"]
    lower_shuffled = bootstrap["normal_minus_mean_shuffled_auroc"]["lower_95"]
    return "confirmed" if lower_zero > 0 and lower_shuffled > 0 else "tentative"
