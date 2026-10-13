"""Evaluation primitives for binary SpatialSense answer generation."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor


_STRICT_BINARY = re.compile(r"^\s*(yes|no)\s*[.!]?\s*$", re.IGNORECASE)


def parse_greedy_binary(text: str) -> str:
    """Map a complete short generation to yes, no, or invalid."""

    match = _STRICT_BINARY.fullmatch(text)
    return match.group(1).lower() if match else "invalid"


def fixed_derangement(
    size: int, seed: int, identity_keys: Sequence[str] | None = None
) -> list[int]:
    """Create a deterministic permutation with no sample or visual self mapping."""

    if size < 2:
        raise ValueError("A derangement requires at least two samples")
    if identity_keys is not None and len(identity_keys) != size:
        raise ValueError("identity_keys length does not match size")
    rng = random.Random(seed)
    for _ in range(10_000):
        order = list(range(size))
        rng.shuffle(order)
        if identity_keys is not None and any(
            identity_keys[order[position]]
            == identity_keys[order[(position + 1) % size]]
            for position in range(size)
        ):
            continue
        mapping = list(range(size))
        for position, target in enumerate(order):
            mapping[target] = order[(position + 1) % size]
        return mapping
    raise RuntimeError("Could not construct a non-self visual permutation")


def auroc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Tie-aware binary AUROC expressed as positive/negative pair ordering."""

    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def binary_classification_metrics(
    labels: Sequence[bool], predictions: Sequence[bool], scores: Sequence[float]
) -> dict[str, Any]:
    if not (len(labels) == len(predictions) == len(scores)) or not labels:
        raise ValueError("labels, predictions, and scores must have equal nonzero length")
    tp = sum(label and pred for label, pred in zip(labels, predictions))
    tn = sum((not label) and (not pred) for label, pred in zip(labels, predictions))
    fp = sum((not label) and pred for label, pred in zip(labels, predictions))
    fn = sum(label and (not pred) for label, pred in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "sample_count": len(labels),
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "auroc": auroc(labels, scores),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def greedy_metrics(
    labels: Sequence[bool], predictions: Sequence[str], relations: Sequence[str]
) -> dict[str, Any]:
    if not (len(labels) == len(predictions) == len(relations)) or not labels:
        raise ValueError("Greedy inputs must have equal nonzero length")
    allowed = {"yes", "no", "invalid"}
    if not set(predictions).issubset(allowed):
        raise ValueError("Greedy prediction outside yes/no/invalid")
    confusion = {
        "yes": {"yes": 0, "no": 0, "invalid": 0},
        "no": {"yes": 0, "no": 0, "invalid": 0},
    }
    for label, prediction in zip(labels, predictions):
        confusion["yes" if label else "no"][prediction] += 1

    class_metrics = {}
    for class_name in ("yes", "no"):
        true_class = [label if class_name == "yes" else not label for label in labels]
        predicted_class = [prediction == class_name for prediction in predictions]
        tp = sum(a and b for a, b in zip(true_class, predicted_class))
        fp = sum((not a) and b for a, b in zip(true_class, predicted_class))
        fn = sum(a and (not b) for a, b in zip(true_class, predicted_class))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        class_metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
        }
    per_relation = {}
    for relation in sorted(set(relations)):
        indices = [index for index, value in enumerate(relations) if value == relation]
        per_relation[relation] = {
            "sample_count": len(indices),
            "accuracy": sum(
                predictions[index] == ("yes" if labels[index] else "no")
                for index in indices
            )
            / len(indices),
        }
    return {
        "sample_count": len(labels),
        "exact_match_accuracy": sum(
            prediction == ("yes" if label else "no")
            for label, prediction in zip(labels, predictions)
        )
        / len(labels),
        "invalid_rate": predictions.count("invalid") / len(labels),
        "class_metrics": class_metrics,
        "macro_f1": sum(row["f1"] for row in class_metrics.values()) / 2,
        "confusion_matrix": confusion,
        "per_relation": per_relation,
    }


def conditional_log_likelihood(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Return answer log-probability sums and supervised token counts per row."""

    if logits.shape[:2] != labels.shape:
        raise ValueError("Logits and labels have incompatible shapes")
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_log_probs = logits[:, :-1].float().log_softmax(dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    sums = token_log_probs.masked_fill(~valid, 0).sum(dim=-1)
    counts = valid.sum(dim=-1)
    if bool(counts.eq(0).any().item()):
        raise RuntimeError("Candidate answer has no supervised tokens")
    return sums, counts


def likelihood_metrics(
    labels: Sequence[bool],
    scores: Sequence[float],
    relations: Sequence[str],
) -> dict[str, Any]:
    predictions = [score > 0 for score in scores]
    result = binary_classification_metrics(labels, predictions, scores)
    per_relation = {}
    for relation in sorted(set(relations)):
        indices = [index for index, value in enumerate(relations) if value == relation]
        subset_labels = [labels[index] for index in indices]
        subset_scores = [scores[index] for index in indices]
        subset_predictions = [predictions[index] for index in indices]
        subset = binary_classification_metrics(
            subset_labels, subset_predictions, subset_scores
        )
        per_relation[relation] = {
            "sample_count": len(indices),
            "accuracy": subset["accuracy"],
            "auroc": subset["auroc"],
        }
    result["per_relation"] = per_relation
    return result
