"""Deterministic contracts for the autonomous stability run."""

from __future__ import annotations

from spacebyte.stability_longrun import (
    best_epoch,
    conclusion,
    metric_row_is_stable,
    paired_bootstrap,
    stopping_reason,
)


def row(epoch: int, normal: float, zero: float, shuffled: float, *, f1=0.51):
    metric = lambda auc, macro_f1=f1: {
        "logit_yes_minus_no": {
            "auroc": auc,
            "macro_f1": macro_f1,
            "confusion_matrix": {"tn": 4, "fp": 1, "fn": 1, "tp": 4},
        }
    }
    return {
        "epoch": epoch,
        "all_finite": True,
        "evaluations": {
            "normal": metric(normal),
            "zero_byte": metric(zero),
            "shuffled_byte": metric(shuffled),
        },
    }


def test_stability_requires_two_consecutive_qualifying_epochs() -> None:
    first = row(3, 0.53, 0.50, 0.50)
    second = row(4, 0.54, 0.50, 0.51)
    assert metric_row_is_stable(first)
    assert stopping_reason([first]) is None
    assert stopping_reason([first, second]) == "stability_confirmed_two_consecutive_epochs"


def test_plateau_requires_three_new_small_best_improvements() -> None:
    rows = [
        row(3, 0.5300, 0.50, 0.52, f1=0.49),
        row(4, 0.5310, 0.50, 0.52, f1=0.49),
        row(5, 0.5320, 0.50, 0.52, f1=0.49),
        row(6, 0.5330, 0.50, 0.52, f1=0.49),
    ]
    assert stopping_reason(rows) == "plateau"


def test_best_selection_uses_auc_then_f1_and_keeps_earlier_tie() -> None:
    rows = [row(3, 0.53, 0.5, 0.5, f1=0.55), row(4, 0.54, 0.5, 0.5, f1=0.50)]
    rows += [row(5, 0.54, 0.5, 0.5, f1=0.51), row(6, 0.54, 0.5, 0.5, f1=0.51)]
    assert best_epoch(rows)["epoch"] == 5


def test_bootstrap_is_paired_deterministic_and_conclusion_is_confirmed() -> None:
    labels = [False, False, False, True, True, True]
    normal = [-3, -2, -1, 1, 2, 3]
    zero = [0, 0, 0, 0, 0, 0]
    shuffled = [[0, 1, -1, 0, -1, 1], [1, -1, 0, -1, 1, 0]]
    first = paired_bootstrap(labels, normal, zero, shuffled, iterations=200, seed=7)
    second = paired_bootstrap(labels, normal, zero, shuffled, iterations=200, seed=7)
    assert first == second
    assert first["normal_auroc"]["lower_95"] == 1.0
    assert conclusion(1.0, 0.5, 0.5, first) == "confirmed"
