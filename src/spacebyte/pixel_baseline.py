"""Pure policy helpers for the autonomous SpatialSense Pixel baseline."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def pixel_best_epoch(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select Normal AUROC, then Macro-F1, retaining an earlier exact tie."""

    if not rows:
        raise ValueError("Cannot select from an empty Pixel history")
    selected = rows[0]
    for candidate in rows[1:]:
        current = selected["normal"]
        proposed = candidate["normal"]
        if (proposed["auroc"], proposed["macro_f1"]) > (
            current["auroc"],
            current["macro_f1"],
        ):
            selected = candidate
    return selected


def visual_margin_is_stable(row: Mapping[str, Any], threshold: float = 0.02) -> bool:
    return (
        row["normal"]["auroc"] - row["zero"]["auroc"] >= threshold
        and row["normal"]["auroc"] - row["shuffled_auroc_mean"] >= threshold
    )


def pixel_stopping_reason(
    history: Sequence[dict[str, Any]],
    *,
    max_epochs: int = 5,
    stability_margin: float = 0.02,
    plateau_delta: float = 0.002,
) -> str | None:
    """Apply the predeclared two-epoch stability/plateau/max rules."""

    if len(history) >= 2 and all(
        visual_margin_is_stable(row, stability_margin) for row in history[-2:]
    ):
        return "visual_margin_stable_two_consecutive_epochs"
    if len(history) >= 3:
        no_improvement = []
        for position in range(1, len(history)):
            previous_best = max(row["normal"]["auroc"] for row in history[:position])
            improvement = history[position]["normal"]["auroc"] - previous_best
            no_improvement.append(improvement < plateau_delta)
        if len(no_improvement) >= 2 and all(no_improvement[-2:]):
            return "normal_auroc_plateau_two_consecutive_epochs"
    if history and history[-1]["epoch"] >= max_epochs:
        return "max_epochs"
    return None


def cache_metadata_matches(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    """Require exact values for every declared cache-identity field."""

    mismatches = [
        key for key, value in expected.items() if key not in observed or observed[key] != value
    ]
    unexpected = sorted(
        set(observed)
        - set(expected)
        - {"feature_shape", "created_at", "generation", "validation"}
    )
    mismatches.extend(f"unexpected:{key}" for key in unexpected)
    return not mismatches, mismatches
