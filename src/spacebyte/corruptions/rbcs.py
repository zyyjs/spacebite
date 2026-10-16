"""Canonical full-stream RBCS corruption supplied for SpaceByte experiments.

The segment and byte decisions intentionally preserve the ordering and
probability semantics of the user-provided reference implementation.  In
particular, the flip/loss decision is made once per selected segment and byte
loss is applied only after every original segment index has been processed.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from numbers import Real
from typing import Any, Final


ALGORITHM_NAME: Final = "RBCS Full-Stream Corruption"
ALGORITHM_VERSION: Final = "rbcs_fullstream_v1"
ALGORITHM_SOURCE: Final = "用户提供的实验代码"


def _validate_probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number in [0, 1]")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


@dataclass(frozen=True)
class RBCSParams:
    """Parameters for one full-stream RBCS corruption scenario."""

    S: int
    P: float
    p_f: float
    q: float

    def __post_init__(self) -> None:
        if isinstance(self.S, bool) or not isinstance(self.S, int):
            raise TypeError("S must be a positive integer")
        if self.S <= 0:
            raise ValueError(f"S must be a positive integer, got {self.S!r}")
        _validate_probability("P", self.P)
        _validate_probability("p_f", self.p_f)
        _validate_probability("q", self.q)


SCENARIOS: Final[dict[str, RBCSParams]] = {
    "Light-Flip": RBCSParams(128, 0.35, 1.0, 0.35),
    "Light-Loss": RBCSParams(128, 0.35, 0.0, 0.35),
    "Light-Mixed": RBCSParams(128, 0.35, 0.5, 0.35),
    "Medium-Flip": RBCSParams(64, 0.55, 1.0, 0.55),
    "Medium-Loss": RBCSParams(64, 0.55, 0.0, 0.55),
    "Medium-Mixed": RBCSParams(64, 0.55, 0.5, 0.55),
    "Heavy-Flip": RBCSParams(32, 0.65, 1.0, 0.65),
    "Heavy-Loss": RBCSParams(32, 0.65, 0.0, 0.65),
    "Heavy-Mixed": RBCSParams(32, 0.65, 0.5, 0.65),
    "Extreme-Flip": RBCSParams(16, 0.75, 1.0, 0.75),
    "Extreme-Loss": RBCSParams(16, 0.75, 0.0, 0.75),
    "Extreme-Mixed": RBCSParams(16, 0.75, 0.5, 0.75),
}


def _coerce_data(data: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes, bytearray, or memoryview")
    return bytes(data)


def _validate_seed(seed: int | None) -> None:
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")


class RBCSCorruptor:
    """Apply canonical RBCS corruption to an entire JPEG file byte stream."""

    @staticmethod
    def corrupt(
        data: bytes | bytearray | memoryview,
        params: RBCSParams,
        seed: int | None = None,
    ) -> bytes:
        """Return corrupted bytes without touching Python's global RNG state."""

        corrupted, _statistics = RBCSCorruptor.corrupt_with_stats(
            data, params, seed=seed
        )
        return corrupted

    @staticmethod
    def corrupt_with_stats(
        data: bytes | bytearray | memoryview,
        params: RBCSParams,
        seed: int | None = None,
        scenario_name: str | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        """Return corrupted bytes and an auditable statistics dictionary."""

        if not isinstance(params, RBCSParams):
            raise TypeError("params must be an RBCSParams instance")
        _validate_seed(seed)
        if scenario_name is not None and not isinstance(scenario_name, str):
            raise TypeError("scenario_name must be a string or None")

        original = _coerce_data(data)
        mutable = bytearray(original)
        deleted: set[int] = set()
        rng = random.Random(seed)
        complete_segment_count = len(mutable) // params.S
        selected_segment_count = 0
        flip_segment_count = 0
        loss_segment_count = 0
        flipped_byte_count = 0

        # Only complete segments are processed. Segment boundaries always use
        # the original byte positions because deletion is deferred to the end.
        for segment in range(complete_segment_count):
            if rng.random() >= params.P:
                continue

            selected_segment_count += 1
            start = segment * params.S
            use_bit_flip = rng.random() < params.p_f
            if use_bit_flip:
                flip_segment_count += 1
            else:
                loss_segment_count += 1

            for index in range(start, start + params.S):
                if rng.random() >= params.q:
                    continue

                if use_bit_flip:
                    bit = rng.randint(0, 7)
                    mutable[index] ^= 1 << bit
                    flipped_byte_count += 1
                else:
                    deleted.add(index)

        corrupted = bytes(
            value for index, value in enumerate(mutable) if index not in deleted
        )
        deleted_byte_count = len(deleted)
        modified_byte_count = flipped_byte_count + deleted_byte_count
        denominator = len(original)

        statistics = {
            "algorithm_name": ALGORITHM_NAME,
            "algorithm_version": ALGORITHM_VERSION,
            "algorithm_source": ALGORITHM_SOURCE,
            "seed": seed,
            "scenario_name": scenario_name,
            "parameters": {
                "S": params.S,
                "P": float(params.P),
                "p_f": float(params.p_f),
                "q": float(params.q),
            },
            "original_length": denominator,
            "corrupted_length": len(corrupted),
            "complete_segment_count": complete_segment_count,
            "selected_segment_count": selected_segment_count,
            "flip_segment_count": flip_segment_count,
            "loss_segment_count": loss_segment_count,
            "flipped_byte_count": flipped_byte_count,
            "deleted_byte_count": deleted_byte_count,
            "modified_byte_count": modified_byte_count,
            "flip_rate": flipped_byte_count / denominator if denominator else 0.0,
            "deletion_rate": deleted_byte_count / denominator if denominator else 0.0,
            "total_modified_rate": modified_byte_count / denominator if denominator else 0.0,
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "corrupted_sha256": hashlib.sha256(corrupted).hexdigest(),
        }
        return corrupted, statistics
