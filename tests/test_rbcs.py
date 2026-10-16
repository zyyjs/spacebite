"""Contracts for the canonical full-stream RBCS corruption module."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from spacebyte.corruptions.rbcs import (
    ALGORITHM_VERSION,
    RBCSCorruptor,
    RBCSParams,
    SCENARIOS,
)


def reference_corrupt(data: bytes, params: RBCSParams, seed: int) -> bytes:
    rng = random.Random(seed)
    mutable = bytearray(data)
    deleted = set()
    for segment in range(len(mutable) // params.S):
        if rng.random() >= params.P:
            continue
        start = segment * params.S
        use_bit_flip = rng.random() < params.p_f
        for index in range(start, start + params.S):
            if rng.random() >= params.q:
                continue
            if use_bit_flip:
                mutable[index] ^= 1 << rng.randint(0, 7)
            else:
                deleted.add(index)
    return bytes(value for index, value in enumerate(mutable) if index not in deleted)


def test_same_seed_is_byte_exact() -> None:
    data = bytes(range(256)) * 8
    first = RBCSCorruptor.corrupt(data, SCENARIOS["Heavy-Mixed"], seed=11)
    second = RBCSCorruptor.corrupt(data, SCENARIOS["Heavy-Mixed"], seed=11)
    assert first == second


@pytest.mark.parametrize("scenario", tuple(SCENARIOS))
def test_local_rng_implementation_matches_reference_semantics(scenario: str) -> None:
    data = bytes(range(256)) * 4 + b"incomplete-tail"
    params = SCENARIOS[scenario]
    assert RBCSCorruptor.corrupt(data, params, seed=20261016) == reference_corrupt(
        data, params, 20261016
    )


def test_different_controlled_seeds_produce_different_results() -> None:
    data = bytes(range(256)) * 16
    assert RBCSCorruptor.corrupt(data, SCENARIOS["Heavy-Mixed"], seed=1) != RBCSCorruptor.corrupt(
        data, SCENARIOS["Heavy-Mixed"], seed=2
    )


def test_global_random_state_is_untouched() -> None:
    random.seed(991)
    before = random.getstate()
    RBCSCorruptor.corrupt(bytes(range(128)), SCENARIOS["Extreme-Mixed"], seed=4)
    assert random.getstate() == before


@pytest.mark.parametrize(
    "params", [RBCSParams(4, 0.0, 0.5, 1.0), RBCSParams(4, 1.0, 0.5, 0.0)]
)
def test_zero_selection_or_modification_probability_is_identity(params) -> None:
    data = bytes(range(18))
    assert RBCSCorruptor.corrupt(data, params, seed=3) == data


def test_full_flip_changes_one_bit_per_complete_segment_byte() -> None:
    data = bytes(range(10))
    output, stats = RBCSCorruptor.corrupt_with_stats(
        data, RBCSParams(4, 1.0, 1.0, 1.0), seed=7
    )
    assert len(output) == len(data)
    assert output[-2:] == data[-2:]
    for original, changed in zip(data[:8], output[:8]):
        difference = original ^ changed
        assert difference > 0 and difference & (difference - 1) == 0
    assert stats["flipped_byte_count"] == 8
    assert stats["deleted_byte_count"] == 0


def test_full_loss_deletes_complete_segments_and_preserves_tail() -> None:
    data = bytes(range(10))
    output, stats = RBCSCorruptor.corrupt_with_stats(
        data, RBCSParams(4, 1.0, 0.0, 1.0), seed=7
    )
    assert output == data[-2:]
    assert stats["deleted_byte_count"] == 8
    assert stats["corrupted_length"] == 2


def test_flip_keeps_length_and_loss_never_increases_it() -> None:
    data = bytes(range(200))
    assert len(RBCSCorruptor.corrupt(data, SCENARIOS["Light-Flip"], seed=9)) == len(data)
    assert len(RBCSCorruptor.corrupt(data, SCENARIOS["Light-Loss"], seed=9)) <= len(data)


def test_mixed_uses_segment_level_flip_and_loss_with_controlled_seed() -> None:
    output, stats = RBCSCorruptor.corrupt_with_stats(
        bytes(range(32)), RBCSParams(4, 1.0, 0.5, 1.0), seed=0
    )
    assert stats["flip_segment_count"] > 0
    assert stats["loss_segment_count"] > 0
    assert stats["flip_segment_count"] + stats["loss_segment_count"] == 8
    assert len(output) == 32 - stats["deleted_byte_count"]


def test_input_bytes_like_object_is_not_modified() -> None:
    source = bytearray(range(64))
    original = source[:]
    result = RBCSCorruptor.corrupt(source, SCENARIOS["Extreme-Flip"], seed=5)
    assert source == original
    assert isinstance(result, bytes)
    assert RBCSCorruptor.corrupt(memoryview(source), RBCSParams(8, 0, 0, 0), seed=1) == bytes(source)


def test_statistics_counts_rates_and_hashes_match_outputs() -> None:
    data = bytes(range(40))
    output, stats = RBCSCorruptor.corrupt_with_stats(
        data,
        RBCSParams(4, 1.0, 0.5, 1.0),
        seed=0,
        scenario_name="controlled",
    )
    assert stats["algorithm_version"] == ALGORITHM_VERSION
    assert stats["scenario_name"] == "controlled"
    assert stats["modified_byte_count"] == stats["flipped_byte_count"] + stats["deleted_byte_count"]
    assert stats["corrupted_length"] == stats["original_length"] - stats["deleted_byte_count"]
    assert stats["total_modified_rate"] == stats["modified_byte_count"] / len(data)
    assert stats["original_sha256"] == hashlib.sha256(data).hexdigest()
    assert stats["corrupted_sha256"] == hashlib.sha256(output).hexdigest()


def test_all_twelve_scenarios_are_exact() -> None:
    expected = {
        "Light-Flip": (128, 0.35, 1.0, 0.35),
        "Light-Loss": (128, 0.35, 0.0, 0.35),
        "Light-Mixed": (128, 0.35, 0.5, 0.35),
        "Medium-Flip": (64, 0.55, 1.0, 0.55),
        "Medium-Loss": (64, 0.55, 0.0, 0.55),
        "Medium-Mixed": (64, 0.55, 0.5, 0.55),
        "Heavy-Flip": (32, 0.65, 1.0, 0.65),
        "Heavy-Loss": (32, 0.65, 0.0, 0.65),
        "Heavy-Mixed": (32, 0.65, 0.5, 0.65),
        "Extreme-Flip": (16, 0.75, 1.0, 0.75),
        "Extreme-Loss": (16, 0.75, 0.0, 0.75),
        "Extreme-Mixed": (16, 0.75, 0.5, 0.75),
    }
    assert list(SCENARIOS) == list(expected)
    assert {
        name: (params.S, params.P, params.p_f, params.q)
        for name, params in SCENARIOS.items()
    } == expected


@pytest.mark.parametrize(
    ("arguments", "exception"),
    [
        ((0, 0.5, 0.5, 0.5), ValueError),
        ((-1, 0.5, 0.5, 0.5), ValueError),
        ((1.5, 0.5, 0.5, 0.5), TypeError),
        ((4, -0.1, 0.5, 0.5), ValueError),
        ((4, 1.1, 0.5, 0.5), ValueError),
        ((4, 0.5, -0.1, 0.5), ValueError),
        ((4, 0.5, 1.1, 0.5), ValueError),
        ((4, 0.5, 0.5, float("nan")), ValueError),
        ((4, 0.5, 0.5, float("inf")), ValueError),
    ],
)
def test_invalid_parameters_raise(arguments, exception) -> None:
    with pytest.raises(exception):
        RBCSParams(*arguments)


def test_invalid_data_params_seed_and_scenario_name_raise() -> None:
    params = RBCSParams(4, 1, 1, 1)
    with pytest.raises(TypeError, match="data"):
        RBCSCorruptor.corrupt([1, 2, 3], params, seed=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="params"):
        RBCSCorruptor.corrupt(b"abc", object(), seed=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seed"):
        RBCSCorruptor.corrupt(b"abc", params, seed="1")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="scenario_name"):
        RBCSCorruptor.corrupt_with_stats(b"abc", params, scenario_name=1)  # type: ignore[arg-type]


def test_empty_input_has_zero_rates_and_valid_hashes() -> None:
    output, stats = RBCSCorruptor.corrupt_with_stats(b"", SCENARIOS["Heavy-Mixed"], seed=3)
    assert output == b""
    assert stats["complete_segment_count"] == 0
    assert stats["flip_rate"] == stats["deletion_rate"] == stats["total_modified_rate"] == 0


def test_verification_cli_writes_bytes_and_decode_report(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jpg"
    output_path = tmp_path / "corrupted.jpg"
    report_path = tmp_path / "report.json"
    Image.new("RGB", (32, 24), (30, 100, 220)).save(input_path, quality=100)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_rbcs.py",
            "--input",
            str(input_path),
            "--scenario",
            "Heavy-Mixed",
            "--seed",
            "20261016",
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text())
    assert output_path.read_bytes() != input_path.read_bytes()
    assert report["scenario_name"] == "Heavy-Mixed"
    assert report["seed"] == 20261016
    assert set(report["jpeg_decode"]) == {"pil_verify", "pil_full_load"}
    assert report["decode_failure_substitution_used"] is False
