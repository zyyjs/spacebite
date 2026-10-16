#!/usr/bin/env python3
"""Apply canonical RBCS corruption to one JPEG and record decode diagnostics."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from spacebyte.corruptions.rbcs import RBCSCorruptor, SCENARIOS


DEFAULT_OUTPUT_ROOT = Path("/data/students/Zhang_jinsong/spacebyte/cache/rbcs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _default_paths(input_path: Path, scenario: str) -> tuple[Path, Path]:
    suffix = input_path.suffix or ".jpg"
    stem = f"{input_path.stem}_{scenario.lower().replace('-', '_')}"
    return DEFAULT_OUTPUT_ROOT / f"{stem}{suffix}", DEFAULT_OUTPUT_ROOT / f"{stem}.json"


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _decode_check(payload: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        result["pil_verify"] = {"success": True, "exception_type": None, "message": None}
    except Exception as error:  # PIL uses several format-specific exception types.
        result["pil_verify"] = {
            "success": False,
            "exception_type": type(error).__name__,
            "message": str(error),
        }

    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            details = {
                "format": image.format,
                "mode": image.mode,
                "size": list(image.size),
            }
        result["pil_full_load"] = {
            "success": True,
            "exception_type": None,
            "message": None,
            **details,
        }
    except Exception as error:
        result["pil_full_load"] = {
            "success": False,
            "exception_type": type(error).__name__,
            "message": str(error),
        }
    return result


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input JPEG does not exist: {args.input}")
    default_output, default_report = _default_paths(args.input, args.scenario)
    output_path = (args.output or default_output).resolve()
    report_path = (args.report or default_report).resolve()
    original = args.input.read_bytes()
    corrupted, statistics = RBCSCorruptor.corrupt_with_stats(
        original,
        SCENARIOS[args.scenario],
        seed=args.seed,
        scenario_name=args.scenario,
    )
    _atomic_bytes(output_path, corrupted)
    report = {
        **statistics,
        "input_path": str(args.input.resolve()),
        "output_path": str(output_path),
        "report_path": str(report_path),
        "jpeg_decode": _decode_check(corrupted),
        "decode_failure_substitution_used": False,
    }
    _atomic_bytes(
        report_path,
        (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
