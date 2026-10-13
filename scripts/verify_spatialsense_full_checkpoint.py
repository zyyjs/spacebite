#!/usr/bin/env python3
"""Strictly reload a saved full-pilot bridge in a fresh Python process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spacebyte.models import ByteQFormerBridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "bridge_state_dict" not in artifact:
        raise RuntimeError("Checkpoint does not contain bridge_state_dict")
    bridge = ByteQFormerBridge()
    incompatible = bridge.load_state_dict(
        artifact["bridge_state_dict"], strict=True
    )
    state = bridge.state_dict()
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "strict": True,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "tensor_count": len(state),
        "parameter_count": sum(value.numel() for value in state.values()),
        "all_finite": all(
            bool(torch.isfinite(value).all().item())
            for value in state.values()
            if value.is_floating_point()
        ),
        "optimizer_step": artifact.get("optimizer_step"),
        "metadata_keys": sorted(artifact.get("metadata", {})),
    }
    if report["missing_keys"] or report["unexpected_keys"] or not report["all_finite"]:
        raise RuntimeError(f"Strict bridge reload failed: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
