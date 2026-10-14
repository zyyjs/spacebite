#!/usr/bin/env python3
"""Strictly verify the bridge and identity-resolved ByteFormer partial state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spacebyte.models import ByteQFormerBridge, build_official_byteformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--byteformer-config", type=Path, required=True)
    parser.add_argument("--byteformer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    bridge = ByteQFormerBridge()
    incompatible = bridge.load_state_dict(artifact["bridge_state_dict"], strict=True)
    byteformer, _, _ = build_official_byteformer(
        args.byteformer_config, args.byteformer_checkpoint
    )
    transformer = byteformer.transformer
    modules = [
        transformer[len(transformer) - 2],
        transformer[len(transformer) - 1],
        byteformer.post_transformer_norm,
    ]
    names_by_id = {id(module): name for name, module in byteformer.named_modules()}
    module_names = [names_by_id[id(module)] for module in modules]
    expected = {
        name
        for name, _ in byteformer.named_parameters()
        if name.startswith(tuple(f"{module_name}." for module_name in module_names))
    }
    source = artifact["byteformer_partial_state_dict"]
    if set(source) != expected:
        raise RuntimeError(
            f"Partial ByteFormer keys differ: missing={sorted(expected - set(source))}, "
            f"unexpected={sorted(set(source) - expected)}"
        )
    target = dict(byteformer.named_parameters())
    with torch.no_grad():
        for name, value in source.items():
            if value.shape != target[name].shape or value.dtype != target[name].dtype:
                raise RuntimeError(f"Shape/dtype mismatch for {name}")
            target[name].copy_(value)
    exact = all(torch.equal(target[name], value) for name, value in source.items())
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "epoch": artifact.get("epoch"),
        "bridge_strict": True,
        "bridge_missing_keys": list(incompatible.missing_keys),
        "bridge_unexpected_keys": list(incompatible.unexpected_keys),
        "byteformer_resolved_modules": module_names,
        "byteformer_partial_tensor_count": len(source),
        "byteformer_partial_parameter_count": sum(value.numel() for value in source.values()),
        "byteformer_exact_after_copy": exact,
        "all_finite": all(
            bool(torch.isfinite(value).all().item())
            for state in (artifact["bridge_state_dict"], source)
            for value in state.values()
            if value.is_floating_point()
        ),
    }
    if (
        report["bridge_missing_keys"]
        or report["bridge_unexpected_keys"]
        or not report["byteformer_exact_after_copy"]
        or not report["all_finite"]
    ):
        raise RuntimeError(f"Checkpoint verification failed: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
