#!/usr/bin/env python3
"""End-to-end verification of BLIP-2 initialization for Byte Q-Former."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from spacebyte.jpeg_bytes import pad_byte_sequences, preprocess_image_to_jpeg_bytes
from spacebyte.models import (
    ByteFormerTokenAdapter,
    ByteQFormerBridge,
    build_official_byteformer,
    freeze_byteformer,
    initialize_byte_qformer_from_blip2_stage1,
)


DEFAULT_BLIP2_CHECKPOINT = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/blip2_stage1/" "blip2_pretrained.pth"
)
DEFAULT_BYTEFORMER_CONFIG = Path(
    "/home/Zhang_jinsong/spacebyte/corenet/projects/byteformer/"
    "imagenet_jpeg_q100/conv_kernel_size=8.yaml"
)
DEFAULT_BYTEFORMER_CHECKPOINT = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/"
    "byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blip2-checkpoint", type=Path, default=DEFAULT_BLIP2_CHECKPOINT
    )
    parser.add_argument(
        "--byteformer-config", type=Path, default=DEFAULT_BYTEFORMER_CONFIG
    )
    parser.add_argument(
        "--byteformer-checkpoint", type=Path, default=DEFAULT_BYTEFORMER_CHECKPOINT
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def tensor_error(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = (left.float() - right.float()).abs()
    return {
        "max_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
    }


def select_jpegs(
    image_dir: Path, opts: argparse.Namespace, count: int = 2
) -> tuple[list[Path], list[torch.Tensor]]:
    selected: list[tuple[Path, torch.Tensor]] = []
    lengths: set[int] = set()
    for path in sorted(image_dir.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in {".jpg", ".jpeg"}
            or path.name.startswith("._")
        ):
            continue
        stream = preprocess_image_to_jpeg_bytes(path, opts)
        if stream.numel() in lengths:
            continue
        selected.append((path, stream))
        lengths.add(stream.numel())
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Found only {len(selected)} distinct JPEG streams")
    selected.sort(key=lambda item: item[1].numel())
    return [item[0] for item in selected], [item[1] for item in selected]


def bridge_forward(
    bridge: ByteQFormerBridge,
    features: torch.Tensor,
    feature_mask: torch.Tensor,
    input_ids: torch.Tensor,
    text_mask: torch.Tensor,
):
    return bridge(
        encoder_hidden_states=features,
        encoder_attention_mask=feature_mask,
        input_ids=input_ids,
        text_attention_mask=text_mask,
    )


def gradient_report(bridge: ByteQFormerBridge) -> dict[str, Any]:
    named = dict(bridge.named_parameters())
    groups = {
        "query_tokens": ["query_tokens"],
        "qformer_self_attention": [
            key for key in named if ".attention.self.query." in key
        ],
        "qformer_cross_attention": [
            key for key in named if ".crossattention.self.query." in key
        ],
        "qformer_ffn": [key for key in named if ".intermediate_query." in key],
        "projector": [key for key in named if key.startswith("llm_projector.")],
    }
    nonzero = {
        group: any(
            named[key].grad is not None and bool((named[key].grad != 0).any().item())
            for key in keys
        )
        for group, keys in groups.items()
    }
    zero_or_missing = [
        key
        for key, parameter in named.items()
        if parameter.requires_grad
        and (parameter.grad is None or not bool((parameter.grad != 0).any().item()))
    ]
    return {"nonzero_by_group": nonzero, "zero_or_missing": zero_or_missing}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cuda_index = device.index if device.index is not None else 0

    torch.manual_seed(20261012)
    bridge = ByteQFormerBridge().to(device).eval()
    synthetic_features = torch.randn(2, 231, 192, device=device)
    synthetic_mask = torch.ones(2, 231, dtype=torch.long, device=device)
    synthetic_ids = torch.randint(1, 30523, (2, 16), device=device)
    synthetic_text_mask = torch.ones(2, 16, dtype=torch.long, device=device)
    with torch.no_grad():
        before_initialization = bridge_forward(
            bridge,
            synthetic_features,
            synthetic_mask,
            synthetic_ids,
            synthetic_text_mask,
        ).projected_features.clone()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_index)
    migration = initialize_byte_qformer_from_blip2_stage1(bridge, args.blip2_checkpoint)
    with torch.no_grad():
        after_initialization = bridge_forward(
            bridge,
            synthetic_features,
            synthetic_mask,
            synthetic_ids,
            synthetic_text_mask,
        ).projected_features
    before_after_error = tensor_error(before_initialization, after_initialization)
    if before_after_error["max_absolute_error"] == 0:
        raise RuntimeError("BLIP-2 initialization did not change the bridge output")

    byteformer, opts, byteformer_load = build_official_byteformer(
        args.byteformer_config, args.byteformer_checkpoint
    )
    adapter = freeze_byteformer(ByteFormerTokenAdapter(byteformer).to(device))
    paths, streams = select_jpegs(args.image_dir, opts)
    byte_batch = pad_byte_sequences(streams).to(device)
    input_ids = torch.tensor(
        [[101, 2023, 2003, 1037, 3231, 102], [101, 2156, 1996, 3861, 102, 0]],
        dtype=torch.long,
        device=device,
    )
    text_mask = input_ids.ne(0).long()
    with torch.no_grad():
        byte_features, byte_mask = adapter(byte_batch)
        baseline = bridge_forward(
            bridge, byte_features, byte_mask, input_ids, text_mask
        )

        changed_padding = byte_features.clone()
        changed_padding[byte_mask == 0] = 1000 * torch.randn_like(
            changed_padding[byte_mask == 0]
        )
        changed_padding_output = bridge_forward(
            bridge, changed_padding, byte_mask, input_ids, text_mask
        )

        extra_features = torch.randn(2, 31, 192, device=device)
        extra_mask = torch.zeros(2, 31, dtype=torch.long, device=device)
        added_byte_padding_output = bridge_forward(
            bridge,
            torch.cat([byte_features, extra_features], dim=1),
            torch.cat([byte_mask, extra_mask], dim=1),
            input_ids,
            text_mask,
        )

        extra_ids = torch.tensor(
            [[300, 301, 302], [400, 401, 402]], dtype=torch.long, device=device
        )
        extra_text_mask = torch.zeros(2, 3, dtype=torch.long, device=device)
        added_text_padding_output = bridge_forward(
            bridge,
            byte_features,
            byte_mask,
            torch.cat([input_ids, extra_ids], dim=1),
            torch.cat([text_mask, extra_text_mask], dim=1),
        )

        deterministic_before = baseline.projected_features.clone()
    repeated_migration = initialize_byte_qformer_from_blip2_stage1(
        bridge, args.blip2_checkpoint
    )
    with torch.no_grad():
        deterministic_after = bridge_forward(
            bridge, byte_features, byte_mask, input_ids, text_mask
        ).projected_features
    deterministic_error = tensor_error(deterministic_before, deterministic_after)

    bridge.train()
    bridge.zero_grad(set_to_none=True)
    train_output = bridge_forward(
        bridge, byte_features.detach(), byte_mask, input_ids, text_mask
    )
    train_output.projected_features.float().square().mean().backward()
    gradients = gradient_report(bridge)
    byteformer_with_grad = [
        name
        for name, parameter in adapter.named_parameters()
        if parameter.grad is not None
    ]
    bridge.eval()

    finite = bool(
        torch.isfinite(baseline.query_features).all().item()
        and torch.isfinite(baseline.projected_features).all().item()
    )
    if not finite:
        raise RuntimeError("Initialized bridge produced NaN or Inf")
    if not all(gradients["nonzero_by_group"].values()):
        raise RuntimeError(f"Missing bridge gradients: {gradients}")
    if byteformer_with_grad:
        raise RuntimeError("Frozen ByteFormer received gradients")
    if deterministic_error["max_absolute_error"] != 0:
        raise RuntimeError("Repeated checkpoint initialization was not deterministic")

    report = {
        "migration": migration,
        "repeated_migration_verification": repeated_migration["verification"],
        "before_vs_after_initialization_error": before_after_error,
        "deterministic_reinitialization_error": deterministic_error,
        "byteformer_checkpoint": byteformer_load,
        "samples": [
            {"path": str(path.resolve()), "jpeg_byte_length": stream.numel()}
            for path, stream in zip(paths, streams)
        ],
        "shapes": {
            "jpeg_byte_batch": list(byte_batch.shape),
            "byte_features": list(byte_features.shape),
            "byte_attention_mask": list(byte_mask.shape),
            "query_features": list(baseline.query_features.shape),
            "projected_features": list(baseline.projected_features.shape),
        },
        "valid_byte_tokens": byte_mask.sum(dim=1).tolist(),
        "all_outputs_finite": finite,
        "changed_byte_padding_value_error": tensor_error(
            baseline.projected_features,
            changed_padding_output.projected_features,
        ),
        "added_byte_padding_error": tensor_error(
            baseline.projected_features,
            added_byte_padding_output.projected_features,
        ),
        "added_text_padding_error": tensor_error(
            baseline.projected_features,
            added_text_padding_output.projected_features,
        ),
        "gradient_check": gradients,
        "byteformer_frozen": not any(
            parameter.requires_grad for parameter in adapter.parameters()
        ),
        "byteformer_parameters_with_gradient": byteformer_with_grad,
        "precision": str(next(bridge.parameters()).dtype),
        "device": str(device),
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(cuda_index) if device.type == "cuda" else 0
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
