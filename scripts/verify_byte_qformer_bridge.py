#!/usr/bin/env python3
"""Verify the frozen ByteFormer -> random instruction-aware Q-Former bridge."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import torch

from spacebyte.jpeg_bytes import pad_byte_sequences, preprocess_image_to_jpeg_bytes
from spacebyte.models import (
    ByteFormerTokenAdapter,
    ByteQFormerBridge,
    build_official_byteformer,
    freeze_byteformer,
)


DEFAULT_CONFIG = Path(
    "/home/Zhang_jinsong/spacebyte/corenet/projects/byteformer/"
    "imagenet_jpeg_q100/conv_kernel_size=8.yaml"
)
DEFAULT_CHECKPOINT = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/"
    "byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--byteformer-config",
        "--config",
        dest="config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--byteformer-checkpoint",
        "--checkpoint",
        dest="checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
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


def parameter_counts(module: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in module.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        ),
    }


def nonzero_gradient_components(
    bridge: ByteQFormerBridge,
) -> tuple[dict[str, bool], list[str]]:
    named = dict(bridge.named_parameters())
    groups = {
        "query_tokens": ["query_tokens"],
        "byte_layernorm": [
            name for name in named if name.startswith("byte_layernorm.")
        ],
        "self_attention": [name for name in named if ".attention.self.query." in name],
        "cross_attention": [
            name for name in named if ".crossattention.self.query." in name
        ],
        "query_ffn": [name for name in named if ".intermediate_query." in name],
        "llm_projector": [name for name in named if name.startswith("llm_projector.")],
    }
    found = {
        group: any(
            named[name].grad is not None and bool((named[name].grad != 0).any().item())
            for name in names
        )
        for group, names in groups.items()
    }
    without_gradient = [
        name
        for name, parameter in named.items()
        if parameter.requires_grad
        and (parameter.grad is None or not bool((parameter.grad != 0).any().item()))
    ]
    return found, without_gradient


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cuda_index = device.index if device.index is not None else 0

    byteformer, opts, checkpoint_report = build_official_byteformer(
        args.config, args.checkpoint
    )
    adapter = freeze_byteformer(ByteFormerTokenAdapter(byteformer).to(device))
    torch.manual_seed(20261012)
    bridge = ByteQFormerBridge().to(device)
    bridge.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(cuda_index)

    paths, streams = select_jpegs(args.image_dir, opts)
    byte_batch = pad_byte_sequences(streams).to(device)
    # Fixed in-vocabulary BERT IDs avoid downloading a tokenizer or any model
    # weights. They exercise the real instruction embedding and masking path.
    instruction_ids = torch.tensor(
        [[101, 2023, 2003, 1037, 3231, 102], [101, 2156, 1996, 3861, 102, 0]],
        dtype=torch.long,
        device=device,
    )
    instruction_mask = instruction_ids.ne(0).long()

    with torch.no_grad():
        byte_features, byte_mask = adapter(byte_batch)
        baseline = bridge(
            encoder_hidden_states=byte_features,
            encoder_attention_mask=byte_mask,
            input_ids=instruction_ids,
            text_attention_mask=instruction_mask,
        )

        changed_byte_features = byte_features.clone()
        changed_byte_features[byte_mask == 0] = 1000 * torch.randn_like(
            changed_byte_features[byte_mask == 0]
        )
        masked_byte_output = bridge(
            encoder_hidden_states=changed_byte_features,
            encoder_attention_mask=byte_mask,
            input_ids=instruction_ids,
            text_attention_mask=instruction_mask,
        )

        extra_byte_features = torch.randn(
            byte_features.shape[0],
            31,
            byte_features.shape[-1],
            device=device,
        )
        extra_byte_mask = torch.zeros(
            byte_features.shape[0], 31, dtype=torch.long, device=device
        )
        added_byte_padding = bridge(
            torch.cat([byte_features, extra_byte_features], dim=1),
            torch.cat([byte_mask, extra_byte_mask], dim=1),
            instruction_ids,
            instruction_mask,
        )

        extra_ids = torch.tensor(
            [[300, 301, 302], [400, 401, 402]], dtype=torch.long, device=device
        )
        extra_text_mask = torch.zeros(2, 3, dtype=torch.long, device=device)
        added_text_padding = bridge(
            byte_features,
            byte_mask,
            torch.cat([instruction_ids, extra_ids], dim=1),
            torch.cat([instruction_mask, extra_text_mask], dim=1),
        )

    bridge.train()
    bridge.zero_grad(set_to_none=True)
    gradient_output = bridge(
        byte_features.detach(), byte_mask, instruction_ids, instruction_mask
    )
    gradient_output.projected_features.square().mean().backward()
    gradient_components, no_gradient_parameters = nonzero_gradient_components(bridge)
    bridge.eval()
    byteformer_parameters_with_gradient = [
        name
        for name, parameter in adapter.named_parameters()
        if parameter.grad is not None
    ]
    if byteformer_parameters_with_gradient:
        raise RuntimeError("Frozen ByteFormer unexpectedly received gradients")
    if not all(gradient_components.values()):
        raise RuntimeError(f"Missing bridge gradients: {gradient_components}")

    qformer_counts = parameter_counts(bridge.qformer)
    counts = {
        "byteformer": parameter_counts(adapter),
        "qformer_without_mlm_head": qformer_counts,
        "query_tokens": bridge.query_tokens.numel(),
        "byte_layernorm": sum(
            parameter.numel() for parameter in bridge.byte_layernorm.parameters()
        ),
        "llm_projector": sum(
            parameter.numel() for parameter in bridge.llm_projector.parameters()
        ),
        "bridge_total": parameter_counts(bridge),
    }
    report: dict[str, Any] = {
        "initialization_mode": bridge.initialization_mode,
        "initialization": {
            "mode": bridge.initialization_mode,
            "external_model_weights_loaded": False,
            "qformer_source": bridge.qformer_source,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "precision": str(next(bridge.parameters()).dtype),
            "device": str(device),
        },
        "qformer_config": {
            "hidden_size": bridge.config.hidden_size,
            "num_hidden_layers": bridge.config.num_hidden_layers,
            "num_attention_heads": bridge.config.num_attention_heads,
            "intermediate_size": bridge.config.intermediate_size,
            "encoder_width": bridge.config.encoder_width,
            "cross_attention_freq": bridge.config.cross_attention_freq,
            "query_length": bridge.config.query_length,
            "vocab_size": bridge.config.vocab_size,
        },
        "checkpoint": checkpoint_report,
        "byteformer_frozen": not any(
            parameter.requires_grad for parameter in adapter.parameters()
        ),
        "byteformer_training_mode": adapter.training,
        "byteformer_parameters_with_gradient": byteformer_parameters_with_gradient,
        "samples": [
            {"path": str(path.resolve()), "jpeg_byte_length": stream.numel()}
            for path, stream in zip(paths, streams)
        ],
        "byte_features_shape": list(byte_features.shape),
        "byte_attention_mask_shape": list(byte_mask.shape),
        "byte_valid_tokens": byte_mask.sum(dim=1).tolist(),
        "query_features_shape": list(baseline.query_features.shape),
        "projected_features_shape": list(baseline.projected_features.shape),
        "query_attention_mask_shape": list(baseline.query_attention_mask.shape),
        "query_token_count": baseline.query_features.shape[1],
        "combined_attention_mask_shape": list(baseline.combined_attention_mask.shape),
        "all_outputs_finite": bool(
            torch.isfinite(baseline.query_features).all().item()
            and torch.isfinite(baseline.projected_features).all().item()
        ),
        "masked_byte_value_error": tensor_error(
            baseline.projected_features, masked_byte_output.projected_features
        ),
        "added_byte_padding_error": tensor_error(
            baseline.projected_features, added_byte_padding.projected_features
        ),
        "added_text_padding_error": tensor_error(
            baseline.projected_features, added_text_padding.projected_features
        ),
        "gradient_components_nonzero": gradient_components,
        "trainable_parameters_without_gradient": no_gradient_parameters,
        "parameter_counts": counts,
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
