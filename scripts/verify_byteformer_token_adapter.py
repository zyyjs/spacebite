#!/usr/bin/env python3
"""Verify ByteFormer token extraction and variable-length padding behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from spacebyte.jpeg_bytes import pad_byte_sequences, preprocess_image_to_jpeg_bytes
from spacebyte.models import ByteFormerTokenAdapter, build_official_byteformer


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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
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


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def select_distinct_jpegs(
    image_dir: Path, opts: argparse.Namespace, count: int = 4
) -> tuple[list[Path], list[torch.Tensor]]:
    paths: list[Path] = []
    byte_streams: list[torch.Tensor] = []
    lengths: set[int] = set()
    candidates = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg"}
        and not path.name.startswith("._")
    )
    for path in candidates:
        stream = preprocess_image_to_jpeg_bytes(path, opts)
        if stream.numel() in lengths:
            continue
        paths.append(path)
        byte_streams.append(stream)
        lengths.add(stream.numel())
        if len(paths) == count:
            break
    if len(paths) < count:
        raise RuntimeError(f"Found only {len(paths)} JPEG streams with distinct lengths")

    ordered = sorted(zip(paths, byte_streams), key=lambda item: item[1].numel())
    return [item[0] for item in ordered], [item[1] for item in ordered]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        cuda_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        torch.cuda.empty_cache()

    model, opts, load_report = build_official_byteformer(
        args.config, args.checkpoint
    )
    parameter_sha256_before = parameter_digest(model)
    model.eval().to(device)
    adapter = ByteFormerTokenAdapter(model).eval()
    if device.type == "cuda":
        # CoreNet imports do not initialize a CUDA context. Reset only after
        # the model transfer has established one for the selected device.
        torch.cuda.reset_peak_memory_stats(cuda_index)

    paths, streams = select_distinct_jpegs(args.image_dir, opts)
    batch = pad_byte_sequences(streams).to(device)

    # Reproduce the upstream advanced-indexing bug before using the adapter.
    original_byte_mask = torch.zeros_like(batch, dtype=torch.float32)
    original_byte_mask[batch == -1].fill_(float("-inf"))
    original_mask_changed = bool(torch.isneginf(original_byte_mask).any().item())
    with torch.inference_mode():
        original_tokens, original_internal_mask = model.get_backbone_inputs(
            batch.clone()
        )
        _, original_final_mask = model.backbone_forward(
            original_tokens, original_internal_mask
        )

        features, attention_mask, trace = adapter.forward_with_trace(batch)

        shortest = streams[0].unsqueeze(0).to(device)
        official_logits = model(shortest.clone())
        single_features, single_mask = adapter(shortest)
        reconstructed_logits = adapter.pool_and_classify(
            single_features, single_mask
        )

        shortest_valid_count = int(attention_mask[0].sum().item())
        batch_consistency = tensor_error(
            single_features[0], features[0, :shortest_valid_count]
        )

        extra_padding = 4096
        padding_only_input = torch.full(
            (1, shortest.shape[1] + extra_padding),
            -1,
            dtype=shortest.dtype,
            device=device,
        )
        padding_only_input[:, : shortest.shape[1]] = shortest
        padded_features, padded_mask = adapter(padding_only_input)
        padded_valid_count = int(padded_mask.sum().item())
        padding_consistency = tensor_error(
            single_features[0], padded_features[0, :padded_valid_count]
        )

    parameter_sha256_after = parameter_digest(model)
    parameter_unchanged = parameter_sha256_before == parameter_sha256_after
    if not parameter_unchanged:
        raise RuntimeError("Model parameters changed during inference")

    final_token_count = features.shape[1]
    valid_counts = attention_mask.sum(dim=1).tolist()
    samples: list[dict[str, Any]] = []
    for path, stream, valid_count in zip(paths, streams, valid_counts):
        samples.append(
            {
                "path": str(path.resolve()),
                "jpeg_byte_length": stream.numel(),
                "batch_padded_length": batch.shape[1],
                "byte_valid_positions": stream.numel(),
                "final_token_count": final_token_count,
                "final_valid_tokens": valid_count,
                "final_padding_tokens": final_token_count - valid_count,
            }
        )

    finite = bool(torch.isfinite(features).all().item()) and bool(
        torch.isfinite(reconstructed_logits).all().item()
    )
    report = {
        "checkpoint": load_report,
        "samples": samples,
        "original_mask_bug": {
            "byte_mask_changed_by_chained_index_fill": original_mask_changed,
            "original_final_masked_counts": torch.isneginf(original_final_mask)
            .sum(dim=1)
            .tolist(),
        },
        "shape_trace": trace,
        "features_shape": list(features.shape),
        "attention_mask_shape": list(attention_mask.shape),
        "classification_reconstruction_error": tensor_error(
            official_logits, reconstructed_logits
        ),
        "single_vs_mixed_batch_error": batch_consistency,
        "same_bytes_extra_padding_error": padding_consistency,
        "same_bytes_extra_padding_valid_tokens": padded_valid_count,
        "output_all_finite": finite,
        "model_parameter_sha256_before": parameter_sha256_before,
        "model_parameter_sha256_after": parameter_sha256_after,
        "model_parameters_unchanged": parameter_unchanged,
        "spatialsense_checkpoint_loaded": False,
        "device": str(device),
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(cuda_index)
            if device.type == "cuda"
            else 0
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
