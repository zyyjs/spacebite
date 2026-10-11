#!/usr/bin/env python3
"""Strict-load and smoke-test Apple's official pretrained ByteFormer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from corenet.data.transforms import image_bytes, image_pil
from spacebyte.models import build_official_byteformer


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
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def official_validation_preprocess(
    image_path: Path, opts: argparse.Namespace
) -> tuple[torch.Tensor, list[int]]:
    """Apply the validation transforms and JPEG encoding from the official YAML."""
    with Image.open(image_path) as image:
        data = {"image": image.convert("RGB")}
        data = image_pil.Resize(opts=opts)(data)
        data = image_pil.CenterCrop(opts=opts)(data)
        data = image_pil.ToTensor(opts=opts)(data)

    pixel_shape = list(data["image"].shape)
    encoded = {"samples": data["image"]}
    encoded = image_bytes.PILSave(opts=opts)(encoded)
    return encoded["samples"].unsqueeze(0), pixel_shape


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model, opts, report = build_official_byteformer(args.config, args.checkpoint)
    model.eval().to(device)
    samples, pixel_shape = official_validation_preprocess(args.image, opts)
    samples = samples.to(device)

    with torch.inference_mode():
        logits = model(samples)

    finite = bool(torch.isfinite(logits).all().item())
    if not finite:
        raise RuntimeError("Forward output contains NaN or Inf")
    if list(logits.shape) != [1, 1000]:
        raise RuntimeError(f"Unexpected output shape: {list(logits.shape)}")

    report.update(
        {
            "source": (
                "https://docs-assets.developer.apple.com/ml-research/models/"
                "cvnets-v2/multimodal_classification/"
                "imagenet_jpeg_q100_k8_w128.pt"
            ),
            "input_image": str(args.image.resolve()),
            "validation_pixel_shape_before_encoding": pixel_shape,
            "model_input_shape": list(samples.shape),
            "model_input_dtype": str(samples.dtype),
            "model_output_shape": list(logits.shape),
            "model_output_dtype": str(logits.dtype),
            "output_all_finite": finite,
            "spatialsense_checkpoint_loaded": False,
            "device": str(device),
        }
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
