"""Pixel inputs derived from the exact JPEG stream used by the Byte branch."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .spatialsense import render_record_to_jpeg_bytes


BLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
BLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class MarkedPixelInput:
    """One normalized Pixel input and its shared Byte-branch provenance."""

    pixel_values: Tensor
    jpeg_bytes: Tensor
    boxes: dict[str, list[int]]
    sample_id: str


def decode_marked_jpeg_to_pixel_values(jpeg_bytes: Tensor) -> Tensor:
    """Decode a Q100 stream and apply the pinned BLIP image normalization.

    Geometry is intentionally untouched here: ``render_record_to_jpeg_bytes``
    has already performed the shared 224x224 letterbox and box drawing.
    """

    if jpeg_bytes.ndim != 1 or jpeg_bytes.dtype not in (torch.int32, torch.int64):
        raise TypeError("jpeg_bytes must be a one-dimensional integer tensor")
    if bool(((jpeg_bytes < 0) | (jpeg_bytes > 255)).any().item()):
        raise ValueError("jpeg_bytes must contain only values in [0, 255]")
    payload = jpeg_bytes.to(dtype=torch.uint8, device="cpu").numpy().tobytes()
    with Image.open(io.BytesIO(payload)) as image:
        rgb = image.convert("RGB")
        if rgb.size != (224, 224):
            raise RuntimeError(f"Expected the shared 224x224 JPEG, got {rgb.size}")
        array = np.asarray(rgb, dtype=np.float32).copy()
    pixels = torch.from_numpy(array).permute(2, 0, 1).div_(255.0)
    mean = torch.tensor(BLIP_IMAGE_MEAN, dtype=pixels.dtype)[:, None, None]
    std = torch.tensor(BLIP_IMAGE_STD, dtype=pixels.dtype)[:, None, None]
    return pixels.sub_(mean).div_(std)


def render_record_to_pixel_values(
    dataset_root: Path, record: Mapping[str, Any]
) -> MarkedPixelInput:
    """Render once as Byte JPEG, then decode that same stream for Pixel input."""

    jpeg_bytes, _rendered, boxes = render_record_to_jpeg_bytes(dataset_root, record)
    return MarkedPixelInput(
        pixel_values=decode_marked_jpeg_to_pixel_values(jpeg_bytes),
        jpeg_bytes=jpeg_bytes,
        boxes=boxes,
        sample_id=str(record["sample_id"]),
    )


def fixed_derangement(size: int, seed: int) -> Tensor:
    """Return a deterministic global permutation with no sample left in place."""

    if size < 2:
        raise ValueError("Shuffled-Pixel requires at least two globally mapped samples")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    identity = torch.arange(size)
    for _ in range(10_000):
        permutation = torch.randperm(size, generator=generator)
        if bool(permutation.ne(identity).all().item()):
            return permutation
    raise RuntimeError("Could not construct a fixed-point-free permutation")
