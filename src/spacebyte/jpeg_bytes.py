"""Official ByteFormer validation preprocessing for JPEG byte inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch import Tensor

from corenet.data.transforms import image_bytes, image_pil


def preprocess_image_to_jpeg_bytes(image_path: Path, opts: Any) -> Tensor:
    """Resize, center-crop, and encode an image as configured by CoreNet."""
    with Image.open(image_path) as image:
        data = {"image": image.convert("RGB")}
        data = image_pil.Resize(opts=opts)(data)
        data = image_pil.CenterCrop(opts=opts)(data)
        data = image_pil.ToTensor(opts=opts)(data)

    encoded = image_bytes.PILSave(opts=opts)({"samples": data["image"]})
    byte_tensor = encoded["samples"]
    if byte_tensor.ndim != 1 or byte_tensor.dtype != torch.int32:
        raise RuntimeError(
            "CoreNet JPEG preprocessing did not return a one-dimensional int32 tensor"
        )
    return byte_tensor


def pad_byte_sequences(sequences: Sequence[Tensor]) -> Tensor:
    """Right-pad variable-length byte streams with CoreNet's -1 sentinel."""
    if not sequences:
        raise ValueError("At least one byte sequence is required")
    max_length = max(sequence.numel() for sequence in sequences)
    batch = torch.full(
        (len(sequences), max_length),
        -1,
        dtype=sequences[0].dtype,
        device=sequences[0].device,
    )
    for index, sequence in enumerate(sequences):
        if sequence.ndim != 1:
            raise ValueError(f"Expected a 1D byte sequence, got {sequence.shape}")
        if sequence.dtype != sequences[0].dtype:
            raise TypeError("All byte sequences must use the same dtype")
        batch[index, : sequence.numel()] = sequence
    return batch
