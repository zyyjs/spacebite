"""Strict loader for Apple's official JPEG Q100 ByteFormer checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from corenet.modeling.models import get_model
from corenet.options.opts import get_training_arguments


EXPECTED_CHECKPOINT_SHA256 = (
    "13646ae929d20f3fb48f0a25e0a945dd59ad070746dbf651c6b2a64856867d82"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_official_byteformer(
    config_path: Path,
    checkpoint_path: Path,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Create the official model and strictly load the unmodified checkpoint."""
    config_path = config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Official checkpoint SHA-256 mismatch: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )

    opts = get_training_arguments(args=["--common.config-file", str(config_path)])
    expected_options = {
        "model.classification.name": "byteformer",
        "model.classification.n_classes": 1000,
        "model.classification.byteformer.mode": "tiny",
        "model.classification.byteformer.conv_kernel_size": 8,
        "model.classification.byteformer.window_sizes": [128],
        "model.classification.byteformer.max_num_tokens": 50000,
        "image_augmentation.resize.size": 256,
        "image_augmentation.center_crop.size": 224,
        "image_augmentation.pil_save.file_encoding": "JPEG",
        "image_augmentation.pil_save.quality": 100,
    }
    actual_options = {key: getattr(opts, key) for key in expected_options}
    mismatches = {
        key: {"expected": expected_options[key], "actual": actual_options[key]}
        for key in expected_options
        if actual_options[key] != expected_options[key]
    }
    if mismatches:
        raise RuntimeError(f"Official config validation failed: {mismatches}")

    model = get_model(opts=opts, category="classification", model_name="byteformer")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected a state dictionary, got {type(state_dict).__name__}"
        )
    incompatible = model.load_state_dict(state_dict, strict=True)
    loaded_state = model.state_dict()
    state_dict_exact_match = all(
        torch.equal(loaded_state[key].cpu(), tensor.cpu())
        for key, tensor in state_dict.items()
    )
    if not state_dict_exact_match:
        raise RuntimeError("Loaded model tensors do not exactly match the checkpoint")

    report = {
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_tensor_count": len(state_dict),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "state_dict_exact_match": state_dict_exact_match,
        "model_options": actual_options,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "classifier_in_features": model.classifier.in_features,
        "classifier_out_features": model.classifier.out_features,
    }
    return model, opts, report
