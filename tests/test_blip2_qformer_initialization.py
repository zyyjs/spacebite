"""Regression tests for audited BLIP-2 Stage-1 Q-Former initialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from spacebyte.jpeg_bytes import pad_byte_sequences, preprocess_image_to_jpeg_bytes
from spacebyte.models import (
    BLIP2_STAGE1_SHA256,
    BLIP2_STAGE1_SIZE_BYTES,
    ByteFormerTokenAdapter,
    ByteQFormerBridge,
    build_official_byteformer,
    freeze_byteformer,
    initialize_byte_qformer_from_blip2_stage1,
)


BLIP2_CHECKPOINT = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/blip2_stage1/" "blip2_pretrained.pth"
)
BYTEFORMER_CONFIG = Path(
    "/home/Zhang_jinsong/spacebyte/corenet/projects/byteformer/"
    "imagenet_jpeg_q100/conv_kernel_size=8.yaml"
)
BYTEFORMER_CHECKPOINT = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/"
    "byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt"
)
IMAGE_DIR = Path(
    "/data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense/images/flickr"
)


def _error(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    difference = (left.float() - right.float()).abs()
    return float(difference.max().item()), float(difference.mean().item())


def _forward(
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


def _two_streams(opts: Any) -> list[torch.Tensor]:
    streams: list[torch.Tensor] = []
    lengths: set[int] = set()
    for path in sorted(IMAGE_DIR.glob("*.jpg")):
        if path.name.startswith("._"):
            continue
        stream = preprocess_image_to_jpeg_bytes(path, opts)
        if stream.numel() in lengths:
            continue
        streams.append(stream)
        lengths.add(stream.numel())
        if len(streams) == 2:
            return streams
    pytest.skip("Two distinct real JPEG streams are unavailable")


@pytest.fixture(scope="module")
def verification() -> dict[str, Any]:
    required = (
        BLIP2_CHECKPOINT,
        BYTEFORMER_CONFIG,
        BYTEFORMER_CHECKPOINT,
        IMAGE_DIR,
    )
    if not all(path.exists() for path in required):
        pytest.skip("BLIP-2/ByteFormer integration assets are unavailable")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20261012)
    bridge = ByteQFormerBridge().to(device).eval()
    synthetic_features = torch.randn(2, 231, 192, device=device)
    synthetic_mask = torch.ones(2, 231, dtype=torch.long, device=device)
    synthetic_ids = torch.randint(1, 30523, (2, 16), device=device)
    synthetic_text_mask = torch.ones(2, 16, dtype=torch.long, device=device)
    with torch.no_grad():
        synthetic_before = _forward(
            bridge,
            synthetic_features,
            synthetic_mask,
            synthetic_ids,
            synthetic_text_mask,
        ).projected_features.clone()

    migration = initialize_byte_qformer_from_blip2_stage1(bridge, BLIP2_CHECKPOINT)
    with torch.no_grad():
        synthetic_after = _forward(
            bridge,
            synthetic_features,
            synthetic_mask,
            synthetic_ids,
            synthetic_text_mask,
        ).projected_features.clone()

    byteformer, opts, byteformer_report = build_official_byteformer(
        BYTEFORMER_CONFIG, BYTEFORMER_CHECKPOINT
    )
    adapter = freeze_byteformer(ByteFormerTokenAdapter(byteformer).to(device))
    byte_batch = pad_byte_sequences(_two_streams(opts)).to(device)
    input_ids = torch.tensor(
        [[101, 2023, 2003, 1037, 3231, 102], [101, 2156, 1996, 3861, 102, 0]],
        dtype=torch.long,
        device=device,
    )
    text_mask = input_ids.ne(0).long()
    with torch.no_grad():
        byte_features, byte_mask = adapter(byte_batch)
        baseline = _forward(bridge, byte_features, byte_mask, input_ids, text_mask)

        changed = byte_features.clone()
        changed[byte_mask == 0] = 1000 * torch.randn_like(changed[byte_mask == 0])
        changed_output = _forward(bridge, changed, byte_mask, input_ids, text_mask)

        extra_features = torch.randn(2, 47, 192, device=device)
        extra_mask = torch.zeros(2, 47, dtype=torch.long, device=device)
        extra_byte_output = _forward(
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
        extra_text_output = _forward(
            bridge,
            byte_features,
            byte_mask,
            torch.cat([input_ids, extra_ids], dim=1),
            torch.cat([text_mask, extra_text_mask], dim=1),
        )
        deterministic_before = baseline.projected_features.clone()

    repeated = initialize_byte_qformer_from_blip2_stage1(bridge, BLIP2_CHECKPOINT)
    with torch.no_grad():
        deterministic_after = _forward(
            bridge, byte_features, byte_mask, input_ids, text_mask
        ).projected_features

    bridge.train()
    bridge.zero_grad(set_to_none=True)
    gradient_output = _forward(
        bridge, byte_features.detach(), byte_mask, input_ids, text_mask
    )
    gradient_output.projected_features.float().square().mean().backward()
    named = dict(bridge.named_parameters())
    gradient_groups = {
        "query_tokens": ["query_tokens"],
        "qformer": [key for key in named if key.startswith("qformer.")],
        "cross_attention": [
            key for key in named if ".crossattention.self.query." in key
        ],
        "projector": [key for key in named if key.startswith("llm_projector.")],
    }
    nonzero_gradients = {
        group: any(
            named[key].grad is not None and bool((named[key].grad != 0).any().item())
            for key in keys
        )
        for group, keys in gradient_groups.items()
    }
    bridge.eval()

    return {
        "migration": migration,
        "repeated": repeated,
        "synthetic_before": synthetic_before,
        "synthetic_after": synthetic_after,
        "byteformer_report": byteformer_report,
        "adapter": adapter,
        "byte_features": byte_features,
        "byte_mask": byte_mask,
        "baseline": baseline,
        "changed_output": changed_output,
        "extra_byte_output": extra_byte_output,
        "extra_text_output": extra_text_output,
        "deterministic_before": deterministic_before,
        "deterministic_after": deterministic_after,
        "nonzero_gradients": nonzero_gradients,
    }


def test_checkpoint_hash_and_structure(verification: dict[str, Any]) -> None:
    report = verification["migration"]
    checkpoint = report["checkpoint"]
    contents = report["contents"]
    assert checkpoint["sha256"] == BLIP2_STAGE1_SHA256
    assert checkpoint["size_bytes"] == BLIP2_STAGE1_SIZE_BYTES
    assert checkpoint["top_level_keys"] == ["model"]
    assert checkpoint["tensor_count"] == 346
    assert contents["qformer_prefix"] == "Qformer."
    assert contents["query_tokens"]["shape"] == [1, 32, 768]
    assert contents["word_embeddings"]["vocab_size"] == 30523
    assert contents["contains_visual_encoder"] is False
    assert contents["contains_mlm_head"] is True


def test_loaded_and_retained_tensors_are_verified(verification: dict[str, Any]) -> None:
    report = verification["migration"]
    assert report["verification"]["all_loaded_tensors_exact"] is True
    assert report["verification"]["loaded_verification_failures"] == []
    assert report["verification"]["all_retained_tensors_unchanged"] is True
    assert report["verification"]["retained_verification_failures"] == []
    assert report["verification"]["strict_false_used"] is False
    assert report["initialization_mode"] == "blip2_stage1_partial"
    assert report["summary"]["loaded_tensor_count"] == 318
    assert report["summary"]["shape_mismatch_tensor_count"] == 12
    mismatch = [
        item for item in report["tensor_records"] if item["status"] == "shape_mismatch"
    ]
    assert all(
        ".crossattention.self." in item["target_key"]
        and item["target_key"].endswith(("key.weight", "value.weight"))
        for item in mismatch
    )


def test_query_tokens_and_embeddings_are_loaded(verification: dict[str, Any]) -> None:
    records = verification["migration"]["tensor_records"]
    by_target = {
        item["target_key"]: item for item in records if item["target_key"] is not None
    }
    assert by_target["query_tokens"]["status"] == "loadable"
    assert (
        by_target["qformer.bert.embeddings.word_embeddings.weight"]["status"]
        == "loadable"
    )
    assert by_target["byte_layernorm.weight"]["status"] == "project_added"
    assert by_target["llm_projector.weight"]["status"] == "project_added"


def test_synthetic_forward_and_initialization_effect(
    verification: dict[str, Any]
) -> None:
    before = verification["synthetic_before"]
    after = verification["synthetic_after"]
    assert after.shape == (2, 32, 4096)
    assert torch.isfinite(after).all()
    maximum, _ = _error(before, after)
    assert maximum > 1e-4


def test_real_jpeg_end_to_end_and_frozen_byteformer(
    verification: dict[str, Any]
) -> None:
    assert verification["byte_features"].shape == (2, 177, 192)
    assert verification["byte_mask"].shape == (2, 177)
    assert verification["baseline"].query_features.shape == (2, 32, 768)
    assert verification["baseline"].projected_features.shape == (2, 32, 4096)
    assert torch.isfinite(verification["baseline"].projected_features).all()
    assert not any(
        parameter.requires_grad for parameter in verification["adapter"].parameters()
    )
    assert all(
        parameter.grad is None for parameter in verification["adapter"].parameters()
    )
    assert verification["byteformer_report"]["missing_keys"] == []
    assert verification["byteformer_report"]["unexpected_keys"] == []


def test_padding_masks(verification: dict[str, Any]) -> None:
    baseline = verification["baseline"].projected_features
    maximum, _ = _error(baseline, verification["changed_output"].projected_features)
    assert maximum <= 1e-6
    maximum, _ = _error(baseline, verification["extra_byte_output"].projected_features)
    assert maximum <= 1e-6
    maximum, _ = _error(baseline, verification["extra_text_output"].projected_features)
    assert maximum <= 5e-6


def test_gradients_and_deterministic_reinitialization(
    verification: dict[str, Any]
) -> None:
    assert all(verification["nonzero_gradients"].values())
    maximum, mean = _error(
        verification["deterministic_before"],
        verification["deterministic_after"],
    )
    assert maximum == 0 and mean == 0
    assert verification["repeated"]["verification"]["all_loaded_tensors_exact"] is True
    assert (
        verification["repeated"]["verification"]["all_retained_tensors_unchanged"]
        is True
    )
