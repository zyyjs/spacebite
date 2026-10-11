"""Integration tests for extraction of official ByteFormer token features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from spacebyte.jpeg_bytes import pad_byte_sequences, preprocess_image_to_jpeg_bytes
from spacebyte.models import ByteFormerTokenAdapter, build_official_byteformer


CONFIG = Path(
    "/home/Zhang_jinsong/spacebyte/corenet/projects/byteformer/"
    "imagenet_jpeg_q100/conv_kernel_size=8.yaml"
)
CHECKPOINT = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/"
    "byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt"
)
IMAGE_DIR = Path(
    "/data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense/images/flickr"
)


def _error(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    difference = (left.float() - right.float()).abs()
    return float(difference.max().item()), float(difference.mean().item())


def _four_streams(opts: Any) -> list[torch.Tensor]:
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
        if len(streams) == 4:
            break
    if len(streams) != 4:
        pytest.skip("Four distinct real JPEG streams are unavailable")
    return sorted(streams, key=lambda stream: stream.numel())


@pytest.fixture(scope="module")
def verification() -> dict[str, Any]:
    if not CONFIG.is_file() or not CHECKPOINT.is_file() or not IMAGE_DIR.is_dir():
        pytest.skip("ByteFormer integration-test assets are unavailable")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, opts, load_report = build_official_byteformer(CONFIG, CHECKPOINT)
    model.eval().to(device)
    adapter = ByteFormerTokenAdapter(model).eval()
    streams = _four_streams(opts)
    batch = pad_byte_sequences(streams).to(device)
    shortest = streams[0].unsqueeze(0).to(device)

    original_byte_mask = torch.zeros_like(batch, dtype=torch.float32)
    original_byte_mask[batch == -1].fill_(float("-inf"))

    with torch.inference_mode():
        features, attention_mask = adapter(batch)
        official_logits = model(shortest.clone())
        single_features, single_mask = adapter(shortest)
        reconstructed_logits = adapter.pool_and_classify(
            single_features, single_mask
        )

        extended = torch.full(
            (1, shortest.shape[1] + 4096),
            -1,
            dtype=shortest.dtype,
            device=device,
        )
        extended[:, : shortest.shape[1]] = shortest
        extended_features, extended_mask = adapter(extended)

    checkpoint_state = torch.load(CHECKPOINT, map_location="cpu")
    parameters_match = all(
        torch.equal(model.state_dict()[key].detach().cpu(), value)
        for key, value in checkpoint_state.items()
    )
    return {
        "adapter": adapter,
        "batch": batch,
        "streams": streams,
        "features": features,
        "attention_mask": attention_mask,
        "official_logits": official_logits,
        "reconstructed_logits": reconstructed_logits,
        "single_features": single_features,
        "single_mask": single_mask,
        "extended_features": extended_features,
        "extended_mask": extended_mask,
        "original_byte_mask": original_byte_mask,
        "load_report": load_report,
        "parameters_match": parameters_match,
    }


def test_a_output_shapes(verification: dict[str, Any]) -> None:
    features = verification["features"]
    attention_mask = verification["attention_mask"]
    byte_tensor = verification["batch"]
    assert features.ndim == 3
    assert features.shape[0] == byte_tensor.shape[0]
    assert features.shape[-1] == 192
    assert attention_mask.shape == features.shape[:2]
    assert set(attention_mask.unique().tolist()).issubset({0, 1})


def test_b_classification_path_equivalence(verification: dict[str, Any]) -> None:
    maximum, mean = _error(
        verification["official_logits"], verification["reconstructed_logits"]
    )
    assert maximum <= 1e-5, {"max_absolute_error": maximum, "mean": mean}


def test_c_mixed_length_batch_and_original_bug(
    verification: dict[str, Any]
) -> None:
    mask = verification["attention_mask"]
    valid_counts = mask.sum(dim=1)
    assert len(set(stream.numel() for stream in verification["streams"])) == 4
    assert not torch.isneginf(verification["original_byte_mask"]).any()
    assert bool((mask[0] == 0).any().item())
    for row, count in zip(mask, valid_counts):
        count = int(count.item())
        assert bool((row[:count] == 1).all().item())
        assert bool((row[count:] == 0).all().item())


def test_d_single_and_batch_consistency(verification: dict[str, Any]) -> None:
    valid_count = int(verification["attention_mask"][0].sum().item())
    maximum, mean = _error(
        verification["single_features"][0],
        verification["features"][0, :valid_count],
    )
    assert maximum <= 1e-5, {"max_absolute_error": maximum, "mean": mean}


def test_e_padding_length_invariance(verification: dict[str, Any]) -> None:
    valid_count = int(verification["extended_mask"].sum().item())
    assert valid_count == verification["single_features"].shape[1]
    maximum, mean = _error(
        verification["single_features"][0],
        verification["extended_features"][0, :valid_count],
    )
    assert maximum <= 1e-5, {"max_absolute_error": maximum, "mean": mean}


def test_f_official_checkpoint_integrity(verification: dict[str, Any]) -> None:
    report = verification["load_report"]
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert report["state_dict_exact_match"] is True
    assert report["classifier_out_features"] == 1000
    assert verification["parameters_match"] is True
