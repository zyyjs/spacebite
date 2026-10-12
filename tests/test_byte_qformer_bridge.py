"""Tests for the random-initialized instruction-aware Byte Q-Former bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from spacebyte.jpeg_bytes import pad_byte_sequences, preprocess_image_to_jpeg_bytes
from spacebyte.models import (
    ByteFormerTokenAdapter,
    ByteQFormerBridge,
    build_official_byteformer,
    freeze_byteformer,
)


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


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _error(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    difference = (left.float() - right.float()).abs()
    return float(difference.max().item()), float(difference.mean().item())


@pytest.fixture(scope="module")
def bridge() -> ByteQFormerBridge:
    torch.manual_seed(20261012)
    model = ByteQFormerBridge().to(_device())
    return model


@pytest.fixture(scope="module")
def synthetic_inputs() -> dict[str, torch.Tensor]:
    device = _device()
    generator = torch.Generator(device="cpu").manual_seed(73)
    byte_features = torch.randn(2, 231, 192, generator=generator).to(device)
    byte_mask = torch.zeros(2, 231, dtype=torch.long, device=device)
    byte_mask[0, :139] = 1
    byte_mask[1, :] = 1
    input_ids = torch.randint(
        1, 30523, (2, 16), generator=generator, dtype=torch.long
    ).to(device)
    text_mask = torch.zeros(2, 16, dtype=torch.long, device=device)
    text_mask[0, :7] = 1
    text_mask[1, :12] = 1
    input_ids[text_mask == 0] = 0
    return {
        "byte_features": byte_features,
        "byte_mask": byte_mask,
        "input_ids": input_ids,
        "text_mask": text_mask,
    }


def _forward(bridge: ByteQFormerBridge, inputs: dict[str, torch.Tensor]):
    return bridge(
        encoder_hidden_states=inputs["byte_features"],
        encoder_attention_mask=inputs["byte_mask"],
        input_ids=inputs["input_ids"],
        text_attention_mask=inputs["text_mask"],
    )


def test_a_shapes_and_mask_semantics(
    bridge: ByteQFormerBridge, synthetic_inputs: dict[str, torch.Tensor]
) -> None:
    bridge.eval()
    with torch.inference_mode():
        output = _forward(bridge, synthetic_inputs)
    assert output.query_features.shape == (2, 32, 768)
    assert output.projected_features.shape == (2, 32, 4096)
    assert output.query_attention_mask.shape == (2, 32)
    assert bool((output.query_attention_mask == 1).all().item())
    assert output.combined_attention_mask.shape == (2, 48)
    assert torch.equal(
        output.combined_attention_mask[:, 32:], synthetic_inputs["text_mask"]
    )
    assert torch.isfinite(output.query_features).all()
    assert torch.isfinite(output.projected_features).all()


def test_b_masked_byte_values_do_not_change_output(
    bridge: ByteQFormerBridge, synthetic_inputs: dict[str, torch.Tensor]
) -> None:
    bridge.eval()
    altered = {key: value.clone() for key, value in synthetic_inputs.items()}
    padding = altered["byte_mask"].eq(0).unsqueeze(-1)
    altered["byte_features"] = torch.where(
        padding,
        torch.randn_like(altered["byte_features"]) * 1000,
        altered["byte_features"],
    )
    with torch.inference_mode():
        baseline = _forward(bridge, synthetic_inputs).projected_features
        changed = _forward(bridge, altered).projected_features
    maximum, mean = _error(baseline, changed)
    assert maximum <= 1e-6, {"max_absolute_error": maximum, "mean": mean}


def test_c_adding_byte_padding_does_not_change_output(
    bridge: ByteQFormerBridge, synthetic_inputs: dict[str, torch.Tensor]
) -> None:
    bridge.eval()
    extended = {key: value.clone() for key, value in synthetic_inputs.items()}
    extra_features = torch.randn(
        2, 47, 192, device=_device(), dtype=extended["byte_features"].dtype
    )
    extended["byte_features"] = torch.cat(
        [extended["byte_features"], extra_features], dim=1
    )
    extended["byte_mask"] = torch.cat(
        [
            extended["byte_mask"],
            torch.zeros(2, 47, dtype=torch.long, device=_device()),
        ],
        dim=1,
    )
    with torch.inference_mode():
        baseline = _forward(bridge, synthetic_inputs).query_features
        changed = _forward(bridge, extended).query_features
    maximum, mean = _error(baseline, changed)
    assert maximum <= 1e-6, {"max_absolute_error": maximum, "mean": mean}


def test_d_text_padding_and_combined_mask(
    bridge: ByteQFormerBridge, synthetic_inputs: dict[str, torch.Tensor]
) -> None:
    bridge.eval()
    extended = {key: value.clone() for key, value in synthetic_inputs.items()}
    extra_ids = torch.randint(1, 30523, (2, 9), device=_device())
    extended["input_ids"] = torch.cat([extended["input_ids"], extra_ids], dim=1)
    extended["text_mask"] = torch.cat(
        [
            extended["text_mask"],
            torch.zeros(2, 9, dtype=torch.long, device=_device()),
        ],
        dim=1,
    )
    with torch.inference_mode():
        baseline = _forward(bridge, synthetic_inputs)
        changed = _forward(bridge, extended)
    maximum, mean = _error(baseline.projected_features, changed.projected_features)
    # Extending the sequence can select a different CUDA matrix kernel. The
    # masked values remain excluded; allow only ordinary FP32 round-off.
    assert maximum <= 5e-6, {"max_absolute_error": maximum, "mean": mean}
    assert baseline.combined_attention_mask.shape[1] == 32 + 16
    assert changed.combined_attention_mask.shape[1] == 32 + 25
    assert int(changed.combined_attention_mask[:, :32].sum().item()) == 64
    assert torch.equal(changed.combined_attention_mask[:, 32:], extended["text_mask"])


def test_e_gradient_flow(
    bridge: ByteQFormerBridge, synthetic_inputs: dict[str, torch.Tensor]
) -> None:
    bridge.train()
    bridge.zero_grad(set_to_none=True)
    output = _forward(bridge, synthetic_inputs)
    output.projected_features.square().mean().backward()

    named = dict(bridge.named_parameters())
    required = {
        "query_tokens": ["query_tokens"],
        "byte_layernorm": ["byte_layernorm.weight", "byte_layernorm.bias"],
        "self_attention": [name for name in named if ".attention.self.query." in name],
        "cross_attention": [
            name for name in named if ".crossattention.self.query." in name
        ],
        "query_ffn": [name for name in named if ".intermediate_query." in name],
        "llm_projector": ["llm_projector.weight", "llm_projector.bias"],
    }
    for component, names in required.items():
        assert names, f"No parameters found for {component}"
        assert any(
            named[name].grad is not None and bool((named[name].grad != 0).any().item())
            for name in names
        ), f"No non-zero gradient found for {component}"

    no_gradient = [
        name
        for name, parameter in named.items()
        if parameter.requires_grad
        and (parameter.grad is None or not bool((parameter.grad != 0).any().item()))
    ]
    # The last layer's text-only FFN is downstream of the final query output.
    # It is therefore expected not to contribute to this query-only loss.
    allowed_prefixes = (
        "qformer.bert.encoder.layer.11.intermediate.",
        "qformer.bert.encoder.layer.11.output.",
    )
    assert all(name.startswith(allowed_prefixes) for name in no_gradient), no_gradient
    bridge.eval()


def _two_real_streams(opts: Any) -> list[torch.Tensor]:
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


def test_f_end_to_end_two_real_jpegs(bridge: ByteQFormerBridge) -> None:
    if not CONFIG.is_file() or not CHECKPOINT.is_file() or not IMAGE_DIR.is_dir():
        pytest.skip("ByteFormer end-to-end assets are unavailable")
    device = _device()
    byteformer, opts, load_report = build_official_byteformer(CONFIG, CHECKPOINT)
    adapter = freeze_byteformer(ByteFormerTokenAdapter(byteformer).to(device))
    assert adapter.training is False
    assert not any(parameter.requires_grad for parameter in adapter.parameters())

    byte_batch = pad_byte_sequences(_two_real_streams(opts)).to(device)
    input_ids = torch.tensor(
        [[101, 2023, 2003, 1037, 3231, 102], [101, 2156, 1996, 3861, 102, 0]],
        dtype=torch.long,
        device=device,
    )
    text_mask = input_ids.ne(0).long()
    # no_grad freezes the encoder computation while producing ordinary tensors
    # that the trainable bridge can safely save for its backward pass.
    with torch.no_grad():
        byte_features, byte_mask = adapter(byte_batch)

    bridge.train()
    bridge.zero_grad(set_to_none=True)
    output = bridge(byte_features.detach(), byte_mask, input_ids, text_mask)
    output.projected_features.float().square().mean().backward()
    bridge.eval()

    with torch.inference_mode():
        output = bridge(byte_features, byte_mask, input_ids, text_mask)

    assert byte_features.shape[0] == 2 and byte_features.shape[-1] == 192
    assert output.projected_features.shape == (2, 32, 4096)
    assert torch.isfinite(output.projected_features).all()
    assert load_report["missing_keys"] == []
    assert load_report["unexpected_keys"] == []
    assert load_report["state_dict_exact_match"] is True
    assert all(parameter.grad is None for parameter in adapter.parameters())
    assert bridge.query_tokens.grad is not None
    assert bool((bridge.query_tokens.grad != 0).any().item())
