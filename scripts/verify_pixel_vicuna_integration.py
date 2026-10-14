#!/usr/bin/env python3
"""Run the one-shot real-weight Pixel -> Vicuna integration verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

from spacebyte.answer_only import audit_answer_only_labels, binary_answer_token_ids
from spacebyte.data import (
    fixed_derangement,
    flatten_spatialsense_annotations,
    render_record_to_jpeg_bytes,
    render_record_to_pixel_values,
)
from spacebyte.models import (
    ImageQFormerBridge,
    PixelVicuna,
    build_official_eva_vit_g,
    build_xinstruct_qformer_tokenizer,
    initialize_image_bridge_from_instructblip,
    load_frozen_vicuna,
    parameter_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--eva-checkpoint", type=Path, required=True)
    parser.add_argument("--instructblip-checkpoint", type=Path, required=True)
    parser.add_argument("--vicuna-path", type=Path, required=True)
    parser.add_argument("--qformer-tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def reset_peak(device: torch.device) -> int:
    torch.cuda.reset_peak_memory_stats(device)
    return int(torch.cuda.memory_allocated(device))


def finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def parameter_probes(module: nn.Module) -> dict[str, tuple[float, float, float, int]]:
    probes = {}
    for name, parameter in module.named_parameters():
        flat = parameter.detach().reshape(-1)
        probes[name] = (
            float(flat[0].float().cpu()),
            float(flat[flat.numel() // 2].float().cpu()),
            float(flat[-1].float().cpu()),
            int(parameter._version),
        )
    return probes


def gradient_report(model: PixelVicuna) -> dict[str, Any]:
    named = dict(model.bridge.named_parameters())

    def norm(name: str) -> float:
        gradient = named[name].grad
        return 0.0 if gradient is None else float(gradient.float().norm().item())

    cross = {}
    for layer in (0, 2, 4, 6, 8, 10):
        cross[str(layer)] = {
            kind: norm(
                f"qformer.bert.encoder.layer.{layer}.crossattention.self.{kind}.weight"
            )
            for kind in ("key", "value")
        }
    self_attention_names = [
        name for name in named if name.endswith("attention.self.query.weight")
        and "crossattention" not in name
    ]
    groups = {
        "query_tokens": norm("query_tokens"),
        "vision_layernorm": max(
            norm(name) for name in named if name.startswith("vision_layernorm.")
        ),
        "qformer_self_attention": max(norm(name) for name in self_attention_names),
        "projector": max(norm(name) for name in named if name.startswith("llm_projector.")),
    }
    all_gradients = [
        parameter.grad for parameter in model.bridge.parameters() if parameter.grad is not None
    ]
    return {
        "group_norms": groups,
        "cross_attention_key_value_norms": cross,
        "all_required_nonzero": all(value > 0 for value in groups.values())
        and all(value > 0 for pair in cross.values() for value in pair.values()),
        "all_gradients_finite": all(finite_tensor(gradient) for gradient in all_gradients),
        "visual_encoder_gradients": [
            name for name, parameter in model.visual_encoder.named_parameters()
            if parameter.grad is not None
        ],
        "vicuna_gradients": [
            name for name, parameter in model.vicuna.named_parameters()
            if parameter.grad is not None
        ],
    }


def select_distinct_images(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for record in records:
        if record["image_path"] not in seen:
            seen.add(record["image_path"])
            selected.append(record)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Could not find {count} distinct train images")


def predicted_logits(output, row: int) -> torch.Tensor:
    positions = output.prepared.labels[row].ne(-100).nonzero().flatten()
    if positions.numel() != 1:
        raise RuntimeError("Expected exactly one answer-only target")
    return output.logits[row, int(positions.item()) - 1]


def compare_logits(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    delta = (left.float() - right.float()).abs()
    return {"max_abs": float(delta.max().item()), "mean_abs": float(delta.mean().item())}


def main(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This verification requires one CUDA GPU")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.manual_seed(20261014)
    torch.cuda.manual_seed_all(20261014)
    torch.use_deterministic_algorithms(True, warn_only=True)

    report: dict[str, Any] = {
        "status": "running",
        "device": {
            "requested": str(device),
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "cuda_device_count": torch.cuda.device_count(),
            "used_gpu_indices": [device.index],
        },
        "weights": {},
    }
    for name, path, source in (
        (
            "eva_clip_g",
            args.eva_checkpoint,
            "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth",
        ),
        (
            "instructblip_vicuna7b_trimmed",
            args.instructblip_checkpoint,
            "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth",
        ),
    ):
        report["weights"][name] = {
            "official_source": source,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    report["lavis_revision"] = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1] / "lavis"), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    reset_peak(device)
    visual_encoder, eva_audit = build_official_eva_vit_g(args.eva_checkpoint)
    visual_encoder.to(device)
    bridge = ImageQFormerBridge().to(device)
    bridge_audit = initialize_image_bridge_from_instructblip(
        bridge, args.instructblip_checkpoint
    )
    vicuna, llm_tokenizer, vicuna_report = load_frozen_vicuna(
        args.vicuna_path, device=device, dtype=torch.float16
    )
    qformer_tokenizer = build_xinstruct_qformer_tokenizer(args.qformer_tokenizer)
    model = PixelVicuna(
        visual_encoder=visual_encoder,
        bridge=bridge,
        vicuna=vicuna,
        llm_tokenizer=llm_tokenizer,
        qformer_tokenizer=qformer_tokenizer,
    ).to(device)
    report["memory_after_model_load"] = memory(device)
    report["weight_loading"] = {
        "eva": eva_audit,
        "image_bridge": bridge_audit,
        "vicuna": vicuna_report,
    }
    report["parameters"] = parameter_summary(model)

    train_records = flatten_spatialsense_annotations(
        args.dataset_root, include_splits={"train"}
    )
    selected = select_distinct_images(train_records, 3)
    marked = [render_record_to_pixel_values(args.dataset_root, record) for record in selected]
    byte_again, _rendered, boxes_again = render_record_to_jpeg_bytes(
        args.dataset_root, selected[0]
    )
    report["shared_visual_semantics"] = {
        "sample_id": marked[0].sample_id,
        "same_jpeg_bytes": bool(torch.equal(marked[0].jpeg_bytes, byte_again)),
        "same_boxes": marked[0].boxes == boxes_again,
        "boxes": marked[0].boxes,
        "jpeg_byte_length": int(marked[0].jpeg_bytes.numel()),
    }
    if not all(report["shared_visual_semantics"][key] for key in ("same_jpeg_bytes", "same_boxes")):
        raise RuntimeError("Pixel and Byte branches did not share the exact marked JPEG")

    single_pixels = marked[0].pixel_values.unsqueeze(0).to(device)
    question = selected[0]["question"]
    answer = selected[0]["answer"]

    model.eval()
    reset_peak(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        forward_one = model(
            single_pixels, question, answer, answer_only_loss=True
        )
    torch.cuda.synchronize(device)
    report["memory_forward"] = memory(device)
    answer_audit = audit_answer_only_labels(
        forward_one.prepared, [answer], llm_tokenizer
    )
    labels = forward_one.prepared.labels
    conservation = int(labels.eq(-100).sum() + labels.ne(-100).sum()) == labels.numel()
    report["shapes"] = {
        "pixel_values": list(forward_one.prepared.pixel_values.shape),
        "vision_features": list(forward_one.prepared.vision_features.shape),
        "vision_attention_mask": list(forward_one.prepared.vision_attention_mask.shape),
        "image_query_features": list(forward_one.prepared.image_query_features.shape),
        "projected_image_queries": list(forward_one.prepared.projected_image_queries.shape),
        "vicuna_inputs_embeds": list(forward_one.prepared.vicuna_inputs_embeds.shape),
        "vicuna_attention_mask": list(forward_one.prepared.vicuna_attention_mask.shape),
        "labels": list(labels.shape),
        "logits": list(forward_one.logits.shape),
    }
    answer_ids = binary_answer_token_ids(llm_tokenizer)
    answer_logits = predicted_logits(forward_one, 0)
    report["single_sample_forward"] = {
        "loss": float(forward_one.loss.item()),
        "all_outputs_finite": all(
            finite_tensor(value)
            for value in (
                forward_one.loss,
                forward_one.logits,
                forward_one.prepared.vision_features,
                forward_one.prepared.image_query_features,
                forward_one.prepared.projected_image_queries,
            )
        ),
        "answer_only_audit": answer_audit,
        "label_conservation": conservation,
        "causal_shift_correct": all(row["causal_shift_is_one"] for row in answer_audit["rows"]),
        "predicted_answer": "yes"
        if answer_logits[answer_ids["yes"]] >= answer_logits[answer_ids["no"]]
        else "no",
    }

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        repeat_a = model(single_pixels, question, answer, answer_only_loss=True)
        repeat_b = model(single_pixels, question, answer, answer_only_loss=True)
    report["determinism"] = compare_logits(repeat_a.logits, repeat_b.logits)

    batch_pixels = torch.stack([item.pixel_values for item in marked]).to(device)
    questions = [record["question"] for record in selected]
    answers = [record["answer"] for record in selected]
    permutation = fixed_derangement(3, 20261014)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        normal = model(batch_pixels, questions, answers, answer_only_loss=True)
        zero = model(
            batch_pixels, questions, answers, answer_only_loss=True, pixel_condition="zero"
        )
        shuffled = model(
            batch_pixels,
            questions,
            answers,
            answer_only_loss=True,
            pixel_condition="shuffled",
            shuffled_indices=permutation,
        )
    normal_next = torch.stack([predicted_logits(normal, index) for index in range(3)])
    zero_next = torch.stack([predicted_logits(zero, index) for index in range(3)])
    shuffled_next = torch.stack([predicted_logits(shuffled, index) for index in range(3)])
    report["pixel_conditions"] = {
        "shuffle_seed": 20261014,
        "global_derangement": permutation.tolist(),
        "sample_ids": [item.sample_id for item in marked],
        "normal_vs_zero": compare_logits(normal_next, zero_next),
        "normal_vs_shuffled": compare_logits(normal_next, shuffled_next),
        "outputs_differ": bool(
            not torch.equal(normal_next, zero_next)
            and not torch.equal(normal_next, shuffled_next)
        ),
    }

    visual_before = parameter_probes(model.visual_encoder)
    vicuna_before = parameter_probes(model.vicuna)
    model.train()
    model.zero_grad(set_to_none=True)
    reset_peak(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        backward_output = model(
            single_pixels, question, answer, answer_only_loss=True
        )
    forward_peak_during_backward = memory(device)
    backward_output.loss.backward()
    torch.cuda.synchronize(device)
    report["memory_loss_forward"] = forward_peak_during_backward
    report["memory_backward"] = memory(device)
    gradients = gradient_report(model)
    gradients["frozen_visual_parameters_unchanged"] = (
        visual_before == parameter_probes(model.visual_encoder)
    )
    gradients["frozen_vicuna_parameters_unchanged"] = (
        vicuna_before == parameter_probes(model.vicuna)
    )
    report["gradients"] = gradients

    # Save only the trainable image bridge; official frozen weights remain external.
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        checkpoint_path = Path(temporary) / "pixel_trainable_roundtrip.pt"
        torch.save(model.bridge.state_dict(), checkpoint_path)
        reloaded = ImageQFormerBridge()
        result = reloaded.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True
        )
        exact = all(
            torch.equal(value.cpu(), reloaded.state_dict()[key])
            for key, value in model.bridge.state_dict().items()
        )
        report["checkpoint_roundtrip"] = {
            "scope": "trainable Image Q-Former, Query Tokens, vision LayerNorm, projector",
            "strict": True,
            "missing_keys": result.missing_keys,
            "unexpected_keys": result.unexpected_keys,
            "all_tensors_exact": exact,
        }

    required = [
        report["single_sample_forward"]["all_outputs_finite"],
        report["single_sample_forward"]["label_conservation"],
        report["single_sample_forward"]["causal_shift_correct"],
        report["determinism"]["max_abs"] == 0.0,
        report["pixel_conditions"]["outputs_differ"],
        gradients["all_required_nonzero"],
        gradients["all_gradients_finite"],
        gradients["visual_encoder_gradients"] == [],
        gradients["vicuna_gradients"] == [],
        gradients["frozen_visual_parameters_unchanged"],
        gradients["frozen_vicuna_parameters_unchanged"],
        report["checkpoint_roundtrip"]["all_tensors_exact"],
    ]
    report["status"] = "passed" if all(required) else "failed"
    report["elapsed_seconds"] = time.perf_counter() - started
    if report["status"] != "passed":
        raise RuntimeError("One or more Pixel-Vicuna verification conditions failed")
    return report


def entrypoint() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = main(args)
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "oom": isinstance(error, torch.cuda.OutOfMemoryError),
        }
        if torch.cuda.is_available():
            device = torch.device(args.device)
            failure["memory_at_failure"] = memory(device)
        args.output.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "shapes": report["shapes"],
        "peak_backward_bytes": report["memory_backward"]["peak_allocated_bytes"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    entrypoint()
