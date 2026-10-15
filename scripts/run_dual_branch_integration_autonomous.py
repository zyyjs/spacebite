#!/usr/bin/env python3
"""One-shot real-weight verification for frozen Pixel+Byte -> Vicuna wiring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from spacebyte.answer_only import audit_answer_only_labels, binary_answer_token_ids
from spacebyte.data import (
    fixed_derangement,
    flatten_spatialsense_annotations,
    render_record_to_pixel_values,
    sha256_file,
)
from spacebyte.models import (
    ByteFormerTokenAdapter,
    ByteQFormerBridge,
    ByteVicuna,
    DualBranchVicuna,
    ImageQFormerBridge,
    PixelVicuna,
    build_official_byteformer,
    build_official_eva_vit_g,
    build_xinstruct_qformer_tokenizer,
    dual_parameter_summary,
    initialize_byte_qformer_from_blip2_stage1,
    initialize_image_bridge_from_instructblip,
    load_frozen_vicuna,
)
from spacebyte.pixel_training import feature_similarity_metrics


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data/students/Zhang_jinsong/spacebyte")
DEFAULT_OUTPUT = DATA_ROOT / "result/fusion/dual_branch_integration"
GB = 1024**3
TERMINATE = {"requested": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT / "dataset/SpatialSense")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pixel-checkpoint",
        type=Path,
        default=DATA_ROOT / "result/pixel/spatialsense_pixel_baseline/best.pt",
    )
    parser.add_argument(
        "--byte-checkpoint",
        type=Path,
        default=DATA_ROOT
        / "result/byteformer/spatialsense_byteformer_stability_longrun/stability_best.pt",
    )
    parser.add_argument(
        "--eva-checkpoint",
        type=Path,
        default=DATA_ROOT / "models/pixel_vicuna_official/eva_vit_g.pth",
    )
    parser.add_argument(
        "--instructblip-checkpoint",
        type=Path,
        default=DATA_ROOT
        / "models/pixel_vicuna_official/instruct_blip_vicuna7b_trimmed.pth",
    )
    parser.add_argument(
        "--byteformer-config",
        type=Path,
        default=ROOT / "corenet/projects/byteformer/imagenet_jpeg_q100/conv_kernel_size=8.yaml",
    )
    parser.add_argument(
        "--byteformer-checkpoint",
        type=Path,
        default=DATA_ROOT
        / "models/byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt",
    )
    parser.add_argument(
        "--blip2-checkpoint",
        type=Path,
        default=DATA_ROOT / "models/blip2_stage1/blip2_pretrained.pth",
    )
    parser.add_argument(
        "--vicuna-path", type=Path, default=DATA_ROOT / "models/vicuna-7b-v1.1"
    )
    parser.add_argument(
        "--qformer-tokenizer",
        type=Path,
        default=DATA_ROOT / "models/bert-base-uncased-tokenizer",
    )
    parser.add_argument(
        "--pixel-cache-root",
        type=Path,
        default=DATA_ROOT / "cache/spatialsense_pixel_eva_g_fp32_forward",
    )
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--minimum-free-gib", type=float, default=22.0)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def acquire_lock(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / "RUNNING.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            lock.unlink()
        else:
            raise RuntimeError(f"Dual integration is already running as PID {pid}")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    atomic_text(output_dir / "run.pid", f"{os.getpid()}\n")
    return lock


def git_metadata() -> dict[str, str]:
    def run(*values: str) -> str:
        return subprocess.check_output(["git", *values], cwd=ROOT, text=True).strip()

    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "status": run("status", "--short"),
    }


def update_status(path: Path, status: dict[str, Any], **changes: Any) -> None:
    status.update(changes)
    write_json(path, status)


def on_signal(signum, _frame) -> None:
    TERMINATE["requested"] = True
    print(f"termination signal {signum} received", flush=True)


def check_termination() -> None:
    if TERMINATE["requested"]:
        raise RuntimeError("verification terminated by signal")


def select_verification_records(
    records: Sequence[Mapping[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Choose distinct non-test images and relations without reading test records."""

    if count < 3:
        raise ValueError("Dual shuffled validation requires at least three samples")
    if any(record["split"] not in {"train", "valid"} for record in records):
        raise RuntimeError("Dual integration received a forbidden split")
    result: list[dict[str, Any]] = []
    images: set[str] = set()
    relations: set[str] = set()
    # Ensure both allowed splits are represented before filling remaining rows.
    for desired_split in ("train", "valid"):
        for record in records:
            if record["split"] != desired_split or record["image_path"] in images:
                continue
            result.append(dict(record))
            images.add(record["image_path"])
            relations.add(record["relation_raw"])
            break
    for record in records:
        if len(result) >= count:
            break
        if record["image_path"] in images or record["relation_raw"] in relations:
            continue
        result.append(dict(record))
        images.add(record["image_path"])
        relations.add(record["relation_raw"])
    if len(result) != count:
        raise RuntimeError("Could not select enough distinct verification records")
    return result


def file_digest_from_int_tensor(values: torch.Tensor) -> str:
    payload = values.to(dtype=torch.uint8, device="cpu").numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def pad_jpeg_batch(values: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    width = max(int(value.numel()) for value in values)
    batch = torch.full((len(values), width), -1, dtype=torch.int32)
    for index, value in enumerate(values):
        batch[index, : value.numel()] = value
    return batch.to(device)


def strict_task_bridge_load(module, checkpoint: Path, key: str) -> dict[str, Any]:
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    source = artifact.get(key)
    if not isinstance(source, Mapping) or not all(torch.is_tensor(v) for v in source.values()):
        raise RuntimeError(f"{checkpoint} has no tensor mapping {key}")
    target = module.state_dict()
    missing = sorted(set(target) - set(source))
    unexpected = sorted(set(source) - set(target))
    mismatches = {
        name: {"source": list(source[name].shape), "target": list(target[name].shape)}
        for name in set(source) & set(target)
        if source[name].shape != target[name].shape
    }
    if missing or unexpected or mismatches:
        raise RuntimeError(
            f"Strict task checkpoint mismatch: missing={missing[:5]} "
            f"unexpected={unexpected[:5]} shapes={list(mismatches)[:5]}"
        )
    incompatible = module.load_state_dict(source, strict=True)
    failed = [
        name for name, value in source.items() if not torch.equal(module.state_dict()[name], value)
    ]
    if failed:
        raise RuntimeError(f"Post-load exact task checkpoint audit failed: {failed[:5]}")
    report = {
        "path": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "tensor_count": len(source),
        "parameter_elements": sum(int(value.numel()) for value in source.values()),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "shape_mismatch": mismatches,
        "post_load_all_tensors_exact": True,
    }
    del artifact, source
    return report


def strict_partial_byteformer_load(byteformer, checkpoint: Path) -> dict[str, Any]:
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    source = artifact.get("byteformer_partial_state_dict")
    if not isinstance(source, Mapping):
        raise RuntimeError("Byte task checkpoint has no partial ByteFormer state")
    parameters = dict(byteformer.named_parameters())
    unexpected = sorted(set(source) - set(parameters))
    mismatches = {
        name: {"source": list(value.shape), "target": list(parameters[name].shape)}
        for name, value in source.items()
        if name in parameters and value.shape != parameters[name].shape
    }
    if unexpected or mismatches:
        raise RuntimeError(
            f"Partial ByteFormer checkpoint mismatch: unexpected={unexpected} shapes={mismatches}"
        )
    with torch.no_grad():
        for name, value in source.items():
            parameters[name].copy_(value)
    failed = [
        name for name, value in source.items() if not torch.equal(parameters[name].detach(), value)
    ]
    if failed:
        raise RuntimeError(f"Partial ByteFormer exact audit failed: {failed[:5]}")
    report = {
        "path": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "loaded_tensor_count": len(source),
        "loaded_parameter_elements": sum(int(value.numel()) for value in source.values()),
        "missing_keys": [],
        "unexpected_keys": unexpected,
        "shape_mismatch": mismatches,
        "post_load_all_tensors_exact": True,
        "loaded_keys": sorted(source),
    }
    del artifact, source
    return report


def load_models(args: argparse.Namespace, device: torch.device):
    image_bridge = ImageQFormerBridge()
    image_official = initialize_image_bridge_from_instructblip(
        image_bridge, args.instructblip_checkpoint
    )
    image_task = strict_task_bridge_load(
        image_bridge, args.pixel_checkpoint, "bridge_state_dict"
    )

    byte_bridge = ByteQFormerBridge()
    byte_official = initialize_byte_qformer_from_blip2_stage1(
        byte_bridge, args.blip2_checkpoint
    )
    byte_task = strict_task_bridge_load(
        byte_bridge, args.byte_checkpoint, "bridge_state_dict"
    )

    byteformer, _, byteformer_official = build_official_byteformer(
        args.byteformer_config, args.byteformer_checkpoint
    )
    byteformer_task = strict_partial_byteformer_load(byteformer, args.byte_checkpoint)
    byte_adapter = ByteFormerTokenAdapter(byteformer)

    eva, eva_official = build_official_eva_vit_g(args.eva_checkpoint)
    image_bridge.to(device)
    byte_bridge.to(device)
    byte_adapter.to(device)
    eva.to(device)

    vicuna, llm_tokenizer, vicuna_report = load_frozen_vicuna(
        args.vicuna_path, device=device, dtype=torch.float16
    )
    vicuna_weight_files = sorted(
        [*args.vicuna_path.glob("*.bin"), *args.vicuna_path.glob("*.safetensors")]
    )
    if not vicuna_weight_files:
        raise RuntimeError("Vicuna directory contains no local weight shards")
    vicuna_report["weight_files"] = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in vicuna_weight_files
    ]
    vicuna_report["weight_bytes_total"] = sum(
        item["bytes"] for item in vicuna_report["weight_files"]
    )
    qformer_tokenizer = build_xinstruct_qformer_tokenizer(args.qformer_tokenizer)
    model = DualBranchVicuna(
        visual_encoder=eva,
        image_bridge=image_bridge,
        byteformer_adapter=byte_adapter,
        byte_bridge=byte_bridge,
        vicuna=vicuna,
        llm_tokenizer=llm_tokenizer,
        qformer_tokenizer=qformer_tokenizer,
    ).to(device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Dual integration must freeze every model parameter")
    report = {
        "eva_official": eva_official,
        "image_bridge_official": {
            key: image_official[key]
            for key in ("loaded", "checkpoint_only", "model_only", "shape_mismatch", "post_load_exact_match")
        },
        "pixel_task_checkpoint": image_task,
        "byteformer_official": byteformer_official,
        "byteformer_task_checkpoint": byteformer_task,
        "byte_qformer_blip2": byte_official["summary"],
        "byte_task_checkpoint": byte_task,
        "vicuna": vicuna_report,
        "parameters": dual_parameter_summary(model),
    }
    return model, report


def snapshot_parameters(model: DualBranchVicuna) -> dict[str, torch.Tensor]:
    probes = {
        "eva": next(model.visual_encoder.parameters()).detach().reshape(-1)[:4096],
        "image_query_tokens": model.image_bridge.query_tokens.detach(),
        "image_projector": model.image_bridge.llm_projector.weight.detach().reshape(-1)[:4096],
        "byteformer": next(model.byteformer_adapter.parameters()).detach().reshape(-1)[:4096],
        "byte_query_tokens": model.byte_bridge.query_tokens.detach(),
        "byte_projector": model.byte_bridge.llm_projector.weight.detach().reshape(-1)[:4096],
        "vicuna_embedding": model.vicuna.get_input_embeddings().weight.detach().reshape(-1)[:4096],
    }
    return {name: value.cpu().clone() for name, value in probes.items()}


def compare_snapshot(model: DualBranchVicuna, before: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    after = snapshot_parameters(model)
    unchanged = {name: torch.equal(value, after[name]) for name, value in before.items()}
    if not all(unchanged.values()):
        raise RuntimeError(f"Frozen parameter probes changed: {unchanged}")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Frozen integration unexpectedly created gradients")
    return {"representative_tensor_exact": unchanged, "all_gradients_none": True}


def load_pixel_cached_features(
    cache_root: Path, records: Sequence[Mapping[str, Any]], jpeg_hashes: Mapping[str, str]
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    manifests: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid"):
        path = cache_root / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            manifests[row["sample_id"]] = row
    features = []
    audits = []
    for record in records:
        row = manifests.get(record["sample_id"])
        if row is None:
            raise RuntimeError(f"Pixel cache lacks sample {record['sample_id']}")
        if row["split"] != record["split"] or row["image_path"] != record["image_path"]:
            raise RuntimeError("Pixel cache discrete sample mapping mismatch")
        artifact = torch.load(
            cache_root / row["feature_relative_path"], map_location="cpu", weights_only=True
        )
        expected_hash = jpeg_hashes[record["sample_id"]]
        if (
            artifact["pixel_cache_key"] != row["pixel_cache_key"]
            or artifact["processed_jpeg_sha256"] != expected_hash
            or artifact["vision_features"].shape != (257, 1408)
            or artifact["vision_features"].dtype != torch.float16
            or not torch.isfinite(artifact["vision_features"]).all()
        ):
            raise RuntimeError("Pixel cache identity, shape, dtype, or finite audit failed")
        features.append(artifact["vision_features"])
        audits.append(
            {
                "sample_id": record["sample_id"],
                "split": record["split"],
                "cache_key": row["pixel_cache_key"],
                "jpeg_sha256": expected_hash,
                "shape": list(artifact["vision_features"].shape),
            }
        )
    batch = torch.stack(features)
    return batch, torch.ones(batch.shape[:2], dtype=torch.long), audits


def tensor_shapes(prepared) -> dict[str, list[int]]:
    names = (
        "pixel_values",
        "pixel_vision_features",
        "pixel_vision_attention_mask",
        "pixel_query_features",
        "projected_pixel_queries",
        "jpeg_bytes",
        "byte_features",
        "byte_attention_mask",
        "byte_query_features",
        "projected_byte_queries",
        "concatenated_visual_tokens",
        "vicuna_inputs_embeds",
        "vicuna_attention_mask",
        "labels",
    )
    return {name: list(getattr(prepared, name).shape) for name in names}


def score_rows(output, token_ids: Mapping[str, int]) -> list[float]:
    rows = []
    for index, answer_slice in enumerate(output.prepared.answer_slices):
        position = answer_slice.start - 1
        logits = output.logits[index, position]
        rows.append(float((logits[token_ids["yes"]] - logits[token_ids["no"]]).item()))
    return rows


def condition_summary(output, reference_logits, token_ids) -> dict[str, Any]:
    logits = output.logits.detach().float()
    difference = (logits - reference_logits).abs()
    labels = output.prepared.labels
    return {
        "logits_finite": bool(torch.isfinite(output.logits).all().item()),
        "loss": float(output.loss.item()),
        "loss_finite": bool(torch.isfinite(output.loss).item()),
        "yes_minus_no_scores": score_rows(output, token_ids),
        "logits_max_abs_difference_from_pixel_byte": float(difference.max().item()),
        "logits_mean_abs_difference_from_pixel_byte": float(difference.mean().item()),
        "top1_changed_count": int(
            output.logits.argmax(-1).ne(reference_logits.argmax(-1)).sum().item()
        ),
        "attention_mask": {
            "shape": list(output.prepared.vicuna_attention_mask.shape),
            "valid": int(output.prepared.vicuna_attention_mask.sum().item()),
            "padding": int(output.prepared.vicuna_attention_mask.eq(0).sum().item()),
        },
        "labels": {
            "shape": list(labels.shape),
            "ignored": int(labels.eq(-100).sum().item()),
            "supervised": int(labels.ne(-100).sum().item()),
            "conservation": int(labels.numel()),
        },
    }


def render_report(report: Mapping[str, Any], path: Path) -> None:
    conditions = report["conditions"]
    condition_lines = [
        "| Condition | Loss | finite | max Δ logits | mean Δ logits | top-1 changed |",
        "|---|---:|:---:|---:|---:|---:|",
    ]
    for name, row in conditions.items():
        condition_lines.append(
            f"| {name} | {row['loss']:.6f} | {row['logits_finite'] and row['loss_finite']} "
            f"| {row['logits_max_abs_difference_from_pixel_byte']:.6g} "
            f"| {row['logits_mean_abs_difference_from_pixel_byte']:.6g} "
            f"| {row['top1_changed_count']} |"
        )
    memory = report["gpu_memory_gib"]
    online_cached = report["online_cached"]
    if online_cached["status"] == "compared":
        online_cached_lines = (
            f"- Pixel projected-query similarity: cosine minimum "
            f"{online_cached['pixel_projected_queries']['minimum_sample_cosine_similarity']:.8f}, "
            f"relative L2 {online_cached['pixel_projected_queries']['relative_l2_error']:.8f}.\n"
            f"- Byte projected-query similarity: cosine minimum "
            f"{online_cached['byte_projected_queries']['minimum_sample_cosine_similarity']:.8f}, "
            f"relative L2 {online_cached['byte_projected_queries']['relative_l2_error']:.8f}."
        )
    else:
        online_cached_lines = (
            "- Online dual-encoder execution ran out of memory; the exception and peak "
            "were retained, and verified cache mode completed without changing the architecture."
        )
    text = f"""# Pixel+Byte Dual-Branch Integration

## Scope

Inference-only wiring of frozen Pixel and Byte branches into frozen Vicuna. No
optimizer, backward pass, parameter update, gating, LoRA, augmentation, or
SpatialSense test data was used.

## Result

Status: **{report['conclusion']}**. Pixel Query Tokens precede Byte Query Tokens,
both fixed weights are 1.0, and every model parameter remained frozen and exact
on representative before/after probes.

## Real shapes

```json
{json.dumps(report['shapes'], indent=2)}
```

## Six conditions

{chr(10).join(condition_lines)}

These values are wiring diagnostics only and are not Accuracy, Macro-F1, or AUROC results.

## Online and cached paths

{online_cached_lines}
- The existing Pixel FP32-forward/FP16-storage cache was used. Because the final Byte checkpoint changed ByteFormer blocks 10/11 and post norm, the stale pre-training Byte cache was not reused; this verifier cached only the selected final-ByteFormer outputs in memory.

## Loading and freezing

- Pixel task bridge: {report['loading']['pixel_task_checkpoint']['tensor_count']} tensors, strict and exact.
- Byte task bridge: {report['loading']['byte_task_checkpoint']['tensor_count']} tensors, strict and exact.
- ByteFormer task delta: {report['loading']['byteformer_task_checkpoint']['loaded_tensor_count']} tensors, exact.
- All trainable parameters after assembly: 0.

## GPU memory

- Loaded: {memory['model_loaded']:.3f} GiB
- Online dual encoders: {memory['online_dual_encoders']:.3f} GiB
- Cached encoder features: {memory['cached_encoder_features']:.3f} GiB
- Complete loss forward: {memory['complete_loss_forward']:.3f} GiB

## Provenance

- Samples: {', '.join(report['sample_ids'])}
- Git: `{report['git']['branch']}` at `{report['git']['commit']}`
- JSON: `{report['json_path']}`
"""
    atomic_text(path, text)


def run(args: argparse.Namespace, status: dict[str, Any], status_path: Path) -> dict[str, Any]:
    required = (
        args.pixel_checkpoint,
        args.byte_checkpoint,
        args.eva_checkpoint,
        args.instructblip_checkpoint,
        args.byteformer_config,
        args.byteformer_checkpoint,
        args.blip2_checkpoint,
        args.vicuna_path,
        args.qformer_tokenizer,
        args.pixel_cache_root,
    )
    absent = [str(path) for path in required if not path.exists()]
    if absent:
        raise FileNotFoundError(f"Required integration inputs are missing: {absent}")
    if args.device != "cuda:1":
        raise RuntimeError("This verification is pinned to cuda:1 unless the code is reviewed")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        raise RuntimeError("cuda:1 is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    if free < args.minimum_free_gib * GB:
        raise RuntimeError(
            f"GPU 1 has only {free / GB:.2f} GiB free; required {args.minimum_free_gib:.2f} GiB"
        )
    gpu_initial = {
        "name": torch.cuda.get_device_name(device),
        "free_gib": free / GB,
        "total_gib": total / GB,
    }

    update_status(status_path, status, phase="select_real_samples")
    # The loader is explicitly restricted at source, so test annotations are
    # never parsed or exposed to the verifier.
    records = flatten_spatialsense_annotations(
        args.dataset_root, include_splits={"train", "valid"}
    )
    selected = select_verification_records(records, args.sample_count)
    rendered = [render_record_to_pixel_values(args.dataset_root, record) for record in selected]
    pixels = torch.stack([item.pixel_values for item in rendered]).to(device)
    jpeg = pad_jpeg_batch([item.jpeg_bytes for item in rendered], device)
    questions = [record["question"] for record in selected]
    answers = [record["answer"] for record in selected]
    jpeg_hashes = {
        record["sample_id"]: file_digest_from_int_tensor(item.jpeg_bytes)
        for record, item in zip(selected, rendered)
    }
    same_source = all(
        record["sample_id"] == item.sample_id
        for record, item in zip(selected, rendered)
    )
    if not same_source:
        raise RuntimeError("Pixel and Byte samples do not share sample IDs")

    check_termination()
    update_status(status_path, status, phase="checkpoint_audit_and_model_load")
    torch.cuda.reset_peak_memory_stats(device)
    model, loading = load_models(args, device)
    loaded_memory = torch.cuda.max_memory_allocated(device) / GB
    before = snapshot_parameters(model)

    update_status(status_path, status, phase="online_dual_forward")
    torch.cuda.reset_peak_memory_stats(device)
    online_output = None
    pixel_features = pixel_mask = byte_features = byte_mask = None
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            encoded = model.encode_online(pixels, jpeg, questions)
        online_pixel, online_byte, pixel_features, pixel_mask, byte_features, byte_mask = encoded
        if not all(
            torch.isfinite(value).all()
            for value in (
                pixel_features,
                byte_features,
                online_pixel.projected_features,
                online_byte.projected_features,
            )
        ):
            raise FloatingPointError("Online encoder or Query output is non-finite")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            online_output = model.forward_from_features(
                pixel_features,
                pixel_mask,
                byte_features,
                byte_mask,
                questions,
                answers,
            )
        if not torch.isfinite(online_output.loss) or not torch.isfinite(
            online_output.logits
        ).all():
            raise FloatingPointError("Online complete loss path is non-finite")
        update_status(status_path, status, online_mode_status="passed")
    except torch.cuda.OutOfMemoryError as error:
        print(f"online dual-encoder mode OOM: {error}", flush=True)
        update_status(status_path, status, online_mode_status="oom")
        online_output = None
        pixel_features = pixel_mask = byte_features = byte_mask = None
        torch.cuda.empty_cache()
    online_encoder_memory = torch.cuda.max_memory_allocated(device) / GB

    # Existing Pixel cache has strict sample/JPEG provenance.  The final Byte
    # checkpoint changed its encoder, so cache only current frozen outputs for
    # these selected samples instead of reusing the stale official cache.
    cached_pixel_cpu, cached_pixel_mask_cpu, pixel_cache_audit = load_pixel_cached_features(
        args.pixel_cache_root, selected, jpeg_hashes
    )
    if byte_features is None or byte_mask is None:
        # OOM in the combined online path does not invalidate cache-mode
        # verification. Produce only the small selected Byte cache sequentially.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            byte_features, byte_mask = model.byteformer_adapter(jpeg)
    cached_byte_cpu = byte_features.detach().to(dtype=torch.float16, device="cpu").clone()
    cached_byte_mask_cpu = byte_mask.detach().to(device="cpu").clone()
    cached_pixel = cached_pixel_cpu.to(device)
    cached_pixel_mask = cached_pixel_mask_cpu.to(device)
    cached_byte = cached_byte_cpu.to(device)
    cached_byte_mask = cached_byte_mask_cpu.to(device)

    update_status(status_path, status, phase="single_branch_consistency")
    pixel_single = PixelVicuna(
        visual_encoder=model.visual_encoder,
        bridge=model.image_bridge,
        vicuna=model.vicuna,
        llm_tokenizer=model.llm_tokenizer,
        qformer_tokenizer=model.qformer_tokenizer,
    ).eval()
    byte_single = ByteVicuna(
        byteformer_adapter=model.byteformer_adapter,
        bridge=model.byte_bridge,
        vicuna=model.vicuna,
        llm_tokenizer=model.llm_tokenizer,
        qformer_tokenizer=model.qformer_tokenizer,
    ).eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        pixel_independent = pixel_single.prepare_inputs_from_vision_features(
            cached_pixel, cached_pixel_mask, questions, answers, answer_only_loss=True
        )
        byte_independent = byte_single.prepare_inputs_from_byte_features(
            cached_byte, cached_byte_mask, questions, answers, answer_only_loss=True
        )

    update_status(status_path, status, phase="cached_dual_and_six_conditions")
    token_ids = binary_answer_token_ids(model.llm_tokenizer)
    permutation = fixed_derangement(args.sample_count, args.seed).to(device)
    conditions: dict[str, Any] = {}
    normal_output = None
    reference_logits = None
    cached_memory = 0.0
    full_loss_memory = 0.0
    cached_prepared = None
    for name in (
        "pixel_byte",
        "pixel_only",
        "byte_only",
        "zero_both",
        "shuffled_pixel",
        "shuffled_byte",
    ):
        check_termination()
        torch.cuda.reset_peak_memory_stats(device)
        kwargs = {"condition": name}
        if name.startswith("shuffled_"):
            kwargs["shuffled_indices"] = permutation
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output = model.forward_from_features(
                cached_pixel,
                cached_pixel_mask,
                cached_byte,
                cached_byte_mask,
                questions,
                answers,
                **kwargs,
            )
        peak = torch.cuda.max_memory_allocated(device) / GB
        cached_memory = max(cached_memory, peak)
        full_loss_memory = max(full_loss_memory, peak)
        if name == "pixel_byte":
            normal_output = output
            cached_prepared = output.prepared
            reference_logits = output.logits.detach().float().cpu()
            reference_for_summary = output.logits.detach().float()
        else:
            reference_for_summary = reference_logits.to(device)
        conditions[name] = condition_summary(output, reference_for_summary, token_ids)
        if name != "pixel_byte":
            del output, reference_for_summary
    assert normal_output is not None and cached_prepared is not None
    label_audit = audit_answer_only_labels(cached_prepared, answers, model.llm_tokenizer)
    if (
        not label_audit["exactly_one_supervised_token_per_sample"]
        or not label_audit["eos_fully_ignored"]
        or label_audit["ignored_token_count"] + label_audit["supervised_token_count"]
        != cached_prepared.labels.numel()
    ):
        raise RuntimeError("Answer-only or causal label audit failed")

    pixel_single_similarity = feature_similarity_metrics(
        pixel_independent.projected_image_queries,
        cached_prepared.projected_pixel_queries,
    )
    byte_single_similarity = feature_similarity_metrics(
        byte_independent.projected_queries,
        cached_prepared.projected_byte_queries,
    )
    if not pixel_single_similarity["torch_equal"] or not byte_single_similarity["torch_equal"]:
        raise RuntimeError("Dual branch Query outputs changed relative to single-branch wrappers")

    # Compare online encoders with the two cache paths by sending each through
    # the exact same frozen Q-Former/projector. An online OOM is explicitly
    # reported rather than converted into a silent architecture change.
    if online_output is not None and pixel_features is not None and pixel_mask is not None:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            online_prepared = model.prepare_inputs_from_features(
                pixel_features,
                pixel_mask,
                byte_features,
                byte_mask,
                questions,
                answers,
            )
        online_cached = {
            "status": "compared",
            "pixel_encoder_features": feature_similarity_metrics(pixel_features, cached_pixel),
            "byte_encoder_features": feature_similarity_metrics(byte_features, cached_byte),
            "pixel_projected_queries": feature_similarity_metrics(
                online_prepared.projected_pixel_queries,
                cached_prepared.projected_pixel_queries,
            ),
            "byte_projected_queries": feature_similarity_metrics(
                online_prepared.projected_byte_queries,
                cached_prepared.projected_byte_queries,
            ),
            "yes_minus_no_absolute_differences": [
                abs(left - right)
                for left, right in zip(
                    score_rows(online_output, token_ids),
                    conditions["pixel_byte"]["yes_minus_no_scores"],
                )
            ],
        }
        for branch in ("pixel_projected_queries", "byte_projected_queries"):
            values = online_cached[branch]
            if (
                not values["all_finite"]
                or values["minimum_sample_cosine_similarity"] < 0.999
            ):
                raise RuntimeError(
                    f"Online/cache projected Query mismatch for {branch}: {values}"
                )
    else:
        online_cached = {"status": "not_compared_due_to_online_oom"}
    update_status(status_path, status, cached_mode_status="passed")

    changed_conditions = ("pixel_only", "byte_only", "zero_both", "shuffled_pixel", "shuffled_byte")
    if any(
        conditions[name]["logits_max_abs_difference_from_pixel_byte"] <= 0
        for name in changed_conditions
    ):
        raise RuntimeError("At least one dual ablation did not change Vicuna logits")
    if any(not row["logits_finite"] or not row["loss_finite"] for row in conditions.values()):
        raise FloatingPointError("A dual condition produced NaN or Inf")

    parameter_audit = compare_snapshot(model, before)
    report_path = args.output_dir / "DUAL_BRANCH_INTEGRATION.md"
    json_path = args.output_dir / "dual_branch_integration.json"
    report = {
        "schema_version": 1,
        "conclusion": (
            "passed" if online_output is not None else "passed_cached_mode_online_oom"
        ),
        "git": git_metadata(),
        "device": args.device,
        "gpu_initial": gpu_initial,
        "sample_ids": [record["sample_id"] for record in selected],
        "sample_splits": [record["split"] for record in selected],
        "test_split_read": False,
        "shared_marked_jpeg": {
            "passed": same_source,
            "jpeg_sha256": jpeg_hashes,
            "pixel_decoded_from_exact_byte_tensor": True,
            "boxes": {record["sample_id"]: item.boxes for record, item in zip(selected, rendered)},
        },
        "fixed_weights": {"pixel_weight": model.pixel_weight, "byte_weight": model.byte_weight},
        "loading": loading,
        "single_branch_consistency": {
            "pixel": pixel_single_similarity,
            "byte": byte_single_similarity,
        },
        "pixel_cache_audit": pixel_cache_audit,
        "byte_cache_policy": {
            "existing_pretraining_cache_reused": False,
            "reason": "final task checkpoint changes ByteFormer transformer.10, transformer.11, and post_transformer_norm",
            "selected_outputs_cached_in_memory": True,
            "stored_dtype": str(cached_byte_cpu.dtype),
        },
        "online_cached": online_cached,
        "shapes": tensor_shapes(cached_prepared),
        "answer_only": label_audit,
        "causal_shift_all_one": all(row["causal_shift_is_one"] for row in label_audit["rows"]),
        "shuffle": {
            "seed": args.seed,
            "mapping": permutation.cpu().tolist(),
            "fixed_point_free": bool(permutation.cpu().ne(torch.arange(args.sample_count)).all()),
        },
        "conditions": conditions,
        "parameter_immutability": parameter_audit,
        "gpu_memory_gib": {
            "model_loaded": loaded_memory,
            "online_dual_encoders": online_encoder_memory,
            "cached_encoder_features": cached_memory,
            "complete_loss_forward": full_loss_memory,
        },
        "json_path": str(json_path.resolve()),
        "report_path": str(report_path.resolve()),
    }
    write_json(json_path, report)
    render_report(report, report_path)
    return report


def write_failure(args, status, status_path, error: BaseException) -> None:
    failure = {
        "state": "failed",
        "phase": status.get("phase"),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "time": now_iso(),
    }
    write_json(args.output_dir / "failure.json", failure)
    atomic_text(
        args.output_dir / "failure_report.md",
        "# Dual-Branch Integration Failure\n\n"
        f"- Phase: `{failure['phase']}`\n"
        f"- Error: `{failure['error_type']}: {failure['error']}`\n"
        "- No training, optimizer, backward pass, or parameter update was performed.\n",
    )
    (args.output_dir / "FAILED").touch()
    update_status(
        status_path,
        status,
        state="failed",
        end_time=now_iso(),
        exit_code=1,
        stop_reason=str(error),
        online_mode_status=status.get("online_mode_status", "failed_or_not_reached"),
        cached_mode_status=status.get("cached_mode_status", "failed_or_not_reached"),
        report_path=str((args.output_dir / "failure_report.md").resolve()),
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(args.output_dir)
    git = git_metadata()
    status_path = args.output_dir / "status.json"
    status = {
        "state": "running",
        "phase": "startup",
        "pid": os.getpid(),
        "device": args.device,
        "start_time": now_iso(),
        "end_time": None,
        "exit_code": None,
        "stop_reason": None,
        "git_commit": git["commit"],
        "online_mode_status": "pending",
        "cached_mode_status": "pending",
        "report_path": str((args.output_dir / "DUAL_BRANCH_INTEGRATION.md").resolve()),
    }
    write_json(status_path, status)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)
    try:
        report = run(args, status, status_path)
        (args.output_dir / "DONE").touch()
        update_status(
            status_path,
            status,
            state="done",
            phase="complete",
            end_time=now_iso(),
            exit_code=0,
            stop_reason="verification_completed",
            online_mode_status=status.get("online_mode_status", "passed"),
            cached_mode_status=status.get("cached_mode_status", "passed"),
            report_path=report["report_path"],
        )
        return 0
    except BaseException as error:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            status["online_mode_status"] = "oom"
            torch.cuda.empty_cache()
        write_failure(args, status, status_path, error)
        return 1
    finally:
        if lock.exists() and lock.read_text().strip() == str(os.getpid()):
            lock.unlink()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
