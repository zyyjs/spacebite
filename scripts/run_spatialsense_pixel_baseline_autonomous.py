#!/usr/bin/env python3
"""Autonomous cached-EVA SpatialSense Pixel-only training and reporting."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from spacebyte.answer_only import audit_answer_only_labels
from spacebyte.data import (
    VISUAL_PREPROCESS_CONFIG,
    atomic_torch_save,
    flatten_spatialsense_annotations,
    render_record_to_pixel_values,
    sha256_file,
)
from spacebyte.evaluation import fixed_derangement
from spacebyte.models import (
    ImageQFormerBridge,
    PixelVicuna,
    build_official_eva_vit_g,
    build_xinstruct_qformer_tokenizer,
    initialize_image_bridge_from_instructblip,
    load_frozen_vicuna,
)
from spacebyte.pixel_baseline import (
    cache_metadata_matches,
    pixel_best_epoch,
    pixel_stopping_reason,
    visual_margin_is_stable,
)
from spacebyte.pixel_training import (
    PIXEL_CACHE_VERSION,
    PIXEL_FEATURE_THRESHOLDS,
    PermutedPixelCachedDataset,
    SpatialSensePixelCachedDataset,
    canonical_sha256,
    collate_pixel_features,
    evaluate_pixel_answer_only,
    feature_similarity_metrics,
    feature_similarity_passes,
    pixel_visual_cache_identity,
    require_frozen_eval,
    validate_discrete_cache_contract,
)
from spacebyte.stability_longrun import conclusion, paired_bootstrap


ROOT = Path("/home/Zhang_jinsong/spacebyte")
DATA_ROOT = Path("/data/students/Zhang_jinsong/spacebyte")
DEFAULT_OUTPUT = DATA_ROOT / "result/pixel/spatialsense_pixel_baseline"
DEFAULT_CACHE = DATA_ROOT / "cache/spatialsense_pixel_eva_g_fp32_forward"
BYTE_REFERENCE = {
    "accuracy": 0.5364,
    "macro_f1": 0.5065,
    "auroc": 0.5420,
    "zero_auroc": 0.4989,
    "shuffled_auroc": 0.5043,
}


class TerminationRequested(RuntimeError):
    pass


class CachedVisionSentinel(torch.nn.Module):
    """Prevents accidental online pixels after the official EVA cache is built."""

    num_features = 1408

    def forward(self, _pixels):
        raise RuntimeError("Training must consume the audited frozen-EVA cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT / "dataset/SpatialSense")
    parser.add_argument(
        "--source-train-manifest",
        type=Path,
        default=DATA_ROOT / "cache/spatialsense_full/manifests/train.jsonl",
    )
    parser.add_argument(
        "--source-valid-manifest",
        type=Path,
        default=DATA_ROOT / "cache/spatialsense_full/manifests/valid.jsonl",
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--cache-forward-precision",
        choices=("fp32", "official_mixed_fp16"),
        default="fp32",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--eva-checkpoint",
        type=Path,
        default=DATA_ROOT / "models/pixel_vicuna_official/eva_vit_g.pth",
    )
    parser.add_argument(
        "--instructblip-checkpoint",
        type=Path,
        default=DATA_ROOT / "models/pixel_vicuna_official/instruct_blip_vicuna7b_trimmed.pth",
    )
    parser.add_argument("--vicuna-path", type=Path, default=DATA_ROOT / "models/vicuna-7b-v1.1")
    parser.add_argument(
        "--qformer-tokenizer",
        type=Path,
        default=DATA_ROOT / "models/bert-base-uncased-tokenizer",
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--smoke-steps", type=int, default=20)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--cache-diagnostic-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_metadata() -> dict[str, str]:
    def run(*values: str) -> str:
        return subprocess.check_output(["git", *values], cwd=ROOT, text=True).strip()

    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "status": run("status", "--short"),
    }


def acquire_lock(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / "RUNNING.lock"
    if lock.exists():
        try:
            existing_pid = int(lock.read_text(encoding="utf-8").strip())
            os.kill(existing_pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            lock.unlink()
        else:
            raise RuntimeError(f"Pixel baseline is already running as PID {existing_pid}")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    atomic_write_text(output_dir / "run.pid", f"{os.getpid()}\n")
    return lock


def update_status(path: Path, status: dict[str, Any], **changes: Any) -> None:
    status.update(changes)
    write_json(path, status)


def validate_args(args: argparse.Namespace) -> None:
    if args.smoke_only and args.cache_diagnostic_only:
        raise ValueError("Smoke and cache-diagnostic modes are mutually exclusive")
    if args.gradient_accumulation != 8 or args.smoke_steps != 20:
        raise ValueError("The experiment fixes accumulation=8 and smoke optimizer steps=20")
    if args.max_epochs != 5 or args.learning_rate != 1e-5 or args.gradient_clipping != 1.0:
        raise ValueError("The fixed Pixel baseline hyperparameters were changed")
    if args.cache_forward_precision != "fp32":
        raise ValueError("Attempt 2 requires FP32 EVA cache forward with FP16 storage")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Pixel baseline requires FP16 CUDA execution")
    required = [
        args.dataset_root / "annotations.json",
        args.source_train_manifest,
        args.source_valid_manifest,
        args.eva_checkpoint,
        args.instructblip_checkpoint,
        args.vicuna_path / "config.json",
        args.qformer_tokenizer / "vocab.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required input paths are missing: {missing}")


def read_source_manifest(path: Path, expected_split: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(row["split"] != expected_split for row in rows):
        raise RuntimeError(f"Source manifest mixes splits: {path}")
    return rows


def audited_records(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for split, manifest_path in (
        ("train", args.source_train_manifest),
        ("valid", args.source_valid_manifest),
    ):
        manifest = read_source_manifest(manifest_path, split)
        raw = flatten_spatialsense_annotations(args.dataset_root, include_splits={split})
        raw_by_id = {row["sample_id"]: row for row in raw}
        if len(raw_by_id) != len(raw):
            raise RuntimeError(f"Raw {split} split has duplicate IDs")
        records = []
        for source in manifest:
            sample_id = source["sample_id"]
            if sample_id not in raw_by_id:
                raise RuntimeError(f"Source manifest ID missing in raw {split}: {sample_id}")
            record = raw_by_id[sample_id]
            for key in ("question", "answer", "relation", "relation_raw", "label"):
                if record[key] != source[key]:
                    raise RuntimeError(f"Raw/source {key} mismatch: {sample_id}")
            records.append(record)
        if len(records) != len(raw):
            raise RuntimeError(f"Raw/source {split} counts differ")
        result[split] = records
    if (len(result["train"]), len(result["valid"])) != (11238, 2638):
        raise RuntimeError("SpatialSense train/valid counts changed")
    if {row["sample_id"] for row in result["train"]} & {
        row["sample_id"] for row in result["valid"]
    }:
        raise RuntimeError("Train and valid sample IDs overlap")
    return result


def prepare_cache_rows(
    args: argparse.Namespace,
    records: dict[str, list[dict[str, Any]]],
    eva_sha256: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    image_hashes: dict[str, str] = {}
    identities: dict[str, dict[str, Any]] = {}
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    for split in ("train", "valid"):
        for record in records[split]:
            image_path = str((args.dataset_root / record["image_path"]).resolve())
            if image_path not in image_hashes:
                image_hashes[image_path] = sha256_file(Path(image_path))
            key, payload = pixel_visual_cache_identity(
                record,
                raw_image_sha256=image_hashes[image_path],
                eva_checkpoint_sha256=eva_sha256,
                cache_forward_precision=args.cache_forward_precision,
            )
            identities.setdefault(key, {"record": record, "payload": payload})
            rows[split].append(
                {
                    "sample_id": record["sample_id"],
                    "split": split,
                    "pixel_cache_key": key,
                    "feature_relative_path": f"features/{key[:2]}/{key}.pt",
                    "image_path": record["image_path"],
                    "question": record["question"],
                    "answer": record["answer"],
                    "relation": record["relation"],
                    "relation_raw": record["relation_raw"],
                    "label": bool(record["label"]),
                }
            )
    return rows, identities


def expected_cache_metadata(
    args: argparse.Namespace,
    rows: dict[str, list[dict[str, Any]]],
    identities: dict[str, dict[str, Any]],
    eva_sha256: str,
) -> dict[str, Any]:
    return {
        "cache_version": PIXEL_CACHE_VERSION,
        "eva_checkpoint_path": str(args.eva_checkpoint.resolve()),
        "eva_checkpoint_sha256": eva_sha256,
        "source_manifest_sha256": {
            "train": sha256_file(args.source_train_manifest),
            "valid": sha256_file(args.source_valid_manifest),
        },
        "sample_manifest_digest": {
            split: canonical_sha256(values) for split, values in rows.items()
        },
        "sample_count": {split: len(values) for split, values in rows.items()},
        "unique_feature_count": len(identities),
        "preprocessing": VISUAL_PREPROCESS_CONFIG,
        "dtype": "torch.float16",
        "cache_forward_precision": args.cache_forward_precision,
        "test_split_used": False,
    }


def manifest_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def cache_is_reusable(
    args: argparse.Namespace,
    expected: dict[str, Any],
    rows: dict[str, list[dict[str, Any]]],
    identities: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    metadata_path = args.cache_root / "cache_metadata.json"
    if not metadata_path.is_file():
        return False, {"reason": "metadata_missing"}
    observed = json.loads(metadata_path.read_text(encoding="utf-8"))
    matches, mismatches = cache_metadata_matches(observed, expected)
    if not matches:
        return False, {"reason": "metadata_mismatch", "mismatches": mismatches}
    for split in ("train", "valid"):
        path = args.cache_root / f"{split}.jsonl"
        if not path.is_file() or path.read_text(encoding="utf-8") != manifest_text(rows[split]):
            return False, {"reason": f"{split}_manifest_mismatch"}
    unique = {row["pixel_cache_key"]: row for values in rows.values() for row in values}
    feature_shape = tuple(observed.get("feature_shape", []))
    if len(feature_shape) != 2:
        return False, {"reason": "feature_shape_missing"}
    for key, row in unique.items():
        path = args.cache_root / row["feature_relative_path"]
        if not path.is_file():
            return False, {"reason": "feature_missing", "key": key}
        try:
            artifact = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            return False, {"reason": "feature_unreadable", "key": key, "error": str(error)}
        feature = artifact.get("vision_features")
        identity = identities[key]
        if (
            artifact.get("pixel_cache_key") != key
            or artifact.get("representative_sample_id")
            != identity["record"]["sample_id"]
            or artifact.get("identity_payload") != identity["payload"]
            or not torch.is_tensor(feature)
            or feature.dtype != torch.float16
            or tuple(feature.shape) != feature_shape
            or not bool(torch.isfinite(feature).all().item())
        ):
            return False, {"reason": "feature_invalid", "key": key}
    return True, {
        "reason": "exact_metadata_manifest_identity_and_feature_match",
        "strict_feature_artifacts_checked": len(unique),
    }


def _cache_artifact(args: argparse.Namespace, key: str) -> dict[str, Any]:
    return torch.load(
        args.cache_root / "features" / key[:2] / f"{key}.pt",
        map_location="cpu",
        weights_only=True,
    )


def _diagnostic_batch_indices(
    pending: list[tuple[str, dict[str, Any]]], batch_size: int
) -> list[int]:
    total = math.ceil(len(pending) / batch_size)
    first_valid = next(
        index for index, (_key, item) in enumerate(pending) if item["record"]["split"] == "valid"
    )
    requested = [
        0,
        1,
        total // 3,
        total // 2,
        max(0, first_valid // batch_size - 1),
        first_valid // batch_size,
        max(0, total - 2),
        total - 1,
    ]
    selected = []
    for index in requested:
        if 0 <= index < total and index not in selected:
            selected.append(index)
    if sum(len(pending[index * batch_size : (index + 1) * batch_size]) for index in selected) < 32:
        raise RuntimeError("Controlled cache diagnostic selected fewer than 32 samples")
    return selected


def _cache_precision_forward(
    visual_encoder: torch.nn.Module,
    pixels: torch.Tensor,
    precision: str,
) -> torch.Tensor:
    if precision == "fp32":
        with torch.autocast(device_type="cuda", enabled=False):
            return visual_encoder(pixels.float()).half()
    if precision == "official_mixed_fp16":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return visual_encoder(pixels).half()
    raise ValueError(f"Unsupported cache forward precision: {precision}")


@torch.inference_mode()
def controlled_cache_diagnostics(
    args: argparse.Namespace,
    visual_encoder: torch.nn.Module,
    identities: dict[str, dict[str, Any]],
    eva_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    """Replay cache batches, compare batch paths, FP32, and downstream bridge."""

    require_frozen_eval(visual_encoder)
    pending = list(identities.items())
    batch_indices = _diagnostic_batch_indices(pending, args.cache_batch_size)
    selected: list[dict[str, Any]] = []
    same_batch_rows = []
    same_cached = []
    same_replayed = []
    preprocessing_sha = canonical_sha256(
        {
            "marked_jpeg_preprocessing": VISUAL_PREPROCESS_CONFIG,
            "pixel_normalization": pending[0][1]["payload"]["pixel_normalization"],
        }
    )

    for batch_index in batch_indices:
        start = batch_index * args.cache_batch_size
        batch = pending[start : start + args.cache_batch_size]
        marked = [
            render_record_to_pixel_values(args.dataset_root, item["record"])
            for _key, item in batch
        ]
        pixels = torch.stack([value.pixel_values for value in marked]).to(device)
        replayed = _cache_precision_forward(
            visual_encoder, pixels, args.cache_forward_precision
        ).cpu()
        cached = torch.stack(
            [_cache_artifact(args, key)["vision_features"] for key, _item in batch]
        )
        metrics = feature_similarity_metrics(cached, replayed)
        same_batch_rows.append(
            {
                "batch_index": batch_index,
                "batch_size": len(batch),
                "metrics": metrics,
            }
        )
        same_cached.append(cached)
        same_replayed.append(replayed)
        for position, ((key, identity), marked_input) in enumerate(zip(batch, marked)):
            artifact = _cache_artifact(args, key)
            processed_sha = hashlib.sha256(
                marked_input.jpeg_bytes.to(torch.uint8).numpy().tobytes()
            ).hexdigest()
            observed = {
                "sample_id": artifact.get("representative_sample_id"),
                "split": identity["record"]["split"],
                "cache_key": artifact.get("pixel_cache_key"),
                "input_content_hash": artifact.get("processed_jpeg_sha256"),
                "checkpoint_sha256": artifact.get("identity_payload", {}).get(
                    "eva_checkpoint_sha256"
                ),
                "preprocessing_sha256": canonical_sha256(
                    {
                        "marked_jpeg_preprocessing": artifact.get("identity_payload", {}).get(
                            "marked_jpeg_preprocessing"
                        ),
                        "pixel_normalization": artifact.get("identity_payload", {}).get(
                            "pixel_normalization"
                        ),
                    }
                ),
                "shape": list(artifact["vision_features"].shape),
                "dtype": str(artifact["vision_features"].dtype),
                "attention_mask": f"all_ones_length_{artifact['vision_features'].shape[0]}",
            }
            expected = {
                "sample_id": identity["record"]["sample_id"],
                "split": identity["record"]["split"],
                "cache_key": key,
                "input_content_hash": processed_sha,
                "checkpoint_sha256": eva_sha256,
                "preprocessing_sha256": preprocessing_sha,
                "shape": list(cached[position].shape),
                "dtype": "torch.float16",
                "attention_mask": f"all_ones_length_{cached.shape[1]}",
            }
            validate_discrete_cache_contract(observed, expected)
            selected.append(
                {
                    "key": key,
                    "identity": identity,
                    "batch_index": batch_index,
                    "batch_position": position,
                    "batch_size": len(batch),
                    "cached": cached[position],
                    "replayed": replayed[position],
                    "pixels": marked_input.pixel_values,
                }
            )

    same_metrics = feature_similarity_metrics(
        torch.cat(same_cached), torch.cat(same_replayed)
    )
    if not same_metrics["torch_equal"]:
        raise RuntimeError(
            f"Same-batch cache-precision replay was not exact: {same_metrics}"
        )

    online_single = []
    for item in selected:
        pixels = item["pixels"].unsqueeze(0).to(device)
        online_single.append(
            _cache_precision_forward(
                visual_encoder, pixels, args.cache_forward_precision
            )[0].cpu()
        )
    cached_selected = torch.stack([item["cached"] for item in selected])
    online_selected = torch.stack(online_single)
    cross_batch_metrics = feature_similarity_metrics(cached_selected, online_selected)
    cross_batch_passed = feature_similarity_passes(cross_batch_metrics)

    failed_exact = [
        (index, item)
        for index, item in enumerate(selected)
        if not torch.equal(item["cached"], online_selected[index]) and item["batch_size"] == 8
    ]
    if len(failed_exact) < 6:
        failed_exact = [
            (index, item)
            for index, item in enumerate(selected)
            if item["batch_size"] == 8
        ]
    if len(failed_exact) < 6:
        raise RuntimeError("Fewer than six full-batch samples for FP32 diagnosis")
    fp32_targets = failed_exact[:6]
    if args.cache_forward_precision == "fp32":
        fp32_encoder = visual_encoder
        fp32_audit = {
            "shape_mismatch": {"tensors": 0, "parameters": 0},
            "official_mixed_fp16_conversion": False,
        }
    else:
        visual_encoder.cpu()
        torch.cuda.empty_cache()
        fp32_encoder, fp32_audit = build_official_eva_vit_g(
            args.eva_checkpoint, mixed_fp16=False
        )
        fp32_encoder.to(device).eval()
        require_frozen_eval(fp32_encoder)
    fp32_batch_outputs: dict[int, torch.Tensor] = {}
    for batch_index in sorted({item["batch_index"] for _index, item in fp32_targets}):
        batch = pending[
            batch_index * args.cache_batch_size : (batch_index + 1) * args.cache_batch_size
        ]
        pixels = torch.stack(
            [
                render_record_to_pixel_values(args.dataset_root, identity["record"]).pixel_values
                for _key, identity in batch
            ]
        ).to(device)
        with torch.autocast(device_type="cuda", enabled=False):
            fp32_batch_outputs[batch_index] = fp32_encoder(pixels.float()).cpu()
    fp32_batch = []
    fp32_single = []
    fp32_samples = []
    for _selected_index, item in fp32_targets:
        with torch.autocast(device_type="cuda", enabled=False):
            single = fp32_encoder(item["pixels"].unsqueeze(0).to(device).float())[0].cpu()
        batched = fp32_batch_outputs[item["batch_index"]][item["batch_position"]]
        fp32_batch.append(batched)
        fp32_single.append(single)
        fp32_samples.append(item["identity"]["record"]["sample_id"])
    fp32_metrics = feature_similarity_metrics(
        torch.stack(fp32_batch), torch.stack(fp32_single)
    )

    fp32_weight_autocast_batched = []
    for batch_index in batch_indices:
        batch = pending[
            batch_index * args.cache_batch_size : (batch_index + 1) * args.cache_batch_size
        ]
        pixels = torch.stack(
            [
                render_record_to_pixel_values(args.dataset_root, identity["record"]).pixel_values
                for _key, identity in batch
            ]
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            fp32_weight_autocast_batched.append(fp32_encoder(pixels).half().cpu())
    fp32_weight_autocast_single = []
    for item in selected:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            fp32_weight_autocast_single.append(
                fp32_encoder(item["pixels"].unsqueeze(0).to(device))[0].half().cpu()
            )
    fp32_weight_autocast_metrics = feature_similarity_metrics(
        torch.cat(fp32_weight_autocast_batched),
        torch.stack(fp32_weight_autocast_single),
    )

    fp32_encoder.cpu()
    torch.cuda.empty_cache()

    bridge = ImageQFormerBridge()
    initialize_image_bridge_from_instructblip(bridge, args.instructblip_checkpoint)
    bridge.requires_grad_(False).to(device).eval()
    tokenizer = build_xinstruct_qformer_tokenizer(args.qformer_tokenizer)
    projected_cached = []
    projected_online = []
    for start in range(0, len(selected), 4):
        group = selected[start : start + 4]
        cached = torch.stack([item["cached"] for item in group]).to(device)
        online = online_selected[start : start + len(group)].to(device)
        questions = [item["identity"]["record"]["question"] for item in group]
        tokens = tokenizer(
            questions,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        mask = torch.ones(cached.shape[:2], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            cached_output = bridge(
                cached,
                mask,
                tokens.input_ids.to(device),
                tokens.attention_mask.to(device),
            ).projected_features.half().cpu()
            online_output = bridge(
                online,
                mask,
                tokens.input_ids.to(device),
                tokens.attention_mask.to(device),
            ).projected_features.half().cpu()
        projected_cached.append(cached_output)
        projected_online.append(online_output)
    downstream_metrics = feature_similarity_metrics(
        torch.cat(projected_cached), torch.cat(projected_online)
    )
    downstream_passed = feature_similarity_passes(downstream_metrics)

    relation_types = sorted(
        {item["identity"]["record"]["relation_raw"] for item in selected}
    )
    report = {
        "passed": cross_batch_passed and downstream_passed,
        "cache_reused": None,
        "encoder_eval": not visual_encoder.training,
        "encoder_trainable_parameters": sum(
            parameter.numel()
            for parameter in visual_encoder.parameters()
            if parameter.requires_grad
        ),
        "execution_context": "torch.inference_mode",
        "cache_generation": {
            "unique_feature_count": len(pending),
            "batch_size": args.cache_batch_size,
            "batch_count": math.ceil(len(pending) / args.cache_batch_size),
            "last_batch_size": len(pending) % args.cache_batch_size or args.cache_batch_size,
            "order_sha256": canonical_sha256([key for key, _item in pending]),
            "forward_precision": args.cache_forward_precision,
            "storage_dtype": "torch.float16",
        },
        "strict_data_mapping": {
            "all_feature_artifacts_checked": len(pending),
            "diagnostic_jpeg_hashes_checked": len(selected),
            "selected_batch_indices": batch_indices,
            "selected_samples": len(selected),
            "relation_types": relation_types,
            "shape": list(cached_selected.shape[1:]),
            "dtype": str(cached_selected.dtype),
            "attention_mask": f"all_ones_length_{cached_selected.shape[1]}",
        },
        "same_batch_cache_precision_replay": {
            "passed": True,
            "aggregate": same_metrics,
            "batches": same_batch_rows,
        },
        "batch1_vs_batch8_cache_precision": {
            "passed": cross_batch_passed,
            "precision": args.cache_forward_precision,
            "thresholds": PIXEL_FEATURE_THRESHOLDS,
            "metrics": cross_batch_metrics,
        },
        "batch1_vs_batch8_fp32": {
            "diagnostic_only": True,
            "checkpoint_load": {
                "shape_mismatch": fp32_audit["shape_mismatch"],
                "official_mixed_fp16_conversion": fp32_audit[
                    "official_mixed_fp16_conversion"
                ],
            },
            "sample_ids": fp32_samples,
            "metrics": fp32_metrics,
        },
        "batch1_vs_batch8_fp16_autocast_with_fp32_weights": {
            "diagnostic_only": True,
            "passed": feature_similarity_passes(fp32_weight_autocast_metrics),
            "thresholds": PIXEL_FEATURE_THRESHOLDS,
            "metrics": fp32_weight_autocast_metrics,
        },
        "image_bridge_downstream": {
            "passed": downstream_passed,
            "metrics": downstream_metrics,
        },
    }
    del bridge, fp32_encoder, fp32_batch_outputs
    gc.collect()
    torch.cuda.empty_cache()
    return report


@torch.no_grad()
def build_or_validate_cache(
    args: argparse.Namespace,
    visual_encoder: torch.nn.Module,
    device: torch.device,
    log,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    records = audited_records(args)
    eva_sha256 = sha256_file(args.eva_checkpoint)
    rows, identities = prepare_cache_rows(args, records, eva_sha256)
    expected = expected_cache_metadata(args, rows, identities, eva_sha256)
    reusable, reuse_audit = cache_is_reusable(args, expected, rows, identities)
    feature_shape = None
    if reusable:
        feature_shape = json.loads(
            (args.cache_root / "cache_metadata.json").read_text(encoding="utf-8")
        )["feature_shape"]
        print("pixel cache: exact reusable cache verified", flush=True)
    elif (args.cache_root / "cache_metadata.json").exists():
        raise RuntimeError(
            f"Existing Pixel cache failed strict data/mapping integrity; refusing rebuild: "
            f"{reuse_audit}"
        )
    else:
        print(f"pixel cache: rebuilding ({reuse_audit})", flush=True)
        args.cache_root.mkdir(parents=True, exist_ok=True)
        pending = list(identities.items())
        for start in range(0, len(pending), args.cache_batch_size):
            batch = pending[start : start + args.cache_batch_size]
            marked = [
                render_record_to_pixel_values(args.dataset_root, item["record"])
                for _key, item in batch
            ]
            pixels = torch.stack([item.pixel_values for item in marked]).to(device)
            features = _cache_precision_forward(
                visual_encoder, pixels, args.cache_forward_precision
            ).cpu()
            if not bool(torch.isfinite(features).all().item()):
                raise FloatingPointError("Non-finite EVA cache feature")
            if feature_shape is None:
                feature_shape = list(features.shape[1:])
            elif list(features.shape[1:]) != feature_shape:
                raise RuntimeError("EVA output shape changed during cache construction")
            for position, ((key, identity), marked_input) in enumerate(zip(batch, marked)):
                relative = Path("features") / key[:2] / f"{key}.pt"
                artifact = {
                    "cache_version": PIXEL_CACHE_VERSION,
                    "pixel_cache_key": key,
                    "vision_features": features[position].contiguous(),
                    "feature_shape": feature_shape,
                    "dtype": "torch.float16",
                    "representative_sample_id": identity["record"]["sample_id"],
                    "identity_payload": identity["payload"],
                    "generation_batch_index": start // args.cache_batch_size,
                    "generation_batch_position": position,
                    "generation_batch_size": len(batch),
                    "processed_jpeg_sha256": hashlib.sha256(
                        marked_input.jpeg_bytes.to(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                }
                atomic_torch_save(artifact, args.cache_root / relative)
            if start == 0 or (start // args.cache_batch_size + 1) % 100 == 0:
                print(
                    f"pixel cache unique={min(start + len(batch), len(pending))}/{len(pending)}",
                    flush=True,
                )
        for split in ("train", "valid"):
            atomic_write_text(args.cache_root / f"{split}.jsonl", manifest_text(rows[split]))
        metadata = {
            **expected,
            "feature_shape": feature_shape,
            "created_at": now_iso(),
        }
        write_json(args.cache_root / "cache_metadata.json", metadata)
    audit = controlled_cache_diagnostics(
        args, visual_encoder, identities, eva_sha256, device
    )
    audit["cache_reused"] = reusable
    metadata_path = args.cache_root / "cache_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["generation"] = audit["cache_generation"]
    metadata["validation"] = {
        "validated_at": now_iso(),
        "method": "strict_discrete_same_batch_replay_and_controlled_numeric_similarity",
        "report_path": str((args.output_dir / "cache_diagnostic.json").resolve()),
        "thresholds": PIXEL_FEATURE_THRESHOLDS,
    }
    write_json(metadata_path, metadata)
    write_json(args.output_dir / "cache_diagnostic.json", audit)
    if not audit["passed"]:
        raise RuntimeError(
            "Controlled Pixel cache diagnostics did not meet the predeclared "
            "floating-point or downstream thresholds; training is blocked"
        )
    report = {
        "reused": reusable,
        "reuse_audit": reuse_audit,
        "metadata": {**expected, "feature_shape": feature_shape},
        "controlled_diagnostics": audit,
        "manifest_paths": {
            split: str((args.cache_root / f"{split}.jsonl").resolve())
            for split in ("train", "valid")
        },
    }
    write_json(args.output_dir / "cache_report.json", report)
    return report, rows


def build_training_model(args: argparse.Namespace, device: torch.device):
    bridge = ImageQFormerBridge()
    initialization = initialize_image_bridge_from_instructblip(
        bridge, args.instructblip_checkpoint
    )
    bridge.to(device)
    vicuna, llm_tokenizer, vicuna_report = load_frozen_vicuna(
        args.vicuna_path, device=device, dtype=torch.float16
    )
    qformer_tokenizer = build_xinstruct_qformer_tokenizer(args.qformer_tokenizer)
    model = PixelVicuna(
        visual_encoder=CachedVisionSentinel(),
        bridge=bridge,
        vicuna=vicuna,
        llm_tokenizer=llm_tokenizer,
        qformer_tokenizer=qformer_tokenizer,
    ).to(device)
    return model, {"image_bridge": initialization, "vicuna": vicuna_report}


def create_optimizer(model: PixelVicuna, args: argparse.Namespace):
    trainable = [parameter for parameter in model.bridge.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.0, growth_interval=2000)
    return optimizer, scheduler, scaler


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    torch.cuda.set_rng_state_all(value["torch_cuda"])


def checkpoint_artifact(
    model: PixelVicuna,
    optimizer,
    scheduler,
    scaler,
    *,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    epoch_completed: int,
    global_step: int,
    best_epoch: int | None,
    best_auroc: float | None,
    epoch_in_progress: int | None = None,
    epoch_order: list[int] | None = None,
    next_micro_index: int = 0,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "bridge_state_dict": {
            name: value.detach().cpu() for name, value in model.bridge.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "amp_scaler_state_dict": scaler.state_dict(),
        "epoch": epoch_completed,
        "epoch_in_progress": epoch_in_progress,
        "epoch_order": epoch_order,
        "next_micro_index": next_micro_index,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_metric": best_auroc,
        "history": history,
        "rng_state": rng_state(),
        "training_config": fixed_training_config(args),
    }


def fixed_training_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clipping": args.gradient_clipping,
        "micro_batch_size": 1,
        "gradient_accumulation": args.gradient_accumulation,
        "precision": "FP16 autocast with GradScaler",
        "eva_cache_forward_precision": args.cache_forward_precision,
        "eva_cache_storage_dtype": "torch.float16",
        "grad_scaler_init_scale": 1024.0,
        "grad_scaler_growth_interval": 2000,
        "max_epochs": args.max_epochs,
        "seed": args.seed,
        "answer_only": True,
        "train_split": "train",
        "selection_split": "valid",
        "test_split_used": False,
        "eva_frozen": True,
        "vicuna_frozen": True,
        "trainable": [
            "Image Q-Former",
            "Image Query Tokens",
            "visual LayerNorm",
            "image-to-Vicuna projector",
        ],
        "scheduler": "constant LambdaLR",
    }


def load_resume(path: Path, model, optimizer, scheduler, scaler, args) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("training_config") != fixed_training_config(args):
        raise RuntimeError("Resume checkpoint training configuration differs")
    incompatible = model.bridge.load_state_dict(artifact["bridge_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict bridge resume failed")
    optimizer.load_state_dict(artifact["optimizer_state_dict"])
    scheduler.load_state_dict(artifact["scheduler_state_dict"])
    scaler.load_state_dict(artifact["amp_scaler_state_dict"])
    restore_rng(artifact["rng_state"])
    return artifact


def trainable_gradient_report(model: PixelVicuna) -> dict[str, float]:
    named = dict(model.bridge.named_parameters())

    def norm(name: str) -> float:
        gradient = named[name].grad
        return 0.0 if gradient is None else float(gradient.float().norm().item())

    return {
        "query_tokens": norm("query_tokens"),
        "vision_layernorm": max(
            norm(name) for name in named if name.startswith("vision_layernorm.")
        ),
        "self_attention": max(
            norm(name)
            for name in named
            if name.endswith("attention.self.query.weight") and "crossattention" not in name
        ),
        "cross_attention_kv": max(
            norm(name)
            for name in named
            if "crossattention.self" in name
            and name.endswith(("key.weight", "value.weight"))
        ),
        "projector": max(
            norm(name) for name in named if name.startswith("llm_projector.")
        ),
    }


def parameter_probes(module: torch.nn.Module) -> dict[str, tuple[float, float, int]]:
    result = {}
    for name, parameter in module.named_parameters():
        flat = parameter.detach().reshape(-1)
        result[name] = (
            float(flat[0].float().cpu()),
            float(flat[-1].float().cpu()),
            int(parameter._version),
        )
    return result


def run_smoke(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    set_seed(args.seed)
    records = audited_records(args)["train"][: args.smoke_steps * args.gradient_accumulation]
    eva, eva_audit = build_official_eva_vit_g(args.eva_checkpoint)
    eva.to(device).eval()
    eva_before = parameter_probes(eva)
    cached = []
    with torch.no_grad():
        for start in range(0, len(records), args.cache_batch_size):
            group = records[start : start + args.cache_batch_size]
            marked = [render_record_to_pixel_values(args.dataset_root, row) for row in group]
            pixels = torch.stack([item.pixel_values for item in marked]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = eva(pixels).half().cpu()
            for row, feature in zip(group, features):
                cached.append({"record": row, "feature": feature})

    with tempfile.TemporaryDirectory(dir=args.output_dir) as temporary:
        cache_probe = Path(temporary) / "smoke_cache_probe.pt"
        atomic_torch_save(
            {
                "sample_id": cached[0]["record"]["sample_id"],
                "vision_features": cached[0]["feature"],
            },
            cache_probe,
        )
        restored_probe = torch.load(cache_probe, map_location="cpu", weights_only=True)
        if (
            restored_probe["sample_id"] != cached[0]["record"]["sample_id"]
            or not torch.equal(restored_probe["vision_features"], cached[0]["feature"])
        ):
            raise RuntimeError("Smoke cache sample mapping/save-load check failed")

    model, load_report = build_training_model(args, device)
    vicuna_before = parameter_probes(model.vicuna)
    optimizer, scheduler, scaler = create_optimizer(model, args)
    bridge_before = {
        name: value.detach().cpu().clone() for name, value in model.bridge.named_parameters()
    }
    losses = []
    gradient_samples = []
    model.train()
    model.vicuna.eval()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.smoke_steps):
        optimizer.zero_grad(set_to_none=True)
        group = cached[
            step * args.gradient_accumulation : (step + 1) * args.gradient_accumulation
        ]
        for item in group:
            feature = item["feature"].unsqueeze(0).to(device)
            mask = torch.ones(feature.shape[:2], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.forward_from_vision_features(
                    feature,
                    mask,
                    item["record"]["question"],
                    item["record"]["answer"],
                    answer_only_loss=True,
                )
                loss = output.loss / len(group)
            if not bool(torch.isfinite(output.loss).item()):
                raise FloatingPointError("Smoke loss is non-finite")
            scaler.scale(loss).backward()
            losses.append(float(output.loss.detach().float().item()))
        scaler.unscale_(optimizer)
        gradients = trainable_gradient_report(model)
        total_norm = float(
            torch.nn.utils.clip_grad_norm_(model.bridge.parameters(), args.gradient_clipping)
        )
        if not math.isfinite(total_norm) or not all(
            math.isfinite(value) and value > 0 for value in gradients.values()
        ):
            raise FloatingPointError(f"Smoke gradients invalid: {gradients}, total={total_norm}")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        gradient_samples.append(gradients)

    changed = [
        name
        for name, parameter in model.bridge.named_parameters()
        if not torch.equal(bridge_before[name], parameter.detach().cpu())
    ]
    if not changed or max(losses) - min(losses) <= 1e-6:
        raise RuntimeError("Smoke did not change trainable parameters or loss")
    if eva_before != parameter_probes(eva) or vicuna_before != parameter_probes(model.vicuna):
        raise RuntimeError("A frozen smoke-test module changed")
    if any(parameter.grad is not None for parameter in eva.parameters()):
        raise RuntimeError("EVA unexpectedly received smoke gradients")
    if any(parameter.grad is not None for parameter in model.vicuna.parameters()):
        raise RuntimeError("Vicuna unexpectedly received smoke gradients")

    sample = cached[0]
    feature = sample["feature"].unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        audit_output = model.forward_from_vision_features(
            feature,
            torch.ones(feature.shape[:2], dtype=torch.long, device=device),
            sample["record"]["question"],
            sample["record"]["answer"],
            answer_only_loss=True,
        )
    label_audit = audit_answer_only_labels(
        audit_output.prepared, [sample["record"]["answer"]], model.llm_tokenizer
    )
    if not label_audit["exactly_one_supervised_token_per_sample"]:
        raise RuntimeError("Smoke answer-only label audit failed")

    artifact = checkpoint_artifact(
        model,
        optimizer,
        scheduler,
        scaler,
        args=args,
        history=[],
        epoch_completed=0,
        global_step=args.smoke_steps,
        best_epoch=None,
        best_auroc=None,
    )
    temporary = args.output_dir / ".smoke_resume.pt"
    atomic_torch_save(artifact, temporary)
    expected_state = {
        name: value.clone() for name, value in artifact["bridge_state_dict"].items()
    }
    with torch.no_grad():
        next(model.bridge.parameters()).add_(1)
    restored = load_resume(temporary, model, optimizer, scheduler, scaler, args)
    temporary.unlink()
    if restored["global_step"] != args.smoke_steps or not all(
        torch.equal(value, model.bridge.state_dict()[name].cpu())
        for name, value in expected_state.items()
    ):
        raise RuntimeError("Smoke checkpoint did not restore model/global step exactly")
    optimizer_state_present = bool(optimizer.state_dict()["state"])
    if not optimizer_state_present:
        raise RuntimeError("Smoke optimizer state was not restored")
    result = {
        "passed": True,
        "optimizer_steps": args.smoke_steps,
        "micro_steps": len(cached),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_range": max(losses) - min(losses),
        "trainable_tensors_updated": len(changed),
        "gradient_norms_last": gradient_samples[-1],
        "eva_no_grad_and_unchanged": True,
        "vicuna_no_grad_and_unchanged": True,
        "checkpoint_model_optimizer_global_step_restored": True,
        "cache_mapping_checked": True,
        "label_audit": label_audit,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "eva_load_summary": {
            key: eva_audit[key]
            for key in ("loaded", "checkpoint_only", "model_only", "shape_mismatch")
        },
        "image_bridge_load_summary": {
            key: load_report["image_bridge"][key]
            for key in ("loaded", "checkpoint_only", "model_only", "shape_mismatch")
        },
        "timestamp": now_iso(),
        "git": git_metadata(),
    }
    write_json(args.output_dir / "smoke_20_steps.json", result)
    del model, optimizer, scheduler, scaler, eva, cached, artifact, expected_state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def compact_evaluation(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": value["sample_count"],
        "duration_seconds": value["duration_seconds"],
        **value["logit_yes_minus_no"],
    }


def evaluate_epoch(model, valid, shuffled_datasets, args, device) -> dict[str, Any]:
    normal_raw = evaluate_pixel_answer_only(
        model,
        valid,
        device=device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    zero_raw = evaluate_pixel_answer_only(
        model,
        valid,
        device=device,
        batch_size=args.eval_batch_size,
        pixel_condition="zero",
        num_workers=args.num_workers,
    )
    shuffled_raw = [
        evaluate_pixel_answer_only(
            model,
            dataset,
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )
        for dataset in shuffled_datasets
    ]
    shuffled = [compact_evaluation(value) for value in shuffled_raw]
    aurocs = [value["auroc"] for value in shuffled]
    normal = compact_evaluation(normal_raw)
    zero = compact_evaluation(zero_raw)
    return {
        "normal": normal,
        "zero": zero,
        "shuffled": shuffled,
        "shuffled_auroc_mean": float(np.mean(aurocs)),
        "shuffled_auroc_std": float(np.std(aurocs, ddof=1)),
        "normal_minus_zero_auroc": normal["auroc"] - zero["auroc"],
        "normal_minus_shuffled_mean_auroc": normal["auroc"] - float(np.mean(aurocs)),
    }


def train_epoch(
    model,
    dataset,
    optimizer,
    scheduler,
    scaler,
    args,
    device,
    *,
    epoch: int,
    order: list[int],
    start_micro_index: int,
    global_step: int,
    curve_handle,
    termination_state: dict[str, bool],
    termination_callback,
) -> tuple[dict[str, Any], int]:
    model.train()
    # Freezing weights and keeping the frozen language model in inference mode
    # are separate requirements; only the image bridge should enter train mode.
    model.vicuna.eval()
    losses = []
    started = time.perf_counter()
    optimizer_steps = 0
    first_label_audit = None
    for group_start in range(start_micro_index, len(order), args.gradient_accumulation):
        group_indices = order[group_start : group_start + args.gradient_accumulation]
        optimizer.zero_grad(set_to_none=True)
        group_losses = []
        for index in group_indices:
            batch = collate_pixel_features([dataset[index]])
            features = batch["vision_features"].to(device, non_blocking=True)
            mask = batch["vision_attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.forward_from_vision_features(
                    features,
                    mask,
                    batch["questions"],
                    batch["answers"],
                    answer_only_loss=True,
                )
                loss = output.loss / len(group_indices)
            if not bool(torch.isfinite(output.loss).item()):
                raise FloatingPointError("Non-finite Pixel answer-only loss")
            if first_label_audit is None:
                first_label_audit = audit_answer_only_labels(
                    output.prepared, batch["answers"], model.llm_tokenizer
                )
            scaler.scale(loss).backward()
            group_losses.append(float(output.loss.detach().float().item()))
        scaler.unscale_(optimizer)
        gradients = trainable_gradient_report(model)
        total_norm = float(
            torch.nn.utils.clip_grad_norm_(model.bridge.parameters(), args.gradient_clipping)
        )
        if not math.isfinite(total_norm) or not all(
            math.isfinite(value) and value > 0 for value in gradients.values()
        ):
            raise FloatingPointError(f"Invalid Pixel gradients: {gradients}, total={total_norm}")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1
        optimizer_steps += 1
        losses.extend(group_losses)
        curve_handle.write(
            json.dumps(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "next_micro_index": group_start + len(group_indices),
                    "mean_loss": float(np.mean(group_losses)),
                    "gradient_norms": gradients,
                    "total_gradient_norm_before_clipping": total_norm,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                }
            )
            + "\n"
        )
        curve_handle.flush()
        if termination_state["requested"]:
            termination_callback(group_start + len(group_indices), global_step)
            raise TerminationRequested("Termination signal received; safe checkpoint saved")
    return {
        "mean_loss": float(np.mean(losses)),
        "min_loss": min(losses),
        "max_loss": max(losses),
        "micro_steps": len(losses),
        "optimizer_steps": optimizer_steps,
        "duration_seconds": time.perf_counter() - started,
        "label_audit": first_label_audit,
    }, global_step


def final_statistics(model, valid, shuffled_datasets, shuffle_seeds, args, device):
    normal = evaluate_pixel_answer_only(
        model, valid, device=device, batch_size=args.eval_batch_size, num_workers=args.num_workers
    )
    zero = evaluate_pixel_answer_only(
        model,
        valid,
        device=device,
        batch_size=args.eval_batch_size,
        pixel_condition="zero",
        num_workers=args.num_workers,
    )
    shuffled = [
        evaluate_pixel_answer_only(
            model,
            dataset,
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )
        for dataset in shuffled_datasets
    ]
    identifiers = [row["sample_id"] for row in normal["outputs"]]
    if identifiers != [row["sample_id"] for row in zero["outputs"]] or any(
        identifiers != [row["sample_id"] for row in value["outputs"]]
        for value in shuffled
    ):
        raise RuntimeError("Final paired evaluation sample order differs")
    labels = [bool(row["label"]) for row in normal["outputs"]]
    normal_scores = [row["score_logit_yes_minus_no"] for row in normal["outputs"]]
    zero_scores = [row["score_logit_yes_minus_no"] for row in zero["outputs"]]
    shuffled_scores = [
        [row["score_logit_yes_minus_no"] for row in value["outputs"]]
        for value in shuffled
    ]
    bootstrap = paired_bootstrap(
        labels,
        normal_scores,
        zero_scores,
        shuffled_scores,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    shuffled_metrics = [compact_evaluation(value) for value in shuffled]
    shuffled_aurocs = [value["auroc"] for value in shuffled_metrics]
    normal_metrics = compact_evaluation(normal)
    zero_metrics = compact_evaluation(zero)
    mean_shuffled = float(np.mean(shuffled_aurocs))
    final_conclusion = conclusion(
        normal_metrics["auroc"], zero_metrics["auroc"], mean_shuffled, bootstrap
    )
    score_path = args.output_dir / "final_sample_scores.jsonl"
    with score_path.open("w", encoding="utf-8") as handle:
        for index, sample_id in enumerate(identifiers):
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "label": labels[index],
                        "normal_score": normal_scores[index],
                        "zero_score": zero_scores[index],
                        "shuffled_scores": {
                            str(seed): shuffled_scores[position][index]
                            for position, seed in enumerate(shuffle_seeds)
                        },
                    }
                )
                + "\n"
            )
    result = {
        "normal": normal_metrics,
        "zero": zero_metrics,
        "shuffled": [
            {"seed": seed, **metrics}
            for seed, metrics in zip(shuffle_seeds, shuffled_metrics)
        ],
        "shuffled_auroc_mean": mean_shuffled,
        "shuffled_auroc_std": float(np.std(shuffled_aurocs, ddof=1)),
        "normal_minus_zero_auroc": normal_metrics["auroc"] - zero_metrics["auroc"],
        "normal_minus_shuffled_mean_auroc": normal_metrics["auroc"] - mean_shuffled,
        "bootstrap": bootstrap,
        "conclusion": final_conclusion,
        "sample_scores_path": str(score_path.resolve()),
    }
    write_json(args.output_dir / "bootstrap.json", bootstrap)
    write_json(args.output_dir / "final_metrics.json", result)
    return result


def render_report(report: dict[str, Any], path: Path) -> None:
    rows = []
    for row in report["epochs"]:
        rows.append(
            f"| {row['epoch']} | {row['normal']['accuracy']:.4f} | "
            f"{row['normal']['macro_f1']:.4f} | {row['normal']['auroc']:.4f} | "
            f"{row['zero']['auroc']:.4f} | {row['shuffled_auroc_mean']:.4f} ± "
            f"{row['shuffled_auroc_std']:.4f} | {row['normal_minus_zero_auroc']:.4f} | "
            f"{row['normal_minus_shuffled_mean_auroc']:.4f} |"
        )
    final = report["final"]
    bootstrap = final["bootstrap"]
    relation_rows = []
    for relation, metrics in final["normal"]["per_relation"].items():
        relation_rows.append(
            f"| {relation} | {metrics['sample_count']} | {metrics['accuracy']:.4f} | "
            f"{metrics['auroc']:.4f} |"
        )
    text = f"""# SpatialSense Pixel-only Baseline

## Experiment

Frozen EVA-CLIP-G cached features → trainable official InstructBLIP Image Q-Former/Query Tokens/visual LayerNorm/projector → frozen Vicuna-7B-v1.1. SpatialSense test was not read.

## Attempt history

Attempt 1 stopped before training because the original audit incorrectly required
bitwise equality between FP16 EVA outputs produced with cache batch size 8 and
online batch size 1. Its evidence is preserved under
`attempt_1_strict_cache_check_failure/`. Attempt 2 keeps strict equality for all
discrete mapping and input identity fields, requires exact same-batch replay, and
uses the predeclared mean-error, relative-L2, and per-sample cosine thresholds
only for controlled cross-batch floating-point comparison. Controlled diagnosis
also found that the official mixed-FP16 path missed the fixed per-sample cosine
threshold, while FP32 batch-size replay was stable; therefore the cache was
regenerated with FP32 EVA forward passes and stored as FP16.

## Stop

Epoch **{report['stop_epoch']}**, reason `{report['stop_reason']}`; best epoch **{report['best_epoch']}**.

## Epoch validation

| Epoch | Normal Acc | Normal Macro-F1 | Normal AUROC | Zero AUROC | 5× Shuffled AUROC | N−Z | N−S |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Final best-checkpoint metrics

- Normal: Accuracy {final['normal']['accuracy']:.4f}, Macro-F1 {final['normal']['macro_f1']:.4f}, AUROC {final['normal']['auroc']:.4f}
- Zero-Pixel AUROC: {final['zero']['auroc']:.4f}
- 5× Shuffled-Pixel AUROC: {final['shuffled_auroc_mean']:.4f} ± {final['shuffled_auroc_std']:.4f}

| Relation | Samples | Accuracy | AUROC |
|---|---:|---:|---:|
{chr(10).join(relation_rows)}

## Bootstrap 95% confidence intervals

- Normal AUROC: [{bootstrap['normal_auroc']['lower_95']:.4f}, {bootstrap['normal_auroc']['upper_95']:.4f}]
- Normal−Zero AUROC: [{bootstrap['normal_minus_zero_auroc']['lower_95']:.4f}, {bootstrap['normal_minus_zero_auroc']['upper_95']:.4f}]
- Normal−mean Shuffled AUROC: [{bootstrap['normal_minus_mean_shuffled_auroc']['lower_95']:.4f}, {bootstrap['normal_minus_mean_shuffled_auroc']['upper_95']:.4f}]

Conclusion: `{final['conclusion']}`.

## Byte-only reference (not rerun)

| Branch | Accuracy | Macro-F1 | Normal AUROC | Zero AUROC | Shuffled AUROC |
|---|---:|---:|---:|---:|---:|
| Pixel-only | {final['normal']['accuracy']:.4f} | {final['normal']['macro_f1']:.4f} | {final['normal']['auroc']:.4f} | {final['zero']['auroc']:.4f} | {final['shuffled_auroc_mean']:.4f} |
| Confirmed Byte-only | 0.5364 | 0.5065 | 0.5420 | 0.4989 | 0.5043 |

## Runtime and artifacts

- Peak GPU memory: {report['peak_gpu_memory_bytes'] / 2**30:.3f} GiB
- Total duration: {report['total_duration_seconds'] / 3600:.3f} hours
- Cache: `{report['cache_root']}`
- Best: `{report['best_checkpoint']}`
- Last: `{report['last_checkpoint']}`
- Log: `{report['train_log']}`
- Code: `{report['git']['branch']}` at `{report['git']['commit']}`
"""
    atomic_write_text(path, text)


def run_autonomous(
    args: argparse.Namespace,
    device: torch.device,
    status: dict[str, Any],
    status_path: Path,
    termination_state: dict[str, bool],
) -> None:
    started = time.perf_counter()
    smoke_path = args.output_dir / "smoke_20_steps.json"
    if not smoke_path.is_file() or not json.loads(smoke_path.read_text())["passed"]:
        raise RuntimeError("A passing 20-step smoke report is required before long-run launch")
    if not args.resume and any(
        (args.output_dir / name).exists() for name in ("best.pt", "last.pt", "DONE")
    ):
        raise RuntimeError("Existing formal artifacts require --resume; refusing to overwrite")
    update_status(status_path, status, state="loading_eva_for_cache")
    eva, eva_audit = build_official_eva_vit_g(
        args.eva_checkpoint,
        mixed_fp16=args.cache_forward_precision == "official_mixed_fp16",
    )
    eva.to(device).eval()
    update_status(status_path, status, state="cache_check_or_build")
    cache_report, _rows = build_or_validate_cache(args, eva, device, sys.stdout)
    del eva
    gc.collect()
    torch.cuda.empty_cache()

    train_dataset = SpatialSensePixelCachedDataset(
        args.cache_root / "train.jsonl", args.cache_root, expected_split="train"
    )
    valid_dataset = SpatialSensePixelCachedDataset(
        args.cache_root / "valid.jsonl", args.cache_root, expected_split="valid"
    )
    shuffle_seeds = [args.seed + offset for offset in range(5)]
    valid_keys = [row["pixel_cache_key"] for row in valid_dataset.records]
    mappings = [
        fixed_derangement(len(valid_dataset), seed, valid_keys) for seed in shuffle_seeds
    ]
    shuffled_datasets = [
        PermutedPixelCachedDataset(valid_dataset, mapping) for mapping in mappings
    ]
    write_json(
        args.output_dir / "shuffled_mappings.json",
        {
            "full_validation_set": True,
            "sample_count": len(valid_dataset),
            "self_mappings": [0] * len(mappings),
            "seeds": shuffle_seeds,
            "mappings": {str(seed): mapping for seed, mapping in zip(shuffle_seeds, mappings)},
        },
    )

    update_status(status_path, status, state="loading_trainable_model")
    model, model_load = build_training_model(args, device)
    optimizer, scheduler, scaler = create_optimizer(model, args)
    history: list[dict[str, Any]] = []
    global_step = 0
    epoch_completed = 0
    resume_epoch = None
    resume_order = None
    resume_micro_index = 0
    best_epoch_value = None
    best_auroc = None
    last_path = args.output_dir / "last.pt"
    best_path = args.output_dir / "best.pt"
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError("--resume requested but last.pt is absent")
        artifact = load_resume(last_path, model, optimizer, scheduler, scaler, args)
        history = artifact["history"]
        global_step = int(artifact["global_step"])
        epoch_completed = int(artifact["epoch"])
        resume_epoch = artifact.get("epoch_in_progress")
        resume_order = artifact.get("epoch_order")
        resume_micro_index = int(artifact.get("next_micro_index", 0))
        best_epoch_value = artifact.get("best_epoch")
        best_auroc = artifact.get("best_metric")
        del artifact
    else:
        initial = checkpoint_artifact(
            model,
            optimizer,
            scheduler,
            scaler,
            args=args,
            history=history,
            epoch_completed=0,
            global_step=0,
            best_epoch=None,
            best_auroc=None,
        )
        atomic_torch_save(initial, last_path)
        del initial

    config = {
        **fixed_training_config(args),
        "cache": cache_report,
        "model_load": {
            "image_bridge": {
                key: model_load["image_bridge"][key]
                for key in ("loaded", "checkpoint_only", "model_only", "shape_mismatch")
            },
            "eva": {
                key: eva_audit[key]
                for key in ("loaded", "checkpoint_only", "model_only", "shape_mismatch")
            },
            "vicuna": model_load["vicuna"],
        },
        "weights": {
            "eva_sha256": sha256_file(args.eva_checkpoint),
            "instructblip_sha256": sha256_file(args.instructblip_checkpoint),
        },
        "git": git_metadata(),
    }
    write_json(args.output_dir / "training_config.json", config)
    curve_path = args.output_dir / "training_curve.jsonl"
    curve_mode = "a" if args.resume and curve_path.exists() else "w"
    stop_reason = None
    torch.cuda.reset_peak_memory_stats(device)
    update_status(
        status_path,
        status,
        state="training",
        current_epoch=resume_epoch or epoch_completed + 1,
        global_step=global_step,
        best_epoch=best_epoch_value,
        best_auroc=best_auroc,
    )
    with curve_path.open(curve_mode, encoding="utf-8") as curve_handle:
        epoch = resume_epoch or epoch_completed + 1
        while epoch <= args.max_epochs:
            if resume_epoch == epoch:
                order = resume_order
                micro_index = resume_micro_index
                if not isinstance(order, list) or sorted(order) != list(range(len(train_dataset))):
                    raise RuntimeError("Resume epoch order is invalid")
            else:
                generator = torch.Generator().manual_seed(args.seed + epoch)
                order = torch.randperm(len(train_dataset), generator=generator).tolist()
                micro_index = 0

            def save_signal_checkpoint(next_micro_index: int, updated_step: int) -> None:
                artifact = checkpoint_artifact(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    args=args,
                    history=history,
                    epoch_completed=epoch - 1,
                    global_step=updated_step,
                    best_epoch=best_epoch_value,
                    best_auroc=best_auroc,
                    epoch_in_progress=epoch,
                    epoch_order=order,
                    next_micro_index=next_micro_index,
                )
                atomic_torch_save(artifact, last_path)

            training, global_step = train_epoch(
                model,
                train_dataset,
                optimizer,
                scheduler,
                scaler,
                args,
                device,
                epoch=epoch,
                order=order,
                start_micro_index=micro_index,
                global_step=global_step,
                curve_handle=curve_handle,
                termination_state=termination_state,
                termination_callback=save_signal_checkpoint,
            )
            update_status(
                status_path,
                status,
                state="evaluating",
                current_epoch=epoch,
                global_step=global_step,
            )
            evaluation = evaluate_epoch(
                model, valid_dataset, shuffled_datasets, args, device
            )
            row = {
                "epoch": epoch,
                "training": training,
                **evaluation,
                "stable_visual_margin": visual_margin_is_stable(evaluation),
            }
            history.append(row)
            selected = pixel_best_epoch(history)
            is_best = selected["epoch"] == epoch
            if is_best:
                best_epoch_value = epoch
                best_auroc = row["normal"]["auroc"]
            artifact = checkpoint_artifact(
                model,
                optimizer,
                scheduler,
                scaler,
                args=args,
                history=history,
                epoch_completed=epoch,
                global_step=global_step,
                best_epoch=best_epoch_value,
                best_auroc=best_auroc,
            )
            atomic_torch_save(artifact, last_path)
            if is_best:
                atomic_torch_save(artifact, best_path)
            del artifact
            write_json(args.output_dir / f"epoch_{epoch}_metrics.json", row)
            update_status(
                status_path,
                status,
                state="training",
                current_epoch=epoch,
                global_step=global_step,
                best_epoch=best_epoch_value,
                best_auroc=best_auroc,
            )
            if termination_state["requested"]:
                raise TerminationRequested(
                    "Termination signal received after epoch checkpoint was saved"
                )
            stop_reason = pixel_stopping_reason(history, max_epochs=args.max_epochs)
            if stop_reason:
                break
            epoch += 1
            resume_epoch = None

    if stop_reason is None:
        raise RuntimeError("Training exited without an autonomous stop reason")
    update_status(status_path, status, state="final_evaluation", stop_reason=stop_reason)
    best_artifact = torch.load(best_path, map_location="cpu", weights_only=False)
    incompatible = model.bridge.load_state_dict(best_artifact["bridge_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Best checkpoint strict reload failed")
    if any(not bool(torch.isfinite(parameter).all().item()) for parameter in model.bridge.parameters()):
        raise FloatingPointError("Best checkpoint contains non-finite trainable parameters")
    final = final_statistics(
        model, valid_dataset, shuffled_datasets, shuffle_seeds, args, device
    )
    report = {
        "state": "complete",
        "stop_epoch": history[-1]["epoch"],
        "stop_reason": stop_reason,
        "best_epoch": best_epoch_value,
        "epochs": history,
        "final": final,
        "byte_reference": BYTE_REFERENCE,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "total_duration_seconds": time.perf_counter() - started,
        "cache_root": str(args.cache_root.resolve()),
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "train_log": str((args.output_dir / "train.log").resolve()),
        "git": git_metadata(),
        "test_split_used": False,
    }
    report_path = args.output_dir / "SPATIALSENSE_PIXEL_BASELINE.md"
    render_report(report, report_path)
    write_json(args.output_dir / "training_report.json", report)
    (args.output_dir / "DONE").touch()
    update_status(
        status_path,
        status,
        state="done",
        end_time=now_iso(),
        current_epoch=history[-1]["epoch"],
        global_step=global_step,
        best_epoch=best_epoch_value,
        best_auroc=best_auroc,
        stop_reason=stop_reason,
        exit_code=0,
    )


def write_failure(
    args: argparse.Namespace,
    status: dict[str, Any],
    status_path: Path,
    error: BaseException,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stop_reason = (
        "terminated_by_signal"
        if isinstance(error, TerminationRequested)
        else f"failure:{type(error).__name__}"
    )
    failure = {
        "state": "failed",
        "stop_reason": stop_reason,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "time": now_iso(),
        "last_checkpoint": str((args.output_dir / "last.pt").resolve())
        if (args.output_dir / "last.pt").exists()
        else None,
    }
    write_json(args.output_dir / "failure.json", failure)
    atomic_write_text(
        args.output_dir / "failure_report.md",
        "# SpatialSense Pixel Baseline Failure\n\n"
        f"Stop reason: `{stop_reason}`\n\n"
        f"Error: `{type(error).__name__}: {error}`\n\n"
        f"Last safe checkpoint: `{failure['last_checkpoint']}`\n",
    )
    (args.output_dir / "FAILED").touch()
    update_status(
        status_path,
        status,
        state="failed",
        end_time=now_iso(),
        stop_reason=stop_reason,
        exit_code=130 if isinstance(error, TerminationRequested) else 1,
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(args.output_dir)
    git = git_metadata()
    status_path = args.output_dir / "status.json"
    status = {
        "state": "starting",
        "pid": os.getpid(),
        "start_time": now_iso(),
        "end_time": None,
        "current_epoch": 0,
        "global_step": 0,
        "best_epoch": None,
        "best_auroc": None,
        "stop_reason": None,
        "exit_code": None,
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "mode": (
            "smoke_only"
            if args.smoke_only
            else "cache_diagnostic_only"
            if args.cache_diagnostic_only
            else "autonomous_training"
        ),
    }
    write_json(status_path, status)
    termination_state = {"requested": False}

    def request_termination(_signum, _frame):
        termination_state["requested"] = True

    signal.signal(signal.SIGTERM, request_termination)
    signal.signal(signal.SIGINT, request_termination)
    try:
        validate_args(args)
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        cuda_probe = torch.empty(1, device=device)
        torch.cuda.synchronize(device)
        del cuda_probe
        update_status(status_path, status, state="cuda_initialized")
        set_seed(args.seed)
        if args.cache_diagnostic_only:
            update_status(status_path, status, state="cache_diagnostic_running")
            eva, _eva_audit = build_official_eva_vit_g(
                args.eva_checkpoint,
                mixed_fp16=args.cache_forward_precision == "official_mixed_fp16",
            )
            eva.to(device).eval()
            report, _rows = build_or_validate_cache(args, eva, device, sys.stdout)
            update_status(
                status_path,
                status,
                state="cache_diagnostic_passed",
                end_time=now_iso(),
                stop_reason="cache_diagnostic_only_complete",
                exit_code=0,
            )
            metrics = report["controlled_diagnostics"][
                "batch1_vs_batch8_cache_precision"
            ]["metrics"]
            print(json.dumps({"cache_diagnostic": "passed", "metrics": metrics}), flush=True)
            return 0
        if args.smoke_only:
            update_status(status_path, status, state="smoke_running")
            smoke = run_smoke(args, device)
            update_status(
                status_path,
                status,
                state="smoke_passed",
                end_time=now_iso(),
                global_step=args.smoke_steps,
                stop_reason="smoke_only_complete",
                exit_code=0,
                best_auroc=None,
            )
            print(json.dumps({"smoke": "passed", **smoke}, ensure_ascii=False), flush=True)
            return 0
        run_autonomous(args, device, status, status_path, termination_state)
        return 0
    except BaseException as error:
        write_failure(args, status, status_path, error)
        traceback.print_exc()
        return 130 if isinstance(error, TerminationRequested) else 1
    finally:
        if lock.exists() and lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock.unlink()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
