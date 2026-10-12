"""Auditable BLIP-2 Stage-1 initialization for :class:`ByteQFormerBridge`."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .byte_qformer_bridge import ByteQFormerBridge


BLIP2_STAGE1_URL = (
    "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/"
    "models/BLIP2/blip2_pretrained.pth"
)
BLIP2_STAGE1_SHA256 = "f31f96e4a97ce9ac7a140ea1961b8a871822ae1cd0a23f0f1122fbefd70db9f0"
BLIP2_STAGE1_SIZE_BYTES = 746_998_955

_CROSS_MISMATCH = re.compile(
    r"^qformer\.bert\.encoder\.layer\.(0|2|4|6|8|10)\."
    r"crossattention\.self\.(key|value)\.weight$"
)
_BLIP2_AUXILIARY_PREFIXES = (
    "temp",
    "ln_vision.",
    "vision_proj.",
    "text_proj.",
    "itm_head.",
)
_PROJECT_ADDED_PREFIXES = ("byte_layernorm.", "llm_projector.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Tensor]]:
    resolved = path.resolve()
    size_bytes = resolved.stat().st_size
    sha256 = file_sha256(resolved)
    if size_bytes != BLIP2_STAGE1_SIZE_BYTES:
        raise RuntimeError(
            f"BLIP-2 Stage-1 size mismatch: expected {BLIP2_STAGE1_SIZE_BYTES}, "
            f"got {size_bytes}"
        )
    if sha256 != BLIP2_STAGE1_SHA256:
        raise RuntimeError(
            f"BLIP-2 Stage-1 SHA-256 mismatch: expected {BLIP2_STAGE1_SHA256}, "
            f"got {sha256}"
        )
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or list(checkpoint) != ["model"]:
        raise RuntimeError(
            "Expected the official checkpoint top level to contain only 'model'"
        )
    state = checkpoint["model"]
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise TypeError("checkpoint['model'] must be a string-to-tensor mapping")
    metadata = {
        "source_url": BLIP2_STAGE1_URL,
        "path": str(resolved),
        "size_bytes": size_bytes,
        "sha256": sha256,
        "top_level_type": type(checkpoint).__name__,
        "top_level_keys": list(checkpoint),
        "state_dict_type": type(state).__name__,
        "tensor_count": len(state),
        "tensor_numel": sum(tensor.numel() for tensor in state.values()),
    }
    return metadata, state


def _target_key(source_key: str) -> str | None:
    if source_key == "query_tokens":
        return source_key
    if source_key.startswith("Qformer."):
        return "qformer." + source_key.removeprefix("Qformer.")
    return None


def _module_group(key: str) -> str:
    lowered = key.lower()
    if key == "query_tokens":
        return "query_tokens"
    if key.startswith(_PROJECT_ADDED_PREFIXES):
        return "project_added"
    if ".cls." in lowered:
        return "mlm_head"
    if key.startswith(_BLIP2_AUXILIARY_PREFIXES):
        return "blip2_auxiliary"
    if ".embeddings." in lowered:
        return "token_and_position_embeddings"
    cross_match = re.search(r"\.layer\.(\d+)\.crossattention\.", key)
    if cross_match:
        return f"cross_attention_layer_{cross_match.group(1)}"
    if ".attention." in lowered:
        return "self_attention"
    if ".intermediate_query." in lowered or ".output_query." in lowered:
        return "query_ffn"
    if ".intermediate." in lowered or ".output." in lowered:
        return "text_ffn"
    return "qformer_other"


def _record(
    *,
    status: str,
    source_key: str | None,
    target_key: str | None,
    source: Tensor | None,
    target: Tensor | None,
    reason: str,
) -> dict[str, Any]:
    grouping_key = target_key or source_key or ""
    return {
        "status": status,
        "module": _module_group(grouping_key),
        "source_key": source_key,
        "target_key": target_key,
        "source_shape": list(source.shape) if source is not None else None,
        "target_shape": list(target.shape) if target is not None else None,
        "source_numel": source.numel() if source is not None else 0,
        "target_numel": target.numel() if target is not None else 0,
        "source_is_parameter": bool(
            source_key is not None and not source_key.endswith("position_ids")
        ),
        "target_is_parameter": bool(
            target_key is not None and not target_key.endswith("position_ids")
        ),
        "reason": reason,
    }


def _build_records(
    source_state: dict[str, Tensor], target_state: dict[str, Tensor]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    mapped_targets: set[str] = set()
    for source_key, source_tensor in source_state.items():
        target_key = _target_key(source_key)
        target_tensor = target_state.get(target_key) if target_key is not None else None
        if target_tensor is None:
            records.append(
                _record(
                    status="checkpoint_only",
                    source_key=source_key,
                    target_key=target_key,
                    source=source_tensor,
                    target=None,
                    reason=(
                        "MLM head is intentionally absent from the bridge"
                        if source_key.startswith("Qformer.cls.")
                        else "BLIP-2 auxiliary or vision-branch tensor"
                    ),
                )
            )
            continue
        mapped_targets.add(target_key)
        if source_tensor.shape == target_tensor.shape:
            status = "loadable"
            reason = "semantic key mapping and shape match"
        else:
            status = "shape_mismatch"
            reason = (
                "BLIP-2 cross-attention expects 1408-wide vision features; "
                "ByteFormer supplies 192-wide features"
            )
        records.append(
            _record(
                status=status,
                source_key=source_key,
                target_key=target_key,
                source=source_tensor,
                target=target_tensor,
                reason=reason,
            )
        )

    for target_key, target_tensor in target_state.items():
        if target_key in mapped_targets:
            continue
        records.append(
            _record(
                status=(
                    "project_added"
                    if target_key.startswith(_PROJECT_ADDED_PREFIXES)
                    else "current_only"
                ),
                source_key=None,
                target_key=target_key,
                source=None,
                target=target_tensor,
                reason="Spacebyte bridge module has no BLIP-2 Stage-1 counterpart",
            )
        )
    return records


def _assert_expected_incompatibilities(records: list[dict[str, Any]]) -> None:
    unexpected: list[dict[str, Any]] = []
    for record in records:
        status = record["status"]
        source_key = record["source_key"] or ""
        target_key = record["target_key"] or ""
        if status == "shape_mismatch" and not _CROSS_MISMATCH.match(target_key):
            unexpected.append(record)
        elif status == "checkpoint_only" and not (
            source_key.startswith("Qformer.cls.")
            or source_key.startswith(_BLIP2_AUXILIARY_PREFIXES)
        ):
            unexpected.append(record)
        elif status in {"project_added", "current_only"} and not target_key.startswith(
            _PROJECT_ADDED_PREFIXES
        ):
            unexpected.append(record)
    if unexpected:
        concise = [
            {
                "status": item["status"],
                "source_key": item["source_key"],
                "target_key": item["target_key"],
            }
            for item in unexpected
        ]
        raise RuntimeError(f"Unexpected BLIP-2 incompatibilities: {concise}")


def _summarize(
    checkpoint: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    status_summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensor_count": 0, "source_numel": 0, "target_numel": 0}
    )
    module_summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "loadable_tensors": 0,
            "loadable_target_numel": 0,
            "shape_mismatch_tensors": 0,
            "shape_mismatch_source_numel": 0,
            "shape_mismatch_target_numel": 0,
            "checkpoint_only_tensors": 0,
            "checkpoint_only_numel": 0,
            "retained_target_tensors": 0,
            "retained_target_numel": 0,
        }
    )
    for item in records:
        status = item["status"]
        status_summary[status]["tensor_count"] += 1
        status_summary[status]["source_numel"] += item["source_numel"]
        status_summary[status]["target_numel"] += item["target_numel"]
        grouped = module_summary[item["module"]]
        if status == "loadable":
            grouped["loadable_tensors"] += 1
            grouped["loadable_target_numel"] += item["target_numel"]
        elif status == "shape_mismatch":
            grouped["shape_mismatch_tensors"] += 1
            grouped["shape_mismatch_source_numel"] += item["source_numel"]
            grouped["shape_mismatch_target_numel"] += item["target_numel"]
        elif status == "checkpoint_only":
            grouped["checkpoint_only_tensors"] += 1
            grouped["checkpoint_only_numel"] += item["source_numel"]
        else:
            grouped["retained_target_tensors"] += 1
            grouped["retained_target_numel"] += item["target_numel"]

    loaded_numel = status_summary["loadable"]["target_numel"]
    loaded_parameter_numel = sum(
        item["target_numel"]
        for item in records
        if item["status"] == "loadable" and item["target_is_parameter"]
    )
    retained_numel = sum(
        item["target_numel"]
        for item in records
        if item["status"] in {"shape_mismatch", "project_added", "current_only"}
    )
    target_numel = loaded_numel + retained_numel
    retained_parameter_numel = sum(
        item["target_numel"]
        for item in records
        if item["status"] in {"shape_mismatch", "project_added", "current_only"}
        and item["target_is_parameter"]
    )
    target_parameter_numel = loaded_parameter_numel + retained_parameter_numel
    checkpoint_parameter_numel = sum(
        item["source_numel"] for item in records if item["source_is_parameter"]
    )
    rendered_modules: dict[str, dict[str, int | float | None]] = {}
    for module, values in module_summary.items():
        target_total = (
            values["loadable_target_numel"]
            + values["shape_mismatch_target_numel"]
            + values["retained_target_numel"]
        )
        source_total = (
            values["loadable_target_numel"]
            + values["shape_mismatch_source_numel"]
            + values["checkpoint_only_numel"]
        )
        rendered_modules[module] = {
            **values,
            "target_migration_fraction": (
                values["loadable_target_numel"] / target_total if target_total else None
            ),
            "checkpoint_consumed_fraction": (
                values["loadable_target_numel"] / source_total if source_total else None
            ),
        }
    return {
        "by_status": dict(status_summary),
        "by_module": rendered_modules,
        "loaded_tensor_count": status_summary["loadable"]["tensor_count"],
        "loaded_target_numel": loaded_numel,
        "loaded_target_parameter_numel": loaded_parameter_numel,
        "loaded_fraction_of_target": loaded_numel / target_numel,
        "loaded_parameter_fraction_of_target": (
            loaded_parameter_numel / target_parameter_numel
        ),
        "loaded_fraction_of_checkpoint": loaded_numel / checkpoint["tensor_numel"],
        "shape_mismatch_tensor_count": status_summary["shape_mismatch"]["tensor_count"],
        "shape_mismatch_source_numel": status_summary["shape_mismatch"]["source_numel"],
        "shape_mismatch_target_numel": status_summary["shape_mismatch"]["target_numel"],
        "retained_random_target_numel": retained_numel,
        "retained_random_parameter_numel": retained_parameter_numel,
        "target_state_numel": target_numel,
        "target_parameter_numel": target_parameter_numel,
        "checkpoint_parameter_numel_excluding_known_buffers": (
            checkpoint_parameter_numel
        ),
    }


def _checkpoint_contents(source_state: dict[str, Tensor]) -> dict[str, Any]:
    word_key = "Qformer.bert.embeddings.word_embeddings.weight"
    return {
        "qformer_prefix": "Qformer.",
        "query_tokens": {
            "key": "query_tokens",
            "shape": list(source_state["query_tokens"].shape),
        },
        "word_embeddings": {
            "key": word_key,
            "shape": list(source_state[word_key].shape),
            "vocab_size": source_state[word_key].shape[0],
        },
        "contains_visual_encoder": any(
            key.startswith("visual_encoder.") for key in source_state
        ),
        "contains_vision_layernorm": any(
            key.startswith("ln_vision.") for key in source_state
        ),
        "contains_mlm_head": any(
            key.startswith("Qformer.cls.") for key in source_state
        ),
        "contains_itm_head": any(key.startswith("itm_head.") for key in source_state),
        "contains_classification_head": any(
            key.startswith("classifier.") for key in source_state
        ),
    }


def inspect_blip2_qformer_checkpoint(
    bridge: ByteQFormerBridge, checkpoint_path: Path
) -> dict[str, Any]:
    checkpoint, source_state = _load_checkpoint(checkpoint_path)
    records = _build_records(source_state, bridge.state_dict())
    _assert_expected_incompatibilities(records)
    return {
        "checkpoint": checkpoint,
        "key_mapping": {
            "query_tokens": "query_tokens",
            "Qformer.<suffix>": "qformer.<suffix>",
            "all_other_checkpoint_keys": None,
        },
        "contents": _checkpoint_contents(source_state),
        "summary": _summarize(checkpoint, records),
        "tensor_records": records,
    }


def initialize_byte_qformer_from_blip2_stage1(
    bridge: ByteQFormerBridge, checkpoint_path: Path
) -> dict[str, Any]:
    """Load only exact semantic/shape matches and verify every decision."""
    checkpoint, source_state = _load_checkpoint(checkpoint_path)
    target_state = bridge.state_dict()
    records = _build_records(source_state, target_state)
    _assert_expected_incompatibilities(records)
    before = {
        key: tensor.detach().cpu().clone() for key, tensor in target_state.items()
    }

    with torch.no_grad():
        for item in records:
            if item["status"] != "loadable":
                continue
            source = source_state[item["source_key"]]
            target = target_state[item["target_key"]]
            target.copy_(source.to(device=target.device, dtype=target.dtype))

    after = bridge.state_dict()
    loaded_failures: list[str] = []
    retained_failures: list[str] = []
    for item in records:
        target_key = item["target_key"]
        if target_key is None or item["status"] == "checkpoint_only":
            continue
        actual = after[target_key].detach().cpu()
        if item["status"] == "loadable":
            expected = source_state[item["source_key"]].detach().cpu()
            if not torch.equal(actual, expected):
                loaded_failures.append(target_key)
        elif item["status"] in {
            "shape_mismatch",
            "project_added",
            "current_only",
        } and not torch.equal(actual, before[target_key]):
            retained_failures.append(target_key)

    if loaded_failures or retained_failures:
        raise RuntimeError(
            "Post-migration tensor verification failed: "
            f"loaded={loaded_failures}, retained={retained_failures}"
        )
    bridge.initialization_mode = "blip2_stage1_partial"
    bridge.initialization_checkpoint_sha256 = checkpoint["sha256"]
    bridge.initialization_checkpoint_path = checkpoint["path"]
    inspection = {
        "initialization_mode": bridge.initialization_mode,
        "checkpoint": checkpoint,
        "key_mapping": {
            "query_tokens": "query_tokens",
            "Qformer.<suffix>": "qformer.<suffix>",
            "all_other_checkpoint_keys": None,
        },
        "contents": _checkpoint_contents(source_state),
        "summary": _summarize(checkpoint, records),
        "tensor_records": records,
    }
    inspection["verification"] = {
        "all_loaded_tensors_exact": True,
        "loaded_verification_failures": loaded_failures,
        "all_retained_tensors_unchanged": True,
        "retained_verification_failures": retained_failures,
        "strict_false_used": False,
    }
    inspection["intentional_initialization"] = {
        "cross_attention_key_value_weights": (
            "retained LAVIS standard random Normal(mean=0, std=0.02); biases load"
        ),
        "byte_layernorm": "PyTorch LayerNorm default: weight=1, bias=0",
        "llm_projector": "PyTorch Linear default Kaiming-uniform weight and bias",
    }
    return inspection
