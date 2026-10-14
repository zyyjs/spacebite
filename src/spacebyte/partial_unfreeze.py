"""Online SpatialSense bytes and auditable partial-ByteFormer fine-tuning."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .answer_only import binary_answer_token_ids
from .data.spatialsense import (
    flatten_spatialsense_annotations,
    render_record_to_jpeg_bytes,
)
from .evaluation import greedy_metrics, likelihood_metrics
from .jpeg_bytes import pad_byte_sequences
from .models import ByteVicuna
from .training import parameter_role


class SpatialSenseOnlineByteDataset(Dataset[dict[str, Any]]):
    """Render marked JPEG-Q100 streams without reading feature-cache tensors."""

    def __init__(
        self,
        dataset_root: Path,
        manifest_path: Path,
        *,
        expected_split: str,
    ) -> None:
        if expected_split not in {"train", "valid"}:
            raise ValueError("Online training dataset accepts only train or valid")
        manifest_rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(row["split"] != expected_split for row in manifest_rows):
            raise RuntimeError(f"Manifest contains rows outside {expected_split!r}")
        raw = flatten_spatialsense_annotations(
            dataset_root, include_splits={expected_split}
        )
        raw_by_id = {row["sample_id"]: row for row in raw}
        if len(raw_by_id) != len(raw):
            raise RuntimeError("Raw split contains duplicate sample IDs")
        records = []
        for manifest_row in manifest_rows:
            sample_id = manifest_row["sample_id"]
            if sample_id not in raw_by_id:
                raise RuntimeError(f"Manifest sample missing from raw split: {sample_id}")
            record = raw_by_id[sample_id]
            for key in ("question", "answer", "relation", "relation_raw", "label"):
                if record[key] != manifest_row[key]:
                    raise RuntimeError(f"Raw/manifest {key} mismatch for {sample_id}")
            record = dict(record)
            record["visual_identity_key"] = manifest_row["visual_cache_key"]
            records.append(record)
        if len(records) != len(raw):
            raise RuntimeError("Raw split and manifest sample counts differ")
        self.dataset_root = dataset_root
        self.manifest_path = manifest_path
        self.expected_split = expected_split
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def metadata(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "sample_id": record["sample_id"],
            "split": record["split"],
            "visual_identity_key": record["visual_identity_key"],
            "question": record["question"],
            "answer": record["answer"],
            "relation": record["relation"],
            "relation_raw": record["relation_raw"],
            "label": bool(record["label"]),
        }

    def visual_bytes(self, index: int) -> torch.Tensor:
        byte_tensor, _, _ = render_record_to_jpeg_bytes(
            self.dataset_root, self.records[index]
        )
        return byte_tensor

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = self.metadata(index)
        result["byte_tensor"] = self.visual_bytes(index)
        result["source_sample_id"] = result["sample_id"]
        result["source_visual_identity_key"] = result["visual_identity_key"]
        return result


class PermutedOnlineByteDataset(Dataset[dict[str, Any]]):
    """Keep target language/label while substituting a different JPEG stream."""

    def __init__(self, dataset: SpatialSenseOnlineByteDataset, mapping: list[int]):
        if sorted(mapping) != list(range(len(dataset))):
            raise ValueError("Visual mapping is not a complete permutation")
        if any(index == source for index, source in enumerate(mapping)):
            raise ValueError("Visual permutation contains a sample self mapping")
        keys = [row["visual_identity_key"] for row in dataset.records]
        if any(keys[index] == keys[source] for index, source in enumerate(mapping)):
            raise ValueError("Visual permutation contains an identity self mapping")
        self.dataset = dataset
        self.mapping = mapping

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.mapping[index]
        result = self.dataset.metadata(index)
        result["byte_tensor"] = self.dataset.visual_bytes(source)
        result["source_sample_id"] = self.dataset.records[source]["sample_id"]
        result["source_visual_identity_key"] = self.dataset.records[source][
            "visual_identity_key"
        ]
        return result


def collate_online_bytes(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty byte batch")
    streams = [item["byte_tensor"] for item in batch]
    return {
        "byte_tensor": pad_byte_sequences(streams),
        "byte_lengths": [int(stream.numel()) for stream in streams],
        "sample_ids": [item["sample_id"] for item in batch],
        "source_sample_ids": [item["source_sample_id"] for item in batch],
        "visual_identity_keys": [item["visual_identity_key"] for item in batch],
        "source_visual_identity_keys": [
            item["source_visual_identity_key"] for item in batch
        ],
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "relations_raw": [item["relation_raw"] for item in batch],
        "labels": [bool(item["label"]) for item in batch],
    }


def configure_byteformer_partial_unfreeze(model: ByteVicuna) -> dict[str, Any]:
    """Resolve by module identity, then unfreeze exactly the final two blocks/norm."""

    adapter = model.byteformer_adapter
    byteformer = getattr(adapter, "byteformer", None)
    if byteformer is None:
        raise TypeError("ByteFormerTokenAdapter does not expose its wrapped model")
    transformer = getattr(byteformer, "transformer", None)
    if not isinstance(transformer, (nn.ModuleList, nn.Sequential)):
        raise TypeError(
            "Expected ByteFormer transformer to be an ordered ModuleList/Sequential"
        )
    if len(transformer) < 2:
        raise RuntimeError("ByteFormer has fewer than two Transformer blocks")
    post_norm = getattr(byteformer, "post_transformer_norm", None)
    if not isinstance(post_norm, nn.Module):
        raise TypeError("ByteFormer post_transformer_norm is not a module")

    adapter.requires_grad_(False)
    selected = [transformer[len(transformer) - 2], transformer[len(transformer) - 1]]
    for module in (*selected, post_norm):
        module.requires_grad_(True)

    names_by_id = {id(module): name for name, module in byteformer.named_modules()}
    selected_names = [names_by_id[id(module)] for module in selected]
    norm_name = names_by_id[id(post_norm)]
    expected_names = [
        f"transformer.{len(transformer) - 2}",
        f"transformer.{len(transformer) - 1}",
    ]
    if selected_names != expected_names or norm_name != "post_transformer_norm":
        raise RuntimeError(
            "Resolved ByteFormer module names differ from the ordered model structure: "
            f"{selected_names}, {norm_name}"
        )
    trainable = [
        (name, parameter)
        for name, parameter in byteformer.named_parameters()
        if parameter.requires_grad
    ]
    allowed_prefixes = tuple(f"{name}." for name in selected_names) + (
        "post_transformer_norm.",
    )
    unexpected = [name for name, _ in trainable if not name.startswith(allowed_prefixes)]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable ByteFormer parameters: {unexpected}")
    frozen_roots = {
        "embeddings": byteformer.embeddings,
        "token_reduction_net": byteformer.token_reduction_net,
        "downsamplers": byteformer.downsamplers,
        "classifier": byteformer.classifier,
    }
    if any(
        parameter.requires_grad
        for module in frozen_roots.values()
        if module is not None
        for parameter in module.parameters()
    ):
        raise RuntimeError("A required frozen ByteFormer component is trainable")
    return {
        "transformer_container_type": type(transformer).__name__,
        "transformer_block_count": len(transformer),
        "unfrozen_modules": selected_names + [norm_name],
        "module_types": {
            name: f"{type(module).__module__}.{type(module).__name__}"
            for name, module in zip(selected_names + [norm_name], selected + [post_norm])
        },
        "parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "parameter_tensor_count": len(trainable),
        "parameters_by_module": {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in zip(selected_names + [norm_name], selected + [post_norm])
        },
        "trainable_parameter_names": [name for name, _ in trainable],
        "confirmed_frozen_modules": list(frozen_roots),
    }


def set_partial_training_mode(model: ByteVicuna, enabled: bool) -> None:
    model.train(enabled)
    if enabled:
        byteformer = model.byteformer_adapter.byteformer
        byteformer.transformer[-2].train(True)
        byteformer.transformer[-1].train(True)
        byteformer.post_transformer_norm.train(True)
        model.vicuna.eval()


def build_partial_unfreeze_optimizer_groups(
    model: ByteVicuna,
    *,
    byteformer_lr: float = 1e-6,
    pretrained_lr: float = 5e-6,
    new_lr: float = 5e-5,
    weight_decay: float = 0.01,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[tuple[str, nn.Parameter]]] = {
        "byteformer_unfrozen": [],
        "blip2_initialized": [],
        "new_modules": [],
    }
    for name, parameter in model.byteformer_adapter.byteformer.named_parameters():
        if parameter.requires_grad:
            grouped["byteformer_unfrozen"].append((f"byteformer.{name}", parameter))
    for name, parameter in model.bridge.named_parameters():
        if parameter.requires_grad:
            grouped[parameter_role(name)].append((f"bridge.{name}", parameter))
    all_trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assigned = [
        id(parameter) for entries in grouped.values() for _, parameter in entries
    ]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(all_trainable):
        raise RuntimeError("Partial-unfreeze optimizer groups are not exhaustive/disjoint")
    learning_rates = {
        "byteformer_unfrozen": byteformer_lr,
        "blip2_initialized": pretrained_lr,
        "new_modules": new_lr,
    }
    groups = [
        {
            "params": [parameter for _, parameter in grouped[name]],
            "lr": learning_rates[name],
            "weight_decay": weight_decay,
            "group_name": name,
        }
        for name in ("byteformer_unfrozen", "blip2_initialized", "new_modules")
    ]
    report = {
        "all_trainable_parameters_assigned_once": True,
        "groups": {
            name: {
                "learning_rate": learning_rates[name],
                "tensor_count": len(entries),
                "parameter_count": sum(parameter.numel() for _, parameter in entries),
                "parameter_names": [parameter_name for parameter_name, _ in entries],
            }
            for name, entries in grouped.items()
        },
        "total_trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    return groups, report


@torch.no_grad()
def evaluate_online_answer_only(
    model: ByteVicuna,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    projected_query_scale: float = 1.0,
    num_workers: int = 0,
) -> dict[str, Any]:
    model.eval()
    answer_ids = binary_answer_token_ids(model.llm_tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_online_bytes,
        pin_memory=num_workers > 0,
    )
    labels: list[bool] = []
    scores: list[float] = []
    relations: list[str] = []
    predictions: list[str] = []
    outputs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch in loader:
        byte_tensor = batch["byte_tensor"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prepared = model.prepare_inputs(
                byte_tensor,
                batch["questions"],
                answers=None,
                projected_query_scale=projected_query_scale,
                text_padding_side="right",
            )
            vicuna_output = model.vicuna(
                inputs_embeds=prepared.inputs_embeds,
                attention_mask=prepared.attention_mask,
                use_cache=False,
                return_dict=True,
            )
        positions = prepared.attention_mask.sum(dim=1).long() - 1
        logits = vicuna_output.logits[
            torch.arange(byte_tensor.shape[0], device=device), positions
        ].float()
        batch_scores = logits[:, answer_ids["yes"]] - logits[:, answer_ids["no"]]
        token_ids = logits.argmax(dim=-1)
        for index, token_id in enumerate(token_ids.tolist()):
            prediction = (
                "yes"
                if token_id == answer_ids["yes"]
                else "no"
                if token_id == answer_ids["no"]
                else "invalid"
            )
            predictions.append(prediction)
            outputs.append(
                {
                    "sample_id": batch["sample_ids"][index],
                    "source_sample_id": batch["source_sample_ids"][index],
                    "label": batch["labels"][index],
                    "single_token_id": token_id,
                    "single_token_prediction": prediction,
                    "score_logit_yes_minus_no": float(batch_scores[index].item()),
                    "jpeg_byte_length": batch["byte_lengths"][index],
                }
            )
        labels.extend(batch["labels"])
        scores.extend(batch_scores.cpu().tolist())
        relations.extend(batch["relations_raw"])
    return {
        "sample_count": len(labels),
        "projected_query_scale": projected_query_scale,
        "duration_seconds": time.perf_counter() - started,
        "single_token_generation": greedy_metrics(labels, predictions, relations),
        "logit_yes_minus_no": likelihood_metrics(labels, scores, relations),
        "outputs": outputs,
    }
