"""Small, auditable training helpers for the SpatialSense overfit check."""

from __future__ import annotations

import re
from typing import Any, Sequence

import torch
from torch import Tensor, nn


_NEW_CROSS_KV = re.compile(
    r"^qformer\.bert\.encoder\.layer\.(0|2|4|6|8|10)\."
    r"crossattention\.self\.(key|value)\.weight$"
)


def parameter_role(name: str) -> str:
    """Assign every trainable bridge parameter to exactly one LR group."""

    if name.startswith("byte_layernorm.") or name.startswith("llm_projector."):
        return "new_modules"
    if _NEW_CROSS_KV.match(name):
        return "new_modules"
    return "blip2_initialized"


def build_bridge_optimizer_groups(
    bridge: nn.Module,
    *,
    pretrained_lr: float = 1e-5,
    new_lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build disjoint, exhaustive parameter groups and an audit report."""

    grouped: dict[str, list[tuple[str, nn.Parameter]]] = {
        "blip2_initialized": [],
        "new_modules": [],
    }
    for name, parameter in bridge.named_parameters():
        if parameter.requires_grad:
            grouped[parameter_role(name)].append((name, parameter))
    all_trainable = {
        id(parameter): name
        for name, parameter in bridge.named_parameters()
        if parameter.requires_grad
    }
    assigned_ids = [
        id(parameter)
        for entries in grouped.values()
        for _, parameter in entries
    ]
    if len(assigned_ids) != len(set(assigned_ids)):
        raise RuntimeError("A bridge parameter appears in more than one optimizer group")
    if set(assigned_ids) != set(all_trainable):
        missing = set(all_trainable) - set(assigned_ids)
        raise RuntimeError(
            f"Optimizer grouping omitted trainable parameters: {[all_trainable[x] for x in missing]}"
        )
    learning_rates = {
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
        for name in ("blip2_initialized", "new_modules")
    ]
    report = {
        "all_trainable_parameters_assigned_once": True,
        "groups": {
            name: {
                "learning_rate": learning_rates[name],
                "tensor_count": len(entries),
                "parameter_count": sum(parameter.numel() for _, parameter in entries),
                "parameter_names": [entry_name for entry_name, _ in entries],
            }
            for name, entries in grouped.items()
        },
        "total_trainable_parameters": sum(
            parameter.numel() for parameter in bridge.parameters() if parameter.requires_grad
        ),
    }
    return groups, report


def assert_frozen_modules_absent(
    optimizer_groups: Sequence[dict[str, Any]], *frozen_modules: nn.Module
) -> None:
    optimized = {
        id(parameter)
        for group in optimizer_groups
        for parameter in group["params"]
    }
    frozen = {
        id(parameter)
        for module in frozen_modules
        for parameter in module.parameters()
    }
    overlap = optimized & frozen
    if overlap:
        raise RuntimeError("Frozen ByteFormer/Vicuna parameter entered optimizer")


def audit_causal_labels(
    prepared: Any,
    answers: Sequence[str],
    tokenizer: Any,
) -> dict[str, Any]:
    """Prove count conservation and answer+EOS supervision for every row."""

    labels = prepared.labels
    if labels is None:
        raise ValueError("Prepared inputs do not contain causal labels")
    if labels.shape != prepared.attention_mask.shape:
        raise RuntimeError(
            f"labels {tuple(labels.shape)} != attention mask {tuple(prepared.attention_mask.shape)}"
        )
    ignored = int(labels.eq(-100).sum().item())
    supervised = int(labels.ne(-100).sum().item())
    if ignored + supervised != labels.numel():
        raise RuntimeError("Ignored and supervised label counts do not cover labels")
    if labels.shape[0] != len(answers):
        raise ValueError("Answer count does not match label batch")
    rows = []
    for index, answer in enumerate(answers):
        actual = labels[index][labels[index].ne(-100)].tolist()
        encoded = tokenizer(
            f"{answer}{tokenizer.eos_token}", add_special_tokens=True
        ).input_ids
        expected = list(encoded[1:])
        if actual != expected:
            raise RuntimeError(
                f"Row {index} supervised ids {actual} != answer+EOS ids {expected}"
            )
        rows.append(
            {
                "index": index,
                "answer": answer,
                "supervised_token_ids": actual,
                "decoded_with_special_tokens": tokenizer.decode(
                    actual, skip_special_tokens=False
                ),
                "expected_token_ids": expected,
            }
        )
    return {
        "shape": list(labels.shape),
        "numel": labels.numel(),
        "ignored": ignored,
        "supervised": supervised,
        "count_conservation": ignored + supervised == labels.numel(),
        "per_sample": rows,
    }


def supervised_token_accuracy(logits: Tensor, labels: Tensor) -> tuple[int, int]:
    """Count next-token predictions over non-ignored causal targets."""

    predictions = logits[:, :-1].argmax(dim=-1)
    targets = labels[:, 1:]
    supervised = targets.ne(-100)
    correct = predictions.eq(targets).logical_and(supervised).sum()
    return int(correct.item()), int(supervised.sum().item())
