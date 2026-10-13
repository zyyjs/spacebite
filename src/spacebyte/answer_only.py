"""Single-answer-token objectives and metrics for SpatialSense."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .data import collate_cached_features
from .evaluation import greedy_metrics, likelihood_metrics
from .models import ByteVicuna


def assert_clean_blip2_initialization(load_report: dict[str, Any]) -> None:
    if load_report.get("bridge_checkpoint_load") is not None:
        raise RuntimeError("Answer-only pilot must not load a prior bridge checkpoint")
    summary = load_report.get("blip2_initialization", {})
    query_summary = summary.get("by_module", {}).get("query_tokens", {})
    if (
        not summary
        or query_summary.get("loadable_tensors") != 1
        or query_summary.get("target_migration_fraction") != 1.0
    ):
        raise RuntimeError("BLIP-2 Stage-1 initialization was not verified")


def binary_answer_token_ids(tokenizer: Any) -> dict[str, int]:
    result = {}
    for answer in ("yes", "no"):
        encoded = tokenizer(
            f"{answer}{tokenizer.eos_token}", add_special_tokens=True
        ).input_ids[1:]
        if len(encoded) != 2 or encoded[-1] != tokenizer.eos_token_id:
            raise RuntimeError(f"Unexpected {answer!r} + EOS tokenization: {encoded}")
        result[answer] = int(encoded[0])
    if result["yes"] == result["no"]:
        raise RuntimeError("yes and no map to the same token")
    return result


def audit_answer_only_labels(prepared: Any, answers: list[str], tokenizer: Any) -> dict[str, Any]:
    labels = prepared.labels
    if labels is None:
        raise ValueError("Prepared inputs have no labels")
    token_ids = binary_answer_token_ids(tokenizer)
    rows = []
    for index, answer in enumerate(answers):
        positions = labels[index].ne(-100).nonzero(as_tuple=False).flatten()
        if positions.numel() != 1:
            raise RuntimeError(
                f"Row {index} has {positions.numel()} supervised tokens, expected one"
            )
        position = int(positions.item())
        if int(labels[index, position].item()) != token_ids[answer]:
            raise RuntimeError("Supervised label is not the requested yes/no token")
        answer_slice = prepared.answer_slices[index]
        if position != answer_slice.start:
            raise RuntimeError("Answer target is not at the first answer position")
        if int(labels[index, answer_slice.start + 1].item()) != -100:
            raise RuntimeError("EOS unexpectedly participates in answer-only loss")
        rows.append(
            {
                "index": index,
                "answer": answer,
                "supervised_position": position,
                "prediction_logit_position": position - 1,
                "token_id": token_ids[answer],
                "eos_position": answer_slice.start + 1,
                "eos_label": -100,
                "causal_shift_is_one": position - (position - 1) == 1,
            }
        )
    return {
        "batch_size": len(answers),
        "supervised_token_count": int(labels.ne(-100).sum().item()),
        "ignored_token_count": int(labels.eq(-100).sum().item()),
        "exactly_one_supervised_token_per_sample": True,
        "eos_fully_ignored": True,
        "rows": rows,
    }


@torch.no_grad()
def next_token_binary_logits(
    model: ByteVicuna,
    byte_features: torch.Tensor,
    byte_attention_mask: torch.Tensor,
    questions: list[str],
    *,
    projected_query_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    prepared = model.prepare_inputs_from_byte_features(
        byte_features,
        byte_attention_mask,
        questions,
        answers=None,
        projected_query_scale=projected_query_scale,
        text_padding_side="right",
    )
    outputs = model.vicuna(
        inputs_embeds=prepared.inputs_embeds,
        attention_mask=prepared.attention_mask,
        use_cache=False,
        return_dict=True,
    )
    positions = prepared.attention_mask.sum(dim=1).long() - 1
    for row, position in enumerate(positions.tolist()):
        if not bool(prepared.attention_mask[row, : position + 1].all().item()):
            raise RuntimeError("Prompt mask is not a contiguous valid prefix")
        if bool(prepared.attention_mask[row, position + 1 :].any().item()):
            raise RuntimeError("Prompt padding is not a contiguous suffix")
    next_logits = outputs.logits[
        torch.arange(outputs.logits.shape[0], device=outputs.logits.device), positions
    ]
    return next_logits, positions, prepared


@torch.no_grad()
def evaluate_answer_only(
    model: ByteVicuna,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    projected_query_scale: float = 1.0,
    num_workers: int = 0,
) -> dict[str, Any]:
    model.eval()
    ids = binary_answer_token_ids(model.llm_tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_cached_features,
        pin_memory=num_workers > 0,
    )
    labels: list[bool] = []
    scores: list[float] = []
    relations: list[str] = []
    generated: list[str] = []
    outputs = []
    started = time.perf_counter()
    for batch in loader:
        features = batch["byte_features"].to(device, non_blocking=True)
        mask = batch["byte_attention_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, positions, _ = next_token_binary_logits(
                model,
                features,
                mask,
                batch["questions"],
                projected_query_scale=projected_query_scale,
            )
        batch_scores = logits[:, ids["yes"]].float() - logits[:, ids["no"]].float()
        token_predictions = logits.argmax(dim=-1)
        for index, token_id in enumerate(token_predictions.tolist()):
            if token_id == ids["yes"]:
                parsed = "yes"
            elif token_id == ids["no"]:
                parsed = "no"
            else:
                parsed = "invalid"
            generated.append(parsed)
            outputs.append(
                {
                    "sample_id": batch["sample_ids"][index],
                    "label": batch["labels"][index],
                    "relation": batch["relations_raw"][index],
                    "single_token_id": token_id,
                    "single_token_text": model.llm_tokenizer.decode([token_id]),
                    "single_token_prediction": parsed,
                    "score_logit_yes_minus_no": float(batch_scores[index].item()),
                    "autoregressive_logit_position": int(positions[index].item()),
                }
            )
        labels.extend(batch["labels"])
        relations.extend(batch["relations_raw"])
        scores.extend(batch_scores.cpu().tolist())
    return {
        "sample_count": len(labels),
        "projected_query_scale": projected_query_scale,
        "duration_seconds": time.perf_counter() - started,
        "answer_token_ids": ids,
        "single_token_generation": greedy_metrics(labels, generated, relations),
        "logit_yes_minus_no": likelihood_metrics(labels, scores, relations),
        "outputs": outputs,
    }


@torch.no_grad()
def audit_answer_eos_nll(
    model: ByteVicuna,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> dict[str, Any]:
    model.eval()
    ids = binary_answer_token_ids(model.llm_tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_cached_features,
        pin_memory=num_workers > 0,
    )
    answer_nlls: list[float] = []
    eos_nlls: list[float] = []
    labels: list[bool] = []
    scores: list[float] = []
    relations: list[str] = []
    started = time.perf_counter()
    for batch in loader:
        features = batch["byte_features"].to(device, non_blocking=True)
        mask = batch["byte_attention_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model.forward_from_byte_features(
                features, mask, batch["questions"], batch["answers"]
            )
        for index, answer_slice in enumerate(output.prepared.answer_slices):
            answer_position = answer_slice.start
            eos_position = answer_position + 1
            answer_target = int(output.prepared.labels[index, answer_position].item())
            eos_target = int(output.prepared.labels[index, eos_position].item())
            if answer_target != ids[batch["answers"][index]]:
                raise RuntimeError("Answer target/token mismatch during audit")
            if eos_target != int(model.llm_tokenizer.eos_token_id):
                raise RuntimeError("EOS target mismatch during audit")
            answer_logits = output.logits[index, answer_position - 1].float()
            eos_logits = output.logits[index, eos_position - 1].float()
            answer_nlls.append(
                float(-answer_logits.log_softmax(-1)[answer_target].item())
            )
            eos_nlls.append(float(-eos_logits.log_softmax(-1)[eos_target].item()))
            scores.append(float((answer_logits[ids["yes"]] - answer_logits[ids["no"]]).item()))
        labels.extend(batch["labels"])
        relations.extend(batch["relations_raw"])
    metrics = likelihood_metrics(labels, scores, relations)
    return {
        "sample_count": len(labels),
        "answer_token_nll": sum(answer_nlls) / len(answer_nlls),
        "eos_nll": sum(eos_nlls) / len(eos_nlls),
        "mean_two_token_nll": (
            sum(answer_nlls) + sum(eos_nlls)
        )
        / (2 * len(answer_nlls)),
        "answer_fraction_of_total_nll": sum(answer_nlls)
        / (sum(answer_nlls) + sum(eos_nlls)),
        "logit_yes_minus_no": metrics,
        "duration_seconds": time.perf_counter() - started,
    }
