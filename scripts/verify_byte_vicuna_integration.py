#!/usr/bin/env python3
"""Verify the complete frozen-ByteFormer/frozen-Vicuna VLM chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from spacebyte.jpeg_bytes import pad_byte_sequences, preprocess_image_to_jpeg_bytes
from spacebyte.models import (
    ByteFormerTokenAdapter,
    ByteQFormerBridge,
    ByteVicuna,
    VICUNA_REPO_ID,
    VICUNA_REVISION,
    build_official_byteformer,
    build_xinstruct_qformer_tokenizer,
    embedding_statistics,
    initialize_byte_qformer_from_blip2_stage1,
    load_frozen_vicuna,
)
from spacebyte.training import audit_causal_labels


DEFAULT_QFORMER_TOKENIZER = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/bert-base-uncased-tokenizer"
)
DEFAULT_BLIP2 = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/blip2_stage1/blip2_pretrained.pth"
)
DEFAULT_BYTEFORMER_CONFIG = Path(
    "/home/Zhang_jinsong/spacebyte/corenet/projects/byteformer/"
    "imagenet_jpeg_q100/conv_kernel_size=8.yaml"
)
DEFAULT_BYTEFORMER_CHECKPOINT = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/"
    "byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt"
)
EXPECTED_WEIGHT_FILES = {
    "pytorch_model-00001-of-00002.bin": {
        "size_bytes": 9_976_634_558,
        "sha256": "5ed572be140240b212049a3e791271eed8a04c40bc732c91a4da4b1469db23b1",
    },
    "pytorch_model-00002-of-00002.bin": {
        "size_bytes": 3_500_315_539,
        "sha256": "d9382da358f0ec38c4fca3bcf1e3e65274ae4c78090e0775b4bb4dea6a518e08",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vicuna-path", type=Path, required=True)
    parser.add_argument(
        "--qformer-tokenizer", type=Path, default=DEFAULT_QFORMER_TOKENIZER
    )
    parser.add_argument("--blip2-checkpoint", type=Path, default=DEFAULT_BLIP2)
    parser.add_argument(
        "--byteformer-config", type=Path, default=DEFAULT_BYTEFORMER_CONFIG
    )
    parser.add_argument(
        "--byteformer-checkpoint", type=Path, default=DEFAULT_BYTEFORMER_CHECKPOINT
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(path: Path) -> dict[str, Any]:
    files = []
    total = 0
    for item in sorted(path.iterdir()):
        if not item.is_file():
            continue
        size = item.stat().st_size
        total += size
        record: dict[str, Any] = {"name": item.name, "size_bytes": size}
        if item.name.startswith("pytorch_model-") and item.suffix == ".bin":
            record["sha256"] = sha256(item)
            expected = EXPECTED_WEIGHT_FILES.get(item.name)
            record["matches_official_size"] = bool(
                expected and size == expected["size_bytes"]
            )
            record["matches_official_sha256"] = bool(
                expected and record["sha256"] == expected["sha256"]
            )
        files.append(record)
    manifest = {
        "repo_id": VICUNA_REPO_ID,
        "revision": VICUNA_REVISION,
        "path": str(path.resolve()),
        "total_file_bytes": total,
        "files": files,
    }
    weights = {
        item["name"]: item
        for item in files
        if item["name"] in EXPECTED_WEIGHT_FILES
    }
    manifest["all_official_weight_checks_passed"] = (
        set(weights) == set(EXPECTED_WEIGHT_FILES)
        and all(
            item["matches_official_size"] and item["matches_official_sha256"]
            for item in weights.values()
        )
    )
    return manifest


def find_streams(image_dir: Path, opts: Any, count: int = 2):
    found = []
    lengths = set()
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        if path.name.startswith("._"):
            continue
        stream = preprocess_image_to_jpeg_bytes(path, opts)
        if stream.numel() in lengths:
            continue
        found.append((path, stream))
        lengths.add(stream.numel())
        if len(found) == count:
            break
    if len(found) != count:
        raise RuntimeError(f"Found only {len(found)} distinct JPEG streams")
    return sorted(found, key=lambda pair: pair[1].numel())


def error(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_fp32 = left.float()
    right_fp32 = right.float()
    difference = (left_fp32 - right_fp32).abs()
    reference_mean = left_fp32.abs().mean().clamp_min(torch.finfo(torch.float32).eps)
    dot = (left_fp32 * right_fp32).sum()
    norm_product = left_fp32.norm() * right_fp32.norm()
    return {
        "max_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "mean_error_fraction_of_reference": float(
            (difference.mean() / reference_mean).item()
        ),
        "cosine_similarity": float((dot / norm_product).item()),
        "top1_agreement": float(
            left_fp32.argmax(dim=-1).eq(right_fp32.argmax(dim=-1)).float().mean().item()
        ),
    }


def logits_for(model: ByteVicuna, prepared) -> torch.Tensor:
    return model.vicuna(
        inputs_embeds=prepared.inputs_embeds,
        attention_mask=prepared.attention_mask,
        use_cache=False,
        return_dict=True,
    ).logits


def cuda_memory(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {"allocated_bytes": 0, "peak_allocated_bytes": 0}
    index = device.index if device.index is not None else 0
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
    }


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device.index if device.index is not None else 0)


def gpu_inventory() -> list[dict[str, Any]]:
    inventory = []
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        inventory.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_bytes": int(total_bytes),
                "free_bytes_at_start": int(free_bytes),
            }
        )
    return inventory


def gradient_report(model: ByteVicuna) -> dict[str, Any]:
    named = dict(model.bridge.named_parameters())
    self_attention = [key for key in named if ".attention.self.query.weight" in key]
    cross_kv = {
        layer: [
            f"qformer.bert.encoder.layer.{layer}.crossattention.self.key.weight",
            f"qformer.bert.encoder.layer.{layer}.crossattention.self.value.weight",
        ]
        for layer in (0, 2, 4, 6, 8, 10)
    }

    def norm(key: str) -> float:
        gradient = named[key].grad
        return 0.0 if gradient is None else float(gradient.float().norm().item())

    cross_report = {
        str(layer): {key.rsplit(".", 2)[-2]: norm(key) for key in keys}
        for layer, keys in cross_kv.items()
    }
    groups = {
        "query_tokens": norm("query_tokens"),
        "byte_layernorm": max(norm(key) for key in named if key.startswith("byte_layernorm.")),
        "qformer_self_attention": max(norm(key) for key in self_attention),
        "projector": max(norm(key) for key in named if key.startswith("llm_projector.")),
    }
    return {
        "group_gradient_norms": groups,
        "cross_attention_key_value_gradient_norms": cross_report,
        "all_required_nonzero": all(value > 0 for value in groups.values())
        and all(
            value > 0
            for layer in cross_report.values()
            for value in layer.values()
        ),
        "byteformer_parameters_with_grad": [
            name
            for name, parameter in model.byteformer_adapter.named_parameters()
            if parameter.grad is not None
        ],
        "vicuna_parameters_with_grad": [
            name
            for name, parameter in model.vicuna.named_parameters()
            if parameter.grad is not None
        ],
        "vicuna_trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.vicuna.parameters()
            if parameter.requires_grad
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(20261012)
    detected_gpus = gpu_inventory() if torch.cuda.is_available() else []
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    reset_peak(device)

    vicuna_manifest = model_manifest(args.vicuna_path)
    if not vicuna_manifest["all_official_weight_checks_passed"]:
        raise RuntimeError(f"Official Vicuna weight integrity check failed: {vicuna_manifest}")

    bridge = ByteQFormerBridge().to(device)
    initialization = initialize_byte_qformer_from_blip2_stage1(
        bridge, args.blip2_checkpoint
    )
    byteformer, byte_opts, byteformer_load = build_official_byteformer(
        args.byteformer_config, args.byteformer_checkpoint
    )
    adapter = ByteFormerTokenAdapter(byteformer).to(device)
    vicuna, llm_tokenizer, vicuna_config = load_frozen_vicuna(
        args.vicuna_path, device=device, dtype=torch.float16
    )
    qformer_tokenizer = build_xinstruct_qformer_tokenizer(args.qformer_tokenizer)
    model = ByteVicuna(
        byteformer_adapter=adapter,
        bridge=bridge,
        vicuna=vicuna,
        llm_tokenizer=llm_tokenizer,
        qformer_tokenizer=qformer_tokenizer,
    ).eval()
    memory_after_load = cuda_memory(device)

    text_only_tokens = llm_tokenizer(
        "USER: Hello\nASSISTANT:", return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        text_only_output = vicuna(
            input_ids=text_only_tokens.input_ids,
            attention_mask=text_only_tokens.attention_mask,
            use_cache=False,
            return_dict=True,
        )
    text_only_report = {
        "input_shape": list(text_only_tokens.input_ids.shape),
        "logits_shape": list(text_only_output.logits.shape),
        "all_finite": bool(torch.isfinite(text_only_output.logits).all().item()),
    }

    selected = find_streams(args.image_dir, byte_opts)
    byte_batch = pad_byte_sequences([stream for _, stream in selected]).to(device)
    single_byte = byte_batch[:1, : selected[0][1].numel()]
    question = "What is the spatial relation of the subject to the object?"
    answer = "left of"
    questions_pair = [question, "Where is the subject relative to the object?"]
    answers_pair = [answer, "above and behind"]

    reset_peak(device)
    generated_text, generated_ids, generated_prepared = model.generate(
        single_byte, question, num_beams=1, max_length=8
    )
    generation_memory = cuda_memory(device)
    if generated_ids.numel() == 0:
        raise RuntimeError("Vicuna generated no token")

    model.eval()
    with torch.no_grad():
        baseline_prepared = model.prepare_inputs(
            byte_batch, questions_pair, answers_pair
        )
        baseline_logits = logits_for(model, baseline_prepared)

        changed_features = baseline_prepared.byte_features.clone()
        padding_positions = baseline_prepared.byte_attention_mask.eq(0)
        if not bool(padding_positions.any().item()):
            raise RuntimeError("Mixed-length JPEG batch has no padded Byte tokens")
        changed_features[padding_positions] = 1000 * torch.randn_like(
            changed_features[padding_positions]
        )
        changed_prepared = model.prepare_inputs_from_byte_features(
            changed_features,
            baseline_prepared.byte_attention_mask,
            questions_pair,
            answers_pair,
        )
        changed_logits = logits_for(model, changed_prepared)

        extended_bytes = torch.nn.functional.pad(byte_batch, (0, 257), value=-1)
        extended_prepared = model.prepare_inputs(
            extended_bytes, questions_pair, answers_pair
        )
        extended_logits = logits_for(model, extended_prepared)

        text_padded_prepared = model.prepare_inputs(
            byte_batch,
            questions_pair,
            answers_pair,
            extra_text_padding=7,
        )
        text_padded_logits = logits_for(model, text_padded_prepared)

        repeated_prepared = model.prepare_inputs(
            byte_batch, questions_pair, answers_pair
        )
        repeated_logits = logits_for(model, repeated_prepared)

        zero_query_prepared = model.prepare_inputs(
            byte_batch,
            questions_pair,
            answers_pair,
            projected_query_scale=0.0,
        )
        zero_query_logits = logits_for(model, zero_query_prepared)

    valid_width = baseline_logits.shape[1]
    baseline_valid_positions = baseline_prepared.attention_mask.bool()
    text_padded_prefix_logits = text_padded_logits[:, :valid_width]
    padding_checks = {
        "changed_byte_padding_content": error(baseline_logits, changed_logits),
        "added_right_byte_padding": error(baseline_logits, extended_logits),
        "added_text_padding_valid_positions": error(
            baseline_logits[baseline_valid_positions],
            text_padded_prefix_logits[baseline_valid_positions],
        ),
        "deterministic_repeat": error(baseline_logits, repeated_logits),
        "zero_query_prefix": error(baseline_logits, zero_query_logits),
    }

    single_for_loss = single_byte
    model.train()
    model.zero_grad(set_to_none=True)
    reset_peak(device)
    loss_output = model(single_for_loss, question, answer)
    loss_forward_memory = cuda_memory(device)
    if not bool(torch.isfinite(loss_output.loss).item()):
        raise RuntimeError("Causal LM loss is NaN or Inf")
    reset_peak(device)
    loss_output.loss.backward()
    backward_memory = cuda_memory(device)
    gradients = gradient_report(model)

    labels = loss_output.prepared.labels
    if labels is None:
        raise RuntimeError("Loss path did not create labels")
    audit_labels = baseline_prepared.labels
    if audit_labels is None:
        raise RuntimeError("Label audit path did not create labels")
    causal_label_audit = audit_causal_labels(
        baseline_prepared, answers_pair, llm_tokenizer
    )
    query_slice = baseline_prepared.query_slice
    text_padding_positions = baseline_prepared.attention_mask.eq(0)
    label_report = {
        "batch_context": "separate two-sample mixed-length label audit",
        "shape": list(audit_labels.shape),
        "numel": audit_labels.numel(),
        "ignored_count": int(audit_labels.eq(-100).sum().item()),
        "supervised_count": int(audit_labels.ne(-100).sum().item()),
        "count_conservation": (
            int(audit_labels.eq(-100).sum().item())
            + int(audit_labels.ne(-100).sum().item())
            == audit_labels.numel()
        ),
        "supervised_answer_eos_audit": causal_label_audit["per_sample"],
        "fixed_prefix_cue_all_ignored": bool(
            audit_labels[:, : query_slice.start].eq(-100).all().item()
        ),
        "query_all_ignored": bool(
            audit_labels[:, query_slice].eq(-100).all().item()
        ),
        "prompt_all_ignored": all(
            bool(audit_labels[index, prompt_slice].eq(-100).all().item())
            for index, prompt_slice in enumerate(baseline_prepared.prompt_slices)
        ),
        "answer_all_supervised": all(
            bool(audit_labels[index, answer_slice].ne(-100).all().item())
            for index, answer_slice in enumerate(baseline_prepared.answer_slices)
        ),
        "padding_position_count": int(text_padding_positions.sum().item()),
        "padding_all_ignored": bool(
            text_padding_positions.any().item()
            and audit_labels[text_padding_positions].eq(-100).all().item()
        ),
    }

    embeds = generated_prepared.inputs_embeds
    qslice = generated_prepared.query_slice
    text_values = torch.cat([embeds[:, : qslice.start], embeds[:, qslice.stop :]], dim=1)
    scale = {
        "projected_byte_queries": embedding_statistics(
            generated_prepared.projected_queries
        ),
        "vicuna_text_embeddings": embedding_statistics(text_values),
    }

    all_finite = bool(
        torch.isfinite(loss_output.logits).all().item()
        and torch.isfinite(generated_prepared.inputs_embeds).all().item()
    )
    requirements = {
        "generation_nonempty_token": generated_ids.numel() > 0,
        "decoded_output_is_string": bool(generated_text)
        and isinstance(generated_text[0], str),
        "generated_non_special_token": bool(
            generated_ids.ne(llm_tokenizer.eos_token_id)
            .logical_and(generated_ids.ne(llm_tokenizer.pad_token_id))
            .any()
            .item()
        ),
        "loss_finite": bool(torch.isfinite(loss_output.loss).item()),
        "text_only_forward_finite": text_only_report["all_finite"],
        "projector_matches_vicuna_hidden_size": (
            bridge.llm_projector.out_features == vicuna.config.hidden_size == 4096
        ),
        "labels_correct": all(
            label_report[key]
            for key in (
                "query_all_ignored",
                "fixed_prefix_cue_all_ignored",
                "prompt_all_ignored",
                "answer_all_supervised",
                "padding_all_ignored",
                "count_conservation",
            )
        ),
        "byte_padding_invariant": padding_checks["changed_byte_padding_content"][
            "max_absolute_error"
        ]
        <= 1e-5,
        "right_byte_padding_invariant": padding_checks["added_right_byte_padding"][
            "max_absolute_error"
        ]
        <= 1e-5,
        # Changing sequence width can select different FP16 CUDA reduction
        # kernels across 32 Vicuna layers. Audit both absolute drift and the
        # resulting distribution/ranking instead of relying on one extremal
        # vocabulary logit alone.
        "text_padding_invariant": (
            padding_checks["added_text_padding_valid_positions"][
                "max_absolute_error"
            ]
            <= 1.0
            and padding_checks["added_text_padding_valid_positions"][
                "mean_absolute_error"
            ]
            <= 3e-2
            and padding_checks["added_text_padding_valid_positions"][
                "cosine_similarity"
            ]
            >= 0.9999
            and padding_checks["added_text_padding_valid_positions"][
                "top1_agreement"
            ]
            >= 0.95
            and padding_checks["added_text_padding_valid_positions"][
                "mean_error_fraction_of_reference"
            ]
            <= 1e-2
        ),
        "deterministic": padding_checks["deterministic_repeat"][
            "max_absolute_error"
        ]
        == 0,
        "query_prefix_changes_logits": padding_checks["zero_query_prefix"][
            "max_absolute_error"
        ]
        > 1e-5,
        "required_gradients_nonzero": gradients["all_required_nonzero"],
        "byteformer_has_no_grad": not gradients["byteformer_parameters_with_grad"],
        "vicuna_has_no_grad": not gradients["vicuna_parameters_with_grad"],
        "vicuna_frozen": gradients["vicuna_trainable_parameter_count"] == 0,
        "all_outputs_finite": all_finite,
    }
    if not all(requirements.values()):
        raise RuntimeError(
            "Integration verification failed: "
            f"requirements={requirements}; padding_checks={padding_checks}"
        )

    report = {
        "vicuna_manifest": vicuna_manifest,
        "vicuna_configuration": vicuna_config,
        "text_only_forward": text_only_report,
        "prompt_protocol": {
            "prefix": "USER: ",
            "image_cue": "image: ",
            "postfix": "\\nASSISTANT:",
            "order": [
                "prefix tokens (including one start token)",
                "image cue tokens (duplicate start token removed)",
                "32 projected Byte Query tokens",
                "question + postfix tokens",
                "answer + EOS tokens (loss path only)",
            ],
        },
        "blip2_initialization": {
            "loaded_tensor_count": initialization["summary"]["loaded_tensor_count"],
            "shape_mismatch_tensor_count": initialization["summary"][
                "shape_mismatch_tensor_count"
            ],
            "cross_attention_key_value_tensors_retained_random": 12,
            "all_loaded_tensors_exact": initialization["verification"][
                "all_loaded_tensors_exact"
            ],
        },
        "byteformer_checkpoint": byteformer_load,
        "samples": [
            {"path": str(path.resolve()), "jpeg_byte_length": stream.numel()}
            for path, stream in selected
        ],
        "shapes": {
            "jpeg_bytes": list(single_byte.shape),
            "byte_features": list(loss_output.prepared.byte_features.shape),
            "byte_attention_mask": list(
                loss_output.prepared.byte_attention_mask.shape
            ),
            "query_features": list(loss_output.prepared.query_features.shape),
            "projected_queries": list(
                loss_output.prepared.projected_queries.shape
            ),
            "text_embeds": [
                loss_output.prepared.inputs_embeds.shape[0],
                loss_output.prepared.inputs_embeds.shape[1] - 32,
                loss_output.prepared.inputs_embeds.shape[2],
            ],
            "inputs_embeds": list(loss_output.prepared.inputs_embeds.shape),
            "attention_mask": list(loss_output.prepared.attention_mask.shape),
            "labels": list(labels.shape),
            "logits": list(loss_output.logits.shape),
        },
        "generation": {
            "token_ids": generated_ids.tolist(),
            "decoded": generated_text,
        },
        "causal_lm_loss": float(loss_output.loss.detach().float().item()),
        "labels": label_report,
        "padding_determinism_and_prefix_checks": padding_checks,
        "text_padding_fp16_acceptance": {
            "max_absolute_error_at_most": 1.0,
            "mean_absolute_error_at_most": 3e-2,
            "mean_error_fraction_of_reference_at_most": 1e-2,
            "cosine_similarity_at_least": 0.9999,
            "top1_agreement_at_least": 0.95,
            "reason": (
                "Changing sequence width may select different FP16 CUDA "
                "reduction kernels across 32 frozen Vicuna layers."
            ),
        },
        "gradients": gradients,
        "embedding_scale": scale,
        "memory": {
            "detected_gpus_at_start": detected_gpus,
            "selected_device": str(device),
            "after_full_model_load": memory_after_load,
            "generation": generation_memory,
            "loss_forward": loss_forward_memory,
            "backward": backward_memory,
        },
        "runtime_strategy": {
            "vicuna_dtype": "torch.float16",
            "use_cache": False,
            "multi_gpu": False,
            "gradient_checkpointing": False,
            "quantization": False,
            "optimizer_step": False,
        },
        "requirements": requirements,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
