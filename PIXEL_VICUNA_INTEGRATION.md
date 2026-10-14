# Pixel → Vicuna integration

## Scope and result

This stage adds and verifies an independent Pixel visual path only. It does not train SpatialSense, fuse Pixel and Byte features, add a gate, alter Vicuna, or use the test split. The real-weight verification passed on one RTX 4090 with FP16 autocast: the official image branch reaches frozen Vicuna, and answer-only loss back-propagates through the Image Q-Former, Query Tokens, visual LayerNorm, and projector.

## Official model and weights

The implementation follows the pinned LAVIS InstructBLIP Vicuna-7B configuration at LAVIS revision `05cb670f47296522e24e7727e366b4a997d28582`:

- Visual encoder: LAVIS EVA-CLIP-G, 224-pixel input, 14-pixel patches, width 1408, 39 instantiated transformer blocks.
- Image Q-Former: 12 layers, hidden size 768, 32 Query Tokens, cross-attention in layers 0/2/4/6/8/10, encoder width 1408.
- Projector: `Linear(768, 4096)`.
- Language model: existing frozen `lmsys/vicuna-7b-v1.1`, revision `694d39416b1744d939760f5257c4bb08b2fae400`.

| Artifact | Official source | Local path | Bytes | SHA-256 |
|---|---|---|---:|---|
| EVA-CLIP-G | LAVIS `eva_vit_g.pth` Google Storage URL | `/data/students/Zhang_jinsong/spacebyte/models/pixel_vicuna_official/eva_vit_g.pth` | 2,025,249,237 | `99d2bb36c6b52c94fe6e2e12373afb27de57ae81378c3d8c53bf0e83b0f4275f` |
| InstructBLIP Vicuna-7B trimmed | LAVIS `instruct_blip_vicuna7b_trimmed.pth` Google Storage URL | `/data/students/Zhang_jinsong/spacebyte/models/pixel_vicuna_official/instruct_blip_vicuna7b_trimmed.pth` | 2,266,496,277 | `6b9553383527466c1382259a3fc97b6d643f72b1ecaa37bf7c4711de0059b8ea` |

The fixed X-InstructBLIP `image_qformer.pth` URL returned HTTP 403, so it was not used or replaced by an unofficial mirror. The selected fallback is the official LAVIS InstructBLIP image branch whose projector is directly compatible with the existing 4096-wide Vicuna.

## Input and supervision protocol

Pixel preprocessing reuses `render_record_to_jpeg_bytes` without duplicating any geometry. The source image is aspect-preserving bicubic letterboxed to 224×224, the object is boxed in blue and the subject in red, and PIL encodes the result as unoptimized JPEG Q100. Pixel processing then decodes that exact JPEG byte stream to RGB and applies the official BLIP mean `(0.48145466, 0.4578275, 0.40821073)` and standard deviation `(0.26862954, 0.26130258, 0.27577711)`. A real sample and an automated unit test both confirmed identical sample ID, JPEG bytes, transformed boxes, and image source between the Pixel and Byte paths.

The Vicuna sequence retains the established X-InstructBLIP-compatible layout: `USER:` tokens, `image:` cue, 32 projected image Query Tokens, question plus `ASSISTANT:` prompt, then answer and EOS. Only the single `yes` or `no` token is supervised. Query, cue, prompt, EOS, and padding labels are `-100`; the real run had 1 supervised and 66 ignored positions, and the causal prediction position was exactly one position before the answer label.

## Actual real-weight shapes

| Tensor | Shape |
|---|---|
| `pixel_values` | `[1, 3, 224, 224]` |
| `vision_features` | `[1, 257, 1408]` |
| `vision_attention_mask` | `[1, 257]` |
| `image_query_features` | `[1, 32, 768]` |
| `projected_image_queries` | `[1, 32, 4096]` |
| `vicuna_inputs_embeds` | `[1, 67, 4096]` |
| `vicuna_attention_mask` / `labels` | `[1, 67]` |
| Vicuna `logits` | `[1, 67, 32001]` |

## Audited loading

The loader copies only explicitly mapped tensors with exact shapes and verifies every copied tensor afterward; it does not call an unchecked `strict=False` load.

- EVA-CLIP-G: 511 tensors / 985,894,528 values loaded exactly; model-only 0; shape mismatch 0. The official source checkpoint has 17 unused tensors / 26,694,400 values: block 39, final `norm`, and classification `head`. These are expected source-only tensors because pinned LAVIS constructs the BLIP-2 encoder with 39 blocks and returns pre-head token features.
- Image branch: all 334 tensors / 188,837,888 values loaded exactly; checkpoint-only 0; model-only 0; shape mismatch 0. Mapping is `Qformer.* → qformer.*`, `query_tokens → query_tokens`, `ln_vision.* → vision_layernorm.*`, and `llm_proj.* → llm_projector.*`.
- A strict save/reload of the trainable image bridge returned `missing_keys=[]`, `unexpected_keys=[]`, with every tensor identical.

## Parameters and gradients

| Module | Total parameters | Trainable parameters | Status |
|---|---:|---:|---|
| EVA-CLIP-G visual encoder | 985,894,528 | 0 | frozen |
| Image Q-Former + Query Tokens + visual LayerNorm + projector | 188,837,376 | 188,837,376 | trainable |
| Vicuna-7B | 6,738,423,808 | 0 | frozen |

The real answer-only backward pass produced finite, nonzero gradients for Query Tokens (norm 2.2366), Q-Former self-attention (2.0330), visual LayerNorm (0.9669), and projector (34.4651). Cross-attention key/value gradients were nonzero in all six layers. EVA and Vicuna had no gradients, and frozen-parameter probes were unchanged. Loss was finite (`0.616268`).

## Ablation interfaces and runtime checks

`Normal-Pixel`, post-encoder `Zero-Pixel`, and explicit globally mapped `Shuffled-Pixel` are available. Shuffling requires a fixed-point-free permutation of at least two samples; batch-size-one internal shuffling is rejected. On three distinct train images, the fixed permutation `[2, 0, 1]` changed next-token logits relative to Normal (`max abs 4.878906`); Zero also changed them (`max abs 4.969727`). Two deterministic evaluation forwards matched exactly.

Absolute peak allocated GPU memory was 15.139 GiB after load, 15.520 GiB for inference forward, 15.990 GiB for the loss forward, and 16.051 GiB after backward. No quantization, LoRA, gradient checkpointing, or multiple GPUs were used. End-to-end verification took 107.3 seconds.

## Reproduction and tests

```bash
conda activate /data/students/Zhang_jinsong/spacebyte/envs/spacebyte-qformer
python -m pip install 'timm==0.4.12'
python scripts/verify_pixel_vicuna_integration.py \
  --dataset-root /data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense \
  --eva-checkpoint /data/students/Zhang_jinsong/spacebyte/models/pixel_vicuna_official/eva_vit_g.pth \
  --instructblip-checkpoint /data/students/Zhang_jinsong/spacebyte/models/pixel_vicuna_official/instruct_blip_vicuna7b_trimmed.pth \
  --vicuna-path /data/students/Zhang_jinsong/spacebyte/models/vicuna-7b-v1.1 \
  --qformer-tokenizer /data/students/Zhang_jinsong/spacebyte/models/bert-base-uncased-tokenizer \
  --output /data/students/Zhang_jinsong/spacebyte/result/pixel/pixel_vicuna_integration.json \
  --device cuda:2
python -m pytest -q
```

The full suite passed: **59 tests passed**, including 5 Pixel-specific tests and all existing Byte regressions. The machine-readable report is `/data/students/Zhang_jinsong/spacebyte/result/pixel/pixel_vicuna_integration.json`.

## Known limitations

This verifies wiring, masking, loading, determinism, ablation behavior, strict checkpoint round-trip, and gradient flow only. It does not establish Pixel-only SpatialSense accuracy, generated-answer quality, robustness, or Pixel/Byte complementarity. Pixel-only training with Normal, Zero-Pixel, and full-validation Shuffled-Pixel controls is the appropriate next stage; fusion and gating remain out of scope until that independent signal is demonstrated.
