# ByteFormer → Byte Q-Former → Vicuna-7B integration

## Scope

This branch connects the already verified ByteFormer token adapter and
BLIP-2-initialized Byte Q-Former to a frozen Vicuna-7B language model. It adds
generation, causal language-model loss, and gradient verification only. No
optimizer step or training is performed.

The frozen/trainable boundary is:

```text
JPEG bytes
  → frozen official JPEG-Q100 ByteFormer
  → Byte tokens [B,T,192] + binary Byte mask [B,T]
  → trainable Byte LayerNorm + BLIP-2-initialized Q-Former + 32 Query Tokens
  → Query features [B,32,768]
  → trainable Linear(768,4096)
  → projected Query prefix [B,32,4096]
  → frozen Vicuna-7B-v1.1
  → generated tokens or causal LM loss
```

ByteFormer and Vicuna have `requires_grad=False` and stay in evaluation mode.
The loss path does **not** use `torch.no_grad()`: Vicuna's operations remain in
the autograd graph so gradients can flow from answer loss to the projector,
Q-Former, cross-attention K/V weights, and Query Tokens.

## Pinned model and source protocol

The pinned LAVIS/X-InstructBLIP submodule commit
`05cb670f47296522e24e7727e366b4a997d28582` explicitly requests **Vicuna
v1.1** in its README. This integration uses the official Hugging Face
repository:

```text
repository: lmsys/vicuna-7b-v1.1
revision:   694d39416b1744d939760f5257c4bb08b2fae400
local path: /data/students/Zhang_jinsong/spacebyte/models/vicuna-7b-v1.1
```

Official main weight-file integrity values used by the verifier:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `pytorch_model-00001-of-00002.bin` | 9,976,634,558 | `5ed572be140240b212049a3e791271eed8a04c40bc732c91a4da4b1469db23b1` |
| `pytorch_model-00002-of-00002.bin` | 3,500,315,539 | `d9382da358f0ec38c4fca3bcf1e3e65274ae4c78090e0775b4bb4dea6a518e08` |

The path is supplied at runtime and is not hard-coded into the model class.
Weights and caches are outside Git.

The existing `spacebyte-qformer` environment was missing the runtime required
by `LlamaTokenizer`; `sentencepiece==0.2.1` was therefore installed in that
environment. No quantization or LoRA dependency was added.

The following behavior is reproduced from the pinned
`blip2_vicuna_xinstruct.py` and its `vicuna7b` YAML:

- `LlamaTokenizer(..., use_fast=False, truncation_side="left")`;
- add a new `[PAD]` token rather than aliasing EOS as padding;
- set BOS, EOS, and UNK to `</s>`;
- resize Vicuna's 32,000-row embeddings to 32,001 rows;
- prefix `USER: `;
- cleaned image cue `image: `, with its independently inserted start token
  removed because a prefix already supplies the start token;
- postfix `\nASSISTANT:`;
- Q-Former receives the natural-language question as instruction tokens;
- projected modality Query tokens precede the question/postfix text;
- answer tokens end with EOS;
- the answer tokenizer's independent first start token is removed before
  concatenation;
- loss batches use right text padding, while generation batches use left text
  padding, matching the two pinned source paths;
- causal-LM labels are `-100` for prefix, cue, all 32 Query tokens, question,
  postfix, and padding; only answer/EOS tokens are supervised.

The resulting order is:

```text
[one start token + USER:] [image:] [32 projected Query tokens]
[question + ASSISTANT:] [answer + EOS]
```

Thus, if all non-Query tokens are denoted by `L`, Vicuna receives
`inputs_embeds`, `attention_mask`, and (for loss) `labels` with sequence length
`32 + L`.

### Audited source discrepancy

The pinned X-InstructBLIP training `forward` has two consecutive appends in the
`use_cues and prefix and clean_tokenization` branch, which duplicates the cue
and modality Query tensor. Its generation path appends them once. Duplicating
them would produce 64 instead of the required 32 modality Query tokens. This
integration follows the pinned generation layout and the explicit 32-Query
contract for both loss and generation. The discrepancy is recorded here rather
than silently reproducing it.

## Tokenizer IDs and model configuration

After the X-InstructBLIP tokenizer mutation:

```text
BOS = </s> = 2
EOS = </s> = 2
UNK = </s> = 2
PAD = [PAD] = 32000
tokenizer length = 32001
```

The original checkpoint configuration is LLaMA causal LM, hidden size 4096,
32 layers, 32 attention heads, maximum position length 2048, and vocabulary
size 32000. Hidden size 4096 is checked against the Byte Q-Former projector at
construction time.

## Interfaces

The model path and all checkpoint/tokenizer paths are arguments:

```python
vicuna, llm_tokenizer, report = load_frozen_vicuna(
    vicuna_path,
    device="cuda:1",
    dtype=torch.float16,
)
model = ByteVicuna(
    byteformer_adapter=adapter,
    bridge=bridge,
    vicuna=vicuna,
    llm_tokenizer=llm_tokenizer,
    qformer_tokenizer=qformer_tokenizer,
)
```

Generation is evaluation-only and wrapped in `torch.no_grad()` by the method.
Both the short verification generation and the differentiable loss path use
`use_cache=False`; the loss path is otherwise a normal differentiable
`forward`.

## Verification

Fast contract tests:

```bash
/data/students/Zhang_jinsong/spacebyte/envs/spacebyte-qformer/bin/python \
  -m pytest -q tests/test_byte_vicuna_integration.py
```

Full real-weight verification:

```bash
/data/students/Zhang_jinsong/spacebyte/envs/spacebyte-qformer/bin/python \
  scripts/verify_byte_vicuna_integration.py \
  --vicuna-path /data/students/Zhang_jinsong/spacebyte/models/vicuna-7b-v1.1 \
  --image-dir /data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense/images/flickr \
  --device cuda:3 \
  --output /data/students/Zhang_jinsong/spacebyte/result/byteformer/byte_vicuna_integration.json
```

The full verifier checks real JPEG execution, generation, label masking,
finite causal loss, Byte/text padding invariance, determinism, Query-prefix
participation, gradient flow through every one of the six cross-attention K/V
pairs, frozen-module gradients, embedding scale, and four GPU-memory stages.

Measured values from the final verification are stored in the JSON report at
the path above. No model weight is written or updated by verification.

## Final measured verification

The successful real-weight run used one RTX 4090 (`cuda:3`), Vicuna FP16,
batch size 1 for generation/loss, and no quantization, multi-GPU placement,
gradient checkpointing, optimizer step, or parameter update. The local model
directory contains 13,477,482,052 bytes across the pinned repository files;
both weight-shard hashes match the values above.

The real JPEG path produced:

```text
JPEG bytes             [1,42448]
ByteFormer features    [1,166,192]
Byte attention mask    [1,166]
Q-Former queries       [1,32,768]
projected queries      [1,32,4096]
Vicuna inputs/labels   [1,60,4096] / [1,60]
Vicuna logits          [1,60,32001]
```

The untrained bridge generated the non-empty decoded string `1: 1:`. This is
only a connectivity result; no semantic quality is claimed. Causal LM loss was
`7.0591597557`. In the two-sample label audit, 113 positions were ignored and
7 answer/EOS positions were supervised; prefix, cue, Queries, prompt, and two
padding positions were all correctly set to `-100`.

Changing masked Byte padding content and adding 257 right Byte-padding bytes
both gave exactly zero logit error. Repeating identical input was also bitwise
identical. Adding seven text-padding positions changed the FP16 CUDA matrix
width and produced max/mean valid-logit differences of `0.7734375` and
`0.0258461`; the mean error was `0.61475%` of mean reference magnitude,
cosine similarity was `0.9999695`, and token top-1 agreement was `96.61%`.
The verifier records all five measurements and applies the documented combined
FP16 acceptance bounds. Zeroing the Query prefix caused max/mean differences
of `29.703125`/`2.516474` and only `24.17%` top-1 agreement, demonstrating that
the Byte Query prefix participates in Vicuna computation.

All required gradients were nonzero. Gradient norms were `4.1059` for Query
Tokens, `0.3370` for Byte LayerNorm, `2.4316` for Q-Former self-attention, and
`96.8930` for the projector. Key/value gradient norms were nonzero in all six
cross-attention layers (0, 2, 4, 6, 8, and 10). ByteFormer and Vicuna had no
gradients, and Vicuna had zero trainable parameters.

Projected Byte Queries had mean/std/mean L2 norm
`-0.00116 / 0.28949 / 18.5168`; ordinary Vicuna text embeddings measured
`-0.000016 / 0.01691 / 1.05853`. These values are diagnostic only: no extra
normalization layer was introduced.

Peak allocated GPU memory was:

| Stage | Bytes | GiB |
| --- | ---: | ---: |
| full model loaded | 14,286,691,328 | 13.306 |
| real-JPEG generation | 14,409,175,552 | 13.420 |
| causal-loss forward | 14,757,213,696 | 13.744 |
| backward | 15,143,890,944 | 14.104 |

The BLIP-2 initializer loaded 318 tensors exactly; its 12 known incompatible
cross-attention K/V tensors remained intentionally initialized for encoder
width 192. The official ByteFormer checkpoint continued to load with no
missing or unexpected keys and exact state-dict equality.

Final regression command: `python -m pytest -q tests`; result: **25 passed**
(the pre-existing 19 checks plus 6 Vicuna-integration contract tests).
