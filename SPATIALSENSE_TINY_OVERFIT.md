# SpatialSense instruction pipeline and tiny-overfit verification

## Scope and outcome

This stage audits the local SpatialSense release, builds a deterministic
candidate-relation instruction pipeline, caches frozen ByteFormer features,
and verifies parameter updates on 36 fixed training annotations. It is an
overfit/connectivity test only, not a validation or test-set result.

The verified training chain is:

```text
raw SpatialSense image + subject/object boxes
  → red subject box + blue object box
  → JPEG Q100 bytes
  → frozen official ByteFormer
  → cached Byte features [T,192] + mask [T]
  → trainable BLIP-2-initialized Byte Q-Former
  → 32 Query features [32,768]
  → trainable Linear(768,4096)
  → frozen Vicuna-7B-v1.1
  → yes/no answer + EOS causal loss
```

The run stopped successfully at optimizer step 50, after two consecutive
evaluations at or above the 90% exact-match criterion. Final fixed-set
exact-match and token accuracy were both 100%.

## Correction to the previous label report

The earlier numbers `113 ignored + 7 supervised = 120` were correct, but they
came from a separate two-sample mixed-length audit with `labels=[2,60]`. The
nearby tensor table showed the one-sample loss path with `labels=[1,60]`, which
made the explanation appear inconsistent. No labels were accidentally
accumulated and the label constructor was not wrong.

The verifier now enforces:

```python
assert labels.shape == attention_mask.shape
assert num_ignored + num_supervised == labels.numel()
```

For every row it also independently tokenizes `answer + EOS` and requires an
exact token-ID match with every non-`-100` label. The real tiny-overfit online
check had `labels=[1,67]`, with 65 ignored tokens and the two supervised IDs
`[694,2]`, decoded as `no</s>`.

## Real data audit

Local source:

```text
/data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense/annotations.json
/data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense/images/flickr
/data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense/images/nyu
```

`annotations.json` is a list of 11,569 image records. Each image record has
`url`, `nsid`, `height`, `width`, `annotations`, and `split`. Each annotation
has `_id`, `predicate`, `subject`, `object`, and an explicit Boolean `label`.
Subject/object dictionaries contain `name`, center `x/y`, and `bbox`.

The true task format is **candidate relation + true/false** (case B), not one
true relation class per sample. Consequently this pipeline generates `yes` or
`no`. It never creates negatives for unannotated relations.

| Split | Image records | Candidate annotations | False | True |
| --- | ---: | ---: | ---: | ---: |
| train | 7,645 | 11,238 | 5,619 | 5,619 |
| valid | 1,553 | 2,638 | 1,319 | 1,319 |
| test | 2,371 | 3,622 | 1,811 | 1,811 |
| total | 11,569 | 17,498 | 8,749 | 8,749 |

Per-relation totals are also exactly balanced within every split:

| Candidate relation | train (F/T) | valid (F/T) | test (F/T) | total |
| --- | ---: | ---: | ---: | ---: |
| above | 388/388 | 99/99 | 144/144 | 1,262 |
| behind | 1,038/1,038 | 187/187 | 254/254 | 2,958 |
| in | 402/402 | 108/108 | 169/169 | 1,358 |
| in front of | 725/725 | 137/137 | 225/225 | 2,174 |
| next to | 496/496 | 123/123 | 162/162 | 1,562 |
| on | 1,479/1,479 | 404/404 | 550/550 | 4,866 |
| to the left of | 350/350 | 74/74 | 89/89 | 1,026 |
| to the right of | 274/274 | 54/54 | 73/73 | 802 |
| under | 467/467 | 133/133 | 145/145 | 1,490 |

There are 4,110 image records with more than one annotation (maximum 12).
The 17,498 annotations cover 17,092 ordered subject/object pairs; 379 pairs
have multiple explicitly judged candidate relations, with a maximum of five.

## Subject/object rendering and JPEG encoding

The raw source images do not contain task boxes. The rendering convention was
read from the archived, previously verified SpatialSense preprocessing code;
none of its classification head, BCE loss, sparse nine-output target, training
script, or checkpoint was reused.

- annotation bbox order: `[y_min, y_max, x_min, x_max]`;
- all 10,440 annotated images match their declared dimensions, so no extra
  annotation-to-decoded-image correction is needed (the implementation still
  handles a mismatch defensively);
- preserve aspect ratio and bicubic-letterbox onto 224×224 gray `(128,128,128)`;
- draw blue object `(0,96,255)` first;
- draw red subject `(255,32,32)` second;
- line width: 3 pixels;
- clamp transformed coordinates independently to `[0,223]`;
- encode the marked RGB image through PIL JPEG with `quality=100`, without
  optimization, matching the official ByteFormer JPEG-Q100 encoding settings.

Nine relation-covering visual checks are in:

```text
/data/students/Zhang_jinsong/spacebyte/result/byteformer/
  spatialsense_tiny_overfit/visualizations
```

Manual inspection of distinct-name examples (for example red bike/blue wall
and red man/blue umbrella) confirmed that subject and object were not swapped.

## Instruction and fixed sample set

Every sample uses one template:

```text
Is the object in the red box {candidate relation} the object in the blue box?
Answer yes or no.
```

Only the canonical answer `yes` or `no` is supervised. Raw left/right phrases
are normalized from `to the left/right of` to `left/right of` in the prompt.
The verified X-InstructBLIP `USER:`, `image:`, 32 Query prefix,
`\nASSISTANT:`, BOS/EOS, padding, and causal-label protocol is unchanged.

The committed manifest uses seed `20261013` and contains 36 unique train
annotations: for each of the nine candidate relations, two explicit false
annotations and two explicit true annotations. The same 36 samples are used
for training and evaluation, by design.

## Frozen ByteFormer feature cache

Cache location:

```text
/data/students/Zhang_jinsong/spacebyte/cache/spatialsense_tiny_overfit
```

Each `<sample_id>.pt` contains FP16 `byte_features=[T,192]`, a binary
`byte_attention_mask=[T]`, sample ID, valid-token count, original marked-JPEG
byte length, feature shape, preprocessing version, and ByteFormer checkpoint
SHA-256. The official checkpoint hash remains:

```text
13646ae929d20f3fb48f0a25e0a945dd59ad070746dbf651c6b2a64856867d82
```

Across 36 records, JPEG streams contain 27,519–52,774 bytes and cached valid
token counts range from 108 to 207. The cache occupies 2,284,984 bytes.
Four deterministic random consistency checks had bitwise-identical masks and
zero online-repeat error. FP32-online versus FP16-cache max error was
`0.0151367`; per-sample mean errors were `0.000763–0.001061`.

A separate pre-training path started from a real 49,484-byte JPEG and produced
`[1,194,192] → [1,32,768] → [1,32,4096]` with finite loss. Its cache max/mean
errors were `0.0077934 / 0.0007630`, confirming the cache did not hide a broken
image-to-byte path.

## Optimization configuration and parameter audit

```text
seed                   20261013
precision              frozen ByteFormer FP32; cache FP16;
                       bridge/Vicuna under FP16 autocast with GradScaler
micro batch size       1
gradient accumulation  8
maximum optimizer steps 500
completed steps        50 (early success)
optimizer              AdamW
gradient clipping      1.0
use_cache              False
```

Explicit name-based groups:

| Group | LR | Tensors | Parameters |
| --- | ---: | ---: | ---: |
| BLIP-2 initialized Q-Former + Query Tokens | 1e-5 | 317 | 172,708,608 |
| six random cross-attention K/V weights + Byte LayerNorm + projector | 1e-4 | 16 | 4,919,680 |
| total | — | 333 | 177,628,288 |

All trainable bridge tensors belonged to exactly one group. ByteFormer and
Vicuna parameter IDs were absent from the optimizer. After training, every
frozen parameter version counter was unchanged, both frozen modules still had
zero trainable parameters and no gradients, and representative Query Tokens,
self-attention, all six random cross-attention key weights, Byte LayerNorm, and
projector parameters all changed by nonzero amounts.

## Overfit results

| Step | Eval loss | Exact match | Token accuracy | Gradient norm | Samples/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4.132359 | 0.00% | 0.00% | — | — |
| 25 | 0.169674 | 91.67% | 95.83% | 5.8026 | 5.19 |
| 50 | 0.00006565 | 100.00% | 100.00% | 0.00494 | 5.21 |

Initially, 34/36 outputs were unparseable and two were `yes`. At step 50 all
36 outputs parsed, with exactly 18 `no` and 18 `yes`. Every relation achieved
4/4 exact matches. The independent evaluation process strictly reloaded the
saved bridge (`missing=[]`, `unexpected=[]`) and reproduced the same loss and
100% metrics.

Core experiment time (initial evaluation through final evaluation) was
84.33 seconds. Full-model load occupied 13.306 GiB and peak allocated GPU
memory was 16.055 GiB on one RTX 4090. No OOM fallback, multi-GPU placement,
gradient checkpointing, quantization, LoRA, or CREMA was used.

Outputs are outside Git:

```text
/data/students/Zhang_jinsong/spacebyte/result/byteformer/
  spatialsense_tiny_overfit/data_audit.json
  spatialsense_tiny_overfit/evaluation.json
  spatialsense_tiny_overfit/train/training_config.json
  spatialsense_tiny_overfit/train/training_log.jsonl
  spatialsense_tiny_overfit/train/training_report.json
  spatialsense_tiny_overfit/train/bridge_final.pt
  spatialsense_tiny_overfit/train/optimizer_final.pt
```

The 710,630,064-byte bridge checkpoint contains only the trainable bridge state
and metadata. The 1,421,310,716-byte optimizer file contains optimizer state;
neither file duplicates frozen ByteFormer or Vicuna weights.

Final regression command:

```bash
CUDA_VISIBLE_DEVICES=1 python -m pytest -q tests
```

Result: **32 passed** (all previous 25 checks plus 7 new pipeline/label/update
checks). The 16 warnings are unchanged dependency deprecation/compatibility
warnings.
