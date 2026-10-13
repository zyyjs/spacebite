# SpatialSense full-cache one-epoch pilot

## Scope and task

This branch implements **Byte-stream-conditioned spatial relation
verification**.  Each example consists of a JPEG byte stream whose subject is
marked by a red box and object by a blue box, plus one explicitly annotated
candidate relation.  The target is exactly `yes` or `no`.  It is not a
nine-class task, and an unannotated relation is never synthesized as a
negative example.

The run is deliberately limited to one pilot epoch.  ByteFormer and Vicuna-7B
remain frozen; Query Tokens, Byte LayerNorm, the complete Q-Former (including
the six Byte-width cross-attention K/V projections), and the LLM projector are
trainable.  The test split is cached for future work but is not loaded by the
training program and is not evaluated here.

## Leakage audit

The audit is run before cache generation:

```bash
PYTHONPATH=src /data/students/Zhang_jinsong/spacebyte/envs/spacebyte-qformer/bin/python \
  scripts/audit_spatialsense_splits.py \
  --dataset-root /data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense \
  --output /data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_full_pilot/split_leakage_audit.json
```

The official counts are train 11,238, valid 2,638, and test 3,622.  There are
17,092 unique processed visual inputs for 17,498 annotations.  All pairwise
split overlaps are zero for sample ID, exact annotation tuple, raw image path,
raw image SHA-256, and processed marked-JPEG SHA-256.  No split contains a
duplicate exact annotation.  There are 379 within-split groups (785 samples)
where the same ordered subject/object boxes have different annotated candidate
relations; these are legitimate independent judgements and are retained.

## Content-addressed cache

The visual cache identity hashes the raw image SHA-256, both bboxes, complete
letterbox and red/blue-box configuration, JPEG quality and encoding version,
and official ByteFormer checkpoint SHA-256.  One FP16 `[T,192]` feature tensor
and binary `[T]` mask are stored per unique visual key.  Split JSONL manifests
retain one row per annotation, so deduplication never loses the relation or
label mapping.  Writes use a temporary file plus `fsync` and atomic rename;
existing files are structurally validated before being resumed.

```bash
PYTHONPATH=src /data/students/Zhang_jinsong/spacebyte/envs/spacebyte-qformer/bin/python \
  scripts/cache_spatialsense_full.py \
  --dataset-root /data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense \
  --cache-root /data/students/Zhang_jinsong/spacebyte/cache/spatialsense_full \
  --device cuda:0 --batch-size 8 --consistency-samples-per-split 10
```

The final cache size, runtime, and online-FP32 versus cached-FP16 errors are
recorded in the external `cache_report.json` and summarized below after the
run.

## Evaluation definitions

Greedy evaluation deterministically generates a short answer and accepts only
a complete `yes`, `no`, `yes.`, or `no.` response (case-insensitive); every
other generation is `invalid`.  It reports exact match, per-relation accuracy,
per-class precision/recall/F1, macro F1, invalid rate, and a 2-by-3 confusion
matrix.

Candidate likelihood evaluates both supervised sequences independently:

```text
log P(yes + EOS | byte, prompt)
log P(no  + EOS | byte, prompt)
```

The score is the difference between their mean token log probabilities.  Mean
normalization is explicit so unequal tokenization lengths cannot favor the
shorter answer; with the pinned tokenizer both candidate sequences have two
supervised tokens.  Metrics are accuracy, balanced accuracy, precision,
recall, F1, tie-aware AUROC, and per-relation accuracy/AUROC.

## Fixed pilot configuration

| Item | Value |
|---|---:|
| epochs | 1 |
| micro batch | 1 |
| gradient accumulation | 8 |
| effective batch | 8 |
| optimizer steps | 1,405 |
| precision | FP16 autocast + GradScaler |
| initial loss scale | 1,024 (dynamic) |
| optimizer | AdamW |
| BLIP-2-initialized LR | 1e-5 |
| new Byte K/V, Byte LN, projector LR | 1e-4 |
| warmup | 100 optimizer steps |
| post-warmup schedule | cosine decay |
| gradient clipping | 1.0 |
| seed | 20261013 |

Full validation is performed at step 0, step 703, and step 1,405.  Best
checkpoint selection uses validation candidate-likelihood accuracy and then
AUROC as the tie-breaker.  No test sample participates in training,
validation, hyperparameter choice, or checkpoint choice.

```bash
PYTHONPATH=src /data/students/Zhang_jinsong/spacebyte/envs/spacebyte-qformer/bin/python \
  scripts/train_spatialsense_full.py \
  --cache-root /data/students/Zhang_jinsong/spacebyte/cache/spatialsense_full \
  --train-manifest /data/students/Zhang_jinsong/spacebyte/cache/spatialsense_full/manifests/train.jsonl \
  --valid-manifest /data/students/Zhang_jinsong/spacebyte/cache/spatialsense_full/manifests/valid.jsonl \
  --vicuna-path /data/students/Zhang_jinsong/spacebyte/models/vicuna-7b-v1.1 \
  --output-dir /data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_full_pilot/train \
  --device cuda:0 --eval-batch-size 4
```

## Run results

### Cache

| Measurement | Result |
|---|---:|
| annotation mappings | 17,498 |
| unique visual cache files | 17,092 |
| mappings deduplicated | 406 |
| validated cache files | 17,092 / 17,092 |
| cache size | 1,077,194,200 bytes (about 1.00 GiB) |
| generation and validation time | 945.84 s |
| online checks | 10 per split, 30 total |
| largest FP32-online vs FP16-cache absolute error | 0.0155411 |
| mean of the 30 per-sample mean absolute errors | 0.000894054 |
| exact masks | 30 / 30 |
| peak cache-generation GPU memory | 273,017,344 bytes |

Every sample ID is globally unique and every annotation maps to one validated
visual key.  The train and valid manifest SHA-256 values used by the training
run are respectively
`3a95c09bdfc188f93464a3829fc63e74af74381533f5f5cc7dd0337c6bb987c6`
and
`3232075c17bbdac9e41ad89a25b7eb024e54f7ee5129b6a190bb616210fd50ce`.

### Answer tokens and causal labels

| Answer | Raw IDs including independent start | Supervised answer + EOS | Tokens | Decoded |
|---|---|---|---|---|
| yes | `[2, 4874, 2]` | `[4874, 2]` | `▁yes`, `</s>` | ` yes</s>` |
| no | `[2, 694, 2]` | `[694, 2]` | `▁no`, `</s>` | ` no</s>` |

Both answers therefore contain a leading SentencePiece space and exactly two
supervised tokens.  A two-row runtime label audit found 132 ignored and four
supervised positions out of 136 total; the non-ignored IDs exactly reconstructed
the requested answer plus EOS.

### Full-validation trajectory

| Step | Greedy exact | Invalid | Greedy macro F1 | Likelihood accuracy | Balanced accuracy | Likelihood F1 | AUROC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | 100.00% | 0.0000 | 50.00% | 50.00% | 0.6667 | 0.4774 |
| 703 | 50.00% | 0.00% | 0.3333 | 50.00% | 50.00% | 0.6667 | 0.5089 |
| 1,405 | 50.00% | 0.00% | 0.3333 | 50.00% | 50.00% | 0.0000 | 0.5159 |

At step 0 the untrained bridge generated longer descriptions, so the strict
parser marked every response invalid.  By step 703 it generated a single
binary class for all examples; at step 1,405 it generated the other single
class for all examples.  Thus 50% accuracy is a consequence of the exactly
balanced validation labels, not evidence of useful discrimination.  The best
checkpoint is step 1,405: likelihood accuracy ties at 50%, and AUROC 0.5159 is
the best tie-break value.

### Byte reliance ablations

| Condition | Greedy exact | Greedy macro F1 | Likelihood accuracy | Likelihood F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| normal Byte | 50.00% | 0.3333 | 50.00% | 0.0000 | 0.5159 |
| Zero-Byte | 50.00% | 0.3333 | 50.00% | 0.0000 | 0.5007 |
| Shuffled-Byte | 50.00% | 0.3333 | 50.00% | 0.0000 | 0.5118 |

Normal likelihood accuracy exceeds neither control.  Its AUROC is only 0.0153
above Zero-Byte and 0.0041 above Shuffled-Byte.  This pilot therefore does not
demonstrate that the model learned to use Byte visual information; its binary
decisions are dominated by relation-text and yes/no priors.

Per-relation likelihood results at epoch end are:

| Relation | Samples | Normal acc./AUROC | Zero acc./AUROC | Shuffled acc./AUROC |
|---|---:|---:|---:|---:|
| above | 198 | 50.0% / 0.5419 | 50.0% / 0.5042 | 50.0% / 0.4590 |
| behind | 374 | 50.0% / 0.5080 | 50.0% / 0.5252 | 50.0% / 0.5164 |
| in | 216 | 50.0% / 0.5305 | 50.0% / 0.4505 | 50.0% / 0.5259 |
| in front of | 274 | 50.0% / 0.4750 | 50.0% / 0.5000 | 50.0% / 0.5462 |
| next to | 246 | 50.0% / 0.5080 | 50.0% / 0.4899 | 50.0% / 0.5119 |
| on | 808 | 50.0% / 0.5406 | 50.0% / 0.5103 | 50.0% / 0.5042 |
| to the left of | 148 | 50.0% / 0.4429 | 50.0% / 0.4310 | 50.0% / 0.5114 |
| to the right of | 108 | 50.0% / 0.5976 | 50.0% / 0.4630 | 50.0% / 0.5430 |
| under | 266 | 50.0% / 0.5216 | 50.0% / 0.4990 | 50.0% / 0.5332 |

### Stability and resources

The first attempt stopped before its first optimizer update because the default
loss scale 65,536 overflowed the gradients.  A no-update diagnostic of the same
first eight samples established that scale 1,024 produced finite nonzero
gradients, after which the model was restarted from the untouched BLIP-2
initialization.  The authorized run then completed all 11,238 micro steps and
1,405 optimizer steps with zero NaN/Inf events.

Loss fell from 4.5405 at the first optimizer step to 0.3811 at the last; the
first-100 and last-100 step mean losses were 0.7568 and 0.3477.  The unscaled
total gradient norm ranged from 0.0977 to 102.7364 before clipping.  At the
last step, cross-attention K/V, projector and Query Token gradient norms were
0.0392, 1.4696 and 0.0175, all nonzero.  Mean logged throughput was 6.58
samples/s.

| Resource/time | Result |
|---|---:|
| model memory after load | 14,286,691,328 bytes |
| peak GPU memory | 17,251,615,232 bytes (about 16.07 GiB) |
| pure training compute (validation excluded) | 1,757.09 s |
| epoch wall time (scheduled validation included) | 2,213.91 s |
| total pilot including final ablations | 2,676.25 s |
| multi-GPU / checkpointing / quantization | no / no / no |

ByteFormer and Vicuna each retained zero trainable parameters and no gradients.
The best bridge was independently reconstructed and loaded with `strict=True`:
334 tensors, 177,628,800 parameters, no missing or unexpected keys, and all
floating-point values finite.

Raw machine-readable artifacts live under
`/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_full_pilot/`
and are intentionally excluded from Git.  No metric was computed on the test
split.

## Tests

The dedicated 13 tests cover split isolation, complete visual identity, cache
mapping and corruption detection, answer tokenization, causal likelihood,
strict generation parsing, Zero-Byte scaling, fixed derangement, optimizer
data isolation, strict checkpoint reload/selection, and scheduler boundaries.
The prior 32 regression tests are also run unchanged; the final suite passed
45/45.
