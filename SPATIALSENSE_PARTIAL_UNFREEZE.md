# SpatialSense ByteFormer Partial-Unfreeze Pilot

The experiment loaded the answer-only pilot's best bridge and recomputed ByteFormer features online from the red/blue-box JPEG-Q100 byte streams. It did not read the old feature cache or the test split. Vicuna, ByteFormer embeddings, Conv1D, every TokenMerging module, blocks 0–9, and the classifier remained frozen.

Runtime inspection resolved the trainable ByteFormer modules as `transformer.10` (444,864 parameters), `transformer.11` (444,864), and `post_transformer_norm` (384), totalling 890,112 parameters. The 20-step smoke test confirmed finite nonzero gradients and updates in every intended optimizer group; all frozen parameter version counters and gradients remained unchanged. Formal training rebuilt the model from the original input checkpoint after the smoke test.

| Epoch | Condition | Accuracy | Macro-F1 | AUROC |
|---:|---|---:|---:|---:|
| 1 | Normal | 0.5008 | 0.3357 | 0.5196 |
| 1 | Zero-Byte | 0.5000 | 0.3333 | 0.4989 |
| 1 | Shuffled-Byte | 0.5011 | 0.3359 | 0.5065 |
| 2 | Normal | 0.5034 | 0.3578 | 0.5174 |
| 2 | Zero-Byte | 0.5000 | 0.3333 | 0.4989 |
| 2 | Shuffled-Byte | 0.5023 | 0.3596 | 0.4997 |
| 3 | Normal | 0.5155 | 0.5155 | 0.5270 |
| 3 | Zero-Byte | 0.5000 | 0.3333 | 0.4989 |
| 3 | Shuffled-Byte | 0.5061 | 0.5061 | 0.4996 |

Normal exceeded the stronger control in AUROC by 1.31, 1.77, and 2.74 percentage points across epochs. The threshold was reached at epoch 3, but not sustained across all three epochs. Peak allocated GPU memory was 16.09 GiB. Training compute took 1:27:50 and total runtime including smoke tests, model reloads, evaluations, and checkpoint writes was 1:48:12.

The selected epoch-3 checkpoint is `/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_byteformer_partial_unfreeze/partial_best.pt`. Its bridge reload is strict with no missing or unexpected keys; all 26 partial ByteFormer tensors reload exactly and all stored floating-point tensors are finite.
