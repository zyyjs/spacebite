# SpatialSense Answer-Only Loss Pilot

This comparison was run from a clean BLIP-2 Stage-1 Q-Former initialization for exactly one epoch. ByteFormer and Vicuna remained frozen. The cache, prompt, FP16 precision, micro-batch size 1, gradient accumulation 8, learning rates, and seed match the preceding full pilot. No test data was loaded.

## Objective audit

Each sample has exactly one supervised full-vocabulary causal-LM target: its `yes` or `no` token. Query tokens, prompt tokens, EOS, and padding are `-100`. The target is read from the logit immediately before its label position, confirming the standard one-position causal shift.

For the preceding `bridge_best.pt`, train answer-token NLL was 0.694905 and EOS NLL was 0.000031; valid answer-token NLL was 0.695420 and EOS NLL was 0.000031. The reported two-token means were therefore 0.347468 and 0.347725. The approximately 0.347 loss did **not** come mainly from EOS: EOS was already almost free, while averaging it with the approximately 0.695 answer NLL divided the reported loss nearly in half.

## Results

The epoch-end `logit_yes-logit_no` results were:

| Split/condition | Accuracy | Positive-class F1 | AUROC |
|---|---:|---:|---:|
| Train | 0.5000 | 0.0000 | 0.5316 |
| Valid Normal | 0.5000 | 0.0000 | 0.5040 |
| Valid Zero-Byte | 0.5000 | 0.0000 | 0.4989 |
| Valid Shuffled-Byte | 0.5000 | 0.0000 | 0.5019 |

The full-vocabulary one-token generation at epoch end produced a valid binary token for every sample. Train and valid exact accuracy were both 0.5000 and binary macro-F1 was 0.3333. The model selected `no` for all samples at the zero decision threshold, so positive-class F1 is 0 despite balanced labels. Step-0, midpoint (703), and epoch-end (1405) validation were performed; the selected checkpoint is step 703 by accuracy then AUROC.

Peak allocated GPU memory was 17,257,984,000 bytes (16.07 GiB). The epoch took 1,611.18 seconds; total wall time including the required evaluations was 1,978.54 seconds. The strict best-checkpoint reload had no missing or unexpected keys and all tensors were finite.

Machine-readable results are under `/data/students/Zhang_jinsong/spacebyte/result/byteformer/spatialsense_answer_loss_pilot/`, including `old_bridge_answer_eos_audit.json`, `train/training_report.json`, the three validation snapshots, train metrics, controls, the training curve, and `train/bridge_best.pt`.
