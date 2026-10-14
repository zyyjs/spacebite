# SpatialSense Pixel-only Autonomous Baseline

This experiment trains only the official InstructBLIP image bridge on the
SpatialSense train split and selects checkpoints only on valid. EVA-CLIP-G and
Vicuna-7B-v1.1 remain frozen. The test split is never loaded.

The controller first validates or builds an audited EVA feature cache. Attempt 2
uses deterministic FP32 EVA forward passes and stores the resulting tensors as
FP16 after the original official mixed-FP16 cache showed batch-size-dependent
outliers. Discrete sample/input/weight/preprocessing identity remains exact;
floating comparisons use fixed mean-error, relative-L2, and per-sample cosine
criteria.
then trains for at most five epochs with AdamW (`lr=1e-5`), micro-batch size 1,
gradient accumulation 8, FP16, and gradient clipping at 1.0. Every epoch it
evaluates Normal-Pixel, Zero-Pixel, and five fixed full-validation
Shuffled-Pixel derangements. It handles checkpoint resume, early stopping,
final paired bootstrap statistics, JSON/Markdown reports, signals, and failure
artifacts without an attached AI or terminal session.

Run the required smoke test:

```bash
/data/students/Zhang_jinsong/spacebyte/envs/spacebyte-qformer/bin/python \
  scripts/run_spatialsense_pixel_baseline_autonomous.py \
  --smoke-only --device cuda:2
```

Launch the detached formal experiment:

```bash
bash scripts/launch_spatialsense_pixel_baseline.sh
```

Resume from the complete `last.pt` training state after an interruption:

```bash
nohup setsid /data/students/Zhang_jinsong/spacebyte/envs/spacebyte-qformer/bin/python -u \
  scripts/run_spatialsense_pixel_baseline_autonomous.py \
  --device cuda:2 --resume \
  >> /data/students/Zhang_jinsong/spacebyte/result/pixel/spatialsense_pixel_baseline/train.log \
  2>&1 < /dev/null &
```

Runtime artifacts are written under
`/data/students/Zhang_jinsong/spacebyte/result/pixel/spatialsense_pixel_baseline/`.
The EVA cache is stored under
`/data/students/Zhang_jinsong/spacebyte/cache/spatialsense_pixel_eva_g/`.
Neither location is committed to Git.

Inspect or terminate the job:

```bash
cat /data/students/Zhang_jinsong/spacebyte/result/pixel/spatialsense_pixel_baseline/status.json
tail -f /data/students/Zhang_jinsong/spacebyte/result/pixel/spatialsense_pixel_baseline/train.log
kill -TERM "$(cat /data/students/Zhang_jinsong/spacebyte/result/pixel/spatialsense_pixel_baseline/run.pid)"
```

The successful terminal artifact is `DONE`; failed execution writes `FAILED`
and `failure_report.md`. The final report is
`SPATIALSENSE_PIXEL_BASELINE.md` in the runtime result directory.
