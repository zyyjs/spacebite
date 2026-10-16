# SpatialSense RBCS decode pilot

This CPU-only pilot selects 90 deterministic SpatialSense validation samples,
renders the existing red-subject/blue-object 224×224 JPEG Q100 representation,
and evaluates Clean plus all canonical RBCS scenarios for JPEG decodability.
It imports the only normative implementation from
`spacebyte.corruptions.rbcs`; it does not load any model or read the test split.

Run:

```bash
python scripts/run_rbcs_decode_pilot.py \
  --dataset-root /data/students/Zhang_jinsong/spacebyte/dataset/SpatialSense \
  --output-dir /data/students/Zhang_jinsong/spacebyte/result/rbcs/decode_pilot \
  --global-seed 20261016
```

The output directory contains the manifest and its SHA-256, one JSONL row for
every Clean/corrupted stream, per-scenario/repeat CSV statistics, exception
counts, representative byte streams, and the final Markdown report. These
runtime artifacts belong under `/data` and must not be committed.
