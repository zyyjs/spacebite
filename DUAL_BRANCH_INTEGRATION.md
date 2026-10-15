# Pixel+Byte dual-branch integration

This stage is an inference-only wiring check. It concatenates frozen Pixel Query
Tokens before frozen Byte Query Tokens, each with fixed scale `1.0`, and supplies
the resulting prefix to the existing frozen Vicuna-7B-v1.1 model. It does not
create an optimizer, run backward, update parameters, add a gate, use LoRA, or
read the SpatialSense test split.

The autonomous verifier uses three distinct real train/valid samples, checks the
online encoders and cache path, compares each branch with its independent model
wrapper, evaluates all six wiring conditions, audits answer-only labels, and
writes either `DONE` or `FAILED` before exiting. The Pixel path reuses the
verified FP32-forward/FP16-storage EVA cache. The old Byte cache is deliberately
not reused because the final Byte checkpoint changed ByteFormer blocks 10/11 and
`post_transformer_norm`; only the selected final-encoder outputs are cached in
memory for this check.

Launch on the required device:

```bash
SPACEBYTE_DEVICE=cuda:1 bash scripts/launch_dual_branch_integration.sh
```

Artifacts are written outside Git:

```text
/data/students/Zhang_jinsong/spacebyte/result/fusion/dual_branch_integration/
```

The final Markdown report is `DUAL_BRANCH_INTEGRATION.md` inside that result
directory. Checkpoints, caches, JSON, logs, and result reports must not be
committed.
