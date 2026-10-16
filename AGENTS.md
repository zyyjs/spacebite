# Project agent rules

## RBCS corruption

- The canonical RBCS implementation is `src/spacebyte/corruptions/rbcs.py`.
- Later experiments must not copy, rewrite, or silently modify the 12 scenario parameters.
- RBCS is full-stream corruption; Pixel decode failures must be recorded explicitly.
- Corrupted data, caches, and results belong under `/data` and must not be committed to Git.
