# RBCS Full-Stream Corruption

`src/spacebyte/corruptions/rbcs.py` is the only canonical RBCS implementation
in this project.

- Name: **RBCS Full-Stream Corruption**
- Version: `rbcs_fullstream_v1`
- Source: user-provided experimental code

RBCS operates on the complete JPEG file byte stream. It is not limited to the
entropy-coded scan, so it may change the JPEG header, image-size fields,
quantization tables, Huffman tables, scan data, EOI, and other markers. The
module never repairs the corrupted file or substitutes a decodable image.

## Parameters and processing

- `S`: segment length in bytes. Only `len(data) // S` complete segments are
  eligible; the final incomplete segment is unchanged.
- `P`: probability of selecting each complete segment.
- `p_f`: once per selected segment, probability of choosing bit flip. Otherwise
  that whole segment uses byte loss. This is a segment-level decision, not a
  byte-level decision.
- `q`: probability that each byte in a selected segment is modified.

Selected flip-segment bytes have exactly one uniformly selected bit toggled.
Selected loss-segment byte indices are collected and deleted only after all
segments have been processed. Segment boundaries therefore always refer to the
original stream and never shift after an earlier loss. Byte loss shortens the
stream; flip does not.

For bytes in complete segments, the theoretical total modification rate is
approximately `P × q`. In a Mixed scenario the expected flip fraction is
`P × q × p_f` and the expected deletion fraction is
`P × q × (1 − p_f)`. Rates over the entire file are slightly lower when there
is an unprocessed incomplete tail, and realized finite-sample rates vary with
the seed.

## Canonical scenarios

| Scenario | S | P | p_f | q |
|---|---:|---:|---:|---:|
| Light-Flip | 128 | 0.35 | 1.0 | 0.35 |
| Light-Loss | 128 | 0.35 | 0.0 | 0.35 |
| Light-Mixed | 128 | 0.35 | 0.5 | 0.35 |
| Medium-Flip | 64 | 0.55 | 1.0 | 0.55 |
| Medium-Loss | 64 | 0.55 | 0.0 | 0.55 |
| Medium-Mixed | 64 | 0.55 | 0.5 | 0.55 |
| Heavy-Flip | 32 | 0.65 | 1.0 | 0.65 |
| Heavy-Loss | 32 | 0.65 | 0.0 | 0.65 |
| Heavy-Mixed | 32 | 0.65 | 0.5 | 0.65 |
| Extreme-Flip | 16 | 0.75 | 1.0 | 0.75 |
| Extreme-Loss | 16 | 0.75 | 0.0 | 0.75 |
| Extreme-Mixed | 16 | 0.75 | 0.5 | 0.75 |

These names and values must not be silently changed or reinterpreted.

## Pixel and Byte behavior

Because full-stream corruption can damage structural JPEG fields, PIL may fail
at `verify()` or full pixel decoding. Pixel decoding failure must be recorded
explicitly and must never be replaced with a black image, zero tensor, or the
original image. ByteFormer can still attempt to consume the corrupted raw bytes
because its input does not require JPEG decoding.

## Python API

```python
from spacebyte.corruptions.rbcs import RBCSCorruptor, RBCSParams, SCENARIOS

corrupted = RBCSCorruptor.corrupt(
    intact_bytes,
    SCENARIOS["Heavy-Mixed"],
    seed=20261016,
)

corrupted, statistics = RBCSCorruptor.corrupt_with_stats(
    intact_bytes,
    RBCSParams(S=32, P=0.65, p_f=0.5, q=0.65),
    seed=20261016,
    scenario_name="Heavy-Mixed",
)
```

The implementation accepts `bytes`, `bytearray`, and `memoryview`, always copies
the input, and uses a local `random.Random(seed)` instance without reading or
modifying global RNG state.

## Command line

```bash
python scripts/verify_rbcs.py \
  --input example.jpg \
  --scenario Heavy-Mixed \
  --seed 20261016 \
  --output /data/students/Zhang_jinsong/spacebyte/cache/rbcs/example_heavy_mixed.jpg \
  --report /data/students/Zhang_jinsong/spacebyte/cache/rbcs/example_heavy_mixed.json
```

The report records corruption statistics plus separate PIL `verify()` and full
`load()` outcomes, including exception types. When output paths are omitted,
artifacts default to `/data/students/Zhang_jinsong/spacebyte/cache/rbcs/` and
remain outside Git.
