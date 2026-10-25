# R35.8 — Parallel stage-zero evaluation

## Configuration

- Application: UHD encrypted FFT3840 horizontal transform
- Input: 3840 x 2160 grayscale image
- CKKS profile: shared Q16/P2
- Hybrid gadget base log: 32
- Window: 8 of 135 horizontal batches
- Key-generation workers: 4
- Stage-zero evaluation workers: 4
- Rayon workers: 8

## R35.8C — Exact-equivalence validation

Serial and parallel stage-zero evaluation were executed using
identical encrypted inputs and deterministic rotation-key seeds.

- Comparison batches: 2
- Output ciphertexts compared: 4
- Rotation-key occurrences per execution: 103
- Serial execution: 2142.132195 seconds
- Parallel execution: 1794.937876 seconds
- Exact RLWE equality: PASS
- Exact CKKS state equality: PASS

The R35.8C timings include sequential key generation and are
not isolated evaluation-kernel benchmarks.

## R35.8D — Eight-batch production benchmark

| Metric | R35.7A (s) | R35.8D (s) |
|---|---:|---:|
| FFT15 | 179.291145 | 180.856552 |
| Stage-zero keygen | 376.339031 | 370.110734 |
| Stage-zero non-keygen | 2729.346883 | 706.973889 |
| Stage-zero total | 3105.685915 | 1077.084623 |
| Stage 1 keygen | 71.913937 | 71.061361 |
| Stage 1 evaluation | 56.537443 | 52.828888 |
| Stage 2 keygen | 60.982214 | 59.927780 |
| Stage 2 evaluation | 47.382851 | 46.381433 |
| Stage 3 keygen | 54.245975 | 52.544281 |
| Stage 3 evaluation | 41.675967 | 40.762382 |
| Complete window | 3617.718472 | 1581.450167 |

## Results

- Stage-zero non-keygen speedup: 3.86x
- Stage-zero total speedup: 2.88x
- Complete window speedup: 2.29x
- Horizontal batches completed: 8/135
- Production benchmark status: PASS

## Interpretation

The stage-zero implementation parallelizes independent ciphertext
batches while retaining source-major, operand-major accumulation
within each ciphertext.

Prepared public operands and bounded Galois-key residency remain
in use.

Exact ciphertext equivalence was demonstrated for two batches.
The eight-batch benchmark establishes successful production
execution and performance, but does not independently establish
eight-batch exact equivalence.

## Next experiments

R35.9B: evaluate eight stage-zero workers against the four-worker
baseline, subject to available memory.

R35.9C: investigate bounded reuse of rotation keys across
repeated offsets without introducing an unbounded global cache.
