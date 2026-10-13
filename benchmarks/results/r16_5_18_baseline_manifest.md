# FHE-rs R16.5.18 — Benchmark Manifest

## Provenance

- Validation: `82ef7beb224980b815ba58fd35fe379c5abc8156`
- Reproducibility: `ccae7788c17e6ca930d93938a1964ea6562ab948`
- Validation status: PASS
- Independent source compilation: PASS

## Platform

- Processor: Apple M4
- Model: Mac16,1
- Memory: 24 GiB
- Operating system: macOS-27.0.1-arm64-arm-64bit-Mach-O
- Compiler: rustc 1.89.0 (29483883e 2025-08-04)
- Cargo: cargo 1.89.0 (c24e10642 2025-06-23)

## Experiment

- Image: 3840 × 2160
- Selected spectral bins: 4096
- CKKS polynomial degree: 32768
- Horizontal ciphertexts: 135
- Horizontal handoff level: 7
- Final level: 13
- Intermediate decryptions: 0

## Validated results

- Relative L2 error: 8.734300537894e-6
- Maximum absolute error: 4.474216955315e-7
- Total runtime: 42802.938 seconds

## Measurement provenance

The accompanying JSON manifest preserves the raw key-value
measurements extracted from the original validation log and
benchmark record.

## Limitations

- Research parameters; security strength not independently validated.
- Peak memory not established.
- Independent compilation does not constitute a second numerical run.
- Recorded phase timings must not be assumed additive without
  verifying their measurement boundaries.
