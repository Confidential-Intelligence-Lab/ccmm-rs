# FHE-rs fuzzing

Fuzzing is a deep-assurance activity and is intentionally separate from the stable Rust development baseline.

Normal development and CI use the pinned stable toolchain in `rust-toolchain.toml`.

LibFuzzer-based fuzzing requires nightly Rust and should be invoked explicitly with `+nightly`; do not change the repository-wide default toolchain.

## Targets

### `fft1_shape`

Exercises valid radix-2 FFT1 shape construction and structural invariants.

### `fft2_shape`

Exercises square and rectangular radix-2 FFT2 shape construction, stage counts, element counts, and butterfly counts.

### `packed_fft2_stage`

Exercises valid packed FFT2 DIF stage construction across row and column axes, forward/inverse directions, spans, and strided column rotations.

## Build

```bash
cargo +nightly fuzz build fft1_shape
cargo +nightly fuzz build fft2_shape
cargo +nightly fuzz build packed_fft2_stage
```

## Smoke campaigns

```bash
cargo +nightly fuzz run fft1_shape -- -max_total_time=15 -timeout=5
cargo +nightly fuzz run fft2_shape -- -max_total_time=15 -timeout=5
cargo +nightly fuzz run packed_fft2_stage -- -max_total_time=15 -timeout=5
```

Longer campaigns belong in deep/nightly assurance rather than normal PR CI.

Generated corpus, crash artifacts, coverage data, and fuzz build output are not committed by default. Any input that exposes a real defect should instead be converted into a deterministic regression test or deliberately curated seed.
