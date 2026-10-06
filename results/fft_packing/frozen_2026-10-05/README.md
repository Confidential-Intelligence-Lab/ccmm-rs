# R16 Full-Slot SIMD FFT2 Packing Evidence

Date: 2026-10-05

## Configuration

- CKKS ring degree: 65536
- CKKS slots: 32768
- Logical FFT tile: 64x64 = 4096 complex slots
- Local FFT depth: 12 stages
- Galois key generation: `RnsGaloisKey::generate_with_ntt_rng`
- Validation oracle:
  - repeated clear packed DIF stage algebra
  - final per-tile comparison against `fft2_pp`
- Research parameterization; not security-bearing

## Results

| Packing | Active slots | Utilization | FFT execution | Final packed REL-L2 | Status |
|---|---:|---:|---:|---:|---|
| 2 tiles / CT | 8192 | 25% | 186.289 s | 1.883722e-11 | PASS |
| 4 tiles / CT | 16384 | 50% | 184.964 s | 1.687028e-11 | PASS |
| 8 tiles / CT | 32768 | 100% | 183.787 s | 1.307253e-11 | PASS |

## Key result

Eight independent encrypted 64x64 FFT2 instances execute simultaneously
inside one CKKS ciphertext while occupying all 32768 slots.

Increasing packed FFT count from 2 to 8 does not increase FFT depth,
rotation-key count, or measured local FFT wall-clock time materially.

The 8-way configuration therefore provides approximately 4x useful FFT
throughput relative to the 2-way configuration.

## Numerical stability

The 8-way packed execution shows smooth error growth across all 12 stages:

- stage 1 REL-L2: 3.222324e-12
- stage 12 REL-L2: 1.307253e-11

All eight final logical tiles lie tightly around 1.30e-11 relative L2.

At 8-way packing there are no inactive slots, so
`MULTI_TILE_FFT_INACTIVE_MAX_ABS=0` reflects an empty inactive region,
not a measured zero-leakage region.

## 1K implication

For a 1024x1024 image with 64x64 logical tiles:

- 16 x 16 = 256 logical tiles
- 8 logical tiles per ciphertext
- 32 physical ciphertexts

This becomes the baseline representation for R17 hierarchical packed FFT2.

## Known issue

`RnsGaloisKey::generate_with_distribution_ntt_rng` produced catastrophic
rotation error in the large-N diagnostic and in the initial multi-tile
validator.

The validated R16 correctness path uses:

`RnsGaloisKey::generate_with_ntt_rng`

The distribution-aware/security-bearing Galois-key path must be investigated
and repaired separately before security-bearing claims are made for this
configuration.
