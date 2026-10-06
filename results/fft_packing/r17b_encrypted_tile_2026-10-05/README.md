# R17b Encrypted Packed Tile-Block DIF Evidence

Date: 2026-10-05

## Purpose

Validate the three new encrypted intra-ciphertext global-column stages
required by the 8x1 packed 1024x1024 FFT2 schedule.

## Configuration

- CKKS ring degree: 65536
- CKKS slots: 32768
- Tile size: 64x64 = 4096 slots
- Packed tiles per ciphertext: 8
- Slot utilization: 100%
- Depth-aware profile: 22 top-level limbs
- Required FFT depth: 20
- Terminal guard limbs: 2

The test begins at execution level 11, matching the complete R17 1K
schedule after:

- 4 global-row stages
- 6 local-row stages
- 1 global-column inter-ciphertext stage

## Encrypted stages

| Input level | Tile span | Slot rotation | Input limbs | Output limbs | Relative L2 |
|---|---:|---:|---:|---:|---:|
| 11 | 8 | 16384 | 11 | 10 | 3.436714e-12 |
| 12 | 4 | 8192 | 10 | 9 | 4.440503e-12 |
| 13 | 2 | 4096 | 9 | 8 | 5.308105e-12 |

Final maximum absolute error: 3.480242e-11.

Status: PASS.

## Galois-key observation

Rotation by 16384 is a half-slot rotation for a 32768-slot CKKS vector.
Left and right therefore have the same Galois exponent:

65537

Only one evaluation key is required at level 11.

The remaining rotations require two exponents each, so the three new
R17 tile-block stages require five distinct level-specific Galois keys.

## Conclusion

The encrypted tile-block DIF primitive required for levels 11-13 of
the packed 1K FFT2 schedule is validated at the exact RNS widths and
execution levels used by the intended 1024x1024 computation.
