# R17c Packed Encrypted 1024x1024 FFT2

Date: 2026-10-05

## Result

A complete 1024x1024 FFT2 was evaluated over encrypted CKKS data using
hierarchical 8-way SIMD tile packing.

Status: PASS

## Representation

- Image: 1024x1024
- Logical elements: 1,048,576
- Logical tile: 64x64
- Logical tile grid: 16x16
- Logical tiles: 256
- Packing: 8x1 vertical
- Tiles per ciphertext: 8
- Physical ciphertexts: 32
- CKKS slots: 32768
- Slot utilization: 100%
- Ring degree: 65536

## Depth

- Top limbs: 22
- FFT levels: 20
- Final limbs: 2

Schedule:

- levels 0-3: global-row inter-ciphertext DIF
- levels 4-9: local-row 8-way SIMD DIF
- level 10: global-column inter-ciphertext DIF
- levels 11-13: global-column tile-block intra-ciphertext DIF
- levels 14-19: local-column 8-way SIMD DIF

## Galois keys

- Keyed levels: 15
- Distinct level-specific Galois keys: 29

The 16384-slot half-vector rotation uses one shared left/right exponent.

## Correctness

Checkpoint relative L2:

- level 4:  2.835398402583e-12
- level 10: 7.998379065481e-12
- level 11: 8.333992727937e-12
- level 14: 1.125218223756e-11
- level 20: 1.627499117330e-11

Final:

- relative L2: 1.627923819358e-11
- maximum absolute error: 9.799912846418e-6
- validation tolerance: 5.0e-3
- status: PASS

## Performance

- encryption: 16185 ms
- Galois key generation: 87190 ms
- encrypted FFT2: 1074923 ms
- decryption: 3283 ms
- total: 1185320 ms
- FFT throughput: 975.489407 image elements/s

## Conclusion

The R17 architecture demonstrates an exact hierarchical encrypted
1024x1024 FFT2 using only 32 fully occupied CKKS ciphertexts.

Eight logical 64x64 FFT tiles execute simultaneously in each ciphertext.
The packed execution preserves the complete 20-level DIF dependency path
and reconstructs the canonical 1024x1024 FFT with relative L2 error of
approximately 1.63e-11.

The validated Galois-key path uses `generate_with_ntt_rng`.
The separate distribution-aware Galois-key generation issue remains an
independent infrastructure item and must be resolved before making
security-bearing claims for this configuration.
