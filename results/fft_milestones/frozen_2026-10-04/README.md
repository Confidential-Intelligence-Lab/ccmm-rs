# Exact Hierarchical Encrypted FFT2 — 2026-10-04 Evidence

All executed configurations use 64x64 logical leaves and polynomial
ring degree N=65536 for exact global execution.

## 128x128

- logical elements: 16,384
- encrypted tiles: 4
- exact FFT levels: 14
- encrypted FFT2: 196.017 s
- throughput: 83.584587 elements/s
- encryption: 0.841 s
- end-to-end: 271.210 s
- relative L2: 1.998011797668e-11
- max absolute error: 2.431848272342e-7
- status: PASS

## 256x256

- logical elements: 65,536
- encrypted tiles: 16
- exact FFT levels: 16
- chain limbs: 18
- encrypted FFT2: 431.875 s
- throughput: 151.747612 elements/s
- encryption: 3.422 s
- end-to-end: 495.740 s
- relative L2: 2.025624607155e-11
- max absolute error: 9.729009753097e-7
- status: PASS

Pre-optimization encrypted FFT2 baseline: 1249.684 s.
Executor speedup: approximately 2.89x.

## 512x512

- logical elements: 262,144
- encrypted tiles: 64
- local stages: 12
- cross-tile stages: 6
- exact FFT levels: 18
- chain limbs: 20
- total Q: 1100 bits
- terminal guard limbs: 2
- encrypted FFT2: 1727.721 s
- throughput: 151.728201 elements/s
- encryption: 14.925 s
- Galois-key generation: 67.771 s
- decrypt/decode: 6.394 s
- end-to-end: 1816.819 s
- relative L2: 2.056933395569e-11
- max absolute error: 3.891851971045e-6
- inactive max absolute: 1.046141462651e-7
- status: PASS

## Scaling observation

256x256 -> 512x512 increases logical elements by 4x.

Encrypted FFT2:
431.875 s -> 1727.721 s

Throughput:
151.747612 -> 151.728201 elements/s

Relative L2:
2.025624607155e-11 -> 2.056933395569e-11

The measured executor therefore exhibits nearly constant throughput across
this scaling step while relative numerical error remains approximately
2e-11.

## Planning ladder

With two terminal guard limbs:

| Image | 64x64 tiles | Exact depth | Chain limbs |
|---|---:|---:|---:|
| 256x256 | 16 | 16 | 18 |
| 512x512 | 64 | 18 | 20 |
| 1024x1024 | 256 | 20 | 22 |
| 2048x2048 | 1024 | 22 | 24 |
| 4096x4096 | 4096 | 24 | 26 |

The complete 26-limb N=65536 pool has a 1430-bit aggregate modulus.

Planning through 4096x4096 is validated. Exact encrypted execution is
validated through 512x512.
