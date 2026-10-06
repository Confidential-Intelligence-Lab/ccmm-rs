# R17a Clear Packed 1K FFT2 Evidence

Date: 2026-10-05

## Architecture

- Image: 1024x1024
- Tile: 64x64
- Logical tile grid: 16x16 = 256 tiles
- Packing: 8x1 vertical
- Tiles per packed vector: 8
- Packed vectors / future ciphertexts: 32
- Slots per packed vector: 32768
- Slot utilization: 100%

## Schedule

- 4 global-row inter-vector DIF stages
- 6 local-row 8-way SIMD DIF stages
- 1 global-column inter-vector DIF stage
- 3 global-column intra-vector DIF stages
- 6 local-column 8-way SIMD DIF stages

Total depth: 20 stages.

The five inter-vector stages contain 16 physical vector-pair
butterflies each, for 80 physical inter-vector butterflies total.

The three intra-vector global-column stages use tile-block offsets
corresponding to 16384, 8192, and 4096 slots.

## Validation

The complete packed physical schedule was reconstructed into the global
DIF physical image and converted once to canonical logical FFT order.

Reference: monolithic fft2_pp over the complete 1024x1024 input.

- Relative L2: 8.079566982475e-15
- Maximum absolute error: 2.518616632596e-9
- Execution time: 89 ms
- Status: PASS

This validates the exact algebra and data layout intended for encrypted
R17 execution.
