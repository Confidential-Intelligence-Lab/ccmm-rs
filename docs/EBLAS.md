# eBLAS: Encrypted Basic Linear Algebra Subprograms

## Purpose

eBLAS is the application-facing linear-algebra layer of the FHE-rs stack
currently developed inside `ccmm-rs`. Its central separation is:

```text
operation semantics != operand privacy != execution backend
```

Applications specify the mathematical operation and privacy relationship.
Concrete cryptographic schedules remain explicit and reproducible, while a
policy layer may select among characterized backends.

## Public and Encrypted Operands

- **PP** — both operands are public/plaintext;
- **CP** — the left operand is encrypted and the right operand is plaintext;
- **PC** — the left operand is plaintext and the right operand is encrypted;
- **CC** — both operands are encrypted.

All four modes are implemented for the applicable current GEMM paths.

## Operations

**Level 1:** ADD, SCALE, AXPY, DOT. ADD preserves level. SCALE consumes one
level while restoring the incoming scale. AXPY consumes one level and aligns
its second operand through the tested modulus-switch path. No relinearization
is required.

**Level 2:** GEMV. GEMV and DOT reduce to validated GEMM machinery.

**Level 3:** GEMM, batched GEMM, and tensor GEMM mapping. For `A: M x K` and
`B: K x N`, `C[i,j] = sum_k A[i,k] * B[k,j]`. Current matrices are
column-major: `index = row + column * rows`.

## Concrete GEMM backends

| Backend | Privacy | Products | Adds | Relin | Rescale |
|---|---|---:|---:|---:|---:|
| Reference | PP | MKN | MN(K-1) | 0 | 0 |
| CpDirect | CP | MKN | MN(K-1) | 0 | MN |
| CcScalar | CC | MKN | MN(K-1) | MKN | MKN |
| CcStructured | CC | MKN | MN(K-1) | MN | MN |

`CpDirect` accumulates ciphertext-plaintext products before one output rescale.
This is CPMM-equivalent at the algebraic level.

`CcScalar` independently completes each scalar `ct x ct -> relin -> rescale`
product and is retained as a reproducible baseline. Current terminology does
not call this scalar baseline CCMM.

`CcStructured` accumulates unrelinearized degree-2 RLWE tensor products across
K before one relinearization and rescale per output. R3.4/R3.5 measured
2.3–3.45x kernel speedup over the scalar CC baseline across the tested
frontier with 4–16x fewer relinearizations/rescales. The count reduction is an
algorithmic property of these schedules; runtime speedups are measurements.

## PC through transpose

PC reuses CP through `A_plain * B_enc = (B_enc^T * A_plain^T)^T`. In the
current scalar-per-entry representation transpose is an object/data
permutation: no CKKS automorphism, rotation, key switch, rescale, or level is
consumed. R3.5h measured about 0.30–0.46% PC overhead for tested shapes. Do
not generalize this to future packed layouts.

## Measurement-driven CC policy

R3.5i freezes the current policy:

```text
K = 1  -> CcScalar
K >= 2 -> CcStructured
```

At K=1 the schedules have the same postprocessing count; scalar is the
canonical/simple baseline, not a claimed measured winner. For K>=2 the choice
is evidence-backed by the current research-4096 characterization. Explicit
backend selection remains available. The policy is implementation/profile
specific, not a theorem for packed, GPU, FPGA, or future-scheme backends.

## Batching and tensor mappings

Batched GEMM defines independent `C_b = A_b * B_b` products with no
broadcasting or cross-batch reduction. Current batching delegates each batch
to validated GEMM: it is **semantic aggregation, not acceleration**. R3.5h
observed approximately linear total latency and stable latency per batch.

The tensor contract deliberately distinguishes `tensor semantics != matrix
batching != CKKS SIMD packing`. Rank-3 tensor GEMM lowers to batched GEMM and
supports PP/CP/PC/CC without introducing new cryptographic arithmetic.

## Accounting and application rule

`GemmOperationCount::for_backend(spec, backend)` is authoritative for
high-level schedule accounting. It intentionally excludes lower-level NTT,
RNS, modular, and memory operations.

New linear-algebra applications should call eBLAS rather than CPMM/CCMM
implementation functions directly unless the experiment explicitly studies a
backend.

## Current non-goals

- bootstrapping is not implemented;
- integer/discrete CKKS is not implemented;
- batching is not yet SIMD/parallel acceleration;
- no GPU/FPGA eBLAS backend is currently claimed;
- future packed representations may change transpose/backend costs.
