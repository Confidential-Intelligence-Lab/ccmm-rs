# eBLAS: Encrypted Basic Linear Algebra Subprograms

## Purpose

eBLAS is the linear-algebra layer for `ccmm-rs`.

It separates:

1. **mathematical semantics** — GEMM, GEMV, DOT, AXPY, transpose, batching;
2. **operand privacy** — PP, CP, PC, CC;
3. **execution backend** — reference, direct CKKS, CPMM-equivalent CP,
   matrix-structured CCMM, and future packed/BLAS/accelerator backends.

Applications should depend on eBLAS semantics rather than directly selecting
low-level CKKS/RNS routines.

## Operand modes

- **PP**: plaintext × plaintext
- **CP**: ciphertext × plaintext
- **PC**: plaintext × ciphertext
- **CC**: ciphertext × ciphertext

All four are part of the semantic contract.  An operand mode being defined
does not imply that an execution backend has already been implemented.

At the R3.5a checkpoint:

| Privacy | Supported backend |
|---|---|
| PP | Reference |
| CP | CpDirect |
| PC | none yet |
| CC | CcScalar, CcStructured |

## GEMM semantics

For

```text
A: M x K
B: K x N
```

eBLAS GEMM returns

```text
C: M x N
C[i,j] = sum_k A[i,k] * B[k,j].
```

The current matrix layout is column major:

```text
index = row + column * rows
```

The privacy mode does not change the mathematical result.

## Current encrypted schedules

### CP GEMM

The existing realistic RNS/NTT ciphertext-plaintext path evaluates `MKN`
ciphertext-plaintext products, accumulates each output dot product at product
scale, and performs one rescale per output entry.

```text
relinearizations = 0
rescales          = M*N
```

### Scalar CC GEMM

The scalar baseline independently performs multiply, relinearize, and rescale
for each encrypted scalar product.

```text
relinearizations = M*K*N
rescales          = M*K*N
```

### Structured CCMM

Matrix-structured CCMM accumulates the complete degree-two RLWE dot product
before postprocessing.

```text
relinearizations = M*N
rescales          = M*N
```

R3.4 measured this distinction directly.  eBLAS exposes both schedules as
separate CC backends so that experiments and future dispatch policy remain
explicit.

## Planned operation families

### Level 3

- GEMM
- batched GEMM

### Level 2

- GEMV
- rank-one / outer-product primitives

### Level 1

- ADD
- SCALE
- AXPY
- DOT

### Layout

- transpose
- packing/unpacking
- representation conversion

## Design constraints

- Operation semantics, privacy mode, and execution backend are orthogonal.
- Unsupported privacy/backend combinations fail explicitly.
- No automatic backend selection is introduced until characterization data
  justify a dispatch policy.
- Bootstrapping is outside the eBLAS kernel contract.  Applications and
  higher-level planners decide when depth restoration is required.
- Physical RNS width and accelerator/provider choices remain below the eBLAS
  semantic interface.
