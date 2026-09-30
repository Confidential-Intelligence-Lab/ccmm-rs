# Batch CPMM and CCMM in FHE-rs

## Capability Reproduction and Architectural Integration

FHE-rs includes an independent native-Rust implementation of the Batch CPMM and CCMM constructions from *Fast Batch Matrix Multiplication in Ciphertexts* (CRYPTO 2026).

The goal of this implementation is not to claim superiority over the reference artifact. Instead, it demonstrates that these structured encrypted matrix-multiplication constructions can be reproduced faithfully inside a broader, modern homomorphic-encryption research library.

FHE-rs is designed as a composable research platform rather than as an implementation of a single encrypted-computing algorithm. Its architecture brings together the components commonly needed to develop, reproduce, and evaluate contemporary homomorphic-encryption techniques under shared representations and interfaces.

---

## FHE-rs Architecture

```text
                         FHE-rs

+------------------------------------------------------------------+
| Applications / Research Workloads                                |
| private AI | secure analytics | tensors | multi-party workflows  |
+-------------------------------+----------------------------------+
                                |
+-------------------------------v----------------------------------+
| eBLAS / Encrypted Linear Algebra                                 |
| GEMM | GEMV | DOT | AXPY | batched GEMM | tensor mappings       |
| public/public | encrypted/public | public/encrypted | enc/enc    |
+-------------------------------+----------------------------------+
                                |
              +-----------------+-----------------+
              |                                   |
+-------------v----------------+    +-------------v----------------+
| General HE Evaluation        |    | Structured Algorithms        |
| CKKS arithmetic              |    | Batch CPMM / Batch CCMM      |
| rotations                    |    | SinC batching                |
| relinearization              |    | RingSwitch                   |
| rescaling                    |    | C-MT                         |
| modulus switching            |    | structured polynomial MM     |
+-------------+----------------+    +-------------+----------------+
              |                                   |
              +-----------------+-----------------+
                                |
+-------------------------------v----------------------------------+
| Cryptographic Infrastructure                                   |
| RLWE ciphertexts | evaluation keys | Galois keys | key switching |
| RNS decomposition | modulus chains | basis transitions           |
+-------------------------------+----------------------------------+
                                |
+-------------------------------v----------------------------------+
| Arithmetic / Execution                                          |
| RNS | NTT | modular arithmetic | 32/64/128-bit backends         |
| reference paths | optimized paths | backend policies             |
+------------------------------------------------------------------+
```

The breadth of this stack is intentional. It provides a common experimental environment in which established techniques can be reproduced, new constructions can be introduced, and alternative representations and execution strategies can be evaluated without rebuilding the surrounding cryptographic infrastructure for every experiment.

Within this architecture, Batch CPMM and CCMM are structured matrix-multiplication algorithms rather than standalone application-specific implementations. They reuse the library's RLWE ciphertext representation, RNS and NTT arithmetic, modulus-chain machinery, evaluation keys, key switching, relinearization, rescaling, and CKKS infrastructure.

At the upper layers, eBLAS separates encrypted linear-algebra operations from their cryptographic realization. This allows structured techniques such as CPMM and CCMM, general CKKS evaluation paths, and future execution backends to be studied behind common workload-level abstractions.

---

## Batch CPMM and CCMM Reproduction

The implementation reproduces the reference artifact's 64x64 and 128x128 matrix configurations using a degree-8192 large ring.

The reproduced execution paths exercise the principal mechanisms required by the constructions, including:

- SinC batching;
- coefficient-preserving ring switching;
- C-MT;
- structured RNS/NTT polynomial matrix multiplication;
- ranked polynomial slicing and reconstruction;
- former/latter ciphertext reconstruction for CCMM;
- relinearization;
- CKKS rescaling;
- decryption and SinC decoding.

### End-to-End Validation

| Operation | Matrix dimension | Scalar degree | Batch count | Relative L2 error | Status |
|---|---:|---:|---:|---:|---|
| Batch CPMM | 64x64 | 256 | 128 | 3.558767985371e-7 | PASS |
| Batch CPMM | 128x128 | 128 | 64 | 1.610681474010e-7 | PASS |
| Batch CCMM | 64x64 | 128 | 64 | 4.101992550405e-7 | PASS |
| Batch CCMM | 128x128 | 64 | 32 | 1.942477178548e-7 | PASS |

For the validated CCMM configurations:

```text
LARGE_RING_DEGREE = 8192
LOG2_SCALE        = 27.993302092216055
SCALE             ≈ 2.671920963825e8
```

The active SD3b level-1 RNS basis used by the reproduced path is:

```text
q0 = 68,712,923,137
q1 =    268,238,849
Q1 = q0 * q1
```

One CKKS rescale removes `q1` after ciphertext multiplication.

---

## Source-Faithful CCMM Execution Path

The validated Batch CCMM path follows the structure of the reference construction while using FHE-rs-native cryptographic and arithmetic infrastructure.

```text
Enc(A), Enc(B)
      |
      +----------------------------+
      |                            |
      |                          C-MT(B)
      |                            |
      v                            v
 ranked slice                 ranked slice
   2d x d                       2d x d
      |                            |
      |                         transpose
      |                            |
      |                          d x 2d
      +-------------+--------------+
                    |
             modular polynomial MM
             (2d x d)(d x 2d)
                    |
                  2d x 2d
                    |
              +-----+-----+
              |           |
           former       latter
           d x 2d       d x 2d
              |           |
          transpose   transpose
              |           |
           2d x d       2d x d
              |           |
            stack         stack
              |           |
             C-MT         C-MT
              |           |
              +-----+-----+
                    |
              addCtSkCt-equivalent
                    |
            quadratic ciphertext
                    |
              relinearization
                    |
                CKKS rescale
                    |
             decrypt / SinC decode
                    |
              compare with A * B
```

For the final CCMM reconstruction, the HEaaN-style semantic relation

```text
former + s * latter
```

maps into the FHE-rs quadratic ciphertext representation as

```text
c0 = b_former
c1 = a_former + b_latter
c2 = a_latter
```

followed by ordinary RNS relinearization and CKKS rescaling.

---

## Reproduction Commands

Run the four capability tests independently:

```bash
cargo test --release batch_cpmm_end_to_end_authors_d64 -- --nocapture
cargo test --release batch_cpmm_end_to_end_authors_d128 -- --nocapture
cargo test --release batch_ccmm_end_to_end_authors_d64 -- --nocapture
cargo test --release batch_ccmm_end_to_end_authors_d128 -- --nocapture
```

Before collecting or reporting results, run the repository hygiene checks:

```bash
cargo fmt --all -- --check
cargo check --release
cargo clippy --release --all-targets -- -D warnings
git diff --check
```

---

## Claim Boundary

These tests establish functional and numerical reproduction of the tested Batch CPMM and CCMM configurations.

They do **not** currently constitute a performance reproduction or a performance comparison with the reference artifact.

The current Batch implementation is intentionally being used first as a correctness and capability baseline. In particular, the modular matrix-product path still performs redundant NTT work and has substantial optimization headroom.

The authors-scale CCMM tests are therefore heavyweight milestone-validation tests rather than routine CI tests.

Current observed unoptimized wall-clock validation times include approximately:

```text
Batch CCMM d=64   ≈ 2491.74 s
Batch CCMM d=128  ≈ 5440.86 s
```

These times should be treated only as baseline implementation measurements, not comparative performance claims.

---

## Research Role of the Batch Implementations

The Batch CPMM and CCMM implementations serve two purposes inside FHE-rs.

First, they provide independently reproduced, source-faithful implementations of sophisticated structured encrypted matrix-multiplication techniques.

Second, they exercise a substantial portion of the FHE-rs stack through a single workload:

- canonical CKKS encoding and decoding;
- RLWE encryption and decryption;
- RNS representations;
- NTT-domain polynomial arithmetic;
- Galois automorphisms and evaluation keys;
- key switching;
- modulus chains and basis transitions;
- relinearization;
- rescaling;
- structured matrix layouts;
- higher-level encrypted linear algebra.

This makes the Batch constructions useful not only as reproduced algorithms, but also as integration workloads for future work on execution policies, accelerators, resilience, alternative arithmetic backends, and new encrypted linear-algebra techniques.

---

---

## Reference

The Batch CPMM and CCMM implementations reproduce the constructions described
by Jung Hee Cheon, Minsik Kang, and Junho Lee in *Fast Batch Matrix
Multiplication in Ciphertexts* (CRYPTO 2026).

Within FHE-rs, these structured algorithms are integrated with the broader
CKKS/RLWE, RNS/NTT, key-management, modulus-management, and eBLAS
infrastructure described above.
