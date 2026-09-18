# ccmm-rs

`ccmm-rs` is a correctness-first native Rust implementation of
ciphertext-plaintext matrix multiplication (CPMM) and
ciphertext-ciphertext matrix multiplication (CCMM) for RLWE/CKKS-style
encrypted matrices.

The implementation is based on the matrix-multiplication constructions
introduced by Jung Hee Cheon, Minsik Kang, and Junho Lee in
**"Fast Batch Matrix Multiplication in Ciphertexts," CRYPTO 2026**.

> **Attribution**
>
> `ccmm-rs` does not propose the CPMM or CCMM algorithms.
> Those constructions are due to Cheon, Kang, and Lee.
> This repository provides an independent native-Rust implementation,
> correctness infrastructure, characterization, and cross-validation
> against the HEaaN-based reference implementation.

## Original paper

Jung Hee Cheon, Minsik Kang, and Junho Lee,
**"Fast Batch Matrix Multiplication in Ciphertexts,"**
*Advances in Cryptology - CRYPTO 2026*,
Lecture Notes in Computer Science, vol. 16801,
pp. 558-590, Springer, 2026.

- DOI: `10.1007/978-3-032-35374-0_18`
- IACR Cryptology ePrint Archive: `2025/1957`

## Implementation

The repository contains a complete correctness-first native Rust vertical
slice for encrypted matrix multiplication:

- dense batch and encoded matrix representations;
- polynomial matrices over `Z_q[X] / (X^N + 1)`;
- modular and negacyclic polynomial arithmetic;
- reference negacyclic NTT/iNTT;
- RLWE key generation, encryption, and decryption;
- degree-2 ciphertext multiplication;
- multiplication/evaluation-key generation;
- gadget decomposition and relinearization;
- two-level CKKS-style scale and modulus management;
- modulus transition and rescaling;
- native CPMM;
- native CCMM;
- deterministic and seeded correctness campaigns;
- native Rust CCMM oracle;
- HEaaN cross-validation;
- release-mode baseline characterization.

## CCMM construction

For ciphertext matrices `(B, A)` and `(D, C)`:

```text
(B, A) x (D, C)
        |
        v
(BD, BC, AD, AC)
        |
        v
(BD, BC + AD, AC)
        |
        v
(c0, c1, c2)
        |
        v
relinearization
        |
        v
(B', A')
        |
        v
rescale Q -> q
        |
        v
(B'', A'')
where

c0 = BD
c1 = BC + AD
c2 = AC

before multiplication-key relinearization.

End-to-end correctness

The primary native implementation gate is:

Dec(CCMM(Enc(M1), Enc(M2))) ~= M1 * M2

For

M1 = [[1, 2],
      [3, 4]]

M2 = [[5, 6],
      [7, 8]]
the expected result is

[[19, 22],
 [43, 50]]

The native Rust oracle reports a maximum absolute error of approximately
1.07e-4.

Run it with:

cargo run --release --bin ccmm_oracle
HEaaN cross-validation

For the same deterministic workload:

Native Rust max absolute error : ~1.07e-4
HEaaN max absolute error       : ~2.74e-5
Rust/HEaaN max disagreement    : ~1.07e-4
Cross-validation tolerance     :  1.00e-3

CCMM_CROSS_VALIDATION=PASS

HEaaN is used only as an external validation oracle and is not part of
the native Rust execution path.

Baseline characterization

Current correctness-oriented parameters:

Ring degree       N = 8
High modulus      Q = 140739635773439
Low modulus       q = 2147483647
Rescale prime     p = 65537
Initial scale     Delta = 65537
Matrix dimension       = 2 x 2
Campaign seeds         = 32

Release-mode baseline:

Secret-key generation mean       0.759 us
Evaluation-key generation mean   7.620 us
Left encryption mean             8.066 us
Right encryption mean            7.359 us

CCMM minimum                     29.125 us
CCMM mean                        57.137 us
CCMM maximum                    102.166 us

Decryption mean                   4.888 us

Maximum absolute error            3.51e-4
Mean absolute error               8.56e-5

These numbers characterize the current correctness-first implementation
and are not intended as performance comparisons against implementations
using different parameters, packing schemes, or arithmetic backends.

Reproducibility

Quality gate:

cargo fmt --all
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings

Native oracle:

cargo run --release --bin ccmm_oracle

Baseline characterization:

cargo run --release --bin ccmm_bench
Current scope

ccmm-rs is currently a research and correctness implementation.

Current limitations include:

deliberately small ring degree for transparent correctness testing;
correctness-oriented polynomial multiplication;
reference NTT rather than a production-optimized NTT backend;
a two-level modulus chain;
coefficient encoding rather than production SIMD/complex-slot packing;
no third-party security audit;
baseline timings are not production-performance claims.
Citation

Please cite the original work:

@inproceedings{CheonKangLee2026FastBatchMM,
  author    = {Jung Hee Cheon and Minsik Kang and Junho Lee},
  title     = {Fast Batch Matrix Multiplication in Ciphertexts},
  booktitle = {Advances in Cryptology -- CRYPTO 2026},
  series    = {Lecture Notes in Computer Science},
  volume    = {16801},
  pages     = {558--590},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.1007/978-3-032-35374-0_18}
}

Preprint:

@article{CheonKangLee2025ePrint,
  author  = {Jung Hee Cheon and Minsik Kang and Junho Lee},
  title   = {Fast Batch Matrix Multiplication in Ciphertexts},
  journal = {IACR Cryptology ePrint Archive},
  volume  = {2025},
  pages   = {1957},
  year    = {2025}
}

For use of the ccmm-rs software artifact itself, citation metadata is
provided in CITATION.cff.

License

MIT. See LICENSE.
