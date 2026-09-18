# ccmm-rs

`ccmm-rs` is a correctness-first native Rust implementation of ciphertext-plaintext matrix multiplication (CPMM) and ciphertext-ciphertext matrix multiplication (CCMM) for RLWE/CKKS-style encrypted matrices.

The implementation is based on the matrix-multiplication constructions introduced by Jung Hee Cheon, Minsik Kang, and Junho Lee in **“Fast Batch Matrix Multiplication in Ciphertexts,” CRYPTO 2026**.

> **Attribution**
>
> `ccmm-rs` does not propose the CPMM or CCMM algorithms. These constructions are due to Cheon, Kang, and Lee. This repository provides an independent native-Rust implementation, correctness infrastructure, characterization, and cross-validation against the HEaaN-based reference implementation.

## Original Paper

Jung Hee Cheon, Minsik Kang, and Junho Lee, **“Fast Batch Matrix Multiplication in Ciphertexts,”** *Advances in Cryptology — CRYPTO 2026*, Lecture Notes in Computer Science, vol. 16801, pp. 558–590, Springer, 2026.

- DOI: `10.1007/978-3-032-35374-0_18`
- IACR Cryptology ePrint Archive: `2025/1957`

## What Is Implemented

The repository contains a complete correctness-first native Rust vertical slice for encrypted matrix multiplication:

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

## CCMM Construction

For ciphertext matrices `(B, A)` and `(D, C)`, the native implementation follows the degree-2 RLWE product structure:

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
```

with

```text
c0 = BD
c1 = BC + AD
c2 = AC
```

before multiplication-key relinearization.

## End-to-End Correctness

The primary native implementation gate is:

```text
Dec(CCMM(Enc(M1), Enc(M2))) ~= M1 * M2
```

For

```text
M1 = [[1, 2],
      [3, 4]]

M2 = [[5, 6],
      [7, 8]]
```

the expected result is

```text
[[19, 22],
 [43, 50]]
```

The deterministic native Rust oracle reports a maximum absolute error of approximately `1.07e-4`.

Run it with:

```bash
cargo run --release --bin ccmm_oracle
```

A successful run terminates with:

```text
RUST_CCMM_STATUS=PASS
```

## HEaaN Cross-Validation

The native implementation has also been cross-validated against the HEaaN-based implementation of the original construction using the same deterministic matrix workload.

| Metric | Result |
| --- | ---: |
| Native Rust max absolute error | ~`1.07e-4` |
| HEaaN max absolute error | ~`2.74e-5` |
| Rust/HEaaN max disagreement | ~`1.07e-4` |
| Cross-validation tolerance | `1.00e-3` |
| Cross-validation status | `PASS` |

HEaaN is used only as an external validation oracle and is **not part of the native Rust execution path**.

## Baseline Characterization

The current correctness-oriented configuration is:

| Parameter | Value |
| --- | ---: |
| Ring degree `N` | `8` |
| High modulus `Q` | `140739635773439` |
| Low modulus `q` | `2147483647` |
| Rescale prime `p` | `65537` |
| Initial scale `Delta` | `65537` |
| Matrix dimensions | `2 x 2` |
| Characterization seeds | `32` |

The release-mode characterization measures:

- secret-key generation;
- evaluation-key generation;
- left and right matrix encryption;
- native CCMM;
- decryption and decoding;
- maximum absolute numerical error;
- mean absolute numerical error.

Run the current characterization with:

```bash
cargo run --release --bin ccmm_bench
```

A successful campaign terminates with:

```text
CCMM_BENCH_STATUS=PASS
```

The benchmark is intended to characterize the current correctness-first implementation. Its timings are **not performance comparisons** against implementations using different cryptographic parameters, packing schemes, ring dimensions, or arithmetic backends.

## Reproducibility

Run the complete quality gate with:

```bash
cargo fmt --all -- --check
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings
```

Run the deterministic native oracle with:

```bash
cargo run --release --bin ccmm_oracle
```

Run the baseline characterization with:

```bash
cargo run --release --bin ccmm_bench
```

The `v0.1.0` correctness baseline contains **101 passing Rust tests with zero Clippy warnings**.

## Current Scope and Limitations

`ccmm-rs` is currently a research and correctness implementation. It deliberately prioritizes transparent semantics and testability over production-scale performance.

Current limitations include:

- a deliberately small ring degree for transparent correctness testing;
- correctness-oriented polynomial multiplication;
- a reference NTT rather than a production-optimized NTT backend;
- a two-level modulus chain;
- coefficient-oriented CKKS-style encoding rather than production SIMD/complex-slot packing;
- no third-party security audit;
- baseline timings that should not be interpreted as production-performance claims.

These constraints make the current release suitable as a **correctness baseline** for subsequent optimization and extension.

## Citation

If you use the CCMM construction, please cite the original work:

```bibtex
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
```

The corresponding preprint is:

```bibtex
@article{CheonKangLee2025ePrint,
  author  = {Jung Hee Cheon and Minsik Kang and Junho Lee},
  title   = {Fast Batch Matrix Multiplication in Ciphertexts},
  journal = {IACR Cryptology ePrint Archive},
  volume  = {2025},
  pages   = {1957},
  year    = {2025}
}
```

For use of the `ccmm-rs` software artifact itself, citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).
