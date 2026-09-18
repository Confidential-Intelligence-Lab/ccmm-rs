# ccmm-rs

`ccmm-rs` is a correctness-first native Rust research implementation of
ciphertext-plaintext matrix multiplication (CPMM) and
ciphertext-ciphertext matrix multiplication (CCMM) for RLWE/CKKS-style
encrypted matrices.

The repository contains two complementary execution paths:

1.  a small, transparent **reference/correctness path** retained as an
    oracle for encrypted matrix multiplication; and
2.  a **Roadmap-2 RNS/NTT CKKS research path** with canonical CKKS SIMD
    semantics, modulus chains, optimized NTT/RNS arithmetic, leveled
    evaluation, evaluation-key management, Galois operations, and
    realistic encrypted matrix-multiplication characterization.

The implementation is based on the matrix-multiplication constructions
introduced by Jung Hee Cheon, Minsik Kang, and Junho Lee in **"Fast
Batch Matrix Multiplication in Ciphertexts," CRYPTO 2026**.

> **Attribution**
>
> `ccmm-rs` does not propose the CPMM or CCMM algorithms. These
> constructions are due to Cheon, Kang, and Lee. This repository
> provides an independent native-Rust implementation, correctness
> infrastructure, RNS/NTT CKKS research stack, characterization, and
> external cross-validation.

## Original Paper

Jung Hee Cheon, Minsik Kang, and Junho Lee, **"Fast Batch Matrix
Multiplication in Ciphertexts,"** *Advances in Cryptology --- CRYPTO
2026*, Lecture Notes in Computer Science, vol. 16801, pp. 558--590,
Springer, 2026.

-   DOI: `10.1007/978-3-032-35374-0_18`
-   IACR Cryptology ePrint Archive: `2025/1957`

## Architecture

The repository deliberately preserves the original correctness-oriented
path while building a more realistic RNS/NTT CKKS stack alongside it.

``` text
ccmm-rs
|
+-- Reference / correctness path
|   |
|   +-- small transparent parameters
|   +-- coefficient-domain polynomial arithmetic
|   +-- RLWE encryption/decryption
|   +-- degree-2 ciphertext products
|   +-- relinearization and rescaling
|   +-- CPMM / CCMM
|   +-- deterministic native oracle
|   +-- HEaaN external cross-validation
|   `-- baseline characterization
|
`-- Roadmap-2 RNS / NTT CKKS path
    |
    +-- RNS modulus chains and basis transitions
    +-- optimized negacyclic NTT/iNTT
    +-- canonical CKKS SIMD embedding
    +-- leveled ciphertext state
    +-- multiply -> relinearize -> rescale
    +-- level-aware evaluation keys
    +-- rotations and conjugation
    +-- Gaussian ciphertext-error infrastructure
    +-- RNS/NTT ciphertext matrices
    `-- realistic N=4096 CCMM characterization
```

The reference implementation is intentionally retained. It provides a
simple correctness oracle independent of the more complex Roadmap-2
machinery.

## Implemented Capabilities

The repository currently includes:

-   dense batch and encoded matrix representations;
-   polynomial matrices over `Z_q[X] / (X^N + 1)`;
-   modular and negacyclic polynomial arithmetic;
-   reference and optimized negacyclic NTT/iNTT infrastructure;
-   RNS polynomial representation and modulus bases;
-   modulus-basis transitions;
-   RLWE key generation, encryption, and decryption;
-   degree-2 RLWE ciphertext multiplication;
-   multiplication/evaluation-key generation;
-   gadget decomposition and relinearization;
-   modulus switching and CKKS-style rescaling;
-   leveled RNS CKKS ciphertext state;
-   level-aware evaluation-key selection;
-   canonical complex-slot CKKS embedding;
-   ciphertext addition and multiplication;
-   rotations and conjugation through Galois automorphisms;
-   native CPMM and CCMM reference paths;
-   RNS/NTT ciphertext-ciphertext matrix multiplication;
-   deterministic and seeded correctness campaigns;
-   Gaussian ciphertext-error sampling for research validation;
-   HEaaN external cross-validation;
-   release-mode reference and realistic characterization.

## Reference CCMM Construction

For ciphertext matrices `(B, A)` and `(D, C)`, the reference
implementation follows the degree-2 RLWE product structure:

``` text
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
rescale
        |
        v
(B'', A'')
```

with

``` text
c0 = BD
c1 = BC + AD
c2 = AC
```

before multiplication-key relinearization.

## Roadmap-2 RNS/NTT CKKS Path

The Roadmap-2 path separates matrix orchestration from cryptographic
arithmetic.

`RnsCkksCiphertextMatrix` stores leveled RNS CKKS ciphertexts in
column-major order. Matrix multiplication is implemented as encrypted
dot products:

``` text
C[i,j] = sum_k A[i,k] * B[k,j]
```

Each ciphertext-ciphertext product delegates to the existing NTT-backed
CKKS evaluator:

``` text
multiply
  -> relinearize
  -> rescale
```

and products in each dot product are accumulated with ciphertext
addition.

For a square `d x d` matrix multiplication, the current implementation
performs:

``` text
ciphertext multiply/relinearize/rescale operations = d^3
ciphertext additions                               = d^2(d - 1)
multiplicative depth                               = 1
```

The matrix layer does not introduce a separate polynomial
multiplication, NTT, relinearization, or rescaling implementation.

## Canonical CKKS SIMD

The Roadmap-2 path includes a canonical CKKS embedding between complex
slots and polynomial coefficients.

For ring degree `N`, the implementation exposes `N/2` complex CKKS
slots. The realistic N=4096 profile therefore provides 2048 complex
slots.

The current canonical embedding is a correctness-oriented reference
implementation. Its transform is `O(N^2)`, so encoding and decoding
costs are reported separately from encrypted matrix-kernel timing.

## Parameter Profiles

The Roadmap-2 parameter infrastructure contains research profiles for
ring degrees:

``` text
N = 4096
N = 8192
N = 16384
```

The N=4096 profile used for the current realistic CCMM characterization
has:

  Parameter                                      Value
  ---------------------------------- -----------------
  Profile                              `research-4096`
  Ring degree `N`                               `4096`
  Complex slots                                 `2048`
  Modulus-chain levels                             `3`
  Aggregate top-chain modulus size          `105 bits`
  Initial scale                                 `2^35`
  Parameter class                           `Research`
  Security-bearing designation                 `false`

The larger profiles provide additional research parameter points for
deeper or larger experiments. A parameter profile being present in the
repository does **not** by itself constitute a production security
claim.

## Security Model and Current Security Scope

`ccmm-rs` distinguishes correctness/research parameters from
security-bearing parameter sets.

The security-analysis work uses published FHE security-guideline
methodology as the primary parameter-selection reference and provides
tooling for direct Lattice Estimator evaluation of experimental or
alternative parameter sets.

The current realistic CCMM benchmark deliberately reports:

``` text
PARAMETER_CLASS=Research
SECURITY_BEARING=false
```

Ciphertext encryption in the R2.10 N=4096 characterization uses:

``` text
ErrorDistribution::DiscreteGaussian { sigma: 3.19 }
```

However, the current evaluation-key path is generated with:

``` text
EVALUATION_KEY_NOISE_BOUND=0
```

This is an explicit limitation, not an implicit security claim.

During Roadmap R2.9, Gaussian ciphertext encryption was validated
independently, while Gaussian noise in the current
evaluation-key/relinearization construction produced unacceptable
numerical amplification. That behavior was localized to the
evaluation-key/relinearization path. A production-style bounded-base or
hybrid/special-modulus key-switch construction, together with complete
auxiliary-modulus accounting, remains future work.

Consequently:

-   the realistic profiles remain research profiles;
-   `security_bearing()` remains false for the current characterized
    profiles;
-   the repository does not claim a production-ready 128-bit-secure CKKS
    instantiation;
-   the current Gaussian sampler is research infrastructure rather than
    a hardened constant-time sampler;
-   no third-party security audit has been performed.

See `security/r2.9b/` for the security/noise validation evidence and
provenance.

## Reference End-to-End Correctness

The deterministic native reference oracle checks:

``` text
Dec(CCMM(Enc(M1), Enc(M2))) ~= M1 * M2
```

for

``` text
M1 = [[1, 2],
      [3, 4]]

M2 = [[5, 6],
      [7, 8]]
```

with expected result

``` text
[[19, 22],
 [43, 50]]
```

Run it with:

``` bash
cargo run --release --bin ccmm_oracle
```

A successful run terminates with:

``` text
RUST_CCMM_STATUS=PASS
```

This oracle uses the deliberately small reference configuration and
should not be interpreted as the realistic RNS/NTT parameter path.

## HEaaN External Cross-Validation

The native reference implementation has also been cross-validated
against the HEaaN-based implementation of the original construction
using the same deterministic matrix workload.

HEaaN is used **only as an external validation oracle**. It is not part
of the native Rust execution path.

The historical reference-path cross-validation reported:

  Metric                                  Result
  -------------------------------- -------------
  Native Rust max absolute error     \~`1.07e-4`
  HEaaN max absolute error           \~`2.74e-5`
  Rust/HEaaN max disagreement        \~`1.07e-4`
  Cross-validation tolerance           `1.00e-3`
  Cross-validation status                 `PASS`

The HEaaN result validates the reference construction independently; it
should not be interpreted as an independent validation of every
Roadmap-2 RNS/NTT component.

## Realistic N=4096 CCMM Characterization

Roadmap R2.10 characterizes encrypted matrix multiplication using the
`research-4096` RNS/NTT CKKS path.

The benchmark uses:

-   ring degree `N=4096`;
-   2048 complex slots;
-   a three-level, 105-bit top modulus chain;
-   initial scale `2^35`;
-   Gaussian ciphertext error with `sigma=3.19`;
-   zero-noise evaluation keys, preserving the explicitly validated R2.9
    scope;
-   release-mode execution;
-   separate timing of encoding, encryption, evaluation-key generation,
    CCMM, and decrypt/decode.

Run it with:

``` bash
cargo run --release --bin rns_ccmm_bench -- 2
```

or, after building once:

``` bash
cargo build --release --bin rns_ccmm_bench
target/release/rns_ccmm_bench 2
target/release/rns_ccmm_bench 4
```

### 2x2 reproducibility

A five-trial N=4096 2x2 campaign produced:

``` text
TRIALS=5
CCMM_US_MIN=518594.875
CCMM_US_MEAN=520023.1414
CCMM_US_MEDIAN=520241.791
CCMM_US_MAX=521630.708
ALL_PASS=true
```

The deterministic campaign reported:

``` text
MAX_SLOT_ERROR=1.922629027223e-6
MEAN_SLOT_ERROR=8.527720653016e-9
RMS_SLOT_ERROR=5.041571872216e-8
```

against a maximum-slot-error tolerance of `1e-3`.

### 4x4 scaling point

The N=4096 4x4 characterization performs:

``` text
SCALAR_MULTIPLIES=64
CIPHERTEXT_ADDITIONS=48
MULTIPLICATIVE_DEPTH=1
```

and measured:

``` text
CCMM_US=4139224.417
MAX_SLOT_ERROR=3.157531649373e-6
MEAN_SLOT_ERROR=8.894645843515e-9
RMS_SLOT_ERROR=6.071045288200e-8
RNS_CCMM_BENCH_STATUS=PASS
```

Relative to the five-trial 2x2 CCMM mean, the 4x4 kernel scales by
approximately:

``` text
4139224.417 / 520023.1414 = 7.96x
```

while the scalar ciphertext-multiplication count grows from 8 to 64,
exactly `8x`.

This is consistent with the expected cubic operation-count growth of the
current matrix algorithm between these two dimensions.

The complete characterization evidence is retained under
`results/r2.10/`.

## Reference Baseline Characterization

The original reference-path benchmark remains available:

``` bash
cargo run --release --bin ccmm_bench
```

It uses small transparent parameters and exists as a
correctness/performance baseline for the reference implementation.

Its timings must not be compared directly with the realistic RNS/NTT
benchmark as if they represented equivalent cryptographic parameter sets
or execution models.

## Reproducibility

Run the complete Rust quality gate with:

``` bash
cargo fmt --all -- --check
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings
```

Run the deterministic reference oracle with:

``` bash
cargo run --release --bin ccmm_oracle
```

Run the reference characterization with:

``` bash
cargo run --release --bin ccmm_bench
```

Run the realistic RNS/NTT characterization with:

``` bash
cargo build --release --bin rns_ccmm_bench
target/release/rns_ccmm_bench 2
target/release/rns_ccmm_bench 4
```

Additional validation and characterization tooling is available under
`tools/`, `scripts/`, `security/`, and `results/`.

## Repository Organization

The principal source areas are:

``` text
src/
  ccmm/       reference CPMM/CCMM orchestration
  ckks/       CKKS parameters, canonical embedding, leveled evaluation
  eval/       reference evaluation/relinearization machinery
  grafting/   RNS evaluation-key and grafting infrastructure
  matrix/     plaintext, polynomial, and ciphertext matrix representations
  ring/       polynomial, RNS, modulus, and NTT arithmetic
  rlwe/       RLWE ciphertexts, keys, encryption, and error distributions
  bin/        oracles, benchmarks, and characterization executables

tools/        validation and cross-check tooling
scripts/      reproducibility/cross-validation scripts
security/     security and noise-analysis evidence
results/      characterization artifacts
```

## Current Scope and Limitations

`ccmm-rs` remains a research implementation. It prioritizes explicit
semantics, correctness evidence, differential validation, and
reproducibility over production deployment.

Important current limitations include:

-   current realistic profiles are research profiles and are not
    designated security-bearing;
-   Gaussian evaluation-key noise is deferred pending a more appropriate
    key-switch construction;
-   auxiliary-modulus accounting must be completed before a production
    security designation;
-   the Gaussian research sampler is not a hardened constant-time
    sampler;
-   the canonical CKKS embedding is currently `O(N^2)`;
-   matrix multiplication currently performs the direct encrypted `d^3`
    dot-product schedule;
-   the realistic matrix characterization currently covers 2x2 and 4x4
    matrices at N=4096;
-   HEaaN cross-validation applies to the reference construction rather
    than constituting blanket validation of the entire Roadmap-2 stack;
-   no third-party security audit has been performed.

These limitations are intentional boundaries on the claims made by the
current artifact.

## Development Status

The original `v0.1.0` release established the correctness baseline.

`v2.0.0` completes the Roadmap-2 architecture described in this repository:
realistic RNS/NTT CKKS execution, canonical SIMD semantics, leveled
evaluation, Galois operations, level-aware evaluation keys, security/noise
characterization, and realistic encrypted matrix multiplication at N=4096.

The realistic parameter profiles remain research profiles and are explicitly
not designated security-bearing. See `docs/SECURITY_AND_PARAMETERS.md` for
the current security-claim boundary.

## Citation

If you use the CCMM construction, please cite the original work:

``` bibtex
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

``` bibtex
@article{CheonKangLee2025ePrint,
  author  = {Jung Hee Cheon and Minsik Kang and Junho Lee},
  title   = {Fast Batch Matrix Multiplication in Ciphertexts},
  journal = {IACR Cryptology ePrint Archive},
  volume  = {2025},
  pages   = {1957},
  year    = {2025}
}
```

For use of the `ccmm-rs` software artifact itself, citation metadata is
provided in [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).
