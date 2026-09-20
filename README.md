# ccmm-rs — evolving toward FHE-rs

> **Project direction.** The repository and Cargo package remain named
> `ccmm-rs` at the R3.5 checkpoint to preserve history and reproducibility.
> The cryptographic stack developed here is evolving into **FHE-rs**, a
> native-Rust research framework for homomorphic encryption. CCMM remains an
> implemented and characterized execution technique, but no longer defines
> the scope of the software.

`ccmm-rs` now contains a correctness-first leveled RNS/NTT CKKS stack,
encrypted linear algebra through **eBLAS**, CPMM, scalar CC execution, and
structured CCMM. The current stack supports realistic N=4096 encrypted
computation, bounded-base Gaussian relinearization, composite RNS scaling,
multiple physical limb widths, PP/CP/PC/CC linear-algebra semantics, batching,
tensor-to-GEMM mappings, and measurement-driven CC backend selection.

```text
applications
    |
    v
eBLAS: operation semantics + operand privacy + backend policy
    |
    +-- CP direct / CPMM-equivalent execution
    +-- scalar CC baseline
    `-- structured CCMM
    |
    v
FHE-rs cryptographic substrate
    |
    +-- leveled CKKS
    +-- key switching / relinearization / rescaling
    +-- SIMD / Galois operations
    +-- RNS / NTT / modular arithmetic
    `-- future resilience and additional HE capabilities
```

Applications should depend on eBLAS or stable evaluator interfaces rather than
reimplementing cryptographic schedules.

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

## Start Here

| Need | Document |
|---|---|
| Build and run a first encrypted computation | [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) |
| Understand eBLAS and its privacy/backend model | [`docs/EBLAS.md`](docs/EBLAS.md) |
| See demonstrated applications | [`docs/APPLICATIONS.md`](docs/APPLICATIONS.md) |
| Understand evidence, assurance, and claim boundaries | [`docs/ASSURANCE.md`](docs/ASSURANCE.md) |
| Inspect security parameters and the precise security claim | [`docs/SECURITY_AND_PARAMETERS.md`](docs/SECURITY_AND_PARAMETERS.md) |
| Understand RNS/composite-scaling architecture | [`docs/RNS_ARCHITECTURE.md`](docs/RNS_ARCHITECTURE.md) |
| Reproduce principal results | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) |
| Add an application, backend, profile, or future scheme | [`docs/EXTENDING_FHE_RS.md`](docs/EXTENDING_FHE_RS.md) |

### R3.5 capability and evidence snapshot

| Capability | Status | Evidence / boundary |
|---|---|---|
| Leveled RNS/NTT CKKS | Implemented + validated | Native Rust tests and realistic N=4096 campaigns |
| Bounded-base Gaussian relinearization | Implemented + validated | sigma=3.19 ciphertext/evaluation-key error; `base_log=20` |
| Underlying `research-4096` RLWE parameterization | Security-parameter gate passed | ~130.3-bit binding classical estimate under documented Lattice Estimator + MATZOV methodology; not a blanket implementation-security claim |
| Composite RNS scaling / 32-, 64-, 128-bit physical limbs | Implemented + validated | Semantic-invariance and composite-rescale campaigns |
| eBLAS PP / CP / PC / CC | Implemented + validated | GEMM, GEMV, DOT, ADD, SCALE, AXPY, transpose |
| Batched GEMM | Implemented + validated | Independent per-batch semantics; no SIMD/parallel acceleration claim |
| Tensor GEMM mapping | Implemented + validated | Rank-3 tensor semantics mapped to batched GEMM |
| Structured CCMM | Implemented + characterized | 2.3–3.45x measured kernel speedup with 4–16x fewer relin/rescale operations over the tested frontier |
| Measurement-driven CC policy | Implemented + validated | K=1 scalar canonical baseline; K>=2 structured under current research-4096 implementation |
| Bootstrapping | Planned | Not implemented |
| Integer/discrete CKKS | Planned | Not implemented |
| Production hardening / side-channel assurance | Not claimed | Research sampler; no third-party security audit |

See [`docs/ASSURANCE.md`](docs/ASSURANCE.md) before reusing a security or
performance statement.

## Original Paper

Jung Hee Cheon, Minsik Kang, and Junho Lee, **"Fast Batch Matrix
Multiplication in Ciphertexts,"** *Advances in Cryptology --- CRYPTO
2026*, Lecture Notes in Computer Science, vol. 16801, pp. 558--590,
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

`ccmm-rs` separates underlying RLWE parameter security from the security
properties of a complete execution path.

For `research-4096`, R3.1 validates the underlying uniform-ternary-secret RLWE
parameterization against a 128-bit classical target using Lattice Estimator
revision `8f1ff7e` and the `RC.MATZOV` reduction-cost model. The binding
estimated work factor is approximately `2^130.3`.

R3.1 also introduces a bounded-base evaluation-key path with `base_log=20`.
At N=4096, discrete-Gaussian ciphertext and evaluation-key errors with
`sigma=3.19` pass end-to-end multiply/relinearize/rescale validation and a
realistic 2x2 ciphertext-ciphertext matrix-multiplication experiment.

The current claim boundary remains explicit:

- the estimator result characterizes the underlying RLWE parameterization;
- evaluation keys encrypt secret-dependent values and therefore retain an
  explicit circular/KDM assumption;
- the Gaussian sampler is research infrastructure and is not hardened or
  constant-time;
- the equivalent R3.1 security workflow has not yet been applied to the
  N=8192 and N=16384 research profiles;
- legacy CRT-gadget and historical zero-noise evaluation-key paths remain
  correctness/provenance paths rather than the bounded security-oriented
  reference;
- no third-party security audit has been performed.

Historical R2.9/R2.10 artifacts retain the security designations and
zero-noise evaluation-key measurements that were correct when those
experiments were frozen.

See `docs/SECURITY_AND_PARAMETERS.md`, `security/r3.1d/`, and
`results/r3.1e/` for the current claim boundary and supporting evidence.

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

## R3.2 Applications

The realistic N=4096 CKKS substrate now supports both ciphertext-plaintext and
ciphertext-ciphertext matrix workloads.

| Application | Primitive | Private operands | Evaluation key | Reference |
|---|---|---|---|---|
| Private linear inference | CPMM | encrypted input, plaintext weights | no | `src/bin/private_linear_inference.rs` |
| Private two-party matrix product | CCMM | both matrices encrypted | yes | `src/bin/private_two_party_matrix_product.rs` |

### Private linear inference

The CPMM application evaluates an encrypted feature vector against plaintext
model weights. Since plaintext multiplication preserves degree-one RLWE, the
path requires neither relinearization nor an evaluation key.

```text
cargo run --release --bin private_linear_inference
```

### Private two-party matrix product

The CCMM application evaluates a matrix product with both operands encrypted.
It uses the R3.1 bounded-base Gaussian relinearization path:

```text
BOUNDED_BASE_LOG=20
CIPHERTEXT_ERROR_SIGMA=3.19
EVALUATION_KEY_ERROR_SIGMA=3.19
```

```text
cargo run --release --bin private_two_party_matrix_product
```

The final 2x2 validation reports maximum matrix error below `1e-9`, well inside
the current `2e-3` numerical acceptance threshold.

Application evidence is retained under:

```text
results/r3.2a/
results/r3.2b/
results/r3.2c/
results/r3.2d/
```

See `docs/APPLICATIONS.md` for the application-facing summary and
`docs/SECURITY_AND_PARAMETERS.md` for the security claim boundary.

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
-   the bounded-base Gaussian evaluation-key path is validated at N=4096,
    but its secret-dependent evaluation-key messages retain an explicit
    circular/KDM assumption;
-   the bounded-base reference introduces no auxiliary special modulus; any
    future special-modulus/hybrid path requires fresh modulus accounting;
-   the Gaussian research sampler is not a hardened constant-time
    sampler;
-   the canonical CKKS embedding is currently `O(N^2)`;
-   scalar CC GEMM remains available as a reproducible baseline, while
    structured CCMM amortizes relinearization/rescaling across the reduction
    dimension; neither path should be interpreted as a universally optimal
    packed or accelerator implementation;

-   current eBLAS batching is semantic aggregation of independent GEMMs, not
    CKKS SIMD batching or parallel execution;

-   the current PC implementation reuses CP through transpose under the
    scalar-per-entry representation; its measured transpose overhead should
    not be generalized to future packed layouts;
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

The realistic parameter profiles remain research profiles. For
`research-4096`, the **underlying uniform-ternary-secret RLWE
parameterization** meets the targeted 128-bit classical-security gate under
the documented Lattice Estimator methodology and MATZOV cost model. This is
a parameter-security statement, not a blanket claim of production security,
circular/KDM security, side-channel resistance, or audit. See
`docs/SECURITY_AND_PARAMETERS.md` for the precise claim boundary.

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
