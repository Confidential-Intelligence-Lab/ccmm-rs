# FHE-rs

**Homomorphic encryption and encrypted linear algebra in Rust.**

FHE-rs is a native Rust research library for building applications that compute
directly on encrypted data. It currently implements leveled CKKS for approximate
numerical computation and provides encrypted vector, matrix, batched, and tensor
operations through eBLAS.

The project is currently developed in the `ccmm-rs` repository. The repository
and Cargo package names are retained for now to preserve the history and
reproducibility of the original encrypted matrix-multiplication work.

## What You Can Build

FHE-rs is intended for applications where data should remain encrypted during
computation. Current examples and validated workloads include:

- private linear inference with encrypted inputs and public model weights;
- encrypted matrix multiplication with one encrypted operand;
- encrypted matrix multiplication with both operands encrypted;
- computations over data encrypted by multiple parties;
- batched matrix multiplication;
- tensor workloads mapped to encrypted matrix multiplication;
- convolution-style workloads expressed through tensor-to-matrix mappings.

The next application layer adds private MLP-style inference, richer tensor
workloads, and proxy re-encryption as a cryptographic service.

## Features

### Homomorphic encryption

FHE-rs currently provides:

- CKKS encoding and decoding for approximate numerical computation;
- encryption and decryption;
- encrypted addition and multiplication;
- ciphertext-plaintext multiplication;
- relinearization and rescaling;
- modulus switching;
- ciphertext rotations and conjugation;
- level-aware evaluation keys;
- multi-level computation without bootstrapping.

### Encrypted linear algebra

The eBLAS layer provides a common interface for encrypted linear algebra:

- GEMM;
- GEMV;
- DOT;
- ADD;
- SCALE;
- AXPY;
- transpose;
- batched GEMM;
- tensor-to-GEMM mappings.

Operations support the common cases where both operands are public, either
operand is encrypted, or both operands are encrypted.

For encrypted-by-encrypted matrix multiplication, FHE-rs includes both a simple
baseline path and an optimized structured path. The library can select between
them using measured backend policy, while still allowing explicit backend
selection for reproducible experiments.

### Arithmetic and execution

Under the public CKKS and eBLAS interfaces, FHE-rs includes:

- residue-number-system (RNS) arithmetic;
- Number Theoretic Transform (NTT) polynomial multiplication;
- configurable modulus chains;
- 32-, 64-, and 128-bit physical arithmetic backends;
- composite scaling and modulus transitions;
- reference and optimized execution paths for validation and characterization.

These implementation details are documented separately so that applications do
not need to depend on them directly.

## Quick Start

Run the complete validation gate:

```bash
cargo fmt --all -- --check
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

Run a private inference example:

```bash
cargo run --release --bin private_linear_inference
```

Run an encrypted two-party matrix product:

```bash
cargo run --release --bin private_two_party_matrix_product
```

See [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) for the application
development workflow.

## Example: Private Linear Inference

The private linear inference example encrypts a feature vector and evaluates it
against plaintext model weights. The encrypted result is then decrypted and
checked against the cleartext reference result.

```text
encrypted features
        |
        v
encrypted x plaintext matrix multiplication
        |
        v
encrypted prediction
        |
        v
decrypt + compare with cleartext reference
```

Because the model weights are public in this example, no multiplication
evaluation key is required.

## Example: Encrypted Two-Party Computation

The two-party matrix example encrypts both input matrices and computes their
product without decrypting either operand during evaluation.

This path exercises encrypted-by-encrypted multiplication, relinearization,
rescaling, and the optimized matrix execution path.

## Performance Highlights

For the currently characterized N=4096 CKKS profile, the optimized structured
encrypted matrix-multiplication path reduces the number of
relinearization/rescaling steps from one per scalar product to one per output
entry.

Across the tested matrix frontier, this produced approximately **2.3x to 3.45x**
lower kernel time than the scalar encrypted baseline, with **4x to 16x fewer**
relinearization/rescaling operations.

These are measured results for the current implementation and parameter profile,
not universal performance claims. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and
[`docs/ASSURANCE.md`](docs/ASSURANCE.md).

## Security and Assurance

The current `research-4096` parameter set has been evaluated against a
128-bit classical-security target for its underlying RLWE problem using the
documented Lattice Estimator methodology and MATZOV reduction-cost model.

FHE-rs remains research software. The current implementation:

- has not undergone a third-party security audit;
- does not claim constant-time or production side-channel hardening;
- uses a research Gaussian sampler;
- relies on an explicit circular/KDM assumption for the current
  secret-dependent evaluation-key construction.

The exact parameter assumptions, estimator configuration, limitations, and
supporting evidence are documented in:

- [`docs/SECURITY_AND_PARAMETERS.md`](docs/SECURITY_AND_PARAMETERS.md)
- [`docs/ASSURANCE.md`](docs/ASSURANCE.md)

## Documentation

| Topic | Document |
|---|---|
| Build and run your first encrypted computation | [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) |
| Applications and examples | [`docs/APPLICATIONS.md`](docs/APPLICATIONS.md) |
| Encrypted linear algebra and backend selection | [`docs/EBLAS.md`](docs/EBLAS.md) |
| Security parameters and claim boundaries | [`docs/SECURITY_AND_PARAMETERS.md`](docs/SECURITY_AND_PARAMETERS.md) |
| Validation, evidence, and assurance | [`docs/ASSURANCE.md`](docs/ASSURANCE.md) |
| RNS and arithmetic architecture | [`docs/RNS_ARCHITECTURE.md`](docs/RNS_ARCHITECTURE.md) |
| Reproduce experiments and measurements | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) |
| Add applications, backends, or future HE capabilities | [`docs/EXTENDING_FHE_RS.md`](docs/EXTENDING_FHE_RS.md) |

## Current Scope

FHE-rs currently provides **leveled CKKS**. In practical terms, an application
must fit within the available modulus/depth budget.

Not yet implemented:

- CKKS bootstrapping;
- integer/discrete CKKS;
- additional homomorphic-encryption schemes;
- GPU or FPGA execution backends;
- production hardening and third-party audit.

These are planned extensions, but they are not required for the current
application and resilience research.

## Architecture

```text
Applications
    |
    v
eBLAS and evaluator APIs
    |
    v
CKKS operations and evaluation
    |
    v
RNS / NTT / modular arithmetic
```

Applications should use eBLAS or stable evaluator interfaces rather than
reconstruct low-level cryptographic schedules.

The original CCMM research remains part of the implementation as one optimized
encrypted matrix-multiplication technique. The broader library is no longer
limited to that workload.

## Research Origin and Attribution

The encrypted matrix-multiplication implementation was originally motivated by
the constructions of Jung Hee Cheon, Minsik Kang, and Junho Lee:

> **Fast Batch Matrix Multiplication in Ciphertexts**, CRYPTO 2026.

FHE-rs does not claim authorship of the CPMM or CCMM algorithms. The repository
provides an independent native-Rust implementation together with CKKS, RNS/NTT,
eBLAS, validation, characterization, and application infrastructure.

For detailed citation information, see [`CITATION.cff`](CITATION.cff).

## Repository Layout

```text
src/
  ckks/       CKKS encoding, state, evaluation, keys, rotations
  eblas/      encrypted linear algebra
  eval/       key switching and relinearization primitives
  grafting/   RNS evaluation-key and modulus-transition infrastructure
  matrix/     plaintext and ciphertext matrix representations
  ring/       modular arithmetic, NTT, RNS, basis conversion
  rlwe/       RLWE keys, ciphertexts, encryption, error distributions
  bin/        examples, applications, validation, characterization

docs/         user, architecture, security, and reproducibility documentation
results/      retained validation and characterization artifacts
security/     security-analysis evidence
```

## License

MIT. See [`LICENSE`](LICENSE).
