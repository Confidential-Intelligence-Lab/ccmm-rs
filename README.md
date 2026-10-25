# FHE-rs

**Encrypted computing in Rust, from modular arithmetic to application-facing encrypted linear algebra and Fourier transforms.**

FHE-rs is an open-source research platform for privacy-preserving computation. It connects cryptographic primitives, polynomial arithmetic, execution planning, and application-level numerical kernels in a modular Rust implementation. The currently implemented homomorphic-encryption scheme is **leveled RNS-CKKS**. The wider architecture is intended to support additional encrypted-computation mechanisms without binding application semantics to one cryptosystem.

The Git repository and Cargo package retain the historical name **`ccmm-rs`**. The software platform is **FHE-rs**; the v3.0 implementation is available on the [`feat/r15-frequency-domain-filtering` branch](https://github.com/Confidential-Intelligence-Lab/ccmm-rs/tree/feat/r15-frequency-domain-filtering). This branch is a release candidate, not a tagged release on `main`.

> **Research software.** The included numerical validation and research parameter profiles do not constitute production cryptographic security certification. Large encrypted-FFT demonstrations use research configurations. See [Security and parameters](docs/SECURITY_AND_PARAMETERS.md) and [Security policy](SECURITY.md).

## Contents

- [Why FHE-rs](#why-fhe-rs)
- [Getting started](#getting-started)
- [Architecture](#architecture)
- [Implemented capabilities](#implemented-capabilities)
- [Encrypted Fourier transforms](#encrypted-fourier-transforms)
- [Examples and reproducible experiments](#examples-and-reproducible-experiments)
- [Parallel execution](#parallel-execution)
- [Optimization and research methodology](#optimization-and-research-methodology)
- [Documentation and references](#documentation-and-references)
- [Development and security](#development-and-security)

## Why FHE-rs

Encrypted applications need more than an encryption API. Numerical workloads require explicit management of ciphertext layout, scale, modulus-chain levels, evaluation keys, data movement, scheduling, and the cost of repeated arithmetic. FHE-rs brings these concerns into one inspectable research stack while keeping **what an operation means** separate from **which operands are private**, **how it is evaluated**, and **how its data is represented**.

The platform supports research across privacy-aware matrix operations and encrypted signal processing, including Fourier transforms on packed and decomposed images. It preserves reference and optimized execution paths so that correctness and performance claims can be checked against explicit numerical oracles and recorded configurations.

## Getting started

Install a compatible Rust toolchain and clone the repository. To reproduce the v3.0 release-candidate source, select its branch explicitly:

```bash
git clone https://github.com/Confidential-Intelligence-Lab/ccmm-rs.git
cd ccmm-rs
git switch feat/r15-frequency-domain-filtering
cargo build --locked --release
cargo test --locked --lib
```

Run application-oriented examples:

```bash
cargo run --locked --release --bin private_linear_inference
cargo run --locked --release --bin private_two_party_matrix_product
cargo run --locked --release --bin eblas_gemm_validation
```

These are research examples, not production deployment recipes. Consult [Getting started](docs/GETTING_STARTED.md) for API conventions and CKKS-state handling.

## Architecture

```text
 Applications / privacy-aware security services
                     |
             eBLAS numerical semantics
       ADD, AXPY, DOT, GEMV, GEMM, batch,
          tensor, correlation, FFT/iFFT
                     |
        Operand privacy + execution policy
              PP | CP | PC | CC
                     |
      Direct / CPMM / structured CCMM /
        packed and decomposed transforms
                     |
      Layout, tiling, key planning, rotation,
         scheduling and parallel execution
                     |
              Leveled RNS-CKKS
       encode / encrypt / evaluate / decrypt
                     |
      RNS / NTT / modular polynomial arithmetic
                     |
             Multicore CPU execution
```

**Semantics, operand privacy, execution mechanism, and representation are separate design dimensions.** The CKKS logical modulus chain must also be distinguished from the physical RNS residue limbs and auxiliary modulus material used by key-switching mechanisms. The current implemented backend is CPU-based; GPU, FPGA, and photonic acceleration are research directions, not validated backends of this release.

See [eBLAS](docs/EBLAS.md) and [RNS architecture](docs/RNS_ARCHITECTURE.md) for detailed contracts.

## Implemented capabilities

| Layer | Capabilities and scope |
|---|---|
| Cryptographic foundation | Leveled RNS-CKKS encoding, encryption, evaluation, decryption and decoding; rescaling, modulus switching, rotations, relinearization, and evaluation keys |
| Polynomial arithmetic | RNS modular arithmetic, NTT transforms, prepared NTT plans, and representation-aware execution |
| Key switching | Bounded and hybrid research mechanisms, auxiliary-basis handling, prepared evaluation material |
| eBLAS | Level-1 operations, DOT, GEMV, GEMM, batch and tensor mappings; applicable PP/CP/PC/CC privacy modes |
| Matrix mechanisms | Direct operations, CPMM-style accumulation, structured CCMM, tiling, and reduction |
| Signal processing | Correlation, FFT1/iFFT1, packed FFT2/iFFT2 components, and encrypted-input/public-transform FFT2 |
| Larger transforms | Decomposed 1024 x 1024 and hierarchical 4096 x 4096 FFT2 experiments; separate native-UHD development path |
| Validation | Cleartext-oracle comparison, numerical error, CKKS scale/level contracts, structural accounting, and experiment provenance |

Not claimed: bootstrapping, general production-secure large-FFT parameters, or validated accelerator backends.

## Encrypted Fourier transforms

The primary demonstrated model is **encrypted input, public transform**: image or signal samples remain encrypted during evaluation; FFT twiddle factors, masks, and transform schedule are public. This is a deliberate privacy boundary, not a claim that transform coefficients are secret. Fully private transform parameters remain a separate research question.

A representative FFT-only application flow is:

```text
Image / signal -> encode + encrypt -> encrypted FFT2
               -> decrypt + decode -> validated spectrum
```

The larger-image evidence concerns the FFT-only flow. It must not be described as an end-to-end encrypted filtering-and-inverse-FFT pipeline.

### FFT experiment ladder

The following scripts expose either a validated execution path or a clearly labeled preflight/discovery interface. **Preflight success does not establish encrypted numerical correctness.**

| Transform | Entry point | Status and interpretation |
|---|---|---|
| FFT1 / iFFT1 | `scripts/run_fft1.sh` | FFT1 validator/characterization path; consult the emitted PASS markers and guide for exact scope |
| FFT2 16 x 16 | `scripts/run_fft2_16.sh preflight` | Discovery-only interface; encrypted run deliberately blocked |
| FFT2 64 x 64 | `scripts/run_fft2_64.sh run` | Encrypted multi-tile FFT validator with cleartext-oracle checks |
| FFT2 128 x 128 | `scripts/run_fft2_128.sh preflight` | Discovery-only interface; encrypted run deliberately blocked |
| FFT2 256 x 256 | `scripts/run_fft2_256.sh preflight` | Factored research implementation identified; public execution contract not yet verified |
| FFT2 1024 x 1024 | `scripts/run_fft2_1k.sh run` | Decomposed encrypted FFT2 using 32 ciphertexts; expected `R17_1K_STATUS=PASS` |
| FFT2 4096 x 4096 | `scripts/run_fft2_4k.sh run` | Hierarchical encrypted FFT2 using 16 macrotiles / 512 ciphertexts; expected `R21_UHD_STATUS=PASS` |

A logical image size is not a ciphertext size. The 1K and 4K paths use decomposition and packing; the 4K workload is not represented by a single ciphertext. The documented 4K hierarchical validator (`ckks_packed_uhd_fft_validation`) is **not the same experiment** as the separately developed native UHD (`3840 x 2160`) implementation.

For the detailed numerical contract, representation, input requirements, and evidence interpretation, read [FFT experiments v3](docs/FFT_EXPERIMENTS_V3.md).

## Examples and reproducible experiments

Run commands from the repository root. Large encrypted experiments can require substantial CPU time and memory; start with preflight and inspect the requirements before initiating full evaluation.

```bash
# Inspect the available 1K and 4K experiment contracts without running FFT2.
bash scripts/run_fft2_1k.sh preflight
bash scripts/run_fft2_4k.sh preflight

# Run the encrypted 64 x 64 validator.
bash scripts/run_fft2_64.sh run

# Run the 1K encrypted FFT2, when host resources are available.
bash scripts/run_fft2_1k.sh run

# Run the full 4K hierarchical FFT2, when its baseline input and hash are present.
bash scripts/run_fft2_4k.sh run
```

The 4K wrapper checks a specific grayscale image against a recorded baseline SHA-256 and refuses partial-execution modes. The input image and local baseline evidence may need to be obtained separately; a fresh clone is **not** guaranteed to contain these large research artifacts. See [FFT experiments v3](docs/FFT_EXPERIMENTS_V3.md) and [Reproducibility](docs/REPRODUCIBILITY.md) before attempting reproduction.

Experimental output and frozen benchmark evidence are not necessarily committed to Git. Do not overwrite the local frozen evidence package (`results/eblas_fft_evidence/r31_5_frozen/`). New runs should use new output directories.

## Parallel execution

The 1K and hierarchical 4K encrypted FFT2 executors use separate local and inter-ciphertext parallelism controls. The checked-in wrappers currently select **fixed, research-tested configurations**:

| Workload | Local workers | Inter-ciphertext workers | Runner |
|---|---:|---:|---|
| 1024 x 1024 FFT2 | 2 | 4 | `scripts/run_fft2_1k.sh` |
| 4096 x 4096 hierarchical FFT2 | 8 | 4 | `scripts/run_fft2_4k.sh` |

```bash
# The wrapper applies its documented 2 / 4 worker configuration.
bash scripts/run_fft2_1k.sh run

# The wrapper applies its documented 8 / 4 worker configuration.
bash scripts/run_fft2_4k.sh run
```

**Important:** the current wrappers export these worker values internally. Setting `R17_1K_LOCAL_PARALLELISM`, `R17_1K_INTER_CT_PARALLELISM`, `R21_UHD_LOCAL_PARALLELISM`, or `R21_UHD_INTER_CT_PARALLELISM` outside the wrapper does **not** override those assignments. User-configurable worker overrides require a separate wrapper change and verification; do not infer override support from the existence of environment variables in the underlying binaries.

Parallel speedup is workload- and machine-dependent. Reproducible comparisons require a recorded Git revision, host configuration, parameter profile, input identity, worker configuration, and numerical result. The README intentionally does not report elapsed times or resident-memory measurements.

## Optimization and research methodology

FHE-rs is organized to evaluate algorithmic improvements without confusing local microbenchmarks with integrated application performance.

| Technique | Role in the research stack |
|---|---|
| RNS / NTT arithmetic | Physical residue-domain polynomial operations under logical CKKS level tracking |
| Composite-level and Grafting mechanisms | Manage auxiliary modulus material and key-switching execution separately from ordinary level consumption |
| CPMM and structured CCMM | Study accumulation, postprocessing, and decomposition choices for encrypted matrix operations |
| Packing and tiling | Map large logical matrices and images to bounded ciphertext/slot layouts |
| Transpose-free separable FFT2 | Combine row/column DIF stages with packed rotations and public masks/twiddles |
| Prepared public operands | Reuse encoded/NTT-ready plaintext operands across repeated ciphertext-plaintext operations |
| NTT residency | Reduce avoidable representation changes in compatible execution sequences |
| Multicore scheduling | Expose independent work at stage and ciphertext granularity |

An earlier prepared-public-operand microbenchmark demonstrated **4.10x amortized improvement for 32 ciphertext-plaintext multiplications**, with exact ciphertext equality. This is **not** a claim of 4.10x end-to-end FFT acceleration. Integrated FFT2 comparisons must use matched inputs, parameters, source provenance, correctness criteria, and clearly defined timing boundaries. See [FFT experiments v3](docs/FFT_EXPERIMENTS_V3.md) and [Research references](docs/RESEARCH_REFERENCES.md).

Numerical agreement with a cleartext FFT oracle is not a cryptographic security proof. Optimization candidates in the repository are not necessarily active in every integrated executor.

## Documentation and references

- [Getting started](docs/GETTING_STARTED.md) — installation, examples, and CKKS state.
- [eBLAS](docs/EBLAS.md) — numerical semantics, operand privacy, and mechanisms.
- [RNS architecture](docs/RNS_ARCHITECTURE.md) — residues, levels, NTT, and key-switching structure.
- [FFT experiments v3](docs/FFT_EXPERIMENTS_V3.md) — transform ladder, experiment interfaces, numerical validation, and evidence.
- [Full-stack characterization](docs/FULL_STACK_CHARACTERIZATION.md) — characterization scope and research layers.
- [Characterization evidence](docs/CHARACTERIZATION_EVIDENCE.md) — evidence inventory and qualification.
- [Reproducibility](docs/REPRODUCIBILITY.md) — provenance and repeatable experiments.
- [Research references](docs/RESEARCH_REFERENCES.md) — CKKS, RNS, related HE libraries, and attribution boundaries.
- [Security and parameters](docs/SECURITY_AND_PARAMETERS.md), [Assurance](docs/ASSURANCE.md), and [Secure development](docs/SECURE_DEVELOPMENT.md).

Foundational research includes Gentry's fully homomorphic encryption work, Cheon–Kim–Kim–Song's CKKS construction, and subsequent full-RNS CKKS methods. [Microsoft SEAL](https://github.com/microsoft/SEAL), [OpenFHE](https://github.com/openfheorg/openfhe-development), and [HElib](https://github.com/homenc/HElib) are external ecosystem references. Their mention does not imply source-code reuse, API compatibility, or benchmark parity. For verified citations and unresolved attribution items, consult the bibliography rather than relying on inferred paper titles.

## Development and security

```bash
cargo fmt --all -- --check
cargo metadata --locked --format-version 1 > /dev/null
cargo check --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --lib
```

The full historical integration-test suite includes expensive research experiments. Release validation should distinguish bounded regression tests from opt-in characterization workloads. Preserve cleartext oracles, CKKS scale/level invariants, input provenance, and parameter-security qualifications when contributing changes.

Security-sensitive reports should follow [SECURITY.md](SECURITY.md). See [Secure development](docs/SECURE_DEVELOPMENT.md) and [Extending FHE-rs](docs/EXTENDING_FHE_RS.md).

**License:** [MIT](LICENSE) · **Platform:** FHE-rs · **Repository / Cargo package:** `ccmm-rs` · **Organization:** Confidential Intelligence Lab.
