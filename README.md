# FHE-rs

**Encrypted computing in native Rust — from RNS arithmetic to application-facing eBLAS.**

FHE-rs is an open-source research platform for homomorphic encrypted computation, privacy-aware linear algebra, and structured signal processing. Its implemented cryptographic foundation is **leveled RNS-CKKS**, while the architecture separates application semantics, operand privacy, execution mechanisms, representation, and physical residue arithmetic. The repository and Cargo package remain named **`ccmm-rs`** to preserve their history.

> **Research software, not a production cryptographic security certification.** Numerical correctness, a successful encrypted evaluation, or a named research profile does not establish a production-security parameter set. Read [Security and parameters](docs/SECURITY_AND_PARAMETERS.md) and [Security policy](SECURITY.md) before use.

## Contents

- [Getting started](#getting-started)
- [Architecture](#architecture)
- [Implemented capabilities](#implemented-capabilities)
- [Algorithms and optimizations](#algorithms-and-optimizations)
- [Examples and experiments](#examples-and-experiments)
- [Documentation and research references](#documentation-and-research-references)
- [Development, security, and contribution](#development-security-and-contribution)

## Getting started

Requires a Rust toolchain compatible with the checked-in `rust-toolchain` configuration (when present) and the locked dependencies.

```bash
git clone https://github.com/Confidential-Intelligence-Lab/ccmm-rs.git
cd ccmm-rs
cargo build --locked --release
cargo test --locked --release --lib
```

A small application-oriented starting point:

```bash
cargo run --locked --release --bin private_linear_inference
cargo run --locked --release --bin private_two_party_matrix_product
```

These binaries are research examples, not production application recipes. See [Getting started](docs/GETTING_STARTED.md) for application design and CKKS-state conventions.

## Architecture

```text
  Applications / privacy-aware security services
                     |
             eBLAS semantics
       (ADD, AXPY, DOT, GEMV, GEMM,
         batch/tensor, FFT, correlation)
                     |
   privacy modes + execution policy + plans
          PP | CP | PC | CC
                     |
   direct / CPMM / structured CCMM /
       packed and decomposed FFT2
                     |
    ciphertext representation, tiling,
     scheduling, rotation/key planning
                     |
            leveled RNS-CKKS
      encode, encrypt, evaluate, decode
                     |
      RNS / NTT / modular arithmetic
                     |
            multicore CPU today
       other accelerators: research roadmap
```

The architectural contract is **operation semantics != operand privacy != execution mechanism != representation**. CKKS logical levels are distinct from physical RNS limbs; Grafting-managed auxiliary modulus material is also distinct from the ordinary level chain. See [eBLAS](docs/EBLAS.md) and [RNS architecture](docs/RNS_ARCHITECTURE.md).

## Implemented capabilities

| Area | Current research implementation |
|---|---|
| Cryptographic foundation | Leveled RNS-CKKS, canonical encoding/decoding, encryption/decryption, rescaling, modulus switching, rotations, relinearization, evaluation keys |
| eBLAS | Level-1 arithmetic, DOT, GEMV, GEMM; batch and tensor mappings |
| Privacy modes | PP, CP, PC, and CC on applicable linear-algebra paths |
| Matrix mechanisms | Direct CP/CC, CPMM-equivalent accumulation, structured CCMM, tiling and K-reduction |
| Structured transforms | Correlation, FFT1/iFFT1, packed FFT2/iFFT2, transpose-free packed DIF stages |
| Large FFT2 evidence | Decomposed 1024 x 1024 and hierarchical 4096 x 4096 grayscale input, with encrypted FFT and decrypted numerical-oracle validation |
| Validation | Numerical error, ciphertext and level/scale invariants, structural operation accounting, reproducible evidence captures |

**Not claimed:** bootstrapping, production-secure large-FFT parameter sets, or validated GPU/FPGA/photonic execution backends. Current large FFT2 experiments use research parameters.

## Algorithms and optimizations

FHE-rs retains reference paths and explicit execution mechanisms to make correctness and performance changes attributable and reproducible.

| Technique | Purpose / scope | Primary reading |
|---|---|---|
| Residue number system (RNS) and NTT | Execute polynomial arithmetic over physical residue limbs while tracking CKKS modulus levels | [RNS architecture](docs/RNS_ARCHITECTURE.md) |
| Composite levels, Grafting and bounded key switching | Separate CKKS logical modulus consumption, auxiliary basis material, and evaluation-key arithmetic | [RNS architecture](docs/RNS_ARCHITECTURE.md) |
| CP accumulation / CPMM and structured CCMM | Reuse output-level postprocessing instead of relinearizing/rescaling every scalar product | [eBLAS](docs/EBLAS.md) |
| Packing, decomposition and provenance | Represent larger logical matrices and images with multiple ciphertexts; preserve tile and reduction semantics | [eBLAS](docs/EBLAS.md), [Reproducibility](docs/REPRODUCIBILITY.md) |
| Transpose-free 2D DIF FFT | Row- and column-stage execution with rotations and public diagonal masks, avoiding encrypted transpose | [FFT experiments](docs/FFT_EXPERIMENTS_V3.md) |
| Prepared public operands | Amortize ciphertext-public multiplication setup across repeated uses; exact ciphertext equivalence tested | [FFT experiments](docs/FFT_EXPERIMENTS_V3.md) |
| NTT-resident execution | Reduce unnecessary representation conversions through prepared CP/rotation sequences | [FFT experiments](docs/FFT_EXPERIMENTS_V3.md) |
| Prepared modular arithmetic / butterfly alternatives | Differentially tested arithmetic candidates; presence of code does not imply every path uses each candidate | [FFT experiments](docs/FFT_EXPERIMENTS_V3.md) |

**Attribution rule:** An end-to-end speedup of the integrated FFT2 implementation is not evidence of an isolated speedup for any one technique. Likewise, a passing numerical test is not a cryptographic security proof.

## Examples and experiments

All commands below run from the repository root. The 1K/4K scripts and validation binaries are part of the **v3.0 release candidate** and must be committed together with this documentation; they are not guaranteed to exist on earlier branches or tags. The large experiments require the benchmark input file and substantial host resources. **This README intentionally omits runtimes and resident-memory measurements.**

| Example | Command | What it exercises |
|---|---|---|
| Application: private linear inference | `cargo run --locked --release --bin private_linear_inference` | Application-level encrypted linear inference |
| Application: two-party matrix product | `cargo run --locked --release --bin private_two_party_matrix_product` | Privacy-aware encrypted matrix multiplication |
| Matrix and eBLAS validators | `cargo run --locked --release --bin eblas_gemm_validation` | GEMM semantics and encrypted validation |
| FFT1 and packed FFT2 evidence | `bash scripts/run_eblas_fft_evidence.sh` | Scalar FFT1, packed FFT1, FFT2 stages, complete packed FFT2, oracle and structural evidence |
| 4K hierarchical FFT2 | `bash scripts/run_fft2_4k.sh` | Verify the original input, then execute a full 4096 x 4096 FFT2 using 16 macrotiles and 512 ciphertexts |

The 4K runner **must reject** a missing or hash-mismatched input, partial-macrotile mode, and optional equivalence-only modes. See [FFT experiments and evidence](docs/FFT_EXPERIMENTS_V3.md) for setup, flags, artifacts, and the distinction between the dedicated 1K validator and hierarchical 4K executor.

The 1K runner is an existing local research script whose reproducibility and revision contract must be checked before final publication; its absence from the currently published repository is a release gate.

## FFT experiment ladder (v3.0)

The public scripts below distinguish **preflight** (interface checks) from **run** (encrypted execution and validation). Run from the repository root. Research parameter profiles are not security certifications. See [FFT experiment guide](docs/FFT_EXPERIMENTS_V3.md) for packing, numerical checks, and evidence policy.

| Workload | Command | Release readiness |
|---|---|---|
| FFT1 | `bash scripts/run_fft1.sh run` | Validator-backed; verify emitted PASS markers |
| 16 x 16 FFT2 | `bash scripts/run_fft2_16.sh preflight` | Discovery only; encrypted run blocked |
| 64 x 64 FFT2 | `bash scripts/run_fft2_64.sh run` | Encrypted multi-tile validator; PASS required |
| 128 x 128 FFT2 | `bash scripts/run_fft2_128.sh preflight` | Discovery only; encrypted run blocked |
| 256 x 256 FFT2 | `bash scripts/run_fft2_256.sh preflight` | Factored FFT2 implementation identified; command contract pending |
| 1024 x 1024 FFT2 | `bash scripts/run_fft2_1k.sh run` | Encrypted decomposed FFT2; PASS required |
| 4096 x 4096 FFT2 | `bash scripts/run_fft2_4k.sh run` | Encrypted hierarchical FFT2; PASS required and verified image required |

No benchmark runtimes or peak resident-memory values are reported here. Existing experiment evidence is preserved without modification.

### Multithreaded encrypted FFT execution

FHE-rs supports parallel execution of encrypted FFT workloads through
configurable worker settings:

- **Local parallelism:** concurrent execution of supported FFT-stage operations.
- **Inter-ciphertext parallelism:** concurrent processing of independent ciphertexts.

The following configurations have been exercised in the research experiments.

**1024×1024 encrypted FFT2**

```bash
R17_1K_LOCAL_PARALLELISM=2 \
R17_1K_INTER_CT_PARALLELISM=4 \
bash scripts/run_fft2_1k.sh run
```

**4096×4096 hierarchical encrypted FFT2**

```bash
R21_UHD_LOCAL_PARALLELISM=8 \
R21_UHD_INTER_CT_PARALLELISM=4 \
bash scripts/run_fft2_4k.sh run
```

These configurations are not necessarily optimal on other systems.
Performance depends on processor topology, memory bandwidth, ciphertext
representation, and key-switching workload.

For reproducibility, record worker settings, hardware configuration,
CKKS parameters, Git revision, and numerical validation results.

The experiment wrappers must honor the supplied environment variables
for these commands to select the requested worker configurations.

See [FFT experiments and reproducibility](docs/FFT_EXPERIMENTS_V3.md)
for additional details.

## Documentation and research references

- [Getting started](docs/GETTING_STARTED.md) — build, applications, and CKKS state.
- [eBLAS](docs/EBLAS.md) — privacy modes, operations, and backends.
- [RNS architecture](docs/RNS_ARCHITECTURE.md) — physical limbs, composite levels, and Grafting.
- [FFT experiments](docs/FFT_EXPERIMENTS_V3.md) — validated workload interfaces, reproducibility and evidence.
- [Reproducibility](docs/REPRODUCIBILITY.md) — historical checkpoints and measurement discipline.
- [Research references](docs/RESEARCH_REFERENCES.md) — foundational papers, implementation context and attribution boundaries.
- [Assurance](docs/ASSURANCE.md), [secure development](docs/SECURE_DEVELOPMENT.md), [security parameters](docs/SECURITY_AND_PARAMETERS.md).

FHE-rs is independent research software. References to Microsoft SEAL, OpenFHE, HEaaN, HElib and research algorithms are for scientific context, not claims of code reuse or compatibility.

## Development, security, and contribution

```bash
cargo fmt --all -- --check
cargo metadata --locked --format-version 1 > /dev/null
cargo check --locked --release --all-targets
cargo clippy --locked --release --all-targets -- -D warnings
cargo test --locked --release --lib
```

Security-sensitive findings should follow [SECURITY.md](SECURITY.md). Changes should preserve cleartext oracles, parameter classification, CKKS state invariants, and representation contracts. Consult [secure development](docs/SECURE_DEVELOPMENT.md) and [extending FHE-rs](docs/EXTENDING_FHE_RS.md).

**License:** MIT ([LICENSE](LICENSE)). **Platform:** FHE-rs. **Repository / Cargo package:** `ccmm-rs`. **Organization:** Confidential Intelligence Lab.
