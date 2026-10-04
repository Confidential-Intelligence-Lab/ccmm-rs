# FHE-rs

**A native-Rust research platform for encrypted computing and
privacy-aware linear algebra.**

FHE-rs is an experimental full-stack platform for expressing, executing,
validating, and studying computation over encrypted data. Its current
cryptographic foundation is leveled RNS-CKKS, while its software
architecture is designed to separate **what an application computes**
from **how encrypted execution is represented and realized**.

The repository remains named **`ccmm-rs`** to preserve development
history and reproducibility. **FHE-rs** is the broader platform
identity.

> **Project status:** active research software. APIs, parameter sets,
> and execution mechanisms may evolve. Functional and research parameter
> profiles are not automatically production-security recommendations.

------------------------------------------------------------------------

## Why FHE-rs?

Fully homomorphic encryption makes it possible to compute on encrypted
data without first decrypting it, but useful encrypted applications
require more than individual cryptographic primitives. They require a
software stack that can connect application semantics to encrypted
linear algebra, representations, cryptographic state, execution
mechanisms, and eventually heterogeneous hardware.

FHE-rs explores that stack.

The central design principle is:

> **Applications express what computation they need through eBLAS.
> FHE-rs determines how that computation is represented and executed
> using privacy-aware and structure-aware mechanisms.**

This separation makes it possible to study encrypted algorithms without
binding the application interface to a single matrix-multiplication
mechanism, data layout, or future hardware backend.

------------------------------------------------------------------------

## Architecture

``` text
Applications / security services
              |
            eBLAS
              |
   +----------+-----------+
   |          |           |
  GEMM      CPMM        CCMM
   |          |           |
   +---- structured ------+
        encrypted ops
   correlation / FFT / ...
              |
     execution profiles
     representations
     decomposition
     scheduling
              |
         RNS-CKKS
              |
   NTT / RNS / keys / RNG
              |
        CPU today
              |
 GPU / FPGA / PIM / photonics
       research roadmap
```

FHE-rs deliberately separates several concerns that are often conflated:

1.  **Mathematical operation** --- GEMM, GEMV, DOT, AXPY, correlation,
    FFT, and related operations.
2.  **Privacy mode** --- which operands are public, private, or
    encrypted.
3.  **Execution mechanism** --- direct encrypted execution, CPMM, CCMM,
    structured transforms, and future mechanisms.
4.  **Representation** --- scalar, packed-slot, tensor, batch, or
    structure-aware layouts.
5.  **Execution policy** --- decomposition, scheduling, backend
    selection, and characterization.
6.  **Cryptographic realization** --- RNS-CKKS arithmetic, NTT, modulus
    switching, rescaling, relinearization, rotations, and evaluation
    keys.

------------------------------------------------------------------------

## eBLAS: encrypted linear algebra

The eBLAS layer provides a disciplined interface for encrypted numerical
computation.

The current hierarchy includes:

``` text
Level 1
  DOT
  AXPY
  SCALE
  ADD

Level 2
  GEMV

Level 3
  GEMM
  Batched GEMM

Structured eBLAS extensions
  CORRELATE1D
  CORRELATE2D
  FFT1 / iFFT1
  FFT2 / iFFT2
```

Correlation and FFT are not standardized BLAS operations. FHE-rs treats
them as **structured eBLAS extensions** with the same engineering
discipline: explicit shapes, privacy modes, deterministic lowering or
execution plans, representation-aware execution, numerical validation,
and reproducible characterization.

### Privacy modes

FHE-rs distinguishes operand privacy from the mechanism used to execute
an operation:

  Mode   Left operand       Right operand
  ------ ------------------ ------------------
  PP     plaintext/public   plaintext/public
  CP     ciphertext         plaintext/public
  PC     plaintext/public   ciphertext
  CC     ciphertext         ciphertext

Packed mechanisms such as CPMM and CCMM are execution mechanisms, not
privacy modes themselves.

------------------------------------------------------------------------

## Matrix multiplication

FHE-rs supports native and decomposed encrypted matrix multiplication,
including arbitrary logical dimensions through tiling and reduction.

Current coverage includes:

  Capability                      PP   CP / CPMM   CC / CCMM
  ---------------------------- ----- ----------- -----------
  Native GEMM                      ✓           ✓           ✓
  M/N decomposition                ✓           ✓           ✓
  K reduction                      ✓           ✓           ✓
  M/K/N decomposition              ✓           ✓           ✓
  Tile/product provenance          ✓           ✓           ✓
  Packed encrypted execution     ---           ✓           ✓

The packed matrix core has been validated for GEMM, GEMV, and DOT using
CPMM/CP and CCMM/CC paths.

FHE-rs also includes characterization infrastructure for comparison with
the reference research artifact on comparable dimension sets.
Performance comparisons should be interpreted as research
characterization rather than parity claims.

------------------------------------------------------------------------

## Correlation

FHE-rs implements structure-aware correlation by lowering correlation
semantics onto the encrypted matrix substrate.

### 1D correlation

The current valid 1D correlation semantics are:

``` text
y[i] = sum_j x[i + j] * h[j]
```

The kernel is not reversed.

Supported execution includes:

-   PP reference/lowered execution;
-   CP through decomposed encrypted GEMM;
-   CC through decomposed encrypted GEMM;
-   arbitrary logical dimensions through decomposition;
-   numerical comparison against cleartext reference results.

### 2D correlation

The current 2D representation uses:

``` text
input:   [B, H, W, C]
filter:  [Kh, Kw, C, F]
output:  [B, OH, OW, F]
```

with the logical matrix lowering:

``` text
M = B * OH * OW
K = Kh * Kw * C
N = F
```

The current implementation covers PP and CP execution. Reproducible
evidence is available through:

``` bash
scripts/run_eblas_correlation_evidence.sh
```

------------------------------------------------------------------------

## FFT

FHE-rs includes cleartext, scalar-encrypted, and packed-encrypted FFT
execution.

### FFT1 / iFFT1

Implemented capabilities include:

-   radix-2 FFT semantics and explicit execution plans;
-   cleartext FFT1/iFFT1;
-   scalar encrypted FFT1 execution;
-   packed CKKS FFT1/iFFT1;
-   public complex slot-vector multiplication;
-   packed rotations;
-   encrypted DIF stages;
-   bit-reversed physical output representation;
-   explicit level/depth accounting.

The packed DIF representation avoids an encrypted bit-reversal
permutation by interpreting the physical output layout directly.

### FFT2 / iFFT2

FHE-rs implements separable packed 2D FFT execution without an encrypted
matrix transpose.

The packed representation uses natural row-major input followed by:

``` text
row DIF stages
      |
column DIF stages using strided rotations
      |
bit-reversed row/column physical layout
```

Row stages use contiguous rotations. Column stages operate directly on
the same row-major ciphertext using strided rotations and public
diagonal masks.

The result is a **transpose-free encrypted FFT2 execution path**.

For a packed `R x C` FFT2 with

``` text
S = log2(R) + log2(C)
```

the current generic execution structure is:

``` text
ciphertexts          = 1
rotations            = 2 * S
public multiplies    = 3 * S
CC multiplies        = 0
relinearizations     = 0
encrypted transposes = 0
forward depth        = S
inverse depth        = S + 1
```

The inverse transform uses one additional public multiplication for
global normalization.

Functional packed FFT validation profiles are explicitly marked:

``` text
SECURITY_BEARING=false
```

They establish encrypted execution semantics, layout, numerical
behavior, and depth; they are not production-security parameter
recommendations.

------------------------------------------------------------------------

## Reproducible FFT evidence

Run:

``` bash
scripts/run_eblas_fft_evidence.sh
```

The runner exercises scalar and packed encrypted FFT paths and produces:

``` text
results/eblas_fft_evidence/
    raw.log
    summary.csv
    metadata.txt
```

The evidence format deliberately distinguishes:

-   **MEASURED** numerical and cryptographic-state observations; and
-   **STRUCTURAL** implementation-derived operation counts.

This prevents modeled operation counts from being presented as
instrumented runtime measurements.

Current evidence covers scalar FFT1 characterization, packed FFT1/iFFT1,
packed FFT2 row/column stages, and complete packed FFT2/iFFT2 execution.

------------------------------------------------------------------------

## RNS-CKKS foundation

The current FHE-rs cryptographic implementation includes the components
required by the implemented encrypted-computing paths, including:

-   CKKS encoding and decoding;
-   encryption and decryption;
-   RNS modulus bases;
-   NTT-domain arithmetic;
-   ciphertext addition and subtraction;
-   public real and complex scaling;
-   packed public slot-vector multiplication;
-   rotations;
-   modulus switching;
-   CKKS rescaling;
-   bounded RNS relinearization;
-   evaluation-key handling;
-   level, scale, and basis tracking.

The platform is currently centered on leveled RNS-CKKS, but the
higher-level architecture is intentionally not designed as a CKKS-only
application interface.

------------------------------------------------------------------------

## Decomposition and representation

Large logical operations are separated from native encrypted execution
dimensions.

The decomposition layer records and preserves execution provenance such
as:

-   logical matrix dimensions;
-   tile coordinates;
-   K-reduction structure;
-   product provenance;
-   packed-lane provenance where applicable;
-   representation and execution profile.

This enables application-level dimensions to be expressed independently
of a particular native encrypted matrix mechanism.

------------------------------------------------------------------------

## Engineering assurance

FHE-rs is developed as a reproducible research platform. Cryptographic
state, numerical semantics, representation, and dependency provenance
are treated as part of correctness.

The automated quality baseline includes:

-   a pinned Rust toolchain;
-   tracked `Cargo.lock`;
-   locked dependency resolution;
-   formatting checks;
-   release-mode compilation across all targets;
-   Clippy with warnings denied;
-   release-mode library tests;
-   selected NTT, RNS, CKKS, eBLAS, FFT1, and FFT2 validation paths;
-   RustSec dependency auditing;
-   `cargo-deny` advisory, license, dependency, and source policy;
-   a crate-wide safe-Rust baseline.

The core crate currently uses:

``` rust
#![forbid(unsafe_code)]
```

Future accelerator or FFI integration should isolate any required unsafe
boundary rather than weakening the core crate globally.

### Local assurance baseline

``` bash
cargo fmt --all -- --check
cargo metadata --locked --format-version 1 > /dev/null
cargo check --locked --release --all-targets
cargo clippy --locked --release --all-targets -- -D warnings
cargo test --locked --release --lib
cargo audit
cargo deny check
```

Changes to numerical or cryptographic subsystems should additionally
execute the relevant validation and evidence binaries.

See:

-   [`SECURITY.md`](SECURITY.md) --- vulnerability reporting and
    security scope;
-   [`docs/SECURE_DEVELOPMENT.md`](docs/SECURE_DEVELOPMENT.md) ---
    secure-development contract;
-   [`deny.toml`](deny.toml) --- dependency, advisory, license, and
    source policy;
-   [`.github/workflows/ci.yml`](.github/workflows/ci.yml) --- automated
    assurance gates.

------------------------------------------------------------------------

## Security and parameter interpretation

FHE-rs is research software.

The repository distinguishes:

-   **functional validation parameters** --- used to establish semantics
    and exercise execution paths;
-   **research parameters** --- used for research experiments and
    characterization;
-   **security-bearing parameters** --- parameters for which a specific
    security claim is intended and supported.

Do not infer a production security claim merely because an operation
executes over ciphertext.

Profiles explicitly reporting:

``` text
SECURITY_BEARING=false
```

must not be presented as production cryptographic parameter sets.

Parameter changes affecting ring degree, modulus chain, error
distribution, secret distribution, gadget decomposition, auxiliary
moduli, or evaluation-key construction are security-sensitive changes.

------------------------------------------------------------------------

## Repository layout

The exact tree evolves with the research program, but the principal
organization is:

``` text
src/
  eblas/              encrypted linear-algebra semantics and execution
  bin/                validation, characterization, and research binaries
  ...                 CKKS, RNS, NTT, CCMM/CPMM, and supporting components

scripts/
  reproducible characterization and evidence runners

results/
  generated research evidence and characterization output

docs/
  design and secure-development documentation

.github/workflows/
  continuous-integration assurance
```

Generated `results/` directories may contain local experimental evidence
and are not necessarily part of the committed source baseline.

------------------------------------------------------------------------

## Validation philosophy

For encrypted numerical software, "the program ran" is not a sufficient
correctness criterion.

Where applicable, FHE-rs validation checks:

``` text
cleartext semantic oracle
        |
encrypted implementation
        |
decoded numerical error
        |
level / scale / RNS basis
        |
physical representation
        |
structural operation cost
```

Typical evidence includes:

-   relative L2 error;
-   maximum absolute error;
-   inactive-slot error;
-   level consumption;
-   output scale;
-   representation/layout;
-   rotations;
-   ciphertext-public multiplications;
-   ciphertext-ciphertext multiplications;
-   relinearizations;
-   encrypted transposes.

Measured quantities and implementation-derived structural quantities are
kept distinguishable.

------------------------------------------------------------------------

## Current research direction

FHE-rs is evolving from cryptographic mechanisms toward an
application-facing encrypted-computing platform.

Near-term work includes:

-   broader eBLAS coverage;
-   encrypted signal and image-processing applications;
-   decomposition of larger structured workloads;
-   stronger automated numerical regression;
-   property-based testing and fuzzing;
-   performance characterization;
-   security-parameter analysis;
-   compiler/runtime interfaces for backend selection.

A representative application direction is encrypted image filtering:

``` text
private image
    |
tile / overlap decomposition
    |
encrypted FFT2
    |
public spectral filter
    |
encrypted iFFT2
    |
reassembly
    |
private filtered image
```

Correct convolution requires overlap-aware decomposition such as
overlap-save or overlap-add rather than independent non-overlapping
tiles.

------------------------------------------------------------------------

## Heterogeneous execution roadmap

The semantic and execution layers are intended to support future
heterogeneous encrypted-computing research.

Potential backend directions include:

-   multicore CPU;
-   GPU;
-   FPGA;
-   processing-in-memory;
-   silicon photonics;
-   heterogeneous combinations of these mechanisms.

These are roadmap directions unless explicitly identified elsewhere as
implemented and validated.

The objective is to preserve the application/eBLAS interface while
allowing execution mechanisms and hardware mappings to evolve underneath
it.

------------------------------------------------------------------------

## Research reproducibility

Characterization and evidence runners should record enough information
to identify:

-   source commit;
-   execution profile;
-   parameter classification;
-   logical shape;
-   representation;
-   privacy mode;
-   numerical error;
-   cryptographic-state transitions;
-   structural operation counts;
-   measured runtime when explicitly instrumented.

Do not substitute modeled structural counts for measured timing or
instrumentation.

------------------------------------------------------------------------

## Project identity

**Platform:** FHE-rs\
**Repository:** `ccmm-rs`\
**Organization:** Confidential Intelligence Lab\
**Primary language:** Rust\
**Current HE foundation:** leveled RNS-CKKS\
**License:** see [`LICENSE`](LICENSE)

The repository name is intentionally retained for history and
reproducibility while the software evolves into the broader FHE-rs
platform.

------------------------------------------------------------------------

## Contributing

FHE-rs is an active research codebase. Contributions should preserve:

-   mathematical semantics;
-   privacy-mode distinctions;
-   cryptographic state invariants;
-   explicit representation contracts;
-   reproducible validation;
-   dependency and security policy;
-   portability unless a backend is explicitly architecture-specific.

Before proposing a change, run the local assurance baseline and the
validators relevant to the modified subsystem.

Security-sensitive findings should follow [`SECURITY.md`](SECURITY.md)
rather than being disclosed through a public issue.

------------------------------------------------------------------------

## License

See [`LICENSE`](LICENSE).
