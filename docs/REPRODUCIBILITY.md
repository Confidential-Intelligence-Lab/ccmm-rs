# Reproducibility and Characterization

## Current R3.5 checkpoint

Later R3.1–R3.5 evidence supersedes older Roadmap-2 configurations when
describing the current implementation. Historical sections below remain for
reproducing earlier checkpoints.

Current research-4096 security-oriented evaluation uses ciphertext and
evaluation-key error sigma 3.19 with bounded `base_log=20`. The underlying
uniform-ternary-secret RLWE parameterization passes the documented targeted
128-bit classical-security gate under Lattice Estimator/MATZOV methodology;
this is not a blanket production-security statement.

Current eBLAS evidence is under `results/r3.5/`, including backend, batch/PC,
and backend-policy artifacts. The frozen policy is `K=1 -> CcScalar`,
`K>=2 -> CcStructured`; explicit selection remains available.

Current standard gate:

```bash
cargo fmt --all -- --check
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
git diff --check
```


This document provides a compact reproduction guide for the principal
correctness, interoperability, and performance evidence in `ccmm-rs`.

The repository contains two intentionally distinct implementation paths:

-   a small-parameter coefficient-domain/reference path used as a
    correctness oracle;
-   a Roadmap-2 RNS/NTT CKKS path used for realistic-dimension
    arithmetic, SIMD semantics, and matrix characterization.

Performance numbers should be interpreted as measurements of the stated
configuration, not as universal CKKS or CCMM performance claims.

## 1. Environment Capture

Before collecting results, record the repository revision and
host/toolchain information:

``` bash
git rev-parse HEAD
git status --short

rustc --version
cargo --version

uname -a
```

For a publishable characterization, also record CPU model, operating
system, compiler version, build profile, and whether the machine was
otherwise idle.

A dirty worktree should be explicitly reported or avoided.

## 2. Standard Repository Gate

The standard correctness and lint gate is:

``` bash
cargo fmt --all -- --check
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings
```

Expected result:

``` text
all tests pass
zero warnings
```

This gate should be run before collecting characterization data and
again before freezing a release.

## 3. Legacy Reference Oracle

The coefficient-domain/reference implementation remains the primary
small-parameter correctness oracle.

Run:

``` bash
cargo run --release --bin ccmm_oracle
```

The legacy oracle uses a small N=8 CKKS-style configuration and
evaluates the encrypted matrix product:

``` text
A = [1  2]
    [3  4]

B = [5  6]
    [7  8]
```

against:

``` text
A * B = [19  22]
        [43  50]
```

The reference path is intentionally retained rather than replaced by the
optimized RNS/NTT path.

## 4. Legacy Baseline Characterization

Run the original coefficient-domain benchmark with:

``` bash
cargo run --release --bin ccmm_bench
```

This benchmark belongs to the reference/baseline path. Do not compare
its absolute timing directly with the Roadmap-2 realistic RNS benchmark
without accounting for the different ring dimensions, representations,
algorithms, and measurement scopes.

## 5. HEaaN Cross-Validation

HEaaN is used as an **external validation implementation**, not as an
internal dependency or performance oracle for `ccmm-rs`.

Where the external HEaaN environment is installed and configured, use:

``` bash
bash scripts/run-ccmm-cross-validation.sh
```

The historical cross-validation retained by the project compared the
native reference result and HEaaN result against the same plaintext
matrix product and used a numerical tolerance of `1e-3`.

HEaaN availability is environment-dependent. Failure to locate an
external HEaaN installation is not a failure of the native Rust test
suite.

## 6. Roadmap-2 Realistic RNS/NTT CCMM

The realistic characterization binary is:

``` bash
cargo run --release --bin rns_ccmm_bench -- 2
```

for a 2x2 matrix and:

``` bash
cargo run --release --bin rns_ccmm_bench -- 4
```

for a 4x4 matrix.

The current benchmark uses:

``` text
PROFILE=research-4096
RING_DEGREE=4096
SLOT_COUNT=2048
CHAIN_LEVELS=3
TOTAL_MODULUS_BITS=105
INPUT_SCALE=2^35
SECURITY_BEARING=false
```

The validated R2.10 configuration uses discrete-Gaussian ciphertext
error with:

``` text
sigma=3.19
```

and a zero-noise evaluation key.

That evaluation-key choice is an explicit R2.9 limitation and must
remain visible when reporting R2.10 results.

## 7. CCMM Operation Counts

For square `d x d` matrices, the current matrix implementation evaluates
each output entry as a ciphertext dot product.

The resulting operation counts are:

``` text
scalar ciphertext multiplications = d^3
ciphertext additions              = d^2 * (d - 1)
```

Therefore:

``` text
2x2 ->  8 multiplications,  4 additions
4x4 -> 64 multiplications, 48 additions
```

These counts are useful when interpreting the observed scaling.

## 8. R2.10 Frozen Characterization

The frozen R2.10 evidence is stored under:

``` text
results/r2.10/
```

The five-trial 2x2 campaign produced:

``` text
TRIALS=5
CCMM_US_MIN=518594.875
CCMM_US_MEAN=520023.1414
CCMM_US_MEDIAN=520241.791
CCMM_US_MAX=521630.708
MAX_SLOT_ERROR=1.922629027223e-6
MEAN_SLOT_ERROR=8.527720653016e-9
RMS_SLOT_ERROR=5.041571872216e-8
ALL_PASS=true
```

A subsequent parameterized 2x2 regression produced:

``` text
CCMM_US=511660.416
MAX_SLOT_ERROR=1.529019013025e-6
MEAN_SLOT_ERROR=5.941524529503e-9
RMS_SLOT_ERROR=3.621732760032e-8
PASS
```

The 4x4 scaling point produced:

``` text
SCALAR_MULTIPLIES=64
CIPHERTEXT_ADDITIONS=48
ENCODE_US=3805357.125
ENCRYPT_LHS_US=125464.958
ENCRYPT_RHS_US=126108.500
EVAL_KEYGEN_US=25100.667
CCMM_US=4139224.417
DECRYPT_DECODE_US=638184.750
MAX_SLOT_ERROR=3.157531649373e-6
MEAN_SLOT_ERROR=8.894645843515e-9
RMS_SLOT_ERROR=6.071045288200e-8
PASS
```

Relative to the five-trial 2x2 mean, the measured 4x4 CCMM kernel time
is approximately:

``` text
4139224.417 / 520023.1414 ~= 7.96x
```

while the scalar multiplication count increases exactly:

``` text
64 / 8 = 8x
```

The measured result is therefore consistent with the operation-count
increase for these two configurations. It is not a general asymptotic or
hardware-independent performance claim.

## 9. Timing Scope

The realistic benchmark reports major phases separately:

``` text
encode
lhs encryption
rhs encryption
evaluation-key generation
CCMM kernel
decrypt/decode
```

The reported `CCMM_US` value is the encrypted matrix-multiplication
kernel measurement. It does not include encoding, input encryption,
evaluation-key generation, or output decrypt/decode.

The current canonical CKKS embedding is a correctness-oriented reference
implementation with O(N\^2) behavior. Its cost is intentionally excluded
from the CCMM kernel timing.

When publishing results, preserve this distinction.

## 10. Realistic N=4096 Matrix Example

The public matrix example provides a smaller usability-oriented
reproduction of the realistic path:

``` bash
cargo run --release --example rns_ccmm_4096
```

The example performs a 2x2 encrypted matrix multiplication using the
N=4096 research profile, canonical CKKS encoding, Gaussian ciphertext
noise with `sigma=3.19`, and the current zero-noise multiplication key.

A validated run produced:

``` text
MAX_SLOT_ERROR=2.226462672361e-6
TOLERANCE=1.000000000000e-3
RNS_CCMM_EXAMPLE_STATUS=PASS
```

This example is intended to show API composition and end-to-end
behavior, not to replace `rns_ccmm_bench` for timing characterization.

## 11. Canonical CKKS SIMD Example

Run:

``` bash
cargo run --release --example ckks_simd_4096
```

This example isolates the Roadmap-2 arithmetic path:

``` text
canonical slots
-> encode
-> RNS encrypt
-> NTT-backed multiply
-> relinearize
-> rescale
-> decrypt
-> canonical decode
```

It uses sparse complex values across the 2048 canonical slots.

The example intentionally uses zero encryption and evaluation-key noise
so that it isolates arithmetic semantics. Gaussian ciphertext-noise
behavior is validated separately by the R2.9/R2.10 evidence.

A validated run produced:

``` text
MAX_SLOT_ERROR=9.261324438176e-7
TOLERANCE=1.000000000000e-3
CKKS_SIMD_EXAMPLE_STATUS=PASS
```

## 12. Recommended Reproduction Sequence

For a clean checkout, the recommended sequence is:

``` bash
cargo fmt --all -- --check
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings

cargo run --release --bin ccmm_oracle
cargo run --release --example ckks_simd_4096
cargo run --release --example rns_ccmm_4096

cargo run --release --bin rns_ccmm_bench -- 2
cargo run --release --bin rns_ccmm_bench -- 4
```

If HEaaN is available:

``` bash
bash scripts/run-ccmm-cross-validation.sh
```

Finally:

``` bash
git diff --check
git status --short
```

## 13. Repeated Performance Trials

For performance characterization, run multiple independent process
invocations rather than repeatedly timing only an inner operation in one
process.

For example:

``` bash
mkdir -p results/local-r2.10-reproduction

for trial in 1 2 3 4 5; do
    cargo run --release --bin rns_ccmm_bench -- 2 \
        > "results/local-r2.10-reproduction/research-4096-2x2-trial-${trial}.txt"
done
```

Retain the raw output files.

At minimum, report:

``` text
trial count
minimum
mean
median
maximum
maximum numerical error
mean numerical error
RMS numerical error
pass/fail tolerance
```

Do not report only the fastest trial.

## 14. Reproducibility Rules

When adding or publishing a characterization result:

1.  identify the exact git commit;
2.  identify the parameter profile;
3.  state whether the profile is security-bearing;
4.  state ciphertext and evaluation-key noise distributions;
5.  state matrix dimensions and operation counts;
6.  state exactly what the timed region includes;
7.  retain raw trial outputs;
8.  report numerical error together with timing;
9.  record host/toolchain information;
10. distinguish measured observations from general performance claims.

## 15. Security Interpretation

The realistic profiles are currently research profiles.

In particular:

``` text
SECURITY_BEARING=false
```

must remain visible in public characterization until the evaluation-key
noise path, auxiliary-modulus accounting, complete estimator analysis,
and associated release gates are satisfied.

See:

``` text
docs/SECURITY_AND_PARAMETERS.md
security/r2.9b/
```

for the security and noise-model boundary.

## 16. Frozen Evidence

The principal frozen Roadmap-2 evidence is organized as:

``` text
security/r2.9b/
    R2.9 security/noise analysis and diagnostics

results/r2.10/
    R2.10 realistic matrix-characterization evidence

examples/
    public N=4096 CKKS and CCMM examples
```

R2.9 and R2.10 artifacts should be treated as historical evidence for
their corresponding commits. New measurements should be stored
separately rather than silently replacing frozen raw results.
