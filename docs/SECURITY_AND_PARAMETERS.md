# Security and Parameter Model

This document defines the current security and parameter-claim boundary for
`ccmm-rs`.

`ccmm-rs` is a correctness-first research implementation. Security claims are
therefore separated into:

1. parameter-security evidence for the underlying RLWE problem;
2. numerical validation of the concrete CKKS evaluation path; and
3. implementation-security properties required for production deployment.

Evidence in one category does not automatically establish the others.


> **FHE-rs v3.0 scope.** FHE-rs is the platform identity; `ccmm-rs` remains the repository and Rust package name. This document separates security-oriented small-parameter CKKS evidence from correctness/performance experiments using large, research-only FFT2 parameters. A numerically successful encrypted computation is not, by itself, a security qualification.

## Research Parameter Profiles

The realistic CKKS profiles are:

| Profile | Ring degree | Initial scale | Modulus chain | Published 128-bit classical budget reference |
|---|---:|---:|---:|---:|
| `research-4096` | 4096 | `2^35` | 3 x ~35-bit primes | ~106 bits |
| `research-8192` | 8192 | `2^40` | 5 x ~40-bit primes | ~214 bits |
| `research-16384` | 16384 | `2^45` | 9 x ~45-bit primes | ~430 bits |

The published budget values are engineering references under their stated
secret, error, and estimator assumptions. They are not independent proofs of
the complete `ccmm-rs` cryptosystem.

R3.1 performs direct security analysis only for `research-4096`. The larger
profiles remain unvalidated by the equivalent R3.1 workflow.

## Research-4096 Security Model

The validated parameter-security model for `research-4096` is:

```text
PROFILE=research-4096
RING_DEGREE=4096
SLOT_COUNT=2048
CHAIN_LEVELS=3
TOP_LEVEL_Q_BITS=105
INITIAL_SCALE=2^35
SECRET_DISTRIBUTION=uniform-ternary
ERROR_DISTRIBUTION=discrete-Gaussian
ERROR_SIGMA=3.19
TARGET_CLASSICAL_SECURITY_BITS=128
ESTIMATOR=Lattice Estimator
REDUCTION_COST_MODEL=RC.MATZOV
```

The corresponding `CkksSecurityModel` records these assumptions explicitly.

The security model describes the underlying RLWE parameterization. It must not
be interpreted as a blanket statement that every execution path using the
profile has equivalent security properties.

## Secret and Error Distributions

The security analysis assumes a uniform ternary secret with coefficients in:

```text
{-1, 0, +1}
```

and discrete-Gaussian error with:

```text
sigma = 3.19
```

Deterministic secrets used in tests and reproducibility benchmarks are
functional fixtures and are not evidence that those fixtures instantiate the
assumed secret distribution.

RNS encryption samples one logical integer error coefficient and projects that
same error into every residue limb. This preserves the interpretation of a
single small RLWE error under CRT reconstruction.

The current Gaussian sampler uses floating-point probability calculations and
data-dependent sampling behavior. It is research infrastructure and is not
presented as a hardened constant-time production sampler.

## R3.1 Evaluation-Key Architecture

Roadmap R2.9 localized a major numerical failure in the original CRT-residue
evaluation-key path:

```text
Gaussian ciphertext + zero-noise evaluation key    -> stable
zero-noise ciphertext + Gaussian evaluation key    -> unstable
```

R3.1a subsequently quantified the source of the amplification. Canonical CRT
residue digits remain close to the size of the RNS primes even when the CRT
partition is made finer, causing evaluation-key error to be multiplied by
large gadget digits.

R3.1 therefore retains the original CRT gadget as a correctness and
differential-testing path but does not use it as the current noisy
security-oriented reference.

## Bounded-Base Key Switching

R3.1b introduced balanced signed bounded-base decomposition:

```text
B = 2^w
x = sum_j d_j B^j
|d_j| <= B/2
```

For `research-4096`, the characterized candidate radices were:

| base_log | Digit count | Statistical trials | Result |
|---:|---:|---:|---|
| 8 | 14 | 10 | PASS |
| 12 | 9 | 10 | PASS |
| 16 | 7 | 10 | PASS |
| 20 | 6 | 10 | PASS |

All 40 end-to-end CKKS executions passed the `1e-3` numerical tolerance with
Gaussian ciphertext error and Gaussian evaluation-key error at `sigma=3.19`.

The selected N=4096 reference operating point is:

```text
BOUNDED_BASE_LOG=20
BOUNDED_BASE=1048576
TOP_LEVEL_BOUNDED_DIGITS=6
```

This selection minimizes measured key-generation and relinearization cost
among the characterized candidates while retaining substantial numerical
margin. It is an engineering choice, not by itself a security proof.

## Modulus and Public-Sample Exposure

The bounded-base reference construction operates directly over the active CKKS
`Q` basis. It introduces no auxiliary special modulus `P`.

Exact exposure accounting is:

| Level | Composite Q | Q bits | Evaluation-key RLWE samples |
|---:|---:|---:|---:|
| 0 | 40564045502413384804664457166849 | 105 | 6 |
| 1 | 1180580361798806077441 | 70 | 4 |
| 2 | 34359697409 | 35 | 2 |

Therefore:

```text
AUXILIARY_MODULUS_PRESENT=false
```

No `P/Q` term is omitted from this bounded-base construction because no `P`
exists in this path.

A future special-modulus or hybrid key-switch implementation would require
fresh accounting for every additional exposed modulus.

## Lattice Estimator Validation

R3.1d evaluates the exact `research-4096` active moduli using:

```text
Lattice Estimator revision:
8f1ff7e20a4d3391e3badff1d76825314db225bc

reduction cost model:
RC.MATZOV

dimension:
n = 4096

secret:
uniform ternary

error:
discrete Gaussian, sigma = 3.19

estimator sample model:
m = oo
```

The number of exposed RLWE evaluation-key ciphertexts is recorded separately.
It is not incorrectly mapped to the estimator's scalar-LWE `m` parameter.

Observed estimates are:

| Level | Primal USVP | Primal BDD | Dual hybrid | Primal hybrid |
|---:|---:|---:|---:|---:|
| 0 | 130.8 | 130.3 | 130.8 | 189.0 |
| 1 | 204.0 | 203.2 | 202.0 | 301.3 |
| 2 | 443.4 | 441.3 | 428.5 | 676.0 |

The binding result is:

```text
TARGET_BITS=128
BINDING_LEVEL=0
BINDING_ATTACK=primal_bdd
BINDING_ESTIMATE_BITS=130.3
SECURITY_MARGIN_BITS=2.3
R3_1D_ESTIMATOR_GATE=PASS
```

Accordingly, the underlying uniform-ternary-secret RLWE parameterization meets
the targeted 128-bit classical-security gate under the documented Lattice
Estimator methodology and MATZOV reduction-cost model.

This is the precise parameter-security claim.

## Circular/KDM Claim Boundary

The bounded multiplication key contains RLWE encryptions of secret-dependent
values of the form:

```text
B^j * s^2
```

The Lattice Estimator analysis characterizes the underlying RLWE hardness. It
does not independently prove circular/KDM security for these secret-dependent
evaluation-key messages.

The bounded-base evaluation-key construction is supported separately by the
R3.1 correctness, decomposition, noise, and end-to-end numerical evidence.

Consequently, `ccmm-rs` does not claim an independent proof of circular/KDM
security for the evaluation-key construction.

## End-to-End Bounded Gaussian CKKS

R3.1 validates the following realistic pipeline at N=4096:

```text
canonical CKKS encode
-> Gaussian RNS encryption
-> NTT ciphertext multiplication
-> bounded-base Gaussian relinearization
-> CKKS rescale
-> RNS decryption
-> canonical CKKS decode
```

Both ciphertext and evaluation-key errors use:

```text
DiscreteGaussian { sigma: 3.19 }
```

The bounded-base path passes the numerical validation gates across the
characterized radices and randomized trials.

## Realistic N=4096 CCMM

R3.1e extends the bounded-Gaussian path to direct encrypted
ciphertext-ciphertext matrix multiplication.

The validated 2x2 experiment uses:

```text
PROFILE=research-4096
RING_DEGREE=4096
SLOT_COUNT=2048
BOUNDED_BASE_LOG=20
CIPHERTEXT_ERROR_SIGMA=3.19
EVALUATION_KEY_ERROR_SIGMA=3.19
SCALAR_CKKS_MULTIPLIES=8
CIPHERTEXT_ADDITIONS=4
```

A representative final run reports:

```text
MAX_MATRIX_ERROR=8.555989805537e-10
MAX_IMAGINARY_RESIDUAL=2.512294996312e-7
TOLERANCE=2.000000000000e-3
R3_1E3_BOUNDED_CCMM_STATUS=PASS
```

The executable deliberately reports:

```text
UNDERLYING_RLWE_TARGET_SECURITY_BITS=128
CIRCULAR_KDM_ASSUMPTION=required
HARDENED_SAMPLER=false
```

rather than describing the complete execution as unconditionally
"128-bit secure."

## Current Claim

The supported R3.1 statement is:

> The `research-4096` underlying uniform-ternary-secret RLWE parameterization
> meets the targeted 128-bit classical-security gate under the documented
> Lattice Estimator methodology and MATZOV cost model. The N=4096 bounded-base
> CKKS evaluation path has been validated end-to-end with discrete-Gaussian
> ciphertext and evaluation-key error at sigma 3.19, including realistic
> ciphertext-ciphertext matrix multiplication.

This statement is intentionally narrower than a production-security claim.

## Proxy Re-Encryption Prototype Claim Boundary

The R3.6 proxy re-encryption application is a correctness prototype built from
the generic RNS source-to-target key-switch primitive. It demonstrates that a
ciphertext under a source secret can be transformed by a proxy and subsequently
decrypted under a target secret while preserving the encoded plaintext.

The current generic CRT-block key-switch path is retained primarily as a
correctness and differential-testing mechanism. In the PRE experiment, adding
Gaussian error to the re-encryption key at the `research-4096` parameter point
caused unacceptable numerical error amplification. The retained passing
prototype therefore uses a zero-noise re-encryption key and reports:

```text
REENCRYPTION_KEY_SECURITY_BEARING=false
PRE_SECURITY_CLAIM=false
CCA_SECURITY_CLAIM=false
COLLUSION_RESISTANCE_CLAIM=false
UNIDIRECTIONAL_SECURITY_CLAIM=false
```

This result does not weaken the separately characterized bounded-base
multiplication/relinearization path. A security-bearing PRE service requires an
appropriate bounded or otherwise noise-controlled generic source-to-target
key-switch construction, followed by its own security analysis and validation.


<!-- FHE_RS_V3_SECURITY_EXTENSION_BEGIN -->
## FHE-rs v3.0 Large-Scale FFT2 Parameter and Claim Boundary

The encrypted FFT2 research program evaluates **encrypted input data with a public Fourier transform**. Public material includes the transform dimensions, FFT stage schedule, twiddle factors, public masks, and declared packing/layout decisions. The ciphertexts protect the input values under the applicable cryptographic assumptions; the execution does **not** claim to hide the transform, its control flow, ciphertext counts, or observable runtime/memory-access patterns. Secret transforms or private filters are different threat models and require separate designs and analyses.

### Large FFT2 research configurations

The following are *experimental execution configurations*, **not 128-bit-secure parameter profiles**. The smaller `research-4096` security estimate above must not be transferred to them merely because they use CKKS or RNS arithmetic.

| Encrypted FFT2 workload | Ring degree | CKKS slots / ciphertext | Active RNS limbs at top level | FFT levels consumed | Security claim |
| --- | ---: | ---: | ---: | ---: | --- |
| 256 x 256 (R30.20K) | 65,536 | 32,768 | 18 | 16 | Research only; `FFT_SECURITY_BEARING=false` |
| 1024 x 1024 (R17 1K) | 65,536 | 32,768 | 22 | 20 | Research only; no 128-bit claim |
| 4096 x 4096 (R21/R31.5 hierarchical) | 65,536 | 32,768 | 26 | 24 | Research only; no 128-bit claim |

The 256 x 256 R30.20K configuration reports an approximately 990-bit top-level modulus and a baseline zero-noise evaluation-key mode. The 26-limb 4K parameter pool has approximately 1,430 aggregate modulus bits. These bit counts are **not security estimates**. The exact prime sequence, secret distribution, encryption and evaluation-key error distributions, key-switch gadget, auxiliary moduli (if any), and evaluation-key exposure must be recorded for each future security-bearing profile and submitted to a separate estimator analysis.

A successful FFT2 numerical test establishes functional correctness to its stated tolerance. It does not validate IND-CPA security, evaluation-key circular/KDM assumptions, side-channel resistance, or an overall 128-bit classical-security target. In particular, zero-noise evaluation keys are research fixtures and **must not** be described as equivalent to the R3.1 bounded-base Gaussian reference.

### Evaluation-key families and security inheritance

- **Baseline/CRT and zero-noise evaluation-key paths:** retained for correctness, regression, and differential benchmarking. Their numerical PASS results are not security-bearing claims.
- **R3.1 bounded-base Gaussian reference:** a distinct, characterized N=4096 construction with the conditional underlying-RLWE estimator result documented above. Its numerical and parameter-security evidence does not automatically extend to the large FFT2 configurations.
- **Hybrid or alternative key-switch paths:** optimization candidates whose security status depends on the exact deployed implementation and parameters. Each requires explicit accounting for gadget digits, all exposed RLWE samples, any special modulus, ciphertext and evaluation-key error, and estimator inputs before a security claim can be made.

### Parallel execution, working sets, and timing

Worker counts, thread-pool settings, inter-ciphertext concurrency, and local FFT concurrency are **performance configuration metadata**, not cryptographic security parameters. Benchmark reports should distinguish configured concurrency from the measured peak number of simultaneously running threads.

Memory reporting must distinguish (i) serialized or logical ciphertext payload, (ii) live cryptographic working set including evaluation keys and temporary RNS/NTT buffers, (iii) operating-system peak resident set size (RSS), and (iv) platform-specific process memory footprint. These measurements are not interchangeable. Where a measurement is missing, it must remain explicitly unreported rather than be inferred from ciphertext count.

For the v3.0 encrypted FFT2 evidence, report separately: key generation, encode/encrypt, FFT evaluation, decrypt/decode, and total application time; relative L2 and maximum absolute error against the corresponding plaintext oracle; exact thread configuration; and peak RSS. Comparison across R17, R21, R30, and R31.5 requires matching workload, input, parameter profile, timing boundary, and software revision. A kernel microbenchmark speedup is not an end-to-end FFT speedup.

### Security qualification checklist for future large FFT profiles

Before changing any `SECURITY_BEARING=false` marker or making a 128-bit claim, document and validate:

1. The exact ring dimension, RNS primes and total modulus, scale/level schedule, and any auxiliary key-switch modulus.
2. Secret and error distributions for encryption **and every evaluation-key family**, including a non-degenerate noise model.
3. Gadget decomposition, key-switch/relinearization/rotation key counts, and the exposure model used for the lattice estimate.
4. A reproducible Lattice Estimator revision, attack and reduction-cost models, estimated binding security, and margin to the stated target.
5. The additional circular/KDM assumptions for secret-dependent evaluation keys, and the implementation-level limits of samplers, constant-time behavior, and side channels.
6. End-to-end numerical validation under the same security-oriented parameter and error configuration, rather than substituting zero-noise correctness fixtures.

### v3.0 FFT evidence locations

Relevant frozen or campaign artifacts include:

```text
results/icpp_r21_campaign/
results/fft_milestones/
results/eblas_fft_evidence/icpp_4k_baseline_r21/
results/eblas_fft_evidence/r31_5_frozen/
```

These locations contain historical and optimized runs with different configurations. Their contents must be checked before selecting a publication table row; the paths themselves do not certify that all metrics are complete or directly comparable. Preserve frozen evidence and its original security markers without rewriting it retroactively.
<!-- FHE_RS_V3_SECURITY_EXTENSION_END -->

## What Is Not Claimed

The current artifact does **not** claim:

- an independent proof of circular/KDM security for evaluation keys encrypting
  secret-dependent messages;
- a hardened or constant-time Gaussian sampler;
- implementation-level side-channel resistance;
- that the R3.1 analysis automatically applies to `research-8192` or
  `research-16384`;
- that every legacy/reference evaluation path has the same security properties
  as the bounded-base path;
- third-party cryptographic or implementation audit;
- production deployment readiness.

## Evidence and Provenance

The principal R3.1 evidence is retained under:

```text
results/r3.1a/
results/r3.1b/
security/r3.1c/
security/r3.1d/
results/r3.1e/
```

Historical R2.9 and R2.10 artifacts remain unchanged. Statements such as
`SECURITY_BEARING=false` and zero-noise evaluation-key measurements in those
artifacts describe the implementation state at the time those experiments
were frozen; they are retained as provenance rather than rewritten
retroactively.

The original R2.9 evidence remains under:

```text
security/r2.9b/
```

and the original realistic matrix characterization remains under:

```text
results/r2.10/
```

## Implementation Status

R3.1 status is:

```text
R3.1a  CRT evaluation-key noise localization          DONE
R3.1b  bounded-base Gaussian key switching             DONE
R3.1c  exact security-exposure accounting              DONE
R3.1d  Lattice Estimator validation                    DONE
R3.1e  security model + bounded execution + N4096 CCMM DONE
```

R3.1 establishes a quantitatively characterized bounded-base reference path.
Further hardening, additional parameter profiles, alternative key-switch
architectures, and independent review remain separate work.
