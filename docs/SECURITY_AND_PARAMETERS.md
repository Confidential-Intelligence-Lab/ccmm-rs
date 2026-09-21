# Security and Parameter Model

This document defines the current security and parameter-claim boundary for
`ccmm-rs`.

`ccmm-rs` is a correctness-first research implementation. Security claims are
therefore separated into:

1. parameter-security evidence for the underlying RLWE problem;
2. numerical validation of the concrete CKKS evaluation path; and
3. implementation-security properties required for production deployment.

Evidence in one category does not automatically establish the others.

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
