# Security and Parameter Model

This document defines the security and parameter-claim boundary for the
current `ccmm-rs` Roadmap-2 research implementation.

`ccmm-rs` is a correctness-first research artifact. The presence of
realistic ring dimensions, RNS modulus chains, discrete-Gaussian
ciphertext error, and security-analysis tooling does **not** by itself
imply that the current parameter profiles are production-security
parameter sets.

## Parameter Classes

The CKKS parameter infrastructure distinguishes correctness-oriented and
research-oriented configurations.

The current realistic research profiles are:

  ---------------------------------------------------------------------------------
  Profile               Ring degree  Initial scale   Modulus-chain        Published
                                                             shape          128-bit
                                                                          classical
                                                                     modulus-budget
                                                                          reference
  ------------------ -------------- -------------- --------------- ----------------
  `research-4096`              4096         `2^35`    3 × \~35-bit       \~106 bits
                                                            primes 

  `research-8192`              8192         `2^40`    5 × \~40-bit       \~214 bits
                                                            primes 

  `research-16384`            16384         `2^45`    9 × \~45-bit       \~430 bits
                                                            primes 
  ---------------------------------------------------------------------------------

The modulus-budget reference values above correspond to the published
FHE security-guideline methodology used during Roadmap R2.9 under the
stated uniform-ternary-secret assumptions.

They are **budget references**, not independent security proofs for the
complete `ccmm-rs` cryptosystem.

The current profiles remain:

``` text
SECURITY_BEARING=false
```

and their `security_model` is intentionally unset.

## Current Research-4096 Profile

The N=4096 profile used for realistic encrypted matrix-multiplication
characterization has:

``` text
PROFILE=research-4096
RING_DEGREE=4096
SLOT_COUNT=2048
CHAIN_LEVELS=3
TOTAL_MODULUS_BITS=105
INPUT_SCALE=2^35
PARAMETER_CLASS=Research
SECURITY_BEARING=false
```

Its modulus chain is constructed from three approximately 35-bit
NTT-friendly primes.

The aggregate top-chain modulus is approximately 105 bits, leaving only
a small margin relative to the published 128-bit classical
modulus-budget reference used in the R2.9 methodology.

This comparison is useful for parameter engineering, but it must not be
presented as a blanket 128-bit security claim.

## Security-Model Representation

The Roadmap-2 parameter model can represent security assumptions
explicitly through:

``` rust
pub enum CkksSecretDistribution {
    UniformTernary,
}

pub enum CkksErrorDistribution {
    DiscreteGaussian { sigma: f64 },
}

pub struct CkksSecurityModel {
    pub classical_security_bits: u32,
    pub secret_distribution: CkksSecretDistribution,
    pub error_distribution: CkksErrorDistribution,
    pub estimator: &'static str,
    pub reduction_cost_model: &'static str,
}
```

A parameter profile is considered security-bearing only when an explicit
security model is attached to it.

The current realistic profiles deliberately do not attach one.

## Secret Distribution

The research implementation uses ternary secrets with coefficients in:

``` text
{-1, 0, +1}
```

with all-zero secrets rejected in the normal secret-generation path.

The security-guideline comparison used during R2.9 was based on the
published uniform-ternary-secret model.

Deterministic ternary secrets used in tests, examples, and
reproducibility benchmarks are functional test fixtures. They are
**not** evidence that those deterministic secrets instantiate the
assumed security distribution.

## Error Distributions

The RLWE infrastructure supports an explicit error-distribution
abstraction:

``` rust
pub enum ErrorDistribution {
    BoundedUniform { bound: i64 },
    DiscreteGaussian { sigma: f64 },
}
```

The older `noise_bound` mechanism remains available as a
correctness/reference substrate.

For the current realistic ciphertext validation, the research
configuration uses:

``` text
DiscreteGaussian { sigma: 3.19 }
```

The Gaussian sampler is intended for research validation. It currently
uses floating-point probability calculations and data-dependent sampling
behavior and is not presented as a hardened constant-time production
sampler.

## Shared RNS Error Semantics

RNS encryption samples one logical integer error coefficient and
projects that same error into every RNS limb.

This is important: independently sampling unrelated error values in each
residue limb would not represent a single small RLWE error under CRT
reconstruction.

Uniform `a` values may be sampled independently in each RNS limb because
the resulting residues represent a uniform value in the CRT product
ring.

## Gaussian Ciphertext Validation

Roadmap R2.9 validated Gaussian ciphertext encryption on the realistic
N=4096 CKKS path.

The validated scope includes:

``` text
canonical CKKS encode
-> Gaussian-noise RNS encryption
-> NTT-backed ciphertext multiplication
-> relinearization with a zero-noise evaluation key
-> rescale
-> RNS decryption
-> canonical CKKS decode
```

A representative validated run used:

``` text
PROFILE=research-4096
GAUSSIAN_ENCRYPTION_SIGMA=3.19
INPUT_SCALE=34359738368
OUTPUT_SCALE≈34360066050.125
MAX_SLOT_ERROR≈1.36e-6
STATUS=PASS
```

R2.10 subsequently used Gaussian ciphertext encryption with `sigma=3.19`
for realistic N=4096 encrypted matrix multiplication.

## Evaluation-Key Limitation

The current security-bearing limitation is concentrated in the
evaluation-key/relinearization path.

R2.9 performed an A/B localization experiment:

``` text
Case A:
Gaussian-noise ciphertexts
+ zero-noise multiplication key
-> numerically stable

Case B:
zero-noise ciphertexts
+ Gaussian-noise multiplication key
-> unacceptable numerical amplification
```

Case A produced slot error on the order of `1e-6`.

Case B produced error on the order of `1e5`.

The experiment therefore localized the current numerical failure to the
noisy evaluation-key/relinearization path.

The evidence is consistent with excessive noise amplification in the
current evaluation-key/gadget construction, but `ccmm-rs` does not claim
a more specific root cause without a complete decomposition analysis.

For this reason, current realistic examples and characterization use:

``` text
EVALUATION_KEY_NOISE_BOUND=0
```

This is an explicit documented limitation.

## Deferred Evaluation-Key Work

A security-bearing evaluation-key path remains future work.

Likely design directions include a conventional bounded-base
decomposition or a hybrid/special-modulus key-switch construction.

Any such design must include:

-   explicit gadget/decomposition semantics;
-   evaluation-key noise analysis;
-   complete auxiliary-modulus accounting;
-   correct `P/Q` treatment where a special modulus is used;
-   numerical validation across supported levels;
-   security-estimator accounting for the complete modulus exposure;
-   regression against the current correctness/reference path.

No current documentation should imply that this work has already been
completed.

## Auxiliary-Modulus Accounting

Security estimates must account for every modulus exposed by the
cryptographic construction.

The current R2.9 security work explicitly leaves complete auxiliary
`P/Q` accounting deferred until the evaluation-key/key-switch
architecture is finalized.

Therefore, the top CKKS ciphertext modulus `Q` alone must not be treated
as sufficient evidence for a production security designation if future
key switching introduces additional modulus exposure.

## Published Security-Guideline Methodology

The primary parameter-budget methodology used during R2.9 follows the
FHE security-guideline work supplied with the project.

The relevant methodology uses the Lattice Estimator with:

``` text
reduction cost model: RC.MATZOV
estimator commit: 8f1ff7e
```

and evaluates attacks including primal and dual families under the
assumptions specified by the guideline methodology.

For a uniform ternary secret, the published 128-bit classical maximum
`log2(q)` reference values used during R2.9 were approximately:

``` text
N=4096   -> 106 bits
N=8192   -> 214 bits
N=16384  -> 430 bits
```

These values are meaningful only under their associated assumptions and
methodology.

They should not be detached from those assumptions and presented as
universal security thresholds.

## Lattice Estimator Tooling

The repository security workflow supports direct Lattice Estimator
evaluation for experimental or alternative parameter sets.

The intended wording for the current artifact is:

> Standard parameter profiles are selected against published FHE
> security guidelines under their stated assumptions. The `ccmm-rs`
> security-analysis tooling additionally supports direct Lattice
> Estimator evaluation for experimental or alternative parameter sets.

Direct estimator runs are analysis evidence. They do not replace the
need to specify the complete cryptographic construction, secret/error
distributions, modulus exposure, attack model, and estimator provenance.

## What Is Validated

The current Roadmap-2 evidence supports the following statements:

-   realistic N=4096/8192/16384 research parameter profiles exist;
-   their modulus-chain sizes can be compared with published
    security-guideline budgets;
-   canonical CKKS encoding/decoding works at realistic N;
-   optimized RNS/NTT arithmetic works at realistic N;
-   Gaussian ciphertext error with `sigma=3.19` is numerically viable on
    the validated N=4096 path;
-   one logical Gaussian error is consistently projected across RNS
    limbs;
-   NTT-backed multiply/relinearize/rescale works with the current
    zero-noise evaluation-key path;
-   realistic N=4096 2x2 and 4x4 encrypted matrix multiplication passes
    the current numerical validation gates;
-   the noisy evaluation-key failure has been experimentally localized.

## What Is Not Claimed

The current artifact does **not** claim:

-   a production-ready 128-bit-secure CKKS instantiation;
-   that `research-4096`, `research-8192`, or `research-16384` are
    security-bearing profiles;
-   a hardened or constant-time Gaussian sampler;
-   security-valid noisy evaluation keys;
-   complete special-modulus or auxiliary-modulus security accounting;
-   resistance to implementation-level side channels;
-   third-party cryptographic or security audit;
-   production deployment readiness.

## Evidence and Reproducibility

Security/noise evidence and tooling are retained under:

``` text
security/r2.9b/
```

Important artifacts include the R2.9 security/noise validation summary,
parameter inventory and analysis tooling, and the A/B evaluation-key
diagnostics.

Realistic matrix-characterization evidence is retained under:

``` text
results/r2.10/
```

The public README intentionally mirrors the claim boundaries defined
here.

## Release Gate for a Future Security-Bearing Profile

A future profile should not be designated security-bearing until, at
minimum:

1.  the complete encryption and evaluation-key distributions are
    specified;
2.  the evaluation-key/key-switch construction is noise viable;
3.  all ciphertext and auxiliary moduli are included in security
    accounting;
4.  the full parameter set is evaluated under a documented estimator
    version and cost model;
5.  the assumptions match the intended secret and error distributions;
6.  realistic end-to-end numerical tests pass with those
    security-bearing distributions;
7.  implementation-level sampler and secret-handling requirements are
    addressed;
8.  the security claim is independently reviewed.

Until those conditions are met, the realistic profiles remain research
profiles and `SECURITY_BEARING=false` remains the correct designation.
