# Assurance and Evidence

FHE-rs treats implementation status and assurance status as separate
properties. Implementation does not imply production hardening, formal
verification, external audit, or universal characterization.

## Evidence classes

| Evidence | Meaning |
|---|---|
| Unit/integration tests | tested implementation invariants |
| Reference/differential validation | agreement with cleartext or independent path |
| External cross-validation | scoped comparison with an external implementation |
| Numerical campaign | error within stated tolerance for stated campaign |
| Security-parameter analysis | mathematical parameters evaluated under explicit assumptions |
| Characterization | measured/derived performance or operation counts for stated configuration |
| Production assurance | constant-time behavior, hardened randomness, audit, deployment review, etc. |

Evidence in one category does not establish another.

## Current matrix

| Capability | Status | Assurance boundary |
|---|---|---|
| Reference RLWE/CKKS | Implemented | deterministic oracle; scoped historical HEaaN cross-validation |
| RNS/NTT CKKS | Implemented/validated | research implementation |
| Gaussian ciphertext error | Implemented/validated | sampler not hardened constant-time |
| Bounded-base eval key | Implemented/validated | circular/KDM assumption remains explicit |
| `research-4096` RLWE parameters | Gate passed | targeted 128-bit classical gate under documented Lattice Estimator + MATZOV methodology |
| Composite RNS scaling | Implemented/validated | correctness evidence, not independent security proof |
| eBLAS PP/CP/PC/CC | Implemented/validated | path-specific semantic evidence |
| Structured CCMM | Implemented/characterized | measured speedup limited to tested frontier |
| Backend policy | Implemented/validated | profile/implementation specific |
| Batched GEMM | Implemented/validated | semantic aggregation; no SIMD/parallel speedup claim |
| Tensor GEMM | Implemented/validated | mapping adds no crypto primitive |
| Bootstrapping | Planned | not implemented |
| Integer/discrete CKKS | Planned | not implemented |
| Third-party audit / production side-channel hardening | Not claimed | none |

## Precise security statement

The `research-4096` underlying uniform-ternary-secret RLWE parameterization
meets the targeted 128-bit classical-security gate under the documented
Lattice Estimator methodology and MATZOV cost model. The binding reported
estimate is approximately **130.3 classical bits**.

This does not independently prove circular/KDM security, constant-time or
side-channel safety, hardened Gaussian sampling, equivalent security for all
profiles/legacy paths, production readiness, or third-party audit.

## Performance claim discipline

For structured CCMM, MKN -> MN relinearization/rescale reduction is an
algorithmic schedule property. The 2.3–3.45x kernel speedup is measured. Do
not turn the 4–16x postprocessing-count reduction into a 4–16x runtime claim.
Likewise, batching is not claimed as acceleration; PC overhead is not
generalized to packed layouts; software width timing is not generalized to
hardware; and backend policy is not generalized beyond the characterized
implementation.

## Evidence expected for a new capability

Provide an API/semantic contract, positive and relevant negative tests, a
cleartext/reference comparison, realistic-profile validation, numerical/error
characterization where applicable, operation accounting, measured performance
when performance is claimed, explicit security/assurance limitations, and a
reproducibility artifact.
