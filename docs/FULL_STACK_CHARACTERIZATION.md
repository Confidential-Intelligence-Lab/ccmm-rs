# FHE-rs Full-Stack Characterization

## Objective

Characterize FHE-rs across modular arithmetic, NTT, RNS,
CKKS/RLWE, structured matrix multiplication, eBLAS, and eFFT.

The paper studies correctness, computational cost, scalability,
cryptographic parameterization, and reproducibility across layers.

## Experimental scope

The initial ring-degree registry contains:

- 2048
- 4096
- 8192
- 16384
- 32768
- 65536

These are candidate degrees, not a claim that every operation
supports every degree or that every configuration is secure.

The target for this characterization is 128-bit classical security.
Unverified security profiles must not be represented as qualified.

## Metrics

Each applicable benchmark should report:

- Correctness and numerical error
- Operation latency and throughput
- Ring degree, slots, scale, levels, and Q/P parameters
- Key generation and evaluation-key footprint
- Memory usage and parallelism
- Operation counts and amortization
- Hardware, toolchain, revision, and execution configuration

Missing metrics must be reported as unavailable, not estimated.

## Evidence classification

- **Measured:** Existing experimental measurements with provenance.
- **Validated:** Functional correctness established; performance
  evidence may be incomplete.
- **Planned:** Workloads or verification not yet completed.

## Existing evidence

The repository already documents:

- R30 NTT benchmark infrastructure.
- CKKS/RLWE correctness and security-oriented characterization.
- Structured CC matrix multiplication and eBLAS characterization.
- Encrypted FFT experiments and prepared-operand optimization.

Existing measurements must retain their original configuration,
security qualification, and experimental identifiers.

## Randomness and security validation

Secure execution uses OS-entropy-seeded cryptographic randomness.
Randomness implementation validation and RLWE parameter-security
verification are separate requirements.

Existing research-4096 parameter-security evidence must not be
generalized to the full six-degree characterization registry.

## Execution policy

Benchmark execution is manual. No FFT or UHD experiment may be
started implicitly by documentation or evidence-aggregation scripts.

New UHD performance experiments are outside this characterization's scope.

## Primary references

- `benchmarks/r30/manifest.json`
- `benchmarks/r30/security_profiles.json`
- `docs/REPRODUCIBILITY.md`
- `docs/SECURITY_AND_PARAMETERS.md`
- `docs/EBLAS.md`
