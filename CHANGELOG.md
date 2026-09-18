# Changelog

## v2.0.0 — 2026-09-18

`v2.0.0` completes the Roadmap-2 redesign of `ccmm-rs`.

### Added

- realistic RNS/NTT CKKS execution;
- optimized negacyclic NTT/RNS arithmetic;
- modulus chains and basis transitions;
- Grafting and hybrid transition infrastructure;
- canonical CKKS complex-slot SIMD encoding;
- depth-parametric leveled CKKS evaluation;
- Galois automorphisms, rotations, and conjugation;
- level-aware multiplication and Galois evaluation keys;
- CKKS research parameter profiles for N=4096, 8192, and 16384;
- discrete-Gaussian RLWE error infrastructure;
- shared logical RNS error projection;
- realistic N=4096 CKKS multiply/relinearize/rescale validation;
- RNS CKKS ciphertext addition;
- `RnsCkksCiphertextMatrix`;
- NTT-backed encrypted matrix multiplication;
- N=4096 2x2 and 4x4 CCMM characterization;
- reproducibility and security/parameter documentation;
- public N=4096 CKKS SIMD and CCMM examples.

### Validation

The v2.0.0 release gate includes:

- 509 passing Rust tests;
- zero Clippy warnings;
- zero Rustdoc warnings;
- deterministic reference CCMM validation;
- canonical CKKS SIMD validation;
- realistic N=4096 encrypted CCMM validation;
- five-trial 2x2 characterization;
- 4x4 scaling characterization.

### Security status

The realistic parameter profiles remain research profiles:

```text
SECURITY_BEARING=false
```

Ciphertext encryption supports discrete-Gaussian error with sigma=3.19 in
the validated realistic path.

The current evaluation-key/relinearization path remains characterized with
zero evaluation-key noise. Security-bearing noisy evaluation keys and
complete auxiliary-modulus accounting are explicitly deferred.

See `docs/SECURITY_AND_PARAMETERS.md`.

## v0.1.0 — 2026-09-17

Initial correctness-oriented release.

The v0.1.0 baseline established:

- native Rust CPMM and CCMM;
- transparent small-parameter RLWE/CKKS-style execution;
- degree-2 ciphertext multiplication;
- relinearization and rescaling;
- deterministic correctness campaigns;
- native CCMM oracle;
- HEaaN external cross-validation;
- baseline characterization.
