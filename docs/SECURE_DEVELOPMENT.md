# Secure Development

This document defines the engineering assurance baseline for FHE-rs / ccmm-rs.

The objective is broader than memory safety. Encrypted-computing software can
remain numerically plausible while violating a level, scale, basis,
representation, randomness, or information-flow contract. Those contracts are
therefore part of correctness.

## 1. Build and dependency reproducibility

The repository tracks `Cargo.lock`.

CI and release validation use locked dependency resolution:

```bash
cargo metadata --locked --format-version 1
cargo check --locked --release --all-targets
cargo clippy --locked --release --all-targets -- -D warnings
cargo test --locked --release
```

The Rust toolchain used by the project is pinned in `rust-toolchain.toml`.

A dependency update is an explicit source change and must be reviewed as such.

## 2. Safe Rust baseline

The core crate currently forbids unsafe Rust:

```rust
#![forbid(unsafe_code)]
```

Do not weaken this crate-wide policy to accommodate a new backend.

If future SIMD, CUDA, FPGA, PIM, photonics, device-runtime, or FFI integration
requires unsafe Rust, isolate the unsafe boundary in the smallest practical
backend crate or module.

Every unsafe boundary must document:

- why unsafe code is necessary;
- caller and callee invariants;
- pointer, lifetime, alignment, ownership, and aliasing assumptions;
- buffer-size and integer-range assumptions;
- concurrency assumptions;
- failure behavior;
- tests that exercise the boundary.

## 3. Information classification

Use explicit operand classes:

- **secret** — plaintexts, secret keys, secret-dependent state;
- **encrypted** — ciphertexts derived from secret information;
- **public** — values intentionally available to the evaluator;
- **derived-public** — metadata that is public by design but derived from a
  public execution contract.

Examples of public FFT data include transform shape, stage span, public
twiddle factors, and public masks.

A change that moves a value from secret/encrypted to public is a
security-sensitive API change.

## 4. Randomness

Cryptographic randomness must come from an explicitly supplied cryptographic
RNG or a repository-approved cryptographic RNG abstraction.

Deterministic seeds are appropriate for tests and reproducibility fixtures,
but must be visibly confined to those uses.

Do not silently introduce deterministic production randomness, log secret RNG
state, reuse test seeds as production secrets, or substitute a
non-cryptographic RNG in a cryptographic path.

## 5. CKKS state invariants

Encrypted operators must preserve or explicitly transition:

- modulus-chain level;
- active RNS basis;
- scale;
- ring degree;
- slot representation and layout;
- evaluation-key level and basis.

Operations that consume a level must expose that transition in tests.

## 6. Numerical correctness

Encrypted numerical kernels require an independent semantic reference.

Where applicable, tests should check:

- decoded result against a cleartext oracle;
- relative L2 error;
- maximum absolute error;
- inactive-slot leakage/error;
- expected output scale;
- expected output level and basis;
- expected physical slot layout.

Passing numerical tolerance alone is not sufficient if the level, scale,
basis, or representation contract is wrong.

## 7. Structural correctness and cost

Where relevant, preserve evidence for:

- ciphertext count;
- rotations;
- ciphertext-public multiplications;
- ciphertext-ciphertext multiplications;
- relinearizations;
- additions;
- transposes or global permutations;
- levels consumed.

Implementation-derived structural counts and measured results must be labeled
separately.

## 8. Parameter claims

Distinguish:

- functional validation parameters;
- research parameters;
- security-bearing parameters.

A profile reporting `SECURITY_BEARING=false` must never be described as a
production-security parameter set.

Changing modulus sizes, ring degree, error distribution, secret distribution,
gadget decomposition, auxiliary moduli, or evaluation-key construction is a
security-sensitive change.

## 9. Input validation and arithmetic

Prefer checked arithmetic for shape, allocation, byte-count, and dimension
calculations where overflow could invalidate a safety or cryptographic
assumption.

Reject invalid dimensions, radix/decomposition shapes, incompatible RNS bases,
invalid level transitions, unsupported NTT parameters, and malformed execution
plans.

## 10. Secret exposure

Never intentionally log:

- secret keys;
- plaintext user data;
- secret seed material;
- decrypted intermediate values from a production path.

## 11. Dependencies and supply chain

Run:

```bash
cargo audit
cargo deny check
```

The `deny.toml` policy checks advisories, wildcard dependency requirements,
duplicate crate versions, approved registries and git sources, and accepted
licenses.

Do not add a blanket ignore simply to make CI green.

## 12. Review expectations

Additional scrutiny is required for changes affecting cryptographic parameters,
secret/randomness handling, NTT/RNS arithmetic, evaluation keys, rescaling,
relinearization, packed-slot mapping, serialization, dependency trust, CI
security policy, or future unsafe/FFI boundaries.

## 13. Local assurance baseline

Before submitting a change:

```bash
cargo fmt --all -- --check
cargo metadata --locked --format-version 1 > /dev/null
cargo check --locked --release --all-targets
cargo clippy --locked --release --all-targets -- -D warnings
cargo test --locked --release --lib
cargo audit
cargo deny check
```

Run relevant numerical and cryptographic validation binaries for affected
subsystems.

## 14. Evidence

Reproducibility runners produce raw evidence, structured summaries, and
metadata. Measured evidence and implementation-derived structural evidence
must remain distinguishable and identify the source commit and parameter
contract.
