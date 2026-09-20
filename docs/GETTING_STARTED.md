# Getting Started with FHE-rs

FHE-rs is a Rust library for building applications that compute on encrypted data.
The project is currently developed inside `ccmm-rs`; the repository and Cargo package
names remain unchanged at R3.5 for history/reproducibility.

## Build and validate

```bash
cargo fmt --all -- --check
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
git diff --check
```

## Run an application

```bash
cargo run --release --bin private_linear_inference
cargo run --release --bin private_two_party_matrix_product
```

## Understand the stack

```text
application -> eBLAS/stable evaluator -> backend schedule -> leveled CKKS -> RNS/NTT
```

Use eBLAS for linear algebra. Choose PP, CP, PC, or CC according to which
operands are private. Do not manually reconstruct CPMM/CCMM schedules in
application code.

## Choose Parameters and Track CKKS State

Current realistic work primarily uses `research-4096`. Read
`SECURITY_AND_PARAMETERS.md` before making security statements. Track level
and scale as semantic state. ADD preserves level; SCALE and AXPY consume one
level; CP GEMM rescales once per output; scalar CC postprocesses each scalar
product; structured CCMM postprocesses once per output. Bootstrapping is not
currently available, so applications must fit the leveled modulus budget.

## Validate a new application

Every new application should have a cleartext oracle, reproducible inputs, an
explicit tolerance, level/scale checks, operation-count reporting where
meaningful, a retained result artifact, and the standard repository gate.
Performance results should state profile, host, build mode, measurement scope,
and whether setup/key generation is included.

## Next reading

- `EBLAS.md` — linear algebra and backend policy
- `APPLICATIONS.md` — demonstrated workloads
- `ASSURANCE.md` — evidence and claim boundaries
- `SECURITY_AND_PARAMETERS.md` — security parameters
- `RNS_ARCHITECTURE.md` — RNS/composite-scaling architecture
- `REPRODUCIBILITY.md` — reproduction procedures
- `EXTENDING_FHE_RS.md` — adding capabilities
