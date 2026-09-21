# Applications

FHE-rs application development is currently hosted inside `ccmm-rs`.
Applications should consume eBLAS or stable evaluator interfaces rather than
reproduce low-level cryptographic schedules.

## Current demonstrated applications

| Application | Privacy | Main mechanism | Status |
|---|---|---|---|
| Private linear inference | CP | encrypted features × plaintext weights | Implemented / validated |
| Private two-party matrix product | CC | encrypted matrix × encrypted matrix | Implemented / validated |
| eBLAS tensor GEMM validation | PP/CP/PC/CC | tensor mapping -> batched GEMM | Implemented / validated |

### Private linear inference

```bash
cargo run --release --bin private_linear_inference
```

The workload encrypts a feature vector and evaluates it against plaintext
model weights. The CP path requires no evaluation key or relinearization.
Historical evidence is in `results/r3.2b/`.

### Private two-party matrix product

```bash
cargo run --release --bin private_two_party_matrix_product
```

Both operands are encrypted. The current security-oriented research path uses
bounded-base Gaussian relinearization with `research-4096`. Historical
evidence is in `results/r3.2c/`. See `SECURITY_AND_PARAMETERS.md` and
`ASSURANCE.md` before making a security claim.

## Private Nonlinear / MLP Inference

`private_mlp_inference` demonstrates a two-layer encrypted inference workload
with a nonlinear activation:

- encrypted 4-element input;
- public 4x3 first-layer weights evaluated with eBLAS CP GEMM;
- element-wise encrypted square activation;
- public 3x2 second-layer weights evaluated with eBLAS CP GEMM;
- encrypted 2-element output decrypted only for validation.

The application uses the `research-8192` profile because the complete workload
consumes three CKKS levels:

```text
encrypted input
    -> eBLAS CP GEMM      level 0 -> 1
    -> square activation  level 1 -> 2
    -> eBLAS CP GEMM      level 2 -> 3
```

The square activation uses the bounded ciphertext multiplication path with
relinearization and rescaling at the active CKKS level.

The current `research-8192` profile is a research parameter set and is not
designated as security-validated. The application reports this explicitly.

Run:

    cargo run --release --bin private_mlp_inference

The retained validation output is available in
`results/r3.6/private-mlp-inference.txt`.

## Applications Being Added Next

The remaining application set focuses on:

1. proxy re-encryption as a cryptographic service.

These additions are intentionally bounded so the project can return to resilience integration after the application layer is complete.
4. proxy re-encryption as a security-service application exercising
   key-switch/re-encryption functionality rather than eBLAS;
5. cross-application characterization of correctness/error, latency, levels,
   operation counts, and backend decisions.

Proxy re-encryption is intentionally an FHE-rs application/security service,
not an eBLAS operation.

## Application development rule

```text
choose parameter profile
 -> generate required keys
 -> encode/encrypt private inputs
 -> express linear algebra through eBLAS
 -> record backend decision or explicit backend
 -> decrypt/decode
 -> compare with a cleartext oracle
 -> report error, levels, and operation counts
```

See `GETTING_STARTED.md` and `EXTENDING_FHE_RS.md`.
