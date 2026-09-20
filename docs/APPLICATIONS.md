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

## R3.6 direction

R3.6 expands the application layer while keeping scope finite enough to return
promptly to resilience research. Target application classes are:

1. private linear/logistic inference through eBLAS;
2. a small private MLP with explicit depth/level accounting;
3. tensor / pointwise-convolution workloads using tensor-to-batched-GEMM;
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
