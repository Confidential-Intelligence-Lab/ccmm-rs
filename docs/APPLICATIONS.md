# Applications

FHE-rs application development is currently hosted inside `ccmm-rs`.
Applications should consume eBLAS or stable evaluator interfaces rather than
reproduce low-level cryptographic schedules.

## Current demonstrated applications

| Application | Privacy / service model | Main mechanism | Status |
|---|---|---|---|
| Private linear inference | CP | encrypted features x plaintext weights | Implemented / validated |
| Private two-party matrix product | CC | encrypted matrix x encrypted matrix | Implemented / validated |
| Private nonlinear / MLP inference | CP + nonlinear CC activation | two CP GEMMs with encrypted square activation | Implemented / validated |
| Private pointwise convolution | CP | NHWC lowering -> encrypted x plaintext GEMM | Implemented / validated |
| eBLAS tensor GEMM validation | PP/CP/PC/CC | tensor mapping -> batched GEMM | Implemented / validated |
| Proxy re-encryption prototype | security service | generic RNS source-to-target key switch | Correctness prototype validated |

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

### Private nonlinear / MLP inference

`private_mlp_inference` demonstrates a two-layer encrypted inference workload
with an encrypted square activation:

```text
encrypted input
    -> eBLAS CP GEMM      level 0 -> 1
    -> square activation  level 1 -> 2
    -> eBLAS CP GEMM      level 2 -> 3
```

The application uses `research-8192` because the complete workload consumes
three CKKS levels. That profile is a research parameter set and is not
designated as security-validated.

```bash
cargo run --release --bin private_mlp_inference
```

Retained evidence:
`results/r3.6/private-mlp-inference.txt`.

### Private pointwise convolution

`private_pointwise_convolution` demonstrates encrypted 1x1 convolution with an
NHWC `[1,2,2,3]` input and public `[1,1,3,2]` filter. The tensor is lowered to
a `[4,3]` matrix, evaluated through eBLAS CP GEMM against `[3,2]` public
weights, and restored to NHWC `[1,2,2,2]`.

The tensor transformation is a layout mapping only. The application introduces
no new cryptographic kernel and does not claim CKKS SIMD packing.

```bash
cargo run --release --bin private_pointwise_convolution
```

Retained evidence:
`results/r3.6/private-pointwise-convolution.txt`.

### Proxy re-encryption correctness prototype

`proxy_reencryption` demonstrates an Alice-to-Bob proxy re-encryption service
constructed from the existing generic RNS source-to-target key-switch
primitive:

```text
Alice ciphertext under s_A
    -> proxy: generic RNS key switch with rk_A->B
    -> ciphertext under s_B
    -> Bob decrypts the preserved plaintext
```

Trusted re-encryption-key setup uses both source and target secret keys. The
proxy execution path itself uses only the ciphertext, re-encryption key, and
public NTT plan; it receives neither secret key and performs no decryption.

The current prototype deliberately uses a zero-noise re-encryption key. A
Gaussian re-encryption key with the current generic CRT-block decomposition at
`research-4096` exhibited unacceptable error amplification. Accordingly this
application validates service semantics and plaintext preservation only. It is
**not** a security-bearing PRE construction and makes no CCA-security,
collusion-resistance, or unidirectional-security claim.

```bash
cargo run --release --bin proxy_reencryption
```

Retained evidence:
`results/r3.6/proxy-reencryption.txt`.

### R3.6 application-layer closeout

The representative R3.6 suite now covers linear inference, nonlinear inference,
encrypted tensor/convolution-style computation, two-party encrypted matrix
computation, and a key-switch-based security service. This is sufficient to
exercise the workload layer for the next resilience-integration phase.

Cross-application characterization remains useful follow-on analysis, but is
not a prerequisite for beginning resilience integration.

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
