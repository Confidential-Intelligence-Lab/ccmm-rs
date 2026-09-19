# Applications

`ccmm-rs` includes application-level examples built on the realistic N=4096
RNS/CKKS path.

## Private Linear Inference

Executable:

```text
cargo run --release --bin private_linear_inference
```

The workload encrypts a feature vector and evaluates it against a plaintext
weight matrix using ciphertext-plaintext matrix multiplication (CPMM).

This path requires no evaluation key or relinearization.

Relevant files:

```text
src/bin/private_linear_inference.rs
results/r3.2b/private-linear-inference.txt
```

## Private Two-Party Matrix Product

Executable:

```text
cargo run --release --bin private_two_party_matrix_product
```

Both matrix operands are encrypted. The workload uses the bounded-base
Gaussian ciphertext-ciphertext matrix multiplication (CCMM) path.

Relevant files:

```text
src/bin/private_two_party_matrix_product.rs
results/r3.2c/private-two-party-matrix-product.txt
```

The current N=4096 reference configuration uses:

```text
ring degree              4096
logical CKKS slots       2048
ciphertext sigma         3.19
evaluation-key sigma     3.19
bounded base_log         20
```

For the precise security and implementation claim boundary, see
`docs/SECURITY_AND_PARAMETERS.md`.
