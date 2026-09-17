# ccmm-rs

`ccmm-rs` is a native Rust implementation of ciphertext-ciphertext
matrix multiplication for RLWE/CKKS-style encrypted matrices.

The project focuses on the cryptographic and systems machinery required
for matrix multiplication:

- matrix and ciphertext representations,
- polynomial/ring arithmetic,
- NTT,
- RLWE encryption and decryption,
- degree-2 ciphertext multiplication,
- multiplication/evaluation keys,
- relinearization,
- scale and level management,
- rescaling,
- ciphertext-plaintext matrix multiplication (CPMM),
- ciphertext-ciphertext matrix multiplication (CCMM).

The implementation is developed independently in Rust and validated
against the Crypto 2026 Submission #720 / HEaaN reference implementation.
