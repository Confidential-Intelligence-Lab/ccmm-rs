//! Native Rust research implementation of encrypted matrix multiplication
//! over RLWE/CKKS-style ciphertexts.
//!
//! # Architecture
//!
//! `ccmm-rs` intentionally contains two complementary paths:
//!
//! - a small-parameter reference/correctness path for transparent CPMM/CCMM
//!   validation; and
//! - a Roadmap-2 RNS/NTT CKKS path for realistic ring dimensions, canonical
//!   CKKS SIMD semantics, leveled evaluation, Galois operations, and encrypted
//!   matrix multiplication.
//!
//! # Public modules
//!
//! - [`ccmm`] — reference CPMM/CCMM orchestration and matrix encryption helpers.
//! - [`ckks`] — CKKS parameters, canonical embedding, leveled ciphertext state,
//!   evaluation, rotations, conjugation, and RNS/NTT multiplication.
//! - [`eval`] — reference evaluation-key and relinearization infrastructure.
//! - [`grafting`] — RNS evaluation-key, key-switch, and grafting infrastructure.
//! - [`matrix`] — batch, polynomial, and ciphertext matrix representations.
//! - [`ring`] — polynomial arithmetic, moduli, RNS representation, and NTTs.
//! - [`rlwe`] — RLWE ciphertexts, keys, encryption, decryption, and error models.
//!
//! # Security status
//!
//! The realistic parameter profiles are currently research profiles and are
//! not designated security-bearing. In particular, the validated Roadmap-2
//! configuration uses discrete-Gaussian ciphertext error while retaining a
//! zero-noise evaluation-key path.
//!
//! See `docs/SECURITY_AND_PARAMETERS.md` for the security and parameter claim
//! boundary, and `docs/REPRODUCIBILITY.md` for reproducibility guidance.
//!
//! # Examples
//!
//! Realistic Roadmap-2 examples are available under `examples/`:
//!
//! - `ckks_simd_4096.rs`
//! - `rns_ccmm_4096.rs`
//!
//! The legacy/reference oracle and characterization binaries remain available
//! under `src/bin/`.

pub mod ccmm;
pub mod ckks;
pub mod eval;
pub mod grafting;
pub mod matrix;
pub mod ring;
pub mod rlwe;
