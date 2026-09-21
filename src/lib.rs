//! FHE-rs: homomorphic encryption and encrypted linear algebra in Rust.
//!
//! FHE-rs is a native Rust research library for building applications that
//! compute directly on encrypted data. The current implementation provides
//! leveled CKKS for approximate numerical computation together with encrypted
//! vector, matrix, batched, and tensor operations through eBLAS.
//!
//! The project is currently developed in the `ccmm-rs` repository. The
//! repository and Cargo package names are retained for now to preserve the
//! history and reproducibility of the original encrypted matrix-multiplication
//! work.
//!
//! # What the library provides
//!
//! - CKKS encoding, encryption, decryption, addition, and multiplication;
//! - relinearization, rescaling, modulus switching, rotations, and conjugation;
//! - RNS/NTT arithmetic and configurable modulus chains;
//! - encrypted linear algebra through [`eblas`];
//! - plaintext/encrypted and encrypted/encrypted operand combinations;
//! - automatic selection between characterized encrypted GEMM backends;
//! - batched GEMM and tensor-to-GEMM mappings;
//! - shared CKKS support for application examples through
//!   [`application_support`].
//!
//! # Architecture
//!
//! Applications should use eBLAS or stable evaluator interfaces rather than
//! reconstructing low-level cryptographic schedules.
//!
//! ```text
//! Applications
//!     |
//!     v
//! eBLAS and evaluator APIs
//!     |
//!     v
//! CKKS evaluation
//!     |
//!     v
//! RNS / NTT / modular arithmetic
//! ```
//!
//! The original CPMM/CCMM work remains part of the implementation as one set
//! of encrypted matrix-multiplication mechanisms. The eBLAS abstraction and
//! its implementation are developed as part of FHE-rs; CPMM and CCMM are
//! attributed to Cheon, Kang, and Lee.
//!
//! # Public modules
//!
//! - [`application_support`] — shared helpers for application examples;
//! - [`ckks`] — CKKS parameters, encoding, ciphertext state, evaluation,
//!   rotations, conjugation, and RNS/NTT multiplication;
//! - [`eblas`] — encrypted Basic Linear Algebra Subprograms;
//! - [`eval`] — key-switch and relinearization infrastructure;
//! - [`grafting`] — RNS evaluation-key and modulus-transition infrastructure;
//! - [`matrix`] — plaintext and ciphertext matrix representations;
//! - [`ring`] — modular arithmetic, polynomial arithmetic, RNS, and NTTs;
//! - [`rlwe`] — RLWE ciphertexts, keys, encryption, and error distributions;
//! - [`ccmm`] — reference CPMM/CCMM orchestration retained for attribution,
//!   validation, and reproducibility.
//!
//! # Security and maturity
//!
//! The `research-4096` parameter set has been evaluated against a 128-bit
//! classical-security target for its underlying RLWE problem using the
//! documented Lattice Estimator methodology and MATZOV reduction-cost model.
//!
//! FHE-rs remains research software. The current implementation has not
//! undergone a third-party security audit, does not claim production
//! side-channel hardening, uses a research Gaussian sampler, and retains an
//! explicit circular/KDM assumption for the current secret-dependent
//! evaluation-key construction.
//!
//! See `docs/SECURITY_AND_PARAMETERS.md` and `docs/ASSURANCE.md` for the exact
//! claim boundaries.
//!
//! # Applications
//!
//! Current runnable applications include:
//!
//! - private linear inference with encrypted inputs and public model weights;
//! - encrypted two-party matrix multiplication;
//! - eBLAS validation and characterization workloads.
//!
//! See `docs/APPLICATIONS.md` and `docs/GETTING_STARTED.md` for application
//! development guidance.
//!
//! # Current scope
//!
//! FHE-rs currently provides leveled CKKS. Bootstrapping, integer/discrete
//! CKKS, additional HE schemes, accelerator backends, and production hardening
//! are future extensions.

pub mod application_support;
pub mod ccmm;
pub mod ckks;
pub mod eblas;
pub mod eval;
pub mod grafting;
pub mod matrix;
pub mod ring;
pub mod rlwe;
