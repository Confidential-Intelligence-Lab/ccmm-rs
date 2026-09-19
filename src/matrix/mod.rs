//! Matrix representations used by the reference and Roadmap-2 paths.
//!
//! The module includes plaintext/batch matrices, polynomial matrices, legacy
//! ciphertext matrices, and [`RnsCkksCiphertextMatrix`] for realistic leveled
//! RNS CKKS matrix evaluation.
//!
pub mod batch_matrix;
pub mod ciphertext;
pub mod encoded;

pub use batch_matrix::BatchMatrix;
pub use ciphertext::{MatrixCiphertext, MatrixCiphertextProduct, MatrixCiphertextQuadraticProduct};
pub use encoded::EncodedMatrix;

pub mod polynomial;
pub use polynomial::PolynomialMatrix;

pub mod rns_ckks_ciphertext;
pub mod rns_ckks_plaintext;
pub use rns_ckks_ciphertext::RnsCkksCiphertextMatrix;
pub use rns_ckks_plaintext::RnsCkksPlaintextMatrix;
