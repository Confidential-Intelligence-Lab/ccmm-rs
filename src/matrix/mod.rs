pub mod batch_matrix;
pub mod ciphertext;
pub mod encoded;

pub use batch_matrix::BatchMatrix;
pub use ciphertext::{MatrixCiphertext, MatrixCiphertextProduct, MatrixCiphertextQuadraticProduct};
pub use encoded::EncodedMatrix;

pub mod polynomial;
pub use polynomial::PolynomialMatrix;
