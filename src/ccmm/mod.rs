pub mod cpmm;
pub mod multiply;

pub use cpmm::{cpmm, decrypt_matrix_raw, encrypt_matrix_with_rng, RlweMatrixCiphertext};

pub use multiply::{
    ccmm, ccmm_quadratic, decrypt_ckks_matrix, encrypt_ckks_matrix_with_rng, CkksMatrixCiphertext,
    CkksMatrixQuadraticProduct,
};
