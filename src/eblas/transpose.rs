//! eBLAS mathematical transpose operations.
use crate::matrix::{BatchMatrix, RnsCkksCiphertextMatrix, RnsCkksPlaintextMatrix};

/// Transposes a cleartext matrix batch.
pub fn transpose_pp<T: Clone>(matrix: &BatchMatrix<T>) -> BatchMatrix<T> {
    matrix.transpose()
}
/// Transposes an encoded CKKS plaintext matrix.
pub fn transpose_plain(matrix: &RnsCkksPlaintextMatrix) -> RnsCkksPlaintextMatrix {
    matrix.transpose()
}
/// Transposes an RNS CKKS ciphertext matrix without homomorphic arithmetic.
pub fn transpose_cipher(matrix: &RnsCkksCiphertextMatrix) -> RnsCkksCiphertextMatrix {
    matrix.transpose()
}

#[cfg(test)]
mod tests {
    use super::transpose_pp;
    use crate::matrix::BatchMatrix;
    #[test]
    fn pp_transpose_rectangular() {
        let a = BatchMatrix::from_vec_column_major(2, 3, 1, vec![1, 4, 2, 5, 3, 6]);
        let t = transpose_pp(&a);
        assert_eq!((t.rows(), t.cols()), (3, 2));
        assert_eq!(t.raw(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(transpose_pp(&t), a);
    }
}
