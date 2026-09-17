use super::EncodedMatrix;

/// Structural ciphertext representation for encrypted matrices.
///
/// The ciphertext consists of two matrix components `(B, A)`.
/// Cryptographic semantics such as keys, levels, scaling, NTT state,
/// relinearization, and rescaling are introduced in later layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixCiphertext<T> {
    b: EncodedMatrix<T>,
    a: EncodedMatrix<T>,
}

/// Unreduced structural result of ciphertext-ciphertext matrix multiplication.
///
/// For ciphertexts `(B, A)` and `(D, C)`, the four bilinear blocks are:
///
/// `(BD, BC, AD, AC)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixCiphertextProduct<T> {
    bd: EncodedMatrix<T>,
    bc: EncodedMatrix<T>,
    ad: EncodedMatrix<T>,
    ac: EncodedMatrix<T>,
}

/// Degree-2 ciphertext product before relinearization.
///
/// For
///
/// `(B + A*s)(D + C*s)`
///
/// the coefficients are:
///
/// `c0 = BD`
/// `c1 = BC + AD`
/// `c2 = AC`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixCiphertextQuadraticProduct<T> {
    c0: EncodedMatrix<T>,
    c1: EncodedMatrix<T>,
    c2: EncodedMatrix<T>,
}

impl<T> MatrixCiphertext<T> {
    pub fn new(b: EncodedMatrix<T>, a: EncodedMatrix<T>) -> Self {
        assert_eq!(
            b.rows(),
            a.rows(),
            "ciphertext components must have matching row counts"
        );
        assert_eq!(
            b.cols(),
            a.cols(),
            "ciphertext components must have matching column counts"
        );
        assert_eq!(
            b.batches(),
            a.batches(),
            "ciphertext components must have matching batch counts"
        );

        Self { b, a }
    }

    pub fn b(&self) -> &EncodedMatrix<T> {
        &self.b
    }

    pub fn a(&self) -> &EncodedMatrix<T> {
        &self.a
    }

    pub fn rows(&self) -> usize {
        self.b.rows()
    }

    pub fn cols(&self) -> usize {
        self.b.cols()
    }

    pub fn batches(&self) -> usize {
        self.b.batches()
    }

    pub fn into_components(self) -> (EncodedMatrix<T>, EncodedMatrix<T>) {
        (self.b, self.a)
    }
}

impl<T> MatrixCiphertextProduct<T> {
    pub fn new(
        bd: EncodedMatrix<T>,
        bc: EncodedMatrix<T>,
        ad: EncodedMatrix<T>,
        ac: EncodedMatrix<T>,
    ) -> Self {
        for other in [&bc, &ad, &ac] {
            assert_eq!(
                bd.rows(),
                other.rows(),
                "product terms must have matching row counts"
            );
            assert_eq!(
                bd.cols(),
                other.cols(),
                "product terms must have matching column counts"
            );
            assert_eq!(
                bd.batches(),
                other.batches(),
                "product terms must have matching batch counts"
            );
        }

        Self { bd, bc, ad, ac }
    }

    pub fn bd(&self) -> &EncodedMatrix<T> {
        &self.bd
    }

    pub fn bc(&self) -> &EncodedMatrix<T> {
        &self.bc
    }

    pub fn ad(&self) -> &EncodedMatrix<T> {
        &self.ad
    }

    pub fn ac(&self) -> &EncodedMatrix<T> {
        &self.ac
    }

    pub fn rows(&self) -> usize {
        self.bd.rows()
    }

    pub fn cols(&self) -> usize {
        self.bd.cols()
    }

    pub fn batches(&self) -> usize {
        self.bd.batches()
    }

    pub fn into_terms(
        self,
    ) -> (
        EncodedMatrix<T>,
        EncodedMatrix<T>,
        EncodedMatrix<T>,
        EncodedMatrix<T>,
    ) {
        (self.bd, self.bc, self.ad, self.ac)
    }
}

impl<T> MatrixCiphertextQuadraticProduct<T> {
    pub fn new(c0: EncodedMatrix<T>, c1: EncodedMatrix<T>, c2: EncodedMatrix<T>) -> Self {
        for other in [&c1, &c2] {
            assert_eq!(
                c0.rows(),
                other.rows(),
                "quadratic terms must have matching row counts"
            );
            assert_eq!(
                c0.cols(),
                other.cols(),
                "quadratic terms must have matching column counts"
            );
            assert_eq!(
                c0.batches(),
                other.batches(),
                "quadratic terms must have matching batch counts"
            );
        }

        Self { c0, c1, c2 }
    }

    pub fn c0(&self) -> &EncodedMatrix<T> {
        &self.c0
    }

    pub fn c1(&self) -> &EncodedMatrix<T> {
        &self.c1
    }

    pub fn c2(&self) -> &EncodedMatrix<T> {
        &self.c2
    }

    pub fn rows(&self) -> usize {
        self.c0.rows()
    }

    pub fn cols(&self) -> usize {
        self.c0.cols()
    }

    pub fn batches(&self) -> usize {
        self.c0.batches()
    }

    pub fn into_terms(self) -> (EncodedMatrix<T>, EncodedMatrix<T>, EncodedMatrix<T>) {
        (self.c0, self.c1, self.c2)
    }
}

impl<T> MatrixCiphertext<T>
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    /// Ciphertext-plaintext matrix multiplication:
    ///
    /// `(B, A) * U = (BU, AU)`.
    pub fn cpmm(&self, plaintext: &EncodedMatrix<T>) -> Self {
        assert_eq!(
            self.b.batches(),
            plaintext.batches(),
            "ciphertext and plaintext batch counts must match"
        );
        assert_eq!(
            self.b.cols(),
            plaintext.rows(),
            "ciphertext and plaintext matrix dimensions must be compatible"
        );

        Self::new(self.b.matmul(plaintext), self.a.matmul(plaintext))
    }

    /// Structural ciphertext-ciphertext matrix multiplication:
    ///
    /// `(B, A) * (D, C) -> (BD, BC, AD, AC)`.
    pub fn ccmm(&self, rhs: &Self) -> MatrixCiphertextProduct<T> {
        assert_eq!(
            self.b.batches(),
            rhs.b.batches(),
            "ciphertext batch counts must match"
        );
        assert_eq!(
            self.b.cols(),
            rhs.b.rows(),
            "ciphertext matrix dimensions must be compatible"
        );

        MatrixCiphertextProduct::new(
            self.b.matmul(&rhs.b),
            self.b.matmul(&rhs.a),
            self.a.matmul(&rhs.b),
            self.a.matmul(&rhs.a),
        )
    }
}

impl<T> MatrixCiphertextProduct<T>
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    /// Converts `(BD, BC, AD, AC)` into the degree-2 ciphertext
    ///
    /// `(c0, c1, c2) = (BD, BC + AD, AC)`.
    pub fn combine_degree_two(self) -> MatrixCiphertextQuadraticProduct<T> {
        let (bd, bc, ad, ac) = self.into_terms();

        MatrixCiphertextQuadraticProduct::new(bd, bc.add(&ad), ac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::BatchMatrix;

    fn encoded(rows: usize, cols: usize, batches: usize, values: &[i64]) -> EncodedMatrix<i64> {
        EncodedMatrix::encode(&BatchMatrix::from_vec_column_major(
            rows,
            cols,
            batches,
            values.to_vec(),
        ))
    }

    #[test]
    fn constructor_preserves_components() {
        let b = encoded(2, 2, 1, &[1, 2, 3, 4]);
        let a = encoded(2, 2, 1, &[5, 6, 7, 8]);

        let ciphertext = MatrixCiphertext::new(b.clone(), a.clone());

        assert_eq!(ciphertext.b(), &b);
        assert_eq!(ciphertext.a(), &a);
        assert_eq!(ciphertext.rows(), 2);
        assert_eq!(ciphertext.cols(), 2);
        assert_eq!(ciphertext.batches(), 1);
    }

    #[test]
    fn cpmm_multiplies_both_components() {
        // B = [[1,2],
        //      [3,4]]
        //
        // A = [[5,6],
        //      [7,8]]
        //
        // U = [[2,1],
        //      [0,3]]
        let b = encoded(2, 2, 1, &[1, 3, 2, 4]);
        let a = encoded(2, 2, 1, &[5, 7, 6, 8]);
        let u = encoded(2, 2, 1, &[2, 0, 1, 3]);

        let result = MatrixCiphertext::new(b, a).cpmm(&u);

        assert_eq!(result.b().raw(), &[2, 6, 7, 15]);
        assert_eq!(result.a().raw(), &[10, 14, 23, 31]);
    }

    #[test]
    fn ccmm_expands_all_four_bilinear_terms() {
        let lhs = MatrixCiphertext::new(
            encoded(2, 2, 1, &[1, 0, 0, 1]),
            encoded(2, 2, 1, &[2, 0, 0, 2]),
        );

        let rhs = MatrixCiphertext::new(
            encoded(2, 2, 1, &[3, 0, 0, 3]),
            encoded(2, 2, 1, &[4, 0, 0, 4]),
        );

        let product = lhs.ccmm(&rhs);

        assert_eq!(product.bd().raw(), &[3, 0, 0, 3]);
        assert_eq!(product.bc().raw(), &[4, 0, 0, 4]);
        assert_eq!(product.ad().raw(), &[6, 0, 0, 6]);
        assert_eq!(product.ac().raw(), &[8, 0, 0, 8]);
    }

    #[test]
    fn degree_two_combination_matches_rlwe_product_algebra() {
        let lhs = MatrixCiphertext::new(
            encoded(2, 2, 1, &[1, 0, 0, 1]),
            encoded(2, 2, 1, &[2, 0, 0, 2]),
        );

        let rhs = MatrixCiphertext::new(
            encoded(2, 2, 1, &[3, 0, 0, 3]),
            encoded(2, 2, 1, &[4, 0, 0, 4]),
        );

        let quadratic = lhs.ccmm(&rhs).combine_degree_two();

        assert_eq!(quadratic.c0().raw(), &[3, 0, 0, 3]);
        assert_eq!(quadratic.c1().raw(), &[10, 0, 0, 10]);
        assert_eq!(quadratic.c2().raw(), &[8, 0, 0, 8]);
    }

    #[test]
    fn deterministic_matrix_semantics_match_reference_product() {
        // X = [[1,2],
        //      [3,4]]
        //
        // Y = [[5,6],
        //      [7,8]]
        //
        // X*Y = [[19,22],
        //        [43,50]]
        //
        // Split X = B + A and Y = D + C. Evaluating the
        // ciphertext product polynomial at the structural test point
        // s = 1 must recover the ordinary matrix product.

        let b = encoded(2, 2, 1, &[1, 1, 1, 1]);
        let a = encoded(2, 2, 1, &[0, 2, 1, 3]);

        let d = encoded(2, 2, 1, &[2, 3, 3, 4]);
        let c = encoded(2, 2, 1, &[3, 4, 3, 4]);

        assert_eq!(b.add(&a).raw(), &[1, 3, 2, 4]);
        assert_eq!(d.add(&c).raw(), &[5, 7, 6, 8]);

        let quadratic = MatrixCiphertext::new(b, a)
            .ccmm(&MatrixCiphertext::new(d, c))
            .combine_degree_two();

        let at_one = quadratic.c0().add(quadratic.c1()).add(quadratic.c2());

        assert_eq!(at_one.raw(), &[19, 43, 22, 50]);
    }

    #[test]
    #[should_panic(expected = "matching row counts")]
    fn constructor_rejects_mismatched_rows() {
        let b = encoded(2, 2, 1, &[0, 0, 0, 0]);
        let a = encoded(3, 2, 1, &[0, 0, 0, 0, 0, 0]);

        let _ = MatrixCiphertext::new(b, a);
    }

    #[test]
    #[should_panic(expected = "ciphertext batch counts must match")]
    fn ccmm_rejects_mismatched_batches() {
        let lhs = MatrixCiphertext::new(
            EncodedMatrix::encode(&BatchMatrix::<i64>::new(2, 2, 1)),
            EncodedMatrix::encode(&BatchMatrix::<i64>::new(2, 2, 1)),
        );

        let rhs = MatrixCiphertext::new(
            EncodedMatrix::encode(&BatchMatrix::<i64>::new(2, 2, 2)),
            EncodedMatrix::encode(&BatchMatrix::<i64>::new(2, 2, 2)),
        );

        let _ = lhs.ccmm(&rhs);
    }
}
