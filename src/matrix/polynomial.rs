use crate::ring::{Modulus, Polynomial};

/// Dense matrix whose entries are polynomials in the same negacyclic ring.
///
/// Storage is column-major:
///
/// `index = row + column * rows`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialMatrix {
    rows: usize,
    cols: usize,
    modulus: Modulus,
    ring_degree: usize,
    data: Vec<Polynomial>,
}

impl PolynomialMatrix {
    pub fn new(rows: usize, cols: usize, modulus: Modulus, ring_degree: usize) -> Self {
        assert!(rows > 0, "matrix rows must be positive");
        assert!(cols > 0, "matrix columns must be positive");
        assert!(ring_degree > 0, "polynomial ring degree must be positive");

        Self {
            rows,
            cols,
            modulus,
            ring_degree,
            data: vec![Polynomial::zero(modulus, ring_degree); rows * cols],
        }
    }

    pub fn from_vec_column_major(rows: usize, cols: usize, data: Vec<Polynomial>) -> Self {
        assert!(rows > 0, "matrix rows must be positive");
        assert!(cols > 0, "matrix columns must be positive");
        assert_eq!(
            data.len(),
            rows * cols,
            "polynomial matrix data length mismatch"
        );

        let first = data
            .first()
            .expect("nonempty polynomial matrix must have data");

        let modulus = first.modulus();
        let ring_degree = first.degree();

        assert!(ring_degree > 0, "polynomial ring degree must be positive");

        for polynomial in &data {
            assert_eq!(
                polynomial.modulus(),
                modulus,
                "all matrix polynomials must use the same modulus"
            );
            assert_eq!(
                polynomial.degree(),
                ring_degree,
                "all matrix polynomials must use the same ring degree"
            );
        }

        Self {
            rows,
            cols,
            modulus,
            ring_degree,
            data,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn modulus(&self) -> Modulus {
        self.modulus
    }

    pub fn ring_degree(&self) -> usize {
        self.ring_degree
    }

    pub fn raw(&self) -> &[Polynomial] {
        &self.data
    }

    pub fn get(&self, row: usize, col: usize) -> &Polynomial {
        assert!(row < self.rows, "row index out of bounds");
        assert!(col < self.cols, "column index out of bounds");

        &self.data[self.index(row, col)]
    }

    pub fn set(&mut self, row: usize, col: usize, value: Polynomial) {
        assert!(row < self.rows, "row index out of bounds");
        assert!(col < self.cols, "column index out of bounds");
        assert_eq!(
            value.modulus(),
            self.modulus,
            "polynomial modulus must match matrix modulus"
        );
        assert_eq!(
            value.degree(),
            self.ring_degree,
            "polynomial degree must match matrix ring degree"
        );

        let index = self.index(row, col);
        self.data[index] = value;
    }

    pub fn add(&self, rhs: &Self) -> Self {
        self.assert_same_shape_and_ring(rhs);

        let data = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(lhs, rhs)| lhs.add(rhs))
            .collect();

        Self::from_vec_column_major(self.rows, self.cols, data)
    }

    /// Matrix multiplication over R_q = ``Z_q[X]/(X^N + 1)``.
    ///
    /// Matrix addition is polynomial addition and scalar
    /// multiplication is negacyclic polynomial multiplication.
    pub fn matmul(&self, rhs: &Self) -> Self {
        assert_eq!(self.cols, rhs.rows, "matrix inner dimensions must match");
        self.assert_same_ring(rhs);

        let mut result = Self::new(self.rows, rhs.cols, self.modulus, self.ring_degree);

        for col in 0..rhs.cols {
            for row in 0..self.rows {
                let mut accumulator = Polynomial::zero(self.modulus, self.ring_degree);

                for k in 0..self.cols {
                    let product = self.get(row, k).negacyclic_mul(rhs.get(k, col));

                    accumulator = accumulator.add(&product);
                }

                result.set(row, col, accumulator);
            }
        }

        result
    }

    /// Matrix multiplication over the same negacyclic ring using an
    /// explicit radix-2 NTT multiplication plan for every scalar
    /// polynomial product.
    pub fn matmul_ntt(&self, rhs: &Self, plan: &crate::ring::NttPlan) -> Self {
        assert_eq!(self.cols, rhs.rows, "matrix inner dimensions must match");

        self.assert_same_ring(rhs);

        assert_eq!(
            self.modulus,
            plan.modulus(),
            "matrix modulus must match NTT plan"
        );

        assert_eq!(
            self.ring_degree,
            plan.degree(),
            "matrix ring degree must match NTT plan"
        );

        let mut result = Self::new(self.rows, rhs.cols, self.modulus, self.ring_degree);

        for col in 0..rhs.cols {
            for row in 0..self.rows {
                let mut accumulator = Polynomial::zero(self.modulus, self.ring_degree);

                for k in 0..self.cols {
                    let product = plan.negacyclic_mul(self.get(row, k), rhs.get(k, col));

                    accumulator = accumulator.add(&product);
                }

                result.set(row, col, accumulator);
            }
        }

        result
    }

    fn index(&self, row: usize, col: usize) -> usize {
        row + col * self.rows
    }

    fn assert_same_ring(&self, rhs: &Self) {
        assert_eq!(
            self.modulus, rhs.modulus,
            "matrix polynomial moduli must match"
        );
        assert_eq!(
            self.ring_degree, rhs.ring_degree,
            "matrix polynomial ring degrees must match"
        );
    }

    fn assert_same_shape_and_ring(&self, rhs: &Self) {
        assert_eq!(self.rows, rhs.rows, "matrix row counts must match");
        assert_eq!(self.cols, rhs.cols, "matrix column counts must match");
        self.assert_same_ring(rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant(modulus: Modulus, degree: usize, value: u64) -> Polynomial {
        let mut coefficients = vec![0; degree];
        coefficients[0] = value;
        Polynomial::new(modulus, coefficients)
    }

    fn reference_matmul(lhs: &PolynomialMatrix, rhs: &PolynomialMatrix) -> PolynomialMatrix {
        assert_eq!(lhs.cols(), rhs.rows());

        let mut result =
            PolynomialMatrix::new(lhs.rows(), rhs.cols(), lhs.modulus(), lhs.ring_degree());

        for row in 0..lhs.rows() {
            for col in 0..rhs.cols() {
                let mut sum = Polynomial::zero(lhs.modulus(), lhs.ring_degree());

                for k in 0..lhs.cols() {
                    sum = sum.add(&lhs.get(row, k).negacyclic_mul(rhs.get(k, col)));
                }

                result.set(row, col, sum);
            }
        }

        result
    }

    #[test]
    fn column_major_layout_is_preserved() {
        let q = Modulus::new(97);
        let n = 4;

        let matrix = PolynomialMatrix::from_vec_column_major(
            2,
            2,
            vec![
                constant(q, n, 1),
                constant(q, n, 3),
                constant(q, n, 2),
                constant(q, n, 4),
            ],
        );

        assert_eq!(matrix.get(0, 0), &constant(q, n, 1));
        assert_eq!(matrix.get(1, 0), &constant(q, n, 3));
        assert_eq!(matrix.get(0, 1), &constant(q, n, 2));
        assert_eq!(matrix.get(1, 1), &constant(q, n, 4));
    }

    #[test]
    fn constant_polynomial_matrix_product_matches_integer_example() {
        let q = Modulus::new(97);
        let n = 4;

        // A = [[1, 2],
        //      [3, 4]]
        let lhs = PolynomialMatrix::from_vec_column_major(
            2,
            2,
            vec![
                constant(q, n, 1),
                constant(q, n, 3),
                constant(q, n, 2),
                constant(q, n, 4),
            ],
        );

        // B = [[5, 6],
        //      [7, 8]]
        let rhs = PolynomialMatrix::from_vec_column_major(
            2,
            2,
            vec![
                constant(q, n, 5),
                constant(q, n, 7),
                constant(q, n, 6),
                constant(q, n, 8),
            ],
        );

        let product = lhs.matmul(&rhs);

        assert_eq!(product.get(0, 0), &constant(q, n, 19));
        assert_eq!(product.get(1, 0), &constant(q, n, 43));
        assert_eq!(product.get(0, 1), &constant(q, n, 22));
        assert_eq!(product.get(1, 1), &constant(q, n, 50));
    }

    #[test]
    fn polynomial_matrix_product_uses_negacyclic_ring_multiplication() {
        let q = Modulus::new(97);

        let lhs = PolynomialMatrix::from_vec_column_major(
            1,
            1,
            vec![Polynomial::new(q, vec![1, 2, 3, 4])],
        );

        let rhs = PolynomialMatrix::from_vec_column_major(
            1,
            1,
            vec![Polynomial::new(q, vec![5, 6, 7, 8])],
        );

        let product = lhs.matmul(&rhs);

        assert_eq!(
            product.get(0, 0),
            &lhs.get(0, 0).negacyclic_mul(rhs.get(0, 0))
        );
    }

    #[test]
    fn rectangular_polynomial_matrix_product_matches_reference() {
        let q = Modulus::new(12_289);

        let lhs = PolynomialMatrix::from_vec_column_major(
            2,
            3,
            vec![
                Polynomial::new(q, vec![1, 2, 0, 0, 0, 0, 0, 0]),
                Polynomial::new(q, vec![3, 1, 0, 0, 0, 0, 0, 0]),
                Polynomial::new(q, vec![2, 4, 0, 0, 0, 0, 0, 0]),
                Polynomial::new(q, vec![1, 5, 0, 0, 0, 0, 0, 0]),
                Polynomial::new(q, vec![6, 2, 0, 0, 0, 0, 0, 0]),
                Polynomial::new(q, vec![4, 3, 0, 0, 0, 0, 0, 0]),
            ],
        );

        let rhs = PolynomialMatrix::from_vec_column_major(
            3,
            2,
            vec![
                Polynomial::new(q, vec![2, 1, 0, 0, 0, 0, 0, 0]),
                Polynomial::new(q, vec![3, 2, 0, 0, 0, 0, 0, 0]),
                Polynomial::new(q, vec![1, 4, 0, 0, 0, 0, 0, 0]),
                Polynomial::new(q, vec![5, 1, 0, 0, 0, 0, 0, 0]),
                Polynomial::new(q, vec![2, 6, 0, 0, 0, 0, 0, 0]),
                Polynomial::new(q, vec![4, 2, 0, 0, 0, 0, 0, 0]),
            ],
        );

        assert_eq!(lhs.matmul(&rhs), reference_matmul(&lhs, &rhs));
    }

    #[test]
    fn addition_is_entrywise_ring_addition() {
        let q = Modulus::new(97);

        let lhs = PolynomialMatrix::from_vec_column_major(
            1,
            2,
            vec![
                Polynomial::new(q, vec![1, 2, 3, 4]),
                Polynomial::new(q, vec![5, 6, 7, 8]),
            ],
        );

        let rhs = PolynomialMatrix::from_vec_column_major(
            1,
            2,
            vec![
                Polynomial::new(q, vec![8, 7, 6, 5]),
                Polynomial::new(q, vec![4, 3, 2, 1]),
            ],
        );

        let sum = lhs.add(&rhs);

        assert_eq!(sum.get(0, 0), &lhs.get(0, 0).add(rhs.get(0, 0)));
        assert_eq!(sum.get(0, 1), &lhs.get(0, 1).add(rhs.get(0, 1)));
    }
    #[test]
    fn ntt_matrix_product_matches_reference_matrix_product() {
        let q = Modulus::new(12_289);
        let degree = 64;

        let plan = crate::ring::make_ntt_plan(q, degree);

        fn poly(q: Modulus, degree: usize, offset: u64) -> Polynomial {
            Polynomial::new(
                q,
                (0..degree)
                    .map(|i| (offset + 7 * i as u64 + (i as u64).pow(2)) % q.value())
                    .collect(),
            )
        }

        let lhs = PolynomialMatrix::from_vec_column_major(
            2,
            2,
            vec![
                poly(q, degree, 1),
                poly(q, degree, 2),
                poly(q, degree, 3),
                poly(q, degree, 4),
            ],
        );

        let rhs = PolynomialMatrix::from_vec_column_major(
            2,
            2,
            vec![
                poly(q, degree, 5),
                poly(q, degree, 6),
                poly(q, degree, 7),
                poly(q, degree, 8),
            ],
        );

        assert_eq!(lhs.matmul_ntt(&rhs, &plan), lhs.matmul(&rhs));
    }
}
