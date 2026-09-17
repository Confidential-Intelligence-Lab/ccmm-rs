use super::BatchMatrix;

/// Matrix-layout encoding used by the CCMM layer.
///
/// `EncodedMatrix` preserves the logical `(batch, row, column)` dimensions
/// and stores values in the same batch-major, column-major order as
/// [`BatchMatrix`]:
///
/// `batch * rows * cols + row + col * rows`.
///
/// This type models matrix layout only. Ring encoding, CKKS scaling,
/// ciphertext packing, and encryption are separate layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedMatrix<T> {
    rows: usize,
    cols: usize,
    batches: usize,
    data: Vec<T>,
}

impl<T> EncodedMatrix<T> {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn batches(&self) -> usize {
        self.batches
    }

    pub fn raw(&self) -> &[T] {
        &self.data
    }

    pub fn get(&self, batch: usize, row: usize, col: usize) -> &T {
        &self.data[self.index(batch, row, col)]
    }

    fn index(&self, batch: usize, row: usize, col: usize) -> usize {
        assert!(batch < self.batches, "batch index out of bounds");
        assert!(row < self.rows, "row index out of bounds");
        assert!(col < self.cols, "column index out of bounds");

        batch * self.rows * self.cols + row + col * self.rows
    }
}

impl<T: Clone> EncodedMatrix<T> {
    /// Encodes a batch matrix without changing its coefficient values.
    pub fn encode(matrix: &BatchMatrix<T>) -> Self {
        Self {
            rows: matrix.rows(),
            cols: matrix.cols(),
            batches: matrix.batches(),
            data: matrix.raw().to_vec(),
        }
    }

    /// Decodes this representation back into a batch matrix.
    pub fn decode(&self) -> BatchMatrix<T> {
        BatchMatrix::from_vec_column_major(self.rows, self.cols, self.batches, self.data.clone())
    }
}

impl<T> EncodedMatrix<T>
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    /// Adds corresponding encoded matrix entries.
    ///
    /// # Panics
    ///
    /// Panics if matrix or batch dimensions differ.
    pub fn add(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.rows, rhs.rows,
            "row counts must match for matrix addition"
        );
        assert_eq!(
            self.cols, rhs.cols,
            "column counts must match for matrix addition"
        );
        assert_eq!(
            self.batches, rhs.batches,
            "batch counts must match for matrix addition"
        );

        let data = self
            .data
            .iter()
            .cloned()
            .zip(rhs.data.iter().cloned())
            .map(|(lhs, rhs)| lhs + rhs)
            .collect();

        Self {
            rows: self.rows,
            cols: self.cols,
            batches: self.batches,
            data,
        }
    }

    /// Multiplies corresponding matrices in each batch.
    ///
    /// If `self` has shape `(batch, m, n)` and `rhs` has shape
    /// `(batch, n, p)`, the result has shape `(batch, m, p)`.
    ///
    /// # Panics
    ///
    /// Panics if batch counts differ or matrix dimensions are incompatible.
    pub fn matmul(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.batches, rhs.batches,
            "batch counts must match for matrix multiplication"
        );
        assert_eq!(self.cols, rhs.rows, "inner matrix dimensions must match");

        let mut data = vec![T::default(); self.batches * self.rows * rhs.cols];

        for batch in 0..self.batches {
            for col in 0..rhs.cols {
                for row in 0..self.rows {
                    let mut sum = T::default();

                    for inner in 0..self.cols {
                        sum = sum
                            + self.get(batch, row, inner).clone()
                                * rhs.get(batch, inner, col).clone();
                    }

                    let index = batch * self.rows * rhs.cols + row + col * self.rows;

                    data[index] = sum;
                }
            }
        }

        Self {
            rows: self.rows,
            cols: rhs.cols,
            batches: self.batches,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(rows: usize, cols: usize, batches: usize, values: &[i64]) -> BatchMatrix<i64> {
        BatchMatrix::from_vec_column_major(rows, cols, batches, values.to_vec())
    }

    #[test]
    fn encode_preserves_dimensions_and_layout() {
        let matrix = filled(2, 3, 2, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        let encoded = EncodedMatrix::encode(&matrix);

        assert_eq!(encoded.rows(), 2);
        assert_eq!(encoded.cols(), 3);
        assert_eq!(encoded.batches(), 2);
        assert_eq!(encoded.raw(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn decode_inverts_encode() {
        let matrix = filled(2, 2, 2, &[1, -2, 3, 4, -5, 6, 7, -8]);

        let decoded = EncodedMatrix::encode(&matrix).decode();

        assert_eq!(decoded, matrix);
    }

    #[test]
    fn addition_is_elementwise() {
        let lhs = EncodedMatrix::encode(&filled(2, 2, 1, &[1, 2, 3, 4]));
        let rhs = EncodedMatrix::encode(&filled(2, 2, 1, &[10, 20, 30, 40]));

        let result = lhs.add(&rhs);

        assert_eq!(result.raw(), &[11, 22, 33, 44]);
    }

    #[test]
    fn batch_matmul_matches_reference_values() {
        // Column-major representations:
        //
        // lhs = [[1, 2, 3],
        //        [4, 5, 6]]
        //
        // rhs = [[7,  8],
        //        [9, 10],
        //        [11,12]]
        let lhs = EncodedMatrix::encode(&filled(2, 3, 1, &[1, 4, 2, 5, 3, 6]));

        let rhs = EncodedMatrix::encode(&filled(3, 2, 1, &[7, 9, 11, 8, 10, 12]));

        let product = lhs.matmul(&rhs);

        // [[58, 64],
        //  [139,154]]
        //
        // Column-major:
        assert_eq!(product.raw(), &[58, 139, 64, 154]);
    }

    #[test]
    #[should_panic(expected = "batch counts must match")]
    fn addition_rejects_mismatched_batches() {
        let lhs = EncodedMatrix::encode(&BatchMatrix::<i64>::new(2, 2, 1));
        let rhs = EncodedMatrix::encode(&BatchMatrix::<i64>::new(2, 2, 2));

        let _ = lhs.add(&rhs);
    }

    #[test]
    #[should_panic(expected = "inner matrix dimensions must match")]
    fn matmul_rejects_incompatible_dimensions() {
        let lhs = EncodedMatrix::encode(&BatchMatrix::<i64>::new(2, 3, 1));
        let rhs = EncodedMatrix::encode(&BatchMatrix::<i64>::new(2, 2, 1));

        let _ = lhs.matmul(&rhs);
    }

    #[test]
    fn generic_values_are_supported() {
        let matrix = BatchMatrix::from_vec_column_major(1, 2, 1, vec![42_u64, 99]);

        let encoded = EncodedMatrix::encode(&matrix);

        assert_eq!(encoded.raw(), &[42, 99]);
    }
}
