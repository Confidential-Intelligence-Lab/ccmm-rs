/// Dense batch of matrices stored batch-major and column-major within
/// each matrix.
///
/// The element at `(batch, row, col)` is stored at:
///
/// `batch * rows * cols + row + col * rows`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchMatrix<T> {
    rows: usize,
    cols: usize,
    batches: usize,
    data: Vec<T>,
}

impl<T: Clone + Default> BatchMatrix<T> {
    /// Constructs a zero/default-filled batch matrix.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is zero.
    pub fn new(rows: usize, cols: usize, batches: usize) -> Self {
        assert!(rows > 0, "matrix must contain at least one row");
        assert!(cols > 0, "matrix must contain at least one column");
        assert!(batches > 0, "matrix must contain at least one batch");

        Self {
            rows,
            cols,
            batches,
            data: vec![T::default(); rows * cols * batches],
        }
    }
}

impl<T> BatchMatrix<T> {
    /// Constructs a batch matrix from column-major storage.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is zero or if the data length does not match
    /// `rows * cols * batches`.
    pub fn from_vec_column_major(rows: usize, cols: usize, batches: usize, data: Vec<T>) -> Self {
        assert!(rows > 0, "matrix must contain at least one row");
        assert!(cols > 0, "matrix must contain at least one column");
        assert!(batches > 0, "matrix must contain at least one batch");
        assert_eq!(
            data.len(),
            rows * cols * batches,
            "batch matrix data length mismatch"
        );

        Self {
            rows,
            cols,
            batches,
            data,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn batches(&self) -> usize {
        self.batches
    }

    pub fn get(&self, batch: usize, row: usize, col: usize) -> &T {
        &self.data[self.index(batch, row, col)]
    }

    pub fn set(&mut self, batch: usize, row: usize, col: usize, value: T) {
        let index = self.index(batch, row, col);
        self.data[index] = value;
    }

    /// Returns the mathematical transpose while preserving column-major storage.
    pub fn transpose(&self) -> Self
    where
        T: Clone,
    {
        let mut data = Vec::with_capacity(self.data.len());
        for batch in 0..self.batches {
            for col in 0..self.rows {
                for row in 0..self.cols {
                    data.push(self.get(batch, col, row).clone());
                }
            }
        }
        Self::from_vec_column_major(self.cols, self.rows, self.batches, data)
    }

    pub fn raw(&self) -> &[T] {
        &self.data
    }

    pub fn raw_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn index(&self, batch: usize, row: usize, col: usize) -> usize {
        assert!(batch < self.batches, "batch index out of bounds");
        assert!(row < self.rows, "row index out of bounds");
        assert!(col < self.cols, "column index out of bounds");

        batch * self.rows * self.cols + row + col * self.rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_are_preserved() {
        let matrix = BatchMatrix::<i64>::new(2, 3, 4);

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.batches(), 4);
        assert_eq!(matrix.raw().len(), 24);
    }

    #[test]
    fn storage_is_batch_major_and_column_major() {
        let mut matrix = BatchMatrix::<i64>::new(2, 2, 2);

        matrix.set(0, 0, 0, 1);
        matrix.set(0, 1, 0, 2);
        matrix.set(0, 0, 1, 3);
        matrix.set(0, 1, 1, 4);

        matrix.set(1, 0, 0, 5);
        matrix.set(1, 1, 0, 6);
        matrix.set(1, 0, 1, 7);
        matrix.set(1, 1, 1, 8);

        assert_eq!(matrix.raw(), &[1, 2, 3, 4, 5, 6, 7, 8]);

        assert_eq!(*matrix.get(0, 0, 0), 1);
        assert_eq!(*matrix.get(0, 1, 1), 4);
        assert_eq!(*matrix.get(1, 0, 0), 5);
        assert_eq!(*matrix.get(1, 1, 1), 8);
    }

    #[test]
    fn from_vec_column_major_preserves_storage() {
        let matrix = BatchMatrix::from_vec_column_major(2, 2, 2, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        assert_eq!(matrix.raw(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(*matrix.get(1, 0, 1), 7);
    }

    #[test]
    fn generic_storage_supports_u64() {
        let mut matrix = BatchMatrix::<u64>::new(2, 2, 1);

        matrix.set(0, 1, 1, 42);

        assert_eq!(*matrix.get(0, 1, 1), 42);
    }

    #[test]
    fn raw_mut_updates_storage() {
        let mut matrix = BatchMatrix::<i64>::new(2, 2, 1);

        matrix.raw_mut()[2] = 17;

        assert_eq!(*matrix.get(0, 0, 1), 17);
    }

    #[test]
    #[should_panic(expected = "at least one row")]
    fn rejects_zero_rows() {
        let _ = BatchMatrix::<i64>::new(0, 2, 1);
    }

    #[test]
    #[should_panic(expected = "at least one column")]
    fn rejects_zero_columns() {
        let _ = BatchMatrix::<i64>::new(2, 0, 1);
    }

    #[test]
    #[should_panic(expected = "at least one batch")]
    fn rejects_zero_batches() {
        let _ = BatchMatrix::<i64>::new(2, 2, 0);
    }

    #[test]
    #[should_panic(expected = "data length mismatch")]
    fn rejects_wrong_data_length() {
        let _ = BatchMatrix::from_vec_column_major(2, 2, 1, vec![1, 2, 3]);
    }
}
