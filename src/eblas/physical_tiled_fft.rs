//! R16.4b.3: explicit physical tile storage for mixed-radix FFT.

use num_complex::Complex64;

#[derive(Debug, Clone, Copy)]
pub struct PhysicalTileLayout {
    pub rows: usize,
    pub cols: usize,
    pub tile_rows: usize,
    pub tile_cols: usize,
    pub grid_rows: usize,
    pub grid_cols: usize,
}

impl PhysicalTileLayout {
    pub fn new(rows: usize, cols: usize, tile_rows: usize, tile_cols: usize) -> Self {
        assert!(rows > 0 && cols > 0);
        assert!(tile_rows > 0 && tile_cols > 0);
        assert_eq!(rows % tile_rows, 0);
        assert_eq!(cols % tile_cols, 0);

        Self {
            rows,
            cols,
            tile_rows,
            tile_cols,
            grid_rows: rows / tile_rows,
            grid_cols: cols / tile_cols,
        }
    }

    pub fn tile_count(&self) -> usize {
        self.grid_rows * self.grid_cols
    }

    pub fn tile_elements(&self) -> usize {
        self.tile_rows * self.tile_cols
    }

    pub fn physical_index(&self, row: usize, col: usize) -> usize {
        assert!(row < self.rows && col < self.cols);

        let tile_row = row / self.tile_rows;
        let tile_col = col / self.tile_cols;
        let local_row = row % self.tile_rows;
        let local_col = col % self.tile_cols;

        let tile_index = tile_row * self.grid_cols + tile_col;
        let local_index = local_row * self.tile_cols + local_col;

        tile_index * self.tile_elements() + local_index
    }

    pub fn pack(&self, input: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(input.len(), self.rows * self.cols);

        let mut physical = vec![Complex64::new(0.0, 0.0); input.len()];

        for row in 0..self.rows {
            for col in 0..self.cols {
                physical[self.physical_index(row, col)] = input[row * self.cols + col];
            }
        }

        physical
    }

    pub fn unpack(&self, physical: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(physical.len(), self.rows * self.cols);

        let mut logical = vec![Complex64::new(0.0, 0.0); physical.len()];

        for row in 0..self.rows {
            for col in 0..self.cols {
                logical[row * self.cols + col] = physical[self.physical_index(row, col)];
            }
        }

        logical
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_layout_roundtrip() {
        for (rows, cols, tr, tc) in [(12, 20, 4, 5), (48, 80, 8, 16), (2160, 3840, 16, 256)] {
            let layout = PhysicalTileLayout::new(rows, cols, tr, tc);

            let input: Vec<Complex64> = (0..rows * cols)
                .map(|i| Complex64::new(i as f64, 0.0))
                .collect();

            let physical = layout.pack(&input);
            let recovered = layout.unpack(&physical);

            assert_eq!(input, recovered);
            assert_eq!(layout.tile_count() * layout.tile_elements(), rows * cols);
        }
    }
}

/// Execute an exact 2D FFT directly on tile-major storage.
///
/// The input and output use PhysicalTileLayout indexing.
/// The transform is unnormalized, with the negative-exponent
/// forward-DFT convention.
///
/// Each 1D dimension N = L * G is decomposed as:
///
/// 1. G-point transforms across physical tiles, at fixed local index.
/// 2. Public twiddle multiplication.
/// 3. L-point transforms across local positions.
/// 4. Explicit frequency-index permutation.
///
/// This is a cleartext execution model for subsequent CKKS mapping.
pub fn physical_tiled_fft2_forward(
    layout: &PhysicalTileLayout,
    physical: &[Complex64],
) -> Vec<Complex64> {
    use super::mixed_radix_fft::{mixed_radix_fft1, MixedRadixDirection, MixedRadixFft1Plan};
    use std::f64::consts::PI;

    assert_eq!(physical.len(), layout.rows * layout.cols);

    let mut state = physical.to_vec();

    let row_plan = MixedRadixFft1Plan::new(layout.grid_rows);
    let col_plan = MixedRadixFft1Plan::new(layout.grid_cols);
    let local_row_plan = MixedRadixFft1Plan::new(layout.tile_rows);
    let local_col_plan = MixedRadixFft1Plan::new(layout.tile_cols);

    // Horizontal FFT: N = tile_cols * grid_cols.
    //
    // For each physical tile row and local row, execute the
    // cross-tile transform, apply twiddles, then the local FFT.
    let mut horizontal = vec![Complex64::new(0.0, 0.0); state.len()];

    for tile_row in 0..layout.grid_rows {
        for local_row in 0..layout.tile_rows {
            for local_col in 0..layout.tile_cols {
                let mut values = Vec::with_capacity(layout.grid_cols);

                for tile_col in 0..layout.grid_cols {
                    let row = tile_row * layout.tile_rows + local_row;
                    let col = tile_col * layout.tile_cols + local_col;
                    values.push(state[layout.physical_index(row, col)]);
                }

                let transformed =
                    mixed_radix_fft1(&col_plan, MixedRadixDirection::Forward, &values);

                for (frequency_tile, value) in transformed.into_iter().enumerate() {
                    let row = tile_row * layout.tile_rows + local_row;
                    let col = frequency_tile * layout.tile_cols + local_col;
                    horizontal[layout.physical_index(row, col)] = value;
                }
            }
        }
    }

    state = horizontal;

    // Horizontal twiddles and local FFT.
    let mut horizontal_output = vec![Complex64::new(0.0, 0.0); state.len()];

    for row in 0..layout.rows {
        for frequency_tile in 0..layout.grid_cols {
            let mut values = Vec::with_capacity(layout.tile_cols);

            for local_col in 0..layout.tile_cols {
                let col = frequency_tile * layout.tile_cols + local_col;
                let angle = -2.0 * PI * (frequency_tile * local_col) as f64 / layout.cols as f64;

                let twiddle = Complex64::new(angle.cos(), angle.sin());
                values.push(state[layout.physical_index(row, col)] * twiddle);
            }

            let transformed =
                mixed_radix_fft1(&local_col_plan, MixedRadixDirection::Forward, &values);

            for (local_frequency, value) in transformed.into_iter().enumerate() {
                // Natural-order frequency index:
                // k = frequency_tile + grid_cols * local_frequency.
                let output_col = frequency_tile + layout.grid_cols * local_frequency;

                horizontal_output[layout.physical_index(row, output_col)] = value;
            }
        }
    }

    state = horizontal_output;

    // Vertical cross-tile transform.
    let mut vertical = vec![Complex64::new(0.0, 0.0); state.len()];

    for col in 0..layout.cols {
        for local_row in 0..layout.tile_rows {
            let mut values = Vec::with_capacity(layout.grid_rows);

            for tile_row in 0..layout.grid_rows {
                let row = tile_row * layout.tile_rows + local_row;
                values.push(state[layout.physical_index(row, col)]);
            }

            let transformed = mixed_radix_fft1(&row_plan, MixedRadixDirection::Forward, &values);

            for (frequency_tile, value) in transformed.into_iter().enumerate() {
                let row = frequency_tile * layout.tile_rows + local_row;
                vertical[layout.physical_index(row, col)] = value;
            }
        }
    }

    state = vertical;

    // Vertical twiddles and local FFT.
    let mut output = vec![Complex64::new(0.0, 0.0); state.len()];

    for col in 0..layout.cols {
        for frequency_tile in 0..layout.grid_rows {
            let mut values = Vec::with_capacity(layout.tile_rows);

            for local_row in 0..layout.tile_rows {
                let row = frequency_tile * layout.tile_rows + local_row;
                let angle = -2.0 * PI * (frequency_tile * local_row) as f64 / layout.rows as f64;

                let twiddle = Complex64::new(angle.cos(), angle.sin());
                values.push(state[layout.physical_index(row, col)] * twiddle);
            }

            let transformed =
                mixed_radix_fft1(&local_row_plan, MixedRadixDirection::Forward, &values);

            for (local_frequency, value) in transformed.into_iter().enumerate() {
                let output_row = frequency_tile + layout.grid_rows * local_frequency;

                output[layout.physical_index(output_row, col)] = value;
            }
        }
    }

    output
}

#[cfg(test)]
mod physical_execution_tests {
    use super::*;
    use crate::eblas::mixed_radix_fft::{mixed_radix_fft2, MixedRadixDirection};

    #[test]
    fn physical_fft_matches_global_reference() {
        for (rows, cols, tr, tc) in [
            (12, 20, 4, 5),
            (48, 80, 8, 16),
            (48, 240, 16, 16),
            (135, 240, 5, 16),
        ] {
            let layout = PhysicalTileLayout::new(rows, cols, tr, tc);

            let input: Vec<Complex64> = (0..rows * cols)
                .map(|i| Complex64::new((i as f64 * 0.17).sin(), (i as f64 * 0.11).cos()))
                .collect();

            let physical = layout.pack(&input);

            let transformed = physical_tiled_fft2_forward(&layout, &physical);

            let actual = layout.unpack(&transformed);

            let expected = mixed_radix_fft2(rows, cols, MixedRadixDirection::Forward, &input);

            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (*a - *b).norm())
                .fold(0.0_f64, f64::max);

            assert!(
                max_error < 1e-8,
                "Physical FFT mismatch for {rows}x{cols}: {max_error:e}"
            );
        }
    }
}
