//! Mixed-radix FFT for lengths composed of factors 2, 3, and 5.
//!
//! Cleartext reference implementation for the R16 UHD FFT campaign.
//! The forward transform is unnormalized; the inverse is normalized.

use num_complex::Complex64;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedRadixDirection {
    Forward,
    Inverse,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedRadixFft1Plan {
    length: usize,
    radices: Vec<usize>,
}

impl MixedRadixFft1Plan {
    pub fn new(length: usize) -> Self {
        assert!(length > 0, "FFT length must be positive");

        let mut remaining = length;
        let mut radices = Vec::new();

        for radix in [2, 3, 5] {
            while remaining % radix == 0 && remaining > 1 {
                radices.push(radix);
                remaining /= radix;
            }
        }

        assert_eq!(
            remaining, 1,
            "mixed-radix FFT supports only factors 2, 3, and 5"
        );

        Self { length, radices }
    }

    pub fn length(&self) -> usize {
        self.length
    }

    pub fn radices(&self) -> &[usize] {
        &self.radices
    }
}

fn transform_recursive(input: &[Complex64], radices: &[usize], sign: f64) -> Vec<Complex64> {
    let n = input.len();

    if n == 1 {
        return input.to_vec();
    }

    let radix = radices[0];
    let m = n / radix;

    let mut branches = Vec::with_capacity(radix);

    for j in 0..radix {
        let subsequence: Vec<Complex64> = (0..m).map(|k| input[j + radix * k]).collect();

        branches.push(transform_recursive(&subsequence, &radices[1..], sign));
    }

    let mut output = vec![Complex64::new(0.0, 0.0); n];

    for k in 0..m {
        for q in 0..radix {
            let output_index = k + m * q;
            let mut accumulator = Complex64::new(0.0, 0.0);

            for (j, branch) in branches.iter().enumerate() {
                let angle = sign * 2.0 * PI * (j * output_index) as f64 / n as f64;

                let twiddle = Complex64::new(angle.cos(), angle.sin());
                accumulator += branch[k] * twiddle;
            }

            output[output_index] = accumulator;
        }
    }

    output
}

pub fn mixed_radix_fft1(
    plan: &MixedRadixFft1Plan,
    direction: MixedRadixDirection,
    input: &[Complex64],
) -> Vec<Complex64> {
    assert_eq!(input.len(), plan.length());

    let sign = match direction {
        MixedRadixDirection::Forward => -1.0,
        MixedRadixDirection::Inverse => 1.0,
    };

    let mut output = transform_recursive(input, plan.radices(), sign);

    if direction == MixedRadixDirection::Inverse {
        let normalization = plan.length() as f64;
        for value in &mut output {
            *value /= normalization;
        }
    }

    output
}

pub fn mixed_radix_fft2(
    rows: usize,
    cols: usize,
    direction: MixedRadixDirection,
    input: &[Complex64],
) -> Vec<Complex64> {
    let row_plan = MixedRadixFft1Plan::new(cols);
    let col_plan = MixedRadixFft1Plan::new(rows);

    assert_eq!(
        input.len(),
        rows.checked_mul(cols).expect("FFT2 shape overflow")
    );

    let mut output = input.to_vec();

    for row in output.chunks_exact_mut(cols) {
        let transformed = mixed_radix_fft1(&row_plan, direction, row);
        row.copy_from_slice(&transformed);
    }

    for col in 0..cols {
        let column: Vec<Complex64> = (0..rows).map(|row| output[row * cols + col]).collect();

        let transformed = mixed_radix_fft1(&col_plan, direction, &column);

        for row in 0..rows {
            output[row * cols + col] = transformed[row];
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_dft(input: &[Complex64]) -> Vec<Complex64> {
        let n = input.len();

        (0..n)
            .map(|k| {
                (0..n)
                    .map(|j| {
                        let angle = -2.0 * PI * (j * k) as f64 / n as f64;
                        input[j] * Complex64::new(angle.cos(), angle.sin())
                    })
                    .sum()
            })
            .collect()
    }

    fn max_error(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x - *y).norm())
            .fold(0.0, f64::max)
    }

    #[test]
    fn mixed_radix_matches_direct_dft() {
        for n in [1, 2, 3, 4, 5, 6, 8, 12, 15, 20, 30, 48, 80] {
            let plan = MixedRadixFft1Plan::new(n);
            let input: Vec<_> = (0..n)
                .map(|i| Complex64::new((i as f64 * 0.17).sin(), 0.0))
                .collect();

            let actual = mixed_radix_fft1(&plan, MixedRadixDirection::Forward, &input);

            let expected = direct_dft(&input);

            assert!(
                max_error(&actual, &expected) < 1e-9,
                "FFT mismatch for n={n}"
            );
        }
    }

    #[test]
    fn rectangular_fft2_roundtrip() {
        for (rows, cols) in [(12, 20), (48, 80), (135, 240)] {
            let input: Vec<_> = (0..rows * cols)
                .map(|i| Complex64::new((i as f64 * 0.013).sin(), 0.0))
                .collect();

            let spectrum = mixed_radix_fft2(rows, cols, MixedRadixDirection::Forward, &input);

            let recovered = mixed_radix_fft2(rows, cols, MixedRadixDirection::Inverse, &spectrum);

            assert!(
                max_error(&input, &recovered) < 1e-8,
                "FFT2 roundtrip mismatch for {rows}x{cols}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "mixed-radix FFT supports only factors")]
    fn unsupported_prime_factor_is_rejected() {
        MixedRadixFft1Plan::new(7);
    }

    #[test]
    fn uhd_shape_is_supported() {
        let rows = MixedRadixFft1Plan::new(2160);
        let cols = MixedRadixFft1Plan::new(3840);

        assert_eq!(rows.radices(), &[2, 2, 2, 2, 3, 3, 3, 5]);
        assert_eq!(cols.radices(), &[2, 2, 2, 2, 2, 2, 2, 2, 3, 5]);
    }
}
