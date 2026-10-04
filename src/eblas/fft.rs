//! Application-facing one-dimensional FFT for eBLAS.
//!
//! This module separates transform semantics from cryptographic execution.
//! The production clear path is an iterative radix-2 Cooley-Tukey FFT.
//! A dense O(N^2) DFT is retained only as a numerical reference oracle.
//!
//! This transform is distinct from the NTT used internally for polynomial
//! arithmetic by the cryptographic substrate.

use num_complex::Complex64;
use std::f64::consts::PI;

/// Direction of a complex Fourier transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftDirection {
    /// Forward transform:
    ///
    /// `X[k] = sum_n x[n] exp(-2*pi*i*k*n/N)`.
    Forward,
    /// Inverse transform, normalized by `1/N`.
    Inverse,
}

/// Shape contract for a one-dimensional radix-2 FFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fft1Shape {
    length: usize,
}

impl Fft1Shape {
    /// Creates a radix-2 FFT shape.
    ///
    /// The current implementation requires a non-zero power-of-two length.
    pub fn new(length: usize) -> Self {
        assert!(length > 0, "eBLAS FFT length must be positive");
        assert!(
            length.is_power_of_two(),
            "eBLAS radix-2 FFT length must be a power of two"
        );

        Self { length }
    }

    /// Number of complex values transformed.
    pub const fn length(self) -> usize {
        self.length
    }

    /// Number of radix-2 butterfly stages.
    pub fn stages(self) -> u32 {
        self.length.trailing_zeros()
    }

    /// Total number of radix-2 butterflies.
    pub fn butterflies(self) -> usize {
        self.length
            .checked_mul(self.stages() as usize)
            .expect("eBLAS FFT butterfly count overflow")
            / 2
    }
}

/// Dense complex DFT reference oracle.
///
/// This intentionally uses O(N^2) work and exists for semantic validation.
/// Production FFT execution must not call this routine.
pub fn dft1_reference(
    shape: Fft1Shape,
    direction: FftDirection,
    input: &[Complex64],
) -> Vec<Complex64> {
    assert_eq!(
        input.len(),
        shape.length(),
        "eBLAS DFT input length must match transform shape"
    );

    let n = shape.length();
    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };

    let mut output = vec![Complex64::new(0.0, 0.0); n];

    for (k, output_value) in output.iter_mut().enumerate() {
        let mut sum = Complex64::new(0.0, 0.0);

        for (sample, &value) in input.iter().enumerate() {
            let angle = sign * 2.0 * PI * (k as f64) * (sample as f64) / (n as f64);

            let twiddle = Complex64::new(angle.cos(), angle.sin());
            sum += value * twiddle;
        }

        *output_value = sum;
    }

    if direction == FftDirection::Inverse {
        let scale = n as f64;
        for value in &mut output {
            *value /= scale;
        }
    }

    output
}

/// Computes an application-facing one-dimensional radix-2 FFT.
///
/// The algorithm is iterative decimation-in-time Cooley-Tukey:
///
/// 1. bit-reverse the input ordering;
/// 2. execute `log2(N)` butterfly stages;
/// 3. normalize by `1/N` for inverse transforms.
///
/// Twiddle factors are public constants.
pub fn fft1_pp(shape: Fft1Shape, direction: FftDirection, input: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(
        input.len(),
        shape.length(),
        "eBLAS FFT input length must match transform shape"
    );

    let n = shape.length();
    let mut values = input.to_vec();

    bit_reverse_permute(&mut values);

    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };

    let mut span = 2usize;

    while span <= n {
        let half = span / 2;
        let angle = sign * 2.0 * PI / span as f64;
        let stage_root = Complex64::new(angle.cos(), angle.sin());

        for base in (0..n).step_by(span) {
            let mut twiddle = Complex64::new(1.0, 0.0);

            for offset in 0..half {
                let even_index = base + offset;
                let odd_index = even_index + half;

                let even = values[even_index];
                let odd = twiddle * values[odd_index];

                values[even_index] = even + odd;
                values[odd_index] = even - odd;

                twiddle *= stage_root;
            }
        }

        span = span.checked_mul(2).expect("eBLAS FFT stage span overflow");
    }

    if direction == FftDirection::Inverse {
        let scale = n as f64;
        for value in &mut values {
            *value /= scale;
        }
    }

    values
}

fn bit_reverse_permute(values: &mut [Complex64]) {
    let n = values.len();

    if n <= 2 {
        return;
    }

    let bits = n.trailing_zeros();

    for index in 0..n {
        let reversed = index.reverse_bits() >> (usize::BITS - bits);

        if reversed > index {
            values.swap(index, reversed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_complex(length: usize, salt: usize) -> Vec<Complex64> {
        (0..length)
            .map(|index| {
                let real_raw = (index * 17 + salt * 13) % 37;
                let imag_raw = (index * 11 + salt * 19) % 31;

                Complex64::new(
                    (real_raw as f64 - 18.0) / 19.0,
                    (imag_raw as f64 - 15.0) / 17.0,
                )
            })
            .collect()
    }

    fn max_abs_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
        assert_eq!(actual.len(), expected.len());

        actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn shape_exposes_radix2_structure() {
        let shape = Fft1Shape::new(8);

        assert_eq!(shape.length(), 8);
        assert_eq!(shape.stages(), 3);
        assert_eq!(shape.butterflies(), 12);
    }

    #[test]
    #[should_panic(expected = "FFT length must be positive")]
    fn shape_rejects_zero_length() {
        let _ = Fft1Shape::new(0);
    }

    #[test]
    #[should_panic(expected = "FFT length must be a power of two")]
    fn shape_rejects_non_power_of_two() {
        let _ = Fft1Shape::new(12);
    }

    #[test]
    fn impulse_has_flat_forward_spectrum() {
        let shape = Fft1Shape::new(8);

        let mut input = vec![Complex64::new(0.0, 0.0); 8];
        input[0] = Complex64::new(1.0, 0.0);

        let output = fft1_pp(shape, FftDirection::Forward, &input);

        for value in output {
            assert!((value - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
        }
    }

    #[test]
    fn constant_signal_maps_to_dc() {
        let shape = Fft1Shape::new(8);
        let input = vec![Complex64::new(2.0, 0.0); 8];

        let output = fft1_pp(shape, FftDirection::Forward, &input);

        assert!((output[0] - Complex64::new(16.0, 0.0)).norm() < 1.0e-12);

        for value in &output[1..] {
            assert!(value.norm() < 1.0e-12);
        }
    }

    #[test]
    fn radix2_fft_matches_dense_dft_reference() {
        for &(length, salt) in &[
            (1usize, 1usize),
            (2, 2),
            (4, 3),
            (8, 4),
            (16, 5),
            (32, 6),
            (64, 7),
            (128, 8),
            (256, 9),
        ] {
            let shape = Fft1Shape::new(length);
            let input = deterministic_complex(length, salt);

            let expected = dft1_reference(shape, FftDirection::Forward, &input);

            let actual = fft1_pp(shape, FftDirection::Forward, &input);

            let error = max_abs_error(&actual, &expected);

            println!(
                "FFT1_PP_DFT_CASE=N{} STAGES{} BUTTERFLIES{} MAX_ABS={:.12e}",
                length,
                shape.stages(),
                shape.butterflies(),
                error,
            );

            assert!(
                error < 1.0e-10,
                "radix-2 FFT maximum absolute error {error:e}"
            );
        }
    }

    #[test]
    fn inverse_fft_matches_dense_inverse_dft_reference() {
        for &(length, salt) in &[
            (1usize, 11usize),
            (2, 12),
            (4, 13),
            (8, 14),
            (16, 15),
            (32, 16),
            (64, 17),
        ] {
            let shape = Fft1Shape::new(length);
            let input = deterministic_complex(length, salt);

            let expected = dft1_reference(shape, FftDirection::Inverse, &input);

            let actual = fft1_pp(shape, FftDirection::Inverse, &input);

            let error = max_abs_error(&actual, &expected);

            println!("IFFT1_PP_DFT_CASE=N{} MAX_ABS={:.12e}", length, error,);

            assert!(
                error < 1.0e-10,
                "radix-2 inverse FFT maximum absolute error {error:e}"
            );
        }
    }

    #[test]
    fn inverse_roundtrip_recovers_input() {
        for &(length, salt) in &[
            (1usize, 21usize),
            (2, 22),
            (4, 23),
            (8, 24),
            (16, 25),
            (32, 26),
            (64, 27),
            (128, 28),
            (256, 29),
        ] {
            let shape = Fft1Shape::new(length);
            let input = deterministic_complex(length, salt);

            let spectrum = fft1_pp(shape, FftDirection::Forward, &input);

            let recovered = fft1_pp(shape, FftDirection::Inverse, &spectrum);

            let error = max_abs_error(&recovered, &input);

            println!("FFT1_PP_ROUNDTRIP_CASE=N{} MAX_ABS={:.12e}", length, error,);

            assert!(
                error < 1.0e-10,
                "FFT roundtrip maximum absolute error {error:e}"
            );
        }
    }
}
