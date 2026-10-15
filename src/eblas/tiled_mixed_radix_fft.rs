//! R16.4b: cleartext tiled mixed-radix FFT.
//!
//! Decomposes N = tile_length * tile_count using a two-stage
//! Cooley-Tukey transform with explicit inter-stage twiddles.

use super::mixed_radix_fft::{mixed_radix_fft1, MixedRadixDirection, MixedRadixFft1Plan};
use num_complex::Complex64;
use std::f64::consts::PI;

pub fn tiled_mixed_radix_fft1(
    tile_length: usize,
    tile_count: usize,
    direction: MixedRadixDirection,
    input: &[Complex64],
) -> Vec<Complex64> {
    assert!(tile_length > 0 && tile_count > 0);
    let n = tile_length
        .checked_mul(tile_count)
        .expect("FFT dimension overflow");
    assert_eq!(input.len(), n);

    let local_plan = MixedRadixFft1Plan::new(tile_length);
    let global_plan = MixedRadixFft1Plan::new(tile_count);

    let sign = match direction {
        MixedRadixDirection::Forward => -1.0,
        MixedRadixDirection::Inverse => 1.0,
    };

    // Decimation-in-time decomposition:
    //
    // n = tile_count * local_index + tile_index
    //
    // First transform the tile_count dimension, then apply
    // twiddles, then transform the tile_length dimension.
    let mut intermediate = vec![Complex64::new(0.0, 0.0); n];

    for tile_index in 0..tile_count {
        let mut values = Vec::with_capacity(tile_length);

        for local_index in 0..tile_length {
            values.push(input[local_index * tile_count + tile_index]);
        }

        let transformed = mixed_radix_fft1(&local_plan, direction, &values);

        for (k, value) in transformed.into_iter().enumerate() {
            intermediate[k * tile_count + tile_index] = value;
        }
    }

    let mut output = vec![Complex64::new(0.0, 0.0); n];

    for k in 0..tile_length {
        let mut values = Vec::with_capacity(tile_count);

        for tile_index in 0..tile_count {
            let angle = sign * 2.0 * PI * (k * tile_index) as f64 / n as f64;
            let twiddle = Complex64::new(angle.cos(), angle.sin());

            values.push(intermediate[k * tile_count + tile_index] * twiddle);
        }

        let transformed = mixed_radix_fft1(&global_plan, direction, &values);

        for (q, value) in transformed.into_iter().enumerate() {
            output[k + tile_length * q] = value;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_matches_global_fft() {
        let cases = [(2, 3), (4, 5), (8, 9), (16, 15), (16, 135), (256, 15)];

        for (tile_length, tile_count) in cases {
            let n = tile_length * tile_count;

            let input: Vec<Complex64> = (0..n)
                .map(|i| Complex64::new((i as f64 * 0.17).sin(), (i as f64 * 0.11).cos()))
                .collect();

            let global_plan = MixedRadixFft1Plan::new(n);

            let reference = mixed_radix_fft1(&global_plan, MixedRadixDirection::Forward, &input);

            let tiled = tiled_mixed_radix_fft1(
                tile_length,
                tile_count,
                MixedRadixDirection::Forward,
                &input,
            );

            let max_error = reference
                .iter()
                .zip(&tiled)
                .map(|(a, b)| (*a - *b).norm())
                .fold(0.0_f64, f64::max);

            assert!(
                max_error < 1e-8,
                "Tiled FFT mismatch for {tile_length}x{tile_count}: {max_error:e}"
            );
        }
    }
}
