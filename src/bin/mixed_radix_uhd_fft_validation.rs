//! R16.2: exact-resolution UHD cleartext FFT validation.

use ccmm_rs::eblas::mixed_radix_fft::{mixed_radix_fft2, MixedRadixDirection};
use num_complex::Complex64;
use std::time::Instant;

const ROWS: usize = 2160;
const COLS: usize = 3840;

fn main() {
    println!("R16_UHD_VALIDATION_BEGIN");
    println!("R16_IMAGE_ROWS={ROWS}");
    println!("R16_IMAGE_COLS={COLS}");
    println!("R16_IMAGE_PIXELS={}", ROWS * COLS);
    println!("R16_IMAGE_RESIZED=false");
    println!("R16_TRANSFORM_MODE=GLOBAL_EXACT_MIXED_RADIX");

    let input: Vec<Complex64> = (0..ROWS * COLS)
        .map(|index| {
            let row = index / COLS;
            let col = index % COLS;

            let value = 0.5
                + 0.2 * (2.0 * std::f64::consts::PI * col as f64 / 64.0).cos()
                + 0.2 * (2.0 * std::f64::consts::PI * row as f64 / 45.0).cos();

            Complex64::new(value, 0.0)
        })
        .collect();

    let start = Instant::now();

    let spectrum = mixed_radix_fft2(ROWS, COLS, MixedRadixDirection::Forward, &input);

    let forward_ms = start.elapsed().as_millis();

    // R16.3: independent analytical spectrum oracle.
    //
    // The synthetic input consists of a DC component and two
    // integer-bin cosine waves. Its complete DFT has five
    // nonzero coefficients with analytically known amplitudes.
    {
        let expected_bins = [
            ((0usize, 0usize), 4_147_200.0),
            ((0, 60), 829_440.0),
            ((0, 3780), 829_440.0),
            ((48, 0), 829_440.0),
            ((2112, 0), 829_440.0),
        ];

        let mut expected = vec![Complex64::new(0.0, 0.0); ROWS * COLS];

        for &((row, col), amplitude) in &expected_bins {
            expected[row * COLS + col] = Complex64::new(amplitude, 0.0);
        }

        let mut error_squared = 0.0_f64;
        let mut reference_squared = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for (actual, reference) in spectrum.iter().zip(&expected) {
            let error = (*actual - *reference).norm();
            error_squared += error * error;
            reference_squared += reference.norm_sqr();
            max_abs = max_abs.max(error);
        }

        let rel_l2 = (error_squared / reference_squared).sqrt();
        let pass = rel_l2 < 1e-10;

        println!("R16_ANALYTICAL_SPECTRUM_REL_L2={rel_l2:.12e}");
        println!("R16_ANALYTICAL_SPECTRUM_MAX_ABS={max_abs:.12e}");
        println!(
            "R16_ANALYTICAL_SPECTRUM_STATUS={}",
            if pass { "PASS" } else { "FAIL" }
        );

        assert!(pass, "UHD analytical spectrum validation failed");
    }

    let inverse_start = Instant::now();

    let recovered = mixed_radix_fft2(ROWS, COLS, MixedRadixDirection::Inverse, &spectrum);

    let inverse_ms = inverse_start.elapsed().as_millis();

    let mut error_squared = 0.0;
    let mut reference_squared = 0.0;
    let mut max_abs = 0.0_f64;

    for (original, reconstructed) in input.iter().zip(&recovered) {
        let error = (*original - *reconstructed).norm();

        error_squared += error * error;
        reference_squared += original.norm_sqr();
        max_abs = max_abs.max(error);
    }

    let rel_l2 = (error_squared / reference_squared).sqrt();

    println!("R16_FORWARD_FFT_MS={forward_ms}");
    println!("R16_INVERSE_FFT_MS={inverse_ms}");
    println!("R16_ROUNDTRIP_REL_L2={rel_l2:.12e}");
    println!("R16_ROUNDTRIP_MAX_ABS={max_abs:.12e}");

    let pass = rel_l2 < 1e-9 && max_abs < 1e-8;

    println!(
        "R16_UHD_VALIDATION_STATUS={}",
        if pass { "PASS" } else { "FAIL" }
    );

    assert!(pass, "UHD FFT round-trip validation failed");
}
