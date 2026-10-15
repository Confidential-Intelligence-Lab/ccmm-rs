//! R16.4b.2: full UHD tiled/global FFT2 equivalence validation.

use ccmm_rs::eblas::mixed_radix_fft::{mixed_radix_fft2, MixedRadixDirection};
use ccmm_rs::eblas::tiled_mixed_radix_fft::tiled_mixed_radix_fft1;
use num_complex::Complex64;
use std::f64::consts::PI;
use std::time::Instant;

const ROWS: usize = 2160;
const COLS: usize = 3840;

const TILE_ROWS: usize = 16;
const TILE_COLS: usize = 256;

fn tiled_fft2(input: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(input.len(), ROWS * COLS);

    let mut intermediate = vec![Complex64::new(0.0, 0.0); ROWS * COLS];

    // Horizontal transforms: 3840 = 256 * 15.
    for row in 0..ROWS {
        let start = row * COLS;

        let transformed = tiled_mixed_radix_fft1(
            TILE_COLS,
            COLS / TILE_COLS,
            MixedRadixDirection::Forward,
            &input[start..start + COLS],
        );

        intermediate[start..start + COLS].copy_from_slice(&transformed);
    }

    let mut output = vec![Complex64::new(0.0, 0.0); ROWS * COLS];

    // Vertical transforms: 2160 = 16 * 135.
    for col in 0..COLS {
        let values: Vec<Complex64> = (0..ROWS)
            .map(|row| intermediate[row * COLS + col])
            .collect();

        let transformed = tiled_mixed_radix_fft1(
            TILE_ROWS,
            ROWS / TILE_ROWS,
            MixedRadixDirection::Forward,
            &values,
        );

        for (row, value) in transformed.into_iter().enumerate() {
            output[row * COLS + col] = value;
        }
    }

    output
}

fn main() {
    println!("R16_TILED_UHD_BEGIN");
    println!("R16_IMAGE_ROWS={ROWS}");
    println!("R16_IMAGE_COLS={COLS}");
    println!("R16_TILE_ROWS={TILE_ROWS}");
    println!("R16_TILE_COLS={TILE_COLS}");
    println!("R16_IMAGE_RESIZED=false");

    let input: Vec<Complex64> = (0..ROWS * COLS)
        .map(|index| {
            let row = index / COLS;
            let col = index % COLS;

            // Deterministic, nontrivial signal containing several
            // frequency components and a small spatial modulation.
            let value = 0.4
                + 0.2 * (2.0 * PI * col as f64 / 64.0).cos()
                + 0.15 * (2.0 * PI * row as f64 / 45.0).sin()
                + 0.1 * (2.0 * PI * (row + col) as f64 / 120.0).cos()
                + 0.05 * ((row * 17 + col * 31) as f64).sin();

            Complex64::new(value, 0.0)
        })
        .collect();

    let reference_start = Instant::now();

    let reference = mixed_radix_fft2(ROWS, COLS, MixedRadixDirection::Forward, &input);

    let reference_ms = reference_start.elapsed().as_millis();

    let tiled_start = Instant::now();
    let tiled = tiled_fft2(&input);
    let tiled_ms = tiled_start.elapsed().as_millis();

    // R16.4b.5: execute the FFT directly on physical tile-major storage.
    {
        use ccmm_rs::eblas::physical_tiled_fft::{physical_tiled_fft2_forward, PhysicalTileLayout};

        let layout = PhysicalTileLayout::new(ROWS, COLS, TILE_ROWS, TILE_COLS);

        let physical_input = layout.pack(&input);

        let start = Instant::now();
        let physical_output = physical_tiled_fft2_forward(&layout, &physical_input);
        let physical_ms = start.elapsed().as_millis();

        let actual = layout.unpack(&physical_output);

        let mut error_squared = 0.0_f64;
        let mut reference_squared = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for (value, expected) in actual.iter().zip(&reference) {
            let error = (*value - *expected).norm();
            error_squared += error * error;
            reference_squared += expected.norm_sqr();
            max_abs = max_abs.max(error);
        }

        let rel_l2 = (error_squared / reference_squared).sqrt();
        let pass = rel_l2 < 1e-12;

        println!("R16_PHYSICAL_TILE_COUNT={}", layout.tile_count());
        println!("R16_PHYSICAL_UHD_MS={physical_ms}");
        println!("R16_PHYSICAL_UHD_REL_L2={rel_l2:.12e}");
        println!("R16_PHYSICAL_UHD_MAX_ABS={max_abs:.12e}");
        println!(
            "R16_PHYSICAL_UHD_STATUS={}",
            if pass { "PASS" } else { "FAIL" }
        );

        assert!(pass, "Physical UHD FFT mismatch");
    }

    let mut error_squared = 0.0_f64;
    let mut reference_squared = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (actual, expected) in tiled.iter().zip(&reference) {
        let error = (*actual - *expected).norm();

        error_squared += error * error;
        reference_squared += expected.norm_sqr();
        max_abs = max_abs.max(error);
    }

    let rel_l2 = (error_squared / reference_squared).sqrt();

    println!("R16_GLOBAL_REFERENCE_MS={reference_ms}");
    println!("R16_TILED_FORWARD_MS={tiled_ms}");
    println!("R16_TILED_GLOBAL_REL_L2={rel_l2:.12e}");
    println!("R16_TILED_GLOBAL_MAX_ABS={max_abs:.12e}");

    let pass = rel_l2 < 1e-12;

    println!(
        "R16_TILED_UHD_STATUS={}",
        if pass { "PASS" } else { "FAIL" }
    );

    assert!(pass, "Full UHD tiled/global FFT2 mismatch");
}
