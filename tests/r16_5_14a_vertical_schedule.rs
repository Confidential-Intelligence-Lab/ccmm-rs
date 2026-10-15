// R16.5.14a: vertical FFT2160 decomposition and row-aligned UHD placement.
// Cleartext structural oracle only. No modifications to production code.
use num_complex::Complex64;
use std::f64::consts::PI;

const HEIGHT: usize = 2160;
const TILE_ROWS: usize = 135;
const LOCAL_ROWS: usize = 16;
const TILE_WIDTH: usize = 256;
const CT_SLOTS: usize = 8192;
const TILE_SLOTS: usize = 4096;

fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * (exponent % n) as f64 / n as f64)
}

// Independent mixed-radix Cooley-Tukey reference for factors 2, 3 and 5.
fn recursive_fft(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    if n == 1 {
        return input.to_vec();
    }
    let radix = [2, 3, 5]
        .into_iter()
        .find(|&r| n % r == 0)
        .expect("unsupported FFT length");
    let m = n / radix;
    let branches: Vec<Vec<Complex64>> = (0..radix)
        .map(|q| recursive_fft(&(0..m).map(|j| input[j * radix + q]).collect::<Vec<_>>()))
        .collect();
    let mut result = vec![Complex64::new(0.0, 0.0); n];
    for (k, value) in result.iter_mut().enumerate() {
        for (q, branch) in branches.iter().enumerate() {
            *value += branch[k % m] * root(n, q * k);
        }
    }
    result
}

// UHD row-aligned ciphertext-slot address for pixel (row, column).
fn address(row: usize, column: usize) -> (usize, usize) {
    assert!(row < HEIGHT && column < 3840);
    let tile_row = row / LOCAL_ROWS;
    let tile_col = column / TILE_WIDTH;
    let local_row = row % LOCAL_ROWS;
    let local_col = column % TILE_WIDTH;
    (
        tile_row * 8 + tile_col / 2,
        (tile_col % 2) * TILE_SLOTS + local_row * TILE_WIDTH + local_col,
    )
}

// FFT2160 = cross-tile FFT135, W2160^(k_cross * local_row), local FFT16.
// Natural frequency output: k_cross + 135 * k_local.
fn decomposed_vertical(input: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(input.len(), HEIGHT);
    let mut cross = vec![vec![Complex64::new(0.0, 0.0); LOCAL_ROWS]; TILE_ROWS];
    for local_row in 0..LOCAL_ROWS {
        let tile_values: Vec<_> = (0..TILE_ROWS)
            .map(|tile_row| input[tile_row * LOCAL_ROWS + local_row])
            .collect();
        let transformed = recursive_fft(&tile_values);
        for k_cross in 0..TILE_ROWS {
            cross[k_cross][local_row] = transformed[k_cross];
        }
    }
    let mut result = vec![Complex64::new(0.0, 0.0); HEIGHT];
    for k_cross in 0..TILE_ROWS {
        let local_values: Vec<_> = (0..LOCAL_ROWS)
            .map(|r| cross[k_cross][r] * root(HEIGHT, k_cross * r))
            .collect();
        let transformed = recursive_fft(&local_values);
        for (k_local, &value) in transformed.iter().enumerate() {
            result[k_cross + TILE_ROWS * k_local] = value;
        }
    }
    result
}

#[test]
fn r16_5_14a_vertical_fft2160_and_uhd_placement() {
    let mut seen = vec![false; 1080 * CT_SLOTS];
    for row in 0..HEIGHT {
        for col in 0..3840 {
            let (ct, slot) = address(row, col);
            let flat = ct * CT_SLOTS + slot;
            assert!(
                !seen[flat],
                "duplicate physical slot at row={row} col={col}"
            );
            seen[flat] = true;
        }
    }
    assert_eq!(
        seen.iter().filter(|&&occupied| occupied).count(),
        HEIGHT * 3840
    );
    // A fixed column retains the same intra-tile column coordinate in all 135
    // tile rows. No vertical half-tile realignment is required for this packing.
    for &col in &[0, 255, 256, 511, 2047, 3839] {
        let slots: Vec<_> = (0..TILE_ROWS)
            .map(|t| address(t * LOCAL_ROWS, col).1)
            .collect();
        assert!(slots.iter().all(|&slot| slot == slots[0]));
    }

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for case in 0..3 {
        let input: Vec<_> = (0..HEIGHT)
            .map(|i| {
                let x = i as f64;
                Complex64::new(
                    0.003 * ((0.071 * x + case as f64).sin() + 0.31 * (0.013 * x).cos()),
                    0.002 * ((0.037 * x).cos() - 0.23 * (0.019 * x + case as f64).sin()),
                )
            })
            .collect();
        let output = decomposed_vertical(&input);
        // Independent direct DFT for all 2160 bins, not another FFT schedule.
        let mut error_sq = 0.0;
        let mut reference_sq = 0.0;
        for (k, actual) in output.iter().enumerate() {
            let reference: Complex64 = input
                .iter()
                .enumerate()
                .map(|(j, &x)| x * root(HEIGHT, k * j))
                .sum();
            let error = (*actual - reference).norm();
            max_abs = max_abs.max(error);
            error_sq += error * error;
            reference_sq += reference.norm_sqr();
        }
        max_rel = max_rel.max((error_sq / reference_sq).sqrt());
    }
    println!("R16_VERTICAL_LENGTH={HEIGHT}");
    println!("R16_VERTICAL_CROSS_TILE_RADIX={TILE_ROWS}");
    println!("R16_VERTICAL_LOCAL_RADIX={LOCAL_ROWS}");
    println!("R16_VERTICAL_PHYSICAL_CIPHERTEXTS=1080");
    println!("R16_VERTICAL_ACTIVE_SLOTS={}", HEIGHT * 3840);
    println!("R16_VERTICAL_CROSS_TILE_HALF_REALIGNMENT=0");
    println!("R16_VERTICAL_REL_L2={max_rel:.12e}");
    println!("R16_VERTICAL_MAX_ABS={max_abs:.12e}");
    assert!(
        max_rel < 1e-10,
        "vertical decomposition differs from independent DFT"
    );
    println!("R16_VERTICAL_STATUS=PASS");
}
