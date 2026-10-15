// R16.5.12d: independent cleartext radix-4 FFT256 schedule and CKKS slot mapping.
// No ciphertext operations, no changes to existing R16 evidence.
use std::f64::consts::PI;

use num_complex::Complex64;

const N: usize = 256;
const TILE: usize = 4096;
const CT_SLOTS: usize = 8192;

fn root(order: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * exponent as f64 / order as f64)
}

fn reverse_base4(mut index: usize) -> usize {
    let mut reversed = 0;
    for _ in 0..4 {
        reversed = reversed * 4 + index % 4;
        index /= 4;
    }
    reversed
}

// Iterative radix-4 DIT. The input is permuted by base-4 digit reversal;
// output is in natural frequency order. Each stage uses the usual W_m^(q*j)
// twiddles followed by a four-point DFT W_4^(p*q).
fn fft256_radix4(input: &[Complex64; N]) -> ([Complex64; N], [usize; 4]) {
    let mut data = [Complex64::new(0.0, 0.0); N];
    for (index, &value) in input.iter().enumerate() {
        data[reverse_base4(index)] = value;
    }

    let mut stage_span = 4;
    let mut stage_spans = [0; 4];
    for recorded_span in &mut stage_spans {
        *recorded_span = stage_span;
        let quarter = stage_span / 4;
        for base in (0..N).step_by(stage_span) {
            for j in 0..quarter {
                let mut branches = [Complex64::new(0.0, 0.0); 4];
                for (q, branch) in branches.iter_mut().enumerate() {
                    *branch = data[base + j + q * quarter] * root(stage_span, q * j);
                }
                for p in 0..4 {
                    let mut value = Complex64::new(0.0, 0.0);
                    for (q, &branch) in branches.iter().enumerate() {
                        value += branch * root(4, p * q);
                    }
                    data[base + j + p * quarter] = value;
                }
            }
        }
        stage_span *= 4;
    }
    (data, stage_spans)
}

fn direct_dft(input: &[Complex64; N]) -> [Complex64; N] {
    let mut output = [Complex64::new(0.0, 0.0); N];
    for (k, result) in output.iter_mut().enumerate() {
        for (j, &value) in input.iter().enumerate() {
            *result += value * root(N, (k * j) % N);
        }
    }
    output
}

// Physical UHD row-aligned packing: each 4096-slot tile contains 16 FFT256
// groups, and each 8192-slot ciphertext holds two adjacent tiles.
fn physical_address(tile_row: usize, tile_column: usize, local: usize) -> (usize, usize) {
    assert!(tile_row < 135 && tile_column < 15 && local < TILE);
    (
        tile_row * 8 + tile_column / 2,
        (tile_column % 2) * TILE + local,
    )
}

#[test]
fn r16_5_12d_radix4_fft256_schedule_and_physical_slots() {
    let mut worst_relative = 0.0_f64;
    let mut worst_absolute = 0.0_f64;
    let mut stage_spans = [0; 4];
    for case in 0..5 {
        let input: [Complex64; N] = std::array::from_fn(|i| {
            let t = i as f64;
            let c = case as f64;
            Complex64::new(
                0.004 * ((t * 0.37 + c * 0.19).sin() + 0.4 * (t * 0.07).cos()),
                0.003 * ((t * 0.23 - c * 0.11).cos() - 0.2 * (t * 0.13).sin()),
            )
        });
        let (actual, spans) = fft256_radix4(&input);
        stage_spans = spans;
        let expected = direct_dft(&input);
        let mut error_sq = 0.0;
        let mut reference_sq = 0.0;
        for (a, b) in actual.iter().zip(expected.iter()) {
            error_sq += (*a - *b).norm_sqr();
            reference_sq += b.norm_sqr();
            worst_absolute = worst_absolute.max((*a - *b).norm());
        }
        worst_relative = worst_relative.max((error_sq / reference_sq).sqrt());
    }

    let mut occupied = vec![false; 1080 * CT_SLOTS];
    let mut active_count = 0usize;
    for tile_row in 0..135 {
        for tile_column in 0..15 {
            for group in 0..16 {
                for index in 0..N {
                    let local = group * N + index;
                    let address = physical_address(tile_row, tile_column, local);
                    assert!(address.0 < 1080 && address.1 < CT_SLOTS);
                    let flat = address.0 * CT_SLOTS + address.1;
                    assert!(!occupied[flat], "duplicate physical slot");
                    occupied[flat] = true;
                    active_count += 1;
                }
            }
        }
    }
    assert_eq!(active_count, 135 * 15 * TILE);
    assert_eq!(stage_spans, [4, 16, 64, 256]);
    println!("R16_RADIX4_FFT256_STAGES={stage_spans:?}");
    println!("R16_RADIX4_FFT256_CASES=5");
    println!("R16_RADIX4_FFT256_GROUPS_PER_TILE=16");
    println!("R16_RADIX4_FFT256_ROW_ALIGNED_CIPHERTEXTS=1080");
    println!(
        "R16_RADIX4_FFT256_UNIQUE_ACTIVE_SLOTS={}",
        (2160usize * 3840usize)
    );
    println!("R16_RADIX4_FFT256_REL_L2={worst_relative:.12e}");
    println!("R16_RADIX4_FFT256_MAX_ABS={worst_absolute:.12e}");
    let pass = worst_relative < 1e-11 && worst_absolute < 1e-11;
    println!(
        "R16_RADIX4_FFT256_STATUS={}",
        if pass { "PASS" } else { "FAIL" }
    );
    assert!(pass, "radix-4 FFT256 did not match independent DFT256");
}
