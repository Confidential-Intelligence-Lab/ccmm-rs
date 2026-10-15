// R16.5.12a: complete horizontal UHD FFT decomposition.
//
// Validates:
//   FFT15 across physical tiles
//   public twiddle multiplication
//   local FFT256
//   natural-frequency permutation
//
// Cleartext mathematical and placement validation only.
// No encrypted FFT256 or security claim.

use std::f64::consts::PI;
use std::time::Instant;

use num_complex::Complex64;

const CROSS: usize = 15;
const LOCAL: usize = 256;
const WIDTH: usize = CROSS * LOCAL;

fn dft(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();

    (0..n)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(j, value)| {
                    let angle = -2.0 * PI * (j * k) as f64 / n as f64;

                    *value * Complex64::from_polar(1.0, angle)
                })
                .sum()
        })
        .collect()
}

fn fft_radix2(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();

    assert!(n.is_power_of_two());

    let mut output = input.to_vec();

    // Bit-reversal permutation.
    let bits = n.trailing_zeros();

    for index in 0..n {
        let reversed = index.reverse_bits() >> (usize::BITS - bits);

        if reversed > index {
            output.swap(index, reversed);
        }
    }

    let mut length = 2;

    while length <= n {
        let half = length / 2;

        for start in (0..n).step_by(length) {
            for offset in 0..half {
                let angle = -2.0 * PI * offset as f64 / length as f64;

                let twiddle = Complex64::from_polar(1.0, angle);

                let even = output[start + offset];

                let odd = output[start + offset + half] * twiddle;

                output[start + offset] = even + odd;

                output[start + offset + half] = even - odd;
            }
        }

        length *= 2;
    }

    output
}

fn horizontal_decomposed(input: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(input.len(), WIDTH);

    // Physical input:
    // tile j contains input[j*256 .. (j+1)*256].
    //
    // For each local position s, perform FFT15 across tiles.
    let mut cross_output = vec![Complex64::new(0.0, 0.0); WIDTH];

    for s in 0..LOCAL {
        let cross_input: Vec<Complex64> = (0..CROSS).map(|j| input[j * LOCAL + s]).collect();

        let transformed = dft(&cross_input);

        for k_cross in 0..CROSS {
            cross_output[k_cross * LOCAL + s] = transformed[k_cross];
        }
    }

    // Public twiddle:
    //
    // W_3840^(k_cross * s)
    //
    // Then perform a local FFT256 for each cross frequency.
    let mut output = vec![Complex64::new(0.0, 0.0); WIDTH];

    for k_cross in 0..CROSS {
        let twiddled: Vec<Complex64> = (0..LOCAL)
            .map(|s| {
                let angle = -2.0 * PI * (k_cross * s) as f64 / WIDTH as f64;

                cross_output[k_cross * LOCAL + s] * Complex64::from_polar(1.0, angle)
            })
            .collect();

        let local_output = fft_radix2(&twiddled);

        // Natural-frequency permutation.
        for (k_local, &value) in local_output.iter().enumerate() {
            let frequency = k_cross + CROSS * k_local;

            output[frequency] = value;
        }
    }

    output
}

fn metrics(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    let mut error_squared = 0.0_f64;
    let mut reference_squared = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (actual, expected) in actual.iter().zip(expected) {
        let error = (*actual - *expected).norm();

        error_squared += error * error;
        reference_squared += expected.norm_sqr();

        max_abs = max_abs.max(error);
    }

    ((error_squared / reference_squared).sqrt(), max_abs)
}

#[test]
fn r16_5_12a_horizontal_fft3840_schedule() {
    let start = Instant::now();

    // Deterministic bounded complex input.
    let input: Vec<Complex64> = (0..WIDTH)
        .map(|index| {
            let x = index as f64;

            Complex64::new(
                0.02 * (0.013 * x).sin() + 0.01 * (0.071 * x).cos(),
                0.015 * (0.019 * x).sin(),
            )
        })
        .collect();

    let decomposed = horizontal_decomposed(&input);

    // Independent O(N^2) reference DFT3840.
    let reference_start = Instant::now();

    let reference = dft(&input);

    let reference_ms = reference_start.elapsed().as_secs_f64() * 1000.0;

    let (rel_l2, max_abs) = metrics(&decomposed, &reference);

    // Verify the frequency permutation is bijective.
    let mut visited = vec![false; WIDTH];

    for k_cross in 0..CROSS {
        for k_local in 0..LOCAL {
            let frequency = k_cross + CROSS * k_local;

            assert!(frequency < WIDTH);
            assert!(!visited[frequency]);

            visited[frequency] = true;
        }
    }

    assert!(visited.iter().all(|value| *value));

    println!("R16_HORIZONTAL_WIDTH={WIDTH}");
    println!("R16_HORIZONTAL_CROSS_RADIX={CROSS}");
    println!("R16_HORIZONTAL_LOCAL_RADIX={LOCAL}");
    println!("R16_HORIZONTAL_TRANSFORMED_VALUES={WIDTH}");
    println!("R16_HORIZONTAL_PERMUTATION_BIJECTIVE=true");
    println!("R16_HORIZONTAL_REL_L2={rel_l2:.12e}");
    println!("R16_HORIZONTAL_MAX_ABS={max_abs:.12e}");
    println!("R16_HORIZONTAL_REFERENCE_MS={reference_ms:.3}");
    println!(
        "R16_HORIZONTAL_TOTAL_MS={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );

    let pass = rel_l2 < 1.0e-10 && max_abs < 1.0e-9;

    println!(
        "R16_HORIZONTAL_STATUS={}",
        if pass { "PASS" } else { "FAIL" }
    );

    assert!(pass, "Horizontal FFT3840 schedule validation failed");
}
