//! R16.5.5: validate an explicit mixed-radix DIT execution schedule.

use ccmm_rs::eblas::mixed_radix_fft::{mixed_radix_fft1, MixedRadixDirection, MixedRadixFft1Plan};
use num_complex::Complex64;
use std::f64::consts::PI;

fn execute_staged_dit(input: &[Complex64], radices: &[usize]) -> Vec<Complex64> {
    let n = input.len();

    if n == 1 {
        assert!(radices.is_empty());
        return input.to_vec();
    }

    let radix = radices[0];
    assert_eq!(n % radix, 0);
    let m = n / radix;

    // Recursively execute the interleaved branches first.
    let branches: Vec<Vec<Complex64>> = (0..radix)
        .map(|j| {
            let branch: Vec<_> = (0..m).map(|k| input[j + radix * k]).collect();

            execute_staged_dit(&branch, &radices[1..])
        })
        .collect();

    let mut output = vec![Complex64::new(0.0, 0.0); n];

    for k in 0..m {
        // Public twiddles, applied before the radix-r DFT.
        let aligned: Vec<_> = (0..radix)
            .map(|j| {
                let angle = -2.0 * PI * (j * k) as f64 / n as f64;
                branches[j][k] * Complex64::new(angle.cos(), angle.sin())
            })
            .collect();

        // Explicit radix-r DFT combine.
        for q in 0..radix {
            let mut accumulator = Complex64::new(0.0, 0.0);

            for (j, value) in aligned.iter().enumerate() {
                let angle = -2.0 * PI * (j * q) as f64 / radix as f64;
                let weight = Complex64::new(angle.cos(), angle.sin());

                accumulator += *value * weight;
            }

            output[k + m * q] = accumulator;
        }
    }

    output
}

#[test]
fn r16_5_5_staged_dit_matches_reference() {
    for length in [15, 135, 16, 256] {
        let plan = MixedRadixFft1Plan::new(length);

        let input: Vec<_> = (0..length)
            .map(|i| Complex64::new((i as f64 * 0.17).sin(), (i as f64 * 0.11).cos()))
            .collect();

        let expected = mixed_radix_fft1(&plan, MixedRadixDirection::Forward, &input);

        let actual = execute_staged_dit(&input, plan.radices());

        let max_error = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0_f64, f64::max);

        println!(
            "R16_5_5_LENGTH={} RADICES={:?} MAX_ERROR={:.9e}",
            length,
            plan.radices(),
            max_error
        );

        assert!(
            max_error < 1e-10,
            "Staged DIT mismatch for length {length}: {max_error:e}"
        );
    }

    println!("R16_5_5_STAGED_DIT=PASS");
}
