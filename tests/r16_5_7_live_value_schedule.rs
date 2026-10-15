//! R16.5.7b: explicit recursive-DIT intermediate placement.
//!
//! Each recursion node owns an output vector. Its child outputs are
//! read from separate buffers and combined into the parent's buffer.
//! This is a symbolic cleartext execution model, not CKKS execution.

use ccmm_rs::eblas::mixed_radix_fft::{mixed_radix_fft1, MixedRadixDirection, MixedRadixFft1Plan};
use num_complex::Complex64;
use std::f64::consts::PI;

#[derive(Debug, Default)]
struct ScheduleMetrics {
    combines: usize,
    moved_positions: usize,
    twiddle_multiplications: usize,
}

fn execute_with_explicit_placement(
    input: &[Complex64],
    radices: &[usize],
    metrics: &mut ScheduleMetrics,
) -> Vec<Complex64> {
    let n = input.len();

    if n == 1 {
        assert!(radices.is_empty());
        return input.to_vec();
    }

    let radix = radices[0];
    assert_eq!(n % radix, 0);

    let m = n / radix;

    let branches: Vec<Vec<Complex64>> = (0..radix)
        .map(|j| {
            let branch: Vec<_> = (0..m).map(|k| input[j + radix * k]).collect();

            execute_with_explicit_placement(&branch, &radices[1..], metrics)
        })
        .collect();

    let mut output = vec![Complex64::new(0.0, 0.0); n];

    for k in 0..m {
        let aligned: Vec<_> = (0..radix)
            .map(|j| {
                let angle = -2.0 * PI * (j * k) as f64 / n as f64;
                let twiddle = Complex64::new(angle.cos(), angle.sin());

                if j != 0 && k != 0 {
                    metrics.twiddle_multiplications += 1;
                }

                branches[j][k] * twiddle
            })
            .collect();

        for q in 0..radix {
            let mut accumulator = Complex64::new(0.0, 0.0);

            for (j, value) in aligned.iter().enumerate() {
                let angle = -2.0 * PI * (j * q) as f64 / radix as f64;

                let weight = Complex64::new(angle.cos(), angle.sin());

                accumulator += *value * weight;
            }

            let destination = k + m * q;

            // A branch-local offset is not a parent-buffer offset.
            // Count such placement changes symbolically.
            if destination != k {
                metrics.moved_positions += 1;
            }

            output[destination] = accumulator;
        }

        metrics.combines += 1;
    }

    output
}

#[test]
fn r16_5_7b_explicit_placement_matches_reference() {
    for length in [15, 135] {
        let plan = MixedRadixFft1Plan::new(length);

        let input: Vec<_> = (0..length)
            .map(|i| Complex64::new((i as f64 * 0.17).sin(), (i as f64 * 0.11).cos()))
            .collect();

        let mut metrics = ScheduleMetrics::default();

        let actual = execute_with_explicit_placement(&input, plan.radices(), &mut metrics);

        let expected = mixed_radix_fft1(&plan, MixedRadixDirection::Forward, &input);

        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0_f64, f64::max);

        assert!(
            max_error < 1e-10,
            "Explicit placement mismatch for {length}: {max_error:e}"
        );

        let expected_combines: usize = plan.radices().iter().map(|&radix| length / radix).sum();
        assert_eq!(metrics.combines, expected_combines);

        println!(
            "R16_5_7B_LENGTH={} COMBINES={} \
             MOVED_POSITIONS={} TWIDDLES={} MAX_ERROR={:.9e}",
            length,
            metrics.combines,
            metrics.moved_positions,
            metrics.twiddle_multiplications,
            max_error
        );
    }

    println!("R16_5_7B_EXPLICIT_PLACEMENT=PASS");
}
