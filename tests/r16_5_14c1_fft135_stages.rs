// R16.5.14c.1: explicit four-stage DIF FFT135.
// Cleartext validation of butterfly indexing, twiddles,
// and final mixed-radix digit reversal.
// No production-code changes.

use num_complex::Complex64;
use std::f64::consts::PI;

const N: usize = 135;
const RADICES: [usize; 4] = [5, 3, 3, 3];

fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * (exponent % n) as f64 / n as f64)
}

fn staged_fft135(input: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(input.len(), N);

    let mut data = input.to_vec();
    let mut span = N;

    for &radix in &RADICES {
        let m = span / radix;
        let mut next = data.clone();

        for base in (0..N).step_by(span) {
            for j in 0..m {
                for k in 0..radix {
                    let mut sum = Complex64::new(0.0, 0.0);

                    for q in 0..radix {
                        sum += data[base + j + q * m] * root(radix, q * k);
                    }

                    // DIF: twiddle follows the small DFT.
                    next[base + j + k * m] = sum * root(span, j * k);
                }
            }
        }

        data = next;
        span = m;
    }

    // DIF output is in mixed-radix digit-reversed order.
    // Construct the permutation explicitly.
    let mut output = vec![Complex64::new(0.0, 0.0); N];

    for (index, value) in data.into_iter().enumerate() {
        let mut remaining = index;
        let mut digits = [0usize; 4];

        for (stage, &radix) in RADICES.iter().rev().enumerate() {
            digits[3 - stage] = remaining % radix;
            remaining /= radix;
        }

        let natural = digits[0]
            + RADICES[0] * (digits[1] + RADICES[1] * (digits[2] + RADICES[2] * digits[3]));

        output[natural] = value;
    }

    output
}

#[test]
fn r16_5_14c1_fft135_staged_schedule() {
    let input: Vec<Complex64> = (0..N)
        .map(|i| {
            let x = i as f64;
            Complex64::new(0.005 * (0.17 * x).sin(), 0.005 * (0.11 * x).cos())
        })
        .collect();

    let actual = staged_fft135(&input);

    let reference: Vec<Complex64> = (0..N)
        .map(|k| (0..N).map(|t| input[t] * root(N, k * t)).sum())
        .collect();

    let error_sq: f64 = actual
        .iter()
        .zip(&reference)
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum();

    let reference_sq: f64 = reference.iter().map(Complex64::norm_sqr).sum();

    let rel_l2 = (error_sq / reference_sq).sqrt();

    println!("R16_FFT135_STAGE_RADICES=5,3,3,3");
    println!("R16_FFT135_STAGE_COUNT=4");
    println!("R16_FFT135_STAGE_REL_L2={rel_l2:.12e}");
    println!(
        "R16_FFT135_STAGE_STATUS={}",
        if rel_l2 < 1e-12 { "PASS" } else { "FAIL" }
    );

    assert!(rel_l2 < 1e-12);
}
