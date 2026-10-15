// R16.5.14c.0: FFT135 four-stage arithmetic schedule.
// Cleartext oracle for the forthcoming encrypted implementation.
// No production-code changes.

use num_complex::Complex64;
use std::f64::consts::PI;

fn fft(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    if n == 1 {
        return input.to_vec();
    }

    let radix = [5, 3]
        .into_iter()
        .find(|&r| n % r == 0)
        .expect("unsupported radix");
    let m = n / radix;

    let branches: Vec<Vec<Complex64>> = (0..radix)
        .map(|j| {
            let samples: Vec<_> = (0..m).map(|t| input[radix * t + j]).collect();
            fft(&samples)
        })
        .collect();

    (0..n)
        .map(|k| {
            (0..radix)
                .map(|j| {
                    let angle = -2.0 * PI * (j * k) as f64 / n as f64;
                    branches[j][k % m] * Complex64::from_polar(1.0, angle)
                })
                .sum()
        })
        .collect()
}

#[test]
fn r16_5_14c0_fft135_four_stage_schedule() {
    const N: usize = 135;
    const RADICES: [usize; 4] = [5, 3, 3, 3];

    assert_eq!(RADICES.iter().product::<usize>(), N);

    let input: Vec<Complex64> = (0..N)
        .map(|i| {
            let x = i as f64;
            Complex64::new(0.005 * (0.17 * x).sin(), 0.005 * (0.11 * x).cos())
        })
        .collect();

    let actual = fft(&input);

    let reference: Vec<Complex64> = (0..N)
        .map(|k| {
            (0..N)
                .map(|t| {
                    let angle = -2.0 * PI * (k * t) as f64 / N as f64;
                    input[t] * Complex64::from_polar(1.0, angle)
                })
                .sum()
        })
        .collect();

    let error_sq: f64 = actual
        .iter()
        .zip(&reference)
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum();

    let reference_sq: f64 = reference.iter().map(Complex64::norm_sqr).sum();

    let rel_l2 = (error_sq / reference_sq).sqrt();

    println!("R16_FFT135_RADICES=5,3,3,3");
    println!("R16_FFT135_TARGET_CKKS_LEVELS=4");
    println!("R16_FFT135_REL_L2={rel_l2:.12e}");

    assert!(rel_l2 < 1e-12);
    println!("R16_FFT135_SCHEDULE_STATUS=PASS");
}
