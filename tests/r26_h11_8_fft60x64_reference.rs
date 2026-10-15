use num_complex::Complex64;
use std::f64::consts::PI;

const N: usize = 3840;
const OUTER: usize = 60;
const INNER: usize = 64;

fn twiddle(k: usize, n: usize) -> Complex64 {
    let angle = -2.0 * PI * k as f64 / n as f64;
    Complex64::new(angle.cos(), angle.sin())
}

fn direct_dft(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();

    (0..n)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(j, &x)| x * twiddle((j * k) % n, n))
                .sum()
        })
        .collect()
}

fn fft_radix4(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();

    if n == 1 {
        return input.to_vec();
    }

    assert_eq!(n % 4, 0);

    let m = n / 4;

    let branches: Vec<Vec<Complex64>> = (0..4)
        .map(|r| {
            let sub: Vec<_> = (0..m).map(|j| input[4 * j + r]).collect();
            fft_radix4(&sub)
        })
        .collect();

    let mut output = vec![Complex64::new(0.0, 0.0); n];

    for (k, value) in output.iter_mut().enumerate() {
        let q = k % m;

        *value = (0..4)
            .map(|r| branches[r][q] * twiddle((r * k) % n, n))
            .sum();
    }

    output
}

fn mixed_fft60x64(input: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(input.len(), N);

    let mut inner_outputs = vec![vec![Complex64::new(0.0, 0.0); INNER]; OUTER];

    for r in 0..OUTER {
        let sequence: Vec<_> = (0..INNER).map(|s| input[OUTER * s + r]).collect();

        inner_outputs[r] = fft_radix4(&sequence);
    }

    let mut output = vec![Complex64::new(0.0, 0.0); N];

    for k in 0..INNER {
        let outer_input: Vec<_> = (0..OUTER)
            .map(|r| inner_outputs[r][k] * twiddle((r * k) % N, N))
            .collect();

        let outer_output = direct_dft(&outer_input);

        for q in 0..OUTER {
            output[k + INNER * q] = outer_output[q];
        }
    }

    output
}

fn relative_l2(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    let error: f64 = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum();

    let reference: f64 = expected.iter().map(|v| v.norm_sqr()).sum();

    (error / reference).sqrt()
}

#[test]
fn native_fft3840_mixed_radix_60x64_matches_direct_dft() {
    let input: Vec<Complex64> = (0..N)
        .map(|i| {
            let x = i as f64;
            Complex64::new(
                (0.017 * x).sin() + 0.2 * (0.031 * x).cos(),
                0.3 * (0.023 * x).sin(),
            )
        })
        .collect();

    let actual = mixed_fft60x64(&input);

    // Independent direct-DFT reference at selected frequency bins.
    // This avoids the cost of a full O(N^2) oracle.
    let bins = [
        0, 1, 2, 3, 7, 31, 63, 64, 65, 127, 255, 256, 511, 512, 959, 1024, 1535, 1919, 1920, 2047,
        2559, 3071, 3583, 3839,
    ];

    let expected: Vec<Complex64> = bins
        .iter()
        .map(|&k| {
            input
                .iter()
                .enumerate()
                .map(|(j, &x)| x * twiddle((j * k) % N, N))
                .sum()
        })
        .collect();

    let observed: Vec<Complex64> = bins.iter().map(|&k| actual[k]).collect();

    let error = relative_l2(&observed, &expected);

    let max_abs = observed
        .iter()
        .zip(&expected)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max);

    println!("R26_H11_8_LENGTH={N}");
    println!("R26_H11_8_OUTER={OUTER}");
    println!("R26_H11_8_INNER={INNER}");
    println!("R26_H11_8_REFERENCE_BINS={}", bins.len());
    println!("R26_H11_8_RELATIVE_L2={error:.12e}");
    println!("R26_H11_8_MAX_ABS_ERROR={max_abs:.12e}");
    println!("R26_H11_8_PADDING=NONE");

    assert!(error < 1.0e-10, "relative L2 error: {error}");
    assert!(max_abs < 1.0e-8, "maximum absolute error: {max_abs}");

    println!("R26_H11_8_STATUS=PASS");
}

// Opposite orientation: 60-point transforms first,
// followed by regular radix-4 FFT64 transforms.
fn mixed_fft64_after_fft60(input: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(input.len(), N);

    let mut outer_outputs = vec![vec![Complex64::new(0.0, 0.0); OUTER]; INNER];

    for r in 0..INNER {
        let sequence: Vec<_> = (0..OUTER).map(|s| input[INNER * s + r]).collect();

        outer_outputs[r] = direct_dft(&sequence);
    }

    let mut output = vec![Complex64::new(0.0, 0.0); N];

    for k in 0..OUTER {
        let inner_input: Vec<_> = (0..INNER)
            .map(|r| outer_outputs[r][k] * twiddle((r * k) % N, N))
            .collect();

        let inner_output = fft_radix4(&inner_input);

        for q in 0..INNER {
            output[k + OUTER * q] = inner_output[q];
        }
    }

    output
}

#[test]
fn native_fft3840_fft60_then_fft64_matches_direct_dft() {
    let input: Vec<Complex64> = (0..N)
        .map(|i| {
            let x = i as f64;
            Complex64::new(
                (0.017 * x).sin() + 0.2 * (0.031 * x).cos(),
                0.3 * (0.023 * x).sin(),
            )
        })
        .collect();

    let actual = mixed_fft64_after_fft60(&input);
    let reverse = mixed_fft60x64(&input);

    let bins = [
        0, 1, 2, 3, 7, 31, 59, 60, 61, 63, 64, 127, 255, 256, 511, 959, 1024, 1535, 1919, 1920,
        2047, 2559, 3071, 3839,
    ];

    let expected: Vec<Complex64> = bins
        .iter()
        .map(|&k| {
            input
                .iter()
                .enumerate()
                .map(|(j, &x)| x * twiddle((j * k) % N, N))
                .sum()
        })
        .collect();

    let observed: Vec<Complex64> = bins.iter().map(|&k| actual[k]).collect();

    let error = relative_l2(&observed, &expected);

    let full_error = relative_l2(&actual, &reverse);

    let max_abs = observed
        .iter()
        .zip(&expected)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max);

    println!("R26_H11_9_ORIENTATION=FFT60_THEN_FFT64");
    println!("R26_H11_9_REFERENCE_BINS={}", bins.len());
    println!("R26_H11_9_RELATIVE_L2={error:.12e}");
    println!("R26_H11_9_MAX_ABS_ERROR={max_abs:.12e}");
    println!("R26_H11_9_FULL_ORIENTATION_RELATIVE_L2={full_error:.12e}");
    println!("R26_H11_9_PADDING=NONE");

    assert!(error < 1.0e-10);
    assert!(max_abs < 1.0e-8);
    assert!(full_error < 1.0e-10);

    println!("R26_H11_9_STATUS=PASS");
}
