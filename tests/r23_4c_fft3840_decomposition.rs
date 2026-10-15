//! R23.4C.2: independent plaintext FFT3840 decomposition oracle.
//!
//! Verify FFT15 -> cross-stage twiddle -> FFT256 against the direct
//! DFT for selected frequencies, without relying on CKKS arithmetic.

use num_complex::Complex64;
use std::f64::consts::PI;

const WIDTH: usize = 3840;
const BRANCHES: usize = 15;
const LOCAL: usize = 256;

fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * (exponent % n) as f64 / n as f64)
}

fn input(n: usize) -> Complex64 {
    let real = ((n * 17 + 11) % 101) as f64 / 101.0 - 0.5;
    let imag = ((n * 29 + 7) % 97) as f64 / 97.0 - 0.5;
    Complex64::new(real, imag)
}

fn direct_dft(k: usize) -> Complex64 {
    (0..WIDTH).map(|n| input(n) * root(WIDTH, n * k)).sum()
}

fn decomposed_dft(h: usize, j: usize) -> Complex64 {
    let mut result = Complex64::new(0.0, 0.0);

    for b in 0..LOCAL {
        let mut fft15 = Complex64::new(0.0, 0.0);

        for a in 0..BRANCHES {
            fft15 += input(LOCAL * a + b) * root(BRANCHES, a * h);
        }

        let cross_twiddle = root(WIDTH, h * b);
        let fft256_twiddle = root(LOCAL, j * b);

        result += fft15 * cross_twiddle * fft256_twiddle;
    }

    result
}

#[test]
fn r23_4c_fft3840_decomposition_matches_direct_dft() {
    let selected = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (7, 0),
        (7, 17),
        (7, 127),
        (7, 255),
        (14, 0),
        (14, 255),
    ];

    let mut maximum_error = 0.0_f64;

    for (h, j) in selected {
        let k = h + BRANCHES * j;
        let expected = direct_dft(k);
        let actual = decomposed_dft(h, j);
        let error = (actual - expected).norm();

        maximum_error = maximum_error.max(error);

        assert!(
            error < 1e-8,
            "FFT3840 mismatch: h={h}, j={j}, k={k}, error={error:e}"
        );
    }

    println!("R23_4C2_TESTED_FREQUENCIES={}", selected.len());
    println!("R23_4C2_MAX_ABSOLUTE_ERROR={maximum_error:.12e}");
    println!("R23_4C2_DECOMPOSITION_STATUS=PASS");
}
