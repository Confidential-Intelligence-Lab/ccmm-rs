use ccmm_rs::eblas::prepared_radix4_fft256::pack_fft256_branch_diagonals;
use num_complex::Complex64;
use std::collections::BTreeMap;
use std::f64::consts::PI;

const N: usize = 256;
const LANE_SLOTS: usize = 4096;
const LANES: usize = 8;
const BRANCHES: [usize; LANES] = [0, 1, 3, 5, 7, 9, 11, 14];
const SPANS: [usize; 4] = [4, 16, 64, 256];

type Diagonals = BTreeMap<isize, Vec<Complex64>>;

fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * (exponent % n) as f64 / n as f64)
}

fn reverse_base4(mut value: usize) -> usize {
    let mut reversed = 0;
    for _ in 0..4 {
        reversed = reversed * 4 + value % 4;
        value /= 4;
    }
    reversed
}

fn production_diagonals(span: usize, branch: usize) -> Diagonals {
    let quarter = span / 4;
    let mut diagonals = Diagonals::new();

    for i in 0..LANE_SLOTS {
        let within = i % span;
        let p = within / quarter;
        let j = within % quarter;

        for q in 0..4 {
            let input_within = j + q * quarter;
            let offset = input_within as isize - within as isize;
            let input_index = (i as isize + offset) as usize;

            let mut coefficient = root(span, q * j) * root(4, p * q);

            if span == 4 {
                coefficient *= root(3840, branch * reverse_base4(input_index % N));
            }

            diagonals
                .entry(offset)
                .or_insert_with(|| vec![Complex64::new(0.0, 0.0); LANE_SLOTS])[i] = coefficient;
        }
    }

    diagonals
}

fn apply_diagonals(input: &[Complex64], diagonals: &Diagonals) -> Vec<Complex64> {
    let mut output = vec![Complex64::new(0.0, 0.0); input.len()];

    for (&offset, diagonal) in diagonals {
        for i in 0..input.len() {
            let source = (i as isize + offset).rem_euclid(input.len() as isize) as usize;

            output[i] += diagonal[i] * input[source];
        }
    }

    output
}

// Independent O(N^2) DFT.
// No radix-4 butterflies or diagonal matrices.
//
// The branch twiddle is applied to the natural-order input.
// The caller supplies digit-reversed physical storage.
fn reference_fft256(physical_input: &[Complex64], branch: usize) -> Vec<Complex64> {
    assert_eq!(physical_input.len(), N);

    let mut natural = vec![Complex64::new(0.0, 0.0); N];

    for (physical_index, &value) in physical_input.iter().enumerate() {
        let natural_index = reverse_base4(physical_index);

        natural[natural_index] = value;
    }

    let mut output = vec![Complex64::new(0.0, 0.0); N];

    for (k, result) in output.iter_mut().enumerate() {
        let mut sum = Complex64::new(0.0, 0.0);

        for (n, &value) in natural.iter().enumerate() {
            sum += value * root(3840, branch * n) * root(N, k * n);
        }

        *result = sum;
    }

    output
}

#[test]
fn r26_h10_independent_fft256_oracle() {
    let input: Vec<Complex64> = (0..LANES * LANE_SLOTS)
        .map(|index| {
            let lane = index / LANE_SLOTS;
            let local = index % LANE_SLOTS;

            Complex64::new(
                ((local % 17) as f64 - 8.0) / 4096.0 * (lane + 1) as f64,
                ((local % 11) as f64 - 5.0) / 8192.0,
            )
        })
        .collect();

    let mut actual = input.clone();

    for span in SPANS {
        let lanes: Vec<Diagonals> = BRANCHES
            .iter()
            .map(|&branch| production_diagonals(span, branch))
            .collect();

        let refs: Vec<&Diagonals> = lanes.iter().collect();

        let packed = pack_fft256_branch_diagonals(&refs, LANE_SLOTS);

        actual = apply_diagonals(&actual, &packed);
    }

    let mut max_error = 0.0_f64;
    let mut error_sq = 0.0_f64;
    let mut reference_sq = 0.0_f64;

    for (lane, &branch) in BRANCHES.iter().enumerate() {
        let lane_start = lane * LANE_SLOTS;

        for block in 0..LANE_SLOTS / N {
            let start = lane_start + block * N;
            let end = start + N;

            let expected = reference_fft256(&input[start..end], branch);

            for (&got, &want) in actual[start..end].iter().zip(&expected) {
                let error = (got - want).norm();

                max_error = max_error.max(error);
                error_sq += error * error;
                reference_sq += want.norm_sqr();
            }
        }
    }

    assert!(reference_sq > 0.0);

    let relative_l2 = (error_sq / reference_sq).sqrt();

    println!("R26_H10_RELATIVE_L2={relative_l2:.12e}");
    println!("R26_H10_MAX_ABS_ERROR={max_error:.12e}");
    println!("R26_H10_LANES={LANES}");
    println!("R26_H10_FFT_LENGTH={N}");
    println!("R26_H10_TRANSFORMS={}", LANES * LANE_SLOTS / N);

    assert!(relative_l2 < 1.0e-10, "Independent FFT256 oracle mismatch");

    println!("R26_H10_STATUS=PASS");
}
