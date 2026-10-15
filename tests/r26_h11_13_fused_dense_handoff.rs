use ccmm_rs::eblas::prepared_radix4_fft256::{pack_fft256_branch_diagonals, Diagonals};
use num_complex::Complex64;
use std::collections::BTreeSet;
use std::f64::consts::PI;

const SLOTS: usize = 32768;
const LANE_SLOTS: usize = 4096;
const LANES: usize = 8;
const BRANCHES: usize = 15;
const SPAN: usize = 4;

fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * (exponent % n) as f64 / n as f64)
}

fn reverse_base4(mut index: usize) -> usize {
    let mut reversed = 0;
    for _ in 0..4 {
        reversed = reversed * 4 + index % 4;
        index /= 4;
    }
    reversed
}

fn stage_diagonals(branch: usize) -> Diagonals {
    let mut result = Diagonals::new();

    for i in 0..LANE_SLOTS {
        let within = i % SPAN;
        let p = within;
        for q in 0..4 {
            let offset = q as isize - within as isize;
            let source = (i as isize + offset) as usize;
            let coefficient = root(4, p * q) * root(3840, branch * reverse_base4(source % 256));

            result
                .entry(offset)
                .or_insert_with(|| vec![Complex64::new(0.0, 0.0); LANE_SLOTS])[i] = coefficient;
        }
    }

    result
}

fn sample(branch: usize, index: usize) -> Complex64 {
    let x = index as f64;
    Complex64::new(
        (0.017 * x + branch as f64 * 0.13).sin() * 0.3,
        (0.021 * x - branch as f64 * 0.07).cos() * 0.2,
    )
}

#[test]
fn fused_branch_placement_matches_dense_first_stage() {
    let mut all_offsets = BTreeSet::new();

    for group in 0..2 {
        let branches: [Option<usize>; LANES] = std::array::from_fn(|lane| {
            let branch = group * LANES + lane;
            (branch < BRANCHES).then_some(branch)
        });

        let individual: Vec<Diagonals> = branches
            .iter()
            .map(|branch| stage_diagonals(branch.unwrap_or(0)))
            .collect();

        let references: Vec<&Diagonals> = individual.iter().collect();

        let dense_diagonals = pack_fft256_branch_diagonals(&references, LANE_SLOTS);

        let mut dense_input = vec![Complex64::new(0.0, 0.0); SLOTS];

        let sparse_inputs: Vec<Vec<Complex64>> = branches
            .iter()
            .map(|branch| {
                (0..SLOTS)
                    .map(|index| match branch {
                        Some(branch) if index < LANE_SLOTS => sample(*branch, index),
                        _ => Complex64::new(0.0, 0.0),
                    })
                    .collect()
            })
            .collect();

        for (lane, source) in sparse_inputs.iter().enumerate() {
            let begin = lane * LANE_SLOTS;
            dense_input[begin..begin + LANE_SLOTS].copy_from_slice(&source[..LANE_SLOTS]);
        }

        let mut expected = vec![Complex64::new(0.0, 0.0); SLOTS];

        for (&offset, coefficients) in &dense_diagonals {
            for (index, coefficient) in coefficients.iter().enumerate() {
                let source = (index as isize + offset).rem_euclid(SLOTS as isize) as usize;
                expected[index] += *coefficient * dense_input[source];
            }
        }

        let mut fused = vec![Complex64::new(0.0, 0.0); SLOTS];

        for (lane, branch) in branches.iter().enumerate() {
            if branch.is_none() {
                continue;
            }

            for (&offset, coefficients) in &individual[lane] {
                // Rotations are expressed relative to the original
                // sparse source ciphertext, not the packed ciphertext.
                let shift = offset - (lane * LANE_SLOTS) as isize;
                all_offsets.insert(shift.rem_euclid(SLOTS as isize));

                for (local, coefficient) in coefficients.iter().enumerate().take(LANE_SLOTS) {
                    let destination = lane * LANE_SLOTS + local;
                    let source = (destination as isize + shift).rem_euclid(SLOTS as isize) as usize;

                    fused[destination] += *coefficient * sparse_inputs[lane][source];
                }
            }
        }

        let mut max_abs = 0.0_f64;
        let mut error_sq = 0.0_f64;
        let mut reference_sq = 0.0_f64;

        for (&actual, &reference) in fused.iter().zip(&expected) {
            let error = (actual - reference).norm();
            max_abs = max_abs.max(error);
            error_sq += error * error;
            reference_sq += reference.norm_sqr();
        }

        let relative_l2 = (error_sq / reference_sq.max(f64::MIN_POSITIVE)).sqrt();

        println!("R26_H11_13_GROUP={group}");
        println!("R26_H11_13_RELATIVE_L2={relative_l2:.12e}");
        println!("R26_H11_13_MAX_ABS_ERROR={max_abs:.12e}");

        assert!(relative_l2 < 1.0e-12);
        assert!(max_abs < 1.0e-12);
    }

    println!(
        "R26_H11_13_DISTINCT_SOURCE_ROTATION_OFFSETS={}",
        all_offsets.len()
    );
    println!("R26_H11_13_LEVEL_MODEL=FUSED_FIRST_STAGE");
    println!("R26_H11_13_ENCRYPTED_VALIDATION=NOT_YET_RUN");
    println!("R26_H11_13_STATUS=PASS");
}
