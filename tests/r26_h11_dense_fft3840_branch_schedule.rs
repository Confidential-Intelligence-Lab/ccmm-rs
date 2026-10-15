use ccmm_rs::eblas::prepared_radix4_fft256::{pack_fft256_branch_diagonals, Diagonals};
use num_complex::Complex64;
use std::f64::consts::PI;

const N: usize = 256;
const LANE_SLOTS: usize = 4096;
const LANES: usize = 8;
const GROUPS: usize = 2;
const BRANCHES: usize = 15;
const SPANS: [usize; 4] = [4, 16, 64, 256];

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

fn diagonals(span: usize, branch: usize) -> Diagonals {
    assert!(SPANS.contains(&span));
    let mut result = Diagonals::new();
    let quarter = span / 4;

    for i in 0..LANE_SLOTS {
        let within = i % span;
        let p = within / quarter;
        let j = within % quarter;

        for q in 0..4 {
            let source_within = j + q * quarter;
            let offset = source_within as isize - within as isize;
            let source = (i as isize + offset) as usize;

            let mut coefficient = root(span, q * j) * root(4, p * q);

            if span == 4 {
                coefficient *= root(3840, branch * reverse_base4(source % N));
            }

            result
                .entry(offset)
                .or_insert_with(|| vec![Complex64::new(0.0, 0.0); LANE_SLOTS])[i] = coefficient;
        }
    }

    result
}

fn apply(input: &[Complex64], diagonals: &Diagonals) -> Vec<Complex64> {
    let mut output = vec![Complex64::new(0.0, 0.0); input.len()];

    for (&offset, values) in diagonals {
        for (i, &coefficient) in values.iter().enumerate() {
            let source = (i as isize + offset).rem_euclid(input.len() as isize) as usize;
            output[i] += coefficient * input[source];
        }
    }

    output
}

fn independent_dft(input: &[Complex64], branch: usize) -> Vec<Complex64> {
    assert_eq!(input.len(), N);
    let mut natural = vec![Complex64::new(0.0, 0.0); N];

    for (physical, &value) in input.iter().enumerate() {
        natural[reverse_base4(physical)] = value;
    }

    (0..N)
        .map(|k| {
            natural
                .iter()
                .enumerate()
                .map(|(n, &value)| value * root(3840, branch * n) * root(N, k * n))
                .sum()
        })
        .collect()
}

#[test]
fn r26_h11_dense_fft3840_branch_schedule() {
    let total_slots = LANES * LANE_SLOTS;
    let mut total_error_sq = 0.0;
    let mut total_reference_sq = 0.0;
    let mut max_error = 0.0_f64;
    let mut inactive_max = 0.0_f64;
    let mut active_transforms = 0;

    for group in 0..GROUPS {
        let branch_map: [Option<usize>; LANES] = std::array::from_fn(|lane| {
            let branch = group * LANES + lane;
            (branch < BRANCHES).then_some(branch)
        });

        let input: Vec<Complex64> = (0..total_slots)
            .map(|index| {
                let lane = index / LANE_SLOTS;
                let local = index % LANE_SLOTS;

                if branch_map[lane].is_none() {
                    Complex64::new(0.0, 0.0)
                } else {
                    Complex64::new(
                        ((local % 17) as f64 - 8.0) * (lane + 1) as f64 / 4096.0,
                        ((local % 11) as f64 - 5.0) / 8192.0,
                    )
                }
            })
            .collect();

        let mut actual = input.clone();

        for span in SPANS {
            let per_lane: Vec<Diagonals> = branch_map
                .iter()
                .map(|branch| diagonals(span, branch.unwrap_or(0)))
                .collect();

            let refs: Vec<&Diagonals> = per_lane.iter().collect();
            let packed = pack_fft256_branch_diagonals(&refs, LANE_SLOTS);

            actual = apply(&actual, &packed);
        }

        for (lane, branch) in branch_map.iter().enumerate() {
            let start = lane * LANE_SLOTS;
            let end = start + LANE_SLOTS;

            match branch {
                Some(branch) => {
                    for block in 0..LANE_SLOTS / N {
                        let first = start + block * N;
                        let last = first + N;
                        let expected = independent_dft(&input[first..last], *branch);

                        for (&got, &want) in actual[first..last].iter().zip(&expected) {
                            let error = (got - want).norm();
                            max_error = max_error.max(error);
                            total_error_sq += error * error;
                            total_reference_sq += want.norm_sqr();
                        }

                        active_transforms += 1;
                    }
                }
                None => {
                    for value in &actual[start..end] {
                        inactive_max = inactive_max.max(value.norm());
                    }
                }
            }
        }

        println!("R26_H11_GROUP_{group}_BRANCHES={branch_map:?}");
    }

    assert!(total_reference_sq > 0.0);
    let relative_l2 = (total_error_sq / total_reference_sq).sqrt();

    println!("R26_H11_RELATIVE_L2={relative_l2:.12e}");
    println!("R26_H11_MAX_ABS_ERROR={max_error:.12e}");
    println!("R26_H11_INACTIVE_LANE_MAX={inactive_max:.12e}");
    println!("R26_H11_ACTIVE_TRANSFORMS={active_transforms}");
    println!("R26_H11_CIPHERTEXT_GROUP_TARGET={GROUPS}");

    assert_eq!(active_transforms, BRANCHES * LANE_SLOTS / N);
    assert!(relative_l2 < 1.0e-10);
    assert!(inactive_max < 1.0e-12);

    println!("R26_H11_SCHEDULE_STATUS=PASS");
}
