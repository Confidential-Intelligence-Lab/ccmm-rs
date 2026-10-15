use ccmm_rs::eblas::prepared_radix4_fft256::pack_fft256_branch_diagonals;
use num_complex::Complex64;
use std::collections::BTreeMap;
use std::f64::consts::PI;

const LANE_SLOTS: usize = 4096;
const LANES: usize = 8;
const PACKED_SLOTS: usize = LANE_SLOTS * LANES;
const SPANS: [usize; 4] = [4, 16, 64, 256];

type Diagonals = BTreeMap<isize, Vec<Complex64>>;

fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * (exponent % n) as f64 / n as f64)
}

fn reverse_base4_4(mut value: usize) -> usize {
    let mut reversed = 0;
    for _ in 0..4 {
        reversed = reversed * 4 + value % 4;
        value /= 4;
    }
    reversed
}

// Production formula from R16.6c4, including the digit-reversed
// first-stage branch twiddle.
fn production_diagonals(slots: usize, span: usize, branch: usize) -> Diagonals {
    assert!(SPANS.contains(&span));
    assert_eq!(slots % 256, 0);
    assert!(branch < 15);

    let quarter = span / 4;
    let mut diagonals = Diagonals::new();

    for i in 0..slots {
        let within = i % span;
        let p = within / quarter;
        let j = within % quarter;

        for q in 0..4 {
            let input_within = j + q * quarter;
            let offset = input_within as isize - within as isize;
            let mut coefficient = root(span, q * j) * root(4, p * q);

            if span == 4 {
                let packed_input = (i as isize + offset) as usize;
                let natural_local = reverse_base4_4(packed_input % 256);
                coefficient *= root(3840, branch * natural_local);
            }

            diagonals
                .entry(offset)
                .or_insert_with(|| vec![Complex64::new(0.0, 0.0); slots])[i] = coefficient;
        }
    }

    diagonals
}

fn packed_diagonals(span: usize, branches: &[usize; LANES]) -> Diagonals {
    assert!(SPANS.contains(&span));

    let lanes: Vec<Diagonals> = branches
        .iter()
        .map(|&branch| production_diagonals(LANE_SLOTS, span, branch))
        .collect();

    let references: Vec<&Diagonals> = lanes.iter().collect();

    pack_fft256_branch_diagonals(&references, LANE_SLOTS)
}

fn apply_diagonals(input: &[Complex64], diagonals: &Diagonals) -> Vec<Complex64> {
    let slots = input.len();
    let mut output = vec![Complex64::new(0.0, 0.0); slots];

    for (&offset, diagonal) in diagonals {
        assert_eq!(diagonal.len(), slots);

        for i in 0..slots {
            let source = (i as isize + offset).rem_euclid(slots as isize) as usize;
            output[i] += diagonal[i] * input[source];
        }
    }

    output
}

#[test]
fn r26_h5_packed_diagonals_match_independent_production_branches() {
    let branches = [0, 1, 3, 5, 7, 9, 11, 14];

    let input: Vec<Complex64> = (0..PACKED_SLOTS)
        .map(|i| {
            let lane = i / LANE_SLOTS;
            Complex64::new(
                ((i * 17 + lane * 31) % 251) as f64 / 251.0,
                ((i * 7 + lane * 13) % 127) as f64 / 127.0,
            )
        })
        .collect();

    for span in SPANS {
        let diagonals = packed_diagonals(span, &branches);
        let actual = apply_diagonals(&input, &diagonals);

        let mut maximum_error = 0.0_f64;

        for (lane, &branch) in branches.iter().enumerate() {
            let start = lane * LANE_SLOTS;
            let end = start + LANE_SLOTS;

            let local = production_diagonals(LANE_SLOTS, span, branch);
            let expected = apply_diagonals(&input[start..end], &local);

            for (got, want) in actual[start..end].iter().zip(expected.iter()) {
                maximum_error = maximum_error.max((*got - *want).norm());
            }
        }

        assert!(
            maximum_error <= 1.0e-12,
            "span={span} maximum_error={maximum_error}"
        );

        println!("R26_H5_SPAN_{span}=PASS");
        println!("R26_H5_SPAN_{span}_MAX_ERROR={maximum_error:.12e}");
    }

    println!("R26_H5_LANES={LANES}");
    println!("R26_H5_PACKED_SLOTS={PACKED_SLOTS}");
    println!("R26_H5_STATUS=PASS");
}
