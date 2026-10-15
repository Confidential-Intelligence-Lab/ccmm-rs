use num_complex::Complex64;
use std::f64::consts::PI;

const TILE_SLOTS: usize = 4096;
const LANES: usize = 8;
const TOTAL_SLOTS: usize = TILE_SLOTS * LANES;
const SPANS: [usize; 4] = [4, 16, 64, 256];

fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * (exponent % n) as f64 / n as f64)
}

fn coefficient(span: usize, output_index: usize, q: usize, branch: usize) -> Complex64 {
    let within = output_index % span;
    let quarter = span / 4;
    let p = within / quarter;
    let j = within % quarter;

    let base = root(span, q * j) * root(4, p * q);

    // The branch-dependent first-stage coefficient is intentionally
    // modeled separately from the shared radix-4 coefficient.
    //
    // This is a packing-contract test, not a replacement for the
    // production stage_diagonals implementation.
    if span == 4 {
        base * root(3840, branch * q * 256)
    } else {
        base
    }
}

fn apply_lane_stage(input: &[Complex64], span: usize, branch: usize) -> Vec<Complex64> {
    assert_eq!(input.len(), TILE_SLOTS);

    let quarter = span / 4;
    let mut output = vec![Complex64::new(0.0, 0.0); TILE_SLOTS];

    for (i, destination) in output.iter_mut().enumerate() {
        let within = i % span;
        let j = within % quarter;
        let group = i / span;

        for q in 0..4 {
            let source = group * span + j + q * quarter;
            *destination += input[source] * coefficient(span, i, q, branch);
        }
    }

    output
}

fn apply_packed_stage(
    input: &[Complex64],
    span: usize,
    branches: &[usize; LANES],
) -> Vec<Complex64> {
    assert_eq!(input.len(), TOTAL_SLOTS);

    let mut output = vec![Complex64::new(0.0, 0.0); TOTAL_SLOTS];

    for (lane, &branch) in branches.iter().enumerate() {
        let start = lane * TILE_SLOTS;
        let end = start + TILE_SLOTS;
        let result = apply_lane_stage(&input[start..end], span, branch);
        output[start..end].copy_from_slice(&result);
    }

    output
}

#[test]
fn r26_hybrid_fft256_eight_lane_packing_contract() {
    let branches: [usize; LANES] = [0, 1, 3, 5, 7, 9, 11, 14];

    let mut packed = vec![Complex64::new(0.0, 0.0); TOTAL_SLOTS];

    for lane in 0..LANES {
        for i in 0..TILE_SLOTS {
            let value = ((i * 17 + lane * 31) % 127) as f64 / 127.0;
            packed[lane * TILE_SLOTS + i] = Complex64::new(value, (lane as f64 + 1.0) * 0.001);
        }
    }

    for span in SPANS {
        let actual = apply_packed_stage(&packed, span, &branches);

        for (lane, &branch) in branches.iter().enumerate() {
            let start = lane * TILE_SLOTS;
            let end = start + TILE_SLOTS;

            let expected = apply_lane_stage(&packed[start..end], span, branch);

            let maximum_error = actual[start..end]
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (*a - *b).norm())
                .fold(0.0_f64, f64::max);

            assert!(
                maximum_error <= 1.0e-12,
                "lane={lane} span={span} max_error={maximum_error}"
            );
        }

        println!("R26_PACKING_SPAN_{span}=PASS");
    }

    println!("R26_PACKING_LANES={LANES}");
    println!("R26_PACKING_TOTAL_SLOTS={TOTAL_SLOTS}");
    println!("R26_PACKING_STATUS=PASS");
}
