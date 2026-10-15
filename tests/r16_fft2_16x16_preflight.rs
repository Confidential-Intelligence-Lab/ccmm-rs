use ccmm_rs::eblas::fft::{
    execute_repeated_packed_fft2_dif_stage_pp, fft2_pp, packed_fft2_dif_stage_schedule, Fft2Shape,
    FftDirection, PackedFft2DifStageDiagonals,
};
use num_complex::Complex64;

fn reverse_four_bits(value: usize) -> usize {
    value.reverse_bits() >> (usize::BITS - 4)
}

#[test]
fn r16_fft2_16x16_plaintext_preflight() {
    let shape = Fft2Shape::new(16, 16);

    let input: Vec<Complex64> = (0..256)
        .map(|index| {
            Complex64::new(
                ((index * 7) % 31) as f64 / 128.0,
                ((index * 11) % 29) as f64 / 256.0,
            )
        })
        .collect();

    let schedule = packed_fft2_dif_stage_schedule(shape);

    assert_eq!(schedule.len(), 8);

    let rotations: Vec<usize> = schedule.iter().map(|stage| stage.rotation).collect();

    assert_eq!(rotations, [8, 4, 2, 1, 128, 64, 32, 16]);

    let mut actual = input.clone();

    for stage in &schedule {
        let diagonals =
            PackedFft2DifStageDiagonals::new(shape, stage.axis, stage.span, FftDirection::Forward);

        actual = execute_repeated_packed_fft2_dif_stage_pp(&actual, &diagonals, 1);
    }

    let reference = fft2_pp(shape, FftDirection::Forward, &input);

    let mut error_squared = 0.0_f64;
    let mut reference_squared = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for row in 0..16 {
        for col in 0..16 {
            let actual_index = row * 16 + col;
            let expected_index = reverse_four_bits(row) * 16 + reverse_four_bits(col);

            let error = (actual[actual_index] - reference[expected_index]).norm();

            error_squared += error * error;
            reference_squared += reference[expected_index].norm_sqr();
            max_abs = max_abs.max(error);
        }
    }

    let rel_l2 = (error_squared / reference_squared).sqrt();

    println!("R16_FFT2_16X16_STAGES={}", schedule.len());
    println!("R16_FFT2_16X16_REL_L2={rel_l2:.12e}");
    println!("R16_FFT2_16X16_MAX_ABS={max_abs:.12e}");

    assert!(rel_l2 < 1.0e-12);
    assert!(max_abs < 1.0e-10);

    println!("R16_FFT2_16X16_PREFLIGHT=PASS");
}
