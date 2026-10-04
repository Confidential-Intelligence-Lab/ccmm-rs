#![no_main]

use libfuzzer_sys::fuzz_target;

use ccmm_rs::eblas::fft::{
    Fft2Shape,
    FftDirection,
    PackedFft2Axis,
    PackedFft2DifStageDiagonals,
};

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let row_exp = (data[0] % 7) as u32;
    let col_exp = (data[1] % 7) as u32;

    let rows = 1usize << row_exp;
    let cols = 1usize << col_exp;

    let shape = Fft2Shape::new(rows, cols);

    let axis = if data[2] & 1 == 0 {
        PackedFft2Axis::Rows
    } else {
        PackedFft2Axis::Columns
    };

    let direction = if data[2] & 2 == 0 {
        FftDirection::Forward
    } else {
        FftDirection::Inverse
    };

    let axis_length = match axis {
        PackedFft2Axis::Rows => cols,
        PackedFft2Axis::Columns => rows,
    };

    /*
     * A radix-2 DIF stage exists only for an axis of length >= 2.
     */
    if axis_length < 2 {
        return;
    }

    let max_stage_exp = axis_length.trailing_zeros();

    /*
     * Generate only contract-valid spans here. Rejection behavior is
     * already covered by unit tests; fuzzing should exercise valid plans
     * deeply rather than treating documented assertion failures as bugs.
     */
    let stage_exp =
        1 + (u32::from(data[3]) % max_stage_exp);

    let span = 1usize << stage_exp;

    let diagonals = PackedFft2DifStageDiagonals::new(
        shape,
        axis,
        span,
        direction,
    );

    assert_eq!(diagonals.shape(), shape);
    assert_eq!(diagonals.axis(), axis);
    assert_eq!(diagonals.span(), span);
    assert_eq!(diagonals.half(), span / 2);

    let expected_rotation = match axis {
        PackedFft2Axis::Rows => span / 2,
        PackedFft2Axis::Columns => {
            (span / 2)
                .checked_mul(cols)
                .expect("bounded column rotation must fit")
        }
    };

    assert_eq!(diagonals.rotation(), expected_rotation);
});
