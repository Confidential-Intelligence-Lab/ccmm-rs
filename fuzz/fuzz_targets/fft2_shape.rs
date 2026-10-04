#![no_main]

use libfuzzer_sys::fuzz_target;

use ccmm_rs::eblas::fft::Fft2Shape;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    /*
     * Keep total logical size bounded while exercising rectangular and
     * degenerate 1xN / Nx1 transforms.
     */
    let row_exp = (data[0] % 11) as u32;
    let col_exp = (data[1] % 11) as u32;

    let rows = 1usize << row_exp;
    let cols = 1usize << col_exp;

    if rows.checked_mul(cols).is_none()
        || rows * cols > (1usize << 16)
    {
        return;
    }

    let shape = Fft2Shape::new(rows, cols);

    assert_eq!(shape.rows(), rows);
    assert_eq!(shape.cols(), cols);
    assert_eq!(shape.elements(), rows * cols);

    assert_eq!(shape.row_stages(), col_exp);
    assert_eq!(shape.column_stages(), row_exp);
    assert_eq!(shape.stages(), row_exp + col_exp);

    let expected = shape.elements()
        * shape.stages() as usize
        / 2;

    assert_eq!(shape.butterflies(), expected);
});
