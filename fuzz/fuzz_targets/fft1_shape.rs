#![no_main]

use libfuzzer_sys::fuzz_target;

use ccmm_rs::eblas::fft::Fft1Shape;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[..8]);

    let raw = u64::from_le_bytes(bytes);

    /*
     * Bound the exponent so valid cases remain useful and cannot request
     * pathological allocations in downstream planning.
     */
    let exponent = (raw % 21) as u32;
    let valid_length = 1usize << exponent;

    let shape = Fft1Shape::new(valid_length);

    assert_eq!(shape.length(), valid_length);
    assert_eq!(shape.stages(), exponent);

    let expected = valid_length
        .checked_mul(exponent as usize)
        .expect("bounded FFT1 butterfly count must fit")
        / 2;

    assert_eq!(shape.butterflies(), expected);
});
