use ccmm_rs::eblas::fft::{fft2_pp, Fft2Shape, FftDirection};
use image::{GrayImage, Luma};
use num_complex::Complex64;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn load_grayscale_square(path: &Path, dimension: usize) -> Vec<Complex64> {
    let image = image::open(path)
        .expect("failed to open clear FFT2 input image")
        .to_luma8();

    assert_eq!(
        image.width() as usize,
        dimension,
        "input image width must match requested FFT dimension"
    );
    assert_eq!(
        image.height() as usize,
        dimension,
        "input image height must match requested FFT dimension"
    );

    image
        .pixels()
        .map(|pixel| Complex64::new(f64::from(pixel[0]), 0.0))
        .collect()
}

fn fftshift2(values: &[Complex64], shape: Fft2Shape) -> Vec<Complex64> {
    assert_eq!(values.len(), shape.elements());

    let mut shifted = vec![Complex64::new(0.0, 0.0); values.len()];

    let row_shift = shape.rows() / 2;
    let col_shift = shape.cols() / 2;

    for row in 0..shape.rows() {
        for col in 0..shape.cols() {
            let source = row * shape.cols() + col;
            let target_row = (row + row_shift) % shape.rows();
            let target_col = (col + col_shift) % shape.cols();
            let target = target_row * shape.cols() + target_col;

            shifted[target] = values[source];
        }
    }

    shifted
}

fn save_log_magnitude_spectrum(path: &Path, spectrum: &[Complex64], shape: Fft2Shape) {
    let shifted = fftshift2(spectrum, shape);

    let log_magnitude: Vec<f64> = shifted.iter().map(|value| value.norm().ln_1p()).collect();

    let max_value = log_magnitude.iter().copied().fold(0.0_f64, f64::max);

    assert!(
        max_value.is_finite() && max_value > 0.0,
        "FFT2 spectrum must contain finite nonzero magnitude"
    );

    let mut image = GrayImage::new(shape.cols() as u32, shape.rows() as u32);

    for row in 0..shape.rows() {
        for col in 0..shape.cols() {
            let index = row * shape.cols() + col;

            let normalized = (log_magnitude[index] / max_value).clamp(0.0, 1.0);
            let pixel = (255.0 * normalized).round() as u8;

            image.put_pixel(col as u32, row as u32, Luma([pixel]));
        }
    }

    image
        .save(path)
        .expect("failed to save clear FFT2 log-magnitude image");
}

fn main() {
    let input = env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: clear_fft2_image <input-image> <dimension> <output-image>");

    let dimension = env::args()
        .nth(2)
        .expect("missing FFT dimension")
        .parse::<usize>()
        .expect("FFT dimension must be a positive power of two");

    let output = env::args()
        .nth(3)
        .map(PathBuf::from)
        .expect("missing output image path");

    assert!(
        dimension > 0 && dimension.is_power_of_two(),
        "FFT dimension must be a positive power of two"
    );

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).expect("failed to create clear FFT2 output directory");
    }

    let shape = Fft2Shape::new(dimension, dimension);
    let input_values = load_grayscale_square(&input, dimension);

    println!("CLEAR_FFT2_BEGIN");
    println!("CLEAR_FFT2_INPUT={}", input.display());
    println!("CLEAR_FFT2_DIMENSION={dimension}x{dimension}");
    println!("CLEAR_FFT2_ELEMENTS={}", shape.elements());
    println!("CLEAR_FFT2_DIRECTION=forward");
    println!("CLEAR_FFT2_VISUALIZATION=FFTShift_Log1pMagnitude_Normalized8Bit");

    let start = Instant::now();

    let spectrum = fft2_pp(shape, FftDirection::Forward, &input_values);

    let elapsed_ms = start.elapsed().as_millis();

    save_log_magnitude_spectrum(&output, &spectrum, shape);

    let max_magnitude = spectrum
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);

    let energy = spectrum.iter().map(|value| value.norm_sqr()).sum::<f64>();

    println!("CLEAR_FFT2_ELAPSED_MS={elapsed_ms}");
    println!("CLEAR_FFT2_MAX_MAGNITUDE={max_magnitude:.12e}");
    println!("CLEAR_FFT2_SPECTRAL_ENERGY={energy:.12e}");
    println!("CLEAR_FFT2_OUTPUT={}", output.display());
    println!("CLEAR_FFT2_STATUS=PASS");
    println!("CLEAR_FFT2_END");
}
