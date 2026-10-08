use ccmm_rs::eblas::fft::{fft2_pp, Fft2Shape, FftDirection};
use image::{imageops, GrayImage, Luma};
use num_complex::Complex64;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const UHD_WIDTH: u32 = 3840;
const UHD_HEIGHT: u32 = 2160;
const FFT_DIM: usize = 4096;

fn save_spectrum(path: &Path, spectrum: &[Complex64]) {
    let mut max_log = 0.0_f64;

    for value in spectrum {
        max_log = max_log.max(value.norm().ln_1p());
    }

    assert!(max_log.is_finite() && max_log > 0.0);

    let mut output = GrayImage::new(FFT_DIM as u32, FFT_DIM as u32);

    for row in 0..FFT_DIM {
        for col in 0..FFT_DIM {
            let source = row * FFT_DIM + col;
            let magnitude = spectrum[source].norm().ln_1p();

            let normalized = (magnitude / max_log).clamp(0.0, 1.0);
            let pixel = (255.0 * normalized).round() as u8;

            let shifted_row = (row + FFT_DIM / 2) % FFT_DIM;
            let shifted_col = (col + FFT_DIM / 2) % FFT_DIM;

            output.put_pixel(shifted_col as u32, shifted_row as u32, Luma([pixel]));
        }
    }

    output.save(path).expect("failed to save UHD FFT spectrum");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    assert_eq!(
        args.len(),
        3,
        "usage: clear_uhd_4k_fft <source-image> <output-directory>"
    );

    let source = PathBuf::from(&args[1]);
    let output_dir = PathBuf::from(&args[2]);

    fs::create_dir_all(&output_dir).expect("failed to create UHD output directory");

    let source_image = image::open(&source)
        .expect("failed to load source image")
        .to_luma8();

    let source_width = source_image.width();
    let source_height = source_image.height();

    assert!(
        source_width >= UHD_WIDTH && source_height >= UHD_HEIGHT,
        "source must be at least 3840x2160"
    );

    let crop_x = (source_width - UHD_WIDTH) / 2;
    let crop_y = (source_height - UHD_HEIGHT) / 2;

    let uhd = imageops::crop_imm(&source_image, crop_x, crop_y, UHD_WIDTH, UHD_HEIGHT).to_image();

    let uhd_path = output_dir.join("01_input_uhd_3840x2160.png");

    uhd.save(&uhd_path).expect("failed to save UHD image");

    let mut padded = GrayImage::new(FFT_DIM as u32, FFT_DIM as u32);

    imageops::replace(&mut padded, &uhd, 0, 0);

    let padded_path = output_dir.join("02_padded_4096x4096.png");

    padded
        .save(&padded_path)
        .expect("failed to save padded image");

    let values: Vec<Complex64> = padded
        .pixels()
        .map(|pixel| Complex64::new(f64::from(pixel[0]), 0.0))
        .collect();

    let shape = Fft2Shape::new(FFT_DIM, FFT_DIM);

    println!("R21_UHD_BEGIN");
    println!("R21_UHD_SOURCE={}", source.display());
    println!("R21_UHD_SOURCE_DIMENSION={source_width}x{source_height}");
    println!("R21_UHD_CROP_ORIGIN={crop_x},{crop_y}");
    println!("R21_UHD_DIMENSION=3840x2160");
    println!("R21_UHD_FFT_DIMENSION=4096x4096");
    println!("R21_UHD_PADDING=ZERO");
    println!("R21_UHD_PLACEMENT=TOP_LEFT");
    println!("R21_UHD_PIXEL_ENCODING=LUMA8");
    println!("R21_UHD_FFT_INPUT_SCALE=RAW_0_TO_255");
    println!("R21_UHD_VISUALIZATION=FFTShift_Log1pMagnitude_Normalized8Bit");

    let start = Instant::now();

    let spectrum = fft2_pp(shape, FftDirection::Forward, &values);

    let fft_ms = start.elapsed().as_millis();

    let spectrum_path = output_dir.join("03_clear_fft_spectrum.png");

    save_spectrum(&spectrum_path, &spectrum);

    let spatial_energy: f64 = values.iter().map(|value| value.norm_sqr()).sum();

    let spectral_energy: f64 = spectrum.iter().map(|value| value.norm_sqr()).sum();

    let expected_spectral_energy = spatial_energy * shape.elements() as f64;

    let parseval_relative_error = if expected_spectral_energy > 0.0 {
        ((spectral_energy - expected_spectral_energy) / expected_spectral_energy).abs()
    } else {
        spectral_energy.abs()
    };

    let status = if parseval_relative_error <= 1.0e-10 {
        "PASS"
    } else {
        "FAIL"
    };

    println!("R21_UHD_FFT_MS={fft_ms}");
    println!("R21_UHD_SPATIAL_ENERGY={spatial_energy:.12e}");
    println!("R21_UHD_SPECTRAL_ENERGY={spectral_energy:.12e}");
    println!("R21_UHD_PARSEVAL_REL_ERROR={parseval_relative_error:.12e}");
    println!("R21_UHD_INPUT={}", uhd_path.display());
    println!("R21_UHD_PADDED={}", padded_path.display());
    println!("R21_UHD_SPECTRUM={}", spectrum_path.display());
    println!("R21_UHD_STATUS={status}");
    println!("R21_UHD_END");

    assert_eq!(status, "PASS", "UHD FFT Parseval validation failed");
}
