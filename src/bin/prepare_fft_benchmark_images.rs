use image::imageops::{resize, FilterType};
use image::GrayImage;
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let input = env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: prepare_fft_benchmark_images <source-image> <output-dir>");

    let output_dir = env::args()
        .nth(2)
        .map(PathBuf::from)
        .expect("usage: prepare_fft_benchmark_images <source-image> <output-dir>");

    fs::create_dir_all(&output_dir).expect("failed to create benchmark output directory");

    let image = image::open(&input)
        .expect("failed to open benchmark source image")
        .to_luma8();

    let width = image.width();
    let height = image.height();

    assert!(
        width >= 4096 && height >= 4096,
        "benchmark source must provide at least 4096 pixels in both dimensions"
    );

    /*
     * Deterministic centered square crop.
     *
     * The selected benchmark source is already square, but retaining this
     * rule makes the preparation procedure explicit and reproducible.
     */
    let side = width.min(height);
    let x = (width - side) / 2;
    let y = (height - side) / 2;

    let square: GrayImage = image::imageops::crop_imm(&image, x, y, side, side).to_image();

    println!("FFT_BENCHMARK_SOURCE={}", input.display());
    println!("FFT_BENCHMARK_SOURCE_WIDTH={width}");
    println!("FFT_BENCHMARK_SOURCE_HEIGHT={height}");
    println!("FFT_BENCHMARK_CROP_SIDE={side}");
    println!("FFT_BENCHMARK_GRAYSCALE=LUMA8");
    println!("FFT_BENCHMARK_RESIZE_FILTER=LANCZOS3");

    for dimension in [16_u32, 32, 128, 256, 512, 1024, 2048, 4096] {
        let output = resize(&square, dimension, dimension, FilterType::Lanczos3);

        let path = output_dir.join(format!(
            "fhe_fft_architecture_gray_{dimension}x{dimension}.png"
        ));

        output.save(&path).expect("failed to save benchmark image");

        println!(
            "FFT_BENCHMARK_OUTPUT dimension={} path={}",
            dimension,
            path.display()
        );
    }

    println!("FFT_BENCHMARK_PREPARATION_STATUS=PASS");
}
