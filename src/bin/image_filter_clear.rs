use ccmm_rs::image_pipeline::{
    absolute_error_image, complex_log_magnitude_image, complex_phase_image, complex_real_image,
    frequency_filter_pp, max_abs_image_error, spatial_filter_pp, ImagePlane, PipelineStage,
    PngStageObserver, SpatialKernel, StageObserver,
};
use std::error::Error;
use std::path::{Path, PathBuf};

const TOLERANCE: f64 = 1.0e-10;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);

    let input = args.next();
    let output_root = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("results/image_filter/clear_demo"));

    if args.next().is_some() {
        return Err("usage: image_filter_clear [input.png] [output-directory]".into());
    }

    let image = match input.as_deref() {
        Some(path) => ImagePlane::load(path)?,
        None => ImagePlane::synthetic(256, 256),
    };

    let clear_dir = output_root.join("clear");
    let comparison_dir = output_root.join("comparison");

    std::fs::create_dir_all(&comparison_dir)?;

    let mut observer = PngStageObserver::new(&clear_dir)?;

    observer.observe(PipelineStage::Input, &image)?;
    observer.observe(PipelineStage::Normalized, &image)?;

    let kernel = SpatialKernel::gaussian3x3();

    /*
     * Symmetric zero padding allows valid 3x3 correlation to retain the
     * original image extent.
     */
    let padded = image.zero_pad(kernel.height() / 2, kernel.width() / 2);

    observer.observe(PipelineStage::Padded, &padded)?;

    let spatial = spatial_filter_pp(&padded, &kernel);

    observer.observe(PipelineStage::SpatialReference, &spatial)?;

    let frequency = frequency_filter_pp(&padded, &kernel);

    let spectrum_magnitude = complex_log_magnitude_image(
        frequency.image_spectrum(),
        frequency.fft_cols(),
        frequency.fft_rows(),
    );
    observer.observe(PipelineStage::SpectrumMagnitude, &spectrum_magnitude)?;

    let spectrum_phase = complex_phase_image(
        frequency.image_spectrum(),
        frequency.fft_cols(),
        frequency.fft_rows(),
    );
    observer.observe(PipelineStage::SpectrumPhase, &spectrum_phase)?;

    let filter_spectrum = complex_log_magnitude_image(
        frequency.filter_spectrum(),
        frequency.fft_cols(),
        frequency.fft_rows(),
    );
    observer.observe(PipelineStage::FilterSpectrum, &filter_spectrum)?;

    let filtered_spectrum = complex_log_magnitude_image(
        frequency.filtered_spectrum(),
        frequency.fft_cols(),
        frequency.fft_rows(),
    );
    observer.observe(PipelineStage::FilteredSpectrumMagnitude, &filtered_spectrum)?;

    let inverse = complex_real_image(
        frequency.inverse_canvas(),
        frequency.fft_cols(),
        frequency.fft_rows(),
    );
    observer.observe(PipelineStage::Inverse, &inverse)?;

    observer.observe(PipelineStage::ValidCrop, frequency.output())?;
    observer.observe(PipelineStage::Output, frequency.output())?;

    let absolute_error = absolute_error_image(&spatial, frequency.output());

    absolute_error.save_png(comparison_dir.join("output_abs_error.png"))?;

    let max_abs = max_abs_image_error(&spatial, frequency.output());

    if max_abs > TOLERANCE {
        return Err(format!(
            "frequency-domain output does not match spatial reference: \
             max_abs={max_abs:e}, tolerance={TOLERANCE:e}"
        )
        .into());
    }

    write_metrics(
        &output_root.join("metrics.csv"),
        max_abs,
        frequency.fft_rows(),
        frequency.fft_cols(),
    )?;

    write_metadata(
        &output_root.join("metadata.txt"),
        input.as_deref(),
        &image,
        &padded,
        frequency.output(),
        &kernel,
        frequency.fft_rows(),
        frequency.fft_cols(),
        max_abs,
    )?;

    println!("IMAGE_FILTER_CLEAR_STATUS=PASS");
    println!("INPUT_WIDTH={}", image.width());
    println!("INPUT_HEIGHT={}", image.height());
    println!("PADDED_WIDTH={}", padded.width());
    println!("PADDED_HEIGHT={}", padded.height());
    println!("FFT_ROWS={}", frequency.fft_rows());
    println!("FFT_COLS={}", frequency.fft_cols());
    println!("OUTPUT_WIDTH={}", frequency.output().width());
    println!("OUTPUT_HEIGHT={}", frequency.output().height());
    println!("KERNEL=gaussian3x3");
    println!("BOUNDARY=zero");
    println!("SEMANTICS=correlation");
    println!("MAX_ABS_SPATIAL_VS_FREQUENCY={max_abs:.12e}");
    println!("OUTPUT_DIRECTORY={}", output_root.display());

    Ok(())
}

fn write_metrics(
    path: &Path,
    max_abs: f64,
    fft_rows: usize,
    fft_cols: usize,
) -> std::io::Result<()> {
    std::fs::write(
        path,
        format!(
            "metric,value\n\
             max_abs_spatial_vs_frequency,{max_abs:.12e}\n\
             fft_rows,{fft_rows}\n\
             fft_cols,{fft_cols}\n"
        ),
    )
}

#[allow(clippy::too_many_arguments)]
fn write_metadata(
    path: &Path,
    input_path: Option<&str>,
    input: &ImagePlane,
    padded: &ImagePlane,
    output: &ImagePlane,
    kernel: &SpatialKernel,
    fft_rows: usize,
    fft_cols: usize,
    max_abs: f64,
) -> std::io::Result<()> {
    let source = input_path.unwrap_or("synthetic");

    let metadata = format!(
        concat!(
            "PIPELINE=image_filter_clear_frequency\n",
            "SOURCE={source}\n",
            "INPUT_WIDTH={input_width}\n",
            "INPUT_HEIGHT={input_height}\n",
            "PADDED_WIDTH={padded_width}\n",
            "PADDED_HEIGHT={padded_height}\n",
            "FFT_ROWS={fft_rows}\n",
            "FFT_COLS={fft_cols}\n",
            "OUTPUT_WIDTH={output_width}\n",
            "OUTPUT_HEIGHT={output_height}\n",
            "KERNEL_WIDTH={kernel_width}\n",
            "KERNEL_HEIGHT={kernel_height}\n",
            "KERNEL=gaussian3x3\n",
            "BOUNDARY=zero\n",
            "SEMANTICS=correlation\n",
            "FREQUENCY_REALIZATION=linear_convolution_with_reversed_kernel\n",
            "MAX_ABS_SPATIAL_VS_FREQUENCY={max_abs:.12e}\n"
        ),
        source = source,
        input_width = input.width(),
        input_height = input.height(),
        padded_width = padded.width(),
        padded_height = padded.height(),
        output_width = output.width(),
        output_height = output.height(),
        kernel_width = kernel.width(),
        kernel_height = kernel.height(),
        fft_rows = fft_rows,
        fft_cols = fft_cols,
        max_abs = max_abs,
    );

    std::fs::write(path, metadata)
}
