//! Image-processing support for clear and encrypted application pipelines.
//!
//! This module owns image representation, deterministic preprocessing,
//! application-level spatial filtering, and visualization. Numerical filtering
//! semantics are delegated to eBLAS rather than reimplemented here.

use crate::eblas::correlation::{correlate_2d_pp, Correlation2dShape};
use crate::eblas::fft::{fft2_pp, Fft2Shape, FftDirection};
use crate::eblas::NhwcShape;
use image::{GrayImage, ImageError, Luma};
use num_complex::Complex64;
use std::path::Path;

/// One normalized grayscale image stored in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePlane {
    width: usize,
    height: usize,
    values: Vec<f64>,
}

impl ImagePlane {
    /// Constructs an image plane and validates its storage extent.
    pub fn new(width: usize, height: usize, values: Vec<f64>) -> Self {
        assert!(width > 0, "image width must be positive");
        assert!(height > 0, "image height must be positive");

        let expected = width
            .checked_mul(height)
            .expect("image element count overflow");

        assert_eq!(
            values.len(),
            expected,
            "image storage length must match width times height"
        );

        Self {
            width,
            height,
            values,
        }
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Deterministic synthetic image used by tests, CI, and demonstrations.
    ///
    /// The image combines a horizontal ramp, a vertical ramp, and a bright
    /// centered square so smoothing remains visually obvious.
    pub fn synthetic(width: usize, height: usize) -> Self {
        assert!(width > 0, "synthetic image width must be positive");
        assert!(height > 0, "synthetic image height must be positive");

        let mut values = Vec::with_capacity(
            width
                .checked_mul(height)
                .expect("synthetic image element count overflow"),
        );

        let square_x0 = width / 4;
        let square_x1 = width - width / 4;
        let square_y0 = height / 4;
        let square_y1 = height - height / 4;

        for y in 0..height {
            for x in 0..width {
                let fx = if width > 1 {
                    x as f64 / (width - 1) as f64
                } else {
                    0.0
                };
                let fy = if height > 1 {
                    y as f64 / (height - 1) as f64
                } else {
                    0.0
                };

                let base = 0.15 + 0.25 * fx + 0.20 * fy;
                let square = if x >= square_x0 && x < square_x1 && y >= square_y0 && y < square_y1 {
                    0.40
                } else {
                    0.0
                };

                values.push((base + square).clamp(0.0, 1.0));
            }
        }

        Self::new(width, height, values)
    }

    /// Loads a PNG-compatible image and converts it to normalized grayscale.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ImageError> {
        let gray = image::open(path)?.to_luma8();
        let (width, height) = gray.dimensions();

        let values = gray
            .pixels()
            .map(|pixel| f64::from(pixel[0]) / 255.0)
            .collect();

        Ok(Self::new(width as usize, height as usize, values))
    }

    /// Saves the plane as an 8-bit grayscale PNG.
    ///
    /// Values are clamped to `[0,1]` for visualization only.
    pub fn save_png<P: AsRef<Path>>(&self, path: P) -> Result<(), ImageError> {
        let width = u32::try_from(self.width).expect("image width exceeds PNG dimensions");
        let height = u32::try_from(self.height).expect("image height exceeds PNG dimensions");

        let mut image = GrayImage::new(width, height);

        for y in 0..self.height {
            for x in 0..self.width {
                let value = self.values[y * self.width + x].clamp(0.0, 1.0);
                let quantized = (value * 255.0).round() as u8;
                image.put_pixel(x as u32, y as u32, Luma([quantized]));
            }
        }

        image.save(path)
    }

    /// Adds symmetric zero padding around the image.
    pub fn zero_pad(&self, pad_y: usize, pad_x: usize) -> Self {
        let width = self
            .width
            .checked_add(pad_x.checked_mul(2).expect("horizontal padding overflow"))
            .expect("padded image width overflow");
        let height = self
            .height
            .checked_add(pad_y.checked_mul(2).expect("vertical padding overflow"))
            .expect("padded image height overflow");

        let mut values = vec![0.0; width * height];

        for y in 0..self.height {
            let source = y * self.width;
            let destination = (y + pad_y) * width + pad_x;

            values[destination..destination + self.width]
                .copy_from_slice(&self.values[source..source + self.width]);
        }

        Self::new(width, height, values)
    }
}

/// Public spatial filter kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialKernel {
    width: usize,
    height: usize,
    values: Vec<f64>,
}

impl SpatialKernel {
    pub fn new(width: usize, height: usize, values: Vec<f64>) -> Self {
        assert!(width > 0, "kernel width must be positive");
        assert!(height > 0, "kernel height must be positive");

        let expected = width
            .checked_mul(height)
            .expect("kernel element count overflow");

        assert_eq!(
            values.len(),
            expected,
            "kernel storage length must match width times height"
        );

        Self {
            width,
            height,
            values,
        }
    }

    /// Normalized 3x3 Gaussian-like smoothing kernel.
    pub fn gaussian3x3() -> Self {
        Self::new(
            3,
            3,
            vec![
                1.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
                2.0 / 16.0,
                4.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
            ],
        )
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// Applies one public kernel using the canonical eBLAS valid-correlation path.
pub fn spatial_filter_pp(image: &ImagePlane, kernel: &SpatialKernel) -> ImagePlane {
    assert!(
        kernel.width <= image.width,
        "kernel width cannot exceed image width"
    );
    assert!(
        kernel.height <= image.height,
        "kernel height cannot exceed image height"
    );

    let input_shape = NhwcShape::new(1, image.height, image.width, 1);
    let shape = Correlation2dShape::new(input_shape, kernel.height, kernel.width, 1);

    let output = correlate_2d_pp(shape, image.values(), kernel.values());

    ImagePlane::new(shape.output_width(), shape.output_height(), output)
}

/// Intermediate values produced by clear frequency-domain filtering.
///
/// These values are intentionally exposed so application and validation
/// binaries can visualize every meaningful stage without embedding image I/O
/// into the FFT implementation itself.
#[derive(Debug, Clone)]
pub struct FrequencyFilterResult {
    fft_rows: usize,
    fft_cols: usize,
    image_spectrum: Vec<Complex64>,
    filter_spectrum: Vec<Complex64>,
    filtered_spectrum: Vec<Complex64>,
    inverse_canvas: Vec<Complex64>,
    output: ImagePlane,
}

impl FrequencyFilterResult {
    pub const fn fft_rows(&self) -> usize {
        self.fft_rows
    }

    pub const fn fft_cols(&self) -> usize {
        self.fft_cols
    }

    pub fn image_spectrum(&self) -> &[Complex64] {
        &self.image_spectrum
    }

    pub fn filter_spectrum(&self) -> &[Complex64] {
        &self.filter_spectrum
    }

    pub fn filtered_spectrum(&self) -> &[Complex64] {
        &self.filtered_spectrum
    }

    pub fn inverse_canvas(&self) -> &[Complex64] {
        &self.inverse_canvas
    }

    pub fn output(&self) -> &ImagePlane {
        &self.output
    }
}

/// Computes the next power of two for a positive extent.
fn next_power_of_two_extent(value: usize) -> usize {
    assert!(value > 0, "FFT extent must be positive");

    value
        .checked_next_power_of_two()
        .expect("FFT extent power-of-two overflow")
}

/// Embeds one real image into the upper-left corner of a complex FFT canvas.
fn embed_image_in_complex_canvas(image: &ImagePlane, rows: usize, cols: usize) -> Vec<Complex64> {
    assert!(rows >= image.height());
    assert!(cols >= image.width());

    let mut canvas = vec![Complex64::new(0.0, 0.0); rows * cols];

    for row in 0..image.height() {
        for col in 0..image.width() {
            canvas[row * cols + col] =
                Complex64::new(image.values()[row * image.width() + col], 0.0);
        }
    }

    canvas
}

/// Embeds the spatial kernel reversed in both axes.
///
/// The existing eBLAS spatial oracle implements correlation:
///
/// `y[i,j] = sum_{u,v} x[i+u,j+v] h[u,v]`.
///
/// Linear convolution with the doubly reversed kernel produces the same
/// values, with the valid-correlation region beginning at
/// `(kernel_height - 1, kernel_width - 1)` in the full convolution.
fn embed_reversed_kernel_in_complex_canvas(
    kernel: &SpatialKernel,
    rows: usize,
    cols: usize,
) -> Vec<Complex64> {
    assert!(rows >= kernel.height());
    assert!(cols >= kernel.width());

    let mut canvas = vec![Complex64::new(0.0, 0.0); rows * cols];

    for row in 0..kernel.height() {
        for col in 0..kernel.width() {
            let source_row = kernel.height() - 1 - row;
            let source_col = kernel.width() - 1 - col;

            canvas[row * cols + col] = Complex64::new(
                kernel.values()[source_row * kernel.width() + source_col],
                0.0,
            );
        }
    }

    canvas
}

/// Computes clear frequency-domain valid correlation.
///
/// The operation is mathematically equivalent to [`spatial_filter_pp`]:
/// correlation is realized through linear convolution with the kernel
/// reversed in both spatial axes.
pub fn frequency_filter_pp(image: &ImagePlane, kernel: &SpatialKernel) -> FrequencyFilterResult {
    assert!(
        kernel.width() <= image.width(),
        "kernel width cannot exceed image width"
    );
    assert!(
        kernel.height() <= image.height(),
        "kernel height cannot exceed image height"
    );

    let full_rows = image
        .height()
        .checked_add(kernel.height() - 1)
        .expect("frequency-filter row extent overflow");
    let full_cols = image
        .width()
        .checked_add(kernel.width() - 1)
        .expect("frequency-filter column extent overflow");

    let fft_rows = next_power_of_two_extent(full_rows);
    let fft_cols = next_power_of_two_extent(full_cols);
    let shape = Fft2Shape::new(fft_rows, fft_cols);

    let image_canvas = embed_image_in_complex_canvas(image, fft_rows, fft_cols);
    let kernel_canvas = embed_reversed_kernel_in_complex_canvas(kernel, fft_rows, fft_cols);

    let image_spectrum = fft2_pp(shape, FftDirection::Forward, &image_canvas);
    let filter_spectrum = fft2_pp(shape, FftDirection::Forward, &kernel_canvas);

    let filtered_spectrum: Vec<Complex64> = image_spectrum
        .iter()
        .zip(&filter_spectrum)
        .map(|(image_value, filter_value)| image_value * filter_value)
        .collect();

    let inverse_canvas = fft2_pp(shape, FftDirection::Inverse, &filtered_spectrum);

    let output_height = image.height() - kernel.height() + 1;
    let output_width = image.width() - kernel.width() + 1;

    let crop_row = kernel.height() - 1;
    let crop_col = kernel.width() - 1;

    let mut output = Vec::with_capacity(output_height * output_width);

    for row in 0..output_height {
        for col in 0..output_width {
            output.push(inverse_canvas[(row + crop_row) * fft_cols + col + crop_col].re);
        }
    }

    FrequencyFilterResult {
        fft_rows,
        fft_cols,
        image_spectrum,
        filter_spectrum,
        filtered_spectrum,
        inverse_canvas,
        output: ImagePlane::new(output_width, output_height, output),
    }
}

/// Creates a normalized log-magnitude visualization of complex data.
pub fn complex_log_magnitude_image(
    values: &[Complex64],
    width: usize,
    height: usize,
) -> ImagePlane {
    assert_eq!(values.len(), width * height);

    let logs: Vec<f64> = values.iter().map(|value| value.norm().ln_1p()).collect();

    let maximum = logs.iter().copied().fold(0.0_f64, f64::max);

    let normalized = if maximum > 0.0 {
        logs.into_iter().map(|value| value / maximum).collect()
    } else {
        vec![0.0; values.len()]
    };

    ImagePlane::new(width, height, normalized)
}

/// Creates a phase visualization mapping `[-pi, pi]` onto `[0,1]`.
pub fn complex_phase_image(values: &[Complex64], width: usize, height: usize) -> ImagePlane {
    assert_eq!(values.len(), width * height);

    let values = values
        .iter()
        .map(|value| (value.arg() + std::f64::consts::PI) / (2.0 * std::f64::consts::PI))
        .collect();

    ImagePlane::new(width, height, values)
}

/// Creates an image from the real component of one complex canvas.
///
/// This is intended for decoded/inverse-transform visualization. Values are
/// preserved numerically and clamped only when the PNG is written.
pub fn complex_real_image(values: &[Complex64], width: usize, height: usize) -> ImagePlane {
    assert_eq!(values.len(), width * height);

    ImagePlane::new(width, height, values.iter().map(|value| value.re).collect())
}

/// Computes absolute pixel-wise error between equally shaped image planes.
pub fn absolute_error_image(lhs: &ImagePlane, rhs: &ImagePlane) -> ImagePlane {
    assert_eq!(lhs.width(), rhs.width());
    assert_eq!(lhs.height(), rhs.height());

    ImagePlane::new(
        lhs.width(),
        lhs.height(),
        lhs.values()
            .iter()
            .zip(rhs.values())
            .map(|(a, b)| (a - b).abs())
            .collect(),
    )
}

/// Maximum absolute pixel difference between equally shaped image planes.
pub fn max_abs_image_error(lhs: &ImagePlane, rhs: &ImagePlane) -> f64 {
    assert_eq!(lhs.width(), rhs.width());
    assert_eq!(lhs.height(), rhs.height());

    lhs.values()
        .iter()
        .zip(rhs.values())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

/// Named visualization checkpoints shared by clear and encrypted pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    Input,
    Normalized,
    Padded,
    SpatialReference,
    SpectrumMagnitude,
    SpectrumPhase,
    FilterSpectrum,
    FilteredSpectrumMagnitude,
    Inverse,
    ValidCrop,
    Reconstructed,
    Output,
}

impl PipelineStage {
    pub const fn filename(self) -> &'static str {
        match self {
            Self::Input => "00_input.png",
            Self::Normalized => "01_normalized.png",
            Self::Padded => "02_padded.png",
            Self::SpatialReference => "03_spatial_reference.png",
            Self::SpectrumMagnitude => "04_fft2_magnitude.png",
            Self::SpectrumPhase => "04_fft2_phase.png",
            Self::FilterSpectrum => "05_filter_spectrum.png",
            Self::FilteredSpectrumMagnitude => "06_filtered_spectrum_magnitude.png",
            Self::Inverse => "07_ifft2.png",
            Self::ValidCrop => "08_valid_crop.png",
            Self::Reconstructed => "09_reconstructed.png",
            Self::Output => "10_output.png",
        }
    }
}

/// Minimal observer interface used by application pipelines.
///
/// Encrypted validation can later implement this interface only after an
/// explicit decrypt/decode checkpoint.
pub trait StageObserver {
    fn observe(&mut self, stage: PipelineStage, image: &ImagePlane) -> Result<(), ImageError>;
}

/// PNG-backed stage observer.
#[derive(Debug, Clone)]
pub struct PngStageObserver {
    directory: std::path::PathBuf,
}

impl PngStageObserver {
    pub fn new<P: AsRef<Path>>(directory: P) -> std::io::Result<Self> {
        std::fs::create_dir_all(directory.as_ref())?;

        Ok(Self {
            directory: directory.as_ref().to_path_buf(),
        })
    }
}

impl StageObserver for PngStageObserver {
    fn observe(&mut self, stage: PipelineStage, image: &ImagePlane) -> Result<(), ImageError> {
        image.save_png(self.directory.join(stage.filename()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_image_is_normalized() {
        let image = ImagePlane::synthetic(32, 24);

        assert_eq!(image.width(), 32);
        assert_eq!(image.height(), 24);
        assert!(image
            .values()
            .iter()
            .all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn zero_padding_preserves_original_pixels() {
        let image = ImagePlane::new(2, 2, vec![0.1, 0.2, 0.3, 0.4]);
        let padded = image.zero_pad(1, 2);

        assert_eq!(padded.width(), 6);
        assert_eq!(padded.height(), 4);

        assert_eq!(padded.values()[8], 0.1);
        assert_eq!(padded.values()[9], 0.2);
        assert_eq!(padded.values()[2 * 6 + 2], 0.3);
        assert_eq!(padded.values()[2 * 6 + 3], 0.4);
    }

    #[test]
    fn gaussian_kernel_is_normalized() {
        let kernel = SpatialKernel::gaussian3x3();
        let sum: f64 = kernel.values().iter().sum();

        assert!((sum - 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn frequency_filter_matches_spatial_correlation() {
        let image = ImagePlane::synthetic(8, 8);
        let kernel = SpatialKernel::gaussian3x3();

        let spatial = spatial_filter_pp(&image, &kernel);
        let frequency = frequency_filter_pp(&image, &kernel);

        assert_eq!(frequency.output().width(), spatial.width());
        assert_eq!(frequency.output().height(), spatial.height());

        let max_abs = max_abs_image_error(&spatial, frequency.output());

        assert!(
            max_abs < 1.0e-12,
            "frequency-domain filter must match spatial correlation: {max_abs:e}"
        );
    }

    #[test]
    fn frequency_filter_uses_power_of_two_fft_canvas() {
        let image = ImagePlane::synthetic(8, 6);
        let kernel = SpatialKernel::gaussian3x3();

        let frequency = frequency_filter_pp(&image, &kernel);

        assert!(frequency.fft_rows().is_power_of_two());
        assert!(frequency.fft_cols().is_power_of_two());

        assert!(frequency.fft_rows() >= image.height() + kernel.height() - 1);
        assert!(frequency.fft_cols() >= image.width() + kernel.width() - 1);
    }

    #[test]
    fn spatial_filter_uses_valid_extent() {
        let image = ImagePlane::synthetic(8, 6);
        let kernel = SpatialKernel::gaussian3x3();

        let output = spatial_filter_pp(&image, &kernel);

        assert_eq!(output.width(), 6);
        assert_eq!(output.height(), 4);
        assert!(output.values().iter().all(|value| value.is_finite()));
    }
}
