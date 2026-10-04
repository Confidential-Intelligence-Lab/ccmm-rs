//! Image-processing support for clear and encrypted application pipelines.
//!
//! This module owns image representation, deterministic preprocessing,
//! application-level spatial filtering, and visualization. Numerical filtering
//! semantics are delegated to eBLAS rather than reimplemented here.

use crate::eblas::correlation::{correlate_2d_pp, Correlation2dShape};
use crate::eblas::NhwcShape;
use image::{GrayImage, ImageError, Luma};
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
    fn spatial_filter_uses_valid_extent() {
        let image = ImagePlane::synthetic(8, 6);
        let kernel = SpatialKernel::gaussian3x3();

        let output = spatial_filter_pp(&image, &kernel);

        assert_eq!(output.width(), 6);
        assert_eq!(output.height(), 4);
        assert!(output.values().iter().all(|value| value.is_finite()));
    }
}
