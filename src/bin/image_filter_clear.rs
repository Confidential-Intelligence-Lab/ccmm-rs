use ccmm_rs::image_pipeline::{
    spatial_filter_pp, ImagePlane, PipelineStage, PngStageObserver, SpatialKernel, StageObserver,
};
use std::error::Error;
use std::path::{Path, PathBuf};

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
    let mut observer = PngStageObserver::new(&clear_dir)?;

    observer.observe(PipelineStage::Input, &image)?;
    observer.observe(PipelineStage::Normalized, &image)?;

    let kernel = SpatialKernel::gaussian3x3();
    let padded = image.zero_pad(kernel.height() / 2, kernel.width() / 2);
    observer.observe(PipelineStage::Padded, &padded)?;

    let filtered = spatial_filter_pp(&padded, &kernel);
    observer.observe(PipelineStage::SpatialReference, &filtered)?;
    observer.observe(PipelineStage::Output, &filtered)?;

    write_metadata(
        &output_root.join("metadata.txt"),
        input.as_deref(),
        &image,
        &padded,
        &filtered,
        &kernel,
    )?;

    println!("IMAGE_FILTER_CLEAR_STATUS=PASS");
    println!("INPUT_WIDTH={}", image.width());
    println!("INPUT_HEIGHT={}", image.height());
    println!("PADDED_WIDTH={}", padded.width());
    println!("PADDED_HEIGHT={}", padded.height());
    println!("OUTPUT_WIDTH={}", filtered.width());
    println!("OUTPUT_HEIGHT={}", filtered.height());
    println!("KERNEL=gaussian3x3");
    println!("BOUNDARY=zero");
    println!("SEMANTICS=correlation");
    println!("OUTPUT_DIRECTORY={}", output_root.display());

    Ok(())
}

fn write_metadata(
    path: &Path,
    input_path: Option<&str>,
    input: &ImagePlane,
    padded: &ImagePlane,
    output: &ImagePlane,
    kernel: &SpatialKernel,
) -> std::io::Result<()> {
    let source = input_path.unwrap_or("synthetic");

    let metadata = format!(
        concat!(
            "PIPELINE=image_filter_clear\n",
            "SOURCE={source}\n",
            "INPUT_WIDTH={input_width}\n",
            "INPUT_HEIGHT={input_height}\n",
            "PADDED_WIDTH={padded_width}\n",
            "PADDED_HEIGHT={padded_height}\n",
            "OUTPUT_WIDTH={output_width}\n",
            "OUTPUT_HEIGHT={output_height}\n",
            "KERNEL_WIDTH={kernel_width}\n",
            "KERNEL_HEIGHT={kernel_height}\n",
            "KERNEL=gaussian3x3\n",
            "BOUNDARY=zero\n",
            "SEMANTICS=correlation\n"
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
    );

    std::fs::write(path, metadata)
}
