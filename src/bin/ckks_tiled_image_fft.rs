use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_16384, research_profile_32768, research_profile_65536, rotation_exponent_left,
    rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsCkksLevelKeys, RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{
    execute_packed_fft2_dif_stage_cp, fft2_pp, packed_fft_dif_butterfly_cp, Fft2Shape,
    FftDirection, PackedFft2Axis, PackedFft2DifStageDiagonals,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout,
    RnsKeygenConfig,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use image::{GrayImage, Luma};
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SIGMA: f64 = 3.19;
const TOLERANCE: f64 = 5.0e-3;

#[derive(Debug, Clone, Copy)]
struct FactoredFft2Plan {
    image_rows: usize,
    image_cols: usize,
    tile_rows: usize,
    tile_cols: usize,
    tiles_per_row: usize,
    tiles_per_col: usize,
    tile_count: usize,
    local_row_stages: usize,
    local_col_stages: usize,
    global_row_stages: usize,
    global_col_stages: usize,
}

impl FactoredFft2Plan {
    fn new(image_rows: usize, image_cols: usize, tile_rows: usize, tile_cols: usize) -> Self {
        assert!(
            image_rows > 0 && image_rows.is_power_of_two(),
            "factored FFT2 image row count must be a positive power of two"
        );
        assert!(
            image_cols > 0 && image_cols.is_power_of_two(),
            "factored FFT2 image column count must be a positive power of two"
        );
        assert!(
            tile_rows > 0 && tile_rows.is_power_of_two(),
            "factored FFT2 tile row count must be a positive power of two"
        );
        assert!(
            tile_cols > 0 && tile_cols.is_power_of_two(),
            "factored FFT2 tile column count must be a positive power of two"
        );
        assert!(
            image_rows % tile_rows == 0,
            "factored FFT2 image rows must be divisible by tile rows"
        );
        assert!(
            image_cols % tile_cols == 0,
            "factored FFT2 image columns must be divisible by tile columns"
        );

        let tiles_per_row = image_cols / tile_cols;
        let tiles_per_col = image_rows / tile_rows;

        assert!(
            tiles_per_row.is_power_of_two(),
            "factored FFT2 tiles per row must be a power of two"
        );
        assert!(
            tiles_per_col.is_power_of_two(),
            "factored FFT2 tiles per column must be a power of two"
        );

        Self {
            image_rows,
            image_cols,
            tile_rows,
            tile_cols,
            tiles_per_row,
            tiles_per_col,
            tile_count: tiles_per_row
                .checked_mul(tiles_per_col)
                .expect("factored FFT2 tile count overflow"),
            local_row_stages: tile_cols.ilog2() as usize,
            local_col_stages: tile_rows.ilog2() as usize,
            global_row_stages: tiles_per_row.ilog2() as usize,
            global_col_stages: tiles_per_col.ilog2() as usize,
        }
    }

    fn image_shape(self) -> Fft2Shape {
        Fft2Shape::new(self.image_rows, self.image_cols)
    }

    fn tile_shape(self) -> Fft2Shape {
        Fft2Shape::new(self.tile_rows, self.tile_cols)
    }

    fn local_stages(self) -> usize {
        self.local_row_stages + self.local_col_stages
    }

    fn global_stages(self) -> usize {
        self.global_row_stages + self.global_col_stages
    }

    fn total_stages(self) -> usize {
        self.local_stages() + self.global_stages()
    }

    fn tile_elements(self) -> usize {
        self.tile_rows
            .checked_mul(self.tile_cols)
            .expect("factored FFT2 tile element count overflow")
    }

    fn image_elements(self) -> usize {
        self.image_rows
            .checked_mul(self.image_cols)
            .expect("factored FFT2 image element count overflow")
    }
}

fn phase_start(index: usize, total: usize, name: &str) -> Instant {
    println!("[{index}/{total}] {name} ... START");
    io::stdout()
        .flush()
        .expect("frequency-filter progress output must flush");
    Instant::now()
}

fn phase_done(index: usize, total: usize, name: &str, start: Instant) -> u128 {
    let elapsed_ms = start.elapsed().as_millis();
    println!("[{index}/{total}] {name} ... DONE elapsed_ms={elapsed_ms}");
    elapsed_ms
}

fn block_sizes(limb_count: usize) -> Vec<usize> {
    vec![1; limb_count]
}

fn bit_reverse(index: usize, length: usize) -> usize {
    if length <= 2 {
        return index;
    }

    let bits = length.trailing_zeros();
    index.reverse_bits() >> (usize::BITS - bits)
}

fn packed_physical_to_logical(physical: &[Complex64], shape: Fft2Shape) -> Vec<Complex64> {
    assert!(physical.len() >= shape.elements());

    let mut logical = vec![Complex64::new(0.0, 0.0); shape.elements()];

    for row in 0..shape.rows() {
        for col in 0..shape.cols() {
            let logical_index = row * shape.cols() + col;

            let physical_row = bit_reverse(row, shape.rows());
            let physical_col = bit_reverse(col, shape.cols());
            let physical_index = physical_row * shape.cols() + physical_col;

            logical[logical_index] = physical[physical_index];
        }
    }

    logical
}

fn decrypt_slots(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let plan = RnsNttPlan::new(
        ciphertext.basis().moduli().to_vec(),
        ciphertext.rlwe().degree(),
    );

    let plaintext = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);

    let modulus = composite_modulus_big(plaintext.basis());

    let coefficients: Vec<f64> = reconstruct_coefficients_big(&plaintext)
        .into_iter()
        .map(|value| {
            centered_representative_big(&value, &modulus)
                .to_f64()
                .expect("centered CKKS coefficient must be representable as f64")
                / ciphertext.scale()
        })
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

fn error_metrics(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());

    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (actual, expected) in actual.iter().zip(expected) {
        let error = *actual - *expected;

        squared_error += error.norm_sqr();
        squared_reference += expected.norm_sqr();
        max_abs = max_abs.max(error.norm());
    }

    let rel_l2 = if squared_reference > 0.0 {
        (squared_error / squared_reference).sqrt()
    } else {
        squared_error.sqrt()
    };

    (rel_l2, max_abs)
}

fn load_grayscale_square(path: &Path, dimension: usize) -> Vec<Complex64> {
    let image = image::open(path)
        .unwrap_or_else(|error| panic!("failed to open image {}: {error}", path.display()))
        .to_luma8();

    /*
     * Use a deterministic high-quality resize into the encrypted square
     * workload. The original image may be arbitrarily high resolution.
     */
    let resized = image::imageops::resize(
        &image,
        dimension as u32,
        dimension as u32,
        image::imageops::FilterType::Lanczos3,
    );

    resized
        .pixels()
        .map(|pixel| Complex64::new(f64::from(pixel[0]) / 255.0, 0.0))
        .collect()
}

fn extract_square_tiles(
    values: &[Complex64],
    image_shape: Fft2Shape,
    tile_shape: Fft2Shape,
) -> Vec<Vec<Complex64>> {
    assert_eq!(values.len(), image_shape.elements());
    assert_eq!(image_shape.rows() % tile_shape.rows(), 0);
    assert_eq!(image_shape.cols() % tile_shape.cols(), 0);

    let tile_rows = image_shape.rows() / tile_shape.rows();
    let tile_cols = image_shape.cols() / tile_shape.cols();
    let mut tiles = Vec::with_capacity(tile_rows * tile_cols);

    for tile_row in 0..tile_rows {
        for tile_col in 0..tile_cols {
            let mut tile = vec![Complex64::new(0.0, 0.0); tile_shape.elements()];

            for row in 0..tile_shape.rows() {
                for col in 0..tile_shape.cols() {
                    let image_row = tile_row * tile_shape.rows() + row;
                    let image_col = tile_col * tile_shape.cols() + col;
                    let image_index = image_row * image_shape.cols() + image_col;
                    let tile_index = row * tile_shape.cols() + col;

                    tile[tile_index] = values[image_index];
                }
            }

            tiles.push(tile);
        }
    }

    tiles
}

fn save_tiled_spectrum_image(
    path: &Path,
    tiles: &[Vec<Complex64>],
    image_shape: Fft2Shape,
    tile_shape: Fft2Shape,
) {
    let tile_rows = image_shape.rows() / tile_shape.rows();
    let tile_cols = image_shape.cols() / tile_shape.cols();

    assert_eq!(tiles.len(), tile_rows * tile_cols);

    let shifted: Vec<Vec<Complex64>> = tiles
        .iter()
        .map(|tile| fft_shift(tile, tile_shape))
        .collect();

    let maximum = shifted
        .iter()
        .flat_map(|tile| tile.iter())
        .map(|value| (1.0 + value.norm()).ln())
        .fold(0.0_f64, f64::max);

    let mut image = GrayImage::new(image_shape.cols() as u32, image_shape.rows() as u32);

    for tile_row in 0..tile_rows {
        for tile_col in 0..tile_cols {
            let tile_index = tile_row * tile_cols + tile_col;
            let tile = &shifted[tile_index];

            for row in 0..tile_shape.rows() {
                for col in 0..tile_shape.cols() {
                    let index = row * tile_shape.cols() + col;
                    let magnitude = (1.0 + tile[index].norm()).ln();

                    let normalized = if maximum > 0.0 {
                        magnitude / maximum
                    } else {
                        0.0
                    };

                    let image_row = tile_row * tile_shape.rows() + row;
                    let image_col = tile_col * tile_shape.cols() + col;

                    image.put_pixel(
                        image_col as u32,
                        image_row as u32,
                        Luma([(normalized.clamp(0.0, 1.0) * 255.0).round() as u8]),
                    );
                }
            }
        }
    }

    image
        .save(path)
        .unwrap_or_else(|error| panic!("failed to save {}: {error}", path.display()));
}

fn save_spatial_image(path: &Path, values: &[Complex64], shape: Fft2Shape) {
    assert_eq!(values.len(), shape.elements());

    let mut image = GrayImage::new(shape.cols() as u32, shape.rows() as u32);

    for row in 0..shape.rows() {
        for col in 0..shape.cols() {
            let index = row * shape.cols() + col;
            let value = values[index].re.clamp(0.0, 1.0);
            let pixel = (value * 255.0).round() as u8;

            image.put_pixel(col as u32, row as u32, Luma([pixel]));
        }
    }

    image
        .save(path)
        .unwrap_or_else(|error| panic!("failed to save {}: {error}", path.display()));
}

fn fft_shift(values: &[Complex64], shape: Fft2Shape) -> Vec<Complex64> {
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

fn main() {
    let input_path = env::args().nth(1).map(PathBuf::from).expect(
        "usage: cargo run --release --bin ckks_encrypted_image_fft -- <input-image> [dimension]",
    );

    let dimension = env::args()
        .nth(2)
        .map(|value| {
            value
                .parse::<usize>()
                .expect("image dimension must be a positive power of two")
        })
        .unwrap_or(128);

    let tile_dimension = env::args()
        .nth(3)
        .map(|value| {
            value
                .parse::<usize>()
                .expect("tile dimension must be a positive power of two")
        })
        .unwrap_or(64);

    let fft_plan = FactoredFft2Plan::new(dimension, dimension, tile_dimension, tile_dimension);

    let image_shape = fft_plan.image_shape();
    let shape = fft_plan.tile_shape();

    let tiles_per_dimension = fft_plan.tiles_per_row;
    assert_eq!(
        fft_plan.tiles_per_row, fft_plan.tiles_per_col,
        "current image harness requires a square tile grid"
    );

    let tile_count = fft_plan.tile_count;

    /*
     * The current local-tile executor consumes only the stages internal to
     * one tile. Exact global composition additionally consumes the cross-tile
     * stages represented by fft_plan.global_stages().
     */
    let local_required_levels = fft_plan.local_stages();
    let required_levels = fft_plan.total_stages();
    let total_phases = required_levels + 3;

    let local_profile = if local_required_levels <= 8 {
        research_profile_16384()
    } else if local_required_levels <= 12 {
        research_profile_32768()
    } else if local_required_levels <= 16 {
        research_profile_65536()
    } else {
        panic!(
            "no current image-FFT research profile supports {local_required_levels} local levels"
        );
    };

    let profile = if required_levels <= 8 {
        research_profile_16384()
    } else if required_levels <= 12 {
        research_profile_32768()
    } else if required_levels <= 16 {
        research_profile_65536()
    } else {
        panic!(
            "no current image-FFT research profile supports {required_levels} exact levels for {dimension}x{dimension}"
        );
    };

    /*
     * Exact execution uses the profile selected for the complete dependency
     * path. The smaller local profile remains useful for architecture/cost
     * reporting and for future depth-reduced execution.
     */
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    let top_basis = chain.top().clone();

    let embedding = CkksCanonicalEmbedding::new(degree);
    let slot_count = embedding.slot_count();
    let top_plan = profile.rns_ntt_plan();

    let local_degree = local_profile.degree();
    let local_slot_count = local_degree / 2;

    assert!(
        shape.elements() <= slot_count,
        "FFT tile requires {} slots but exact profile {} provides only {}",
        shape.elements(),
        profile.name(),
        slot_count
    );
    assert!(
        shape.elements() <= local_slot_count,
        "FFT tile requires {} slots but local profile {} provides only {}",
        shape.elements(),
        local_profile.name(),
        local_slot_count
    );
    assert!(
        required_levels <= chain.max_level(),
        "exact FFT requires {required_levels} levels but profile {} supports only {}",
        profile.name(),
        chain.max_level()
    );

    assert!(!local_profile.security_bearing());
    assert!(!profile.security_bearing());
    println!("FFT_PARAMETERIZATION_BEGIN");
    println!("FFT_PROFILE={}", profile.name());
    println!("FFT_SECURITY_BEARING={}", profile.security_bearing());
    println!("FFT_RING_DEGREE={degree}");
    println!("FFT_SLOT_COUNT={slot_count}");
    println!("FFT_ROWS={}", shape.rows());
    println!("FFT_COLS={}", shape.cols());
    println!("FFT_ELEMENTS={}", shape.elements());
    println!("FFT_CHAIN_LIMBS={}", profile.modulus_values().len());
    println!("FFT_TOTAL_MODULUS_BITS={}", profile.total_modulus_bits());
    println!("FFT_INITIAL_SCALE={scale:.12e}");
    println!("FFT_ENCRYPTION_SIGMA={SIGMA:.6}");
    println!("FFT_GALOIS_KEY_NOISE_BOUND=0");
    println!("FFT_PLAINTEXT_MODULUS=2");
    println!("FFT_GADGET_BLOCK_POLICY=singleton");
    println!("FFT_DIRECTION=forward");
    println!("FFT_EXPECTED_LEVELS_CONSUMED={required_levels}");
    println!("FFT_TRANSPOSES=0");

    for (index, modulus) in profile.modulus_values().iter().enumerate() {
        println!("FFT_MODULUS_{index}={modulus}");
    }

    println!("FFT_PARAMETERIZATION_END");
    io::stdout()
        .flush()
        .expect("FFT parameterization output must flush");

    let output_dir = PathBuf::from(format!(
        "results/image_filter/encrypted_frequency_filter/tiled_fft_{dimension}x{dimension}_tile_{tile_dimension}x{tile_dimension}"
    ));

    fs::create_dir_all(&output_dir)
        .expect("failed to create tiled encrypted image FFT result directory");

    println!("TILED_FFT_PARAMETERIZATION_BEGIN");
    println!("TILED_FFT_IMAGE_ROWS={}", image_shape.rows());
    println!("TILED_FFT_IMAGE_COLS={}", image_shape.cols());
    println!("TILED_FFT_TILE_ROWS={}", shape.rows());
    println!("TILED_FFT_TILE_COLS={}", shape.cols());
    println!("TILED_FFT_TILES_PER_DIMENSION={tiles_per_dimension}");
    println!("TILED_FFT_TILE_COUNT={tile_count}");
    println!("TILED_FFT_PROFILE={}", profile.name());
    println!("TILED_FFT_RING_DEGREE={degree}");
    println!("TILED_FFT_SLOT_COUNT={slot_count}");
    println!("TILED_FFT_LEVELS_PER_TILE={required_levels}");
    println!("TILED_FFT_SECURITY_BEARING={}", profile.security_bearing());
    println!("TILED_FFT_SEMANTICS=LOCAL_TILE_SPECTRA");
    println!("TILED_FFT_GLOBAL_EXACT=false");
    println!("TILED_FFT_PARAMETERIZATION_END");

    println!("FHE_FFT_PLAN_BEGIN");
    println!(
        "FHE_FFT_PLAN_IMAGE_DIMENSION={}x{}",
        fft_plan.image_rows, fft_plan.image_cols
    );
    println!(
        "FHE_FFT_PLAN_TILE_DIMENSION={}x{}",
        fft_plan.tile_rows, fft_plan.tile_cols
    );
    println!("FHE_FFT_PLAN_IMAGE_ELEMENTS={}", fft_plan.image_elements());
    println!("FHE_FFT_PLAN_TILE_ELEMENTS={}", fft_plan.tile_elements());
    println!("FHE_FFT_PLAN_TILES_PER_ROW={}", fft_plan.tiles_per_row);
    println!("FHE_FFT_PLAN_TILES_PER_COL={}", fft_plan.tiles_per_col);
    println!("FHE_FFT_PLAN_TILE_COUNT={}", fft_plan.tile_count);
    println!(
        "FHE_FFT_PLAN_LOCAL_ROW_STAGES={}",
        fft_plan.local_row_stages
    );
    println!(
        "FHE_FFT_PLAN_GLOBAL_ROW_STAGES={}",
        fft_plan.global_row_stages
    );
    println!(
        "FHE_FFT_PLAN_LOCAL_COLUMN_STAGES={}",
        fft_plan.local_col_stages
    );
    println!(
        "FHE_FFT_PLAN_GLOBAL_COLUMN_STAGES={}",
        fft_plan.global_col_stages
    );
    println!("FHE_FFT_PLAN_LOCAL_STAGES={}", fft_plan.local_stages());
    println!(
        "FHE_FFT_PLAN_CROSS_TILE_STAGES={}",
        fft_plan.global_stages()
    );
    println!(
        "FHE_FFT_PLAN_TOTAL_EXACT_STAGES={}",
        fft_plan.total_stages()
    );
    println!(
        "FHE_FFT_PLAN_SELECTED_LOCAL_PROFILE={}",
        local_profile.name()
    );
    println!("FHE_FFT_PLAN_EXACT_EXECUTION_PROFILE={}", profile.name());
    println!(
        "FHE_FFT_PLAN_EXACT_EXECUTION_REQUIRED_LEVELS={}",
        fft_plan.total_stages()
    );
    println!("FHE_FFT_PLAN_LOCAL_RING_DEGREE={local_degree}");
    println!("FHE_FFT_PLAN_LOCAL_SLOT_COUNT={local_slot_count}");
    println!(
        "FHE_FFT_PLAN_TILE_SLOT_UTILIZATION={:.6}",
        fft_plan.tile_elements() as f64 / local_slot_count as f64
    );
    println!("FHE_FFT_PLAN_GLOBAL_EXACT_TARGET=true");
    println!("FHE_FFT_PLAN_CURRENT_EXECUTOR_GLOBAL_EXACT=false");
    println!(
        "FHE_FFT_PLAN_SECURITY_BEARING={}",
        profile.security_bearing()
    );
    println!("FHE_FFT_PLAN_END");

    println!("IMAGE_FFT_INPUT={}", input_path.display());
    println!("IMAGE_FFT_OUTPUT_DIR={}", output_dir.display());

    let logical_image = load_grayscale_square(&input_path, dimension);

    let input_image_name = format!("01_input_{dimension}x{dimension}.png");
    save_spatial_image(
        &output_dir.join(&input_image_name),
        &logical_image,
        image_shape,
    );

    let logical_tiles = extract_square_tiles(&logical_image, image_shape, shape);

    assert_eq!(logical_tiles.len(), tile_count);

    let clear_spectra: Vec<Vec<Complex64>> = logical_tiles
        .iter()
        .map(|tile| fft2_pp(shape, FftDirection::Forward, tile))
        .collect();

    save_tiled_spectrum_image(
        &output_dir.join("02_clear_local_fft_log_magnitude.png"),
        &clear_spectra,
        image_shape,
        shape,
    );

    let plaintexts: Vec<_> = logical_tiles
        .iter()
        .map(|tile| {
            let mut physical_input = vec![Complex64::new(0.0, 0.0); slot_count];
            physical_input[..shape.elements()].copy_from_slice(tile);
            encode_rns(&physical_input, &embedding, &top_basis, scale)
        })
        .collect();

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4652_4551_4649_4c01);

    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let total_start = Instant::now();

    let encryption_start = phase_start(1, total_phases, "tile encryption");

    let ciphertexts: Vec<RnsCkksCiphertext> = plaintexts
        .iter()
        .enumerate()
        .map(|(tile_index, plaintext)| {
            let mut encryption_rng =
                ChaCha20Rng::seed_from_u64(0x4652_4551_4649_4c02 ^ tile_index as u64);

            let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                plaintext,
                2,
                ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                &secret,
                &top_plan,
                &mut encryption_rng,
            );

            RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain)
        })
        .collect();

    let timing_encryption_ms = phase_done(1, total_phases, "tile encryption", encryption_start);

    assert_eq!(ciphertexts.len(), tile_count);

    /*
     * Exact factored DIF schedule:
     *
     *   global row stages   : cross-ciphertext, no rotations
     *   local row stages    : packed intra-ciphertext rotations
     *   global column stages: cross-ciphertext, no rotations
     *   local column stages : packed intra-ciphertext rotations
     *
     * Therefore Galois keys are generated only for local stages.
     */
    let mut level_rotations = Vec::with_capacity(fft_plan.local_stages());

    let mut level = fft_plan.global_row_stages;

    let mut rotation = shape.cols() / 2;
    while rotation > 0 {
        level_rotations.push((level, rotation));
        level += 1;
        rotation /= 2;
    }

    assert_eq!(
        level,
        fft_plan.global_row_stages + fft_plan.local_row_stages
    );

    /*
     * Global column stages occupy the next levels and require no Galois keys.
     */
    level += fft_plan.global_col_stages;

    let mut row_rotation = shape.rows() / 2;
    while row_rotation > 0 {
        level_rotations.push((level, row_rotation * shape.cols()));
        level += 1;
        row_rotation /= 2;
    }

    assert_eq!(level, fft_plan.total_stages());
    assert_eq!(level_rotations.len(), fft_plan.local_stages());

    println!("FHE_FFT_ROTATION_KEY_SCHEDULE_BEGIN");
    for &(key_level, key_rotation) in &level_rotations {
        println!("FHE_FFT_ROTATION_KEY level={key_level} rotation={key_rotation}");
    }
    println!("FHE_FFT_ROTATION_KEY_SCHEDULE_END");

    let mut evaluation_keys = RnsCkksEvaluationKeys::new();
    let mut timing_galois_ms = vec![0_u128; level_rotations.len()];

    for (phase_offset, &(level, rotation)) in level_rotations.iter().enumerate() {
        let phase_index = phase_offset + 2;
        let phase_name = format!("level-{level} Galois keygen");
        let keygen_start = phase_start(phase_index, total_phases, &phase_name);

        let level_basis = chain.level(level).clone();
        let level_layout =
            RnsGadgetLayout::new(level_basis.clone(), block_sizes(level_basis.len()));
        let level_plan = RnsNttPlan::new(level_basis.moduli().to_vec(), degree);

        println!(
            "FFT_GALOIS_LEVEL level={level} rotation={rotation} limbs={} degree={degree}",
            level_layout.full_basis().len()
        );
        io::stdout()
            .flush()
            .expect("Galois-level parameterization output must flush");

        let mut level_keys = RnsCkksLevelKeys::new(level, level_basis);

        let left_exponent = rotation_exponent_left(degree, rotation);
        let right_exponent = rotation_exponent_right(degree, rotation);

        println!(
            "      level={level} rotation={rotation} direction=left exponent={left_exponent} ... START"
        );
        println!(
            "      level={level} rotation={rotation} direction=right exponent={right_exponent} ... START"
        );
        io::stdout()
            .flush()
            .expect("Galois-key progress output must flush");

        let ((left_key, left_ms), (right_key, right_ms)) = std::thread::scope(|scope| {
            let left_handle = scope.spawn(|| {
                let start = Instant::now();
                let mut rng =
                    ChaCha20Rng::seed_from_u64(0x4652_4551_4649_4d00 ^ ((level as u64) << 8));

                let key = RnsGaloisKey::generate_with_ntt_rng(
                    RnsKeygenConfig {
                        degree,
                        plaintext_modulus: 2,
                        noise_bound: 0,
                        layout: level_layout.clone(),
                        plan: &level_plan,
                    },
                    &secret,
                    left_exponent,
                    &mut rng,
                );

                (key, start.elapsed().as_millis())
            });

            let right_handle = scope.spawn(|| {
                let start = Instant::now();
                let mut rng = ChaCha20Rng::seed_from_u64(
                    0x4652_4551_4649_4d00 ^ ((level as u64) << 8) ^ 1_u64,
                );

                let key = RnsGaloisKey::generate_with_ntt_rng(
                    RnsKeygenConfig {
                        degree,
                        plaintext_modulus: 2,
                        noise_bound: 0,
                        layout: level_layout.clone(),
                        plan: &level_plan,
                    },
                    &secret,
                    right_exponent,
                    &mut rng,
                );

                (key, start.elapsed().as_millis())
            });

            (
                left_handle
                    .join()
                    .expect("left Galois-key generation worker panicked"),
                right_handle
                    .join()
                    .expect("right Galois-key generation worker panicked"),
            )
        });

        println!(
            "      level={level} rotation={rotation} direction=left exponent={left_exponent} ... DONE elapsed_ms={left_ms}"
        );
        println!(
            "      level={level} rotation={rotation} direction=right exponent={right_exponent} ... DONE elapsed_ms={right_ms}"
        );

        level_keys.insert_galois_key(left_key);
        level_keys.insert_galois_key(right_key);

        evaluation_keys.insert_level(level_keys);

        timing_galois_ms[phase_offset] =
            phase_done(phase_index, total_phases, &phase_name, keygen_start);
    }

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    /*
     * Exact factored encrypted FFT2.
     *
     * Schedule:
     *
     *   global row DIF stages
     *   local row DIF stages
     *   global column DIF stages
     *   local column DIF stages
     *
     * Global stages communicate across ciphertext tiles using
     * packed_fft_dif_butterfly_cp. Local stages use the existing packed
     * intra-ciphertext FFT2 stage primitive.
     *
     * No encrypted data are decrypted between stages.
     */
    let fft_phase = level_rotations.len() + 2;
    let forward_fft2_start = phase_start(fft_phase, total_phases, "exact factored forward FFT2");

    let mut values = ciphertexts;
    let mut execution_level = 0usize;

    /*
     * ---------------------------------------------------------------
     * Global row stages.
     * ---------------------------------------------------------------
     */
    let mut span_tiles = fft_plan.tiles_per_row;

    while span_tiles >= 2 {
        let half_tiles = span_tiles / 2;
        let global_span = span_tiles
            .checked_mul(fft_plan.tile_cols)
            .expect("global row FFT span overflow");

        let stage_plan = RnsNttPlan::new(chain.level(execution_level).moduli().to_vec(), degree);

        let stage_input = values;
        let mut stage_output: Vec<Option<RnsCkksCiphertext>> =
            (0..tile_count).map(|_| None).collect();

        for tile_row in 0..fft_plan.tiles_per_col {
            for group_start in (0..fft_plan.tiles_per_row).step_by(span_tiles) {
                for tile_offset in 0..half_tiles {
                    let upper_col = group_start + tile_offset;
                    let lower_col = upper_col + half_tiles;

                    let upper_index = tile_row * fft_plan.tiles_per_row + upper_col;
                    let lower_index = tile_row * fft_plan.tiles_per_row + lower_col;

                    let mut twiddles = vec![Complex64::new(0.0, 0.0); slot_count];

                    for local_row in 0..fft_plan.tile_rows {
                        for local_col in 0..fft_plan.tile_cols {
                            let slot = local_row * fft_plan.tile_cols + local_col;

                            let twiddle_index = tile_offset * fft_plan.tile_cols + local_col;

                            let angle = -2.0 * std::f64::consts::PI * twiddle_index as f64
                                / global_span as f64;

                            twiddles[slot] = Complex64::new(angle.cos(), angle.sin());
                        }
                    }

                    let (upper, lower) = packed_fft_dif_butterfly_cp(
                        &evaluator,
                        &stage_input[upper_index],
                        &stage_input[lower_index],
                        &twiddles,
                        &embedding,
                        &chain,
                        &stage_plan,
                    );

                    stage_output[upper_index] = Some(upper);
                    stage_output[lower_index] = Some(lower);
                }
            }
        }

        values = stage_output
            .into_iter()
            .map(|value| value.expect("global row FFT tile was not produced"))
            .collect();

        execution_level += 1;
        span_tiles /= 2;
    }

    assert_eq!(execution_level, fft_plan.global_row_stages);

    /*
     * ---------------------------------------------------------------
     * Local row stages.
     * ---------------------------------------------------------------
     */
    let mut span = fft_plan.tile_cols;

    while span >= 2 {
        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Rows,
            span,
            FftDirection::Forward,
        );

        let stage_plan = RnsNttPlan::new(chain.level(execution_level).moduli().to_vec(), degree);

        values = values
            .iter()
            .map(|value| {
                execute_packed_fft2_dif_stage_cp(
                    value,
                    &diagonals,
                    &evaluator,
                    &embedding,
                    &chain,
                    &stage_plan,
                )
            })
            .collect();

        execution_level += 1;
        span /= 2;
    }

    assert_eq!(
        execution_level,
        fft_plan.global_row_stages + fft_plan.local_row_stages
    );

    /*
     * ---------------------------------------------------------------
     * Global column stages.
     * ---------------------------------------------------------------
     */
    span_tiles = fft_plan.tiles_per_col;

    while span_tiles >= 2 {
        let half_tiles = span_tiles / 2;
        let global_span = span_tiles
            .checked_mul(fft_plan.tile_rows)
            .expect("global column FFT span overflow");

        let stage_plan = RnsNttPlan::new(chain.level(execution_level).moduli().to_vec(), degree);

        let stage_input = values;
        let mut stage_output: Vec<Option<RnsCkksCiphertext>> =
            (0..tile_count).map(|_| None).collect();

        for tile_col in 0..fft_plan.tiles_per_row {
            for group_start in (0..fft_plan.tiles_per_col).step_by(span_tiles) {
                for tile_offset in 0..half_tiles {
                    let upper_row = group_start + tile_offset;
                    let lower_row = upper_row + half_tiles;

                    let upper_index = upper_row * fft_plan.tiles_per_row + tile_col;
                    let lower_index = lower_row * fft_plan.tiles_per_row + tile_col;

                    let mut twiddles = vec![Complex64::new(0.0, 0.0); slot_count];

                    for local_row in 0..fft_plan.tile_rows {
                        let twiddle_index = tile_offset * fft_plan.tile_rows + local_row;

                        let angle =
                            -2.0 * std::f64::consts::PI * twiddle_index as f64 / global_span as f64;

                        let twiddle = Complex64::new(angle.cos(), angle.sin());

                        for local_col in 0..fft_plan.tile_cols {
                            let slot = local_row * fft_plan.tile_cols + local_col;

                            twiddles[slot] = twiddle;
                        }
                    }

                    let (upper, lower) = packed_fft_dif_butterfly_cp(
                        &evaluator,
                        &stage_input[upper_index],
                        &stage_input[lower_index],
                        &twiddles,
                        &embedding,
                        &chain,
                        &stage_plan,
                    );

                    stage_output[upper_index] = Some(upper);
                    stage_output[lower_index] = Some(lower);
                }
            }
        }

        values = stage_output
            .into_iter()
            .map(|value| value.expect("global column FFT tile was not produced"))
            .collect();

        execution_level += 1;
        span_tiles /= 2;
    }

    assert_eq!(
        execution_level,
        fft_plan.global_row_stages + fft_plan.local_row_stages + fft_plan.global_col_stages
    );

    /*
     * ---------------------------------------------------------------
     * Local column stages.
     * ---------------------------------------------------------------
     */
    span = fft_plan.tile_rows;

    while span >= 2 {
        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Columns,
            span,
            FftDirection::Forward,
        );

        let stage_plan = RnsNttPlan::new(chain.level(execution_level).moduli().to_vec(), degree);

        values = values
            .iter()
            .map(|value| {
                execute_packed_fft2_dif_stage_cp(
                    value,
                    &diagonals,
                    &evaluator,
                    &embedding,
                    &chain,
                    &stage_plan,
                )
            })
            .collect();

        execution_level += 1;
        span /= 2;
    }

    assert_eq!(execution_level, fft_plan.total_stages());
    assert_eq!(execution_level, required_levels);

    for value in &values {
        assert_eq!(value.level(), required_levels);
    }

    let timing_forward_fft2_ms = phase_done(
        fft_phase,
        total_phases,
        "exact factored forward FFT2",
        forward_fft2_start,
    );

    /*
     * Decrypt only after the complete global encrypted FFT.
     */
    let decrypt_phase = fft_phase + 1;
    let decrypt_decode_start = phase_start(decrypt_phase, total_phases, "global decrypt/decode");

    let mut global_physical = vec![Complex64::new(0.0, 0.0); fft_plan.image_elements()];

    let mut inactive_max_abs = 0.0_f64;

    for tile_row in 0..fft_plan.tiles_per_col {
        for tile_col in 0..fft_plan.tiles_per_row {
            let tile_index = tile_row * fft_plan.tiles_per_row + tile_col;

            let decoded = decrypt_slots(&values[tile_index], &secret, &embedding);

            inactive_max_abs = inactive_max_abs.max(
                decoded[shape.elements()..]
                    .iter()
                    .map(|value| value.norm())
                    .fold(0.0_f64, f64::max),
            );

            for local_row in 0..fft_plan.tile_rows {
                for local_col in 0..fft_plan.tile_cols {
                    let tile_slot = local_row * fft_plan.tile_cols + local_col;

                    let global_row = tile_row * fft_plan.tile_rows + local_row;
                    let global_col = tile_col * fft_plan.tile_cols + local_col;

                    let global_index = global_row * fft_plan.image_cols + global_col;

                    global_physical[global_index] = decoded[tile_slot];
                }
            }
        }
    }

    /*
     * The factored DIF representation has the same global physical
     * bit-reversed row/column ordering as the monolithic packed FFT2.
     */
    let decoded_global = packed_physical_to_logical(&global_physical, image_shape);

    let timing_decrypt_decode_ms = phase_done(
        decrypt_phase,
        total_phases,
        "global decrypt/decode",
        decrypt_decode_start,
    );

    let timing_total_ms = total_start.elapsed().as_millis();

    /*
     * Independent clear global oracle.
     */
    let clear_global = fft2_pp(image_shape, FftDirection::Forward, &logical_image);

    let (rel_l2, max_abs) = error_metrics(&decoded_global, &clear_global);

    let status = if rel_l2 <= TOLERANCE && max_abs <= TOLERANCE && inactive_max_abs <= TOLERANCE {
        "PASS"
    } else {
        "FAIL"
    };

    println!("FFT_TIMING_ENCRYPTION_MS={timing_encryption_ms}");

    for (index, elapsed_ms) in timing_galois_ms.iter().enumerate() {
        println!("FFT_TIMING_GALOIS_KEY_{index}_MS={elapsed_ms}");
    }

    println!("FFT_TIMING_FORWARD_FFT2_MS={timing_forward_fft2_ms}");
    println!("FFT_TIMING_DECRYPT_DECODE_MS={timing_decrypt_decode_ms}");
    println!("FFT_TIMING_TOTAL_MS={timing_total_ms}");

    println!("FACTORED_FFT_RESULT_BEGIN");
    println!(
        "FACTORED_FFT_IMAGE_DIMENSION={}x{}",
        fft_plan.image_rows, fft_plan.image_cols
    );
    println!(
        "FACTORED_FFT_TILE_DIMENSION={}x{}",
        fft_plan.tile_rows, fft_plan.tile_cols
    );
    println!("FACTORED_FFT_TILE_COUNT={}", fft_plan.tile_count);
    println!("FACTORED_FFT_LOCAL_PROFILE={}", local_profile.name());
    println!("FACTORED_FFT_EXACT_PROFILE={}", profile.name());
    println!("FACTORED_FFT_LOCAL_STAGES={}", fft_plan.local_stages());
    println!(
        "FACTORED_FFT_CROSS_TILE_STAGES={}",
        fft_plan.global_stages()
    );
    println!("FACTORED_FFT_TOTAL_LEVELS={execution_level}");
    println!("FACTORED_FFT_GLOBAL_EXACT=true");
    println!("FACTORED_FFT_REL_L2={rel_l2:.12e}");
    println!("FACTORED_FFT_MAX_ABS={max_abs:.12e}");
    println!("FACTORED_FFT_INACTIVE_MAX_ABS={inactive_max_abs:.12e}");
    println!("FACTORED_FFT_TOLERANCE={TOLERANCE:.12e}");
    println!("FACTORED_FFT_RESULT_STATUS={status}");
    println!("FACTORED_FFT_RESULT_END");

    assert_eq!(status, "PASS");
}
