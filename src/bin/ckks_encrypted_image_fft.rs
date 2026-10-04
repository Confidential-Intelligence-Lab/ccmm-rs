use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_16384, research_profile_32768, research_profile_65536, rotation_exponent_left,
    rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsCkksLevelKeys, RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{execute_packed_fft2_dif_cp, fft2_pp, Fft2Shape, FftDirection};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout,
    RnsKeygenConfig,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use image::{GrayImage, ImageBuffer, Luma};
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

fn save_spectrum_image(path: &Path, spectrum: &[Complex64], shape: Fft2Shape) {
    assert_eq!(spectrum.len(), shape.elements());

    let shifted = fft_shift(spectrum, shape);

    let log_magnitude: Vec<f64> = shifted
        .iter()
        .map(|value| (1.0 + value.norm()).ln())
        .collect();

    let maximum = log_magnitude.iter().copied().fold(0.0_f64, f64::max);

    let mut image: GrayImage = ImageBuffer::new(shape.cols() as u32, shape.rows() as u32);

    for row in 0..shape.rows() {
        for col in 0..shape.cols() {
            let index = row * shape.cols() + col;

            let normalized = if maximum > 0.0 {
                log_magnitude[index] / maximum
            } else {
                0.0
            };

            image.put_pixel(
                col as u32,
                row as u32,
                Luma([(normalized.clamp(0.0, 1.0) * 255.0).round() as u8]),
            );
        }
    }

    image
        .save(path)
        .unwrap_or_else(|error| panic!("failed to save {}: {error}", path.display()));
}

fn save_spectrum_error_image(
    path: &Path,
    actual: &[Complex64],
    expected: &[Complex64],
    shape: Fft2Shape,
) {
    assert_eq!(actual.len(), expected.len());

    let errors: Vec<f64> = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .collect();

    let maximum = errors.iter().copied().fold(0.0_f64, f64::max);

    let mut image = GrayImage::new(shape.cols() as u32, shape.rows() as u32);

    for row in 0..shape.rows() {
        for col in 0..shape.cols() {
            let index = row * shape.cols() + col;

            let normalized = if maximum > 0.0 {
                errors[index] / maximum
            } else {
                0.0
            };

            image.put_pixel(
                col as u32,
                row as u32,
                Luma([(normalized.clamp(0.0, 1.0) * 255.0).round() as u8]),
            );
        }
    }

    image
        .save(path)
        .unwrap_or_else(|error| panic!("failed to save {}: {error}", path.display()));
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
                .expect("FFT dimension must be a positive power of two")
        })
        .unwrap_or(16);

    assert!(
        dimension > 0 && dimension.is_power_of_two(),
        "FFT dimension must be a positive power of two"
    );

    let shape = Fft2Shape::new(dimension, dimension);
    let required_levels = shape.stages() as usize;
    let total_phases = required_levels + 3;

    let profile = if required_levels <= 8 {
        research_profile_16384()
    } else if required_levels <= 12 {
        research_profile_32768()
    } else if required_levels <= 16 {
        research_profile_65536()
    } else {
        panic!(
            "no current image-FFT research profile supports {required_levels} levels for {dimension}x{dimension}"
        );
    };

    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    let top_basis = chain.top().clone();

    let embedding = CkksCanonicalEmbedding::new(degree);
    let slot_count = embedding.slot_count();
    let top_plan = profile.rns_ntt_plan();

    assert!(
        shape.elements() <= slot_count,
        "FFT workload requires {} slots but profile {} provides only {}",
        shape.elements(),
        profile.name(),
        slot_count
    );
    assert!(
        required_levels <= chain.max_level(),
        "FFT workload requires {required_levels} levels but profile {} supports only {}",
        profile.name(),
        chain.max_level()
    );
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
    println!(
        "FFT_EXPECTED_LEVELS_CONSUMED={}",
        shape.rows().ilog2() + shape.cols().ilog2()
    );
    println!("FFT_TRANSPOSES=0");

    for (index, modulus) in profile.modulus_values().iter().enumerate() {
        println!("FFT_MODULUS_{index}={modulus}");
    }

    println!("FFT_PARAMETERIZATION_END");
    io::stdout()
        .flush()
        .expect("FFT parameterization output must flush");

    let output_dir = PathBuf::from(format!(
        "results/image_filter/encrypted_frequency_filter/image_fft_{dimension}x{dimension}"
    ));

    fs::create_dir_all(&output_dir).expect("failed to create encrypted image FFT result directory");

    println!("IMAGE_FFT_INPUT={}", input_path.display());
    println!("IMAGE_FFT_OUTPUT_DIR={}", output_dir.display());

    let logical_input = load_grayscale_square(&input_path, dimension);

    let input_image_name = format!("01_input_{dimension}x{dimension}.png");
    save_spatial_image(&output_dir.join(&input_image_name), &logical_input, shape);

    let clear_spectrum = fft2_pp(shape, FftDirection::Forward, &logical_input);

    let mut physical_input = vec![Complex64::new(0.0, 0.0); slot_count];
    physical_input[..shape.elements()].copy_from_slice(&logical_input);

    let plaintext = encode_rns(&physical_input, &embedding, &top_basis, scale);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4652_4551_4649_4c01);

    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let total_start = Instant::now();

    /*
     * Phase 1: fresh encryption at the top of the CKKS chain.
     */
    let encryption_start = phase_start(1, total_phases, "encryption");

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x4652_4551_4649_4c02);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &top_plan,
        &mut encryption_rng,
    );

    let ciphertext = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

    let timing_encryption_ms = phase_done(1, total_phases, "encryption", encryption_start);

    /*
     * Derive the packed DIF FFT2 Galois schedule from the square shape.
     * Row stages rotate within each row; column stages rotate by the
     * corresponding row stride.
     */
    let mut level_rotations = Vec::with_capacity(required_levels);

    let mut level = 0usize;
    let mut rotation = shape.cols() / 2;
    while rotation > 0 {
        level_rotations.push((level, rotation));
        level += 1;
        rotation /= 2;
    }

    let mut row_rotation = shape.rows() / 2;
    while row_rotation > 0 {
        level_rotations.push((level, row_rotation * shape.cols()));
        level += 1;
        row_rotation /= 2;
    }

    assert_eq!(level_rotations.len(), required_levels);

    let mut evaluation_keys = RnsCkksEvaluationKeys::new();
    let mut timing_galois_ms = vec![0_u128; required_levels];

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
     * Phase 4: encrypted forward FFT2.
     */
    let fft_phase = required_levels + 2;
    let forward_fft2_start = phase_start(fft_phase, total_phases, "forward FFT2");

    let encrypted_spectrum = execute_packed_fft2_dif_cp(
        shape,
        FftDirection::Forward,
        &ciphertext,
        &evaluator,
        &embedding,
        &chain,
    );

    let timing_forward_fft2_ms =
        phase_done(fft_phase, total_phases, "forward FFT2", forward_fft2_start);

    assert_eq!(encrypted_spectrum.level(), required_levels);

    /*
     * Phase 5: decrypt/decode the encrypted frequency-domain result.
     *
     * The packed DIF representation is converted explicitly back to logical
     * row-major frequency-bin order before comparison with fft2_pp.
     */
    let decrypt_phase = required_levels + 3;
    let decrypt_decode_start = phase_start(decrypt_phase, total_phases, "decrypt/decode");

    let decoded_physical = decrypt_slots(&encrypted_spectrum, &secret, &embedding);

    let decoded_logical = packed_physical_to_logical(&decoded_physical, shape);

    let timing_decrypt_decode_ms = phase_done(
        decrypt_phase,
        total_phases,
        "decrypt/decode",
        decrypt_decode_start,
    );

    let timing_total_ms = total_start.elapsed().as_millis();

    let (rel_l2, max_abs) = error_metrics(&decoded_logical, &clear_spectrum);

    let inactive_max_abs = decoded_physical[shape.elements()..]
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);

    let status = if rel_l2 <= TOLERANCE && max_abs <= TOLERANCE && inactive_max_abs <= TOLERANCE {
        "PASS"
    } else {
        "FAIL"
    };

    println!("WIDE_RNS_FFT_PROFILE={}", profile.name());
    println!(
        "WIDE_RNS_FFT_SECURITY_BEARING={}",
        profile.security_bearing()
    );
    println!("WIDE_RNS_FFT_RING_DEGREE={degree}");
    println!("WIDE_RNS_FFT_SLOT_COUNT={slot_count}");
    println!(
        "WIDE_RNS_FFT_CHAIN_LIMBS={}",
        profile.modulus_values().len()
    );
    println!(
        "WIDE_RNS_FFT_TOTAL_MODULUS_BITS={}",
        profile.total_modulus_bits()
    );
    println!(
        "WIDE_RNS_FFT_Q_EXCEEDS_U128={}",
        profile.total_modulus_bits() > 128
    );

    println!("WIDE_RNS_FFT_ROWS={}", shape.rows());
    println!("WIDE_RNS_FFT_COLS={}", shape.cols());

    println!("WIDE_RNS_FFT_INPUT_LEVEL={}", ciphertext.level());
    println!("WIDE_RNS_FFT_OUTPUT_LEVEL={}", encrypted_spectrum.level());
    println!(
        "WIDE_RNS_FFT_LEVELS_CONSUMED={}",
        encrypted_spectrum.level() - ciphertext.level()
    );

    println!("WIDE_RNS_FFT_TRANSPOSES=0");
    println!("WIDE_RNS_FFT_REL_L2={rel_l2:.12e}");
    println!("WIDE_RNS_FFT_MAX_ABS={max_abs:.12e}");
    println!("WIDE_RNS_FFT_INACTIVE_MAX_ABS={inactive_max_abs:.12e}");
    println!("WIDE_RNS_FFT_TOLERANCE={TOLERANCE:.12e}");
    println!("WIDE_RNS_FFT_STATUS={status}");

    save_spectrum_image(
        &output_dir.join("02_clear_fft_log_magnitude.png"),
        &clear_spectrum,
        shape,
    );

    save_spectrum_image(
        &output_dir.join("03_encrypted_fft_log_magnitude.png"),
        &decoded_logical,
        shape,
    );

    save_spectrum_error_image(
        &output_dir.join("04_fft_absolute_error.png"),
        &decoded_logical,
        &clear_spectrum,
        shape,
    );

    println!(
        "IMAGE_FFT_INPUT_IMAGE={}",
        output_dir.join(&input_image_name).display()
    );
    println!(
        "IMAGE_FFT_CLEAR_SPECTRUM_IMAGE={}",
        output_dir.join("02_clear_fft_log_magnitude.png").display()
    );
    println!(
        "IMAGE_FFT_ENCRYPTED_SPECTRUM_IMAGE={}",
        output_dir
            .join("03_encrypted_fft_log_magnitude.png")
            .display()
    );
    println!(
        "IMAGE_FFT_ERROR_IMAGE={}",
        output_dir.join("04_fft_absolute_error.png").display()
    );

    println!("FFT_TIMING_ENCRYPTION_MS={timing_encryption_ms}");
    for (level, elapsed_ms) in timing_galois_ms.iter().enumerate() {
        println!("FFT_TIMING_GALOIS_LEVEL_{level}_MS={elapsed_ms}");
    }

    println!("FFT_TIMING_FORWARD_FFT2_MS={timing_forward_fft2_ms}");
    println!("FFT_TIMING_DECRYPT_DECODE_MS={timing_decrypt_decode_ms}");
    println!("FFT_TIMING_TOTAL_MS={timing_total_ms}");

    println!("FFT_SPECTRUM_BEGIN");
    for (index, (clear, encrypted)) in clear_spectrum.iter().zip(&decoded_logical).enumerate() {
        println!(
            "FFT_BIN index={index} clear_re={:.12e} clear_im={:.12e} encrypted_re={:.12e} encrypted_im={:.12e} abs_error={:.12e}",
            clear.re,
            clear.im,
            encrypted.re,
            encrypted.im,
            (*encrypted - *clear).norm()
        );
    }
    println!("FFT_SPECTRUM_END");

    println!("FFT_RESULT_BEGIN");
    println!("FFT_RESULT_DIMENSION={}x{}", shape.rows(), shape.cols());
    println!("FFT_RESULT_PROFILE={}", profile.name());
    println!(
        "FFT_RESULT_LEVELS={}",
        encrypted_spectrum.level() - ciphertext.level()
    );
    println!("FFT_RESULT_REL_L2={rel_l2:.12e}");
    println!("FFT_RESULT_MAX_ABS={max_abs:.12e}");
    println!("FFT_RESULT_INACTIVE_MAX_ABS={inactive_max_abs:.12e}");
    println!("FFT_RESULT_TOLERANCE={TOLERANCE:.12e}");
    println!("FFT_RESULT_STATUS={status}");
    println!("FFT_RESULT_END");

    assert!(rel_l2 <= TOLERANCE);
    assert!(max_abs <= TOLERANCE);
    assert!(inactive_max_abs <= TOLERANCE);
}
