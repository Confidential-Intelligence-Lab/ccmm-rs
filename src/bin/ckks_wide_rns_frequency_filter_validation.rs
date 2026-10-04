use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_16384, rotation_exponent_left, rotation_exponent_right,
    CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext, RnsCkksEvaluationKeys,
    RnsCkksEvaluator, RnsCkksLevelKeys, RnsGaloisKey,
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
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::io::{self, Write};
use std::time::Instant;

const SIGMA: f64 = 3.19;
const TOLERANCE: f64 = 5.0e-3;

fn phase_start(index: usize, name: &str) -> Instant {
    println!("[{index}/11] {name} ... START");
    io::stdout()
        .flush()
        .expect("frequency-filter progress output must flush");
    Instant::now()
}

fn phase_done(index: usize, name: &str, start: Instant) -> u128 {
    let elapsed_ms = start.elapsed().as_millis();
    println!("[{index}/11] {name} ... DONE elapsed_ms={elapsed_ms}");
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

fn main() {
    let profile = research_profile_16384();
    let degree = profile.degree();
    let scale = profile.initial_scale();

    let shape = Fft2Shape::new(16, 16);
    let chain = profile.modulus_chain();
    let top_basis = chain.top().clone();

    let embedding = CkksCanonicalEmbedding::new(degree);
    let slot_count = embedding.slot_count();
    let top_plan = profile.rns_ntt_plan();

    assert_eq!(degree, 16384);
    assert_eq!(top_basis.len(), 9);
    assert_eq!(profile.total_modulus_bits(), 405);
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
    println!("FFT_SIGMA={SIGMA:.6}");
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

    let logical_input: Vec<Complex64> = (0..shape.elements())
        .map(|index| {
            let row = index / shape.cols();
            let col = index % shape.cols();

            /*
             * Deterministic bounded grayscale-like 16x16 pattern.
             */
            let value = ((row * 17 + col * 11 + ((row ^ col) * 3)) % 256) as f64 / 255.0;

            Complex64::new(value, 0.0)
        })
        .collect();

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
    let encryption_start = phase_start(1, "encryption");

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

    let timing_encryption_ms = phase_done(1, "encryption", encryption_start);

    /*
     * Forward 16x16 packed DIF FFT2:
     *
     * row stages:
     *   level 0 -> 1 : span 16, rotation 8
     *   level 1 -> 2 : span  8, rotation 4
     *   level 2 -> 3 : span  4, rotation 2
     *   level 3 -> 4 : span  2, rotation 1
     *
     * column stages:
     *   level 4 -> 5 : span 16, rotation 128
     *   level 5 -> 6 : span  8, rotation 64
     *   level 6 -> 7 : span  4, rotation 32
     *   level 7 -> 8 : span  2, rotation 16
     */
    let level_rotations = [
        (0usize, 8usize),
        (1usize, 4usize),
        (2usize, 2usize),
        (3usize, 1usize),
        (4usize, 128usize),
        (5usize, 64usize),
        (6usize, 32usize),
        (7usize, 16usize),
    ];

    let mut evaluation_keys = RnsCkksEvaluationKeys::new();
    let mut timing_galois_ms = [0_u128; 8];

    for (phase_offset, (level, rotation)) in level_rotations.into_iter().enumerate() {
        let phase_index = phase_offset + 2;
        let phase_name = format!("level-{level} Galois keygen");
        let keygen_start = phase_start(phase_index, &phase_name);

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

        for (tag, exponent) in [
            (0_u64, rotation_exponent_left(degree, rotation)),
            (1_u64, rotation_exponent_right(degree, rotation)),
        ] {
            let direction = if tag == 0 { "left" } else { "right" };

            println!(
                "      level={level} rotation={rotation} direction={direction} exponent={exponent} ... START"
            );
            io::stdout()
                .flush()
                .expect("Galois-key progress output must flush");

            let individual_start = Instant::now();

            let mut rng =
                ChaCha20Rng::seed_from_u64(0x4652_4551_4649_4d00 ^ ((level as u64) << 8) ^ tag);

            let key = RnsGaloisKey::generate_with_ntt_rng(
                RnsKeygenConfig {
                    degree,
                    plaintext_modulus: 2,
                    noise_bound: 0,
                    layout: level_layout.clone(),
                    plan: &level_plan,
                },
                &secret,
                exponent,
                &mut rng,
            );

            println!(
                "      level={level} rotation={rotation} direction={direction} exponent={exponent} ... DONE elapsed_ms={}",
                individual_start.elapsed().as_millis()
            );

            level_keys.insert_galois_key(key);
        }

        evaluation_keys.insert_level(level_keys);

        timing_galois_ms[phase_offset] = phase_done(phase_index, &phase_name, keygen_start);
    }

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    /*
     * Phase 4: encrypted forward FFT2.
     */
    let forward_fft2_start = phase_start(10, "forward FFT2");

    let encrypted_spectrum = execute_packed_fft2_dif_cp(
        shape,
        FftDirection::Forward,
        &ciphertext,
        &evaluator,
        &embedding,
        &chain,
    );

    let timing_forward_fft2_ms = phase_done(10, "forward FFT2", forward_fft2_start);

    assert_eq!(encrypted_spectrum.level(), 8);

    /*
     * Phase 5: decrypt/decode the encrypted frequency-domain result.
     *
     * The packed DIF representation is converted explicitly back to logical
     * row-major frequency-bin order before comparison with fft2_pp.
     */
    let decrypt_decode_start = phase_start(11, "decrypt/decode");

    let decoded_physical = decrypt_slots(&encrypted_spectrum, &secret, &embedding);

    let decoded_logical = packed_physical_to_logical(&decoded_physical, shape);

    let timing_decrypt_decode_ms = phase_done(11, "decrypt/decode", decrypt_decode_start);

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

    assert!(rel_l2 <= TOLERANCE);
    assert!(max_abs <= TOLERANCE);
    assert!(inactive_max_abs <= TOLERANCE);
}
