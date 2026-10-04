use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_8192, rotation_exponent_left, rotation_exponent_right, CkksCanonicalEmbedding,
    CkksChainState, RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsCkksLevelKeys,
    RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{
    execute_packed_fft2_dif_stage_cp, execute_packed_fft2_dif_stage_pp, Fft2Shape, FftDirection,
    PackedFft2Axis, PackedFft2DifStageDiagonals,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const SIGMA: f64 = 3.19;
const TOLERANCE: f64 = 5.0e-3;

fn block_sizes(limb_count: usize) -> Vec<usize> {
    vec![1; limb_count]
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

fn main() {
    let profile = research_profile_8192();
    let degree = profile.degree();
    let scale = profile.initial_scale();

    let chain = profile.modulus_chain();
    let basis = chain.top().clone();

    let embedding = CkksCanonicalEmbedding::new(degree);
    let slot_count = embedding.slot_count();
    let plan = profile.rns_ntt_plan();

    /*
     * Deliberately use the same logical 2x4 geometry as the existing
     * functional FFT2-stage validator, but exercise only the row span-2
     * stage first.
     *
     * Rows / span 2 -> rotation 1.
     */
    let shape = Fft2Shape::new(2, 4);
    let axis = PackedFft2Axis::Rows;
    let span = 2usize;

    assert_eq!(degree, 8192);
    assert!(profile.total_modulus_bits() > 128);
    assert_eq!(basis.len(), 5);

    let logical_input: Vec<Complex64> = (0..shape.elements())
        .map(|index| {
            let re = ((index * 7 + 3) % 17) as f64 - 8.0;
            let im = ((index * 11 + 5) % 19) as f64 - 9.0;
            Complex64::new(re / 256.0, im / 256.0)
        })
        .collect();

    let mut physical_input = vec![Complex64::new(0.0, 0.0); slot_count];

    physical_input[..shape.elements()].copy_from_slice(&logical_input);

    let plaintext = encode_rns(&physical_input, &embedding, &basis, scale);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5749_4445_4632_5301);

    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x5749_4445_4632_5302);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &plan,
        &mut encryption_rng,
    );

    let ciphertext = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

    let diagonals = PackedFft2DifStageDiagonals::new(shape, axis, span, FftDirection::Forward);

    assert_eq!(diagonals.rotation(), 1);

    let mut level_keys = RnsCkksLevelKeys::new(0, basis.clone());

    for (tag, exponent) in [
        (0_u64, rotation_exponent_left(degree, diagonals.rotation())),
        (1_u64, rotation_exponent_right(degree, diagonals.rotation())),
    ] {
        let mut rng = ChaCha20Rng::seed_from_u64(0x5749_4445_4632_5400 ^ tag);

        let key = RnsGaloisKey::generate_with_rng(
            degree,
            2,
            0,
            &secret,
            exponent,
            RnsGadgetLayout::new(basis.clone(), block_sizes(basis.len())),
            &mut rng,
        );

        level_keys.insert_galois_key(key);
    }

    let mut evaluation_keys = RnsCkksEvaluationKeys::new();
    evaluation_keys.insert_level(level_keys);

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let expected = execute_packed_fft2_dif_stage_pp(&logical_input, &diagonals);

    let encrypted = execute_packed_fft2_dif_stage_cp(
        &ciphertext,
        &diagonals,
        &evaluator,
        &embedding,
        &chain,
        &plan,
    );

    let actual = decrypt_slots(&encrypted, &secret, &embedding);

    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for index in 0..shape.elements() {
        let error = actual[index] - expected[index];

        squared_error += error.norm_sqr();
        squared_reference += expected[index].norm_sqr();
        max_abs = max_abs.max(error.norm());
    }

    let rel_l2 = if squared_reference > 0.0 {
        (squared_error / squared_reference).sqrt()
    } else {
        squared_error.sqrt()
    };

    let inactive_max_abs = actual[shape.elements()..]
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);

    assert_eq!(
        encrypted.level(),
        ciphertext.level() + 1,
        "wide-RNS packed FFT2 DIF stage must consume exactly one level"
    );

    assert!(
        (encrypted.scale() / ciphertext.scale() - 1.0).abs() < 1.0e-12,
        "wide-RNS packed FFT2 DIF stage must preserve nominal scale"
    );

    let status = if rel_l2 <= TOLERANCE && max_abs <= TOLERANCE && inactive_max_abs <= TOLERANCE {
        "PASS"
    } else {
        "FAIL"
    };

    println!("WIDE_RNS_FFT2_STAGE_PROFILE={}", profile.name());
    println!(
        "WIDE_RNS_FFT2_STAGE_SECURITY_BEARING={}",
        profile.security_bearing()
    );
    println!("WIDE_RNS_FFT2_STAGE_RING_DEGREE={degree}");
    println!("WIDE_RNS_FFT2_STAGE_SLOT_COUNT={slot_count}");
    println!(
        "WIDE_RNS_FFT2_STAGE_CHAIN_LIMBS={}",
        profile.modulus_values().len()
    );
    println!(
        "WIDE_RNS_FFT2_STAGE_TOTAL_MODULUS_BITS={}",
        profile.total_modulus_bits()
    );
    println!(
        "WIDE_RNS_FFT2_STAGE_Q_EXCEEDS_U128={}",
        profile.total_modulus_bits() > 128
    );
    println!("WIDE_RNS_FFT2_STAGE_ROWS={}", shape.rows());
    println!("WIDE_RNS_FFT2_STAGE_COLS={}", shape.cols());
    println!("WIDE_RNS_FFT2_STAGE_AXIS={axis:?}");
    println!("WIDE_RNS_FFT2_STAGE_SPAN={span}");
    println!("WIDE_RNS_FFT2_STAGE_ROTATION={}", diagonals.rotation());
    println!("WIDE_RNS_FFT2_STAGE_INPUT_LEVEL={}", ciphertext.level());
    println!("WIDE_RNS_FFT2_STAGE_OUTPUT_LEVEL={}", encrypted.level());
    println!(
        "WIDE_RNS_FFT2_STAGE_LEVELS_CONSUMED={}",
        encrypted.level() - ciphertext.level()
    );
    println!("WIDE_RNS_FFT2_STAGE_REL_L2={rel_l2:.12e}");
    println!("WIDE_RNS_FFT2_STAGE_MAX_ABS={max_abs:.12e}");
    println!("WIDE_RNS_FFT2_STAGE_INACTIVE_MAX_ABS={inactive_max_abs:.12e}");
    println!("WIDE_RNS_FFT2_STAGE_TOLERANCE={TOLERANCE:.12e}");
    println!("WIDE_RNS_FFT2_STAGE_STATUS={status}");

    assert!(rel_l2 <= TOLERANCE);
    assert!(max_abs <= TOLERANCE);
    assert!(inactive_max_abs <= TOLERANCE);
}
