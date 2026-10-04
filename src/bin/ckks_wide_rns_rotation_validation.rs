use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_8192, rotation_exponent_left, CkksCanonicalEmbedding, CkksChainState,
    RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsCkksLevelKeys, RnsGaloisKey,
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

const SIGMA: f64 = 3.2;
const ROTATION: usize = 1;
const ACTIVE_SLOTS: usize = 8;
const TOLERANCE: f64 = 1.0e-4;

fn block_sizes(limb_count: usize) -> Vec<usize> {
    vec![1; limb_count]
}

fn decode_slots(
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

    assert_eq!(degree, 8192);
    assert!(profile.total_modulus_bits() > 128);
    assert!(ACTIVE_SLOTS < slot_count);

    let mut input = vec![Complex64::new(0.0, 0.0); slot_count];

    for (index, slot) in input.iter_mut().take(ACTIVE_SLOTS).enumerate() {
        *slot = Complex64::new(
            (index as f64 + 1.0) / 32.0,
            -((2 * index + 1) as f64) / 64.0,
        );
    }

    let plaintext = encode_rns(&input, &embedding, &basis, scale);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5749_4445_524e_5301);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x5749_4445_524e_5302);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &plan,
        &mut encryption_rng,
    );

    let ciphertext = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

    let exponent = rotation_exponent_left(degree, ROTATION);

    let mut key_rng = ChaCha20Rng::seed_from_u64(0x5749_4445_524e_5303);

    let galois_key = RnsGaloisKey::generate_with_rng(
        degree,
        2,
        0,
        &secret,
        exponent,
        RnsGadgetLayout::new(basis.clone(), block_sizes(basis.len())),
        &mut key_rng,
    );

    let mut level_keys = RnsCkksLevelKeys::new(0, basis.clone());
    level_keys.insert_galois_key(galois_key);

    let mut evaluation_keys = RnsCkksEvaluationKeys::new();
    evaluation_keys.insert_level(level_keys);

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);
    let rotated = evaluator.rotate_left(&ciphertext, ROTATION);

    assert_eq!(rotated.level(), ciphertext.level());
    assert_eq!(rotated.basis(), ciphertext.basis());

    let decoded = decode_slots(&rotated, &secret, &embedding);

    let expected: Vec<Complex64> = (0..slot_count)
        .map(|index| input[(index + ROTATION) % slot_count])
        .collect();

    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (&actual, &reference) in decoded.iter().zip(&expected) {
        let error = actual - reference;
        squared_error += error.norm_sqr();
        squared_reference += reference.norm_sqr();
        max_abs = max_abs.max(error.norm());
    }

    let rel_l2 = if squared_reference > 0.0 {
        (squared_error / squared_reference).sqrt()
    } else {
        squared_error.sqrt()
    };

    let q_exceeds_u128 = profile.total_modulus_bits() > 128;
    let status = if max_abs <= TOLERANCE { "PASS" } else { "FAIL" };

    println!("WIDE_RNS_ROTATION_PROFILE={}", profile.name());
    println!("WIDE_RNS_ROTATION_RING_DEGREE={degree}");
    println!("WIDE_RNS_ROTATION_SLOT_COUNT={slot_count}");
    println!(
        "WIDE_RNS_ROTATION_CHAIN_LIMBS={}",
        profile.modulus_values().len()
    );
    println!(
        "WIDE_RNS_ROTATION_TOTAL_MODULUS_BITS={}",
        profile.total_modulus_bits()
    );
    println!("WIDE_RNS_ROTATION_Q_EXCEEDS_U128={q_exceeds_u128}");
    println!(
        "WIDE_RNS_ROTATION_SECURITY_BEARING={}",
        profile.security_bearing()
    );
    println!("WIDE_RNS_ROTATION_STEPS={ROTATION}");
    println!("WIDE_RNS_ROTATION_LEVEL_BEFORE={}", ciphertext.level());
    println!("WIDE_RNS_ROTATION_LEVEL_AFTER={}", rotated.level());
    println!("WIDE_RNS_ROTATION_REL_L2={rel_l2:.12e}");
    println!("WIDE_RNS_ROTATION_MAX_ABS={max_abs:.12e}");
    println!("WIDE_RNS_ROTATION_TOLERANCE={TOLERANCE:.12e}");
    println!("WIDE_RNS_ROTATION_STATUS={status}");

    assert!(
        max_abs <= TOLERANCE,
        "wide-RNS rotation error {max_abs:e} exceeds tolerance {TOLERANCE:e}"
    );
}
