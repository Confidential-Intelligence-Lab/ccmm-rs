use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_8192, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::fft::packed_fft_dif_butterfly_cp;
use ccmm_rs::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng};
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
                .expect("centered CKKS coefficient must fit f64")
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
    let profile = research_profile_8192();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    let basis = chain.top().clone();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slot_count = embedding.slot_count();
    let top_plan = profile.rns_ntt_plan();

    let logical_length = 8usize;

    let mut a_slots = vec![Complex64::new(0.0, 0.0); slot_count];
    let mut b_slots = vec![Complex64::new(0.0, 0.0); slot_count];
    let mut twiddles = vec![Complex64::new(1.0, 0.0); slot_count];

    for index in 0..logical_length {
        a_slots[index] = Complex64::new(index as f64 + 1.0, 0.25 * index as f64);
        b_slots[index] = Complex64::new(0.5 * index as f64 - 1.0, -0.125 * index as f64);

        let angle = -2.0 * std::f64::consts::PI * index as f64 / logical_length as f64;
        twiddles[index] = Complex64::new(angle.cos(), angle.sin());
    }

    let clear_upper: Vec<Complex64> = a_slots.iter().zip(&b_slots).map(|(&a, &b)| a + b).collect();

    let clear_lower: Vec<Complex64> = a_slots
        .iter()
        .zip(&b_slots)
        .zip(&twiddles)
        .map(|((&a, &b), &w)| w * (a - b))
        .collect();

    let plain_a = encode_rns(&a_slots, &embedding, &basis, scale);
    let plain_b = encode_rns(&b_slots, &embedding, &basis, scale);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4255_5454_4552_0001);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let mut rng_a = ChaCha20Rng::seed_from_u64(0x4255_5454_4552_0002);
    let mut rng_b = ChaCha20Rng::seed_from_u64(0x4255_5454_4552_0003);

    let rlwe_a = encrypt_rns_raw_with_distribution_ntt_rng(
        &plain_a,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &top_plan,
        &mut rng_a,
    );

    let rlwe_b = encrypt_rns_raw_with_distribution_ntt_rng(
        &plain_b,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &top_plan,
        &mut rng_b,
    );

    let a = RnsCkksCiphertext::new(rlwe_a, CkksChainState::top(&chain, scale), &chain);

    let b = RnsCkksCiphertext::new(rlwe_b, CkksChainState::top(&chain, scale), &chain);

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let plan = RnsNttPlan::new(a.basis().moduli().to_vec(), degree);

    let (upper, lower) =
        packed_fft_dif_butterfly_cp(&evaluator, &a, &b, &twiddles, &embedding, &chain, &plan);

    assert_eq!(upper.level(), 1);
    assert_eq!(lower.level(), 1);

    let decoded_upper = decrypt_slots(&upper, &secret, &embedding);
    let decoded_lower = decrypt_slots(&lower, &secret, &embedding);

    let (upper_rel_l2, upper_max_abs) = error_metrics(&decoded_upper, &clear_upper);
    let (lower_rel_l2, lower_max_abs) = error_metrics(&decoded_lower, &clear_lower);

    let status = if upper_rel_l2 <= TOLERANCE
        && upper_max_abs <= TOLERANCE
        && lower_rel_l2 <= TOLERANCE
        && lower_max_abs <= TOLERANCE
    {
        "PASS"
    } else {
        "FAIL"
    };

    println!("PACKED_FFT_BUTTERFLY_PROFILE={}", profile.name());
    println!("PACKED_FFT_BUTTERFLY_RING_DEGREE={degree}");
    println!("PACKED_FFT_BUTTERFLY_INPUT_LEVEL=0");
    println!("PACKED_FFT_BUTTERFLY_OUTPUT_LEVEL={}", upper.level());
    println!("PACKED_FFT_BUTTERFLY_LEVELS_CONSUMED={}", upper.level());
    println!("PACKED_FFT_BUTTERFLY_UPPER_REL_L2={upper_rel_l2:.12e}");
    println!("PACKED_FFT_BUTTERFLY_UPPER_MAX_ABS={upper_max_abs:.12e}");
    println!("PACKED_FFT_BUTTERFLY_LOWER_REL_L2={lower_rel_l2:.12e}");
    println!("PACKED_FFT_BUTTERFLY_LOWER_MAX_ABS={lower_max_abs:.12e}");
    println!("PACKED_FFT_BUTTERFLY_TOLERANCE={TOLERANCE:.12e}");
    println!("PACKED_FFT_BUTTERFLY_STATUS={status}");

    assert_eq!(status, "PASS");
}
