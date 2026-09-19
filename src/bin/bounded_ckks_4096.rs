use ccmm_rs::ckks::{
    multiply_relinearize_rescale_rns_ckks_bounded_with_ntt, research_4096_security_model,
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    BoundedRnsKeygenConfig, BoundedRnsMultiplicationKey,
};
use ccmm_rs::ring::{Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn encode_rns(
    slots: &[Complex64],
    embedding: &CkksCanonicalEmbedding,
    basis: &ccmm_rs::ring::ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    let coefficients = embedding.slots_to_coefficients(slots);
    let residues = basis
        .moduli()
        .iter()
        .copied()
        .map(|modulus| {
            let q = i128::from(modulus.value());
            Polynomial::new(
                modulus,
                coefficients
                    .iter()
                    .map(|&value| {
                        let signed = (value * scale).round() as i128;
                        signed.rem_euclid(q) as u64
                    })
                    .collect(),
            )
        })
        .collect();
    RnsPolynomial::from_residues(residues)
}

fn centered(value: u128, modulus: u128) -> i128 {
    if value > modulus / 2 {
        value as i128 - modulus as i128
    } else {
        value as i128
    }
}

fn main() {
    const BASE_LOG: u32 = 20;
    const SIGMA: f64 = 3.19;
    const TOLERANCE: f64 = 1.0e-3;

    let profile = research_profile_4096();
    let security_model = research_4096_security_model();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let slot_count = profile.slot_count();
    let scale = profile.initial_scale();
    let basis = chain.top().clone();
    let top_plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);

    // Fixed seeding is for reproducibility only. Deployment key generation
    // must use an appropriate cryptographic entropy source.
    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x31E2_0000);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let lhs_slots: Vec<Complex64> = (0..slot_count)
        .map(|index| {
            let x = index as f64;
            Complex64::new(0.10 + 0.00002 * x, -0.08 + 0.00001 * x)
        })
        .collect();
    let rhs_slots: Vec<Complex64> = (0..slot_count)
        .map(|index| {
            let x = index as f64;
            Complex64::new(-0.15 + 0.000015 * x, 0.12 - 0.000008 * x)
        })
        .collect();
    let expected: Vec<Complex64> = lhs_slots
        .iter()
        .zip(&rhs_slots)
        .map(|(&lhs, &rhs)| lhs * rhs)
        .collect();

    let lhs_plaintext = encode_rns(&lhs_slots, &embedding, &basis, scale);
    let rhs_plaintext = encode_rns(&rhs_slots, &embedding, &basis, scale);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    let mut lhs_rng = ChaCha20Rng::seed_from_u64(0x31E2_1001);
    let lhs_rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &lhs_plaintext,
        2,
        distribution,
        &secret,
        &top_plan,
        &mut lhs_rng,
    );
    let mut rhs_rng = ChaCha20Rng::seed_from_u64(0x31E2_1002);
    let rhs_rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &rhs_plaintext,
        2,
        distribution,
        &secret,
        &top_plan,
        &mut rhs_rng,
    );

    let lhs = RnsCkksCiphertext::new(lhs_rlwe, CkksChainState::top(&chain, scale), &chain);
    let rhs = RnsCkksCiphertext::new(rhs_rlwe, CkksChainState::top(&chain, scale), &chain);

    let layout = BoundedGadgetLayout::new(basis.clone(), BASE_LOG);
    let mut key_rng = ChaCha20Rng::seed_from_u64(0x31E2_2000);
    let multiplication_key = BoundedRnsMultiplicationKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout: layout.clone(),
            plan: &top_plan,
        },
        &secret,
        distribution,
        &mut key_rng,
    );

    let result = multiply_relinearize_rescale_rns_ckks_bounded_with_ntt(
        &lhs,
        &rhs,
        &multiplication_key,
        &chain,
        &top_plan,
    );
    assert_eq!(result.level(), 1);

    let result_plan = RnsNttPlan::new(result.basis().moduli().to_vec(), degree);
    let decrypted = decrypt_rns_raw_with_ntt(result.rlwe(), &secret, &result_plan);
    let modulus = decrypted.composite_modulus();
    let decoded_coefficients: Vec<f64> = decrypted
        .reconstruct_coefficients()
        .into_iter()
        .map(|value| centered(value, modulus) as f64 / result.scale())
        .collect();
    let observed = embedding.coefficients_to_slots(&decoded_coefficients);

    let mut max_slot_error = 0.0_f64;
    let mut sum_error = 0.0_f64;
    let mut sum_squared_error = 0.0_f64;
    for (&actual, &reference) in observed.iter().zip(&expected) {
        let error = (actual - reference).norm();
        max_slot_error = max_slot_error.max(error);
        sum_error += error;
        sum_squared_error += error * error;
    }
    let mean_slot_error = sum_error / observed.len() as f64;
    let rms_slot_error = (sum_squared_error / observed.len() as f64).sqrt();

    println!("R3_1E2_BOUNDED_CKKS_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("PARAMETER_CLASS={:?}", profile.parameter_class());
    println!("PROFILE_SECURITY_BEARING={}", profile.security_bearing());
    println!(
        "VALIDATED_SECURITY_MODEL_BITS={}",
        security_model.classical_security_bits
    );
    println!("EXECUTION_PATH=bounded-base-gaussian");
    println!("RING_DEGREE={degree}");
    println!("SLOT_COUNT={slot_count}");
    println!("INPUT_SCALE={scale:.17e}");
    println!("OUTPUT_SCALE={:.17e}", result.scale());
    println!("BOUNDED_BASE_LOG={BASE_LOG}");
    println!("BOUNDED_BASE={}", layout.base());
    println!("BOUNDED_DIGITS={}", layout.digit_count());
    println!("CIPHERTEXT_ERROR_DISTRIBUTION=discrete-gaussian");
    println!("CIPHERTEXT_ERROR_SIGMA={SIGMA}");
    println!("EVALUATION_KEY_ERROR_DISTRIBUTION=discrete-gaussian");
    println!("EVALUATION_KEY_ERROR_SIGMA={SIGMA}");
    println!("CIRCULAR_KDM_ASSUMPTION=required");
    println!("HARDENED_SAMPLER=false");
    println!("REPRODUCIBILITY_SECRET_SEED_FIXED=true");
    println!("MAX_SLOT_ERROR={max_slot_error:.12e}");
    println!("MEAN_SLOT_ERROR={mean_slot_error:.12e}");
    println!("RMS_SLOT_ERROR={rms_slot_error:.12e}");
    println!("TOLERANCE={TOLERANCE:.12e}");

    assert!(
        max_slot_error <= TOLERANCE,
        "R3.1e.2 bounded Gaussian CKKS path exceeded tolerance: max_slot_error={max_slot_error:e}"
    );
    println!("R3_1E2_BOUNDED_CKKS_STATUS=PASS");
}
