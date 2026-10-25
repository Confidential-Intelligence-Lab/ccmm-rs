use std::hint::black_box;
use std::time::Instant;

use ccmm_rs::grafting::decrypt_rns_raw_with_ntt;
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big,
    PreparedRnsNttPlan,
};
use num_traits::ToPrimitive;

use ccmm_rs::ckks::{
    mod_switch_rns_ckks_to_next, multiply_plain_rns_ckks_with_ntt,
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next, research_profile_16384,
    research_profile_32768, research_profile_4096, research_profile_65536, research_profile_8192,
    CkksCanonicalEmbedding, CkksChainState, PreparedRnsPlaintextNtt, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::grafting::encrypt_rns_raw_with_distribution_ntt_rng;
use ccmm_rs::ring::{ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn encode_rns(
    slots: &[Complex64],
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
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
        .iter()
        .map(|value| {
            centered_representative_big(value, &modulus)
                .to_f64()
                .expect("CKKS coefficient conversion failed")
                / ciphertext.scale()
        })
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

fn validate_slots(name: &str, actual: &[Complex64], expected: &[Complex64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());

    let mut max_error = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    for (&a, &e) in actual.iter().zip(expected) {
        let error = (a - e).norm();
        assert!(error.is_finite(), "{name}: non-finite error");
        max_error = max_error.max(error);
        sum_sq += error * error;
    }

    let rms_error = (sum_sq / actual.len() as f64).sqrt();

    println!(
        "R30_13B2_ACCURACY OPERATION={name} MAX_ERROR={max_error:.12e} RMS_ERROR={rms_error:.12e} TOLERANCE={tolerance:.12e}"
    );

    assert!(
        max_error <= tolerance,
        "{name}: maximum error {max_error:e} exceeds {tolerance:e}"
    );
}

fn assert_exact_ciphertext(lhs: &RnsCkksCiphertext, rhs: &RnsCkksCiphertext) {
    assert_eq!(lhs.level(), rhs.level());
    assert_eq!(lhs.scale(), rhs.scale());
    assert_eq!(lhs.basis(), rhs.basis());
    assert_eq!(lhs.rlwe().limbs().len(), rhs.rlwe().limbs().len());

    for (a, b) in lhs.rlwe().limbs().iter().zip(rhs.rlwe().limbs()) {
        assert_eq!(a.a(), b.a());
        assert_eq!(a.b(), b.b());
    }

    println!("R30_13B2_PREPARED_CP_EXACT_EQUALITY=PASS");
}

fn benchmark<F, T>(name: &str, warmups: usize, repeats: usize, mut operation: F)
where
    F: FnMut() -> T,
{
    assert!(repeats > 0);

    for _ in 0..warmups {
        black_box(operation());
    }

    let mut samples = Vec::with_capacity(repeats);

    for _ in 0..repeats {
        let start = Instant::now();
        let result = operation();
        let elapsed = start.elapsed().as_secs_f64() * 1.0e3;
        black_box(result);
        samples.push(elapsed);
    }

    samples.sort_by(|a, b| a.total_cmp(b));

    let median = samples[repeats / 2];
    let p95_index = ((repeats * 95).div_ceil(100)).saturating_sub(1);
    let p95 = samples[p95_index];

    println!(
        "R30_13B_RESULT OPERATION={name} MEDIAN_MS={median:.6} P95_MS={p95:.6} REPEATS={repeats}"
    );
}

fn main() {
    let degree_requested: usize = std::env::var("R30_13B_DEGREE")
        .unwrap_or_else(|_| "4096".to_string())
        .parse()
        .expect("R30_13B_DEGREE must be an integer");

    let profile = match degree_requested {
        4096 => research_profile_4096(),
        8192 => research_profile_8192(),
        16384 => research_profile_16384(),
        32768 => research_profile_32768(),
        65536 => research_profile_65536(),
        _ => panic!("Unsupported R30_13B_DEGREE: {degree_requested}"),
    };

    println!("R30_13B3_PROFILE_SELECTION=PASS");
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let basis = chain.top().clone();

    assert!(!profile.security_bearing());
    assert!(chain.len() >= 2);

    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);

    let slots: Vec<Complex64> = (0..profile.slot_count())
        .map(|index| Complex64::new(0.10 + index as f64 * 0.00001, 0.0))
        .collect();

    let plaintext = encode_rns(&slots, &embedding, &basis, scale);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x3013_B001);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };
    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x3013_B002);

    let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        distribution,
        &secret,
        &plan,
        &mut encryption_rng,
    );

    let ciphertext = RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, scale), &chain);

    let keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &keys);

    let warmups: usize = std::env::var("R30_13B_WARMUPS")
        .unwrap_or_else(|_| "3".to_string())
        .parse()
        .expect("Invalid R30_13B_WARMUPS");

    let repeats: usize = std::env::var("R30_13B_REPEATS")
        .unwrap_or_else(|_| "20".to_string())
        .parse()
        .expect("Invalid R30_13B_REPEATS");

    assert!(repeats > 0 && repeats <= 1000);
    assert!(warmups <= 1000);

    println!("R30_13B_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("RING_DEGREE={degree}");
    println!("RNS_LIMBS={}", basis.len());
    println!("SECURITY_CLASSIFICATION=RESEARCH_ONLY");
    println!("SECURITY_QUALIFIED=false");
    println!("WARMUPS={warmups}");
    println!("REPEATS={repeats}");

    let added = evaluator.add(&ciphertext, &ciphertext);
    assert_eq!(added.level(), ciphertext.level());
    assert_eq!(added.scale(), ciphertext.scale());

    let multiplied =
        multiply_plain_rns_ckks_with_ntt(&ciphertext, &plaintext, scale, &chain, &plan);
    assert_eq!(multiplied.level(), ciphertext.level());

    let rescaled = rescale_rns_ckks_to_next(&ciphertext, &chain);
    assert_eq!(rescaled.level(), ciphertext.level() + 1);

    let switched = mod_switch_rns_ckks_to_next(&ciphertext, &chain);
    assert_eq!(switched.level(), ciphertext.level() + 1);
    assert_eq!(switched.scale(), ciphertext.scale());

    println!("R30_13B2_VERSION=1");

    const ACCURACY_TOLERANCE: f64 = 2.0e-3;

    let expected_add: Vec<Complex64> = slots.iter().map(|&x| x + x).collect();

    validate_slots(
        "add_cc",
        &decode_slots(&added, &secret, &embedding),
        &expected_add,
        ACCURACY_TOLERANCE,
    );

    let expected_cp: Vec<Complex64> = slots.iter().map(|&x| x * x).collect();

    validate_slots(
        "multiply_cp_ntt",
        &decode_slots(&multiplied, &secret, &embedding),
        &expected_cp,
        ACCURACY_TOLERANCE,
    );

    let prepared_plan = PreparedRnsNttPlan::new(&plan);
    let prepared_plaintext = PreparedRnsPlaintextNtt::new(&plaintext, &prepared_plan);

    let prepared_result = multiply_plain_rns_ckks_with_prepared_ntt(
        &ciphertext,
        &prepared_plaintext,
        scale,
        &chain,
        &prepared_plan,
    );

    assert_exact_ciphertext(&multiplied, &prepared_result);

    validate_slots(
        "multiply_cp_prepared",
        &decode_slots(&prepared_result, &secret, &embedding),
        &expected_cp,
        ACCURACY_TOLERANCE,
    );

    println!("R30_13B2_NUMERICAL_VALIDATION=PASS");

    println!("R30_13B_STRUCTURAL_VALIDATION=PASS");

    benchmark("add_cc", warmups, repeats, || {
        evaluator.add(black_box(&ciphertext), black_box(&ciphertext))
    });

    benchmark("multiply_cp_ntt", warmups, repeats, || {
        multiply_plain_rns_ckks_with_ntt(
            black_box(&ciphertext),
            black_box(&plaintext),
            scale,
            &chain,
            &plan,
        )
    });

    benchmark("rescale", warmups, repeats, || {
        rescale_rns_ckks_to_next(black_box(&ciphertext), &chain)
    });

    benchmark("mod_switch", warmups, repeats, || {
        mod_switch_rns_ckks_to_next(black_box(&ciphertext), &chain)
    });

    benchmark("prepared_plan_construction", warmups, repeats, || {
        PreparedRnsNttPlan::new(black_box(&plan))
    });

    benchmark("plaintext_preparation", warmups, repeats, || {
        PreparedRnsPlaintextNtt::new(black_box(&plaintext), &prepared_plan)
    });

    benchmark("multiply_cp_prepared", warmups, repeats, || {
        multiply_plain_rns_ckks_with_prepared_ntt(
            black_box(&ciphertext),
            black_box(&prepared_plaintext),
            scale,
            &chain,
            &prepared_plan,
        )
    });

    println!("R30_13B2_STATUS=PASS");

    println!("R30_13B_STATUS=PASS");
}
