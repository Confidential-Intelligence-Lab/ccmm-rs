use ccmm_rs::ckks::{
    mod_switch_rns_ckks_to_next, research_profile_4096, CkksCanonicalEmbedding, CkksChainState,
    RnsCkksCiphertext,
};
use ccmm_rs::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng};
use ccmm_rs::ring::{ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const SIGMA: f64 = 3.19;
const TOLERANCE: f64 = 3.0e-3;

fn encode_rns(
    value: f64,
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    let slots = vec![Complex64::new(value, 0.0); embedding.slot_count()];
    let coefficients = embedding.slots_to_coefficients(&slots);

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
                    .map(|&coefficient| {
                        ((coefficient * scale).round() as i128).rem_euclid(q) as u64
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

#[allow(clippy::too_many_arguments)]
fn encrypt_scalar(
    value: f64,
    seed: u64,
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
    secret: &[i8],
    plan: &RnsNttPlan,
    chain: &ccmm_rs::ring::ModulusChain,
    distribution: ErrorDistribution,
) -> RnsCkksCiphertext {
    let plaintext = encode_rns(value, embedding, basis, scale);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        distribution,
        secret,
        plan,
        &mut rng,
    );

    RnsCkksCiphertext::new(rlwe, CkksChainState::top(chain, scale), chain)
}

fn decode_scalar(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> f64 {
    let plan = RnsNttPlan::new(
        ciphertext.basis().moduli().to_vec(),
        ciphertext.rlwe().degree(),
    );
    let plaintext = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);
    let modulus = plaintext.composite_modulus();

    let coefficients: Vec<f64> = plaintext
        .reconstruct_coefficients()
        .into_iter()
        .map(|value| centered(value, modulus) as f64 / ciphertext.scale())
        .collect();

    let slots = embedding.coefficients_to_slots(&coefficients);
    slots.iter().map(|slot| slot.re).sum::<f64>() / slots.len() as f64
}

fn main() {
    let profile = research_profile_4096();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let basis = chain.top().clone();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x35D2_0000);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let values = [-4.5_f64, -2.0, -0.625, -0.03125, 0.0, 0.125, 1.75, 3.9375];

    let mut max_plain_error_before = 0.0_f64;
    let mut max_plain_error_after = 0.0_f64;
    let mut max_transition_error = 0.0_f64;
    let mut structural_pass = true;

    for (index, &value) in values.iter().enumerate() {
        let ciphertext = encrypt_scalar(
            value,
            0x35D2_1000 ^ index as u64,
            &embedding,
            &basis,
            scale,
            &secret,
            &plan,
            &chain,
            distribution,
        );
        let before = decode_scalar(&ciphertext, &secret, &embedding);
        let switched = mod_switch_rns_ckks_to_next(&ciphertext, &chain);
        let after = decode_scalar(&switched, &secret, &embedding);

        max_plain_error_before = max_plain_error_before.max((before - value).abs());
        max_plain_error_after = max_plain_error_after.max((after - value).abs());
        max_transition_error = max_transition_error.max((after - before).abs());

        structural_pass &= switched.level() == ciphertext.level() + 1;
        structural_pass &= switched.scale() == ciphertext.scale();
        structural_pass &= switched.basis() == chain.level(switched.level());

        println!(
            "CASE_{index}_VALUE={value:.12} BEFORE={before:.12} AFTER={after:.12} TRANSITION_ERROR={:.12e}",
            (after - before).abs()
        );
    }

    let semantic_pass = max_plain_error_before <= TOLERANCE
        && max_plain_error_after <= TOLERANCE
        && max_transition_error <= TOLERANCE;

    println!("R3_5D2A_RNS_MOD_SWITCH_VERSION=1");
    println!("PROFILE=research-4096");
    println!("CASE_COUNT={}", values.len());
    println!("INPUT_LEVEL=0");
    println!("OUTPUT_LEVEL=1");
    println!("INPUT_SCALE={scale:.12e}");
    println!("OUTPUT_SCALE={scale:.12e}");
    println!("MAX_PLAINTEXT_ERROR_BEFORE={max_plain_error_before:.12e}");
    println!("MAX_PLAINTEXT_ERROR_AFTER={max_plain_error_after:.12e}");
    println!("MAX_TRANSITION_ERROR={max_transition_error:.12e}");
    println!(
        "STRUCTURAL_STATUS={}",
        if structural_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "SEMANTIC_STATUS={}",
        if semantic_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "R3_5D2A_RNS_MOD_SWITCH_STATUS={}",
        if structural_pass && semantic_pass {
            "PASS"
        } else {
            "FAIL"
        }
    );

    assert!(
        structural_pass,
        "RNS CKKS modulus-switch structural validation failed"
    );
    assert!(
        semantic_pass,
        "RNS CKKS modulus-switch semantic validation failed"
    );
}
