use ccmm_rs::ckks::{
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::{add_cc, scale_cp};
use ccmm_rs::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng};
use ccmm_rs::matrix::RnsCkksCiphertextMatrix;
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
                    .map(|&x| ((x * scale).round() as i128).rem_euclid(q) as u64)
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

#[allow(clippy::too_many_arguments)]
fn encrypt_vector(
    values: &[f64],
    seed_base: u64,
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
    secret: &[i8],
    plan: &RnsNttPlan,
    chain: &ccmm_rs::ring::ModulusChain,
    distribution: ErrorDistribution,
) -> RnsCkksCiphertextMatrix {
    let data = values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            encrypt_scalar(
                value,
                seed_base ^ index as u64,
                embedding,
                basis,
                scale,
                secret,
                plan,
                chain,
                distribution,
            )
        })
        .collect();

    RnsCkksCiphertextMatrix::from_vec_column_major(values.len(), 1, data)
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
        .map(|x| centered(x, modulus) as f64 / ciphertext.scale())
        .collect();

    let slots = embedding.coefficients_to_slots(&coefficients);
    slots.iter().map(|slot| slot.re).sum::<f64>() / slots.len() as f64
}

fn decode_vector(
    matrix: &RnsCkksCiphertextMatrix,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<f64> {
    (0..matrix.rows())
        .map(|row| decode_scalar(matrix.get(row, 0), secret, embedding))
        .collect()
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn main() {
    let profile = research_profile_4096();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let initial_scale = profile.initial_scale();
    let basis = chain.top().clone();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x35D1_0000);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let x = [1.25, -2.0, 0.375, 4.5];
    let y = [-0.5, 1.25, 2.0, -3.0];
    let alpha = -0.625_f64;

    let x_ct = encrypt_vector(
        &x,
        0x35D1_1000,
        &embedding,
        &basis,
        initial_scale,
        &secret,
        &plan,
        &chain,
        distribution,
    );
    let y_ct = encrypt_vector(
        &y,
        0x35D1_2000,
        &embedding,
        &basis,
        initial_scale,
        &secret,
        &plan,
        &chain,
        distribution,
    );

    let keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &keys);

    let added = add_cc(&evaluator, &x_ct, &y_ct);
    let scaled = scale_cp(&x_ct, alpha, &embedding, &chain, &plan);

    let add_expected: Vec<f64> = x.iter().zip(y).map(|(&a, b)| a + b).collect();
    let scale_expected: Vec<f64> = x.iter().map(|&value| alpha * value).collect();

    let add_actual = decode_vector(&added, &secret, &embedding);
    let scale_actual = decode_vector(&scaled, &secret, &embedding);

    let add_error = max_error(&add_actual, &add_expected);
    let scale_error = max_error(&scale_actual, &scale_expected);

    let input_level = x_ct.get(0, 0).level();
    let add_level = added.get(0, 0).level();
    let scale_level = scaled.get(0, 0).level();

    let input_scale = x_ct.get(0, 0).scale();
    let add_scale = added.get(0, 0).scale();
    let scale_scale = scaled.get(0, 0).scale();
    let dropped = chain
        .dropped_modulus(input_level)
        .expect("top level must have a dropped modulus")
        .value() as f64;
    let expected_scale_after_scale = input_scale * dropped / dropped;
    let scale_relative_error =
        ((scale_scale - expected_scale_after_scale) / expected_scale_after_scale).abs();

    let add_pass = add_error <= TOLERANCE && add_level == input_level && add_scale == input_scale;
    let scale_pass = scale_error <= TOLERANCE
        && scale_level == input_level + 1
        && scale_relative_error <= 1.0e-12;

    println!("R3_5D1_EBLAS_LEVEL1_VERSION=1");
    println!("PROFILE=research-4096");
    println!("VECTOR_LENGTH={}", x.len());
    println!("ALPHA={alpha:.12}");
    println!("INPUT_LEVEL={input_level}");
    println!("INPUT_SCALE={input_scale:.12e}");
    println!("ADD_LEVEL={add_level}");
    println!("ADD_SCALE={add_scale:.12e}");
    println!("ADD_MAX_ERROR={add_error:.12e}");
    println!("ADD_STATUS={}", if add_pass { "PASS" } else { "FAIL" });
    println!("SCALE_LEVEL={scale_level}");
    println!("SCALE_SCALE={scale_scale:.12e}");
    println!("SCALE_RELATIVE_SCALE_ERROR={scale_relative_error:.12e}");
    println!("SCALE_MAX_ERROR={scale_error:.12e}");
    println!("SCALE_STATUS={}", if scale_pass { "PASS" } else { "FAIL" });
    println!(
        "R3_5D1_EBLAS_LEVEL1_STATUS={}",
        if add_pass && scale_pass {
            "PASS"
        } else {
            "FAIL"
        }
    );

    assert!(add_pass, "eBLAS ADD validation failed");
    assert!(scale_pass, "eBLAS SCALE validation failed");
}
