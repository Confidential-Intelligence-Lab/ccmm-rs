use std::time::Instant;

use ccmm_rs::ckks::{
    research_4096_security_model, research_profile_4096, CkksCanonicalEmbedding, CkksChainState,
    RnsCkksCiphertext,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    BoundedRnsKeygenConfig, BoundedRnsMultiplicationKey,
};
use ccmm_rs::matrix::RnsCkksCiphertextMatrix;
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

fn centered(value: u128, modulus: u128) -> i128 {
    if value > modulus / 2 {
        value as i128 - modulus as i128
    } else {
        value as i128
    }
}

struct EncryptionContext<'a> {
    embedding: &'a CkksCanonicalEmbedding,
    basis: &'a ModulusBasis,
    scale: f64,
    secret: &'a [i8],
    plan: &'a RnsNttPlan,
    chain: &'a ccmm_rs::ring::ModulusChain,
    distribution: ErrorDistribution,
}

fn encrypt_scalar(value: f64, seed: u64, context: &EncryptionContext<'_>) -> RnsCkksCiphertext {
    let slots = vec![Complex64::new(value, 0.0); context.secret.len() / 2];

    let plaintext = encode_rns(&slots, context.embedding, context.basis, context.scale);

    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        context.distribution,
        context.secret,
        context.plan,
        &mut rng,
    );

    RnsCkksCiphertext::new(
        rlwe,
        CkksChainState::top(context.chain, context.scale),
        context.chain,
    )
}

fn decode_scalar(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> (f64, f64) {
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

    let mean = slots.iter().map(|slot| slot.re).sum::<f64>() / slots.len() as f64;
    let max_imag = slots
        .iter()
        .map(|slot| slot.im.abs())
        .fold(0.0_f64, f64::max);

    (mean, max_imag)
}

fn main() {
    const BASE_LOG: u32 = 20;
    const SIGMA: f64 = 3.19;
    const TOLERANCE: f64 = 2.0e-3;

    let profile = research_profile_4096();
    let security_model = research_4096_security_model();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let basis = chain.top().clone();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x320C_0000);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let encryption_context = EncryptionContext {
        embedding: &embedding,
        basis: &basis,
        scale,
        secret: &secret,
        plan: &plan,
        chain: &chain,
        distribution,
    };

    let layout = BoundedGadgetLayout::new(basis.clone(), BASE_LOG);

    let keygen_start = Instant::now();
    let mut key_rng = ChaCha20Rng::seed_from_u64(0x320C_1000);
    let multiplication_key = BoundedRnsMultiplicationKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout: layout.clone(),
            plan: &plan,
        },
        &secret,
        distribution,
        &mut key_rng,
    );
    let keygen_us = keygen_start.elapsed().as_secs_f64() * 1.0e6;

    // Two parties contribute encrypted 2x2 matrices.
    //
    // Party A:
    // A = [[ 0.20,  0.40],
    //      [-0.30,  0.50]]
    //
    // Party B:
    // B = [[ 0.60, -0.20],
    //      [ 0.10,  0.70]]
    //
    // C = A B =
    //     [[ 0.16,  0.24],
    //      [-0.13,  0.41]]
    let party_a = [0.20, -0.30, 0.40, 0.50];
    let party_b = [0.60, 0.10, -0.20, 0.70];
    let expected = [[0.16, 0.24], [-0.13, 0.41]];

    let encryption_start = Instant::now();

    let encrypted_a = RnsCkksCiphertextMatrix::from_vec_column_major(
        2,
        2,
        party_a
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                encrypt_scalar(value, 0x320C_2000 + index as u64, &encryption_context)
            })
            .collect(),
    );

    let encrypted_b = RnsCkksCiphertextMatrix::from_vec_column_major(
        2,
        2,
        party_b
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                encrypt_scalar(value, 0x320C_3000 + index as u64, &encryption_context)
            })
            .collect(),
    );

    let encryption_us = encryption_start.elapsed().as_secs_f64() * 1.0e6;

    let evaluation_start = Instant::now();

    let encrypted_product =
        encrypted_a.matmul_bounded_with_ntt(&encrypted_b, &multiplication_key, &chain, &plan);

    let evaluation_us = evaluation_start.elapsed().as_secs_f64() * 1.0e6;

    assert_eq!(encrypted_product.rows(), 2);
    assert_eq!(encrypted_product.cols(), 2);
    assert_eq!(encrypted_product.level(), 1);

    let mut max_error = 0.0_f64;
    let mut max_imag = 0.0_f64;

    for (row, expected_row) in expected.iter().enumerate() {
        for (col, &expected_value) in expected_row.iter().enumerate() {
            let (observed, imag) =
                decode_scalar(encrypted_product.get(row, col), &secret, &embedding);

            let error = (observed - expected_value).abs();
            max_error = max_error.max(error);
            max_imag = max_imag.max(imag);

            println!(
                "R3_2C_CELL_{}_{}_EXPECTED={:.12e} OBSERVED={:.12e} ABS_ERROR={:.12e}",
                row, col, expected_value, observed, error
            );
        }
    }

    println!("R3_2C_PRIVATE_TWO_PARTY_MATRIX_PRODUCT_VERSION=1");
    println!("APPLICATION=private-two-party-matrix-product");
    println!("PROFILE={}", profile.name());
    println!("RING_DEGREE={degree}");
    println!("SLOT_COUNT={}", profile.slot_count());
    println!(
        "UNDERLYING_RLWE_TARGET_SECURITY_BITS={}",
        security_model.classical_security_bits
    );
    println!("PARTY_A_ENCRYPTED_VALUES=4");
    println!("PARTY_B_ENCRYPTED_VALUES=4");
    println!("SCALAR_CKKS_MULTIPLIES=8");
    println!("CIPHERTEXT_ADDITIONS=4");
    println!("EVALUATION_KEY_REQUIRED=true");
    println!("BOUNDED_BASE_LOG={BASE_LOG}");
    println!("BOUNDED_BASE={}", layout.base());
    println!("BOUNDED_DIGITS={}", layout.digit_count());
    println!("CIPHERTEXT_ERROR_SIGMA={SIGMA}");
    println!("EVALUATION_KEY_ERROR_SIGMA={SIGMA}");
    println!("KEYGEN_US={keygen_us:.3}");
    println!("ENCRYPTION_8_VALUES_US={encryption_us:.3}");
    println!("PRIVATE_TWO_PARTY_PRODUCT_US={evaluation_us:.3}");
    println!("MAX_MATRIX_ERROR={max_error:.12e}");
    println!("MAX_IMAGINARY_RESIDUAL={max_imag:.12e}");
    println!("TOLERANCE={TOLERANCE:.12e}");
    println!("CIRCULAR_KDM_ASSUMPTION=required");
    println!("HARDENED_SAMPLER=false");

    assert!(
        max_error <= TOLERANCE,
        "private two-party matrix product exceeded tolerance: max_error={max_error:e}"
    );

    println!("R3_2C_PRIVATE_TWO_PARTY_MATRIX_PRODUCT_STATUS=PASS");
}
