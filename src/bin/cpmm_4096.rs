use std::time::Instant;

use ccmm_rs::ckks::{
    research_4096_security_model, research_profile_4096, CkksCanonicalEmbedding, CkksChainState,
    RnsCkksCiphertext,
};
use ccmm_rs::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng};
use ccmm_rs::matrix::{RnsCkksCiphertextMatrix, RnsCkksPlaintextMatrix};
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

fn encode_plain_scalar(
    value: f64,
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    let slots = vec![Complex64::new(value, 0.0); embedding.slot_count()];
    encode_rns(&slots, embedding, basis, scale)
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

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x320A_0000);
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

    // Column-major:
    // A = [[ 0.50, -0.25],
    //      [ 0.75,  0.125]]
    // U = [[ 0.25,  0.50],
    //      [-0.50,  0.25]]
    let lhs_values = [0.50, 0.75, -0.25, 0.125];
    let rhs_values = [0.25, -0.50, 0.50, 0.25];

    let encryption_start = Instant::now();
    let lhs = RnsCkksCiphertextMatrix::from_vec_column_major(
        2,
        2,
        lhs_values
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                encrypt_scalar(value, 0x320A_1000 + index as u64, &encryption_context)
            })
            .collect(),
    );
    let encryption_us = encryption_start.elapsed().as_secs_f64() * 1.0e6;

    let encoding_start = Instant::now();
    let rhs = RnsCkksPlaintextMatrix::from_vec_column_major(
        2,
        2,
        scale,
        rhs_values
            .iter()
            .map(|&value| encode_plain_scalar(value, &embedding, &basis, scale))
            .collect(),
    );
    let plaintext_encoding_us = encoding_start.elapsed().as_secs_f64() * 1.0e6;

    let cpmm_start = Instant::now();
    let result = lhs.matmul_plain_with_ntt(&rhs, &chain, &plan);
    let cpmm_us = cpmm_start.elapsed().as_secs_f64() * 1.0e6;

    assert_eq!(result.rows(), 2);
    assert_eq!(result.cols(), 2);
    assert_eq!(result.level(), 1);

    let expected = [[0.25, 0.1875], [0.125, 0.40625]];

    let mut max_error = 0.0_f64;
    let mut max_imag = 0.0_f64;

    for (row, expected_row) in expected.iter().enumerate() {
        for (col, &expected_value) in expected_row.iter().enumerate() {
            let (observed, imag) = decode_scalar(result.get(row, col), &secret, &embedding);

            let error = (observed - expected_value).abs();
            max_error = max_error.max(error);
            max_imag = max_imag.max(imag);

            println!(
                "R3_2A_CELL_{}_{}_EXPECTED={:.12e} OBSERVED={:.12e} ABS_ERROR={:.12e}",
                row, col, expected_value, observed, error
            );
        }
    }

    println!("R3_2A_CPMM_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("RING_DEGREE={degree}");
    println!("SLOT_COUNT={}", profile.slot_count());
    println!(
        "UNDERLYING_RLWE_TARGET_SECURITY_BITS={}",
        security_model.classical_security_bits
    );
    println!("MATRIX_ROWS=2");
    println!("MATRIX_INNER=2");
    println!("MATRIX_COLS=2");
    println!("CIPHERTEXT_PLAINTEXT_MULTIPLIES=8");
    println!("CIPHERTEXT_ADDITIONS=4");
    println!("OUTPUT_RESCALES=4");
    println!("EVALUATION_KEY_REQUIRED=false");
    println!("INPUT_CIPHERTEXT_SCALE={scale:.17e}");
    println!("INPUT_PLAINTEXT_SCALE={scale:.17e}");
    println!("OUTPUT_SCALE={:.17e}", result.scale());
    println!("CIPHERTEXT_ERROR_SIGMA={SIGMA}");
    println!("ENCRYPTION_4_CIPHERTEXTS_US={encryption_us:.3}");
    println!("PLAINTEXT_ENCODING_4_VALUES_US={plaintext_encoding_us:.3}");
    println!("CPMM_US={cpmm_us:.3}");
    println!("MAX_MATRIX_ERROR={max_error:.12e}");
    println!("MAX_IMAGINARY_RESIDUAL={max_imag:.12e}");
    println!("TOLERANCE={TOLERANCE:.12e}");
    println!("CIRCULAR_KDM_ASSUMPTION_FOR_CPMM=false");
    println!("HARDENED_SAMPLER=false");

    assert!(
        max_error <= TOLERANCE,
        "realistic 2x2 CPMM exceeded tolerance: max_error={max_error:e}"
    );

    println!("R3_2A_CPMM_STATUS=PASS");
}
