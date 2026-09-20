use ccmm_rs::ckks::{
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
};
use ccmm_rs::eblas::{
    dot_cc, dot_cp, dot_pp, gemv_cc, gemv_cp, gemv_pp, DotShape, GemmBackend, GemvShape,
    MatrixShape,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    BoundedRnsKeygenConfig, BoundedRnsMultiplicationKey,
};
use ccmm_rs::matrix::{BatchMatrix, RnsCkksCiphertextMatrix, RnsCkksPlaintextMatrix};
use ccmm_rs::ring::{ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const SIGMA: f64 = 3.19;
const BASE_LOG: u32 = 20;
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

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn decode_matrix(
    matrix: &RnsCkksCiphertextMatrix,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(matrix.rows() * matrix.cols());
    for col in 0..matrix.cols() {
        for row in 0..matrix.rows() {
            out.push(decode_scalar(matrix.get(row, col), secret, embedding));
        }
    }
    out
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

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x35C0_0001);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&x| x == 0) {
        secret[0] = 1;
    }

    let layout = BoundedGadgetLayout::new(basis.clone(), BASE_LOG);
    let mut key_rng = ChaCha20Rng::seed_from_u64(0x35C0_0002);
    let key = BoundedRnsMultiplicationKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout,
            plan: &plan,
        },
        &secret,
        distribution,
        &mut key_rng,
    );

    let gemv_shape = GemvShape::new(MatrixShape::new(2, 3), MatrixShape::new(3, 1));
    let matrix_values = vec![0.125, -0.0625, 0.03125, 0.09375, -0.046875, 0.078125];
    let vector_values = vec![0.0625, -0.03125, 0.125];

    let matrix_pp = BatchMatrix::from_vec_column_major(2, 3, 1, matrix_values.clone());
    let vector_pp = BatchMatrix::from_vec_column_major(3, 1, 1, vector_values.clone());
    let gemv_ref = gemv_pp(gemv_shape, &matrix_pp, &vector_pp);
    let gemv_expected = gemv_ref.raw().to_vec();

    let matrix_ct = RnsCkksCiphertextMatrix::from_vec_column_major(
        2,
        3,
        matrix_values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                encrypt_scalar(
                    v,
                    0x35C0_1000 + i as u64,
                    &embedding,
                    &basis,
                    scale,
                    &secret,
                    &plan,
                    &chain,
                    distribution,
                )
            })
            .collect(),
    );

    let vector_pt = RnsCkksPlaintextMatrix::from_vec_column_major(
        3,
        1,
        scale,
        vector_values
            .iter()
            .map(|&v| encode_rns(v, &embedding, &basis, scale))
            .collect(),
    );

    let vector_ct = RnsCkksCiphertextMatrix::from_vec_column_major(
        3,
        1,
        vector_values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                encrypt_scalar(
                    v,
                    0x35C0_2000 + i as u64,
                    &embedding,
                    &basis,
                    scale,
                    &secret,
                    &plan,
                    &chain,
                    distribution,
                )
            })
            .collect(),
    );

    let gemv_cp_result = gemv_cp(gemv_shape, &matrix_ct, &vector_pt, &chain, &plan);
    let gemv_cc_scalar = gemv_cc(
        gemv_shape,
        GemmBackend::CcScalar,
        &matrix_ct,
        &vector_ct,
        &key,
        &chain,
        &plan,
    );
    let gemv_cc_structured = gemv_cc(
        gemv_shape,
        GemmBackend::CcStructured,
        &matrix_ct,
        &vector_ct,
        &key,
        &chain,
        &plan,
    );

    let gemv_cp_error = max_error(
        &decode_matrix(&gemv_cp_result, &secret, &embedding),
        &gemv_expected,
    );
    let gemv_scalar_error = max_error(
        &decode_matrix(&gemv_cc_scalar, &secret, &embedding),
        &gemv_expected,
    );
    let gemv_structured_error = max_error(
        &decode_matrix(&gemv_cc_structured, &secret, &embedding),
        &gemv_expected,
    );

    let dot_shape = DotShape::new(4);
    let lhs_values = vec![0.125, -0.0625, 0.03125, 0.09375];
    let rhs_values = vec![0.0625, 0.125, -0.03125, 0.046875];

    let lhs_pp = BatchMatrix::from_vec_column_major(1, 4, 1, lhs_values.clone());
    let rhs_pp = BatchMatrix::from_vec_column_major(4, 1, 1, rhs_values.clone());
    let dot_ref = dot_pp(dot_shape, &lhs_pp, &rhs_pp);
    let dot_expected = dot_ref.raw().to_vec();

    let lhs_ct = RnsCkksCiphertextMatrix::from_vec_column_major(
        1,
        4,
        lhs_values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                encrypt_scalar(
                    v,
                    0x35C0_3000 + i as u64,
                    &embedding,
                    &basis,
                    scale,
                    &secret,
                    &plan,
                    &chain,
                    distribution,
                )
            })
            .collect(),
    );

    let rhs_pt = RnsCkksPlaintextMatrix::from_vec_column_major(
        4,
        1,
        scale,
        rhs_values
            .iter()
            .map(|&v| encode_rns(v, &embedding, &basis, scale))
            .collect(),
    );

    let rhs_ct = RnsCkksCiphertextMatrix::from_vec_column_major(
        4,
        1,
        rhs_values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                encrypt_scalar(
                    v,
                    0x35C0_4000 + i as u64,
                    &embedding,
                    &basis,
                    scale,
                    &secret,
                    &plan,
                    &chain,
                    distribution,
                )
            })
            .collect(),
    );

    let dot_cp_result = dot_cp(dot_shape, &lhs_ct, &rhs_pt, &chain, &plan);
    let dot_cc_scalar = dot_cc(
        dot_shape,
        GemmBackend::CcScalar,
        &lhs_ct,
        &rhs_ct,
        &key,
        &chain,
        &plan,
    );
    let dot_cc_structured = dot_cc(
        dot_shape,
        GemmBackend::CcStructured,
        &lhs_ct,
        &rhs_ct,
        &key,
        &chain,
        &plan,
    );

    let dot_cp_error = max_error(
        &decode_matrix(&dot_cp_result, &secret, &embedding),
        &dot_expected,
    );
    let dot_scalar_error = max_error(
        &decode_matrix(&dot_cc_scalar, &secret, &embedding),
        &dot_expected,
    );
    let dot_structured_error = max_error(
        &decode_matrix(&dot_cc_structured, &secret, &embedding),
        &dot_expected,
    );

    for error in [
        gemv_cp_error,
        gemv_scalar_error,
        gemv_structured_error,
        dot_cp_error,
        dot_scalar_error,
        dot_structured_error,
    ] {
        assert!(error <= TOLERANCE);
    }

    println!("R3_5C_EBLAS_GEMV_DOT_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("GEMV_M=2");
    println!("GEMV_K=3");
    println!("GEMV_CP_MAX_ERROR={gemv_cp_error:.12e}");
    println!("GEMV_CC_SCALAR_MAX_ERROR={gemv_scalar_error:.12e}");
    println!("GEMV_CC_STRUCTURED_MAX_ERROR={gemv_structured_error:.12e}");
    println!("GEMV_STATUS=PASS");
    println!("DOT_K=4");
    println!("DOT_CP_MAX_ERROR={dot_cp_error:.12e}");
    println!("DOT_CC_SCALAR_MAX_ERROR={dot_scalar_error:.12e}");
    println!("DOT_CC_STRUCTURED_MAX_ERROR={dot_structured_error:.12e}");
    println!("DOT_STATUS=PASS");
    println!("R3_5C_EBLAS_GEMV_DOT_STATUS=PASS");
}
