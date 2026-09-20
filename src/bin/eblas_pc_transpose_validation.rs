use ccmm_rs::ckks::{
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
};
use ccmm_rs::eblas::{
    gemm_cc, gemm_cp, gemm_pc, gemm_pp, GemmBackend, GemmShape, GemmSpec, MatrixShape, PrivacyMode,
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

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn main() {
    let m = 2;
    let k = 2;
    let n = 2;

    let lhs_values = vec![0.125, -0.0625, 0.03125, 0.09375];
    let rhs_values = vec![0.0625, 0.125, -0.03125, 0.046875];

    let pp_spec = GemmSpec::new(
        GemmShape::new(MatrixShape::new(m, k), MatrixShape::new(k, n)),
        PrivacyMode::Pp,
    );
    let cp_spec = GemmSpec::new(pp_spec.shape(), PrivacyMode::Cp);
    let pc_spec = GemmSpec::new(pp_spec.shape(), PrivacyMode::Pc);
    let cc_spec = GemmSpec::new(pp_spec.shape(), PrivacyMode::Cc);

    let lhs_pp = BatchMatrix::from_vec_column_major(m, k, 1, lhs_values.clone());
    let rhs_pp = BatchMatrix::from_vec_column_major(k, n, 1, rhs_values.clone());
    let reference = gemm_pp(pp_spec, &lhs_pp, &rhs_pp);
    let expected = reference.raw().to_vec();

    let profile = research_profile_4096();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let basis = chain.top().clone();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x35B0_0001);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&x| x == 0) {
        secret[0] = 1;
    }

    let layout = BoundedGadgetLayout::new(basis.clone(), BASE_LOG);
    let mut key_rng = ChaCha20Rng::seed_from_u64(0x35B0_0002);
    let multiplication_key = BoundedRnsMultiplicationKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout,
            plan: &plan,
        },
        &secret,
        distribution,
        &mut key_rng,
    );

    let lhs_ct = RnsCkksCiphertextMatrix::from_vec_column_major(
        m,
        k,
        lhs_values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                encrypt_scalar(
                    v,
                    0x35B0_1000 + i as u64,
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

    let rhs_plain = RnsCkksPlaintextMatrix::from_vec_column_major(
        k,
        n,
        scale,
        rhs_values
            .iter()
            .map(|&v| encode_rns(v, &embedding, &basis, scale))
            .collect(),
    );

    let rhs_ct = RnsCkksCiphertextMatrix::from_vec_column_major(
        k,
        n,
        rhs_values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                encrypt_scalar(
                    v,
                    0x35B0_2000 + i as u64,
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

    let cp = gemm_cp(cp_spec, &lhs_ct, &rhs_plain, &chain, &plan);
    let lhs_plain = RnsCkksPlaintextMatrix::from_vec_column_major(
        m,
        k,
        scale,
        lhs_values
            .iter()
            .map(|&v| encode_rns(v, &embedding, &basis, scale))
            .collect(),
    );
    let pc = gemm_pc(pc_spec, &lhs_plain, &rhs_ct, &chain, &plan);
    let cc_scalar = gemm_cc(
        cc_spec,
        GemmBackend::CcScalar,
        &lhs_ct,
        &rhs_ct,
        &multiplication_key,
        &chain,
        &plan,
    );
    let cc_structured = gemm_cc(
        cc_spec,
        GemmBackend::CcStructured,
        &lhs_ct,
        &rhs_ct,
        &multiplication_key,
        &chain,
        &plan,
    );

    let cp_values = decode_matrix(&cp, &secret, &embedding);
    let pc_values = decode_matrix(&pc, &secret, &embedding);
    let scalar_values = decode_matrix(&cc_scalar, &secret, &embedding);
    let structured_values = decode_matrix(&cc_structured, &secret, &embedding);

    let cp_error = max_error(&cp_values, &expected);
    let pc_error = max_error(&pc_values, &expected);
    let scalar_error = max_error(&scalar_values, &expected);
    let structured_error = max_error(&structured_values, &expected);
    let cross_error = max_error(&scalar_values, &structured_values);

    assert!(cp_error <= TOLERANCE);
    assert!(pc_error <= TOLERANCE);
    assert!(scalar_error <= TOLERANCE);
    assert!(structured_error <= TOLERANCE);

    println!("R3_5E_EBLAS_PC_TRANSPOSE_VALIDATION_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("M={m}");
    println!("K={k}");
    println!("N={n}");
    println!("PP_STATUS=PASS");
    println!("CP_MAX_ERROR={cp_error:.12e}");
    println!("CP_STATUS=PASS");
    println!("PC_MAX_ERROR={pc_error:.12e}");
    println!("PC_STATUS=PASS");
    println!("CC_SCALAR_MAX_ERROR={scalar_error:.12e}");
    println!("CC_SCALAR_STATUS=PASS");
    println!("CC_STRUCTURED_MAX_ERROR={structured_error:.12e}");
    println!("CC_STRUCTURED_STATUS=PASS");
    println!("CC_CROSS_PATH_MAX_ERROR={cross_error:.12e}");
    println!("R3_5E_EBLAS_PC_TRANSPOSE_STATUS=PASS");
}
