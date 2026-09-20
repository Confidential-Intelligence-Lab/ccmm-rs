use ccmm_rs::ckks::{
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
};
use ccmm_rs::eblas::{
    batched_gemm_cc, batched_gemm_cp, batched_gemm_pc, batched_gemm_pp, BatchedGemmShape,
    BatchedGemmSpec, GemmBackend, GemmShape, MatrixShape, PrivacyMode,
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

fn slice_batch(matrix: &BatchMatrix<f64>, batch: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(matrix.rows() * matrix.cols());
    for col in 0..matrix.cols() {
        for row in 0..matrix.rows() {
            out.push(*matrix.get(batch, row, col));
        }
    }
    out
}

fn main() {
    let batches = 3;
    let m = 2;
    let k = 3;
    let n = 4;

    let lhs_values = vec![
        // batch 0
        0.125, -0.0625, 0.03125, 0.09375, -0.046875, 0.015625, // batch 1
        -0.09375, 0.046875, 0.0625, -0.03125, 0.015625, 0.109375, // batch 2
        0.078125, 0.0234375, -0.0546875, 0.0390625, 0.1015625, -0.0703125,
    ];

    let rhs_values = vec![
        // batch 0
        0.0625, -0.03125, 0.015625, 0.125, 0.046875, -0.0625, -0.03125, 0.078125, 0.09375, 0.046875,
        -0.015625, 0.109375, // batch 1
        -0.046875, 0.03125, 0.078125, 0.09375, -0.015625, 0.0625, 0.015625, 0.109375, -0.03125,
        -0.078125, 0.046875, 0.125, // batch 2
        0.109375, -0.0625, 0.0234375, -0.03125, 0.0859375, 0.046875, 0.0703125, -0.0390625,
        0.1171875, 0.0546875, 0.015625, -0.09375,
    ];

    let gemm_shape = GemmShape::new(MatrixShape::new(m, k), MatrixShape::new(k, n));
    let batch_shape = BatchedGemmShape::new(batches, gemm_shape);

    let pp_spec = BatchedGemmSpec::new(batch_shape, PrivacyMode::Pp);
    let cp_spec = BatchedGemmSpec::new(batch_shape, PrivacyMode::Cp);
    let pc_spec = BatchedGemmSpec::new(batch_shape, PrivacyMode::Pc);
    let cc_spec = BatchedGemmSpec::new(batch_shape, PrivacyMode::Cc);

    let lhs_pp = BatchMatrix::from_vec_column_major(m, k, batches, lhs_values.clone());
    let rhs_pp = BatchMatrix::from_vec_column_major(k, n, batches, rhs_values.clone());
    let reference = batched_gemm_pp(pp_spec, &lhs_pp, &rhs_pp);

    let profile = research_profile_4096();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let basis = chain.top().clone();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x35F0_0001);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&x| x == 0) {
        secret[0] = 1;
    }

    let layout = BoundedGadgetLayout::new(basis.clone(), BASE_LOG);
    let mut key_rng = ChaCha20Rng::seed_from_u64(0x35F0_0002);
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

    let mut lhs_ct = Vec::with_capacity(batches);
    let mut rhs_ct = Vec::with_capacity(batches);
    let mut lhs_plain = Vec::with_capacity(batches);
    let mut rhs_plain = Vec::with_capacity(batches);

    for batch in 0..batches {
        let lhs_batch = slice_batch(&lhs_pp, batch);
        let rhs_batch = slice_batch(&rhs_pp, batch);

        lhs_plain.push(RnsCkksPlaintextMatrix::from_vec_column_major(
            m,
            k,
            scale,
            lhs_batch
                .iter()
                .map(|&v| encode_rns(v, &embedding, &basis, scale))
                .collect(),
        ));

        rhs_plain.push(RnsCkksPlaintextMatrix::from_vec_column_major(
            k,
            n,
            scale,
            rhs_batch
                .iter()
                .map(|&v| encode_rns(v, &embedding, &basis, scale))
                .collect(),
        ));

        lhs_ct.push(RnsCkksCiphertextMatrix::from_vec_column_major(
            m,
            k,
            lhs_batch
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    encrypt_scalar(
                        v,
                        0x35F1_0000 + (batch * 100 + i) as u64,
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
        ));

        rhs_ct.push(RnsCkksCiphertextMatrix::from_vec_column_major(
            k,
            n,
            rhs_batch
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    encrypt_scalar(
                        v,
                        0x35F2_0000 + (batch * 100 + i) as u64,
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
        ));
    }

    let cp = batched_gemm_cp(cp_spec, &lhs_ct, &rhs_plain, &chain, &plan);
    let pc = batched_gemm_pc(pc_spec, &lhs_plain, &rhs_ct, &chain, &plan);
    let cc_scalar = batched_gemm_cc(
        cc_spec,
        GemmBackend::CcScalar,
        &lhs_ct,
        &rhs_ct,
        &multiplication_key,
        &chain,
        &plan,
    );
    let cc_structured = batched_gemm_cc(
        cc_spec,
        GemmBackend::CcStructured,
        &lhs_ct,
        &rhs_ct,
        &multiplication_key,
        &chain,
        &plan,
    );

    let mut max_cp_error = 0.0_f64;
    let mut max_pc_error = 0.0_f64;
    let mut max_cc_scalar_error = 0.0_f64;
    let mut max_cc_structured_error = 0.0_f64;
    let mut max_cc_cross_error = 0.0_f64;

    for batch in 0..batches {
        let expected = slice_batch(&reference, batch);

        let cp_values = decode_matrix(&cp[batch], &secret, &embedding);
        let pc_values = decode_matrix(&pc[batch], &secret, &embedding);
        let scalar_values = decode_matrix(&cc_scalar[batch], &secret, &embedding);
        let structured_values = decode_matrix(&cc_structured[batch], &secret, &embedding);

        let cp_error = max_error(&cp_values, &expected);
        let pc_error = max_error(&pc_values, &expected);
        let scalar_error = max_error(&scalar_values, &expected);
        let structured_error = max_error(&structured_values, &expected);
        let cc_cross = max_error(&scalar_values, &structured_values);

        max_cp_error = max_cp_error.max(cp_error);
        max_pc_error = max_pc_error.max(pc_error);
        max_cc_scalar_error = max_cc_scalar_error.max(scalar_error);
        max_cc_structured_error = max_cc_structured_error.max(structured_error);
        max_cc_cross_error = max_cc_cross_error.max(cc_cross);

        println!("BATCH_{batch}_CP_MAX_ERROR={cp_error:.12e}");
        println!("BATCH_{batch}_PC_MAX_ERROR={pc_error:.12e}");
        println!("BATCH_{batch}_CC_SCALAR_MAX_ERROR={scalar_error:.12e}");
        println!("BATCH_{batch}_CC_STRUCTURED_MAX_ERROR={structured_error:.12e}");
        println!("BATCH_{batch}_CC_CROSS_MAX_ERROR={cc_cross:.12e}");

        assert_eq!(cp[batch].level(), lhs_ct[batch].level() + 1);
        assert_eq!(pc[batch].level(), rhs_ct[batch].level() + 1);
        assert_eq!(cc_scalar[batch].level(), lhs_ct[batch].level() + 1);
        assert_eq!(cc_structured[batch].level(), lhs_ct[batch].level() + 1);

        // Batching must preserve the scale semantics of the underlying
        // single-GEMM kernels. The post-rescale scale need not equal the
        // incoming scale exactly because it depends on the dropped modulus.
        let cp_scale = cp[batch].scale();
        let pc_scale = pc[batch].scale();
        let cc_scalar_scale = cc_scalar[batch].scale();
        let cc_structured_scale = cc_structured[batch].scale();

        assert!(cp_scale.is_finite() && cp_scale > 0.0);
        assert!(pc_scale.is_finite() && pc_scale > 0.0);
        assert!(cc_scalar_scale.is_finite() && cc_scalar_scale > 0.0);
        assert!(cc_structured_scale.is_finite() && cc_structured_scale > 0.0);

        assert!(((pc_scale - cp_scale) / cp_scale).abs() <= 1.0e-12);
        assert!(((cc_scalar_scale - cp_scale) / cp_scale).abs() <= 1.0e-12);
        assert!(((cc_structured_scale - cp_scale) / cp_scale).abs() <= 1.0e-12);
    }

    assert!(max_cp_error <= TOLERANCE);
    assert!(max_pc_error <= TOLERANCE);
    assert!(max_cc_scalar_error <= TOLERANCE);
    assert!(max_cc_structured_error <= TOLERANCE);

    println!("R3_5F_EBLAS_BATCHED_GEMM_VALIDATION_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("BATCHES={batches}");
    println!("M={m}");
    println!("K={k}");
    println!("N={n}");
    println!("CP_MAX_ERROR={max_cp_error:.12e}");
    println!("CP_STATUS=PASS");
    println!("PC_MAX_ERROR={max_pc_error:.12e}");
    println!("PC_STATUS=PASS");
    println!("CC_SCALAR_MAX_ERROR={max_cc_scalar_error:.12e}");
    println!("CC_SCALAR_STATUS=PASS");
    println!("CC_STRUCTURED_MAX_ERROR={max_cc_structured_error:.12e}");
    println!("CC_STRUCTURED_STATUS=PASS");
    println!("CC_CROSS_PATH_MAX_ERROR={max_cc_cross_error:.12e}");
    println!("LEVEL_TRANSITION=0_TO_1");
    println!("CROSS_PATH_SCALE_ALIGNMENT_STATUS=PASS");
    println!("BATCH_ISOLATION_STATUS=PASS");
    println!("R3_5F_EBLAS_BATCHED_GEMM_STATUS=PASS");
}
