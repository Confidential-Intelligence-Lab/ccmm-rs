use ccmm_rs::ckks::{
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
};
use ccmm_rs::eblas::{
    batch_matrix_to_tensor3, batched_gemm_cc, batched_gemm_cp, batched_gemm_pc, batched_gemm_pp,
    tensor3_to_batch_matrix, tensor_gemm_spec, GemmBackend, PrivacyMode, TensorBatchShape,
    TensorGemmShape,
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

fn extract_batch(matrix: &BatchMatrix<f64>, batch: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(matrix.rows() * matrix.cols());
    for col in 0..matrix.cols() {
        for row in 0..matrix.rows() {
            out.push(*matrix.get(batch, row, col));
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
    let lhs_shape = TensorBatchShape::new(3, 2, 3);
    let rhs_shape = TensorBatchShape::new(3, 3, 4);
    let tensor_shape = TensorGemmShape::new(lhs_shape, rhs_shape);
    let output_shape = tensor_shape.output();

    let lhs_values = vec![
        0.125, -0.0625, 0.03125, 0.09375, -0.046875, 0.015625, -0.09375, 0.046875, 0.0625,
        -0.03125, 0.015625, 0.109375, 0.078125, 0.0234375, -0.0546875, 0.0390625, 0.1015625,
        -0.0703125,
    ];

    let rhs_values = vec![
        0.0625, -0.03125, 0.015625, 0.125, 0.046875, -0.0625, -0.03125, 0.078125, 0.09375,
        0.046875, -0.015625, 0.109375, -0.046875, 0.03125, 0.078125, 0.09375, -0.015625, 0.0625,
        0.015625, 0.109375, -0.03125, -0.078125, 0.046875, 0.125, 0.109375, -0.0625, 0.0234375,
        -0.03125, 0.0859375, 0.046875, 0.0703125, -0.0390625, 0.1171875, 0.0546875, 0.015625,
        -0.09375,
    ];

    let lhs_pp = tensor3_to_batch_matrix(lhs_shape, lhs_values.clone());
    let rhs_pp = tensor3_to_batch_matrix(rhs_shape, rhs_values.clone());

    let (lhs_round_shape, lhs_round_values) = batch_matrix_to_tensor3(&lhs_pp);
    let (rhs_round_shape, rhs_round_values) = batch_matrix_to_tensor3(&rhs_pp);
    assert_eq!(lhs_round_shape, lhs_shape);
    assert_eq!(rhs_round_shape, rhs_shape);
    assert_eq!(lhs_round_values, lhs_values);
    assert_eq!(rhs_round_values, rhs_values);

    let pp_spec = tensor_gemm_spec(tensor_shape, PrivacyMode::Pp);
    let cp_spec = tensor_gemm_spec(tensor_shape, PrivacyMode::Cp);
    let pc_spec = tensor_gemm_spec(tensor_shape, PrivacyMode::Pc);
    let cc_spec = tensor_gemm_spec(tensor_shape, PrivacyMode::Cc);

    let reference = batched_gemm_pp(pp_spec, &lhs_pp, &rhs_pp);
    let (reference_shape, reference_values) = batch_matrix_to_tensor3(&reference);
    assert_eq!(reference_shape, output_shape);
    assert_eq!(reference_values.len(), output_shape.elements());

    let profile = research_profile_4096();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let basis = chain.top().clone();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x35A2_0001);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&x| x == 0) {
        secret[0] = 1;
    }

    let layout = BoundedGadgetLayout::new(basis.clone(), BASE_LOG);
    let mut key_rng = ChaCha20Rng::seed_from_u64(0x35A2_0002);
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

    let mut lhs_ct = Vec::with_capacity(lhs_shape.batches());
    let mut rhs_ct = Vec::with_capacity(rhs_shape.batches());
    let mut lhs_plain = Vec::with_capacity(lhs_shape.batches());
    let mut rhs_plain = Vec::with_capacity(rhs_shape.batches());

    for batch in 0..lhs_shape.batches() {
        let lhs_batch = extract_batch(&lhs_pp, batch);
        let rhs_batch = extract_batch(&rhs_pp, batch);

        lhs_plain.push(RnsCkksPlaintextMatrix::from_vec_column_major(
            lhs_shape.rows(),
            lhs_shape.cols(),
            scale,
            lhs_batch
                .iter()
                .map(|&v| encode_rns(v, &embedding, &basis, scale))
                .collect(),
        ));
        rhs_plain.push(RnsCkksPlaintextMatrix::from_vec_column_major(
            rhs_shape.rows(),
            rhs_shape.cols(),
            scale,
            rhs_batch
                .iter()
                .map(|&v| encode_rns(v, &embedding, &basis, scale))
                .collect(),
        ));

        lhs_ct.push(RnsCkksCiphertextMatrix::from_vec_column_major(
            lhs_shape.rows(),
            lhs_shape.cols(),
            lhs_batch
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    encrypt_scalar(
                        v,
                        0x35A3_0000 + (batch * 100 + i) as u64,
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
            rhs_shape.rows(),
            rhs_shape.cols(),
            rhs_batch
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    encrypt_scalar(
                        v,
                        0x35A4_0000 + (batch * 100 + i) as u64,
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

    let mut cp_tensor = Vec::with_capacity(output_shape.elements());
    let mut pc_tensor = Vec::with_capacity(output_shape.elements());
    let mut cc_scalar_tensor = Vec::with_capacity(output_shape.elements());
    let mut cc_structured_tensor = Vec::with_capacity(output_shape.elements());

    for batch in 0..output_shape.batches() {
        cp_tensor.extend(decode_matrix(&cp[batch], &secret, &embedding));
        pc_tensor.extend(decode_matrix(&pc[batch], &secret, &embedding));
        cc_scalar_tensor.extend(decode_matrix(&cc_scalar[batch], &secret, &embedding));
        cc_structured_tensor.extend(decode_matrix(&cc_structured[batch], &secret, &embedding));

        assert_eq!(cp[batch].level(), lhs_ct[batch].level() + 1);
        assert_eq!(pc[batch].level(), rhs_ct[batch].level() + 1);
        assert_eq!(cc_scalar[batch].level(), lhs_ct[batch].level() + 1);
        assert_eq!(cc_structured[batch].level(), lhs_ct[batch].level() + 1);

        let cp_scale = cp[batch].scale();
        assert!(((pc[batch].scale() - cp_scale) / cp_scale).abs() <= 1.0e-12);
        assert!(((cc_scalar[batch].scale() - cp_scale) / cp_scale).abs() <= 1.0e-12);
        assert!(((cc_structured[batch].scale() - cp_scale) / cp_scale).abs() <= 1.0e-12);
    }

    let cp_error = max_error(&cp_tensor, &reference_values);
    let pc_error = max_error(&pc_tensor, &reference_values);
    let cc_scalar_error = max_error(&cc_scalar_tensor, &reference_values);
    let cc_structured_error = max_error(&cc_structured_tensor, &reference_values);
    let cc_cross_error = max_error(&cc_scalar_tensor, &cc_structured_tensor);

    assert!(cp_error <= TOLERANCE);
    assert!(pc_error <= TOLERANCE);
    assert!(cc_scalar_error <= TOLERANCE);
    assert!(cc_structured_error <= TOLERANCE);

    println!("R3_5G2_EBLAS_TENSOR_GEMM_VALIDATION_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!(
        "LHS_SHAPE=[{},{},{}]",
        lhs_shape.batches(),
        lhs_shape.rows(),
        lhs_shape.cols()
    );
    println!(
        "RHS_SHAPE=[{},{},{}]",
        rhs_shape.batches(),
        rhs_shape.rows(),
        rhs_shape.cols()
    );
    println!(
        "OUTPUT_SHAPE=[{},{},{}]",
        output_shape.batches(),
        output_shape.rows(),
        output_shape.cols()
    );
    println!("TENSOR_ROUNDTRIP_STATUS=PASS");
    println!("CP_MAX_ERROR={cp_error:.12e}");
    println!("CP_STATUS=PASS");
    println!("PC_MAX_ERROR={pc_error:.12e}");
    println!("PC_STATUS=PASS");
    println!("CC_SCALAR_MAX_ERROR={cc_scalar_error:.12e}");
    println!("CC_SCALAR_STATUS=PASS");
    println!("CC_STRUCTURED_MAX_ERROR={cc_structured_error:.12e}");
    println!("CC_STRUCTURED_STATUS=PASS");
    println!("CC_CROSS_PATH_MAX_ERROR={cc_cross_error:.12e}");
    println!("LEVEL_TRANSITION=0_TO_1");
    println!("CROSS_PATH_SCALE_ALIGNMENT_STATUS=PASS");
    println!("TENSOR_BATCH_ISOLATION_STATUS=PASS");
    println!("SIMD_PACKING_USED=NO");
    println!("NEW_CRYPTO_KERNEL_USED=NO");
    println!("R3_5G2_EBLAS_TENSOR_GEMM_STATUS=PASS");
}
