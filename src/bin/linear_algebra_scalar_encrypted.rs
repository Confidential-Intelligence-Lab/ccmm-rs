use std::env;
use std::time::Instant;

use ccmm_rs::ckks::{
    research_4096_security_model, research_profile_4096, CkksCanonicalEmbedding, CkksChainState,
    RnsCkksCiphertext,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    BoundedRnsKeygenConfig, BoundedRnsMultiplicationKey,
};
use ccmm_rs::matrix::{RnsCkksCiphertextMatrix, RnsCkksPlaintextMatrix};
use ccmm_rs::ring::{ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const VERSION: u32 = 1;
const SIGMA: f64 = 3.19;
const BASE_LOG: u32 = 20;
const TOLERANCE_CP: f64 = 2.0e-3;
const TOLERANCE_CC: f64 = 3.0e-3;

#[derive(Debug, Clone, Copy)]
struct Workload {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
}

fn workloads(profile: &str) -> Vec<Workload> {
    match profile {
        "small" => vec![
            Workload {
                name: "square-1",
                m: 1,
                k: 1,
                n: 1,
            },
            Workload {
                name: "square-2",
                m: 2,
                k: 2,
                n: 2,
            },
            Workload {
                name: "gemv-1x4-by-4x1",
                m: 1,
                k: 4,
                n: 1,
            },
            Workload {
                name: "rect-2x4-by-4x2",
                m: 2,
                k: 4,
                n: 2,
            },
        ],
        "frontier" => vec![
            Workload {
                name: "square-4",
                m: 4,
                k: 4,
                n: 4,
            },
            Workload {
                name: "gemv-1x8-by-8x1",
                m: 1,
                k: 8,
                n: 1,
            },
            Workload {
                name: "mlp-1x8-by-8x4",
                m: 1,
                k: 8,
                n: 4,
            },
            Workload {
                name: "mlp-1x16-by-16x8",
                m: 1,
                k: 16,
                n: 8,
            },
            Workload {
                name: "square-8",
                m: 8,
                k: 8,
                n: 8,
            },
        ],
        other => panic!("unknown profile {other}; expected small or frontier"),
    }
}

fn deterministic_value(index: usize, salt: u64) -> f64 {
    let mut x = index as u64 ^ salt;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    let mixed = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
    let signed = (mixed % 2001) as i64 - 1000;
    signed as f64 / 8192.0
}

fn cleartext_values(rows: usize, cols: usize, salt: u64) -> Vec<f64> {
    let mut values = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            values.push(deterministic_value(row + col * rows, salt));
        }
    }
    values
}

fn cleartext_matmul(lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; m * n];
    for col in 0..n {
        for row in 0..m {
            let mut sum = 0.0;
            for inner in 0..k {
                let lhs_value = lhs[row + inner * m];
                let rhs_value = rhs[inner + col * k];
                sum += lhs_value * rhs_value;
            }
            out[row + col * m] = sum;
        }
    }
    out
}

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
    let slots = vec![Complex64::new(value, 0.0); context.embedding.slot_count()];
    let plaintext = encode_rns(
        slots.as_slice(),
        context.embedding,
        context.basis,
        context.scale,
    );

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

fn decode_matrix(
    matrix: &RnsCkksCiphertextMatrix,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> (Vec<f64>, f64) {
    let mut values = Vec::with_capacity(matrix.rows() * matrix.cols());
    let mut max_imag = 0.0_f64;

    for col in 0..matrix.cols() {
        for row in 0..matrix.rows() {
            let (value, imag) = decode_scalar(matrix.get(row, col), secret, embedding);
            values.push(value);
            max_imag = max_imag.max(imag);
        }
    }

    (values, max_imag)
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max)
}

fn mean_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum::<f64>()
        / actual.len() as f64
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let workload_profile = args
        .windows(2)
        .find(|pair| pair[0] == "--profile")
        .map(|pair| pair[1].as_str())
        .unwrap_or("small");

    let profile = research_profile_4096();
    let security_model = research_4096_security_model();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let basis = chain.top().clone();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x34C1_0000);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let layout = BoundedGadgetLayout::new(basis.clone(), BASE_LOG);
    let keygen_start = Instant::now();
    let mut key_rng = ChaCha20Rng::seed_from_u64(0x34C1_1000);
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

    let context = EncryptionContext {
        embedding: &embedding,
        basis: &basis,
        scale,
        secret: &secret,
        plan: &plan,
        chain: &chain,
        distribution,
    };

    println!("R3_4C_SCALAR_ENCRYPTED_BASELINE_VERSION={VERSION}");
    println!("PROFILE={}", profile.name());
    println!("RING_DEGREE={degree}");
    println!("SLOT_COUNT={}", profile.slot_count());
    println!(
        "UNDERLYING_RLWE_TARGET_SECURITY_BITS={}",
        security_model.classical_security_bits
    );
    println!("CIPHERTEXT_ERROR_SIGMA={SIGMA}");
    println!("EVALUATION_KEY_ERROR_SIGMA={SIGMA}");
    println!("BOUNDED_BASE_LOG={BASE_LOG}");
    println!("BOUNDED_DIGITS={}", layout.digit_count());
    println!("KEYGEN_US={keygen_us:.3}");
    println!("BOOTSTRAPPING=none");
    println!("WORKLOAD_PROFILE={workload_profile}");

    for workload in workloads(workload_profile) {
        let lhs_values = cleartext_values(workload.m, workload.k, 0x34C1_2001);
        let rhs_values = cleartext_values(workload.k, workload.n, 0x34C1_3002);
        let expected =
            cleartext_matmul(&lhs_values, &rhs_values, workload.m, workload.k, workload.n);

        let lhs_encrypt_start = Instant::now();
        let lhs = RnsCkksCiphertextMatrix::from_vec_column_major(
            workload.m,
            workload.k,
            lhs_values
                .iter()
                .enumerate()
                .map(|(index, &value)| encrypt_scalar(value, 0x34C1_4000 + index as u64, &context))
                .collect(),
        );
        let lhs_encrypt_us = lhs_encrypt_start.elapsed().as_secs_f64() * 1.0e6;

        let rhs_plain_start = Instant::now();
        let rhs_plain = RnsCkksPlaintextMatrix::from_vec_column_major(
            workload.k,
            workload.n,
            scale,
            rhs_values
                .iter()
                .map(|&value| encode_plain_scalar(value, &embedding, &basis, scale))
                .collect(),
        );
        let rhs_plain_encode_us = rhs_plain_start.elapsed().as_secs_f64() * 1.0e6;

        let cp_start = Instant::now();
        let cp_result = lhs.matmul_plain_with_ntt(&rhs_plain, &chain, &plan);
        let cp_us = cp_start.elapsed().as_secs_f64() * 1.0e6;

        let cp_decode_start = Instant::now();
        let (cp_values, cp_max_imag) = decode_matrix(&cp_result, &secret, &embedding);
        let cp_decode_us = cp_decode_start.elapsed().as_secs_f64() * 1.0e6;
        let cp_max_error = max_error(&cp_values, &expected);
        let cp_mean_error = mean_error(&cp_values, &expected);

        let rhs_encrypt_start = Instant::now();
        let rhs_cipher = RnsCkksCiphertextMatrix::from_vec_column_major(
            workload.k,
            workload.n,
            rhs_values
                .iter()
                .enumerate()
                .map(|(index, &value)| encrypt_scalar(value, 0x34C1_8000 + index as u64, &context))
                .collect(),
        );
        let rhs_encrypt_us = rhs_encrypt_start.elapsed().as_secs_f64() * 1.0e6;

        let cc_start = Instant::now();
        let cc_result =
            lhs.matmul_bounded_with_ntt(&rhs_cipher, &multiplication_key, &chain, &plan);
        let cc_us = cc_start.elapsed().as_secs_f64() * 1.0e6;

        let cc_decode_start = Instant::now();
        let (cc_values, cc_max_imag) = decode_matrix(&cc_result, &secret, &embedding);
        let cc_decode_us = cc_decode_start.elapsed().as_secs_f64() * 1.0e6;
        let cc_max_error = max_error(&cc_values, &expected);
        let cc_mean_error = mean_error(&cc_values, &expected);

        let products = workload.m * workload.k * workload.n;
        let output_entries = workload.m * workload.n;
        let additions = workload.m * workload.n * workload.k.saturating_sub(1);

        assert!(
            cp_max_error <= TOLERANCE_CP,
            "{} scalar CP-GEMM exceeded tolerance: {}",
            workload.name,
            cp_max_error
        );
        assert!(
            cc_max_error <= TOLERANCE_CC,
            "{} scalar CC-GEMM exceeded tolerance: {}",
            workload.name,
            cc_max_error
        );

        println!();
        println!("WORKLOAD={}", workload.name);
        println!("M={}", workload.m);
        println!("K={}", workload.k);
        println!("N={}", workload.n);
        println!("OUTPUT_ENTRIES={output_entries}");
        println!("DOT_PRODUCTS={output_entries}");
        println!("INNER_PRODUCTS_PER_OUTPUT={}", workload.k);
        println!("SCALAR_PRODUCTS={products}");
        println!("CIPHERTEXT_ADDITIONS={additions}");

        println!("CP_LHS_CIPHERTEXTS={}", workload.m * workload.k);
        println!("CP_RHS_PLAINTEXTS={}", workload.k * workload.n);
        println!("CP_CT_PT_MULTIPLIES={products}");
        println!("CP_OUTPUT_RESCALES={output_entries}");
        println!("CP_EVALUATION_KEY_REQUIRED=false");
        println!("CP_LHS_ENCRYPT_US={lhs_encrypt_us:.3}");
        println!("CP_RHS_ENCODE_US={rhs_plain_encode_us:.3}");
        println!("CP_KERNEL_US={cp_us:.3}");
        println!("CP_DECODE_US={cp_decode_us:.3}");
        println!("CP_MAX_ERROR={cp_max_error:.12e}");
        println!("CP_MEAN_ERROR={cp_mean_error:.12e}");
        println!("CP_MAX_IMAGINARY_RESIDUAL={cp_max_imag:.12e}");
        println!("CP_STATUS=PASS");

        println!("CC_LHS_CIPHERTEXTS={}", workload.m * workload.k);
        println!("CC_RHS_CIPHERTEXTS={}", workload.k * workload.n);
        println!("CC_CT_CT_MULTIPLIES={products}");
        println!("CC_RELINEARIZATIONS={products}");
        println!("CC_RESCALES={products}");
        println!("CC_LHS_ENCRYPT_US={lhs_encrypt_us:.3}");
        println!("CC_RHS_ENCRYPT_US={rhs_encrypt_us:.3}");
        println!("CC_KERNEL_US={cc_us:.3}");
        println!("CC_DECODE_US={cc_decode_us:.3}");
        println!("CC_MAX_ERROR={cc_max_error:.12e}");
        println!("CC_MEAN_ERROR={cc_mean_error:.12e}");
        println!("CC_MAX_IMAGINARY_RESIDUAL={cc_max_imag:.12e}");
        println!("CC_STATUS=PASS");
    }

    println!();
    println!("R3_4C_SCALAR_ENCRYPTED_BASELINE_STATUS=PASS");
}
