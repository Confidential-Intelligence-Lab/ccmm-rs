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
use ccmm_rs::matrix::RnsCkksCiphertextMatrix;
use ccmm_rs::ring::{ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const SIGMA: f64 = 3.19;
const BASE_LOG: u32 = 20;
const TOLERANCE: f64 = 3.0e-3;

#[derive(Clone, Copy)]
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
    (0..rows * cols)
        .map(|index| deterministic_value(index, salt))
        .collect()
}

fn cleartext_matmul(lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for col in 0..n {
        for row in 0..m {
            for inner in 0..k {
                out[row + col * m] += lhs[row + inner * m] * rhs[inner + col * k];
            }
        }
    }
    out
}

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

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x34D1_0000);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&x| x == 0) {
        secret[0] = 1;
    }

    let layout = BoundedGadgetLayout::new(basis.clone(), BASE_LOG);
    let mut key_rng = ChaCha20Rng::seed_from_u64(0x34D1_1000);
    let key = BoundedRnsMultiplicationKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout: layout.clone(),
            plan: &plan,
        },
        &secret,
        distribution,
        &mut key_rng,
    );

    println!("R3_4D_STRUCTURED_CCMM_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("RING_DEGREE={degree}");
    println!(
        "UNDERLYING_RLWE_TARGET_SECURITY_BITS={}",
        security_model.classical_security_bits
    );
    println!("CIPHERTEXT_ERROR_SIGMA={SIGMA}");
    println!("EVALUATION_KEY_ERROR_SIGMA={SIGMA}");
    println!("BOUNDED_BASE_LOG={BASE_LOG}");
    println!("BOOTSTRAPPING=none");
    println!("WORKLOAD_PROFILE={workload_profile}");

    for workload in workloads(workload_profile) {
        let lhs_values = cleartext_values(workload.m, workload.k, 0x34D1_2001);
        let rhs_values = cleartext_values(workload.k, workload.n, 0x34D1_3002);
        let expected =
            cleartext_matmul(&lhs_values, &rhs_values, workload.m, workload.k, workload.n);

        let lhs = RnsCkksCiphertextMatrix::from_vec_column_major(
            workload.m,
            workload.k,
            lhs_values
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    encrypt_scalar(
                        v,
                        0x34D1_4000 + i as u64,
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

        let rhs = RnsCkksCiphertextMatrix::from_vec_column_major(
            workload.k,
            workload.n,
            rhs_values
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    encrypt_scalar(
                        v,
                        0x34D1_8000 + i as u64,
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

        let start = Instant::now();
        let scalar = lhs.matmul_bounded_with_ntt(&rhs, &key, &chain, &plan);
        let scalar_us = start.elapsed().as_secs_f64() * 1.0e6;

        let start = Instant::now();
        let structured = lhs.matmul_structured_bounded_with_ntt(&rhs, &key, &chain, &plan);
        let structured_us = start.elapsed().as_secs_f64() * 1.0e6;

        let scalar_values = decode_matrix(&scalar, &secret, &embedding);
        let structured_values = decode_matrix(&structured, &secret, &embedding);

        let scalar_error = max_error(&scalar_values, &expected);
        let structured_error = max_error(&structured_values, &expected);
        let cross_error = max_error(&scalar_values, &structured_values);

        assert!(scalar_error <= TOLERANCE);
        assert!(structured_error <= TOLERANCE);

        let products = workload.m * workload.k * workload.n;
        let outputs = workload.m * workload.n;

        println!();
        println!("WORKLOAD={}", workload.name);
        println!("M={}", workload.m);
        println!("K={}", workload.k);
        println!("N={}", workload.n);
        println!("SCALAR_PRODUCTS={products}");
        println!("OUTPUT_ENTRIES={outputs}");
        println!("SCALAR_RELINEARIZATIONS={products}");
        println!("STRUCTURED_RELINEARIZATIONS={outputs}");
        println!("SCALAR_RESCALES={products}");
        println!("STRUCTURED_RESCALES={outputs}");
        println!(
            "RELINEARIZATION_REDUCTION_FACTOR={:.6}",
            products as f64 / outputs as f64
        );
        println!("SCALAR_CC_KERNEL_US={scalar_us:.3}");
        println!("STRUCTURED_CCMM_KERNEL_US={structured_us:.3}");
        println!("STRUCTURED_SPEEDUP_X={:.6}", scalar_us / structured_us);
        println!("SCALAR_MAX_ERROR={scalar_error:.12e}");
        println!("STRUCTURED_MAX_ERROR={structured_error:.12e}");
        println!("CROSS_PATH_MAX_ERROR={cross_error:.12e}");
        println!("STATUS=PASS");
    }

    println!();
    println!("R3_4D_STRUCTURED_CCMM_STATUS=PASS");
}
