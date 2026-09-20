//! R3.5h2 eBLAS batch-scaling and PC-overhead characterization.
//!
//! Setup is excluded from timed regions. Batch timing covers the eBLAS batched
//! GEMM call; PC timing covers the complete eBLAS PC call including its
//! representation transposes.

use ccmm_rs::ckks::{
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
};
use ccmm_rs::eblas::{
    batched_gemm_cc, batched_gemm_cp, gemm_cp, gemm_pc, BatchedGemmOperationCount,
    BatchedGemmShape, BatchedGemmSpec, GemmBackend, GemmShape, GemmSpec, MatrixShape, PrivacyMode,
};
use ccmm_rs::grafting::{
    encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout, BoundedRnsKeygenConfig,
    BoundedRnsMultiplicationKey,
};
use ccmm_rs::matrix::{RnsCkksCiphertextMatrix, RnsCkksPlaintextMatrix};
use ccmm_rs::ring::{ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::hint::black_box;
use std::time::Instant;

const SIGMA: f64 = 3.19;
const BASE_LOG: u32 = 20;
const WARMUPS: usize = 1;
const REPEATS: usize = 3;

#[derive(Clone, Copy)]
struct Timing {
    median_us: f64,
    min_us: f64,
    max_us: f64,
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

fn values(rows: usize, cols: usize, salt: u64) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            out.push(deterministic_value(row + col * rows, salt));
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

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn benchmark<F, T>(mut f: F) -> Timing
where
    F: FnMut() -> T,
{
    for _ in 0..WARMUPS {
        black_box(f());
    }

    let mut samples = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let start = Instant::now();
        black_box(f());
        samples.push(start.elapsed().as_secs_f64() * 1.0e6);
    }

    let min_us = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max_us = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let median_us = median(samples);

    Timing {
        median_us,
        min_us,
        max_us,
    }
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

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x35A8_B001);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&x| x == 0) {
        secret[0] = 1;
    }

    let layout = BoundedGadgetLayout::new(basis.clone(), BASE_LOG);
    let mut key_rng = ChaCha20Rng::seed_from_u64(0x35A8_B002);
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

    println!("R3_5H2_EBLAS_BATCH_PC_CHARACTERIZATION_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("DEGREE={degree}");
    println!("WARMUPS={WARMUPS}");
    println!("REPEATS={REPEATS}");
    println!("TIMING_STATISTIC=median");
    println!("SETUP_INCLUDED_IN_TIMING=NO");

    // Batch scaling: B independent 2x3 * 3x2 GEMMs.
    let m = 2;
    let k = 3;
    let n = 2;
    let gemm_shape = GemmShape::new(MatrixShape::new(m, k), MatrixShape::new(k, n));

    for &batches in &[1_usize, 2, 4, 8] {
        let mut lhs_ct = Vec::with_capacity(batches);
        let mut rhs_ct = Vec::with_capacity(batches);
        let mut rhs_pt = Vec::with_capacity(batches);

        for batch in 0..batches {
            let lhs_values = values(m, k, 0x35A8_C000 + batch as u64);
            let rhs_values = values(k, n, 0x35A8_D000 + batch as u64);

            lhs_ct.push(RnsCkksCiphertextMatrix::from_vec_column_major(
                m,
                k,
                lhs_values
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        encrypt_scalar(
                            v,
                            0x35A8_E000 + (batch * 100 + i) as u64,
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

            rhs_pt.push(RnsCkksPlaintextMatrix::from_vec_column_major(
                k,
                n,
                scale,
                rhs_values
                    .iter()
                    .map(|&v| encode_rns(v, &embedding, &basis, scale))
                    .collect(),
            ));

            rhs_ct.push(RnsCkksCiphertextMatrix::from_vec_column_major(
                k,
                n,
                rhs_values
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        encrypt_scalar(
                            v,
                            0x35A8_F000 + (batch * 100 + i) as u64,
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

        let batch_shape = BatchedGemmShape::new(batches, gemm_shape);
        let cp_spec = BatchedGemmSpec::new(batch_shape, PrivacyMode::Cp);
        let cc_spec = BatchedGemmSpec::new(batch_shape, PrivacyMode::Cc);

        let cp_timing = benchmark(|| batched_gemm_cp(cp_spec, &lhs_ct, &rhs_pt, &chain, &plan));
        let scalar_timing = benchmark(|| {
            batched_gemm_cc(
                cc_spec,
                GemmBackend::CcScalar,
                &lhs_ct,
                &rhs_ct,
                &multiplication_key,
                &chain,
                &plan,
            )
        });
        let structured_timing = benchmark(|| {
            batched_gemm_cc(
                cc_spec,
                GemmBackend::CcStructured,
                &lhs_ct,
                &rhs_ct,
                &multiplication_key,
                &chain,
                &plan,
            )
        });

        let cp_count = BatchedGemmOperationCount::for_backend(cp_spec, GemmBackend::CpDirect);
        let scalar_count = BatchedGemmOperationCount::for_backend(cc_spec, GemmBackend::CcScalar);
        let structured_count =
            BatchedGemmOperationCount::for_backend(cc_spec, GemmBackend::CcStructured);

        println!(
            "BATCH_RESULT,{batches},Cp,CpDirect,{:.3},{:.3},{:.3},{:.3},{},{},{},{}",
            cp_timing.median_us,
            cp_timing.min_us,
            cp_timing.max_us,
            cp_timing.median_us / batches as f64,
            cp_count.scalar_products,
            cp_count.additions,
            cp_count.relinearizations,
            cp_count.rescales
        );
        println!(
            "BATCH_RESULT,{batches},Cc,CcScalar,{:.3},{:.3},{:.3},{:.3},{},{},{},{}",
            scalar_timing.median_us,
            scalar_timing.min_us,
            scalar_timing.max_us,
            scalar_timing.median_us / batches as f64,
            scalar_count.scalar_products,
            scalar_count.additions,
            scalar_count.relinearizations,
            scalar_count.rescales
        );
        println!(
            "BATCH_RESULT,{batches},Cc,CcStructured,{:.3},{:.3},{:.3},{:.3},{},{},{},{}",
            structured_timing.median_us,
            structured_timing.min_us,
            structured_timing.max_us,
            structured_timing.median_us / batches as f64,
            structured_count.scalar_products,
            structured_count.additions,
            structured_count.relinearizations,
            structured_count.rescales
        );
    }

    // PC-vs-CP overhead on two representative rectangular GEMMs.
    for &(name, m, k, n) in &[
        ("pc-small", 2_usize, 3_usize, 4_usize),
        ("pc-rect", 4, 8, 4),
    ] {
        let lhs_values = values(m, k, 0x35B0_1000 + m as u64);
        let rhs_values = values(k, n, 0x35B0_2000 + n as u64);

        let lhs_ct = RnsCkksCiphertextMatrix::from_vec_column_major(
            m,
            k,
            lhs_values
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    encrypt_scalar(
                        v,
                        0x35B0_3000 + i as u64,
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

        let lhs_pt = RnsCkksPlaintextMatrix::from_vec_column_major(
            m,
            k,
            scale,
            lhs_values
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
                        0x35B0_4000 + i as u64,
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
            k,
            n,
            scale,
            rhs_values
                .iter()
                .map(|&v| encode_rns(v, &embedding, &basis, scale))
                .collect(),
        );

        let shape = GemmShape::new(MatrixShape::new(m, k), MatrixShape::new(k, n));
        let cp_spec = GemmSpec::new(shape, PrivacyMode::Cp);
        let pc_spec = GemmSpec::new(shape, PrivacyMode::Pc);

        let cp_timing = benchmark(|| gemm_cp(cp_spec, &lhs_ct, &rhs_pt, &chain, &plan));
        let pc_timing = benchmark(|| gemm_pc(pc_spec, &lhs_pt, &rhs_ct, &chain, &plan));

        println!(
            "PC_RESULT,{name},{m},{k},{n},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6}",
            cp_timing.median_us,
            cp_timing.min_us,
            cp_timing.max_us,
            pc_timing.median_us,
            pc_timing.min_us,
            pc_timing.max_us,
            pc_timing.median_us / cp_timing.median_us,
        );
    }

    println!("R3_5H2_EBLAS_BATCH_PC_CHARACTERIZATION_STATUS=PASS");
}
