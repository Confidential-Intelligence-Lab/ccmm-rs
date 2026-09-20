//! R3.5h1 eBLAS backend characterization.
//!
//! Setup, encoding, encryption, and key generation are deliberately outside
//! timed regions. Timings characterize only the executable eBLAS GEMM call.

use ccmm_rs::ckks::{
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
};
use ccmm_rs::eblas::{
    gemm_cc, gemm_cp, GemmBackend, GemmOperationCount, GemmShape, GemmSpec, MatrixShape,
    PrivacyMode,
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
struct Workload {
    family: &'static str,
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
}

const WORKLOADS: &[Workload] = &[
    Workload {
        family: "crossover",
        name: "dot-1",
        m: 1,
        k: 1,
        n: 1,
    },
    Workload {
        family: "crossover",
        name: "dot-2",
        m: 1,
        k: 2,
        n: 1,
    },
    Workload {
        family: "crossover",
        name: "dot-4",
        m: 1,
        k: 4,
        n: 1,
    },
    Workload {
        family: "crossover",
        name: "dot-8",
        m: 1,
        k: 8,
        n: 1,
    },
    Workload {
        family: "crossover",
        name: "dot-16",
        m: 1,
        k: 16,
        n: 1,
    },
    Workload {
        family: "shape",
        name: "square-2",
        m: 2,
        k: 2,
        n: 2,
    },
    Workload {
        family: "shape",
        name: "square-4",
        m: 4,
        k: 4,
        n: 4,
    },
    Workload {
        family: "shape",
        name: "mlp-1x8x4",
        m: 1,
        k: 8,
        n: 4,
    },
    Workload {
        family: "shape",
        name: "rect-4x8x4",
        m: 4,
        k: 8,
        n: 4,
    },
    Workload {
        family: "shape",
        name: "rect-8x4x8",
        m: 8,
        k: 4,
        n: 8,
    },
];

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

fn print_result(
    workload: Workload,
    privacy: PrivacyMode,
    backend: GemmBackend,
    timing: Timing,
    count: GemmOperationCount,
) {
    let outputs = workload.m * workload.n;

    println!(
        "RESULT,{},{},{},{},{},{:?},{:?},{:.3},{:.3},{:.3},{:.3},{},{},{},{}",
        workload.family,
        workload.name,
        workload.m,
        workload.k,
        workload.n,
        privacy,
        backend,
        timing.median_us,
        timing.min_us,
        timing.max_us,
        timing.median_us / outputs as f64,
        count.scalar_products,
        count.additions,
        count.relinearizations,
        count.rescales,
    );
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

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x35A8_0001);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&x| x == 0) {
        secret[0] = 1;
    }

    let layout = BoundedGadgetLayout::new(basis.clone(), BASE_LOG);
    let mut key_rng = ChaCha20Rng::seed_from_u64(0x35A8_0002);
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

    println!("R3_5H1_EBLAS_BACKEND_CHARACTERIZATION_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("DEGREE={degree}");
    println!("WARMUPS={WARMUPS}");
    println!("REPEATS={REPEATS}");
    println!("TIMING_STATISTIC=median");
    println!("TIMED_REGION=eblas-gemm-call-only");
    println!("SETUP_INCLUDED_IN_TIMING=NO");
    println!("CSV_HEADER=family,name,m,k,n,privacy,backend,median_us,min_us,max_us,us_per_output,scalar_products,additions,relinearizations,rescales");

    for (case_index, workload) in WORKLOADS.iter().copied().enumerate() {
        let lhs_values = values(workload.m, workload.k, 0x35A8_1000 + case_index as u64);
        let rhs_values = values(workload.k, workload.n, 0x35A8_2000 + case_index as u64);

        let shape = GemmShape::new(
            MatrixShape::new(workload.m, workload.k),
            MatrixShape::new(workload.k, workload.n),
        );
        let cp_spec = GemmSpec::new(shape, PrivacyMode::Cp);
        let cc_spec = GemmSpec::new(shape, PrivacyMode::Cc);

        let lhs_ct = RnsCkksCiphertextMatrix::from_vec_column_major(
            workload.m,
            workload.k,
            lhs_values
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    encrypt_scalar(
                        v,
                        0x35A9_0000 + (case_index * 1000 + i) as u64,
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
            workload.k,
            workload.n,
            scale,
            rhs_values
                .iter()
                .map(|&v| encode_rns(v, &embedding, &basis, scale))
                .collect(),
        );

        let rhs_ct = RnsCkksCiphertextMatrix::from_vec_column_major(
            workload.k,
            workload.n,
            rhs_values
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    encrypt_scalar(
                        v,
                        0x35AA_0000 + (case_index * 1000 + i) as u64,
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

        let cp_timing = benchmark(|| gemm_cp(cp_spec, &lhs_ct, &rhs_plain, &chain, &plan));
        let scalar_timing = benchmark(|| {
            gemm_cc(
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
            gemm_cc(
                cc_spec,
                GemmBackend::CcStructured,
                &lhs_ct,
                &rhs_ct,
                &multiplication_key,
                &chain,
                &plan,
            )
        });

        let cp_count = GemmOperationCount::for_backend(cp_spec, GemmBackend::CpDirect);
        let scalar_count = GemmOperationCount::for_backend(cc_spec, GemmBackend::CcScalar);
        let structured_count = GemmOperationCount::for_backend(cc_spec, GemmBackend::CcStructured);

        print_result(
            workload,
            PrivacyMode::Cp,
            GemmBackend::CpDirect,
            cp_timing,
            cp_count,
        );
        print_result(
            workload,
            PrivacyMode::Cc,
            GemmBackend::CcScalar,
            scalar_timing,
            scalar_count,
        );
        print_result(
            workload,
            PrivacyMode::Cc,
            GemmBackend::CcStructured,
            structured_timing,
            structured_count,
        );

        println!(
            "SPEEDUP,{},{:.6}",
            workload.name,
            scalar_timing.median_us / structured_timing.median_us
        );
    }

    println!("R3_5H1_EBLAS_BACKEND_CHARACTERIZATION_STATUS=PASS");
}
