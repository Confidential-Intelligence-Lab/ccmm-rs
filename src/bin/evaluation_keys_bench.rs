use std::hint::black_box;
use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use ccmm_rs::ckks::{
    apply_rns_galois_automorphism, conjugation_exponent, LogicalPayloadBytes, RnsGaloisKey,
};
use ccmm_rs::grafting::{
    hybrid_relinearize, hybrid_relinearize_helper_prime, hybrid_relinearize_prepared,
    rns_relinearize, HelperPrimeNttPlan, HybridMultiplicationKey, HybridQuadraticCiphertext,
    MixedGadgetLayout, Pow2GraftedBasis, PreparedHybridMultiplicationKey, RnsGadgetLayout,
    RnsMultiplicationKey, RnsQuadraticCiphertext,
};
use ccmm_rs::ring::{Modulus, ModulusBasis};

const HELPER_PRIME: u64 = 2_013_265_921;
const SPROUT_BITS: u32 = 8;
const BENCH_TRIALS: usize = 16;

fn main() {
    println!("CKKS_EVAL_KEY_BENCH_VERSION=1");
    println!("PARAMETER_CLASS=CORRECTNESS_ORIENTED");
    println!("SECURITY_BEARING=false");
    println!("SPROUT_BITS={SPROUT_BITS}");
    println!("HELPER_PRIME={HELPER_PRIME}");

    for degree in [8_usize, 16, 32, 64, 128, 256, 512, 1024] {
        run_case(degree);
    }

    println!("CKKS_EVAL_KEY_BENCH_STATUS=PASS");
}

fn run_case(degree: usize) {
    let basis = basis();
    let secret = ternary_secret(degree);

    let ordinary_layout = RnsGadgetLayout::new(basis.clone(), vec![1, 1, 1]);

    let mixed_layout = MixedGadgetLayout::new(basis.clone(), vec![1, 1, 1], SPROUT_BITS);

    /*
     * Measure generation independently. Each path gets a
     * deterministic but distinct RNG stream.
     */
    let ordinary_keygen = measure_once_ns(|| {
        let mut rng = ChaCha20Rng::seed_from_u64(0xE800_0000 ^ degree as u64);

        black_box(RnsMultiplicationKey::generate_with_rng(
            degree,
            16,
            1,
            &secret,
            ordinary_layout.clone(),
            &mut rng,
        ));
    });

    let exponent = conjugation_exponent(degree);

    let galois_keygen = measure_once_ns(|| {
        let mut rng = ChaCha20Rng::seed_from_u64(0xE810_0000 ^ degree as u64);

        black_box(RnsGaloisKey::generate_with_rng(
            degree,
            16,
            1,
            &secret,
            exponent,
            ordinary_layout.clone(),
            &mut rng,
        ));
    });

    let hybrid_keygen = measure_once_ns(|| {
        let mut rng = ChaCha20Rng::seed_from_u64(0xE820_0000 ^ degree as u64);

        black_box(HybridMultiplicationKey::generate_with_rng(
            degree,
            16,
            1,
            &secret,
            mixed_layout.clone(),
            &mut rng,
        ));
    });

    /*
     * Build persistent material used by the operation
     * benchmark.
     */
    let mut ordinary_rng = ChaCha20Rng::seed_from_u64(0xE830_0000 ^ degree as u64);

    let ordinary_key = RnsMultiplicationKey::generate_with_rng(
        degree,
        16,
        1,
        &secret,
        ordinary_layout.clone(),
        &mut ordinary_rng,
    );

    let mut galois_rng = ChaCha20Rng::seed_from_u64(0xE840_0000 ^ degree as u64);

    let galois_key = RnsGaloisKey::generate_with_rng(
        degree,
        16,
        1,
        &secret,
        exponent,
        ordinary_layout,
        &mut galois_rng,
    );

    let mut hybrid_rng = ChaCha20Rng::seed_from_u64(0xE850_0000 ^ degree as u64);

    let hybrid_key = HybridMultiplicationKey::generate_with_rng(
        degree,
        16,
        1,
        &secret,
        mixed_layout,
        &mut hybrid_rng,
    );

    let helper = HelperPrimeNttPlan::new(SPROUT_BITS, degree, Modulus::new(HELPER_PRIME));

    let prepare_start = Instant::now();

    let prepared = PreparedHybridMultiplicationKey::prepare(&hybrid_key, &helper);

    let prepare_ns = prepare_start.elapsed().as_nanos();

    let grafted = Pow2GraftedBasis::new(basis, SPROUT_BITS);

    let product = quadratic(&grafted, degree);

    /*
     * Ordinary RNS baseline derived from the exact same
     * deterministic quadratic coefficients as the hybrid case.
     */
    let ordinary_product = RnsQuadraticCiphertext::from_rns_polynomials(
        product.c0().ordinary().clone(),
        product.c1().ordinary().clone(),
        product.c2().ordinary().clone(),
    );

    /*
     * A real RNS RLWE ciphertext with the same degree and basis.
     * Using one evaluation-key entry avoids introducing unrelated
     * encryption/setup work into the execution benchmark.
     */
    let galois_input = ordinary_key.entry(0).clone();

    let iterations = iterations_for_degree(degree);

    /*
     * Warm all measured hybrid execution paths.
     */
    for _ in 0..8 {
        black_box(hybrid_relinearize(
            black_box(&product),
            black_box(&hybrid_key),
        ));

        black_box(hybrid_relinearize_helper_prime(
            black_box(&product),
            black_box(&hybrid_key),
            black_box(&helper),
        ));

        black_box(hybrid_relinearize_prepared(
            black_box(&product),
            black_box(&prepared),
            black_box(&helper),
        ));

        black_box(rns_relinearize(
            black_box(&ordinary_product),
            black_box(&ordinary_key),
        ));

        black_box(apply_rns_galois_automorphism(
            black_box(&galois_input),
            black_box(&galois_key),
        ));
    }

    let mut ordinary_relin_trials = Vec::with_capacity(BENCH_TRIALS);

    let mut galois_trials = Vec::with_capacity(BENCH_TRIALS);

    for _ in 0..BENCH_TRIALS {
        ordinary_relin_trials.push(measure_ns(iterations, || {
            black_box(rns_relinearize(
                black_box(&ordinary_product),
                black_box(&ordinary_key),
            ));
        }));

        galois_trials.push(measure_ns(iterations, || {
            black_box(apply_rns_galois_automorphism(
                black_box(&galois_input),
                black_box(&galois_key),
            ));
        }));
    }

    let ordinary_relin_ns = median(&ordinary_relin_trials);

    let galois_ns = median(&galois_trials);

    let mut direct_trials = Vec::with_capacity(BENCH_TRIALS);

    for _ in 0..BENCH_TRIALS {
        direct_trials.push(measure_ns(iterations, || {
            black_box(hybrid_relinearize(
                black_box(&product),
                black_box(&hybrid_key),
            ));
        }));
    }

    let hybrid_direct_ns = median(&direct_trials);

    let mut helper_trials = Vec::with_capacity(BENCH_TRIALS);

    for _ in 0..BENCH_TRIALS {
        helper_trials.push(measure_ns(iterations, || {
            black_box(hybrid_relinearize_helper_prime(
                black_box(&product),
                black_box(&hybrid_key),
                black_box(&helper),
            ));
        }));
    }

    let hybrid_helper_ns = median(&helper_trials);

    let mut prepared_trials = Vec::with_capacity(BENCH_TRIALS);

    for _ in 0..BENCH_TRIALS {
        prepared_trials.push(measure_ns(iterations, || {
            black_box(hybrid_relinearize_prepared(
                black_box(&product),
                black_box(&prepared),
                black_box(&helper),
            ));
        }));
    }

    let hybrid_prepared_ns = median(&prepared_trials);

    println!("DEGREE={degree}");
    println!("LEVEL=0");
    println!("ITERATIONS={iterations}");
    println!("BENCH_TRIALS={BENCH_TRIALS}");

    println!("ORDINARY_RELINEARIZE_MEDIAN_NS={ordinary_relin_ns:.3}");
    println!(
        "ORDINARY_RELINEARIZE_MEAN_NS={:.3}",
        mean(&ordinary_relin_trials),
    );
    println!(
        "ORDINARY_RELINEARIZE_MIN_NS={:.3}",
        minimum(&ordinary_relin_trials),
    );
    println!(
        "ORDINARY_RELINEARIZE_MAX_NS={:.3}",
        maximum(&ordinary_relin_trials),
    );

    println!("GALOIS_AUTOMORPHISM_KEYSWITCH_MEDIAN_NS={galois_ns:.3}");
    println!(
        "GALOIS_AUTOMORPHISM_KEYSWITCH_MEAN_NS={:.3}",
        mean(&galois_trials),
    );
    println!(
        "GALOIS_AUTOMORPHISM_KEYSWITCH_MIN_NS={:.3}",
        minimum(&galois_trials),
    );
    println!(
        "GALOIS_AUTOMORPHISM_KEYSWITCH_MAX_NS={:.3}",
        maximum(&galois_trials),
    );

    println!("ORDINARY_MULTIPLICATION_KEYGEN_NS={ordinary_keygen}");

    println!("GALOIS_KEYGEN_NS={galois_keygen}");

    println!("HYBRID_KEYGEN_NS={hybrid_keygen}");

    println!("PREPARED_KEY_BUILD_NS={prepare_ns}");

    println!(
        "ORDINARY_MULTIPLICATION_KEY_BYTES={}",
        ordinary_key.logical_payload_bytes(),
    );

    println!("GALOIS_KEY_BYTES={}", galois_key.logical_payload_bytes(),);

    println!("HYBRID_KEY_BYTES={}", hybrid_key.logical_payload_bytes(),);

    println!("PREPARED_KEY_BYTES={}", prepared.logical_payload_bytes(),);

    println!("HYBRID_DIRECT_MEDIAN_NS={hybrid_direct_ns:.3}");
    println!("HYBRID_DIRECT_MEAN_NS={:.3}", mean(&direct_trials));
    println!("HYBRID_DIRECT_MIN_NS={:.3}", minimum(&direct_trials));
    println!("HYBRID_DIRECT_MAX_NS={:.3}", maximum(&direct_trials));

    println!("HYBRID_HELPER_MEDIAN_NS={hybrid_helper_ns:.3}");
    println!("HYBRID_HELPER_MEAN_NS={:.3}", mean(&helper_trials));
    println!("HYBRID_HELPER_MIN_NS={:.3}", minimum(&helper_trials));
    println!("HYBRID_HELPER_MAX_NS={:.3}", maximum(&helper_trials));

    println!("HYBRID_PREPARED_MEDIAN_NS={hybrid_prepared_ns:.3}");
    println!("HYBRID_PREPARED_MEAN_NS={:.3}", mean(&prepared_trials));
    println!("HYBRID_PREPARED_MIN_NS={:.3}", minimum(&prepared_trials));
    println!("HYBRID_PREPARED_MAX_NS={:.3}", maximum(&prepared_trials));

    println!(
        "HELPER_VS_DIRECT_SPEEDUP={:.6}",
        hybrid_direct_ns / hybrid_helper_ns,
    );

    println!(
        "PREPARED_VS_DIRECT_SPEEDUP={:.6}",
        hybrid_direct_ns / hybrid_prepared_ns,
    );

    println!(
        "PREPARED_VS_HELPER_SPEEDUP={:.6}",
        hybrid_helper_ns / hybrid_prepared_ns,
    );

    if hybrid_direct_ns > hybrid_prepared_ns {
        println!(
            "PREPARED_BREAK_EVEN_OPERATIONS={:.3}",
            prepare_ns as f64 / (hybrid_direct_ns - hybrid_prepared_ns),
        );
    } else {
        println!("PREPARED_BREAK_EVEN_OPERATIONS=INF");
    }

    println!("CASE_STATUS=PASS");
    println!();
}

fn iterations_for_degree(degree: usize) -> usize {
    match degree {
        8 => 2_000,
        16 => 1_000,
        32 => 500,
        64 => 250,
        128 => 100,
        256 => 50,
        512 => 20,
        1024 => 10,
        _ => 5,
    }
}

fn measure_once_ns<F>(operation: F) -> u128
where
    F: FnOnce(),
{
    let start = Instant::now();

    operation();

    start.elapsed().as_nanos()
}

fn measure_ns<F>(iterations: usize, mut operation: F) -> f64
where
    F: FnMut(),
{
    let start = Instant::now();

    for _ in 0..iterations {
        operation();
    }

    start.elapsed().as_nanos() as f64 / iterations as f64
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn minimum(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn maximum(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();

    sorted.sort_by(|lhs, rhs| {
        lhs.partial_cmp(rhs)
            .expect("benchmark values must be finite")
    });

    let middle = sorted.len() / 2;

    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn basis() -> ModulusBasis {
    ModulusBasis::new(vec![
        Modulus::new(12_289),
        Modulus::new(40_961),
        Modulus::new(65_537),
    ])
}

fn ternary_secret(degree: usize) -> Vec<i8> {
    (0..degree)
        .map(|index| match index % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 1,
        })
        .collect()
}

fn quadratic(basis: &Pow2GraftedBasis, degree: usize) -> HybridQuadraticCiphertext {
    let modulus = basis.composite_modulus();

    let component = |offset: u128| {
        basis.from_coefficients(
            &(0..degree)
                .map(|index| (offset + 17 * index as u128 + 5 * (index as u128).pow(2)) % modulus)
                .collect::<Vec<_>>(),
        )
    };

    HybridQuadraticCiphertext::new(component(3), component(29), component(71))
}
