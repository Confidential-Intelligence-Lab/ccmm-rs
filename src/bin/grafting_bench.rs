use std::hint::black_box;
use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use ccmm_rs::grafting::{
    hybrid_relinearize, hybrid_relinearize_helper_prime, hybrid_relinearize_prepared,
    HelperPrimeNttPlan, HybridMultiplicationKey, HybridQuadraticCiphertext, MixedGadgetLayout,
    Pow2GraftedBasis, PreparedHybridMultiplicationKey,
};
use ccmm_rs::ring::{Modulus, ModulusBasis};

const HELPER_PRIME: u64 = 2_013_265_921;

fn main() {
    println!("GRAFTING_BENCH_VERSION=1");

    for degree in [8_usize, 16, 32, 64, 128, 256, 512, 1024] {
        for bits in [4_u32, 6, 8] {
            if !helper_is_admissible(degree, bits) {
                continue;
            }

            run_case(degree, bits);
        }
    }

    println!("GRAFTING_BENCH_STATUS=PASS");
}

fn run_case(degree: usize, bits: u32) {
    let ordinary_basis = basis_for_degree(degree);

    let grafted = Pow2GraftedBasis::new(ordinary_basis.clone(), bits);

    let layout = MixedGadgetLayout::new(ordinary_basis, vec![1, 1, 1], bits);

    let ternary = ternary_secret(degree);

    let product = quadratic(&grafted, degree);

    let mut rng = ChaCha20Rng::seed_from_u64(0xBEEF_0000 ^ degree as u64 ^ ((bits as u64) << 32));

    let key = HybridMultiplicationKey::generate_with_rng(degree, 16, 1, &ternary, layout, &mut rng);

    let helper = HelperPrimeNttPlan::new(bits, degree, Modulus::new(HELPER_PRIME));

    let prepare_start = Instant::now();

    let prepared = PreparedHybridMultiplicationKey::prepare(&key, &helper);

    let prepare_ns = prepare_start.elapsed().as_nanos();

    /*
     * Sprout-only multiplication benchmark using a real
     * evaluation-key operand.
     */
    let sprout_entry = key.entry(0).sprout();

    let prepared_sprout_entry = prepared.sprout_entry(0);

    let sprout_digit = product.c2().sprout().clone();

    /*
     * Keep total benchmark work roughly similar across
     * degrees while avoiding very short measurements.
     */
    let iterations = match degree {
        8 => 2_000,
        16 => 1_000,
        32 => 500,
        64 => 250,
        128 => 100,
        256 => 50,
        512 => 20,
        1024 => 10,
        _ => 5,
    };

    /*
     * Warm each path before measurement.
     */
    for _ in 0..8 {
        black_box(hybrid_relinearize(black_box(&product), black_box(&key)));

        black_box(hybrid_relinearize_helper_prime(
            black_box(&product),
            black_box(&key),
            black_box(&helper),
        ));

        black_box(hybrid_relinearize_prepared(
            black_box(&product),
            black_box(&prepared),
            black_box(&helper),
        ));
    }

    let sprout_direct = measure_ns(iterations, || {
        black_box(black_box(&sprout_digit).negacyclic_mul(black_box(sprout_entry.b())));
    });

    let sprout_helper_uncached = measure_ns(iterations, || {
        black_box(helper.negacyclic_mul(black_box(&sprout_digit), black_box(sprout_entry.b())));
    });

    let sprout_helper_cached = measure_ns(iterations, || {
        black_box(helper.negacyclic_mul_prepared(
            black_box(&sprout_digit),
            black_box(prepared_sprout_entry.b()),
        ));
    });

    let direct = measure_ns(iterations, || {
        black_box(hybrid_relinearize(black_box(&product), black_box(&key)));
    });

    let helper_uncached = measure_ns(iterations, || {
        black_box(hybrid_relinearize_helper_prime(
            black_box(&product),
            black_box(&key),
            black_box(&helper),
        ));
    });

    let helper_cached = measure_ns(iterations, || {
        black_box(hybrid_relinearize_prepared(
            black_box(&product),
            black_box(&prepared),
            black_box(&helper),
        ));
    });

    println!("DEGREE={degree}");
    println!("SPROUT_BITS={bits}");
    println!("ITERATIONS={iterations}");

    println!("PREPARE_KEY_NS={prepare_ns}");

    println!("SPROUT_DIRECT_MEAN_NS={sprout_direct:.3}");

    println!("SPROUT_HELPER_UNCACHED_MEAN_NS={sprout_helper_uncached:.3}");

    println!("SPROUT_HELPER_CACHED_MEAN_NS={sprout_helper_cached:.3}");

    println!(
        "SPROUT_UNCACHED_SPEEDUP={:.6}",
        sprout_direct / sprout_helper_uncached,
    );

    println!(
        "SPROUT_CACHED_SPEEDUP={:.6}",
        sprout_direct / sprout_helper_cached,
    );

    println!("DIRECT_MEAN_NS={direct:.3}");

    println!("HELPER_UNCACHED_MEAN_NS={helper_uncached:.3}");

    println!("HELPER_CACHED_MEAN_NS={helper_cached:.3}");

    println!("UNCACHED_SPEEDUP={:.6}", direct / helper_uncached,);

    println!("CACHED_SPEEDUP={:.6}", direct / helper_cached,);

    println!(
        "CACHE_VS_UNCACHED_SPEEDUP={:.6}",
        helper_uncached / helper_cached,
    );

    /*
     * Approximate number of relinearizations required
     * to amortize one-time EK preparation.
     */
    if direct > helper_cached {
        let saving = direct - helper_cached;

        println!(
            "CACHE_BREAK_EVEN_RELINEARIZATIONS={:.3}",
            prepare_ns as f64 / saving,
        );
    } else {
        println!("CACHE_BREAK_EVEN_RELINEARIZATIONS=INF");
    }

    println!("CASE_STATUS=PASS");
    println!();
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

fn helper_is_admissible(degree: usize, bits: u32) -> bool {
    let max = (1_u128 << bits) - 1;

    let bound = degree as u128 * max * max;

    u128::from(HELPER_PRIME) > 2 * bound
}

fn basis_for_degree(_degree: usize) -> ModulusBasis {
    /*
     * These moduli are adequate for the ordinary-path
     * correctness/benchmark representation. The helper
     * NTT itself uses HELPER_PRIME independently.
     */
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
