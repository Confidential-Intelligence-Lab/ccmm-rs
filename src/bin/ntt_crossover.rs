use std::hint::black_box;
use std::time::{Duration, Instant};

use ccmm_rs::ring::{make_ntt_plan, Modulus, Polynomial};

const MODULUS: u64 = 12_289;

fn deterministic_polynomial(modulus: Modulus, degree: usize, offset: u64) -> Polynomial {
    Polynomial::new(
        modulus,
        (0..degree)
            .map(|index| {
                let i = index as u64;

                (offset + 17 * i + 5 * i * i + 3 * i * i * i) % modulus.value()
            })
            .collect(),
    )
}

fn iterations_for_degree(degree: usize) -> usize {
    match degree {
        16 => 20_000,
        32 => 10_000,
        64 => 5_000,
        128 => 2_000,
        256 => 500,
        512 => 100,
        1024 => 20,
        _ => 10,
    }
}

fn time_average<F>(iterations: usize, mut f: F) -> Duration
where
    F: FnMut(),
{
    let start = Instant::now();

    for _ in 0..iterations {
        f();
    }

    start.elapsed() / iterations as u32
}

fn nanos(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000_000.0
}

fn main() {
    let modulus = Modulus::new(MODULUS);

    println!("NTT_CROSSOVER_VERSION=1");
    println!("MODULUS={MODULUS}");
    println!();

    for degree in [16_usize, 32, 64, 128, 256, 512, 1024] {
        if (modulus.value() - 1) % (2 * degree) as u64 != 0 {
            println!("DEGREE_{degree}_SKIPPED=INCOMPATIBLE_MODULUS");
            continue;
        }

        let plan = make_ntt_plan(modulus, degree);

        let lhs = deterministic_polynomial(modulus, degree, 7);

        let rhs = deterministic_polynomial(modulus, degree, 29);

        let reference = lhs.negacyclic_mul(&rhs);

        let optimized = plan.negacyclic_mul(&lhs, &rhs);

        assert_eq!(
            optimized, reference,
            "NTT/reference mismatch at degree {degree}"
        );

        let iterations = iterations_for_degree(degree);

        // Warm-up.
        for _ in 0..10 {
            black_box(lhs.negacyclic_mul(&rhs));

            black_box(plan.negacyclic_mul(&lhs, &rhs));
        }

        let naive = time_average(iterations, || {
            black_box(lhs.negacyclic_mul(&rhs));
        });

        let ntt = time_average(iterations, || {
            black_box(plan.negacyclic_mul(&lhs, &rhs));
        });

        let naive_ns = nanos(naive);
        let ntt_ns = nanos(ntt);

        let speedup = naive_ns / ntt_ns;

        println!("DEGREE={degree}");
        println!("ITERATIONS={iterations}");
        println!("NAIVE_NS={naive_ns:.3}");
        println!("NTT_NS={ntt_ns:.3}");
        println!("SPEEDUP={speedup:.6}");
        println!("CORRECTNESS=PASS");
        println!();
    }

    println!("NTT_CROSSOVER_STATUS=PASS");
}
