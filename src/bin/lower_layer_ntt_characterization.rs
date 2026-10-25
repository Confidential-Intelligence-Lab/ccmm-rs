use std::hint::black_box;
use std::time::Instant;

use ccmm_rs::ring::{make_ntt_plan, Modulus, Polynomial};

const DEGREE: usize = 4096;
const MODULI: [u64; 3] = [34_359_410_689, 34_359_451_649, 34_359_697_409];

const WARMUPS: usize = 3;
const REPETITIONS: usize = 10;

fn polynomial(modulus: Modulus, offset: u64) -> Polynomial {
    let q = modulus.value();

    Polynomial::new(
        modulus,
        (0..DEGREE)
            .map(|i| {
                let x = i as u64;
                (offset as u128 + 17 * x as u128 + 5 * (x as u128).pow(2) + 3 * (x as u128).pow(3))
                    as u64
                    % q
            })
            .collect(),
    )
}

fn measure<F, T>(mut operation: F) -> Vec<f64>
where
    F: FnMut() -> T,
{
    for _ in 0..WARMUPS {
        black_box(operation());
    }

    let mut samples = Vec::with_capacity(REPETITIONS);

    for _ in 0..REPETITIONS {
        let start = Instant::now();
        black_box(operation());
        samples.push(start.elapsed().as_secs_f64() * 1e9);
    }

    samples
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    (sorted[4] + sorted[5]) / 2.0
}

fn main() {
    println!("LOWER_LAYER_NTT_VERSION=1");
    println!("SECURITY_MODEL=classical");
    println!("SECURITY_TARGET_BITS=128");
    println!("SECURITY_CLAIM_SCOPE=historically_assessed_rlwe_tuple");
    println!("FULL_IMPLEMENTATION_QUALIFIED=false");
    println!("DEGREE={DEGREE}");
    println!("WARMUPS={WARMUPS}");
    println!("REPETITIONS={REPETITIONS}");

    for (limb, &q) in MODULI.iter().enumerate() {
        assert_eq!((q - 1) % (2 * DEGREE as u64), 0);

        let modulus = Modulus::new(q);
        let plan = make_ntt_plan(modulus, DEGREE);

        let lhs = polynomial(modulus, 7);
        let rhs = polynomial(modulus, 29);

        let transformed = plan.forward_radix2(&lhs);
        let recovered = plan.inverse_radix2(&transformed);

        assert_eq!(recovered, lhs, "NTT round-trip failed");

        let product = plan.negacyclic_mul(&lhs, &rhs);

        // Check the complete multiplication against an
        // independent schoolbook implementation.
        let reference = lhs.negacyclic_mul(&rhs);

        assert_eq!(product, reference, "Polynomial product mismatch");

        let forward = measure(|| plan.forward_radix2(&lhs));

        let inverse = measure(|| plan.inverse_radix2(black_box(&transformed)));

        let multiply = measure(|| plan.negacyclic_mul(&lhs, &rhs));

        println!("LIMB={limb}");
        println!("MODULUS={q}");
        println!("FORWARD_NTT_MEDIAN_NS={:.3}", median(&forward));
        println!("INVERSE_NTT_MEDIAN_NS={:.3}", median(&inverse));
        println!("NTT_MUL_MEDIAN_NS={:.3}", median(&multiply));
        println!("CORRECTNESS=PASS");
    }

    println!("LOWER_LAYER_NTT_STATUS=PASS");
}
