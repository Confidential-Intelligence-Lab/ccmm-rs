use std::hint::black_box;
use std::time::Instant;

use ccmm_rs::ring::{
    composite_modulus_big, reconstruct_coefficients_big, rns_from_big_coefficients, Modulus,
    ModulusBasis,
};
use num_bigint::BigUint;

const PRIMES: [u64; 8] = [
    1125899903827969,
    1125899902124033,
    1125899887312897,
    1125899886395393,
    1125899885740033,
    1125899884167169,
    1125899884036097,
    1125899883642881,
];

const WARMUPS: usize = 3;
const REPEATS: usize = 15;

fn measure<F, T>(mut operation: F) -> f64
where
    F: FnMut() -> T,
{
    for _ in 0..WARMUPS {
        black_box(operation());
    }

    let mut samples = Vec::with_capacity(REPEATS);

    for _ in 0..REPEATS {
        let start = Instant::now();
        black_box(operation());
        samples.push(start.elapsed().as_secs_f64() * 1e9);
    }

    samples.sort_by(|a, b| a.total_cmp(b));
    samples[REPEATS / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 3, "usage: binary DEGREE LIMBS");

    let degree: usize = args[1].parse().expect("degree");
    let limbs: usize = args[2].parse().expect("limbs");

    assert!([256, 1024, 4096].contains(&degree));
    assert!([2, 3, 4, 6, 8].contains(&limbs));

    let moduli: Vec<Modulus> = PRIMES[..limbs].iter().copied().map(Modulus::new).collect();

    let basis = ModulusBasis::new(moduli.clone());
    let composite = composite_modulus_big(&basis);

    let lhs: Vec<BigUint> = (0..degree)
        .map(|i| {
            let x = BigUint::from(i as u64 + 1);
            (&composite - BigUint::from(1_u64) - &x * &x) % &composite
        })
        .collect();

    let rhs: Vec<BigUint> = (0..degree)
        .map(|i| {
            let x = BigUint::from(i as u64 + 3);
            (&x * &x * BigUint::from(1234567_u64)) % &composite
        })
        .collect();

    let expected: Vec<BigUint> = lhs
        .iter()
        .zip(&rhs)
        .map(|(a, b)| (a + b) % &composite)
        .collect();

    let lhs_rns = rns_from_big_coefficients(moduli.clone(), &lhs);
    let rhs_rns = rns_from_big_coefficients(moduli.clone(), &rhs);

    assert_eq!(reconstruct_coefficients_big(&lhs_rns), lhs);
    assert_eq!(reconstruct_coefficients_big(&rhs_rns), rhs);

    let result_rns = lhs_rns.add(&rhs_rns);
    assert_eq!(reconstruct_coefficients_big(&result_rns), expected);

    let bigint_add_ns = measure(|| {
        lhs.iter()
            .zip(&rhs)
            .map(|(a, b)| (a + b) % &composite)
            .collect::<Vec<_>>()
    });

    let encode_ns =
        measure(|| rns_from_big_coefficients(black_box(moduli.clone()), black_box(&lhs)));

    let rns_add_ns = measure(|| black_box(&lhs_rns).add(black_box(&rhs_rns)));

    let reconstruct_ns = measure(|| reconstruct_coefficients_big(black_box(&result_rns)));

    println!(
        "RESULT,{degree},{limbs},{},{bigint_add_ns:.3},\
         {encode_ns:.3},{rns_add_ns:.3},{reconstruct_ns:.3},PASS",
        composite.bits()
    );
    println!("WIDE_ARITHMETIC_STATUS=PASS");
}
