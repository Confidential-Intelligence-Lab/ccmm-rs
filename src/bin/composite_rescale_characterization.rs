use std::hint::black_box;
use std::time::Instant;

use ccmm_rs::ring::{
    composite_modulus_big, composite_rescale_reference_coefficients,
    composite_rescale_reference_polynomial, reconstruct_coefficients_big,
    rescale_composite_level_to_next, rns_from_big_coefficients, CompositeModulusChain, Modulus,
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
    assert_eq!(args.len(), 3, "usage: binary DEGREE DROP_COUNT");

    let degree: usize = args[1].parse().expect("degree");
    let drop_count: usize = args[2].parse().expect("drop count");

    assert!([256, 1024, 4096, 8192, 16384, 32768, 65536].contains(&degree));
    assert!([1, 2, 3, 4, 6].contains(&drop_count));

    let moduli: Vec<Modulus> = PRIMES.iter().copied().map(Modulus::new).collect();

    let basis = ModulusBasis::new(moduli.clone());

    let chain = CompositeModulusChain::new(basis.clone(), vec![8 - drop_count, drop_count]);

    assert_eq!(chain.logical_level_count(), 2);
    assert_eq!(
        chain
            .dropped_logical_level(0)
            .expect("trailing group")
            .physical_limb_count(),
        drop_count
    );

    let composite = composite_modulus_big(&basis);
    let divisor = chain.rescale_divisor_big(0).expect("divisor");

    let coefficients: Vec<BigUint> = (0..degree)
        .map(|i| {
            let x = BigUint::from(i as u64 + 1);
            let value = &x * &x * BigUint::from(1234567_u64);
            (&composite - BigUint::from(1_u64) - value) % &composite
        })
        .collect();

    let polynomial = rns_from_big_coefficients(moduli, &coefficients);

    let production = rescale_composite_level_to_next(&polynomial, &chain, 0);

    let expected = composite_rescale_reference_coefficients(&polynomial, &chain, 0);

    let reference = composite_rescale_reference_polynomial(&polynomial, &chain, 0);

    assert_eq!(production, reference);
    assert_eq!(reconstruct_coefficients_big(&production), expected);
    assert_eq!(production.basis(), &chain.active_basis(1));

    let production_ns =
        measure(|| rescale_composite_level_to_next(black_box(&polynomial), black_box(&chain), 0));

    let reference_ns = measure(|| {
        composite_rescale_reference_coefficients(black_box(&polynomial), black_box(&chain), 0)
    });

    let reconstruction_ns = measure(|| reconstruct_coefficients_big(black_box(&production)));

    println!(
        "RESULT,{degree},{drop_count},{},{},{},\
         {production_ns:.3},{reference_ns:.3},\
         {reconstruction_ns:.3},PASS",
        composite.bits(),
        divisor.bits(),
        chain.active_composite_modulus_big(1).bits(),
    );
    println!("COMPOSITE_RESCALE_STATUS=PASS");
}
