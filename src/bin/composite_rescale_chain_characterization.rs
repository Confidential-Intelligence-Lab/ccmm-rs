use std::hint::black_box;
use std::time::Instant;

use ccmm_rs::ring::{
    composite_modulus_big, composite_rescale_reference_coefficients, reconstruct_coefficients_big,
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

fn median_ns<F, T>(mut operation: F) -> f64
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

fn layout(name: &str) -> Vec<usize> {
    match name {
        "single" => vec![1; 8],
        "paired" => vec![2; 4],
        "mixed" => vec![2, 1, 2, 3],
        _ => panic!("unknown layout: {name}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 3, "usage: binary DEGREE LAYOUT");

    let degree: usize = args[1].parse().expect("degree");
    assert!([256, 1024, 4096, 8192, 16384, 32768, 65536].contains(&degree));

    let name = args[2].as_str();
    let groups = layout(name);

    let moduli: Vec<Modulus> = PRIMES.iter().copied().map(Modulus::new).collect();
    let basis = ModulusBasis::new(moduli.clone());
    let chain = CompositeModulusChain::new(basis.clone(), groups);

    let q = composite_modulus_big(&basis);
    let coefficients: Vec<BigUint> = (0..degree)
        .map(|i| {
            let x = BigUint::from(i as u64 + 1);
            let value = &x * &x * BigUint::from(1234567_u64);
            (&q - BigUint::from(1_u64) - value) % &q
        })
        .collect();

    let mut current = rns_from_big_coefficients(moduli, &coefficients);
    let mut exact = coefficients;
    let mut cumulative_divisor = BigUint::from(1_u8);
    let mut total_production_ns = 0.0_f64;
    let mut total_reference_ns = 0.0_f64;

    for level in 0..chain.max_level() {
        let divisor = chain.rescale_divisor_big(level).expect("divisor");
        let next_q = chain.active_composite_modulus_big(level + 1);

        let expected: Vec<BigUint> = exact.iter().map(|value| value / &divisor).collect();

        let reference = composite_rescale_reference_coefficients(&current, &chain, level);

        assert_eq!(reference, expected);

        let production = rescale_composite_level_to_next(&current, &chain, level);

        assert_eq!(production.basis(), &chain.active_basis(level + 1));
        assert_eq!(reconstruct_coefficients_big(&production), expected);

        cumulative_divisor *= &divisor;
        assert_eq!(&q / &cumulative_divisor, next_q);

        let production_ns = median_ns(|| {
            rescale_composite_level_to_next(black_box(&current), black_box(&chain), level)
        });

        let reference_ns = median_ns(|| {
            composite_rescale_reference_coefficients(black_box(&current), black_box(&chain), level)
        });

        total_production_ns += production_ns;
        total_reference_ns += reference_ns;

        println!(
            "LEVEL,{degree},{name},{level},{},{},{},\
             {production_ns:.3},{reference_ns:.3},PASS",
            chain.active_composite_modulus_big(level).bits(),
            divisor.bits(),
            next_q.bits(),
        );

        current = production;
        exact = expected;
    }

    assert_eq!(chain.max_level(), chain.logical_level_count() - 1);
    assert_eq!(reconstruct_coefficients_big(&current), exact);

    println!(
        "SUMMARY,{degree},{name},{},{total_production_ns:.3},\
         {total_reference_ns:.3},PASS",
        chain.max_level(),
    );
    println!("COMPOSITE_RESCALE_CHAIN_STATUS=PASS");
}
