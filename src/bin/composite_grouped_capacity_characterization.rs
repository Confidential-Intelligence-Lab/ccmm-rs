use std::hint::black_box;
use std::time::Instant;

use ccmm_rs::ring::{
    composite_modulus_big, reconstruct_coefficients_big, rescale_composite_level_to_next,
    rns_from_big_coefficients, CompositeModulusChain, Modulus, ModulusBasis,
};
use num_bigint::BigUint;

const WARMUPS: usize = 3;
const REPEATS: usize = 15;

const PRIMES_30: [u64; 8] = [
    1073479681, 1071513601, 1070727169, 1068236801, 1065484289, 1064697857, 1062862849, 1062469633,
];
const PRIMES_35: [u64; 8] = [
    34359214081,
    34357116929,
    34356068353,
    34353184769,
    34352398337,
    34351742977,
    34350039041,
    34346106881,
];
const PRIMES_40: [u64; 8] = [
    1099510054913,
    1099507695617,
    1099506515969,
    1099504549889,
    1099503894529,
    1099503370241,
    1099502714881,
    1099500617729,
];
const PRIMES_50: [u64; 8] = [
    1125899903827969,
    1125899902124033,
    1125899887312897,
    1125899886395393,
    1125899885740033,
    1125899884167169,
    1125899884036097,
    1125899883642881,
];
const PRIMES_55: [u64; 8] = [
    36028797014376449,
    36028797013327873,
    36028797010444289,
    36028797005856769,
    36028797001138177,
    36028796997599233,
    36028796996681729,
    36028796992749569,
];

fn median_ns<F, T>(mut f: F) -> f64
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
        samples.push(start.elapsed().as_secs_f64() * 1e9);
    }

    samples.sort_by(|a, b| a.total_cmp(b));
    samples[REPEATS / 2]
}

fn primes(bits: usize) -> &'static [u64; 8] {
    match bits {
        30 => &PRIMES_30,
        35 => &PRIMES_35,
        40 => &PRIMES_40,
        50 => &PRIMES_50,
        55 => &PRIMES_55,
        _ => panic!("unsupported prime width"),
    }
}

fn grouping(name: &str, limbs: usize) -> Vec<usize> {
    match name {
        "paired" => {
            if limbs == 2 {
                vec![1, 1]
            } else {
                let mut groups = vec![2; limbs / 2];
                if limbs % 2 != 0 {
                    groups.push(1);
                }
                groups
            }
        }
        "mixed" => match limbs {
            2 => vec![1, 1],
            3 => vec![1, 2],
            4 => vec![1, 1, 2],
            6 => vec![2, 1, 3],
            8 => vec![2, 1, 2, 3],
            _ => panic!("unsupported mixed grouping"),
        },
        _ => panic!("unsupported layout"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "usage: binary DEGREE PRIME_BITS LIMBS LAYOUT"
    );

    let degree: usize = args[1].parse().expect("degree");
    let bits: usize = args[2].parse().expect("prime bits");
    let limbs: usize = args[3].parse().expect("limbs");
    let name = args[4].as_str();

    assert!([2048, 4096, 8192, 16384, 32768, 65536].contains(&degree));
    assert!([2, 3, 4, 6, 8].contains(&limbs));

    let groups = grouping(name, limbs);
    assert_eq!(groups.iter().sum::<usize>(), limbs);
    assert!(groups.len() >= 2);

    let moduli: Vec<Modulus> = primes(bits)[..limbs]
        .iter()
        .copied()
        .map(Modulus::new)
        .collect();

    let basis = ModulusBasis::new(moduli.clone());
    let chain = CompositeModulusChain::new(basis.clone(), groups.clone());
    let q = composite_modulus_big(&basis);

    let coefficients: Vec<BigUint> = (0..degree)
        .map(|i| {
            let x = BigUint::from(i as u64 + 1);
            let value = &x * &x * BigUint::from(1234567_u64);
            (&q - BigUint::from(1_u64) - value) % &q
        })
        .collect();

    let input = rns_from_big_coefficients(moduli, &coefficients);
    let mut current = input.clone();
    let mut exact = coefficients;
    let mut sum_transition_ns = 0.0_f64;

    for level in 0..chain.max_level() {
        let divisor = chain.rescale_divisor_big(level).expect("divisor");
        let dropped = chain
            .dropped_logical_level(level)
            .expect("dropped logical level")
            .physical_limb_count();

        let expected: Vec<BigUint> = exact.iter().map(|value| value / &divisor).collect();

        let production = rescale_composite_level_to_next(&current, &chain, level);

        assert_eq!(production.basis(), &chain.active_basis(level + 1));
        assert_eq!(reconstruct_coefficients_big(&production), expected);

        let elapsed_ns = median_ns(|| {
            rescale_composite_level_to_next(black_box(&current), black_box(&chain), level)
        });

        println!(
            "LEVEL,{degree},{bits},{limbs},{name},{level},{dropped},{},{},{elapsed_ns:.3},PASS",
            chain.active_composite_modulus_big(level).bits(),
            chain.active_composite_modulus_big(level + 1).bits(),
        );

        sum_transition_ns += elapsed_ns;
        current = production;
        exact = expected;
    }

    assert_eq!(reconstruct_coefficients_big(&current), exact);

    let chain_ns = median_ns(|| {
        let mut state = input.clone();
        for level in 0..chain.max_level() {
            state = rescale_composite_level_to_next(black_box(&state), black_box(&chain), level);
        }
        state
    });

    let group_text = groups
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("-");

    println!(
        "SUMMARY,{degree},{bits},{limbs},{name},{group_text},{},{},{sum_transition_ns:.3},{chain_ns:.3},PASS",
        q.bits(),
        chain.max_level(),
    );
    println!("COMPOSITE_GROUPED_CAPACITY_STATUS=PASS");
}
