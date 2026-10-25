use std::hint::black_box;
use std::time::Instant;

use ccmm_rs::ring::{make_ntt_plan, Modulus, Polynomial, PreparedNttPlan};

const DEGREES: [usize; 6] = [2048, 4096, 8192, 16384, 32768, 65536];
const MODULI: [(usize, [u64; 8]); 5] = [
    (
        30,
        [
            1073479681, 1071513601, 1070727169, 1068236801, 1065484289, 1064697857, 1062862849,
            1062469633,
        ],
    ),
    (
        35,
        [
            34359214081,
            34357116929,
            34356068353,
            34353184769,
            34352398337,
            34351742977,
            34350039041,
            34346106881,
        ],
    ),
    (
        40,
        [
            1099510054913,
            1099507695617,
            1099506515969,
            1099504549889,
            1099503894529,
            1099503370241,
            1099502714881,
            1099500617729,
        ],
    ),
    (
        50,
        [
            1125899903827969,
            1125899902124033,
            1125899887312897,
            1125899886395393,
            1125899885740033,
            1125899884167169,
            1125899884036097,
            1125899883642881,
        ],
    ),
    (
        55,
        [
            36028797014376449,
            36028797013327873,
            36028797010444289,
            36028797005856769,
            36028797001138177,
            36028796997599233,
            36028796996681729,
            36028796992749569,
        ],
    ),
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

fn dense(modulus: Modulus, q: u64, n: usize, seed: u64) -> Polynomial {
    Polynomial::new(
        modulus,
        (0..n)
            .map(|i| {
                let x = i as u128;
                ((seed as u128 + 17 * x + 5 * x * x + 3 * x * x * x) % q as u128) as u64
            })
            .collect(),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 4, "usage: binary DEGREE BITS LIMBS");

    let n: usize = args[1].parse().expect("degree");
    let bits: usize = args[2].parse().expect("bits");
    let limbs: usize = args[3].parse().expect("limbs");

    assert!(DEGREES.contains(&n));
    assert!((1..=8).contains(&limbs));

    let qs = MODULI
        .iter()
        .find(|(width, _)| *width == bits)
        .expect("unsupported modulus width")
        .1;

    let mut plans = Vec::new();
    let mut prepared_plans = Vec::new();
    let mut left = Vec::new();
    let mut right = Vec::new();

    let mut setup_ns = 0.0;
    let mut prepared_setup_ns = 0.0;
    let mut scalar_ns = 0.0;

    for (limb, &q) in qs.iter().take(limbs).enumerate() {
        assert_eq!((q - 1) % (2 * n as u64), 0);

        let modulus = Modulus::new(q);

        let start = Instant::now();
        let plan = make_ntt_plan(modulus, n);
        setup_ns += start.elapsed().as_secs_f64() * 1e9;

        let start = Instant::now();
        let prepared = PreparedNttPlan::new(&plan);
        prepared_setup_ns += start.elapsed().as_secs_f64() * 1e9;

        let lhs = dense(modulus, q, n, 11 + limb as u64);
        let rhs = dense(modulus, q, n, 31 + limb as u64);

        let spectrum = plan.forward_radix2(&lhs);
        assert_eq!(prepared.forward(&lhs), spectrum);
        assert_eq!(prepared.inverse(&spectrum), lhs);

        // Exact independent reference at the smallest ring degree.
        if n == 2048 {
            assert_eq!(plan.negacyclic_mul(&lhs, &rhs), lhs.negacyclic_mul(&rhs));
        }

        // Exact negacyclic identity and monomial checks at every degree.
        let mut one = vec![0_u64; n];
        one[0] = 1;
        let identity = Polynomial::new(modulus, one);
        assert_eq!(plan.negacyclic_mul(&lhs, &identity), lhs);

        let shift = n / 4;
        let mut monomial = vec![0_u64; n];
        monomial[shift] = 1;
        let monomial = Polynomial::new(modulus, monomial);

        let source: Vec<u64> = (0..n)
            .map(|i| {
                let x = i as u128;
                ((11_u128 + limb as u128 + 17 * x + 5 * x * x + 3 * x * x * x) % q as u128) as u64
            })
            .collect();

        let mut expected = vec![0_u64; n];
        for (i, &value) in source.iter().enumerate() {
            let destination = i + shift;
            if destination < n {
                expected[destination] = value;
            } else {
                expected[destination - n] = modulus.sub(0, value);
            }
        }
        assert_eq!(
            plan.negacyclic_mul(&lhs, &monomial),
            Polynomial::new(modulus, expected)
        );

        scalar_ns += measure(|| modulus.mul(black_box(1234567), black_box(7654321)));

        plans.push(plan);
        prepared_plans.push(prepared);
        left.push(lhs);
        right.push(rhs);
    }

    let forward_ns = measure(|| {
        (0..limbs)
            .map(|i| prepared_plans[i].forward(black_box(&left[i])))
            .collect::<Vec<_>>()
    });

    let spectra: Vec<_> = (0..limbs)
        .map(|i| prepared_plans[i].forward(&left[i]))
        .collect();

    let inverse_ns = measure(|| {
        (0..limbs)
            .map(|i| prepared_plans[i].inverse(black_box(&spectra[i])))
            .collect::<Vec<_>>()
    });

    let multiplication_ns = measure(|| {
        (0..limbs)
            .map(|i| plans[i].negacyclic_mul(black_box(&left[i]), black_box(&right[i])))
            .collect::<Vec<_>>()
    });

    println!(
        "RESULT,{n},{bits},{limbs},{setup_ns:.3},\
         {prepared_setup_ns:.3},{scalar_ns:.3},\
         {forward_ns:.3},{inverse_ns:.3},\
         {multiplication_ns:.3},PASS"
    );
    println!("PARAMETER_SWEEP_STATUS=PASS");
}
