use std::hint::black_box;
use std::time::Instant;

use ccmm_rs::ring::{make_ntt_plan, Modulus, Polynomial, PreparedNttPlan};

const DEGREES: [usize; 6] = [2048, 4096, 8192, 16384, 32768, 65536];
const MODULI: [u64; 3] = [34_359_214_081, 34_357_116_929, 34_356_068_353];
const WARMUPS: usize = 3;
const REPETITIONS: usize = 15;

fn polynomial(modulus: Modulus, degree: usize, offset: u64) -> Polynomial {
    let q = modulus.value() as u128;

    Polynomial::new(
        modulus,
        (0..degree)
            .map(|index| {
                let x = index as u128;
                ((offset as u128 + 17 * x + 5 * x * x + 3 * x * x * x) % q) as u64
            })
            .collect(),
    )
}

fn measure<F, T>(mut operation: F) -> (f64, f64)
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

    samples.sort_by(|a, b| a.total_cmp(b));

    (samples[REPETITIONS / 2], samples[14])
}

fn main() {
    println!("RING_SWEEP_VERSION=1");
    println!("WARMUPS={WARMUPS}");
    println!("REPETITIONS={REPETITIONS}");
    println!("SECURITY_CLAIM_SCOPE=ARITHMETIC_ONLY");
    println!("FULL_IMPLEMENTATION_QUALIFIED=false");

    let requested_degree = std::env::args().nth(1).map(|arg| {
        let degree: usize = arg.parse().expect("invalid ring degree");
        assert!(DEGREES.contains(&degree), "unsupported ring degree");
        degree
    });

    for degree in DEGREES {
        if requested_degree.is_some_and(|requested| requested != degree) {
            continue;
        }

        let mut plans = Vec::new();
        let mut inputs = Vec::new();
        let mut spectra = Vec::new();

        for (limb, &q) in MODULI.iter().enumerate() {
            assert_eq!((q - 1) % (2 * degree as u64), 0);

            let modulus = Modulus::new(q);
            let plan = make_ntt_plan(modulus, degree);
            let prepared = PreparedNttPlan::new(&plan);
            let input = polynomial(modulus, degree, 7);

            let reference = plan.forward_radix2(&input);
            let spectrum = prepared.forward(&input);

            assert_eq!(spectrum, reference);
            assert_eq!(prepared.inverse(&spectrum), input);

            let (reference_forward, _) = measure(|| plan.forward_radix2(black_box(&input)));
            let (prepared_forward, _) = measure(|| prepared.forward(black_box(&input)));

            let (reference_inverse, _) = measure(|| plan.inverse_radix2(black_box(&reference)));
            let (prepared_inverse, _) = measure(|| prepared.inverse(black_box(&spectrum)));

            println!("DEGREE={degree}");
            println!("LIMB={limb}");
            println!("MODULUS={q}");
            println!("REFERENCE_FORWARD_MEDIAN_NS={reference_forward:.3}");
            println!("PREPARED_FORWARD_MEDIAN_NS={prepared_forward:.3}");
            println!(
                "FORWARD_SPEEDUP={:.6}",
                reference_forward / prepared_forward
            );
            println!("REFERENCE_INVERSE_MEDIAN_NS={reference_inverse:.3}");
            println!("PREPARED_INVERSE_MEDIAN_NS={prepared_inverse:.3}");
            println!(
                "INVERSE_SPEEDUP={:.6}",
                reference_inverse / prepared_inverse
            );
            println!("CORRECTNESS=PASS");

            plans.push(prepared);
            inputs.push(input);
            spectra.push(spectrum);
        }

        let (rns_forward, rns_forward_p95) = measure(|| {
            (0..3)
                .map(|i| plans[i].forward(black_box(&inputs[i])))
                .collect::<Vec<_>>()
        });

        let (rns_inverse, rns_inverse_p95) = measure(|| {
            (0..3)
                .map(|i| plans[i].inverse(black_box(&spectra[i])))
                .collect::<Vec<_>>()
        });

        println!("DEGREE={degree}");
        println!("RNS_LIMBS=3");
        println!("RNS_FORWARD_MEDIAN_NS={rns_forward:.3}");
        println!("RNS_FORWARD_P95_NS={rns_forward_p95:.3}");
        println!("RNS_INVERSE_MEDIAN_NS={rns_inverse:.3}");
        println!("RNS_INVERSE_P95_NS={rns_inverse_p95:.3}");
        println!("RNS_CORRECTNESS=PASS");
    }

    println!("RING_SWEEP_STATUS=PASS");
}
