use std::hint::black_box;
use std::time::Instant;

use ccmm_rs::ckks::{
    research_profile_16384, research_profile_32768, research_profile_4096, research_profile_65536,
    research_profile_8192, CkksParameterProfile,
};
use ccmm_rs::ring::{make_ntt_plan, Modulus, Polynomial, PreparedNttPlan};

const WARMUPS: usize = 5;
const REPETITIONS: usize = 30;

fn select_profile(degree: usize) -> CkksParameterProfile {
    match degree {
        4096 => research_profile_4096(),
        8192 => research_profile_8192(),
        16384 => research_profile_16384(),
        32768 => research_profile_32768(),
        65536 => research_profile_65536(),
        _ => panic!("Unsupported degree: {degree}"),
    }
}

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

fn summarize(mut samples: Vec<f64>) -> (f64, f64) {
    assert_eq!(samples.len(), REPETITIONS);
    samples.sort_by(|a, b| a.total_cmp(b));
    ((samples[14] + samples[15]) / 2.0, samples[28])
}

fn emit(label: &str, samples: Vec<f64>) {
    let (median, p95) = summarize(samples);
    println!("{label}_MEDIAN_NS={median:.3}");
    println!("{label}_P95_NS={p95:.3}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    assert_eq!(
        args.len(),
        5,
        "Usage: lower_layer_campaign --profile DEGREE --mode streaming|resident"
    );
    assert_eq!(args[1], "--profile");
    assert_eq!(args[3], "--mode");

    let degree: usize = args[2].parse().expect("Invalid degree");
    let mode = args[4].as_str();

    assert!(matches!(mode, "streaming" | "resident"));

    let profile = select_profile(degree);
    assert!(!profile.security_bearing());

    let moduli = profile.modulus_values();

    println!("LOWER_LAYER_CAMPAIGN_VERSION=2");
    println!("PROFILE={}", profile.name());
    println!("DEGREE={degree}");
    println!("MODE={mode}");
    println!("RNS_LIMBS={}", moduli.len());
    println!("MODULUS_BITS_SUM={}", profile.total_modulus_bits());
    println!("WARMUPS={WARMUPS}");
    println!("REPETITIONS={REPETITIONS}");
    println!("SECURITY_CLAIM_SCOPE=research_only_unqualified");
    println!("FULL_IMPLEMENTATION_QUALIFIED=false");

    let setup_start = Instant::now();

    if mode == "resident" {
        let mut plans = Vec::new();
        let mut inputs = Vec::new();
        let mut spectra = Vec::new();

        for &q in moduli {
            let modulus = Modulus::new(q);
            let reference = make_ntt_plan(modulus, degree);
            let prepared = PreparedNttPlan::new(&reference);
            let input = polynomial(modulus, degree, 7);
            let spectrum = prepared.forward(&input);

            assert_eq!(reference.forward_radix2(&input), spectrum);
            assert_eq!(prepared.inverse(&spectrum), input);

            plans.push(prepared);
            inputs.push(input);
            spectra.push(spectrum);
        }

        println!("SETUP_MS={:.6}", setup_start.elapsed().as_secs_f64() * 1e3);

        emit(
            "RNS_FORWARD",
            measure(|| {
                plans
                    .iter()
                    .zip(&inputs)
                    .map(|(plan, input)| plan.forward(black_box(input)))
                    .collect::<Vec<_>>()
            }),
        );

        emit(
            "RNS_INVERSE",
            measure(|| {
                plans
                    .iter()
                    .zip(&spectra)
                    .map(|(plan, spectrum)| plan.inverse(black_box(spectrum)))
                    .collect::<Vec<_>>()
            }),
        );
    } else {
        let mut setup_ns = 0.0;

        for &q in moduli {
            let modulus = Modulus::new(q);

            let start = Instant::now();
            let reference = make_ntt_plan(modulus, degree);
            let prepared = PreparedNttPlan::new(&reference);
            let input = polynomial(modulus, degree, 7);
            let spectrum = prepared.forward(&input);
            setup_ns += start.elapsed().as_secs_f64() * 1e9;

            assert_eq!(reference.forward_radix2(&input), spectrum);
            assert_eq!(prepared.inverse(&spectrum), input);
        }

        println!("SETUP_MS={:.6}", setup_ns / 1e6);

        emit(
            "RNS_FORWARD",
            measure(|| {
                moduli
                    .iter()
                    .map(|&q| {
                        let modulus = Modulus::new(q);
                        let reference = make_ntt_plan(modulus, degree);
                        let prepared = PreparedNttPlan::new(&reference);
                        let input = polynomial(modulus, degree, 7);
                        prepared.forward(&input)
                    })
                    .collect::<Vec<_>>()
            }),
        );

        emit(
            "RNS_INVERSE",
            measure(|| {
                moduli
                    .iter()
                    .map(|&q| {
                        let modulus = Modulus::new(q);
                        let reference = make_ntt_plan(modulus, degree);
                        let prepared = PreparedNttPlan::new(&reference);
                        let input = polynomial(modulus, degree, 7);
                        let spectrum = prepared.forward(&input);
                        prepared.inverse(&spectrum)
                    })
                    .collect::<Vec<_>>()
            }),
        );
    }

    println!("RNS_CORRECTNESS=PASS");
    println!("LOWER_LAYER_CAMPAIGN_STATUS=PASS");
}
