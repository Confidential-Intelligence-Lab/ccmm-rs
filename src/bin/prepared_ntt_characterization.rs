use std::hint::black_box;
use std::time::Instant;

use ccmm_rs::ckks::{
    research_profile_16384, research_profile_32768, research_profile_4096, research_profile_65536,
    research_profile_8192, CkksParameterProfile,
};
use ccmm_rs::ring::{make_ntt_plan, Modulus, Polynomial, PreparedNttPlan};

fn select_profile() -> CkksParameterProfile {
    let args: Vec<String> = std::env::args().collect();
    let degree = match args.as_slice() {
        [_] => 4096,
        [_, flag, value] if flag == "--profile" => {
            value.parse::<usize>().expect("Invalid profile degree")
        }
        _ => panic!("Usage: prepared_ntt_characterization [--profile DEGREE]"),
    };

    match degree {
        4096 => research_profile_4096(),
        8192 => research_profile_8192(),
        16384 => research_profile_16384(),
        32768 => research_profile_32768(),
        65536 => research_profile_65536(),
        _ => panic!("Unsupported profile degree: {degree}"),
    }
}

const WARMUPS: usize = 5;
const REPETITIONS: usize = 30;

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

fn elapsed_ns<F, T>(mut f: F) -> f64
where
    F: FnMut() -> T,
{
    let start = Instant::now();
    black_box(f());
    start.elapsed().as_secs_f64() * 1e9
}

fn summarize(mut samples: Vec<f64>) -> (f64, f64) {
    assert_eq!(samples.len(), REPETITIONS);
    samples.sort_by(|a, b| a.total_cmp(b));

    let median = (samples[14] + samples[15]) / 2.0;
    let p95 = samples[28];

    (median, p95)
}

fn main() {
    let profile = select_profile();
    let degree = profile.degree();
    let moduli = profile.modulus_values();

    assert!(!profile.security_bearing());

    println!("PREPARED_NTT_VERSION=2");
    println!("PROFILE={}", profile.name());
    println!("DEGREE={degree}");
    println!("RNS_LIMBS={}", moduli.len());
    println!("MODULUS_BITS_SUM={}", profile.total_modulus_bits());
    println!("WARMUPS={WARMUPS}");
    println!("REPETITIONS={REPETITIONS}");
    println!("SECURITY_CLAIM_SCOPE=research_only_unqualified");
    println!("SECURITY_BEARING=false");
    println!("FULL_IMPLEMENTATION_QUALIFIED=false");

    for (limb, &q) in moduli.iter().enumerate() {
        assert_eq!((q - 1) % (2 * degree as u64), 0);

        let modulus = Modulus::new(q);

        let reference_start = Instant::now();
        let reference = make_ntt_plan(modulus, degree);
        let reference_setup_ns = reference_start.elapsed().as_secs_f64() * 1e9;

        let prepared_start = Instant::now();
        let prepared = PreparedNttPlan::new(&reference);
        let prepared_setup_ns = prepared_start.elapsed().as_secs_f64() * 1e9;

        let input = polynomial(modulus, degree, 7);

        let reference_spectrum = reference.forward_radix2(&input);
        let prepared_spectrum = prepared.forward(&input);

        assert_eq!(
            prepared_spectrum, reference_spectrum,
            "Forward NTT backend mismatch"
        );

        assert_eq!(reference.inverse_radix2(&reference_spectrum), input);
        assert_eq!(prepared.inverse(&prepared_spectrum), input);

        let mut reference_forward = Vec::with_capacity(REPETITIONS);
        let mut prepared_forward = Vec::with_capacity(REPETITIONS);
        let mut reference_inverse = Vec::with_capacity(REPETITIONS);
        let mut prepared_inverse = Vec::with_capacity(REPETITIONS);

        for iteration in 0..(WARMUPS + REPETITIONS) {
            // Alternate backend order to reduce systematic timing bias.
            let reverse = iteration % 2 == 1;

            let (rf, pf, ri, pi) = if reverse {
                let pf = elapsed_ns(|| prepared.forward(black_box(&input)));
                let rf = elapsed_ns(|| reference.forward_radix2(black_box(&input)));
                let pi = elapsed_ns(|| prepared.inverse(black_box(&prepared_spectrum)));
                let ri = elapsed_ns(|| reference.inverse_radix2(black_box(&reference_spectrum)));
                (rf, pf, ri, pi)
            } else {
                let rf = elapsed_ns(|| reference.forward_radix2(black_box(&input)));
                let pf = elapsed_ns(|| prepared.forward(black_box(&input)));
                let ri = elapsed_ns(|| reference.inverse_radix2(black_box(&reference_spectrum)));
                let pi = elapsed_ns(|| prepared.inverse(black_box(&prepared_spectrum)));
                (rf, pf, ri, pi)
            };

            if iteration >= WARMUPS {
                reference_forward.push(rf);
                prepared_forward.push(pf);
                reference_inverse.push(ri);
                prepared_inverse.push(pi);
            }
        }

        let (rf_med, rf_p95) = summarize(reference_forward);
        let (pf_med, pf_p95) = summarize(prepared_forward);
        let (ri_med, ri_p95) = summarize(reference_inverse);
        let (pi_med, pi_p95) = summarize(prepared_inverse);

        println!("LIMB={limb}");
        println!("MODULUS={q}");
        println!("REFERENCE_SETUP_NS={reference_setup_ns:.3}");
        println!("PREPARED_SETUP_NS={prepared_setup_ns:.3}");
        println!("REFERENCE_FORWARD_MEDIAN_NS={rf_med:.3}");
        println!("PREPARED_FORWARD_MEDIAN_NS={pf_med:.3}");
        println!("FORWARD_SPEEDUP={:.6}", rf_med / pf_med);
        println!("REFERENCE_FORWARD_P95_NS={rf_p95:.3}");
        println!("PREPARED_FORWARD_P95_NS={pf_p95:.3}");
        println!("REFERENCE_INVERSE_MEDIAN_NS={ri_med:.3}");
        println!("PREPARED_INVERSE_MEDIAN_NS={pi_med:.3}");
        println!("INVERSE_SPEEDUP={:.6}", ri_med / pi_med);
        println!("REFERENCE_INVERSE_P95_NS={ri_p95:.3}");
        println!("PREPARED_INVERSE_P95_NS={pi_p95:.3}");
        println!("CORRECTNESS=PASS");
    }

    println!("PREPARED_NTT_STATUS=PASS");
}
