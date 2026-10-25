use std::fs::{self, File};
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use ccmm_rs::grafting::decrypt_rns_raw_with_ntt;
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big,
    PreparedRnsNttPlan,
};
use num_traits::ToPrimitive;

use ccmm_rs::ckks::{
    mod_switch_rns_ckks_to_next, multiply_plain_rns_ckks_with_ntt,
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next, research_profile_16384,
    research_profile_32768, research_profile_4096, research_profile_65536, research_profile_8192,
    CkksCanonicalEmbedding, CkksChainState, PreparedRnsPlaintextNtt, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::grafting::encrypt_rns_raw_with_distribution_ntt_rng;
use ccmm_rs::ring::{ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn encode_rns(
    slots: &[Complex64],
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    let coefficients = embedding.slots_to_coefficients(slots);

    let residues = basis
        .moduli()
        .iter()
        .copied()
        .map(|modulus| {
            let q = i128::from(modulus.value());
            Polynomial::new(
                modulus,
                coefficients
                    .iter()
                    .map(|&value| {
                        let signed = (value * scale).round() as i128;
                        signed.rem_euclid(q) as u64
                    })
                    .collect(),
            )
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

fn decode_slots(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let plan = RnsNttPlan::new(
        ciphertext.basis().moduli().to_vec(),
        ciphertext.rlwe().degree(),
    );

    let plaintext = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);

    let modulus = composite_modulus_big(plaintext.basis());

    let coefficients: Vec<f64> = reconstruct_coefficients_big(&plaintext)
        .iter()
        .map(|value| {
            centered_representative_big(value, &modulus)
                .to_f64()
                .expect("CKKS coefficient conversion failed")
                / ciphertext.scale()
        })
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

fn validate_slots(name: &str, actual: &[Complex64], expected: &[Complex64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());

    let mut max_error = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    for (&a, &e) in actual.iter().zip(expected) {
        let error = (a - e).norm();
        assert!(error.is_finite(), "{name}: non-finite error");
        max_error = max_error.max(error);
        sum_sq += error * error;
    }

    let rms_error = (sum_sq / actual.len() as f64).sqrt();

    println!(
        "R30_13B2_ACCURACY OPERATION={name} MAX_ERROR={max_error:.12e} RMS_ERROR={rms_error:.12e} TOLERANCE={tolerance:.12e}"
    );

    assert!(
        max_error <= tolerance,
        "{name}: maximum error {max_error:e} exceeds {tolerance:e}"
    );
}

fn assert_exact_ciphertext(lhs: &RnsCkksCiphertext, rhs: &RnsCkksCiphertext) {
    assert_eq!(lhs.level(), rhs.level());
    assert_eq!(lhs.scale(), rhs.scale());
    assert_eq!(lhs.basis(), rhs.basis());
    assert_eq!(lhs.rlwe().limbs().len(), rhs.rlwe().limbs().len());

    for (a, b) in lhs.rlwe().limbs().iter().zip(rhs.rlwe().limbs()) {
        assert_eq!(a.a(), b.a());
        assert_eq!(a.b(), b.b());
    }

    println!("R30_13B2_PREPARED_CP_EXACT_EQUALITY=PASS");
}

fn elapsed_ms<F, T>(operation: F) -> f64
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = operation();
    let elapsed = start.elapsed().as_secs_f64() * 1.0e3;
    black_box(result);
    elapsed
}

fn percentile(samples: &[f64], percentile: usize) -> f64 {
    assert!(!samples.is_empty());
    assert!((1..=100).contains(&percentile));

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let index = (sorted.len() * percentile).div_ceil(100).saturating_sub(1);

    sorted[index]
}

fn median(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty());

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let n = sorted.len();

    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn env_usize(name: &str, default: usize, max: usize) -> usize {
    let value = std::env::var(name)
        .ok()
        .map(|v| {
            v.parse::<usize>()
                .expect("Invalid integer environment variable")
        })
        .unwrap_or(default);

    assert!(value > 0 && value <= max, "{name} outside allowed range");
    value
}

fn main() {
    let degree_requested: usize = std::env::var("R30_13B_DEGREE")
        .unwrap_or_else(|_| "4096".to_string())
        .parse()
        .expect("R30_13B_DEGREE must be an integer");

    let profile = match degree_requested {
        4096 => research_profile_4096(),
        8192 => research_profile_8192(),
        16384 => research_profile_16384(),
        32768 => research_profile_32768(),
        65536 => research_profile_65536(),
        _ => panic!("Unsupported R30_13B_DEGREE: {degree_requested}"),
    };

    println!("R30_13B3_PROFILE_SELECTION=PASS");
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let basis = chain.top().clone();

    assert!(!profile.security_bearing());
    assert!(chain.len() >= 2);

    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);

    let slots: Vec<Complex64> = (0..profile.slot_count())
        .map(|index| Complex64::new(0.10 + index as f64 * 0.00001, 0.0))
        .collect();

    let plaintext = encode_rns(&slots, &embedding, &basis, scale);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x3013_B001);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };
    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x3013_B002);

    let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        distribution,
        &secret,
        &plan,
        &mut encryption_rng,
    );

    let ciphertext = RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, scale), &chain);

    let keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &keys);

    let warmups: usize = std::env::var("R30_13B_WARMUPS")
        .unwrap_or_else(|_| "3".to_string())
        .parse()
        .expect("Invalid R30_13B_WARMUPS");

    let repeats: usize = std::env::var("R30_13B_REPEATS")
        .unwrap_or_else(|_| "20".to_string())
        .parse()
        .expect("Invalid R30_13B_REPEATS");

    assert!(repeats > 0 && repeats <= 1000);
    assert!(warmups <= 1000);

    println!("R30_13B_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("RING_DEGREE={degree}");
    println!("RNS_LIMBS={}", basis.len());
    println!("SECURITY_CLASSIFICATION=RESEARCH_ONLY");
    println!("SECURITY_QUALIFIED=false");
    println!("R30_13C_LEGACY_WARMUPS={warmups}");
    println!("R30_13C_LEGACY_REPEATS={repeats}");

    println!("R30_13C_VERSION=1");
    println!("R30_13C_SECURITY_CLASSIFICATION=RESEARCH_ONLY");

    let added = evaluator.add(&ciphertext, &ciphertext);
    assert_eq!(added.level(), ciphertext.level());
    assert_eq!(added.scale(), ciphertext.scale());

    let multiplied =
        multiply_plain_rns_ckks_with_ntt(&ciphertext, &plaintext, scale, &chain, &plan);
    assert_eq!(multiplied.level(), ciphertext.level());

    let rescaled = rescale_rns_ckks_to_next(&ciphertext, &chain);
    assert_eq!(rescaled.level(), ciphertext.level() + 1);

    let switched = mod_switch_rns_ckks_to_next(&ciphertext, &chain);
    assert_eq!(switched.level(), ciphertext.level() + 1);
    assert_eq!(switched.scale(), ciphertext.scale());

    println!("R30_13B2_VERSION=1");

    const ACCURACY_TOLERANCE: f64 = 2.0e-3;

    let expected_add: Vec<Complex64> = slots.iter().map(|&x| x + x).collect();

    validate_slots(
        "add_cc",
        &decode_slots(&added, &secret, &embedding),
        &expected_add,
        ACCURACY_TOLERANCE,
    );

    let expected_cp: Vec<Complex64> = slots.iter().map(|&x| x * x).collect();

    validate_slots(
        "multiply_cp_ntt",
        &decode_slots(&multiplied, &secret, &embedding),
        &expected_cp,
        ACCURACY_TOLERANCE,
    );

    let prepared_plan = PreparedRnsNttPlan::new(&plan);
    let prepared_plaintext = PreparedRnsPlaintextNtt::new(&plaintext, &prepared_plan);

    let prepared_result = multiply_plain_rns_ckks_with_prepared_ntt(
        &ciphertext,
        &prepared_plaintext,
        scale,
        &chain,
        &prepared_plan,
    );

    assert_exact_ciphertext(&multiplied, &prepared_result);

    validate_slots(
        "multiply_cp_prepared",
        &decode_slots(&prepared_result, &secret, &embedding),
        &expected_cp,
        ACCURACY_TOLERANCE,
    );

    println!("R30_13B2_NUMERICAL_VALIDATION=PASS");

    let paired_warmups = env_usize("R30_13C_WARMUPS", 5, 100);
    let paired_trials = env_usize("R30_13C_TRIALS", 30, 1000);

    let output_dir = PathBuf::from(
        std::env::var("R30_13C_OUTPUT_DIR")
            .unwrap_or_else(|_| "results/benchmarks/r30_13c".to_string()),
    );

    fs::create_dir_all(&output_dir).expect("Cannot create R30.13c output directory");

    let csv_path = output_dir.join(format!("paired_cp_n{}_l{}.csv", degree, basis.len()));

    assert!(
        !csv_path.exists(),
        "Output already exists; choose a fresh R30_13C_OUTPUT_DIR"
    );

    let file = File::create(&csv_path).expect("Cannot create paired benchmark CSV");

    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "degree,limbs,trial,order,ordinary_ms,prepared_ms,paired_speedup"
    )
    .unwrap();

    for trial in 0..paired_warmups {
        if trial % 2 == 0 {
            black_box(multiply_plain_rns_ckks_with_ntt(
                &ciphertext,
                &plaintext,
                scale,
                &chain,
                &plan,
            ));
            black_box(multiply_plain_rns_ckks_with_prepared_ntt(
                &ciphertext,
                &prepared_plaintext,
                scale,
                &chain,
                &prepared_plan,
            ));
        } else {
            black_box(multiply_plain_rns_ckks_with_prepared_ntt(
                &ciphertext,
                &prepared_plaintext,
                scale,
                &chain,
                &prepared_plan,
            ));
            black_box(multiply_plain_rns_ckks_with_ntt(
                &ciphertext,
                &plaintext,
                scale,
                &chain,
                &plan,
            ));
        }
    }

    let mut ordinary_samples = Vec::with_capacity(paired_trials);
    let mut prepared_samples = Vec::with_capacity(paired_trials);
    let mut speedups = Vec::with_capacity(paired_trials);

    for trial in 0..paired_trials {
        let ordinary_first = trial % 2 == 0;

        let (ordinary_ms, prepared_ms) = if ordinary_first {
            (
                elapsed_ms(|| {
                    multiply_plain_rns_ckks_with_ntt(
                        black_box(&ciphertext),
                        black_box(&plaintext),
                        scale,
                        &chain,
                        &plan,
                    )
                }),
                elapsed_ms(|| {
                    multiply_plain_rns_ckks_with_prepared_ntt(
                        black_box(&ciphertext),
                        black_box(&prepared_plaintext),
                        scale,
                        &chain,
                        &prepared_plan,
                    )
                }),
            )
        } else {
            let prepared = elapsed_ms(|| {
                multiply_plain_rns_ckks_with_prepared_ntt(
                    black_box(&ciphertext),
                    black_box(&prepared_plaintext),
                    scale,
                    &chain,
                    &prepared_plan,
                )
            });

            let ordinary = elapsed_ms(|| {
                multiply_plain_rns_ckks_with_ntt(
                    black_box(&ciphertext),
                    black_box(&plaintext),
                    scale,
                    &chain,
                    &plan,
                )
            });

            (ordinary, prepared)
        };

        assert!(ordinary_ms.is_finite() && ordinary_ms > 0.0);
        assert!(prepared_ms.is_finite() && prepared_ms > 0.0);

        let speedup = ordinary_ms / prepared_ms;

        ordinary_samples.push(ordinary_ms);
        prepared_samples.push(prepared_ms);
        speedups.push(speedup);

        writeln!(
            writer,
            "{degree},{},{trial},{},{ordinary_ms:.9},{prepared_ms:.9},{speedup:.9}",
            basis.len(),
            if ordinary_first {
                "ordinary_first"
            } else {
                "prepared_first"
            }
        )
        .unwrap();
    }

    writer.flush().expect("Cannot flush paired CSV");

    println!("R30_13C2_VERSION=1");
    println!("R30_13C2_WARMUPS={paired_warmups}");
    println!("R30_13C2_TRIALS={paired_trials}");

    let mut plan_samples = Vec::with_capacity(paired_trials);
    let mut preparation_samples = Vec::with_capacity(paired_trials);

    for trial in 0..(paired_warmups + paired_trials) {
        let plan_ms = elapsed_ms(|| PreparedRnsNttPlan::new(black_box(&plan)));

        let preparation_ms =
            elapsed_ms(|| PreparedRnsPlaintextNtt::new(black_box(&plaintext), &prepared_plan));

        if trial >= paired_warmups {
            plan_samples.push(plan_ms);
            preparation_samples.push(preparation_ms);
        }
    }

    let plan_median = median(&plan_samples);
    let preparation_median = median(&preparation_samples);
    let setup_median_sum = plan_median + preparation_median;

    let ordinary_median_for_model = median(&ordinary_samples);
    let prepared_median_for_model = median(&prepared_samples);

    let amortization_path =
        output_dir.join(format!("amortization_n{}_l{}.csv", degree, basis.len()));

    assert!(
        !amortization_path.exists(),
        "Amortization CSV already exists; use a fresh output directory"
    );

    let amortization_file =
        File::create(&amortization_path).expect("Cannot create amortization CSV");

    let mut amortization_writer = BufWriter::new(amortization_file);

    writeln!(
        amortization_writer,
        "degree,limbs,reuses,plan_median_ms,preparation_median_ms,ordinary_median_ms,prepared_median_ms,modeled_speedup"
    ).unwrap();

    for reuses in [1_usize, 2, 4, 8, 16, 32] {
        let baseline = reuses as f64 * ordinary_median_for_model;
        let optimized = setup_median_sum + reuses as f64 * prepared_median_for_model;

        assert!(optimized > 0.0);

        let modeled_speedup = baseline / optimized;

        writeln!(
            amortization_writer,
            "{degree},{},{reuses},{plan_median:.9},{preparation_median:.9},{ordinary_median_for_model:.9},{prepared_median_for_model:.9},{modeled_speedup:.9}",
            basis.len()
        ).unwrap();

        println!(
            "R30_13C2_AMORTIZATION DEGREE={degree} REUSES={reuses} MODELED_SPEEDUP={modeled_speedup:.6}"
        );
    }

    amortization_writer
        .flush()
        .expect("Cannot flush amortization CSV");

    println!(
        "R30_13C2_SETUP DEGREE={degree} PLAN_MEDIAN_MS={plan_median:.6} PREPARATION_MEDIAN_MS={preparation_median:.6} SETUP_MEDIAN_SUM_MS={setup_median_sum:.6}"
    );

    println!("R30_13C2_AMORTIZATION_CSV={}", amortization_path.display());

    println!("R30_13C2_STATUS=PASS");

    let ordinary_median = median(&ordinary_samples);
    let prepared_median = median(&prepared_samples);

    println!(
        "R30_13C_RESULT DEGREE={degree} LIMBS={} TRIALS={paired_trials} ORDINARY_MEDIAN_MS={ordinary_median:.6} PREPARED_MEDIAN_MS={prepared_median:.6} SPEEDUP_RATIO={:.6} PAIRED_SPEEDUP_MEDIAN={:.6} ORDINARY_P95_MS={:.6} PREPARED_P95_MS={:.6}",
        basis.len(),
        ordinary_median / prepared_median,
        median(&speedups),
        percentile(&ordinary_samples, 95),
        percentile(&prepared_samples, 95),
    );

    println!("R30_13C_CSV={}", csv_path.display());
    println!("R30_13C_STATUS=PASS");
}
