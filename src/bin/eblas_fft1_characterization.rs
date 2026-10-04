use ccmm_rs::ckks::{
    research_profile_16384, research_profile_32768, research_profile_4096, research_profile_8192,
    CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext, RnsCkksEvaluationKeys,
    RnsCkksEvaluator,
};
use ccmm_rs::eblas::fft::{
    execute_fft1_plan, execute_fft1_plan_cp, Fft1Plan, Fft1Shape, FftDirection,
};
use ccmm_rs::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng};
use ccmm_rs::matrix::RnsCkksCiphertextMatrix;
use ccmm_rs::ring::{ModulusBasis, ModulusChain, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::time::Instant;

const SIGMA: f64 = 3.19;
const TOLERANCE: f64 = 3.0e-3;

fn encode_rns(
    value: f64,
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    let slots = vec![Complex64::new(value, 0.0); embedding.slot_count()];
    let coefficients = embedding.slots_to_coefficients(&slots);

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
                    .map(|&x| ((x * scale).round() as i128).rem_euclid(q) as u64)
                    .collect(),
            )
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

fn centered(value: u128, modulus: u128) -> i128 {
    if value > modulus / 2 {
        value as i128 - modulus as i128
    } else {
        value as i128
    }
}

#[allow(clippy::too_many_arguments)]
fn encrypt_scalar(
    value: f64,
    seed: u64,
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
    secret: &[i8],
    plan: &RnsNttPlan,
    chain: &ModulusChain,
) -> RnsCkksCiphertextMatrix {
    let plaintext = encode_rns(value, embedding, basis, scale);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        secret,
        plan,
        &mut rng,
    );

    let ciphertext = RnsCkksCiphertext::new(rlwe, CkksChainState::top(chain, scale), chain);

    RnsCkksCiphertextMatrix::from_vec_column_major(1, 1, vec![ciphertext])
}

fn decode_complex_scalar(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Complex64 {
    let plan = RnsNttPlan::new(
        ciphertext.basis().moduli().to_vec(),
        ciphertext.rlwe().degree(),
    );

    let plaintext = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);
    let modulus = plaintext.composite_modulus();

    let coefficients: Vec<f64> = plaintext
        .reconstruct_coefficients()
        .into_iter()
        .map(|x| centered(x, modulus) as f64 / ciphertext.scale())
        .collect();

    let slots = embedding.coefficients_to_slots(&coefficients);

    slots.iter().copied().sum::<Complex64>() / slots.len() as f64
}

fn deterministic_input(length: usize, salt: usize) -> Vec<Complex64> {
    let amplitude = 0.125 / length as f64;

    (0..length)
        .map(|index| {
            let raw = ((index * 17 + salt * 11) % 31) as f64;
            let normalized = (raw - 15.0) / 15.0;
            Complex64::new(amplitude * normalized, 0.0)
        })
        .collect()
}

fn profile_for(name: &str) -> ccmm_rs::ckks::CkksParameterProfile {
    match name {
        "research-4096" => research_profile_4096(),
        "research-8192" => research_profile_8192(),
        "research-16384" => research_profile_16384(),
        "research-32768" => research_profile_32768(),
        _ => panic!("unknown FFT1 characterization profile: {name}"),
    }
}

fn run_case(profile_name: &str, length: usize, salt: usize) {
    let profile = profile_for(profile_name);
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let initial_scale = profile.initial_scale();
    let basis = chain.top().clone();

    let fft_shape = Fft1Shape::new(length);
    let fft_plan = Fft1Plan::new(fft_shape);

    let required_levels = fft_plan.stage_count().saturating_sub(1);

    assert!(
        required_levels <= chain.max_level(),
        "FFT1 characterization case exceeds available chain depth"
    );

    let embedding = CkksCanonicalEmbedding::new(degree);
    let top_ntt_plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4654_5431_0000_0000 ^ degree as u64);

    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let input = deterministic_input(length, salt);

    let encryption_start = Instant::now();

    let encrypted: Vec<RnsCkksCiphertextMatrix> = input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            encrypt_scalar(
                value.re,
                0x4654_5431_454e_4300 ^ ((salt as u64) << 16) ^ index as u64,
                &embedding,
                &basis,
                initial_scale,
                &secret,
                &top_ntt_plan,
                &chain,
            )
        })
        .collect();

    let encrypt_ms = encryption_start.elapsed().as_secs_f64() * 1_000.0;

    let keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &keys);

    let execution_start = Instant::now();

    let encrypted_output = execute_fft1_plan_cp(
        &fft_plan,
        FftDirection::Forward,
        &encrypted,
        &evaluator,
        &embedding,
        &chain,
    );

    let execute_ms = execution_start.elapsed().as_secs_f64() * 1_000.0;

    let decryption_start = Instant::now();

    let actual: Vec<Complex64> = encrypted_output
        .iter()
        .map(|value| decode_complex_scalar(value.get(0, 0), &secret, &embedding))
        .collect();

    let decrypt_ms = decryption_start.elapsed().as_secs_f64() * 1_000.0;

    let expected = execute_fft1_plan(&fft_plan, FftDirection::Forward, &input);

    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (observed, reference) in actual.iter().zip(&expected) {
        let error = *observed - *reference;

        squared_error += error.norm_sqr();
        squared_reference += reference.norm_sqr();
        max_abs = max_abs.max(error.norm());
    }

    let rel_l2 = if squared_reference > 0.0 {
        (squared_error / squared_reference).sqrt()
    } else {
        squared_error.sqrt()
    };

    let input_level = encrypted[0].level();
    let output_level = encrypted_output[0].level();
    let levels_consumed = output_level - input_level;

    let input_max_abs = input
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);

    let expected_max_abs = expected
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);

    let status = rel_l2 <= TOLERANCE && max_abs <= TOLERANCE;

    println!(
        "FFT1_CHARACTERIZATION \
PROFILE={} \
N={} \
RING_DEGREE={} \
CHAIN_LEVELS={} \
STAGES={} \
BUTTERFLIES={} \
INPUT_LEVEL={} \
OUTPUT_LEVEL={} \
LEVELS_CONSUMED={} \
INPUT_SCALE={:.12e} \
OUTPUT_SCALE={:.12e} \
INPUT_MAX_ABS={:.12e} \
EXPECTED_MAX_ABS={:.12e} \
REL_L2={:.12e} \
MAX_ABS={:.12e} \
ENCRYPT_MS={:.3} \
EXECUTE_MS={:.3} \
DECRYPT_MS={:.3} \
STATUS={}",
        profile_name,
        length,
        degree,
        chain.len(),
        fft_plan.stage_count(),
        fft_plan.butterfly_count(),
        input_level,
        output_level,
        levels_consumed,
        encrypted[0].scale(),
        encrypted_output[0].scale(),
        input_max_abs,
        expected_max_abs,
        rel_l2,
        max_abs,
        encrypt_ms,
        execute_ms,
        decrypt_ms,
        if status { "PASS" } else { "FAIL" },
    );

    assert_eq!(
        levels_consumed, required_levels,
        "FFT1 characterization level consumption mismatch"
    );

    assert!(
        status,
        "FFT1 characterization failed for profile {profile_name}, N={length}: rel_l2={rel_l2:e}, max_abs={max_abs:e}"
    );
}

fn main() {
    let extended = std::env::var_os("EBLAS_FFT1_EXTENDED").is_some();

    println!("EBLAS_FFT1_CHARACTERIZATION_VERSION=1");
    println!("EBLAS_FFT1_REPRESENTATION=one_ciphertext_per_logical_sample");
    println!("EBLAS_FFT1_INPUT_POLICY=max_L1_spectral_bound_0.125");

    let smoke_cases = [
        ("research-4096", 2usize, 1usize),
        ("research-4096", 4usize, 2usize),
        ("research-4096", 8usize, 3usize),
        ("research-8192", 16usize, 4usize),
    ];

    for &(profile, length, salt) in &smoke_cases {
        run_case(profile, length, salt);
    }

    if extended {
        let extended_cases = [
            ("research-8192", 32usize, 5usize),
            ("research-16384", 64usize, 6usize),
        ];

        for &(profile, length, salt) in &extended_cases {
            run_case(profile, length, salt);
        }
    }

    println!("EBLAS_FFT1_CHARACTERIZATION_STATUS=PASS");
}
