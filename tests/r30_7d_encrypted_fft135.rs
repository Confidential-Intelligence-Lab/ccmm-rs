// R30_7D_ENCRYPTED_FFT135_NUMERICAL
//
// Opt-in production-geometry CKKS correctness test.
// Run with:
// R30_7D_RUN=1 cargo test --release --test r30_7d_encrypted_fft135 -- --nocapture

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    mod_switch_rns_ckks_to_next, research_profile_32768_for_levels, CkksCanonicalEmbedding,
    CkksChainState, RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::encrypted_fft2160::execute_encrypted_fft135;
use ccmm_rs::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;

use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::time::Instant;

const BRANCHES: usize = 135;
const SIGMA: f64 = 3.19;

fn decode(
    ct: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let plan = RnsNttPlan::new(ct.basis().moduli().to_vec(), embedding.degree());
    let decrypted = decrypt_rns_raw_with_ntt(ct.rlwe(), secret, &plan);
    let modulus = composite_modulus_big(decrypted.basis());

    let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
        .into_iter()
        .map(|value| {
            centered_representative_big(&value, &modulus)
                .to_f64()
                .expect("coefficient conversion")
                / ct.scale()
        })
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

#[test]
fn r30_7d_encrypted_fft135_impulse_oracle() {
    if std::env::var_os("R30_7D_RUN").is_none() {
        println!("R30_7D_STATUS=SKIPPED_OPT_IN");
        return;
    }

    let started = Instant::now();

    let profile = research_profile_32768_for_levels(13, 3);
    let degree = profile.degree();
    let chain = profile.modulus_chain();
    let scale = profile.initial_scale();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    assert_eq!(degree, 32768);
    assert_eq!(slots, 16384);
    assert!(chain.max_level() >= 13);

    let basis0 = chain.top().clone();
    let plan0 = RnsNttPlan::new(basis0.moduli().to_vec(), degree);

    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x307D_1350);
    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    // One nonzero vertical sample. The unnormalized DFT of an
    // impulse at index zero equals its amplitude at every frequency.
    let amplitude = 1.0e-7_f64;
    let mut ciphertexts = Vec::with_capacity(BRANCHES);

    for branch in 0..BRANCHES {
        let mut values = vec![Complex64::new(0.0, 0.0); slots];

        if branch == 0 {
            values[0] = Complex64::new(amplitude, 0.0);
        }

        let plaintext = encode_rns(&values, &embedding, &basis0, scale);

        let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
            &plaintext,
            2,
            ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
            &secret,
            &plan0,
            &mut rng,
        );

        let mut ct = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

        for level in 0..7 {
            assert_eq!(ct.level(), level);
            ct = mod_switch_rns_ckks_to_next(&ct, &chain);
        }

        assert_eq!(ct.level(), 7);
        ciphertexts.push(ct);
    }

    println!("R30_7D_INPUT_CIPHERTEXTS={}", ciphertexts.len());
    println!("R30_7D_INPUT_LEVEL=7");

    let execution_started = Instant::now();

    let outputs = execute_encrypted_fft135(ciphertexts, &embedding, &evaluator, &chain);

    println!(
        "R30_7D_FFT135_EXECUTION_MS={:.3}",
        execution_started.elapsed().as_secs_f64() * 1000.0
    );

    assert_eq!(outputs.len(), BRANCHES);
    assert!(outputs.iter().all(|ct| ct.level() == 11));

    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut max_absolute_error = 0.0_f64;

    for (index, ct) in outputs.iter().enumerate() {
        let decoded = decode(ct, &secret, &embedding);
        let actual = decoded[0];
        let expected = Complex64::new(amplitude, 0.0);

        let error = (actual - expected).norm();

        squared_error += error * error;
        squared_reference += expected.norm_sqr();
        max_absolute_error = max_absolute_error.max(error);

        if index < 5 {
            println!(
                "R30_7D_OUTPUT_{}=({:.12e},{:.12e})",
                index, actual.re, actual.im
            );
        }
    }

    let relative_l2 = (squared_error / squared_reference).sqrt();

    println!("R30_7D_OUTPUT_CIPHERTEXTS={}", outputs.len());
    println!("R30_7D_OUTPUT_LEVEL=11");
    println!("R30_7D_RELATIVE_L2={relative_l2:.12e}");
    println!("R30_7D_MAX_ABSOLUTE_ERROR={max_absolute_error:.12e}");
    println!(
        "R30_7D_TOTAL_MS={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );

    assert!(
        relative_l2.is_finite() && relative_l2 < 0.1,
        "encrypted FFT135 impulse relative error exceeds tolerance"
    );

    println!("R30_7D_STATUS=PASS");
}

// R30_7E_COMPLEX_IMPULSE_ORACLE
//
// Nonzero-index complex impulse verifies frequency ordering,
// twiddle direction, and complex arithmetic.
//
// Opt-in:
// R30_7E_RUN=1 cargo test --release --test r30_7d_encrypted_fft135 r30_7e -- --nocapture

#[test]
fn r30_7e_encrypted_fft135_complex_impulse_oracle() {
    if std::env::var_os("R30_7E_RUN").is_none() {
        println!("R30_7E_STATUS=SKIPPED_OPT_IN");
        return;
    }

    use ccmm_rs::eblas::encrypted_fft2160::Fft135Plan;
    use std::f64::consts::PI;

    const IMPULSE_INDEX: usize = 17;

    let started = Instant::now();

    let profile = research_profile_32768_for_levels(13, 3);
    let degree = profile.degree();
    let chain = profile.modulus_chain();
    let scale = profile.initial_scale();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    assert_eq!(degree, 32768);
    assert_eq!(slots, 16384);
    assert!(chain.max_level() >= 13);

    let basis0 = chain.top().clone();
    let plan0 = RnsNttPlan::new(basis0.moduli().to_vec(), degree);

    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x307E_1350);
    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let amplitude = Complex64::new(8.0e-8, -6.0e-8);
    let mut ciphertexts = Vec::with_capacity(BRANCHES);

    for branch in 0..BRANCHES {
        let mut values = vec![Complex64::new(0.0, 0.0); slots];

        if branch == IMPULSE_INDEX {
            values[0] = amplitude;
        }

        let plaintext = encode_rns(&values, &embedding, &basis0, scale);

        let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
            &plaintext,
            2,
            ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
            &secret,
            &plan0,
            &mut rng,
        );

        let mut ct = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

        for level in 0..7 {
            assert_eq!(ct.level(), level);
            ct = mod_switch_rns_ckks_to_next(&ct, &chain);
        }

        ciphertexts.push(ct);
    }

    assert_eq!(ciphertexts.len(), BRANCHES);
    assert!(ciphertexts.iter().all(|ct| ct.level() == 7));

    println!("R30_7E_INPUT_CIPHERTEXTS=135");
    println!("R30_7E_IMPULSE_INDEX={IMPULSE_INDEX}");
    println!(
        "R30_7E_AMPLITUDE=({:.12e},{:.12e})",
        amplitude.re, amplitude.im
    );

    let execution_started = Instant::now();

    let outputs = execute_encrypted_fft135(ciphertexts, &embedding, &evaluator, &chain);

    println!(
        "R30_7E_FFT135_EXECUTION_MS={:.3}",
        execution_started.elapsed().as_secs_f64() * 1000.0
    );

    assert_eq!(outputs.len(), BRANCHES);
    assert!(outputs.iter().all(|ct| ct.level() == 11));

    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut max_absolute_error = 0.0_f64;
    let mut max_phase_error = 0.0_f64;

    for (output_index, ct) in outputs.iter().enumerate() {
        let frequency = Fft135Plan::natural_frequency(output_index);

        let angle = -2.0 * PI * (frequency * IMPULSE_INDEX) as f64 / BRANCHES as f64;

        let expected = amplitude * Complex64::from_polar(1.0, angle);
        let decoded = decode(ct, &secret, &embedding);
        let actual = decoded[0];

        let error = (actual - expected).norm();

        squared_error += error * error;
        squared_reference += expected.norm_sqr();
        max_absolute_error = max_absolute_error.max(error);

        let phase_difference = actual * expected.conj();
        let phase_error = phase_difference.arg().abs();
        max_phase_error = max_phase_error.max(phase_error);

        if output_index < 12 {
            println!(
                "R30_7E_OUTPUT_{}=FREQ_{},ACTUAL=({:.12e},{:.12e}),EXPECTED=({:.12e},{:.12e})",
                output_index, frequency, actual.re, actual.im, expected.re, expected.im
            );
        }
    }

    let relative_l2 = (squared_error / squared_reference).sqrt();

    println!("R30_7E_OUTPUT_CIPHERTEXTS={}", outputs.len());
    println!("R30_7E_OUTPUT_LEVEL=11");
    println!("R30_7E_RELATIVE_L2={relative_l2:.12e}");
    println!("R30_7E_MAX_ABSOLUTE_ERROR={max_absolute_error:.12e}");
    println!("R30_7E_MAX_PHASE_ERROR_RAD={max_phase_error:.12e}");
    println!(
        "R30_7E_TOTAL_MS={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );

    assert!(
        relative_l2.is_finite() && relative_l2 < 1.0e-3,
        "R30.7E encrypted FFT135 relative L2 error exceeds 1e-3"
    );

    assert!(
        max_phase_error.is_finite() && max_phase_error < 1.0e-2,
        "R30.7E encrypted FFT135 phase error exceeds 0.01 radians"
    );

    println!("R30_7E_STATUS=PASS");
}
