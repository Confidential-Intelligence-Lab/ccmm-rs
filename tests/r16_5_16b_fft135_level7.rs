// R16.5.16b: encrypted FFT135 bridge, levels 7 through 11, 4096 active slots.
// Standalone bridge test. Input is encrypted synthetic data modulus-switched to level 7;
// this does NOT yet consume real horizontal FFT ciphertexts.
// Explicitly ignored by default because this encrypts 135 full CKKS vectors.
use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    mod_switch_rns_ckks_to_next, research_profile_32768_for_levels, CkksCanonicalEmbedding,
    CkksChainState, RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::level1::{multiply_complex_slots_cp_prepared, PreparedComplexSlotsCp};
use ccmm_rs::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big,
    PreparedRnsNttPlan, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::f64::consts::PI;
use std::time::Instant;

const N: usize = 135;
const RADICES: [usize; 4] = [5, 3, 3, 3];
const SIGMA: f64 = 3.19;

fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * (exponent % n) as f64 / n as f64)
}

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
                .expect("CKKS coefficient must fit f64")
                / ct.scale()
        })
        .collect();
    embedding.coefficients_to_slots(&coefficients)
}

// Matches the mixed-radix DIF permutation validated in R16.5.14c.1.
fn natural_index(index: usize) -> usize {
    let mut remaining = index;
    let mut digits = [0usize; 4];
    for (stage, &radix) in RADICES.iter().rev().enumerate() {
        digits[3 - stage] = remaining % radix;
        remaining /= radix;
    }
    assert_eq!(remaining, 0);
    digits[0] + RADICES[0] * (digits[1] + RADICES[1] * (digits[2] + RADICES[2] * digits[3]))
}

#[test]
#[ignore = "135 encrypted vectors; run explicitly with --ignored --nocapture"]
fn r16_5_16b_fft135_level7() {
    let started = Instant::now();
    let profile = research_profile_32768_for_levels(13, 3);
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    assert!(chain.max_level() >= 13);
    assert_eq!(degree, 32768);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();
    assert_eq!(slots, 16384);
    const ACTIVE: usize = 4096;
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
    let mut rng = ChaCha20Rng::seed_from_u64(0x1651_4C02);
    let sample: Vec<Complex64> = (0..N)
        .map(|t| {
            let x = t as f64;
            Complex64::new(0.001 * (0.17 * x).sin(), 0.001 * (0.11 * x).cos())
        })
        .collect();
    let encrypt_started = Instant::now();
    let mut ciphertexts: Vec<RnsCkksCiphertext> = sample
        .iter()
        .map(|&value| {
            let mut values = vec![Complex64::new(0.0, 0.0); slots];
            for (slot, item) in values.iter_mut().enumerate().take(ACTIVE) {
                *item = value * Complex64::new(1.0 + (slot % 256) as f64 * 0.0001, 0.0);
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
            {
                let mut ct =
                    RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);
                for expected_level in 1..=7 {
                    ct = mod_switch_rns_ckks_to_next(&ct, &chain);
                    assert_eq!(ct.level(), expected_level);
                }
                ct
            }
        })
        .collect();
    println!(
        "R16_16B_FFT135_ENCRYPT_MS={:.3}",
        encrypt_started.elapsed().as_secs_f64() * 1000.0
    );
    let keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &keys);
    let mut span = N;
    for (stage, &radix) in RADICES.iter().enumerate() {
        let stage_started = Instant::now();
        let m = span / radix;
        let level = ciphertexts[0].level();
        assert_eq!(level, 7 + stage);
        let plan = RnsNttPlan::new(chain.level(level).moduli().to_vec(), degree);
        let prepared_plan = PreparedRnsNttPlan::new(&plan);
        let mut next: Vec<Option<RnsCkksCiphertext>> = vec![None; N];
        for base in (0..N).step_by(span) {
            for j in 0..m {
                for k in 0..radix {
                    let mut output: Option<RnsCkksCiphertext> = None;
                    for q in 0..radix {
                        let input = &ciphertexts[base + j + q * m];
                        let coefficient = root(radix, q * k) * root(span, j * k);
                        let term = if (coefficient - Complex64::new(1.0, 0.0)).norm() < 1e-14 {
                            mod_switch_rns_ckks_to_next(input, &chain)
                        } else {
                            let coefficients = vec![coefficient; slots];
                            let prepared = PreparedComplexSlotsCp::new(
                                input,
                                &coefficients,
                                &embedding,
                                &chain,
                                &plan,
                            );
                            multiply_complex_slots_cp_prepared(
                                input,
                                &prepared,
                                &chain,
                                &prepared_plan,
                            )
                        };
                        output = Some(match output {
                            Some(acc) => evaluator.add(&acc, &term),
                            None => term,
                        });
                    }
                    let destination = base + j + k * m;
                    assert!(next[destination].is_none());
                    next[destination] = output;
                }
            }
        }
        ciphertexts = next
            .into_iter()
            .map(|ct| ct.expect("missing butterfly output"))
            .collect();
        span = m;
        for ct in &ciphertexts {
            assert_eq!(ct.level(), 8 + stage);
            assert_eq!(ct.scale(), scale);
        }
        println!("R16_16B_FFT135_STAGE_{}_RADIX={radix}", stage + 1);
        println!("R16_16B_FFT135_STAGE_{}_LEVEL={}", stage + 1, stage + 8);
        println!(
            "R16_16B_FFT135_STAGE_{}_MS={:.3}",
            stage + 1,
            stage_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    assert_eq!(span, 1);
    let mut reference = vec![Complex64::new(0.0, 0.0); N];
    for (k, expected) in reference.iter_mut().enumerate() {
        *expected = (0..N).map(|t| sample[t] * root(N, k * t)).sum();
    }
    let validation_started = Instant::now();
    let mut error_sq = 0.0;
    let mut reference_sq = 0.0;
    let mut max_abs = 0.0_f64;
    let mut seen = [false; N];
    for (index, ct) in ciphertexts.iter().enumerate() {
        let k = natural_index(index);
        assert!(!seen[k]);
        seen[k] = true;
        let decoded = decode(ct, &secret, &embedding);
        let actual = decoded[0];
        for slot in [1usize, 127, 255, 256, 4095] {
            let factor = 1.0 + (slot % 256) as f64 * 0.0001;
            let slot_error = (decoded[slot] - reference[k] * factor).norm();
            assert!(
                slot_error.is_finite() && slot_error < 1e-3,
                "slot {slot} at FFT135 output {k}: error {slot_error:e}"
            );
        }
        let error = (actual - reference[k]).norm();
        error_sq += error * error;
        reference_sq += reference[k].norm_sqr();
        max_abs = max_abs.max(error);
    }
    assert!(seen.into_iter().all(|x| x));
    let rel_l2 = (error_sq / reference_sq).sqrt();
    println!("R16_16B_INPUT_LEVEL=7");
    println!("R16_16B_ACTIVE_SLOTS={ACTIVE}");
    println!("R16_16B_FINAL_LEVEL={}", ciphertexts[0].level());
    println!("R16_16B_FFT135_REL_L2={rel_l2:.12e}");
    println!("R16_16B_FFT135_MAX_ABS={max_abs:.12e}");
    println!(
        "R16_16B_FFT135_VALIDATION_MS={:.3}",
        validation_started.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "R16_16B_FFT135_TOTAL_MS={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    let pass = ciphertexts[0].level() == 11
        && rel_l2.is_finite()
        && max_abs.is_finite()
        && rel_l2 < 1e-2
        && max_abs < 1e-3;
    println!(
        "R16_16B_FFT135_STATUS={}",
        if pass { "PASS" } else { "FAIL" }
    );
    assert!(
        pass,
        "encrypted FFT135 accuracy outside initial research thresholds"
    );
}
