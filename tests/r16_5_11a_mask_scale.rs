// R16.5.11a: UHD half-tile CKKS masking and scale experiment.
//
// Research correctness experiment only; no security claim.
// Does not modify the established FFT15 or R15 evidence.

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    multiply_plain_rns_ckks_with_prepared_ntt, research_profile_16384, CkksCanonicalEmbedding,
    CkksChainState, PreparedRnsPlaintextNtt, RnsCkksCiphertext,
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
use std::time::Instant;

const TILE_SLOTS: usize = 4096;

fn decode(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let degree = embedding.degree();

    let plan = RnsNttPlan::new(ciphertext.basis().moduli().to_vec(), degree);

    let decrypted = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);

    let modulus = composite_modulus_big(decrypted.basis());

    let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
        .into_iter()
        .map(|value| {
            centered_representative_big(&value, &modulus)
                .to_f64()
                .expect("CKKS coefficient must fit f64")
                / ciphertext.scale()
        })
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

fn report(
    label: &str,
    actual: &[Complex64],
    input: &[Complex64],
    level: usize,
    output_scale: f64,
    elapsed_ms: f64,
) -> (f64, f64, f64) {
    let mut error_squared = 0.0_f64;
    let mut reference_squared = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut leakage = 0.0_f64;

    for slot in 0..actual.len() {
        let expected = if slot < TILE_SLOTS {
            input[slot]
        } else {
            Complex64::new(0.0, 0.0)
        };

        let error = (actual[slot] - expected).norm();

        error_squared += error * error;
        reference_squared += expected.norm_sqr();
        max_abs = max_abs.max(error);

        if slot >= TILE_SLOTS {
            leakage = leakage.max(actual[slot].norm());
        }
    }

    let rel_l2 = (error_squared / reference_squared).sqrt();

    println!("R16_MASK_CASE={label}");
    println!("R16_MASK_LEVEL={level}");
    println!("R16_MASK_OUTPUT_SCALE={output_scale:.12e}");
    println!("R16_MASK_REL_L2={rel_l2:.12e}");
    println!("R16_MASK_MAX_ABS={max_abs:.12e}");
    println!("R16_MASK_INACTIVE_LEAKAGE={leakage:.12e}");
    println!("R16_MASK_EXECUTION_MS={elapsed_ms:.3}");

    (rel_l2, max_abs, leakage)
}

#[test]
fn r16_5_11a_uhd_half_tile_mask_scales() {
    let profile = research_profile_16384();

    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    let basis = chain.top().clone();

    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    assert_eq!(degree, 16384);
    assert_eq!(slots, 8192);
    assert_eq!(TILE_SLOTS * 2, slots);

    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let prepared_plan = PreparedRnsNttPlan::new(&plan);

    let secret: Vec<i8> = (0..degree)
        .map(|index| match index % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let input: Vec<Complex64> = (0..slots)
        .map(|slot| {
            if slot < TILE_SLOTS {
                Complex64::new(
                    0.05 + 0.01 * (slot as f64 * 0.013).sin(),
                    0.02 * (slot as f64 * 0.017).cos(),
                )
            } else {
                Complex64::new(
                    -0.04 + 0.01 * (slot as f64 * 0.011).cos(),
                    -0.03 * (slot as f64 * 0.019).sin(),
                )
            }
        })
        .collect();

    let mask: Vec<Complex64> = (0..slots)
        .map(|slot| {
            if slot < TILE_SLOTS {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x1651_1A00);

    let plaintext = encode_rns(&input, &embedding, &basis, scale);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        &secret,
        &plan,
        &mut rng,
    );

    let ciphertext = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

    println!("R16_MASK_DEGREE={degree}");
    println!("R16_MASK_SLOTS={slots}");
    println!("R16_MASK_TILE_SLOTS={TILE_SLOTS}");
    println!("R16_MASK_INITIAL_SCALE={scale:.12e}");

    let mut baseline_pass = false;

    for exponent in [0, 8, 16, 24] {
        let plaintext_scale = 2.0_f64.powi(exponent);

        let mask_plaintext = encode_rns(&mask, &embedding, &basis, plaintext_scale);

        let prepared = PreparedRnsPlaintextNtt::new(&mask_plaintext, &prepared_plan);

        let start = Instant::now();

        let result = multiply_plain_rns_ckks_with_prepared_ntt(
            &ciphertext,
            &prepared,
            plaintext_scale,
            &chain,
            &prepared_plan,
        );

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        assert_eq!(result.level(), ciphertext.level());
        assert_eq!(result.scale(), ciphertext.scale() * plaintext_scale);

        let actual = decode(&result, &secret, &embedding);

        let label = format!("RAW_2_POW_{exponent}");

        let (rel_l2, max_abs, leakage) = report(
            &label,
            &actual,
            &input,
            result.level(),
            result.scale(),
            elapsed_ms,
        );

        if exponent == 0 {
            baseline_pass = rel_l2 < 1.0e-3 && max_abs < 1.0e-3 && leakage < 1.0e-3;
        }
    }

    // Established eBLAS implementation: public multiplication
    // followed by one CKKS rescale.
    let prepared_mask = PreparedComplexSlotsCp::new(&ciphertext, &mask, &embedding, &chain, &plan);

    let start = Instant::now();

    let rescaled =
        multiply_complex_slots_cp_prepared(&ciphertext, &prepared_mask, &chain, &prepared_plan);

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(rescaled.level(), ciphertext.level() + 1);
    assert_eq!(rescaled.scale(), ciphertext.scale());

    let actual = decode(&rescaled, &secret, &embedding);

    let (baseline_rel_l2, baseline_max_abs, baseline_leakage) = report(
        "RESCALED_BASELINE",
        &actual,
        &input,
        rescaled.level(),
        rescaled.scale(),
        elapsed_ms,
    );

    let rescaled_pass =
        baseline_rel_l2 < 1.0e-3 && baseline_max_abs < 1.0e-3 && baseline_leakage < 1.0e-3;

    println!(
        "R16_MASK_SCALE1_STATUS={}",
        if baseline_pass { "PASS" } else { "FAIL" }
    );

    println!(
        "R16_MASK_RESCALED_STATUS={}",
        if rescaled_pass { "PASS" } else { "FAIL" }
    );

    // The level-preserving scale-1 case is exploratory.
    // The established rescaled path must remain numerically valid.
    assert!(
        rescaled_pass,
        "Rescaled UHD half-tile masking baseline failed"
    );
}
