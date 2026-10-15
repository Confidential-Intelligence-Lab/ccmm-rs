// R16.5.11c: encrypted FFT15 across a physically packed UHD tile row.
// Research-only numerical validation. No security claim.
use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_16384, rotate_left_bounded_rns_ckks_with_prepared_execution,
    rotate_right_bounded_rns_ckks_with_prepared_execution, rotation_exponent_left,
    rotation_exponent_right, BoundedRnsGaloisKey, CkksCanonicalEmbedding, CkksChainState,
    PreparedBoundedRnsGaloisKey, RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::level1::{multiply_complex_slots_cp_prepared, PreparedComplexSlotsCp};
use ccmm_rs::eblas::mixed_radix_ckks::{
    packed_fft15_twiddled_radix3_cp_prepared, packed_radix5_dft_cp_prepared,
    PreparedFft15TwiddledRadix3Cp, PreparedRadix5Cp,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    BoundedRnsKeygenConfig,
};
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

const HALF: usize = 4096;
const TILES: usize = 15;
const PHYSICAL: usize = 8;
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
                .expect("CKKS coefficient must fit f64")
                / ct.scale()
        })
        .collect();
    embedding.coefficients_to_slots(&coefficients)
}

fn elapsed(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[test]
fn r16_5_11c_encrypted_physical_fft15() {
    let total_start = Instant::now();
    let profile = research_profile_16384();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    assert!(chain.max_level() >= 3);
    let embedding = CkksCanonicalEmbedding::new(degree);
    assert_eq!(degree, 16384);
    assert_eq!(embedding.slot_count(), 2 * HALF);

    let basis0 = chain.top().clone();
    let basis1 = chain.level(1).clone();
    let basis2 = chain.level(2).clone();
    let basis3 = chain.level(3).clone();
    let plan0 = RnsNttPlan::new(basis0.moduli().to_vec(), degree);
    let plan1 = RnsNttPlan::new(basis1.moduli().to_vec(), degree);
    let plan2 = RnsNttPlan::new(basis2.moduli().to_vec(), degree);
    let plan3 = RnsNttPlan::new(basis3.moduli().to_vec(), degree);
    let prepared0 = PreparedRnsNttPlan::new(&plan0);
    let prepared1 = PreparedRnsNttPlan::new(&plan1);
    let prepared2 = PreparedRnsNttPlan::new(&plan2);
    let prepared3 = PreparedRnsNttPlan::new(&plan3);

    // Deterministic secret and randomness are strictly for reproducible tests.
    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();
    let mut rng = ChaCha20Rng::seed_from_u64(0x1651_1C00);
    let inputs: [Vec<Complex64>; TILES] = std::array::from_fn(|tile| {
        (0..HALF)
            .map(|slot| {
                let t = tile as f64;
                let s = slot as f64;
                Complex64::new(
                    0.005 * (0.31 * t + 0.013 * s).sin(),
                    0.005 * (0.19 * t - 0.017 * s).cos(),
                )
            })
            .collect()
    });
    let packing_start = Instant::now();
    let physical: [RnsCkksCiphertext; PHYSICAL] = std::array::from_fn(|index| {
        let mut values = vec![Complex64::new(0.0, 0.0); 2 * HALF];
        values[..HALF].copy_from_slice(&inputs[2 * index]);
        if 2 * index + 1 < TILES {
            values[HALF..].copy_from_slice(&inputs[2 * index + 1]);
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
        RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain)
    });
    let packing_ms = elapsed(packing_start);

    let keygen_start = Instant::now();
    let left_key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout: BoundedGadgetLayout::new(basis0.clone(), 8),
            plan: &plan0,
        },
        &secret,
        rotation_exponent_left(degree, HALF),
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &mut rng,
    );
    let right_key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout: BoundedGadgetLayout::new(basis3.clone(), 8),
            plan: &plan3,
        },
        &secret,
        rotation_exponent_right(degree, HALF),
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &mut rng,
    );
    let keygen_ms = elapsed(keygen_start);
    let preparation_start = Instant::now();
    let prepared_left = PreparedBoundedRnsGaloisKey::new(&left_key, &plan0);
    let prepared_right = PreparedBoundedRnsGaloisKey::new(&right_key, &plan3);
    let low_mask: Vec<Complex64> = (0..2 * HALF)
        .map(|s| Complex64::new(if s < HALF { 1.0 } else { 0.0 }, 0.0))
        .collect();
    let prepared_mask =
        PreparedComplexSlotsCp::new(&physical[0], &low_mask, &embedding, &chain, &plan0);
    let movement_preparation_ms = elapsed(preparation_start);
    let keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &keys);

    let extraction_start = Instant::now();
    let aligned: Vec<RnsCkksCiphertext> = (0..TILES)
        .map(|tile| {
            let source = &physical[tile / 2];
            if tile % 2 == 0 {
                multiply_complex_slots_cp_prepared(source, &prepared_mask, &chain, &prepared0)
            } else {
                let shifted = rotate_left_bounded_rns_ckks_with_prepared_execution(
                    source,
                    HALF,
                    &prepared_left,
                    &chain,
                    &plan0,
                    &prepared0,
                );
                multiply_complex_slots_cp_prepared(&shifted, &prepared_mask, &chain, &prepared0)
            }
        })
        .collect();
    let extraction_ms = elapsed(extraction_start);
    assert_eq!(aligned.len(), TILES);
    for ct in &aligned {
        assert_eq!(ct.level(), 1);
        assert_eq!(ct.scale(), scale);
    }

    let radix5_preparation_start = Instant::now();
    let radix5 = PreparedRadix5Cp::new(&aligned[0], &embedding, &chain, &plan1);
    let radix5_preparation_ms = elapsed(radix5_preparation_start);
    let radix5_start = Instant::now();
    let branches: [[RnsCkksCiphertext; 5]; 3] = std::array::from_fn(|j| {
        packed_radix5_dft_cp_prepared(
            &evaluator,
            [
                &aligned[j],
                &aligned[j + 3],
                &aligned[j + 6],
                &aligned[j + 9],
                &aligned[j + 12],
            ],
            &radix5,
            &chain,
            &prepared1,
        )
    });
    let radix5_ms = elapsed(radix5_start);
    for branch in &branches {
        for ct in branch {
            assert_eq!(ct.level(), 2);
            assert_eq!(ct.scale(), scale);
        }
    }

    let radix3_preparation_start = Instant::now();
    let radix3 = PreparedFft15TwiddledRadix3Cp::new(&branches[0][0], &embedding, &chain, &plan2);
    let radix3_preparation_ms = elapsed(radix3_preparation_start);
    let radix3_start = Instant::now();
    let groups: [[RnsCkksCiphertext; 3]; 5] = std::array::from_fn(|k| {
        packed_fft15_twiddled_radix3_cp_prepared(
            &evaluator,
            [&branches[0][k], &branches[1][k], &branches[2][k]],
            k,
            &radix3,
            &chain,
            &prepared2,
        )
    });
    let radix3_ms = elapsed(radix3_start);
    let frequency_outputs: Vec<&RnsCkksCiphertext> = (0..TILES)
        .map(|frequency| &groups[frequency % 5][frequency / 5])
        .collect();
    for ct in &frequency_outputs {
        assert_eq!(ct.level(), 3);
        assert_eq!(ct.scale(), scale);
    }

    let repacking_start = Instant::now();
    let repacked: Vec<RnsCkksCiphertext> = (0..PHYSICAL)
        .map(|index| {
            let low = frequency_outputs[2 * index];
            if 2 * index + 1 < TILES {
                let high = rotate_right_bounded_rns_ckks_with_prepared_execution(
                    frequency_outputs[2 * index + 1],
                    HALF,
                    &prepared_right,
                    &chain,
                    &plan3,
                    &prepared3,
                );
                evaluator.add(low, &high)
            } else {
                low.clone()
            }
        })
        .collect();
    let repacking_ms = elapsed(repacking_start);
    assert_eq!(repacked.len(), PHYSICAL);
    for ct in &repacked {
        assert_eq!(ct.level(), 3);
        assert_eq!(ct.scale(), scale);
    }

    // Only the final physical ciphertexts are decrypted.
    let validation_start = Instant::now();
    let mut error_sq = 0.0;
    let mut reference_sq = 0.0;
    let mut max_abs = 0.0_f64;
    let mut inactive_leakage = 0.0_f64;
    for (index, ct) in repacked.iter().enumerate() {
        let actual = decode(ct, &secret, &embedding);
        for half in 0..2 {
            let frequency = 2 * index + half;
            for slot in 0..HALF {
                let measured = actual[half * HALF + slot];
                if frequency == TILES {
                    inactive_leakage = inactive_leakage.max(measured.norm());
                    continue;
                }
                let expected: Complex64 = (0..TILES)
                    .map(|tile| {
                        let angle =
                            -2.0 * std::f64::consts::PI * (frequency * tile) as f64 / TILES as f64;
                        inputs[tile][slot] * Complex64::new(angle.cos(), angle.sin())
                    })
                    .sum();
                let error = (measured - expected).norm();
                error_sq += error * error;
                reference_sq += expected.norm_sqr();
                max_abs = max_abs.max(error);
            }
        }
    }
    let validation_ms = elapsed(validation_start);
    let rel_l2 = (error_sq / reference_sq).sqrt();
    println!("R16_PHYSICAL_FFT15_INPUT_CIPHERTEXTS={PHYSICAL}");
    println!("R16_PHYSICAL_FFT15_OUTPUT_CIPHERTEXTS={}", repacked.len());
    println!("R16_PHYSICAL_FFT15_TRANSFORMED_VALUES={}", TILES * HALF);
    println!("R16_PHYSICAL_FFT15_FINAL_LEVEL=3");
    println!(
        "R16_PHYSICAL_FFT15_FINAL_SCALE={:.12e}",
        repacked[0].scale()
    );
    println!("R16_PHYSICAL_FFT15_PACKING_MS={packing_ms:.3}");
    println!("R16_PHYSICAL_FFT15_KEYGEN_MS={keygen_ms:.3}");
    println!("R16_PHYSICAL_FFT15_MOVEMENT_PREPARATION_MS={movement_preparation_ms:.3}");
    println!("R16_PHYSICAL_FFT15_EXTRACTION_MS={extraction_ms:.3}");
    println!("R16_PHYSICAL_FFT15_RADIX5_PREPARATION_MS={radix5_preparation_ms:.3}");
    println!("R16_PHYSICAL_FFT15_RADIX5_MS={radix5_ms:.3}");
    println!("R16_PHYSICAL_FFT15_RADIX3_PREPARATION_MS={radix3_preparation_ms:.3}");
    println!("R16_PHYSICAL_FFT15_RADIX3_MS={radix3_ms:.3}");
    println!("R16_PHYSICAL_FFT15_REPACKING_MS={repacking_ms:.3}");
    println!("R16_PHYSICAL_FFT15_VALIDATION_MS={validation_ms:.3}");
    println!("R16_PHYSICAL_FFT15_REL_L2={rel_l2:.12e}");
    println!("R16_PHYSICAL_FFT15_MAX_ABS={max_abs:.12e}");
    println!("R16_PHYSICAL_FFT15_INACTIVE_LEAKAGE={inactive_leakage:.12e}");
    println!("R16_PHYSICAL_FFT15_TOTAL_MS={:.3}", elapsed(total_start));
    let pass =
        rel_l2.is_finite() && rel_l2 < 1e-3 && max_abs.is_finite() && inactive_leakage < 1e-3;
    println!(
        "R16_PHYSICAL_FFT15_STATUS={}",
        if pass { "PASS" } else { "FAIL" }
    );
    assert!(pass, "physical encrypted FFT15 numerical validation failed");
}
