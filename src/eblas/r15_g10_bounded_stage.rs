use super::*;

#[test]
#[ignore = "large bounded Gaussian FFT stage benchmark"]
fn r15_g10_bounded_gaussian_image_fft_stage_64x64() {
    use crate::application_support::ckks::encode_rns;
    use crate::ckks::{
        research_profile_32768, rotation_exponent_left, rotation_exponent_right,
        BoundedRnsGaloisKey, CkksChainState, RnsCkksEvaluationKeys,
    };
    use crate::grafting::{
        decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
        BoundedRnsKeygenConfig,
    };
    use crate::ring::{
        centered_representative_big, composite_modulus_big, reconstruct_coefficients_big,
        PreparedRnsNttPlan, RnsNttPlan,
    };
    use crate::rlwe::ErrorDistribution;
    use num_traits::ToPrimitive;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use std::time::Instant;

    const BASE_LOG: u32 = 8;
    const SIGMA: f64 = 3.19;
    const TOLERANCE: f64 = 5.0e-3;

    let profile = research_profile_32768();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    let basis = chain.top().clone();

    let embedding = CkksCanonicalEmbedding::new(degree);
    let slot_count = embedding.slot_count();

    let shape = Fft2Shape::new(64, 64);
    let active_slots = shape.elements();

    assert_eq!(degree, 32768);
    assert_eq!(active_slots, 4096);
    assert!(active_slots <= slot_count);

    let input: Vec<Complex64> = (0..slot_count)
        .map(|index| {
            if index < active_slots {
                let value = ((index * 17 + 11) % 251) as f64 / 251.0;
                Complex64::new(value, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();

    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let prepared_plan = PreparedRnsNttPlan::new(&plan);

    let plaintext = encode_rns(&input, &embedding, &basis, scale);

    let secret: Vec<i8> = (0..degree)
        .map(|index| match index % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x1531_0001);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &plan,
        &mut encryption_rng,
    );

    let ciphertext = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

    let diagonals =
        PackedFft2DifStageDiagonals::new(shape, PackedFft2Axis::Rows, 64, FftDirection::Forward);

    let rotation = diagonals.rotation();
    assert_eq!(rotation, 32);

    let layout = BoundedGadgetLayout::new(basis.clone(), BASE_LOG);
    let digits = layout.digit_count();
    let limbs = basis.moduli().len();

    let key_payload_bytes = 2_u128 * 2 * degree as u128 * limbs as u128 * digits as u128 * 8;

    println!("R15_G10_DEGREE={degree}");
    println!("R15_G10_SLOTS={slot_count}");
    println!("R15_G10_ACTIVE_SLOTS={active_slots}");
    println!("R15_G10_LIMBS={limbs}");
    println!("R15_G10_BASE_LOG={BASE_LOG}");
    println!("R15_G10_GADGET_DIGITS={digits}");
    println!("R15_G10_SIGMA={SIGMA}");
    println!("R15_G10_INITIAL_SCALE={scale:.12e}");
    println!("R15_G10_STAGE_TYPE=local-row");
    println!("R15_G10_STAGE_ROTATION={rotation}");
    println!(
        "R15_G10_KEY_PAYLOAD_MIB={:.3}",
        key_payload_bytes as f64 / 1048576.0
    );

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let mut key_rng = ChaCha20Rng::seed_from_u64(0x1531_0002);

    let keygen_start = Instant::now();

    let left_key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout: layout.clone(),
            plan: &plan,
        },
        &secret,
        rotation_exponent_left(degree, rotation),
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &mut key_rng,
    );

    let left_keygen_ms = keygen_start.elapsed().as_secs_f64() * 1000.0;

    let preparation_start = Instant::now();
    let prepared_left = PreparedBoundedRnsGaloisKey::new(&left_key, &plan);
    drop(left_key);
    let left_preparation_ms = preparation_start.elapsed().as_secs_f64() * 1000.0;

    println!("R15_G10_LEFT_KEYGEN_MS={left_keygen_ms:.3}");
    println!("R15_G10_LEFT_PREPARATION_MS={left_preparation_ms:.3}");

    let keygen_start = Instant::now();

    let right_key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout,
            plan: &plan,
        },
        &secret,
        rotation_exponent_right(degree, rotation),
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &mut key_rng,
    );

    let right_keygen_ms = keygen_start.elapsed().as_secs_f64() * 1000.0;

    let preparation_start = Instant::now();
    let prepared_right = PreparedBoundedRnsGaloisKey::new(&right_key, &plan);
    drop(right_key);
    let right_preparation_ms = preparation_start.elapsed().as_secs_f64() * 1000.0;

    println!("R15_G10_RIGHT_KEYGEN_MS={right_keygen_ms:.3}");
    println!("R15_G10_RIGHT_PREPARATION_MS={right_preparation_ms:.3}");

    let preparation_start = Instant::now();

    let prepared_fft_stage = PreparedRepeatedPackedFft2DifStage::new(
        &ciphertext,
        &diagonals,
        1,
        &embedding,
        &chain,
        &plan,
    );

    let diagonal_preparation_ms = preparation_start.elapsed().as_secs_f64() * 1000.0;

    println!("R15_G10_DIAGONAL_PREPARATION_MS={diagonal_preparation_ms:.3}");

    let execution_start = Instant::now();

    let output = execute_repeated_packed_fft2_dif_stage_cp_bounded_prepared(
        &ciphertext,
        &prepared_fft_stage,
        &evaluator,
        (&prepared_left, &prepared_right),
        &chain,
        &plan,
        &prepared_plan,
    );

    let execution_ms = execution_start.elapsed().as_secs_f64() * 1000.0;

    println!("R15_G10_EXECUTION_MS={execution_ms:.3}");
    println!("R15_G10_OUTPUT_LEVEL={}", output.level());
    println!("R15_G10_OUTPUT_SCALE={:.12e}", output.scale());

    assert_eq!(output.level(), 1);
    assert!(output.scale().is_finite());
    assert!(output.scale() > 0.0);

    let result_plan = RnsNttPlan::new(output.basis().moduli().to_vec(), degree);

    let decrypted = decrypt_rns_raw_with_ntt(output.rlwe(), &secret, &result_plan);

    let modulus = composite_modulus_big(decrypted.basis());

    let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
        .into_iter()
        .map(|value| {
            centered_representative_big(&value, &modulus)
                .to_f64()
                .expect("decoded coefficient must fit f64")
                / output.scale()
        })
        .collect();

    let actual = embedding.coefficients_to_slots(&coefficients);

    let reference = execute_repeated_packed_fft2_dif_stage_pp(&input, &diagonals, 1);

    let mut max_abs = 0.0_f64;
    let mut error_sq = 0.0_f64;
    let mut reference_sq = 0.0_f64;
    let mut inactive_max_abs = 0.0_f64;

    for index in 0..slot_count {
        assert!(
            actual[index].re.is_finite() && actual[index].im.is_finite(),
            "non-finite decoded slot at index {index}"
        );

        if index < active_slots {
            let error = (actual[index] - reference[index]).norm();
            max_abs = max_abs.max(error);
            error_sq += error * error;
            reference_sq += reference[index].norm_sqr();
        } else {
            inactive_max_abs = inactive_max_abs.max(actual[index].norm());
        }
    }

    assert!(reference_sq > 0.0);

    let rel_l2 = (error_sq / reference_sq).sqrt();

    println!("R15_G10_REL_L2={rel_l2:.12e}");
    println!("R15_G10_MAX_ABS={max_abs:.12e}");
    println!("R15_G10_INACTIVE_MAX_ABS={inactive_max_abs:.12e}");
    println!("R15_G10_TOLERANCE={TOLERANCE:.12e}");

    let passed = rel_l2 <= TOLERANCE && max_abs <= TOLERANCE && inactive_max_abs <= TOLERANCE;

    println!("R15_G10_STATUS={}", if passed { "PASS" } else { "FAIL" });

    assert!(passed, "bounded Gaussian image FFT stage failed");
}
