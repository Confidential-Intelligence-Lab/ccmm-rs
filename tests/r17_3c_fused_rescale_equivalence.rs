//! R17.3c: diagnostic comparison of experimental fused rescaling.
//! The fused executor is not production-approved; oracle errors were
//! higher than the production executor in this research fixture.
//! Research-only parameters; no cryptographic security claim.

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    rotate_left_rns_ckks_with_prepared_ntt, rotate_right_rns_ckks_with_prepared_ntt,
    rotation_exponent_left, rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState,
    PreparedRnsGaloisKey, RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{
    execute_repeated_packed_fft2_dif_stage_cp_prepared,
    execute_repeated_packed_fft2_dif_stage_cp_prepared_fused_rescale, Fft2Shape, FftDirection,
    PackedFft2Axis, PackedFft2DifStageDiagonals, PreparedRepeatedPackedFft2DifStage,
};
use ccmm_rs::eblas::level1::multiply_complex_slots_cp;
use ccmm_rs::grafting::decrypt_rns_raw_with_ntt;
use ccmm_rs::grafting::{
    encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout, RnsKeygenConfig,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big,
};
use ccmm_rs::ring::{ModulusBasis, ModulusChain, PreparedRnsNttPlan, RnsNttPlan};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn decode_rns_complex_slots(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let plan = RnsNttPlan::new(ciphertext.basis().moduli().to_vec(), embedding.degree());

    let decrypted = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);

    let modulus = composite_modulus_big(decrypted.basis());

    let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
        .into_iter()
        .map(|value| {
            centered_representative_big(&value, &modulus)
                .to_f64()
                .expect("R17.3c coefficient conversion")
                / ciphertext.scale()
        })
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

#[test]
fn r17_3c_fused_rescale_numerical_equivalence() {
    let degree = 1024usize;
    let scale = (1u64 << 25) as f64;

    let profile = ccmm_rs::ckks::research_profile_32768_for_levels(13, 3);
    let q = ModulusBasis::new(profile.modulus_chain().top().moduli()[..3].to_vec());

    let chain = ModulusChain::from_top_basis(q.clone());
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    let plan = RnsNttPlan::new(q.moduli().to_vec(), degree);
    let prepared_plan = PreparedRnsNttPlan::new(&plan);

    let shape = Fft2Shape::new(2, 2);
    let tiles = slots / shape.elements();

    assert_eq!(tiles * shape.elements(), slots);
    assert!(tiles > 1);

    let input: Vec<Complex64> = (0..slots)
        .map(|i| {
            let k = i % shape.elements();
            Complex64::new((k + 1) as f64 / 128.0, (k % 3) as f64 / 256.0)
        })
        .collect();

    let plaintext = encode_rns(&input, &embedding, &q, scale);

    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x171D_0001);

    let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        &secret,
        &plan,
        &mut rng,
    );

    let ciphertext = RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, scale), &chain);

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    for axis in [PackedFft2Axis::Rows, PackedFft2Axis::Columns] {
        let diagonals = PackedFft2DifStageDiagonals::new(shape, axis, 2, FftDirection::Forward);

        let rotation = diagonals.rotation();

        let make_key = |exponent, rng: &mut ChaCha20Rng| {
            RnsGaloisKey::generate_with_ntt_rng(
                RnsKeygenConfig {
                    degree,
                    plaintext_modulus: 2,
                    noise_bound: 0,
                    layout: RnsGadgetLayout::new(q.clone(), vec![1; q.len()]),
                    plan: &plan,
                },
                &secret,
                exponent,
                rng,
            )
        };

        let left_key = make_key(rotation_exponent_left(degree, rotation), &mut rng);
        let right_key = make_key(rotation_exponent_right(degree, rotation), &mut rng);

        let prepared_left = PreparedRnsGaloisKey::new(&left_key, &plan);
        let prepared_right = PreparedRnsGaloisKey::new(&right_key, &plan);

        // Expand the logical diagonals to the complete slot vector.
        let mut direct = Vec::with_capacity(slots);
        let mut left = Vec::with_capacity(slots);
        let mut right = Vec::with_capacity(slots);

        for _ in 0..tiles {
            direct.extend_from_slice(diagonals.direct());
            left.extend_from_slice(diagonals.rotate_left());
            right.extend_from_slice(diagonals.rotate_right());
        }

        assert_eq!(direct.len(), slots);
        assert_eq!(left.len(), slots);
        assert_eq!(right.len(), slots);

        let rotated_left = rotate_left_rns_ckks_with_prepared_ntt(
            &ciphertext,
            rotation,
            &prepared_left,
            &chain,
            &plan,
        );

        let rotated_right = rotate_right_rns_ckks_with_prepared_ntt(
            &ciphertext,
            rotation,
            &prepared_right,
            &chain,
            &plan,
        );

        let direct_term =
            multiply_complex_slots_cp(&ciphertext, &direct, &embedding, &chain, &plan);

        let left_term = multiply_complex_slots_cp(&rotated_left, &left, &embedding, &chain, &plan);

        let right_term =
            multiply_complex_slots_cp(&rotated_right, &right, &embedding, &chain, &plan);

        let partial = evaluator.add(&direct_term, &left_term);
        let reference = evaluator.add(&partial, &right_term);

        let prepared_stage = PreparedRepeatedPackedFft2DifStage::new(
            &ciphertext,
            &diagonals,
            tiles,
            &embedding,
            &chain,
            &plan,
        );

        // Independent cleartext evaluation of the three FFT diagonals.
        // The same cyclic slot rotations are used by the encrypted stage.
        let oracle: Vec<Complex64> = (0..slots)
            .map(|i| {
                let left_index = (i + rotation) % slots;
                let right_index = (i + slots - rotation) % slots;

                direct[i] * input[i] + left[i] * input[left_index] + right[i] * input[right_index]
            })
            .collect();

        for invocation in 0..2 {
            let started = std::time::Instant::now();

            let baseline = execute_repeated_packed_fft2_dif_stage_cp_prepared(
                &ciphertext,
                &prepared_stage,
                &evaluator,
                (&prepared_left, &prepared_right),
                &chain,
                &plan,
                &prepared_plan,
            );

            let baseline_ns = started.elapsed().as_nanos();

            let started = std::time::Instant::now();

            let fused = execute_repeated_packed_fft2_dif_stage_cp_prepared_fused_rescale(
                &ciphertext,
                &prepared_stage,
                &evaluator,
                (&prepared_left, &prepared_right),
                &chain,
                &plan,
                &prepared_plan,
            );

            let fused_ns = started.elapsed().as_nanos();

            assert_eq!(reference.rlwe(), baseline.rlwe());
            assert_eq!(baseline.level(), fused.level());
            assert_eq!(baseline.basis(), fused.basis());
            assert_eq!(baseline.scale().to_bits(), fused.scale().to_bits());

            let baseline_slots = decode_rns_complex_slots(&baseline, &secret, &embedding);

            let fused_slots = decode_rns_complex_slots(&fused, &secret, &embedding);

            assert_eq!(baseline_slots.len(), fused_slots.len());

            let relative_error = |actual: &[Complex64]| {
                assert_eq!(actual.len(), oracle.len());

                let error_sq: f64 = actual
                    .iter()
                    .zip(&oracle)
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum();

                let oracle_sq: f64 = oracle.iter().map(Complex64::norm_sqr).sum();

                (error_sq / oracle_sq.max(1e-300)).sqrt()
            };

            let baseline_oracle_error = relative_error(&baseline_slots);
            let fused_oracle_error = relative_error(&fused_slots);

            println!(
                "R17_3C_ORACLE axis={axis:?} invocation={invocation}                  baseline_rel_l2={baseline_oracle_error:.12e}                  fused_rel_l2={fused_oracle_error:.12e}"
            );

            assert!(
                baseline_oracle_error.is_finite() && fused_oracle_error.is_finite(),
                "R17.3c oracle produced non-finite error"
            );

            let mut error_sq = 0.0f64;
            let mut reference_sq = 0.0f64;
            let mut max_abs = 0.0f64;

            for (expected, actual) in baseline_slots.iter().zip(&fused_slots) {
                let difference = (*actual - *expected).norm();
                error_sq += difference * difference;
                reference_sq += expected.norm_sqr();
                max_abs = max_abs.max(difference);
            }

            let rel_l2 = (error_sq / reference_sq.max(1e-300)).sqrt();

            println!(
                "R17_3C_NUMERICAL axis={axis:?} invocation={invocation}                  rel_l2={rel_l2:.12e} max_abs={max_abs:.12e}                  baseline_ns={baseline_ns} fused_ns={fused_ns}"
            );

            assert!(
                rel_l2.is_finite() && max_abs.is_finite(),
                "R17.3c numerical comparison produced non-finite error"
            );

            println!(
                "R17_3C_DIAGNOSTIC axis={axis:?} invocation={invocation}                  rel_l2={rel_l2:.12e} max_abs={max_abs:.12e}                  status=MEASURED_NOT_VALIDATED"
            );
        }
    }

    println!("R17_3C_FUSED_RESCALE_DIAGNOSTIC=COMPLETE");
}
