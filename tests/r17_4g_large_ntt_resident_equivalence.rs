//! R17.4g: production-sized exact RLWE equivalence.
//! Research-only configuration; not a cryptographic security claim.
//! Run explicitly with --ignored --nocapture.

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    rotate_left_rns_ckks_with_prepared_ntt, rotate_right_rns_ckks_with_prepared_ntt,
    rotation_exponent_left, rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState,
    PreparedRnsGaloisKey, RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{
    execute_repeated_packed_fft2_dif_stage_cp_prepared, Fft2Shape, FftDirection, PackedFft2Axis,
    PackedFft2DifStageDiagonals, PreparedRepeatedPackedFft2DifStage,
};
use ccmm_rs::eblas::level1::multiply_complex_slots_cp;
use ccmm_rs::grafting::{
    encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout, RnsKeygenConfig,
};
use ccmm_rs::ring::{ModulusBasis, ModulusChain, PreparedRnsNttPlan, RnsNttPlan};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[test]
#[ignore = "expensive 65536-degree R17.4g equivalence gate"]
fn r17_4g_large_ntt_resident_exact_rlwe_equivalence() {
    let degree = 65536usize;
    let scale = (1u64 << 55) as f64;

    let profile = ccmm_rs::ckks::research_profile_65536();
    let q = ModulusBasis::new(profile.modulus_chain().top().moduli().to_vec());

    let chain = ModulusChain::from_top_basis(q.clone());
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    let plan = RnsNttPlan::new(q.moduli().to_vec(), degree);
    let prepared_plan = PreparedRnsNttPlan::new(&plan);

    let shape = Fft2Shape::new(64, 64);
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
        let diagonals = PackedFft2DifStageDiagonals::new(shape, axis, 64, FftDirection::Forward);

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

        for invocation in 0..1 {
            let optimized = execute_repeated_packed_fft2_dif_stage_cp_prepared(
                &ciphertext,
                &prepared_stage,
                &evaluator,
                (&prepared_left, &prepared_right),
                &chain,
                &plan,
                &prepared_plan,
            );

            let resident =
                ccmm_rs::eblas::fft::
                    execute_repeated_packed_fft2_dif_stage_cp_prepared_ntt_resident(
                        &ciphertext,
                        &prepared_stage,
                        &evaluator,
                        (&prepared_left, &prepared_right),
                        &chain,
                        &plan,
                        &prepared_plan,
                    );

            let shared_b =
                ccmm_rs::eblas::fft::execute_repeated_packed_fft2_dif_stage_cp_prepared_ntt_resident_shared_b(
                    &ciphertext,
                    &prepared_stage,
                    &evaluator,
                    (&prepared_left, &prepared_right),
                    &chain,
                    &plan,
                    &prepared_plan,
                );

            assert_eq!(
                resident.rlwe(),
                shared_b.rlwe(),
                "R17.5d.4b shared-b RLWE mismatch"
            );

            assert_eq!(
                resident.state(),
                shared_b.state(),
                "R17.5d.4b shared-b CKKS state mismatch"
            );

            println!("R17_5D4B_EXACT_RLWE status=PASS");

            assert_eq!(
                optimized.rlwe(),
                resident.rlwe(),
                "R17.4 exact RLWE mismatch axis={axis:?} invocation={invocation}"
            );
            assert_eq!(
                optimized.state(),
                resident.state(),
                "R17.4 CKKS state mismatch axis={axis:?} invocation={invocation}"
            );

            println!("R17_4G_EXACT_RLWE axis={axis:?} invocation={invocation} status=PASS");

            assert_eq!(
                reference.rlwe(),
                optimized.rlwe(),
                "RLWE mismatch axis={axis:?} invocation={invocation}",
            );

            assert_eq!(reference.level(), optimized.level());
            assert_eq!(reference.basis(), optimized.basis());
            assert_eq!(reference.scale().to_bits(), optimized.scale().to_bits(),);

            println!("R17_4G_REFERENCE_RLWE axis={axis:?} invocation={invocation} status=PASS");
        }
    }
    println!("R17_4G_EXACT_RLWE_STATUS=PASS");
}
