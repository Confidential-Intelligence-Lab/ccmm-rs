//! R16.5.18h.2c: Gaussian QP hybrid prepared FFT-stage regression.
//! Research parameters only; no cryptographic security claim.

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_32768_for_levels, rotation_exponent_left, rotation_exponent_right,
    CkksCanonicalEmbedding, CkksChainState, PreparedHybridRnsGaloisKey, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::fft::{
    execute_repeated_packed_fft2_dif_stage_cp_hybrid_prepared,
    execute_repeated_packed_fft2_dif_stage_pp, Fft2Shape, FftDirection, PackedFft2Axis,
    PackedFft2DifStageDiagonals, PreparedRepeatedPackedFft2DifStage,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, Modulus,
    ModulusBasis, ModulusChain, PreparedRnsNttPlan, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[test]
fn r16_5_18h2c_gaussian_hybrid_prepared_fft_stage() {
    let degree = 1024usize;
    let scale = (1u64 << 25) as f64;

    let validated_profile = research_profile_32768_for_levels(13, 3);
    let validated_q = validated_profile.modulus_chain().top().clone();

    let q = ModulusBasis::new(validated_q.moduli()[..3].to_vec());

    let p = ModulusBasis::new(vec![
        Modulus::new(1_125_899_908_022_273),
        Modulus::new(1_125_899_908_612_097),
    ]);

    for modulus in q.moduli().iter().chain(p.moduli()) {
        assert_eq!(
            modulus.value() % (2 * degree as u64),
            1,
            "modulus must support the degree-1024 negacyclic NTT"
        );
    }

    let chain = ModulusChain::from_top_basis(q.clone());
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    let plan = RnsNttPlan::new(q.moduli().to_vec(), degree);
    let prepared_plan = PreparedRnsNttPlan::new(&plan);

    let mut qp_moduli = q.moduli().to_vec();
    qp_moduli.extend_from_slice(p.moduli());
    let qp_plan = RnsNttPlan::new(qp_moduli, degree);

    let shape = Fft2Shape::new(2, 2);
    let tiles = slots / shape.elements();

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

    let mut rng = ChaCha20Rng::seed_from_u64(0x1802_0c01);

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

        let make_key = |exponent: usize, rng: &mut ChaCha20Rng| {
            PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(q.clone(), 8),
                p.clone(),
                &secret,
                exponent,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                2,
                &qp_plan,
                rng,
            )
        };

        let left = make_key(rotation_exponent_left(degree, rotation), &mut rng);
        let right = make_key(rotation_exponent_right(degree, rotation), &mut rng);

        let prepared_stage = PreparedRepeatedPackedFft2DifStage::new(
            &ciphertext,
            &diagonals,
            tiles,
            &embedding,
            &chain,
            &plan,
        );

        let result = execute_repeated_packed_fft2_dif_stage_cp_hybrid_prepared(
            &ciphertext,
            &prepared_stage,
            &evaluator,
            (&left, &right),
            &chain,
            &qp_plan,
            &prepared_plan,
        );

        assert_eq!(result.level(), ciphertext.level() + 1);
        assert_eq!(result.basis(), chain.level(result.level()));

        let result_plan = RnsNttPlan::new(result.basis().moduli().to_vec(), degree);

        let decrypted = decrypt_rns_raw_with_ntt(result.rlwe(), &secret, &result_plan);

        let modulus = composite_modulus_big(decrypted.basis());

        let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
            .iter()
            .map(|value| {
                centered_representative_big(value, &modulus)
                    .to_f64()
                    .expect("coefficient conversion")
                    / result.scale()
            })
            .collect();

        let actual = embedding.coefficients_to_slots(&coefficients);

        let expected = execute_repeated_packed_fft2_dif_stage_pp(&input, &diagonals, tiles);

        let mut error_sq = 0.0_f64;
        let mut reference_sq = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for (a, e) in actual.iter().zip(&expected) {
            let error = (*a - *e).norm();
            error_sq += error * error;
            reference_sq += e.norm_sqr();
            max_abs = max_abs.max(error);
        }

        let rel_l2 = (error_sq / reference_sq).sqrt();

        println!(
            "R16_5_18H2C_AXIS={axis:?} ROTATION={rotation} \
             MAX_ABS={max_abs:.12e} REL_L2={rel_l2:.12e} \
             LEVEL={} SCALE={:.9e}",
            result.level(),
            result.scale(),
        );

        assert!(max_abs.is_finite());
        assert!(rel_l2.is_finite());

        assert!(
            max_abs < 1.0e-3 && rel_l2 < 1.0e-3,
            "Gaussian hybrid FFT-stage numerical error exceeds tolerance"
        );
    }

    println!("R16_5_18H2C_GAUSSIAN_HYBRID_FFT_STAGE=PASS");
}
