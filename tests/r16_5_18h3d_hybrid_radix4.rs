// R16.5.18h.3d: Gaussian QP hybrid radix-4 FFT stage.
// Research parameters only. Independent plaintext diagonal oracle.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next,
    research_profile_32768_for_levels, rotate_left_hybrid_rns_ckks_with_prepared_ntt,
    rotate_right_hybrid_rns_ckks_with_prepared_ntt, rotation_exponent_left,
    rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState, PreparedHybridRnsGaloisKey,
    PreparedRnsPlaintextNtt, RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator,
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

type Diagonals = BTreeMap<isize, Vec<Complex64>>;
type HybridKeys = BTreeMap<isize, PreparedHybridRnsGaloisKey>;

fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * exponent as f64 / n as f64)
}

fn stage_diagonals(slots: usize, span: usize) -> Diagonals {
    assert_eq!(span, 4);
    let quarter = span / 4;
    let mut diagonals = Diagonals::new();

    for i in 0..slots {
        let within = i % span;
        let p = within / quarter;
        let j = within % quarter;

        for q in 0..4 {
            let input_within = j + q * quarter;
            let offset = input_within as isize - within as isize;
            let coefficient = root(span, q * j) * root(4, p * q);

            diagonals
                .entry(offset)
                .or_insert_with(|| vec![Complex64::new(0.0, 0.0); slots])[i] = coefficient;
        }
    }

    diagonals
}

fn plaintext_stage(input: &[Complex64], diagonals: &Diagonals) -> Vec<Complex64> {
    let slots = input.len();
    let mut output = vec![Complex64::new(0.0, 0.0); slots];

    for (&offset, diagonal) in diagonals {
        for i in 0..slots {
            let source = (i as isize + offset).rem_euclid(slots as isize) as usize;
            output[i] += diagonal[i] * input[source];
        }
    }

    output
}

fn decode(
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
                .expect("coefficient conversion")
                / ciphertext.scale()
        })
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

#[test]
fn r16_5_18h3d_gaussian_hybrid_radix4_stage() {
    const DEGREE: usize = 1024;
    const SCALE: f64 = 33_554_432.0;
    const SIGMA: f64 = 3.19;

    let validated = research_profile_32768_for_levels(13, 3);
    let validated_q = validated.modulus_chain().top().clone();

    let q = ModulusBasis::new(validated_q.moduli()[..3].to_vec());
    let p = ModulusBasis::new(vec![
        Modulus::new(1_125_899_908_022_273),
        Modulus::new(1_125_899_908_612_097),
    ]);

    for modulus in q.moduli().iter().chain(p.moduli()) {
        assert_eq!(modulus.value() % (2 * DEGREE as u64), 1);
    }

    let chain = ModulusChain::from_top_basis(q.clone());
    let embedding = CkksCanonicalEmbedding::new(DEGREE);
    let slots = embedding.slot_count();

    let q_plan = RnsNttPlan::new(q.moduli().to_vec(), DEGREE);
    let prepared_plan = PreparedRnsNttPlan::new(&q_plan);

    let mut qp_moduli = q.moduli().to_vec();
    qp_moduli.extend_from_slice(p.moduli());
    let qp_plan = RnsNttPlan::new(qp_moduli, DEGREE);

    let input: Vec<Complex64> = (0..slots)
        .map(|i| {
            let k = i % 4;
            Complex64::new((k + 1) as f64 / 128.0, (k % 3) as f64 / 256.0)
        })
        .collect();

    let diagonals = stage_diagonals(slots, 4);
    let expected = plaintext_stage(&input, &diagonals);

    assert_eq!(
        diagonals.keys().copied().collect::<Vec<_>>(),
        vec![-3, -2, -1, 0, 1, 2, 3],
    );

    let secret: Vec<i8> = (0..DEGREE)
        .map(|i| match i % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x1803_000d);

    let plaintext = encode_rns(&input, &embedding, &q, SCALE);
    let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &q_plan,
        &mut rng,
    );

    let ciphertext = RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, SCALE), &chain);

    let keys: HybridKeys = diagonals
        .keys()
        .filter(|&&offset| offset != 0)
        .map(|&offset| {
            let exponent = if offset > 0 {
                rotation_exponent_left(DEGREE, offset as usize)
            } else {
                rotation_exponent_right(DEGREE, (-offset) as usize)
            };

            let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(q.clone(), 8),
                p.clone(),
                &secret,
                exponent,
                ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                2,
                &qp_plan,
                &mut rng,
            );

            (offset, key)
        })
        .collect();

    assert_eq!(keys.len(), 6);

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let plain_scale = chain
        .dropped_modulus(ciphertext.level())
        .expect("rescale level")
        .value() as f64;

    let mut accumulator: Option<RnsCkksCiphertext> = None;

    for (&offset, diagonal) in &diagonals {
        let rotated = if offset > 0 {
            Some(rotate_left_hybrid_rns_ckks_with_prepared_ntt(
                &ciphertext,
                offset as usize,
                keys.get(&offset).expect("left hybrid key"),
                &chain,
                &qp_plan,
            ))
        } else if offset < 0 {
            Some(rotate_right_hybrid_rns_ckks_with_prepared_ntt(
                &ciphertext,
                (-offset) as usize,
                keys.get(&offset).expect("right hybrid key"),
                &chain,
                &qp_plan,
            ))
        } else {
            None
        };

        let source = rotated.as_ref().unwrap_or(&ciphertext);
        assert_eq!(source.state(), ciphertext.state());

        let encoded = encode_rns(diagonal, &embedding, ciphertext.basis(), plain_scale);
        let prepared_plain = PreparedRnsPlaintextNtt::new(&encoded, &prepared_plan);

        let term = multiply_plain_rns_ckks_with_prepared_ntt(
            source,
            &prepared_plain,
            plain_scale,
            &chain,
            &prepared_plan,
        );

        accumulator = Some(match accumulator {
            Some(previous) => evaluator.add(&previous, &term),
            None => term,
        });
    }

    let result = rescale_rns_ckks_to_next(&accumulator.expect("nonempty radix-4 stage"), &chain);

    assert_eq!(result.level(), 1);
    assert_eq!(result.basis(), chain.level(1));

    let actual = decode(&result, &secret, &embedding);

    let mut error_sq = 0.0_f64;
    let mut reference_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (&value, &reference) in actual.iter().zip(expected.iter()) {
        let error = (value - reference).norm();
        error_sq += error * error;
        reference_sq += reference.norm_sqr();
        max_abs = max_abs.max(error);
    }

    let relative = (error_sq / reference_sq).sqrt();

    println!("R16_5_18H3D_DEGREE={DEGREE}");
    println!("R16_5_18H3D_SPAN=4");
    println!("R16_5_18H3D_ROTATION_KEYS={}", keys.len());
    println!("R16_5_18H3D_LEVEL={}", result.level());
    println!("R16_5_18H3D_SCALE={:.12e}", result.scale());
    println!("R16_5_18H3D_MAX_ABS={max_abs:.12e}");
    println!("R16_5_18H3D_REL_L2={relative:.12e}");

    assert!(max_abs.is_finite() && relative.is_finite());
    assert!(
        relative < 1.0e-3,
        "Gaussian hybrid radix-4 stage exceeded numerical tolerance"
    );

    println!("R16_5_18H3D_GAUSSIAN_HYBRID_RADIX4=PASS");
}
