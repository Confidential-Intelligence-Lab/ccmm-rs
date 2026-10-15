use std::time::Instant;
// R16.5.18h.4: complete Gaussian QP hybrid radix-4 FFT256.
// Research parameters only. Independent plaintext diagonal oracle.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_32768_for_levels, rotate_right_hybrid_rns_ckks_with_prepared_ntt,
    rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState, PreparedHybridRnsGaloisKey,
    RnsCkksCiphertext,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, Modulus,
    ModulusBasis, ModulusChain, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Diagonals = BTreeMap<isize, Vec<Complex64>>;

fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * exponent as f64 / n as f64)
}

fn stage_diagonals(slots: usize, span: usize) -> Diagonals {
    assert!([4, 16, 64, 256].contains(&span));
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

fn error_metrics(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());

    let mut error_sq = 0.0;
    let mut reference_sq = 0.0;
    let mut max_abs = 0.0_f64;

    for (&a, &e) in actual.iter().zip(expected) {
        let delta = (a - e).norm();
        error_sq += delta * delta;
        reference_sq += e.norm_sqr();
        max_abs = max_abs.max(delta);
    }

    assert!(reference_sq > 0.0);
    ((error_sq / reference_sq).sqrt(), max_abs)
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
fn r16_5_18h7_hybrid_rotation_sensitivity() {
    const DEGREE: usize = 1024;
    const SCALE: f64 = 33_554_432.0;
    const SIGMA: f64 = 3.19;

    let validated = research_profile_32768_for_levels(13, 3);
    let validated_q = validated.modulus_chain().top().clone();

    let q = ModulusBasis::new(validated_q.moduli()[..5].to_vec());
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

    assert_eq!(slots % 256, 0);

    let natural: Vec<Complex64> = (0..slots)
        .map(|i| {
            let x = i as f64;
            Complex64::new(0.001 * (0.013 * x).sin(), 0.001 * (0.017 * x).cos())
        })
        .collect();

    let reverse_base4_4 = |mut i: usize| {
        let mut reversed = 0;
        for _ in 0..4 {
            reversed = reversed * 4 + i % 4;
            i /= 4;
        }
        reversed
    };

    let packed: Vec<Complex64> = (0..slots)
        .map(|i| natural[(i / 256) * 256 + reverse_base4_4(i % 256)])
        .collect();

    let expected: Vec<Complex64> = (0..slots)
        .map(|i| {
            let block = (i / 256) * 256;
            let k = i % 256;
            (0..256)
                .map(|j| natural[block + j] * root(256, (k * j) % 256))
                .sum()
        })
        .collect();

    let spans = [4usize, 16, 64, 256];

    let diagonals: Vec<Diagonals> = spans
        .iter()
        .map(|&span| stage_diagonals(slots, span))
        .collect();

    for (&span, stage) in spans.iter().zip(&diagonals) {
        assert!(!stage.is_empty(), "empty radix-4 stage: {span}");
    }

    let mut stage_reference = packed.clone();
    let mut stage_references = Vec::new();

    for stage in &diagonals {
        stage_reference = plaintext_stage(&stage_reference, stage);
        stage_references.push(stage_reference.clone());
    }

    let (oracle_rel, oracle_max) = error_metrics(stage_references.last().unwrap(), &expected);

    println!("R16_5_18H5_PLAINTEXT_ORACLE_REL_L2={oracle_rel:.12e}");
    println!("R16_5_18H5_PLAINTEXT_ORACLE_MAX_ABS={oracle_max:.12e}");

    assert!(
        oracle_rel < 1.0e-10,
        "plaintext radix-4 schedule disagrees with DFT256"
    );

    let secret: Vec<i8> = (0..DEGREE)
        .map(|i| match i % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x1803_000d);

    let plaintext = encode_rns(&packed, &embedding, &q, SCALE);
    let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &q_plan,
        &mut rng,
    );

    let initial_ciphertext =
        RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, SCALE), &chain);

    let decoded_input = decode(&initial_ciphertext, &secret, &embedding);

    let offset: isize = -3;
    let exponent = rotation_exponent_right(DEGREE, 3);

    let expected: Vec<Complex64> = (0..slots)
        .map(|i| {
            let source = (i as isize + offset).rem_euclid(slots as isize) as usize;
            decoded_input[source]
        })
        .collect();

    let p2_values = [1_125_899_908_022_273_u64, 1_125_899_908_612_097_u64];

    let p3_candidate = 1_125_899_908_630_529_u64;

    assert_eq!(p3_candidate % (2 * DEGREE as u64), 1);
    assert!(!q.moduli().iter().any(|m| m.value() == p3_candidate));

    for p_count in [2usize, 3usize] {
        let mut p_values = p2_values.to_vec();

        if p_count == 3 {
            p_values.push(p3_candidate);
        }

        let auxiliary_basis =
            ModulusBasis::new(p_values.iter().copied().map(Modulus::new).collect());

        for modulus in auxiliary_basis.moduli() {
            assert_eq!(modulus.value() % (2 * DEGREE as u64), 1);
            assert!(
                !q.moduli().iter().any(|m| m.value() == modulus.value()),
                "Q/P overlap"
            );
        }

        let mut qp_moduli = q.moduli().to_vec();
        qp_moduli.extend_from_slice(auxiliary_basis.moduli());

        // Actual Rust NTT compatibility gate.
        let qp_plan = RnsNttPlan::new(qp_moduli, DEGREE);

        println!("R16_5_18H7_P{}_RUST_NTT_PLAN=PASS", p_count);

        for base_log in [2_u32, 3_u32, 4_u32, 8_u32] {
            let layout = BoundedGadgetLayout::new(q.clone(), base_log);

            let mut key_rng = ChaCha20Rng::seed_from_u64(
                0x1807_0000 + (p_count as u64) * 100 + u64::from(base_log),
            );

            let key_start = Instant::now();

            let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                layout,
                auxiliary_basis.clone(),
                &secret,
                exponent,
                ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                2,
                &qp_plan,
                &mut key_rng,
            );

            let keygen_seconds = key_start.elapsed().as_secs_f64();

            let rotation_start = Instant::now();

            let rotated = rotate_right_hybrid_rns_ckks_with_prepared_ntt(
                &initial_ciphertext,
                3,
                &key,
                &chain,
                &qp_plan,
            );

            let rotation_seconds = rotation_start.elapsed().as_secs_f64();

            assert_eq!(rotated.level(), 0);
            assert_eq!(rotated.state(), initial_ciphertext.state());

            let actual = decode(&rotated, &secret, &embedding);

            let (relative_l2, max_abs) = error_metrics(&actual, &expected);

            assert!(relative_l2.is_finite());
            assert!(max_abs.is_finite());

            println!(
                "R16_5_18H7_P={} BASE_LOG={} REL_L2={:.12e} MAX_ABS={:.12e} KEYGEN_S={:.6} ROTATE_S={:.6}",
                p_count,
                base_log,
                relative_l2,
                max_abs,
                keygen_seconds,
                rotation_seconds,
            );
        }
    }

    println!("R16_5_18H7_ROTATION_SENSITIVITY=PASS");
}
