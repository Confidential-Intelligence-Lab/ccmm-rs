// R30_8C_ENCRYPTED_FFT2160_NUMERICAL
//
// Opt-in numerical validation of the production encrypted FFT2160.
// R30_8C_RUN=1 cargo test --release --test r30_8c_encrypted_fft2160 -- --nocapture
//
// Research-only deterministic key material. Never use these keys in production.

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    mod_switch_rns_ckks_to_next, research_profile_65536_for_levels, rotation_exponent_left,
    rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState, PreparedHybridRnsGaloisKey,
    RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::encrypted_fft2160::{
    execute_encrypted_fft2160, fft16_stage_diagonals, Fft135Plan,
};
use ccmm_rs::eblas::prepared_radix4_fft256::{RotationKeys, StageContext};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, ModulusBasis,
    PreparedRnsNttPlan, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;

use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::f64::consts::PI;
use std::time::Instant;

const BRANCHES: usize = 135;
const ROWS: usize = 2160;
const IMPULSE: usize = 422;
const SIGMA: f64 = 3.19;

fn reverse_base4_2(i: usize) -> usize {
    assert!(i < 16);
    (i % 4) * 4 + i / 4
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
                .expect("coefficient conversion")
                / ct.scale()
        })
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

fn generate_keys(
    diagonals: &ccmm_rs::eblas::prepared_radix4_fft256::Diagonals,
    basis: &ModulusBasis,
    auxiliary_basis: &ModulusBasis,
    qp_plan: &RnsNttPlan,
    degree: usize,
    secret: &[i8],
    rng: &mut ChaCha20Rng,
) -> RotationKeys {
    diagonals
        .keys()
        .filter(|&&offset| offset != 0)
        .map(|&offset| {
            let exponent = if offset > 0 {
                rotation_exponent_left(degree, offset as usize)
            } else {
                rotation_exponent_right(degree, (-offset) as usize)
            };

            let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(basis.clone(), 8),
                auxiliary_basis.clone(),
                secret,
                exponent,
                ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                2,
                qp_plan,
                rng,
            );

            (offset, key)
        })
        .collect()
}

#[test]
fn r30_8c_encrypted_fft2160_complex_impulse_oracle() {
    if std::env::var_os("R30_8C_RUN").is_none() {
        println!("R30_8C_STATUS=SKIPPED_OPT_IN");
        return;
    }

    let total_started = Instant::now();

    let profile = research_profile_65536_for_levels(13, 3);
    let degree = profile.degree();
    let chain = profile.modulus_chain();
    let scale = profile.initial_scale();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    assert_eq!(degree, 65536);
    assert_eq!(slots, 32768);
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

    let mut rng = ChaCha20Rng::seed_from_u64(0x308C_2160);
    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let amplitude = Complex64::new(8.0e-8, -6.0e-8);

    // Production packing: 135 ciphertexts, each with 16 local rows
    // at stride 256. The local rows are radix-4 digit-reversed.
    let mut ciphertexts = Vec::with_capacity(BRANCHES);

    for branch in 0..BRANCHES {
        let mut values = vec![Complex64::new(0.0, 0.0); slots];

        for packed_local in 0..16 {
            let natural_local = reverse_base4_2(packed_local);
            let vertical_index = branch * 16 + natural_local;

            if vertical_index == IMPULSE {
                values[packed_local * 256] = amplitude;
            }
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

    println!("R30_8C_INPUT_CIPHERTEXTS={}", ciphertexts.len());
    println!("R30_8C_INPUT_LEVEL=7");
    println!("R30_8C_IMPULSE_INDEX={IMPULSE}");

    // FFT16 requires hybrid-QP rotation keys at levels 11 and 12.
    // R30_8C_VALIDATED_65536_AUXILIARY_BASIS
    // Q occupies the first 16 validated primes. P uses the next two.
    let validated_pool = research_profile_65536_for_levels(15, 3);
    let validated_moduli = validated_pool.modulus_chain().top().moduli().to_vec();

    assert_eq!(basis0.moduli().len(), 16);
    assert_eq!(validated_moduli.len(), 18);
    assert_eq!(&validated_moduli[..16], basis0.moduli());

    let auxiliary_basis = ModulusBasis::new(validated_moduli[16..18].to_vec());

    for modulus in auxiliary_basis.moduli() {
        assert_eq!(
            (modulus.value() - 1) % (2 * degree as u64),
            0,
            "auxiliary modulus must support degree-65536 NTT"
        );
        assert!(!basis0.moduli().contains(modulus), "Q/P modulus overlap");
    }

    println!(
        "R30_8C_AUXILIARY_MODULI={},{}",
        auxiliary_basis.moduli()[0].value(),
        auxiliary_basis.moduli()[1].value()
    );

    let bases: Vec<ModulusBasis> = [11usize, 12]
        .iter()
        .map(|&level| chain.level(level).clone())
        .collect();

    for basis in &bases {
        for modulus in auxiliary_basis.moduli() {
            assert!(
                !basis.moduli().contains(modulus),
                "hybrid Q/P modulus overlap"
            );
        }
    }

    let plans: Vec<RnsNttPlan> = bases
        .iter()
        .map(|basis| RnsNttPlan::new(basis.moduli().to_vec(), degree))
        .collect();

    let prepared: Vec<PreparedRnsNttPlan> = plans.iter().map(PreparedRnsNttPlan::new).collect();

    let qp_plans: Vec<RnsNttPlan> = bases
        .iter()
        .map(|basis| {
            let mut moduli = basis.moduli().to_vec();
            moduli.extend_from_slice(auxiliary_basis.moduli());
            RnsNttPlan::new(moduli, degree)
        })
        .collect();

    let keygen_started = Instant::now();

    let rotation_keys: Vec<RotationKeys> = [4usize, 16]
        .iter()
        .enumerate()
        .map(|(stage, &span)| {
            let diagonals = fft16_stage_diagonals(slots, span, 0, 0);

            generate_keys(
                &diagonals,
                &bases[stage],
                &auxiliary_basis,
                &qp_plans[stage],
                degree,
                &secret,
                &mut rng,
            )
        })
        .collect();

    println!(
        "R30_8C_HYBRID_QP_KEYGEN_MS={:.3}",
        keygen_started.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "R30_8C_ROTATION_KEY_COUNTS={},{}",
        rotation_keys[0].len(),
        rotation_keys[1].len()
    );

    let ctx0 = StageContext {
        keys: &rotation_keys[0],
        evaluator: &evaluator,
        chain: &chain,
        qp_plan: &qp_plans[0],
        prepared: &prepared[0],
    };

    let ctx1 = StageContext {
        keys: &rotation_keys[1],
        evaluator: &evaluator,
        chain: &chain,
        qp_plan: &qp_plans[1],
        prepared: &prepared[1],
    };

    let execution_started = Instant::now();

    let outputs = execute_encrypted_fft2160(
        ciphertexts,
        0,
        &embedding,
        &evaluator,
        &chain,
        [&prepared[0], &prepared[1]],
        [&ctx0, &ctx1],
    );

    println!(
        "R30_8C_FFT2160_EXECUTION_MS={:.3}",
        execution_started.elapsed().as_secs_f64() * 1000.0
    );

    assert_eq!(outputs.len(), BRANCHES);
    assert!(outputs.iter().all(|ct| ct.level() == 13));

    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut max_absolute_error = 0.0_f64;
    let mut max_phase_error = 0.0_f64;
    let mut seen = vec![false; ROWS];

    for (output_index, ct) in outputs.iter().enumerate() {
        let k_cross = Fft135Plan::natural_frequency(output_index);
        let decoded = decode(ct, &secret, &embedding);

        for k_local in 0..16 {
            let frequency = k_cross + BRANCHES * k_local;
            assert!(!seen[frequency]);
            seen[frequency] = true;

            let angle = -2.0 * PI * (frequency * IMPULSE % ROWS) as f64 / ROWS as f64;
            let expected = amplitude * Complex64::from_polar(1.0, angle);
            let actual = decoded[k_local * 256];

            let error = (actual - expected).norm();
            squared_error += error * error;
            squared_reference += expected.norm_sqr();
            max_absolute_error = max_absolute_error.max(error);

            let ratio = actual / expected;
            max_phase_error = max_phase_error.max(ratio.arg().abs());

            if output_index < 2 && k_local < 4 {
                println!(
                    "R30_8C_SAMPLE_K{}_ACTUAL=({:.12e},{:.12e})",
                    frequency, actual.re, actual.im
                );
                println!(
                    "R30_8C_SAMPLE_K{}_EXPECTED=({:.12e},{:.12e})",
                    frequency, expected.re, expected.im
                );
            }
        }
    }

    assert!(seen.iter().all(|&value| value));

    let relative_l2 = (squared_error / squared_reference).sqrt();

    println!("R30_8C_OUTPUT_CIPHERTEXTS={}", outputs.len());
    println!("R30_8C_OUTPUT_LEVEL=13");
    println!("R30_8C_VALIDATED_FREQUENCIES={ROWS}");
    println!("R30_8C_RELATIVE_L2={relative_l2:.12e}");
    println!("R30_8C_MAX_ABSOLUTE_ERROR={max_absolute_error:.12e}");
    println!("R30_8C_MAX_PHASE_ERROR_RAD={max_phase_error:.12e}");
    println!(
        "R30_8C_TOTAL_MS={:.3}",
        total_started.elapsed().as_secs_f64() * 1000.0
    );

    assert!(
        relative_l2.is_finite() && relative_l2 < 0.1,
        "FFT2160 relative L2 exceeds initial research tolerance"
    );

    assert!(
        max_absolute_error.is_finite(),
        "FFT2160 maximum absolute error is nonfinite"
    );

    println!("R30_8C_STATUS=PASS");
}
