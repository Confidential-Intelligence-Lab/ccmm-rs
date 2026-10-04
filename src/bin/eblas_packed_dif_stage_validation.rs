use ccmm_rs::ckks::{
    research_profile_4096, rotation_exponent_left, rotation_exponent_right, CkksCanonicalEmbedding,
    CkksChainState, RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsCkksLevelKeys,
    RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{
    execute_packed_fft1_dif_stage_cp, execute_packed_fft1_dif_stage_pp, FftDirection,
    PackedFft1DifStageDiagonals,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout,
};
use ccmm_rs::ring::{ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const SIGMA: f64 = 3.19;
const TOLERANCE: f64 = 3.0e-3;
const LOGICAL_LENGTH: usize = 8;
const SPANS: [usize; 3] = [8, 4, 2];

fn block_sizes(limb_count: usize) -> Vec<usize> {
    vec![1; limb_count]
}

fn encode_slots_rns(
    slots: &[Complex64],
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    assert_eq!(slots.len(), embedding.slot_count());

    let coefficients = embedding.slots_to_coefficients(slots);

    let residues = basis
        .moduli()
        .iter()
        .copied()
        .map(|modulus| {
            let q = i128::from(modulus.value());

            Polynomial::new(
                modulus,
                coefficients
                    .iter()
                    .map(|&coefficient| {
                        ((coefficient * scale).round() as i128).rem_euclid(q) as u64
                    })
                    .collect(),
            )
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

fn centered(value: u128, modulus: u128) -> i128 {
    if value > modulus / 2 {
        value as i128 - modulus as i128
    } else {
        value as i128
    }
}

fn decrypt_slots(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let plan = RnsNttPlan::new(
        ciphertext.basis().moduli().to_vec(),
        ciphertext.rlwe().degree(),
    );

    let plaintext = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);
    let modulus = plaintext.composite_modulus();

    let coefficients: Vec<f64> = plaintext
        .reconstruct_coefficients()
        .into_iter()
        .map(|value| centered(value, modulus) as f64 / ciphertext.scale())
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

fn main() {
    let profile = research_profile_4096();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let basis = chain.top().clone();

    let embedding = CkksCanonicalEmbedding::new(degree);
    let slot_count = embedding.slot_count();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

    let logical_input: Vec<Complex64> = (0..LOGICAL_LENGTH)
        .map(|index| {
            let re = ((index * 7 + 3) % 17) as f64 - 8.0;
            let im = ((index * 11 + 5) % 19) as f64 - 9.0;
            Complex64::new(re / 128.0, im / 128.0)
        })
        .collect();

    let mut physical_input = vec![Complex64::new(0.0, 0.0); slot_count];
    physical_input[..LOGICAL_LENGTH].copy_from_slice(&logical_input);

    let plaintext = encode_slots_rns(&physical_input, &embedding, &basis, scale);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4449_465f_5354_0001);

    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x4449_465f_5354_1001);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &plan,
        &mut encryption_rng,
    );

    let ciphertext = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

    let mut level_keys = RnsCkksLevelKeys::new(0, basis.clone());

    for (index, shift) in [1usize, 2, 4].into_iter().enumerate() {
        for (tag, exponent) in [
            (0u64, rotation_exponent_left(degree, shift)),
            (1u64, rotation_exponent_right(degree, shift)),
        ] {
            let mut rng =
                ChaCha20Rng::seed_from_u64(0x4449_465f_5354_2001 ^ ((index as u64) << 8) ^ tag);

            let key = RnsGaloisKey::generate_with_rng(
                degree,
                2,
                0,
                &secret,
                exponent,
                RnsGadgetLayout::new(basis.clone(), block_sizes(basis.len())),
                &mut rng,
            );

            level_keys.insert_galois_key(key);
        }
    }

    let mut evaluation_keys = RnsCkksEvaluationKeys::new();
    evaluation_keys.insert_level(level_keys);

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    println!("PACKED_DIF_STAGE_SLOT_COUNT={slot_count}");
    println!("PACKED_DIF_STAGE_LOGICAL_LENGTH={LOGICAL_LENGTH}");
    println!("PACKED_DIF_STAGE_INPUT_LEVEL={}", ciphertext.level());

    for span in SPANS {
        let diagonals =
            PackedFft1DifStageDiagonals::new(LOGICAL_LENGTH, span, FftDirection::Forward);

        let expected = execute_packed_fft1_dif_stage_pp(&logical_input, &diagonals);

        let encrypted = execute_packed_fft1_dif_stage_cp(
            &ciphertext,
            &diagonals,
            &evaluator,
            &embedding,
            &chain,
            &plan,
        );

        let actual = decrypt_slots(&encrypted, &secret, &embedding);

        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for index in 0..LOGICAL_LENGTH {
            let error = actual[index] - expected[index];

            squared_error += error.norm_sqr();
            squared_reference += expected[index].norm_sqr();
            max_abs = max_abs.max(error.norm());
        }

        let rel_l2 = if squared_reference > 0.0 {
            (squared_error / squared_reference).sqrt()
        } else {
            squared_error.sqrt()
        };

        let inactive_max_abs = actual[LOGICAL_LENGTH..]
            .iter()
            .map(|value| value.norm())
            .fold(0.0_f64, f64::max);

        assert_eq!(encrypted.level(), ciphertext.level() + 1);
        assert!((encrypted.scale() / ciphertext.scale() - 1.0).abs() < 1.0e-12);
        assert!(rel_l2 <= TOLERANCE);
        assert!(max_abs <= TOLERANCE);
        assert!(inactive_max_abs <= TOLERANCE);

        println!(
            "PACKED_DIF_STAGE_SPAN={span} ROTATION={} OUTPUT_LEVEL={} LEVELS_CONSUMED={} REL_L2={rel_l2:.12e} MAX_ABS={max_abs:.12e} INACTIVE_MAX_ABS={inactive_max_abs:.12e} STATUS=PASS",
            diagonals.rotation(),
            encrypted.level(),
            encrypted.level() - ciphertext.level(),
        );
    }

    println!("PACKED_DIF_STAGE_STATUS=PASS");
}
