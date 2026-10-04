use ccmm_rs::ckks::{
    research_profile_4096, rotation_exponent_left, CkksCanonicalEmbedding, CkksChainState,
    RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsCkksLevelKeys, RnsGaloisKey,
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
const SHIFTS: [usize; 4] = [1, 2, 4, 8];

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
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x524f_5441_5445_0001);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let slot_count = embedding.slot_count();

    let input_slots: Vec<Complex64> = (0..slot_count)
        .map(|index| {
            let re = ((index * 7 + 3) % 31) as f64 - 15.0;
            let im = ((index * 13 + 5) % 29) as f64 - 14.0;
            Complex64::new(re / 256.0, im / 256.0)
        })
        .collect();

    let plaintext = encode_slots_rns(&input_slots, &embedding, &basis, scale);

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x524f_5441_5445_1001);

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

    for (index, shift) in SHIFTS.into_iter().enumerate() {
        let exponent = rotation_exponent_left(degree, shift);
        let mut rng = ChaCha20Rng::seed_from_u64(0x524f_5441_5445_2001 ^ index as u64);

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

    let mut evaluation_keys = RnsCkksEvaluationKeys::new();
    evaluation_keys.insert_level(level_keys);

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    println!("PACKED_ROTATION_SLOT_COUNT={slot_count}");
    println!("PACKED_ROTATION_INPUT_LEVEL={}", ciphertext.level());
    println!("PACKED_ROTATION_INPUT_SCALE={:.12e}", ciphertext.scale());

    let mut global_max_abs = 0.0_f64;

    for shift in SHIFTS {
        let rotated = evaluator.rotate_left(&ciphertext, shift);
        let actual = decrypt_slots(&rotated, &secret, &embedding);

        let expected: Vec<Complex64> = (0..slot_count)
            .map(|index| input_slots[(index + shift) % slot_count])
            .collect();

        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        let mut max_abs = 0.0_f64;
        let mut max_slot = 0usize;

        for (index, (&observed, &reference)) in actual.iter().zip(&expected).enumerate() {
            let error = observed - reference;
            let abs = error.norm();

            squared_error += error.norm_sqr();
            squared_reference += reference.norm_sqr();

            if abs > max_abs {
                max_abs = abs;
                max_slot = index;
            }
        }

        let rel_l2 = if squared_reference > 0.0 {
            (squared_error / squared_reference).sqrt()
        } else {
            squared_error.sqrt()
        };

        global_max_abs = global_max_abs.max(max_abs);

        assert_eq!(
            rotated.level(),
            ciphertext.level(),
            "packed rotation must preserve CKKS level"
        );
        assert_eq!(
            rotated.basis(),
            ciphertext.basis(),
            "packed rotation must preserve CKKS basis"
        );
        assert!(
            (rotated.scale() / ciphertext.scale() - 1.0).abs() < 1.0e-12,
            "packed rotation must preserve CKKS scale"
        );
        assert!(
            rel_l2 <= TOLERANCE,
            "packed left rotation {shift} relative L2 error {rel_l2:e} exceeds tolerance"
        );
        assert!(
            max_abs <= TOLERANCE,
            "packed left rotation {shift} maximum error {max_abs:e} exceeds tolerance"
        );

        println!(
            "PACKED_ROTATION_SHIFT={shift} OUTPUT_LEVEL={} REL_L2={rel_l2:.12e} MAX_ABS={max_abs:.12e} MAX_ERROR_SLOT={max_slot} STATUS=PASS",
            rotated.level()
        );

        for index in 0..4 {
            println!(
                "PACKED_ROTATION_SHIFT_{shift}_SLOT_{index} OUTPUT=({:.6e},{:.6e}) EXPECTED=({:.6e},{:.6e})",
                actual[index].re,
                actual[index].im,
                expected[index].re,
                expected[index].im,
            );
        }
    }

    println!("PACKED_ROTATION_GLOBAL_MAX_ABS={global_max_abs:.12e}");
    println!("PACKED_ROTATION_STATUS=PASS");
}
