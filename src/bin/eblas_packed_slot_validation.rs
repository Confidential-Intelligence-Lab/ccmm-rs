use ccmm_rs::ckks::{
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
};
use ccmm_rs::eblas::multiply_complex_slots_cp;
use ccmm_rs::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng};
use ccmm_rs::ring::{ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const SIGMA: f64 = 3.19;
const TOLERANCE: f64 = 3.0e-3;

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

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5041_434b_4544_0001);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let slot_count = embedding.slot_count();

    let input_slots: Vec<Complex64> = (0..slot_count)
        .map(|index| {
            let a = ((index * 7 + 3) % 17) as f64 - 8.0;
            let b = ((index * 11 + 5) % 19) as f64 - 9.0;

            Complex64::new(a / 128.0, b / 128.0)
        })
        .collect();

    let public_slots: Vec<Complex64> = (0..slot_count)
        .map(|index| match index % 4 {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        })
        .collect();

    let expected: Vec<Complex64> = input_slots
        .iter()
        .zip(&public_slots)
        .map(|(&value, &factor)| value * factor)
        .collect();

    let plaintext = encode_slots_rns(&input_slots, &embedding, &basis, scale);

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x5041_434b_4544_1001);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &plan,
        &mut encryption_rng,
    );

    let ciphertext = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

    let result = multiply_complex_slots_cp(&ciphertext, &public_slots, &embedding, &chain, &plan);

    let actual = decrypt_slots(&result, &secret, &embedding);

    assert_eq!(actual.len(), expected.len());

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

    println!("PACKED_CP_SLOT_COUNT={slot_count}");
    println!("PACKED_CP_INPUT_LEVEL={}", ciphertext.level());
    println!("PACKED_CP_OUTPUT_LEVEL={}", result.level());
    println!(
        "PACKED_CP_LEVELS_CONSUMED={}",
        result.level() - ciphertext.level()
    );
    println!("PACKED_CP_INPUT_SCALE={:.12e}", ciphertext.scale());
    println!("PACKED_CP_OUTPUT_SCALE={:.12e}", result.scale());
    println!("PACKED_CP_REL_L2={rel_l2:.12e}");
    println!("PACKED_CP_MAX_ABS_ERROR={max_abs:.12e}");
    println!("PACKED_CP_MAX_ERROR_SLOT={max_slot}");

    for index in 0..8 {
        println!(
            "PACKED_CP_SLOT_{} INPUT=({:.6e},{:.6e}) FACTOR=({:.1},{:.1}) OUTPUT=({:.6e},{:.6e}) EXPECTED=({:.6e},{:.6e})",
            index,
            input_slots[index].re,
            input_slots[index].im,
            public_slots[index].re,
            public_slots[index].im,
            actual[index].re,
            actual[index].im,
            expected[index].re,
            expected[index].im,
        );
    }

    assert_eq!(
        result.level(),
        ciphertext.level() + 1,
        "packed CP multiplication must consume exactly one CKKS level"
    );

    assert!(
        (result.scale() / ciphertext.scale() - 1.0).abs() < 1.0e-12,
        "packed CP multiplication must restore the input scale"
    );

    assert!(
        rel_l2 <= TOLERANCE,
        "packed CP relative L2 error {rel_l2:e} exceeds tolerance {TOLERANCE:e}"
    );

    assert!(
        max_abs <= TOLERANCE,
        "packed CP maximum absolute error {max_abs:e} exceeds tolerance {TOLERANCE:e}"
    );

    println!("PACKED_CP_STATUS=PASS");
}
