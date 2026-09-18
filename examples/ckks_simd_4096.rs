use ccmm_rs::ckks::{
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_ntt_rng, RnsGadgetLayout, RnsKeygenConfig,
    RnsMultiplicationKey,
};
use ccmm_rs::ring::{Polynomial, RnsNttPlan, RnsPolynomial};
use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn encode_slots(
    slots: &[Complex64],
    embedding: &CkksCanonicalEmbedding,
    basis: &ccmm_rs::ring::ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    let raw = embedding.slots_to_coefficients(slots);

    let signed: Vec<i128> = raw
        .iter()
        .map(|&coefficient| {
            let scaled = coefficient * scale;
            assert!(scaled.is_finite(), "scaled CKKS coefficient must be finite");
            scaled.round() as i128
        })
        .collect();

    let residues = basis
        .moduli()
        .iter()
        .copied()
        .map(|modulus| {
            let q = i128::from(modulus.value());

            Polynomial::new(
                modulus,
                signed
                    .iter()
                    .map(|&value| {
                        let residue = ((value % q) + q) % q;
                        residue as u64
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

fn main() {
    let profile = research_profile_4096();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let slot_count = profile.slot_count();
    let scale = profile.initial_scale();
    let top_basis = chain.top().clone();
    let top_plan = RnsNttPlan::new(top_basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);

    let secret: Vec<i8> = (0..degree)
        .map(|index| match index % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 1,
        })
        .collect();

    let mut lhs_slots = vec![Complex64::new(0.0, 0.0); slot_count];
    let mut rhs_slots = vec![Complex64::new(0.0, 0.0); slot_count];

    lhs_slots[0] = Complex64::new(0.50, 0.25);
    lhs_slots[1] = Complex64::new(-0.75, 0.125);
    lhs_slots[17] = Complex64::new(0.20, -0.30);
    lhs_slots[257] = Complex64::new(-0.40, 0.10);
    lhs_slots[1023] = Complex64::new(0.125, 0.375);
    lhs_slots[2047] = Complex64::new(-0.25, -0.50);

    rhs_slots[0] = Complex64::new(0.25, -0.50);
    rhs_slots[1] = Complex64::new(0.50, 0.25);
    rhs_slots[17] = Complex64::new(-0.10, 0.20);
    rhs_slots[257] = Complex64::new(0.30, -0.15);
    rhs_slots[1023] = Complex64::new(-0.50, 0.125);
    rhs_slots[2047] = Complex64::new(0.20, 0.40);

    let lhs_plaintext = encode_slots(&lhs_slots, &embedding, &top_basis, scale);
    let rhs_plaintext = encode_slots(&rhs_slots, &embedding, &top_basis, scale);

    let mut lhs_rng = ChaCha20Rng::seed_from_u64(0xCB10_0001);
    let mut rhs_rng = ChaCha20Rng::seed_from_u64(0xCB10_0002);
    let mut key_rng = ChaCha20Rng::seed_from_u64(0xCB10_0003);

    // This example isolates arithmetic semantics with zero encryption/key noise.
    // Gaussian ciphertext-noise validation is covered separately in the
    // repository's R2.9/R2.10 tests and characterization artifacts.
    let lhs_rlwe =
        encrypt_rns_raw_with_ntt_rng(&lhs_plaintext, 2, 0, &secret, &top_plan, &mut lhs_rng);

    let rhs_rlwe =
        encrypt_rns_raw_with_ntt_rng(&rhs_plaintext, 2, 0, &secret, &top_plan, &mut rhs_rng);

    let lhs = RnsCkksCiphertext::new(lhs_rlwe, CkksChainState::top(&chain, scale), &chain);

    let rhs = RnsCkksCiphertext::new(rhs_rlwe, CkksChainState::top(&chain, scale), &chain);

    let multiplication_key = RnsMultiplicationKey::generate_with_ntt_rng(
        RnsKeygenConfig {
            degree,
            plaintext_modulus: 2,
            noise_bound: 0,
            layout: RnsGadgetLayout::new(top_basis.clone(), vec![1, 2]),
            plan: &top_plan,
        },
        &secret,
        &mut key_rng,
    );

    let product = ccmm_rs::ckks::multiply_relinearize_rescale_rns_ckks_with_ntt(
        &lhs,
        &rhs,
        &multiplication_key,
        &chain,
        &top_plan,
    );

    let result_plan = RnsNttPlan::new(product.rlwe().basis().moduli().to_vec(), degree);

    let decrypted = decrypt_rns_raw_with_ntt(product.rlwe(), &secret, &result_plan);

    let modulus = decrypted.composite_modulus();

    let coefficients: Vec<f64> = decrypted
        .reconstruct_coefficients()
        .into_iter()
        .map(|value| centered(value, modulus) as f64 / product.scale())
        .collect();

    let observed = embedding.coefficients_to_slots(&coefficients);

    let expected: Vec<Complex64> = lhs_slots
        .iter()
        .zip(&rhs_slots)
        .map(|(&lhs, &rhs)| lhs * rhs)
        .collect();

    let mut max_error = 0.0_f64;
    let mut max_error_slot = 0_usize;

    for (index, (&actual, &reference)) in observed.iter().zip(&expected).enumerate() {
        let error = (actual - reference).norm();

        if error > max_error {
            max_error = error;
            max_error_slot = index;
        }
    }

    println!("PROFILE={}", profile.name());
    println!("SECURITY_BEARING={}", profile.security_bearing());
    println!("RING_DEGREE={degree}");
    println!("SLOT_COUNT={slot_count}");
    println!("INPUT_SCALE={scale:.6}");
    println!("OUTPUT_SCALE={:.6}", product.scale());
    println!("MAX_SLOT_ERROR={max_error:.12e}");
    println!("MAX_ERROR_SLOT={max_error_slot}");

    for index in [0_usize, 1, 17, 257, 1023, 2047] {
        println!(
            "SLOT={index} EXPECTED={:?} OBSERVED={:?} ERROR={:.6e}",
            expected[index],
            observed[index],
            (observed[index] - expected[index]).norm()
        );
    }

    let tolerance = 1.0e-3;
    println!("TOLERANCE={tolerance:.12e}");

    if max_error < tolerance {
        println!("CKKS_SIMD_EXAMPLE_STATUS=PASS");
    } else {
        println!("CKKS_SIMD_EXAMPLE_STATUS=FAIL");
        std::process::exit(1);
    }
}
