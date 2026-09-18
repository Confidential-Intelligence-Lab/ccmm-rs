use ccmm_rs::ckks::{
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsCkksLevelKeys,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout,
    RnsKeygenConfig, RnsMultiplicationKey,
};
use ccmm_rs::matrix::RnsCkksCiphertextMatrix;
use ccmm_rs::ring::{ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn encode_scalar(
    value: f64,
    slot_count: usize,
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    let slots = vec![Complex64::new(value, 0.0); slot_count];
    let raw = embedding.slots_to_coefficients(&slots);

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
                    .map(|&coefficient| {
                        let residue = ((coefficient % q) + q) % q;
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

    assert_eq!(degree, 4096);
    assert_eq!(slot_count, 2048);

    // Deterministic ternary secret for this reproducible research example.
    let secret: Vec<i8> = (0..degree)
        .map(|index| match index % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 1,
        })
        .collect();

    // Column-major matrices:
    //
    // A = [  0.25    0.50  ]
    //     [ -0.25    0.125 ]
    //
    // B = [ 0.50   -0.25 ]
    //     [ 0.25    0.50 ]
    //
    // A * B =
    //
    //     [  0.25      0.1875 ]
    //     [ -0.09375    0.125  ]
    let lhs_values = [0.25, -0.25, 0.50, 0.125];
    let rhs_values = [0.50, 0.25, -0.25, 0.50];
    let expected = [[0.25, 0.1875], [-0.09375, 0.125]];

    let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };

    let encrypt_scalar = |value: f64, seed: u64| {
        let plaintext = encode_scalar(value, slot_count, &embedding, &top_basis, scale);
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        let inner = encrypt_rns_raw_with_distribution_ntt_rng(
            &plaintext,
            2,
            distribution,
            &secret,
            &top_plan,
            &mut rng,
        );

        RnsCkksCiphertext::new(inner, CkksChainState::top(&chain, scale), &chain)
    };

    let lhs = RnsCkksCiphertextMatrix::from_vec_column_major(
        2,
        2,
        lhs_values
            .iter()
            .enumerate()
            .map(|(index, &value)| encrypt_scalar(value, 0xCC11_0000 + index as u64))
            .collect(),
    );

    let rhs = RnsCkksCiphertextMatrix::from_vec_column_major(
        2,
        2,
        rhs_values
            .iter()
            .enumerate()
            .map(|(index, &value)| encrypt_scalar(value, 0xCC12_0000 + index as u64))
            .collect(),
    );

    // Current validated R2.9/R2.10 scope:
    // ciphertext encryption uses Gaussian sigma=3.19, while the evaluation
    // key remains zero-noise. The research profile is not security-bearing.
    let mut key_rng = ChaCha20Rng::seed_from_u64(0xCC13_0000);

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

    let mut level_keys = RnsCkksLevelKeys::new(0, top_basis);
    level_keys.set_multiplication_key(multiplication_key);

    let mut evaluation_keys = RnsCkksEvaluationKeys::new();
    evaluation_keys.insert_level(level_keys);

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let result = lhs.matmul_with_ntt(&rhs, &evaluator, &top_plan);

    let result_plan = RnsNttPlan::new(result.get(0, 0).basis().moduli().to_vec(), degree);

    let mut max_slot_error = 0.0_f64;

    println!("PROFILE={}", profile.name());
    println!("SECURITY_BEARING={}", profile.security_bearing());
    println!("RING_DEGREE={degree}");
    println!("SLOT_COUNT={slot_count}");
    println!("INPUT_SCALE={scale:.6}");
    println!("OUTPUT_SCALE={:.6}", result.scale());
    println!();

    for (row, expected_row) in expected.iter().enumerate() {
        for (col, &expected_value) in expected_row.iter().enumerate() {
            let ciphertext = result.get(row, col);
            let decrypted = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), &secret, &result_plan);
            let modulus = decrypted.composite_modulus();

            let coefficients: Vec<f64> = decrypted
                .reconstruct_coefficients()
                .into_iter()
                .map(|value| centered(value, modulus) as f64 / ciphertext.scale())
                .collect();

            let slots = embedding.coefficients_to_slots(&coefficients);

            let entry_max_error = slots
                .iter()
                .map(|&actual| (actual - Complex64::new(expected_value, 0.0)).norm())
                .fold(0.0_f64, f64::max);

            max_slot_error = max_slot_error.max(entry_max_error);

            println!(
                "C[{row},{col}] expected={expected_value:.8} \
                 observed_slot0={:.8}+{:.8}i \
                 max_slot_error={entry_max_error:.6e}",
                slots[0].re, slots[0].im
            );
        }
    }

    let tolerance = 1.0e-3;

    println!();
    println!("MAX_SLOT_ERROR={max_slot_error:.12e}");
    println!("TOLERANCE={tolerance:.12e}");

    if max_slot_error < tolerance {
        println!("RNS_CCMM_EXAMPLE_STATUS=PASS");
    } else {
        println!("RNS_CCMM_EXAMPLE_STATUS=FAIL");
        std::process::exit(1);
    }
}
