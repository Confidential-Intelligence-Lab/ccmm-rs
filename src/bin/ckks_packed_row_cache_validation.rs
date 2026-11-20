use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_65536_for_levels, rotation_exponent_left, rotation_exponent_right,
    CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext, RnsCkksEvaluationKeys,
    RnsCkksEvaluator, RnsCkksLevelKeys, RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{
    execute_packed_1k_fft_rows_cp, execute_packed_1k_fft_rows_cp_cached,
    PackedRowPreparationContext,
};
use ccmm_rs::grafting::{
    encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout, RnsKeygenConfig,
};
use ccmm_rs::ring::RnsNttPlan;
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::time::Instant;

const START_LEVEL: usize = 2;
const EXPECTED_FINAL_LEVEL: usize = 12;
const CT_PER_MACRO: usize = 32;
const SIGMA: f64 = 3.19;

fn main() {
    let profile = research_profile_65536_for_levels(24, 2);
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    let embedding = CkksCanonicalEmbedding::new(degree);

    let basis = chain.level(START_LEVEL).clone();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

    assert_eq!(degree, 65_536);
    assert_eq!(basis.len(), 24);

    println!("ROW_CACHE_VALIDATION_BEGIN");
    println!("ROW_CACHE_START_LEVEL={START_LEVEL}");
    println!("ROW_CACHE_START_LIMBS={}", basis.len());
    println!("ROW_CACHE_SECURITY_BEARING=false");

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x524f_5743_4143_4845);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&coefficient| coefficient == 0) {
        secret[0] = 1;
    }

    let keygen_start = Instant::now();
    let mut evaluation_keys = RnsCkksEvaluationKeys::new();
    let mut key_count = 0usize;

    for (level, rotation) in (6usize..12).zip([32usize, 16, 8, 4, 2, 1]) {
        let level_basis = chain.level(level).clone();
        let layout = RnsGadgetLayout::new(level_basis.clone(), vec![1; level_basis.len()]);
        let level_plan = RnsNttPlan::new(level_basis.moduli().to_vec(), degree);

        let left = rotation_exponent_left(degree, rotation);
        let right = rotation_exponent_right(degree, rotation);
        let mut level_keys = RnsCkksLevelKeys::new(level, level_basis);

        for (index, exponent) in [left, right].into_iter().enumerate() {
            if index == 1 && left == right {
                continue;
            }

            let mut rng = ChaCha20Rng::seed_from_u64(
                0x524f_574b_4559_0000 ^ ((level as u64) << 8) ^ (index as u64),
            );

            let key = RnsGaloisKey::generate_with_ntt_rng(
                RnsKeygenConfig {
                    degree,
                    plaintext_modulus: 2,
                    noise_bound: 0,
                    layout: layout.clone(),
                    plan: &level_plan,
                },
                &secret,
                exponent,
                &mut rng,
            );

            level_keys.insert_galois_key(key);
            key_count += 1;
        }

        evaluation_keys.insert_level(level_keys);
        println!("ROW_CACHE_KEYGEN_LEVEL={level} rotation={rotation}");
    }

    println!("ROW_CACHE_KEY_COUNT={key_count}");
    println!("ROW_CACHE_KEYGEN_MS={}", keygen_start.elapsed().as_millis());

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let make_macrotile = |macro_index: usize| -> Vec<RnsCkksCiphertext> {
        (0..CT_PER_MACRO)
            .map(|ct_index| {
                let slots = (0..embedding.slot_count())
                    .map(|slot| {
                        let value = (slot % 257 + ct_index * 7 + macro_index * 13) as f64 / 1024.0;
                        Complex64::new(value, 0.0)
                    })
                    .collect::<Vec<_>>();

                let plaintext = encode_rns(&slots, &embedding, &basis, scale);

                let mut rng = ChaCha20Rng::seed_from_u64(
                    0x524f_5745_4e43_0000 ^ ((macro_index as u64) << 16) ^ ct_index as u64,
                );

                let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                    &plaintext,
                    2,
                    ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                    &secret,
                    &plan,
                    &mut rng,
                );

                RnsCkksCiphertext::new(
                    rlwe,
                    CkksChainState::new(&chain, START_LEVEL, scale),
                    &chain,
                )
            })
            .collect()
    };

    let mut cache = PackedRowPreparationContext::new();

    let first_input = make_macrotile(0);
    let first_start = Instant::now();
    let first_output = execute_packed_1k_fft_rows_cp_cached(
        first_input,
        &evaluator,
        &embedding,
        &chain,
        2,
        4,
        &mut cache,
    );

    assert_eq!(first_output.len(), CT_PER_MACRO);
    assert!(first_output
        .iter()
        .all(|ct| ct.level() == EXPECTED_FINAL_LEVEL));

    println!(
        "ROW_CACHE_FIRST_MACRO_MS={}",
        first_start.elapsed().as_millis()
    );

    drop(first_output);

    let second_input = make_macrotile(1);
    let reference_input = second_input.clone();

    let cached_start = Instant::now();
    let cached = execute_packed_1k_fft_rows_cp_cached(
        second_input,
        &evaluator,
        &embedding,
        &chain,
        2,
        4,
        &mut cache,
    );
    let cached_ms = cached_start.elapsed().as_millis();

    let uncached_start = Instant::now();
    let uncached =
        execute_packed_1k_fft_rows_cp(reference_input, &evaluator, &embedding, &chain, 2, 4);
    let uncached_ms = uncached_start.elapsed().as_millis();

    assert_eq!(cached.len(), CT_PER_MACRO);
    assert_eq!(uncached.len(), CT_PER_MACRO);

    for (index, (lhs, rhs)) in cached.iter().zip(uncached.iter()).enumerate() {
        assert_eq!(
            lhs.rlwe(),
            rhs.rlwe(),
            "cached/uncached RLWE mismatch at ciphertext {index}"
        );
        assert_eq!(
            lhs.state(),
            rhs.state(),
            "cached/uncached state mismatch at ciphertext {index}"
        );
        assert_eq!(lhs.level(), EXPECTED_FINAL_LEVEL);
        assert_eq!(lhs.basis().len(), 14);
    }

    println!("ROW_CACHE_CIPHERTEXTS_VERIFIED={CT_PER_MACRO}");
    println!("ROW_CACHE_CACHED_MS={cached_ms}");
    println!("ROW_CACHE_UNCACHED_MS={uncached_ms}");

    drop(cached);
    drop(uncached);
    drop(cache);

    println!("ROW_CACHE_RELEASED=1");
    println!("ROW_CACHE_EXACT_RLWE_STATUS=PASS");
}
