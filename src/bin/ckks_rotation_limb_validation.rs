use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_65536_for_levels, rotate_left_rns_ckks_with_ntt,
    rotate_left_rns_ckks_with_prepared_ntt, rotation_exponent_left, CkksCanonicalEmbedding,
    CkksChainState, PreparedRnsGaloisKey, RnsCkksCiphertext, RnsGaloisKey,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout,
    RnsKeygenConfig,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const SIGMA: f64 = 3.19;
const DEGREE: usize = 65536;
const ROTATION: usize = 32;

fn block_sizes(limbs: usize) -> Vec<usize> {
    vec![1; limbs]
}

fn rotate_left_clear(values: &[Complex64], amount: usize) -> Vec<Complex64> {
    let amount = amount % values.len();

    (0..values.len())
        .map(|index| values[(index + amount) % values.len()])
        .collect()
}

fn error_metrics(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());

    let mut error_sq = 0.0_f64;
    let mut reference_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (actual, expected) in actual.iter().zip(expected) {
        let error = *actual - *expected;
        let abs = error.norm();

        error_sq += error.norm_sqr();
        reference_sq += expected.norm_sqr();
        max_abs = max_abs.max(abs);
    }

    (
        error_sq.sqrt() / reference_sq.sqrt().max(f64::MIN_POSITIVE),
        max_abs,
    )
}

fn decrypt_slots(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let basis = ciphertext.basis();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), ciphertext.rlwe().degree());

    let decrypted = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);

    let modulus = composite_modulus_big(basis);

    let coefficients = reconstruct_coefficients_big(&decrypted);

    let scale = ciphertext.state().scale();

    let decoded_coefficients = coefficients
        .iter()
        .map(|value| {
            centered_representative_big(value, &modulus)
                .to_f64()
                .expect("centered CKKS coefficient must fit f64")
                / scale
        })
        .collect::<Vec<_>>();

    embedding.coefficients_to_slots(&decoded_coefficients)
}

fn main() {
    println!("ROTATION_LIMB_VALIDATION_BEGIN");
    println!("ROTATION_LIMB_DEGREE={DEGREE}");
    println!("ROTATION_LIMB_ROTATION={ROTATION}");

    for limbs in [14usize, 15, 16, 17, 18, 19, 20, 22] {
        /*
         * research_profile_65536_for_levels() returns
         * required_levels + terminal_guard_limbs limbs.
         */
        let profile = research_profile_65536_for_levels(limbs, 0);

        let chain = profile.modulus_chain();
        let basis = chain.top().clone();
        let scale = profile.initial_scale();
        let embedding = CkksCanonicalEmbedding::new(DEGREE);

        assert_eq!(basis.len(), limbs);

        let slot_count = embedding.slot_count();

        let input = (0..slot_count)
            .map(|index| {
                Complex64::new(
                    ((index % 257) as f64) / 257.0,
                    ((index % 131) as f64) / 131.0,
                )
            })
            .collect::<Vec<_>>();

        let expected = rotate_left_clear(&input, ROTATION);

        let plaintext = encode_rns(&input, &embedding, &basis, scale);

        let plan = RnsNttPlan::new(basis.moduli().to_vec(), DEGREE);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x524f_5441_5445_0001 ^ limbs as u64);

        let mut secret: Vec<i8> = (0..DEGREE)
            .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
            .collect();

        if secret.iter().all(|&value| value == 0) {
            secret[0] = 1;
        }

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x524f_5441_5445_0002 ^ limbs as u64);

        let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
            &plaintext,
            2,
            ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
            &secret,
            &plan,
            &mut encryption_rng,
        );

        let ciphertext = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

        let layout = RnsGadgetLayout::new(basis.clone(), block_sizes(limbs));

        let exponent = rotation_exponent_left(DEGREE, ROTATION);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x524f_5441_5445_0003 ^ limbs as u64);

        println!(
            "ROTATION_LIMB_CASE_BEGIN limbs={} exponent={}",
            limbs, exponent
        );

        let key = RnsGaloisKey::generate_with_distribution_ntt_rng(
            RnsKeygenConfig {
                degree: DEGREE,
                plaintext_modulus: 2,
                noise_bound: 0,
                layout,
                plan: &plan,
            },
            &secret,
            exponent,
            ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
            &mut key_rng,
        );

        /*
         * Ordinary NTT-backed rotation.
         */
        let ordinary = rotate_left_rns_ckks_with_ntt(&ciphertext, ROTATION, &key, &chain, &plan);

        let ordinary_decoded = decrypt_slots(&ordinary, &secret, &embedding);

        let (ordinary_rel_l2, ordinary_max_abs) = error_metrics(&ordinary_decoded, &expected);

        println!(
            "ROTATION_LIMB_ORDINARY limbs={} rel_l2={:.12e} max_abs={:.12e}",
            limbs, ordinary_rel_l2, ordinary_max_abs,
        );

        /*
         * Prepared NTT-backed rotation using exactly the same
         * canonical Galois key.
         */
        let prepared_key = PreparedRnsGaloisKey::new(&key, &plan);

        let prepared = rotate_left_rns_ckks_with_prepared_ntt(
            &ciphertext,
            ROTATION,
            &prepared_key,
            &chain,
            &plan,
        );

        let prepared_decoded = decrypt_slots(&prepared, &secret, &embedding);

        let (prepared_rel_l2, prepared_max_abs) = error_metrics(&prepared_decoded, &expected);

        println!(
            "ROTATION_LIMB_PREPARED limbs={} rel_l2={:.12e} max_abs={:.12e}",
            limbs, prepared_rel_l2, prepared_max_abs,
        );

        println!("ROTATION_LIMB_CASE_END limbs={limbs}");
    }

    println!("ROTATION_LIMB_VALIDATION_END");
}
