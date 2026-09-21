use std::time::Instant;

use ccmm_rs::application_support::ckks::{decode_scalar, encode_rns};
use ccmm_rs::ckks::{
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
};
use ccmm_rs::grafting::{
    encrypt_rns_raw_with_distribution_ntt_rng, rns_key_switch_with_ntt, RnsGadgetLayout,
    RnsKeySwitchKey, RnsKeygenConfig,
};
use ccmm_rs::ring::RnsNttPlan;
use ccmm_rs::rlwe::ErrorDistribution;

use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn ternary_secret(degree: usize, seed: u64) -> Vec<i8> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut secret: Vec<i8> = (0..degree).map(|_| rng.gen_range(-1_i8..=1_i8)).collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    secret
}

fn main() {
    const SIGMA: f64 = 3.19;
    const TOLERANCE: f64 = 2.0e-3;

    let profile = research_profile_4096();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let basis = chain.top().clone();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    let alice_secret = ternary_secret(degree, 0x360E_0001);
    let bob_secret = ternary_secret(degree, 0x360E_0002);
    assert_ne!(alice_secret, bob_secret);

    let value = 0.375_f64;
    let slots = vec![Complex64::new(value, 0.0); profile.slot_count()];
    let plaintext = encode_rns(&slots, &embedding, &basis, scale);

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x360E_1000);
    let encryption_start = Instant::now();
    let alice_raw = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        distribution,
        &alice_secret,
        &plan,
        &mut encryption_rng,
    );
    let encryption_us = encryption_start.elapsed().as_secs_f64() * 1.0e6;

    let alice_ciphertext =
        RnsCkksCiphertext::new(alice_raw, CkksChainState::top(&chain, scale), &chain);

    let (alice_observed, alice_imag) = decode_scalar(&alice_ciphertext, &alice_secret, &embedding);
    let alice_error = (alice_observed - value).abs();

    // Three-limb research-4096 basis: preserve the existing tested generic
    // key-switch decomposition [1, 2].
    assert_eq!(
        basis.len(),
        3,
        "proxy re-encryption example expects the research-4096 three-limb basis"
    );
    let layout = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);

    let mut rekey_rng = ChaCha20Rng::seed_from_u64(0x360E_2000);
    let rekey_start = Instant::now();
    // The current generic CRT-block RNS key-switch decomposition is used here
    // in its zero-noise correctness mode. A Gaussian re-encryption key at this
    // parameter point produces unacceptable error amplification; a
    // security-bearing PRE service therefore requires a bounded generic
    // source-to-target key-switch construction.
    let rekey = RnsKeySwitchKey::generate_with_ntt_rng(
        RnsKeygenConfig {
            degree,
            plaintext_modulus: 2,
            noise_bound: 0,
            layout,
            plan: &plan,
        },
        &alice_secret,
        &bob_secret,
        &mut rekey_rng,
    );
    let rekey_us = rekey_start.elapsed().as_secs_f64() * 1.0e6;

    // Proxy execution path: ciphertext + re-encryption key + public plan only.
    // Neither Alice's nor Bob's secret key is used below.
    let proxy_start = Instant::now();
    let bob_raw = rns_key_switch_with_ntt(alice_ciphertext.rlwe(), &rekey, &plan);
    let proxy_us = proxy_start.elapsed().as_secs_f64() * 1.0e6;

    let bob_ciphertext = RnsCkksCiphertext::new(bob_raw, alice_ciphertext.state().clone(), &chain);

    let (bob_observed, bob_imag) = decode_scalar(&bob_ciphertext, &bob_secret, &embedding);
    let bob_error = (bob_observed - value).abs();
    let preservation_error = (bob_observed - alice_observed).abs();
    let max_imag = alice_imag.max(bob_imag);

    println!("R3_6E_PROXY_REENCRYPTION_VERSION=1");
    println!("APPLICATION=proxy-re-encryption");
    println!("SERVICE=proxy-re-encryption");
    println!("CONSTRUCTION=RNS_KEY_SWITCH");
    println!("DIRECTION=Alice->Bob");
    println!("PROFILE={}", profile.name());
    println!("PARAMETER_SECURITY_VALIDATED=false");
    println!("RING_DEGREE={degree}");
    println!("SLOT_COUNT={}", profile.slot_count());

    println!("TRUSTED_SETUP_REQUIRES_SOURCE_SECRET=true");
    println!("TRUSTED_SETUP_REQUIRES_TARGET_SECRET=true");

    println!("PROXY_INPUTS=ciphertext,re-encryption-key,public-ntt-plan");
    println!("PROXY_REQUIRES_SOURCE_SECRET=false");
    println!("PROXY_REQUIRES_TARGET_SECRET=false");
    println!("PROXY_DECRYPTION_PERFORMED=false");

    println!("SOURCE_LEVEL={}", alice_ciphertext.level());
    println!("TARGET_LEVEL={}", bob_ciphertext.level());
    println!(
        "LEVEL_PRESERVED={}",
        alice_ciphertext.level() == bob_ciphertext.level()
    );

    println!("CIPHERTEXT_ERROR_SIGMA={SIGMA}");
    println!("REENCRYPTION_KEY_NOISE_BOUND=0");
    println!("REENCRYPTION_KEY_SECURITY_BEARING=false");
    println!("HARDENED_SAMPLER=false");

    println!("PRE_SECURITY_CLAIM=false");
    println!("CCA_SECURITY_CLAIM=false");
    println!("COLLUSION_RESISTANCE_CLAIM=false");
    println!("UNIDIRECTIONAL_SECURITY_CLAIM=false");

    println!("ENCRYPTION_US={encryption_us:.3}");
    println!("REENCRYPTION_KEYGEN_US={rekey_us:.3}");
    println!("PROXY_REENCRYPTION_US={proxy_us:.3}");

    println!("SOURCE_EXPECTED={value:.12e}");
    println!("SOURCE_OBSERVED={alice_observed:.12e}");
    println!("SOURCE_ABS_ERROR={alice_error:.12e}");
    println!("TARGET_EXPECTED={value:.12e}");
    println!("TARGET_OBSERVED={bob_observed:.12e}");
    println!("TARGET_ABS_ERROR={bob_error:.12e}");
    println!("PLAINTEXT_PRESERVATION_ERROR={preservation_error:.12e}");
    println!("MAX_IMAGINARY_RESIDUAL={max_imag:.12e}");
    println!("TOLERANCE={TOLERANCE:.12e}");

    assert!(alice_error <= TOLERANCE);
    assert!(bob_error <= TOLERANCE);
    assert!(preservation_error <= TOLERANCE);
    assert_eq!(alice_ciphertext.level(), bob_ciphertext.level());

    println!("SOURCE_CIPHERTEXT_DECRYPTS_UNDER_ALICE=PASS");
    println!("REENCRYPTED_CIPHERTEXT_DECRYPTS_UNDER_BOB=PASS");
    println!("PLAINTEXT_PRESERVATION=PASS");
    println!("R3_6E_PROXY_REENCRYPTION_STATUS=PASS");
}
