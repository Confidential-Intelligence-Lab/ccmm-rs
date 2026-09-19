use crate::grafting::{
    rns_relinearize, rns_relinearize_with_ntt, rns_tensor_with_ntt, RnsMultiplicationKey,
    RnsQuadraticCiphertext,
};
use crate::ring::{ModulusChain, RnsNttPlan, RnsPolynomial};
use crate::rlwe::tensor;

use super::{rescale_rns_ckks_to_next, RnsCkksCiphertext};

/// Multiplies, relinearizes, and rescales two RNS CKKS ciphertexts
/// at the same active modulus-chain level.
///
/// The operation preserves the CKKS state invariant:
///
/// ```text
/// level i, scales Delta_l and Delta_r
///     -> multiply/relinearize at level i
///     -> scale Delta_l * Delta_r
///     -> rescale by chain.dropped_modulus(i)
///     -> level i + 1
/// ```
pub fn multiply_relinearize_rescale_rns_ckks(
    lhs: &RnsCkksCiphertext,
    rhs: &RnsCkksCiphertext,
    multiplication_key: &RnsMultiplicationKey,
    chain: &ModulusChain,
) -> RnsCkksCiphertext {
    lhs.assert_matches_chain(chain);
    rhs.assert_matches_chain(chain);

    assert_eq!(
        lhs.level(),
        rhs.level(),
        "RNS CKKS multiplication requires matching levels"
    );

    assert_eq!(
        lhs.basis(),
        rhs.basis(),
        "RNS CKKS multiplication requires matching bases"
    );

    assert_eq!(
        multiplication_key.layout().full_basis(),
        lhs.basis(),
        "RNS multiplication key basis must match active CKKS level"
    );

    assert!(
        chain.has_next_level(lhs.level()),
        "RNS CKKS multiply-rescale requires a next chain level"
    );

    assert_eq!(
        lhs.rlwe().degree(),
        rhs.rlwe().degree(),
        "RNS CKKS ciphertext degrees must match"
    );

    let limb_count = lhs.basis().len();

    let mut c0_residues = Vec::with_capacity(limb_count);

    let mut c1_residues = Vec::with_capacity(limb_count);

    let mut c2_residues = Vec::with_capacity(limb_count);

    for limb_index in 0..limb_count {
        let quadratic = tensor(lhs.rlwe().limb(limb_index), rhs.rlwe().limb(limb_index));

        c0_residues.push(quadratic.c0().clone());

        c1_residues.push(quadratic.c1().clone());

        c2_residues.push(quadratic.c2().clone());
    }

    let quadratic = RnsQuadraticCiphertext::from_rns_polynomials(
        RnsPolynomial::from_residues(c0_residues),
        RnsPolynomial::from_residues(c1_residues),
        RnsPolynomial::from_residues(c2_residues),
    );

    let relinearized = rns_relinearize(&quadratic, multiplication_key);

    let product_state = lhs.state().after_multiply(rhs.state(), chain);

    let product = RnsCkksCiphertext::new(relinearized, product_state, chain);

    rescale_rns_ckks_to_next(&product, chain)
}

/// NTT-backed counterpart of
/// `multiply_relinearize_rescale_rns_ckks`.
///
/// Ciphertext tensor products and relinearization gadget products use
/// per-limb radix-2 negacyclic NTT plans. The CKKS state transition and
/// nearest-rounding RNS rescale are identical to the reference path.
pub fn multiply_relinearize_rescale_rns_ckks_with_ntt(
    lhs: &RnsCkksCiphertext,
    rhs: &RnsCkksCiphertext,
    multiplication_key: &RnsMultiplicationKey,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertext {
    lhs.assert_matches_chain(chain);
    rhs.assert_matches_chain(chain);

    assert_eq!(
        lhs.level(),
        rhs.level(),
        "RNS CKKS multiplication requires matching levels"
    );
    assert_eq!(
        lhs.basis(),
        rhs.basis(),
        "RNS CKKS multiplication requires matching bases"
    );
    assert_eq!(
        multiplication_key.layout().full_basis(),
        lhs.basis(),
        "RNS multiplication key basis must match active CKKS level"
    );
    assert!(
        chain.has_next_level(lhs.level()),
        "RNS CKKS multiply-rescale requires a next chain level"
    );
    assert_eq!(
        lhs.rlwe().degree(),
        rhs.rlwe().degree(),
        "RNS CKKS ciphertext degrees must match"
    );
    assert_eq!(
        plan.degree(),
        lhs.rlwe().degree(),
        "RNS NTT plan degree must match CKKS ciphertext degree"
    );
    assert_eq!(
        plan.moduli(),
        lhs.basis().moduli(),
        "RNS NTT plan basis must match active CKKS level"
    );

    let quadratic = rns_tensor_with_ntt(lhs.rlwe(), rhs.rlwe(), plan);
    let relinearized = rns_relinearize_with_ntt(&quadratic, multiplication_key, plan);

    let product_state = lhs.state().after_multiply(rhs.state(), chain);
    let product = RnsCkksCiphertext::new(relinearized, product_state, chain);

    rescale_rns_ckks_to_next(&product, chain)
}

/// Evaluates a leveled product chain over RNS CKKS ciphertexts.
///
/// `lhs` and `rhs` are multiplied at the initial level using
/// `multiplication_keys[0]`. Each element of `remaining_operands`
/// is then multiplied into the accumulated ciphertext at the next
/// active level using the corresponding subsequent key.
///
/// Therefore:
///
/// ```text
/// multiplication_keys.len()
///     == remaining_operands.len() + 1
/// ```
///
/// Each multiplication performs relinearization followed by one
/// modulus-chain rescale.
pub fn evaluate_rns_ckks_product_chain(
    lhs: &RnsCkksCiphertext,
    rhs: &RnsCkksCiphertext,
    remaining_operands: &[RnsCkksCiphertext],
    multiplication_keys: &[RnsMultiplicationKey],
    chain: &ModulusChain,
) -> RnsCkksCiphertext {
    assert_eq!(
        multiplication_keys.len(),
        remaining_operands.len() + 1,
        "leveled CKKS evaluation requires one multiplication key per depth"
    );

    let mut accumulator =
        multiply_relinearize_rescale_rns_ckks(lhs, rhs, &multiplication_keys[0], chain);

    for (operand, key) in remaining_operands.iter().zip(&multiplication_keys[1..]) {
        accumulator = multiply_relinearize_rescale_rns_ckks(&accumulator, operand, key, chain);
    }

    accumulator
}

/// Multiplies two same-level RNS CKKS ciphertexts, relinearizes with the
/// bounded signed power-of-two multiplication key, and rescales once.
///
/// This is the public R3.1 bounded-base evaluation path. The `research-4096`
/// validated reference uses `base_log=20` and discrete-Gaussian error with
/// sigma 3.19 for ciphertext and evaluation-key generation.
pub fn multiply_relinearize_rescale_rns_ckks_bounded_with_ntt(
    lhs: &RnsCkksCiphertext,
    rhs: &RnsCkksCiphertext,
    multiplication_key: &crate::grafting::BoundedRnsMultiplicationKey,
    chain: &crate::ring::ModulusChain,
    plan: &crate::ring::RnsNttPlan,
) -> RnsCkksCiphertext {
    assert_eq!(
        lhs.level(),
        rhs.level(),
        "bounded CKKS multiplication requires matching levels"
    );
    assert_eq!(
        lhs.basis(),
        rhs.basis(),
        "bounded CKKS multiplication requires matching bases"
    );
    assert_eq!(
        lhs.basis(),
        multiplication_key.layout().full_basis(),
        "bounded multiplication-key basis must match ciphertext basis"
    );
    assert_eq!(
        plan.moduli(),
        lhs.basis().moduli(),
        "NTT plan basis must match bounded CKKS ciphertext basis"
    );

    let quadratic = crate::grafting::rns_tensor_with_ntt(lhs.rlwe(), rhs.rlwe(), plan);

    let relinearized =
        crate::grafting::bounded_rns_relinearize_with_ntt(&quadratic, multiplication_key, plan);

    let product_state = lhs.state().after_multiply(rhs.state(), chain);
    let product = RnsCkksCiphertext::new(relinearized, product_state, chain);

    super::rescale_rns_ckks_to_next(&product, chain)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use num_complex::Complex64;

    use crate::ckks::{CkksCanonicalEmbedding, CkksChainState, CkksSlotEncoder};
    use crate::grafting::{RnsGadgetLayout, RnsMultiplicationKey, RnsRlweCiphertext};
    use crate::ring::{Modulus, ModulusBasis, ModulusChain, Polynomial};
    use crate::rlwe::RlweCiphertext;

    use super::*;

    fn chain() -> ModulusChain {
        ModulusChain::from_top_basis(ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]))
    }

    fn ternary(degree: usize) -> Vec<i8> {
        (0..degree)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 1,
            })
            .collect()
    }

    fn zero_ciphertext(
        chain: &ModulusChain,
        level: usize,
        degree: usize,
        scale: f64,
    ) -> RnsCkksCiphertext {
        let basis = chain.level(level);

        let limbs = basis
            .moduli()
            .iter()
            .copied()
            .map(|modulus| {
                RlweCiphertext::new(
                    Polynomial::zero(modulus, degree),
                    Polynomial::zero(modulus, degree),
                )
            })
            .collect();

        RnsCkksCiphertext::new(
            RnsRlweCiphertext::from_limbs(limbs),
            CkksChainState::new(chain, level, scale),
            chain,
        )
    }

    #[test]
    fn research_4096_gaussian_encryption_multiply_rescale_matches_slots() {
        use crate::ckks::research_profile_4096;
        use crate::grafting::{
            decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, RnsKeygenConfig,
        };
        use crate::rlwe::ErrorDistribution;

        let profile = research_profile_4096();
        let chain = profile.modulus_chain();
        let degree = profile.degree();
        let scale = profile.initial_scale();
        let top_basis = chain.top().clone();
        let top_plan = RnsNttPlan::new(top_basis.moduli().to_vec(), degree);

        let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };

        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 1,
            })
            .collect();

        let mut lhs_slots = vec![Complex64::new(0.0, 0.0); profile.slot_count()];
        let mut rhs_slots = vec![Complex64::new(0.0, 0.0); profile.slot_count()];

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

        let embedding = CkksCanonicalEmbedding::new(degree);

        let encode_rns = |slots: &[Complex64]| {
            let raw = embedding.slots_to_coefficients(slots);

            let signed: Vec<i128> = raw
                .iter()
                .map(|&coefficient| {
                    let scaled = coefficient * scale;
                    assert!(scaled.is_finite(), "scaled CKKS coefficient must be finite");
                    scaled.round() as i128
                })
                .collect();

            let residues = top_basis
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
        };

        let lhs_plaintext = encode_rns(&lhs_slots);
        let rhs_plaintext = encode_rns(&rhs_slots);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(0x29B5_0001);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(0x29B5_0002);
        let mut key_rng = ChaCha20Rng::seed_from_u64(0x29B5_0003);

        let lhs_rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
            &lhs_plaintext,
            2,
            distribution,
            &secret,
            &top_plan,
            &mut lhs_rng,
        );

        let rhs_rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
            &rhs_plaintext,
            2,
            distribution,
            &secret,
            &top_plan,
            &mut rhs_rng,
        );

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

        let product = multiply_relinearize_rescale_rns_ckks_with_ntt(
            &lhs,
            &rhs,
            &multiplication_key,
            &chain,
            &top_plan,
        );

        let result_plan = RnsNttPlan::new(product.rlwe().basis().moduli().to_vec(), degree);

        let decrypted = decrypt_rns_raw_with_ntt(product.rlwe(), &secret, &result_plan);

        let modulus = decrypted.composite_modulus();
        let output_scale = product.state().scale();

        let coefficients: Vec<f64> = decrypted
            .reconstruct_coefficients()
            .into_iter()
            .map(|value| centered(value, modulus) as f64 / output_scale)
            .collect();

        let observed = embedding.coefficients_to_slots(&coefficients);
        let expected = reference_slot_product(&lhs_slots, &rhs_slots);

        let mut max_error = 0.0_f64;
        let mut max_error_slot = 0_usize;

        for (index, (&actual, &reference)) in observed.iter().zip(&expected).enumerate() {
            let error = (actual - reference).norm();

            if error > max_error {
                max_error = error;
                max_error_slot = index;
            }
        }

        println!("GAUSSIAN_ENCRYPTION_PROFILE={}", profile.name());
        println!("GAUSSIAN_ENCRYPTION_SIGMA=3.19");
        println!("GAUSSIAN_ENCRYPTION_INPUT_SCALE={scale:.6}");
        println!("GAUSSIAN_ENCRYPTION_OUTPUT_SCALE={output_scale:.6}");
        println!("GAUSSIAN_ENCRYPTION_MAX_SLOT_ERROR={max_error:.12e}");
        println!("GAUSSIAN_ENCRYPTION_MAX_ERROR_SLOT={max_error_slot}");

        assert!(
            max_error < 1.0e-3,
            "Gaussian-encryption CKKS multiply-rescale error \
             {max_error:.12e} at slot {max_error_slot} \
             exceeds 1e-3"
        );
    }

    #[test]

    fn research_4096_ntt_ckks_multiply_rescale_matches_slots() {
        use crate::ckks::research_profile_4096;
        use crate::grafting::decrypt_rns_quadratic_raw;
        use crate::grafting::{
            decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_ntt_rng, RnsKeygenConfig,
        };
        use crate::ring::{RnsNttPlan, RnsPolynomial};

        let profile = research_profile_4096();
        let chain = profile.modulus_chain();
        let degree = profile.degree();
        let scale = profile.initial_scale();

        assert_eq!(degree, 4096);
        assert_eq!(profile.slot_count(), 2048);
        assert_eq!(chain.len(), 3);

        let top_basis = chain.top().clone();
        let top_plan = RnsNttPlan::new(top_basis.moduli().to_vec(), degree);

        /*
         * Deterministic ternary secret.
         */
        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 1,
            })
            .collect();

        /*
         * Sparse but genuinely complex SIMD vectors.
         *
         * The reference canonical embedding is currently O(N^2), so this
         * remains a smoke test rather than a benchmark.
         */
        let mut lhs_slots = vec![Complex64::new(0.0, 0.0); profile.slot_count()];
        let mut rhs_slots = vec![Complex64::new(0.0, 0.0); profile.slot_count()];

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

        let embedding = CkksCanonicalEmbedding::new(degree);

        /*
         * Canonical CKKS encoding without a temporary single modulus:
         *
         * slots
         *   -> real canonical coefficients
         *   -> multiply by Delta
         *   -> nearest integer
         *   -> project signed integer into every RNS limb
         */
        let encode_rns = |slots: &[Complex64]| {
            let raw = embedding.slots_to_coefficients(slots);

            let signed: Vec<i128> = raw
                .iter()
                .map(|&coefficient| {
                    let scaled = coefficient * scale;

                    assert!(scaled.is_finite(), "scaled CKKS coefficient must be finite");

                    scaled.round() as i128
                })
                .collect();

            let residues = top_basis
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
        };

        let lhs_plaintext = encode_rns(&lhs_slots);
        let rhs_plaintext = encode_rns(&rhs_slots);

        /*
         * R2.9a.3g.2 numerical isolation:
         *
         * 1. floating canonical-embedding round trip;
         * 2. quantized/RNS CKKS encoding round trip.
         *
         * These diagnostics intentionally run before encryption.
         */
        let report_roundtrip = |label: &str, observed: &[Complex64], reference: &[Complex64]| {
            let mut max_error = 0.0_f64;
            let mut max_error_slot = 0_usize;

            for (index, (&actual, &expected)) in observed.iter().zip(reference).enumerate() {
                let error = (actual - expected).norm();

                if error > max_error {
                    max_error = error;
                    max_error_slot = index;
                }
            }

            println!("{label}_MAX_ERROR={max_error:.12e}");
            println!("{label}_MAX_ERROR_SLOT={max_error_slot}");

            const DIAGNOSTIC_SLOTS: [usize; 12] =
                [0, 1, 17, 257, 511, 512, 513, 1023, 1024, 1536, 1537, 2047];

            for index in DIAGNOSTIC_SLOTS {
                let actual = observed[index];
                let expected = reference[index];
                let delta = actual - expected;

                println!(
                    "{label}_SLOT index={index} error={:.12e} \
                         expected_re={:.12e} expected_im={:.12e} \
                         actual_re={:.12e} actual_im={:.12e} \
                         error_re={:.12e} error_im={:.12e}",
                    delta.norm(),
                    expected.re,
                    expected.im,
                    actual.re,
                    actual.im,
                    delta.re,
                    delta.im,
                );
            }
        };

        let lhs_raw_coefficients = embedding.slots_to_coefficients(&lhs_slots);
        let rhs_raw_coefficients = embedding.slots_to_coefficients(&rhs_slots);

        let lhs_float_roundtrip = embedding.coefficients_to_slots(&lhs_raw_coefficients);
        let rhs_float_roundtrip = embedding.coefficients_to_slots(&rhs_raw_coefficients);

        report_roundtrip("LHS_FLOAT_ROUNDTRIP", &lhs_float_roundtrip, &lhs_slots);
        report_roundtrip("RHS_FLOAT_ROUNDTRIP", &rhs_float_roundtrip, &rhs_slots);

        let decode_encoded = |encoded: &RnsPolynomial| {
            let modulus = encoded.composite_modulus();

            let coefficients: Vec<f64> = encoded
                .reconstruct_coefficients()
                .into_iter()
                .map(|value| centered(value, modulus) as f64 / scale)
                .collect();

            embedding.coefficients_to_slots(&coefficients)
        };

        let lhs_quantized_roundtrip = decode_encoded(&lhs_plaintext);
        let rhs_quantized_roundtrip = decode_encoded(&rhs_plaintext);

        report_roundtrip(
            "LHS_QUANTIZED_ROUNDTRIP",
            &lhs_quantized_roundtrip,
            &lhs_slots,
        );
        report_roundtrip(
            "RHS_QUANTIZED_ROUNDTRIP",
            &rhs_quantized_roundtrip,
            &rhs_slots,
        );

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(0x290A_3A01);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(0x290A_3A02);

        /*
         * Zero noise deliberately isolates large-N CKKS arithmetic
         * correctness. Noise characterization belongs to R2.9c.
         */
        let lhs_rlwe =
            encrypt_rns_raw_with_ntt_rng(&lhs_plaintext, 2, 0, &secret, &top_plan, &mut lhs_rng);

        let rhs_rlwe =
            encrypt_rns_raw_with_ntt_rng(&rhs_plaintext, 2, 0, &secret, &top_plan, &mut rhs_rng);

        let lhs = RnsCkksCiphertext::new(lhs_rlwe, CkksChainState::top(&chain, scale), &chain);

        let rhs = RnsCkksCiphertext::new(rhs_rlwe, CkksChainState::top(&chain, scale), &chain);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x290A_3A03);

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

        /*
         * R2.9a.3g.2 stage localization:
         *
         * Isolate realistic-N NTT tensor and relinearization before
         * CKKS rescaling.
         */
        let diagnostic_quadratic = rns_tensor_with_ntt(lhs.rlwe(), rhs.rlwe(), &top_plan);

        let diagnostic_quadratic_dec = decrypt_rns_quadratic_raw(&diagnostic_quadratic, &secret);

        let diagnostic_relinearized =
            rns_relinearize_with_ntt(&diagnostic_quadratic, &multiplication_key, &top_plan);

        let diagnostic_relinearized_dec =
            decrypt_rns_raw_with_ntt(&diagnostic_relinearized, &secret, &top_plan);

        /*
         * Zero encryption/key noise means relinearization should
         * preserve the decrypted degree-2 product exactly.
         */
        let tensor_relinearize_exact = diagnostic_quadratic_dec == diagnostic_relinearized_dec;

        println!("PRE_RESCALE_RELINEARIZATION_EXACT={tensor_relinearize_exact}");

        /*
         * Decode both pre-rescale products at Delta^2.
         *
         * These floating-point errors are diagnostic only. Exact
         * RNS equality above is the stronger relinearization gate.
         */
        let pre_rescale_scale = scale * scale;
        let pre_rescale_modulus = diagnostic_quadratic_dec.composite_modulus();

        let decode_pre_rescale = |polynomial: &RnsPolynomial| {
            let coefficients: Vec<f64> = polynomial
                .reconstruct_coefficients()
                .into_iter()
                .map(|value| centered(value, pre_rescale_modulus) as f64 / pre_rescale_scale)
                .collect();

            embedding.coefficients_to_slots(&coefficients)
        };

        let quadratic_slots = decode_pre_rescale(&diagnostic_quadratic_dec);

        let relinearized_slots = decode_pre_rescale(&diagnostic_relinearized_dec);

        let expected_pre_rescale = reference_slot_product(&lhs_slots, &rhs_slots);

        let mut quadratic_max_error = 0.0_f64;
        let mut quadratic_max_slot = 0_usize;

        let mut relinearized_max_error = 0.0_f64;
        let mut relinearized_max_slot = 0_usize;

        for index in 0..expected_pre_rescale.len() {
            let quadratic_error = (quadratic_slots[index] - expected_pre_rescale[index]).norm();

            if quadratic_error > quadratic_max_error {
                quadratic_max_error = quadratic_error;
                quadratic_max_slot = index;
            }

            let relinearized_error =
                (relinearized_slots[index] - expected_pre_rescale[index]).norm();

            if relinearized_error > relinearized_max_error {
                relinearized_max_error = relinearized_error;
                relinearized_max_slot = index;
            }
        }

        println!("PRE_RESCALE_TENSOR_MAX_SLOT_ERROR={quadratic_max_error:.12e}");
        println!("PRE_RESCALE_TENSOR_MAX_ERROR_SLOT={quadratic_max_slot}");
        println!("PRE_RESCALE_RELINEARIZED_MAX_SLOT_ERROR={relinearized_max_error:.12e}");
        println!("PRE_RESCALE_RELINEARIZED_MAX_ERROR_SLOT={relinearized_max_slot}");

        /*
         * R2.9a.3g.2 rescale isolation.
         *
         * Build an RLWE ciphertext carrying the exact decrypted
         * pre-rescale product entirely in b, with a = 0.
         *
         * Running this through the production CKKS rescale isolates
         * polynomial rescaling from component-wise RLWE/secret effects.
         */
        let carrier_limbs = diagnostic_quadratic_dec
            .residues()
            .iter()
            .map(|message| {
                let modulus = message.modulus();

                RlweCiphertext::new(message.clone(), Polynomial::zero(modulus, degree))
            })
            .collect();

        let carrier_rlwe = RnsRlweCiphertext::from_limbs(carrier_limbs);

        let carrier_state = lhs.state().after_multiply(rhs.state(), &chain);

        let carrier = RnsCkksCiphertext::new(carrier_rlwe, carrier_state, &chain);

        let carrier_rescaled = rescale_rns_ckks_to_next(&carrier, &chain);

        /*
         * Because a = 0, the rescaled message is simply the b
         * component. Reassemble it as an RNS polynomial.
         */
        let carrier_rescaled_polynomial = RnsPolynomial::from_residues(
            carrier_rescaled
                .rlwe()
                .limbs()
                .iter()
                .map(|limb| limb.b().clone())
                .collect(),
        );

        /*
         * Independent exact oracle:
         *
         *   1. reconstruct the pre-rescale polynomial modulo Q;
         *   2. center each coefficient;
         *   3. perform signed nearest division by q_drop;
         *   4. project the result into the surviving RNS basis.
         *
         * research-4096 Q fits in i128/u128, so no BigInt dependency
         * is necessary here.
         */
        let source_q = diagnostic_quadratic_dec.composite_modulus();

        let source_q_i128 =
            i128::try_from(source_q).expect("research-4096 source modulus must fit i128");

        let diagnostic_dropped = chain
            .dropped_modulus(0)
            .expect("research-4096 top level must rescale");

        let divisor = i128::from(diagnostic_dropped.value());

        let oracle_signed: Vec<i128> = diagnostic_quadratic_dec
            .reconstruct_coefficients()
            .into_iter()
            .map(|value| {
                let canonical = i128::try_from(value).expect("coefficient must fit i128");

                let centered_value = if canonical > source_q_i128 / 2 {
                    canonical - source_q_i128
                } else {
                    canonical
                };

                if centered_value >= 0 {
                    (centered_value + divisor / 2) / divisor
                } else {
                    -((-centered_value + divisor / 2) / divisor)
                }
            })
            .collect();

        let oracle_residues = chain
            .level(1)
            .moduli()
            .iter()
            .copied()
            .map(|modulus| {
                let q = i128::from(modulus.value());

                Polynomial::new(
                    modulus,
                    oracle_signed
                        .iter()
                        .map(|&value| {
                            let residue = ((value % q) + q) % q;
                            residue as u64
                        })
                        .collect(),
                )
            })
            .collect();

        let oracle_rescaled = RnsPolynomial::from_residues(oracle_residues);

        let carrier_exact = carrier_rescaled_polynomial == oracle_rescaled;

        println!("PLAINTEXT_CARRIER_RESCALE_EXACT={carrier_exact}");

        /*
         * Decode the plaintext-carrier result using the same output
         * scale as the production rescale.
         */
        let carrier_modulus = carrier_rescaled_polynomial.composite_modulus();

        let carrier_coefficients: Vec<f64> = carrier_rescaled_polynomial
            .reconstruct_coefficients()
            .into_iter()
            .map(|value| centered(value, carrier_modulus) as f64 / carrier_rescaled.scale())
            .collect();

        let carrier_slots = embedding.coefficients_to_slots(&carrier_coefficients);

        let expected_carrier = reference_slot_product(&lhs_slots, &rhs_slots);

        let mut carrier_max_error = 0.0_f64;
        let mut carrier_max_slot = 0_usize;

        for (index, (&observed, &reference)) in
            carrier_slots.iter().zip(&expected_carrier).enumerate()
        {
            let error = (observed - reference).norm();

            if error > carrier_max_error {
                carrier_max_error = error;
                carrier_max_slot = index;
            }
        }

        println!("PLAINTEXT_CARRIER_MAX_SLOT_ERROR={carrier_max_error:.12e}");
        println!("PLAINTEXT_CARRIER_MAX_ERROR_SLOT={carrier_max_slot}");

        let result = multiply_relinearize_rescale_rns_ckks_with_ntt(
            &lhs,
            &rhs,
            &multiplication_key,
            &chain,
            &top_plan,
        );

        assert_eq!(result.level(), 1);
        assert_eq!(result.basis(), chain.level(1));

        let dropped = chain
            .dropped_modulus(0)
            .expect("research-4096 top level must rescale");

        let expected_scale = scale * scale / dropped.value() as f64;

        assert!(
            (result.scale() - expected_scale).abs() <= expected_scale * 1.0e-12,
            "unexpected large-N CKKS output scale: actual={}, expected={}",
            result.scale(),
            expected_scale,
        );

        /*
         * The result lives at level 1, so use an NTT plan for the
         * reduced active basis.
         */
        let next_plan = RnsNttPlan::new(chain.level(1).moduli().to_vec(), degree);

        let decrypted = decrypt_rns_raw_with_ntt(result.rlwe(), &secret, &next_plan);

        /*
         * research-4096 has a 109-bit top Q. After one rescale the
         * remaining basis is smaller still, so exact reconstruction is
         * representable by the current u128 CRT implementation.
         */
        let active_modulus = decrypted.composite_modulus();

        let decoded_coefficients: Vec<f64> = decrypted
            .reconstruct_coefficients()
            .into_iter()
            .map(|value| centered(value, active_modulus) as f64 / result.scale())
            .collect();

        let actual = embedding.coefficients_to_slots(&decoded_coefficients);

        let expected = reference_slot_product(&lhs_slots, &rhs_slots);

        let mut max_error = 0.0_f64;
        let mut max_error_slot = 0_usize;
        let mut errors = Vec::with_capacity(actual.len());
        let mut diagnostics = Vec::with_capacity(actual.len());
        let mut squared_error_sum = 0.0_f64;

        for (index, (&observed, &reference)) in actual.iter().zip(&expected).enumerate() {
            let delta = observed - reference;
            let error = delta.norm();

            errors.push(error);
            diagnostics.push((index, error, observed, reference, delta));
            squared_error_sum += error * error;

            if error > max_error {
                max_error = error;
                max_error_slot = index;
            }
        }

        let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
        let rms_error = (squared_error_sum / errors.len() as f64).sqrt();

        let mut sorted_errors = errors.clone();
        sorted_errors.sort_by(f64::total_cmp);

        let median_error = if sorted_errors.len() % 2 == 0 {
            let upper = sorted_errors.len() / 2;
            (sorted_errors[upper - 1] + sorted_errors[upper]) / 2.0
        } else {
            sorted_errors[sorted_errors.len() / 2]
        };

        let percentile = |percent: usize| {
            let index = (sorted_errors.len() - 1) * percent / 100;
            sorted_errors[index]
        };

        diagnostics.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));

        println!("PROFILE={}", profile.name());
        println!("DEGREE={degree}");
        println!("SLOTS={}", profile.slot_count());
        println!("INPUT_SCALE={scale:.6}");
        println!("OUTPUT_SCALE={:.6}", result.scale());
        println!("DROPPED_MODULUS={}", dropped.value());
        println!("MAX_SLOT_ERROR={max_error:.12e}");
        println!("MAX_ERROR_SLOT={max_error_slot}");
        println!("MEAN_SLOT_ERROR={mean_error:.12e}");
        println!("RMS_SLOT_ERROR={rms_error:.12e}");
        println!("MEDIAN_SLOT_ERROR={median_error:.12e}");
        println!("P90_SLOT_ERROR={:.12e}", percentile(90));
        println!("P99_SLOT_ERROR={:.12e}", percentile(99));

        println!("TOP_ERROR_SLOTS_BEGIN");
        for &(index, error, observed, reference, delta) in diagnostics.iter().take(16) {
            println!(
                "SLOT_DIAG index={index} error={error:.12e} \
                 expected_re={:.12e} expected_im={:.12e} \
                 actual_re={:.12e} actual_im={:.12e} \
                 error_re={:.12e} error_im={:.12e}",
                reference.re, reference.im, observed.re, observed.im, delta.re, delta.im,
            );
        }
        println!("TOP_ERROR_SLOTS_END");

        const PROBE_SLOTS: [usize; 13] = [
            0, 1, 2, 255, 256, 511, 512, 513, 1023, 1024, 1535, 1536, 2047,
        ];

        println!("PROBE_SLOTS_BEGIN");
        for index in PROBE_SLOTS {
            let observed = actual[index];
            let reference = expected[index];
            let delta = observed - reference;
            let error = delta.norm();

            println!(
                "SLOT_PROBE index={index} error={error:.12e} \
                 expected_re={:.12e} expected_im={:.12e} \
                 actual_re={:.12e} actual_im={:.12e} \
                 error_re={:.12e} error_im={:.12e}",
                reference.re, reference.im, observed.re, observed.im, delta.re, delta.im,
            );
        }
        println!("PROBE_SLOTS_END");

        /*
         * This is intentionally a smoke-test bound, not the eventual
         * R2.9c precision claim. The latter will characterize error
         * statistically across values, scales, depths, and profiles.
         */
        let tolerance = 1.0e-3;

        assert!(
            max_error <= tolerance,
            "research-4096 CKKS multiply error exceeded smoke bound: \
             max_error={max_error:e}, \
             slot={max_error_slot}, \
             tolerance={tolerance:e}"
        );

        println!("RESEARCH_4096_CKKS_NUMERICAL_STATUS=PASS");
    }

    #[test]
    fn ntt_rns_ckks_multiply_matches_reference_exactly() {
        use crate::grafting::{encrypt_rns_raw_with_ntt_rng, RnsKeygenConfig};
        use crate::ring::{RnsNttPlan, RnsPolynomial};

        let chain = chain();
        let degree = 8;
        let basis = chain.top().clone();
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
        let secret = ternary(degree);

        let lhs_message = RnsPolynomial::from_coefficients(
            basis.moduli().to_vec(),
            &[1_u128, 2, 3, 4, 5, 6, 7, 8],
        );

        let rhs_message = RnsPolynomial::from_coefficients(
            basis.moduli().to_vec(),
            &[8_u128, 7, 6, 5, 4, 3, 2, 1],
        );

        for seed in 0_u64..32 {
            let mut lhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xB810);
            let mut rhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xB820);

            let lhs_rlwe =
                encrypt_rns_raw_with_ntt_rng(&lhs_message, 2, 0, &secret, &plan, &mut lhs_rng);

            let rhs_rlwe =
                encrypt_rns_raw_with_ntt_rng(&rhs_message, 2, 0, &secret, &plan, &mut rhs_rng);

            let scale = 256.0;

            let lhs = RnsCkksCiphertext::new(lhs_rlwe, CkksChainState::top(&chain, scale), &chain);

            let rhs = RnsCkksCiphertext::new(rhs_rlwe, CkksChainState::top(&chain, scale), &chain);

            let mut key_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xB830);

            let key = RnsMultiplicationKey::generate_with_ntt_rng(
                RnsKeygenConfig {
                    degree,
                    plaintext_modulus: 2,
                    noise_bound: 0,
                    layout: RnsGadgetLayout::new(basis.clone(), vec![1, 2]),
                    plan: &plan,
                },
                &secret,
                &mut key_rng,
            );

            let reference = multiply_relinearize_rescale_rns_ckks(&lhs, &rhs, &key, &chain);

            let optimized =
                multiply_relinearize_rescale_rns_ckks_with_ntt(&lhs, &rhs, &key, &chain, &plan);

            assert_eq!(
                optimized, reference,
                "NTT CKKS multiply/rescale diverged for seed {seed}"
            );

            assert_eq!(optimized.level(), 1);
            assert_eq!(optimized.basis(), chain.level(1));

            let expected_scale = scale * scale
                / chain
                    .dropped_modulus(0)
                    .expect("top level must have a dropped modulus")
                    .value() as f64;

            assert_eq!(optimized.scale(), expected_scale);
        }
    }

    #[test]
    fn multiply_relinearize_rescale_advances_level_and_scale() {
        let chain = chain();

        let degree = 8;

        let lhs = zero_ciphertext(&chain, 0, degree, 256.0);

        let rhs = zero_ciphertext(&chain, 0, degree, 512.0);

        let layout = RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]);

        let mut rng = ChaCha20Rng::seed_from_u64(0xD400);

        let key = RnsMultiplicationKey::generate_with_rng(
            degree,
            2,
            0,
            &ternary(degree),
            layout,
            &mut rng,
        );

        let result = multiply_relinearize_rescale_rns_ckks(&lhs, &rhs, &key, &chain);

        assert_eq!(result.level(), 1);

        assert_eq!(result.basis(), chain.level(1));

        let expected_scale = 256.0 * 512.0 / 65_537.0;

        assert!((result.scale() - expected_scale).abs() < 1.0e-12);

        for limb in result.rlwe().limbs() {
            assert!(limb.b().coefficients().iter().all(|&value| value == 0));

            assert!(limb.a().coefficients().iter().all(|&value| value == 0));
        }
    }

    #[test]
    fn multiply_relinearize_rescale_works_from_middle_level() {
        let chain = chain();

        let degree = 8;

        let lhs = zero_ciphertext(&chain, 1, degree, 128.0);

        let rhs = zero_ciphertext(&chain, 1, degree, 256.0);

        let layout = RnsGadgetLayout::new(chain.level(1).clone(), vec![1, 1]);

        let mut rng = ChaCha20Rng::seed_from_u64(0xD401);

        let key = RnsMultiplicationKey::generate_with_rng(
            degree,
            2,
            0,
            &ternary(degree),
            layout,
            &mut rng,
        );

        let result = multiply_relinearize_rescale_rns_ckks(&lhs, &rhs, &key, &chain);

        assert_eq!(result.level(), 2);

        assert_eq!(result.basis(), chain.level(2));

        let expected_scale = 128.0 * 256.0 / 40_961.0;

        assert!((result.scale() - expected_scale).abs() < 1.0e-12);
    }

    #[test]
    #[should_panic(expected = "RNS multiplication key basis must match active CKKS level")]
    fn rejects_multiplication_key_for_wrong_level() {
        let chain = chain();

        let degree = 8;

        let lhs = zero_ciphertext(&chain, 1, degree, 1.0);

        let rhs = zero_ciphertext(&chain, 1, degree, 1.0);

        let wrong_layout = RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]);

        let mut rng = ChaCha20Rng::seed_from_u64(0xD402);

        let wrong_key = RnsMultiplicationKey::generate_with_rng(
            degree,
            2,
            0,
            &ternary(degree),
            wrong_layout,
            &mut rng,
        );

        let _ = multiply_relinearize_rescale_rns_ckks(&lhs, &rhs, &wrong_key, &chain);
    }

    fn centered(value: u128, modulus: u128) -> i128 {
        let value = value as i128;

        let modulus = modulus as i128;

        if value > modulus / 2 {
            value - modulus
        } else {
            value
        }
    }

    fn encode_coefficients(values: &[f64], scale: f64, composite_modulus: u128) -> Vec<u128> {
        values
            .iter()
            .map(|&value| {
                let scaled = (value * scale).round() as i128;

                let modulus = composite_modulus as i128;

                ((scaled % modulus) + modulus) as u128 % composite_modulus
            })
            .collect()
    }

    fn encrypt_rns_values(
        chain: &ModulusChain,
        level: usize,
        values: &[f64],
        scale: f64,
        secret: &[i8],
        seed: u64,
    ) -> RnsCkksCiphertext {
        use crate::rlwe::{encrypt_raw_with_rng, RlweParameters, SecretKey};

        let basis = chain.level(level);

        let degree = values.len();

        let encoded = encode_coefficients(values, scale, basis.composite_modulus());

        let limbs = basis
            .moduli()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, modulus)| {
                let secret_polynomial = Polynomial::new(
                    modulus,
                    secret
                        .iter()
                        .map(|&value| match value {
                            -1 => modulus.value() - 1,
                            0 => 0,
                            1 => 1,
                            _ => {
                                panic!("secret must be ternary")
                            }
                        })
                        .collect(),
                );

                let secret_key = SecretKey::from_polynomial(secret_polynomial);

                let message = Polynomial::new(
                    modulus,
                    encoded
                        .iter()
                        .map(|&value| (value % u128::from(modulus.value())) as u64)
                        .collect(),
                );

                let params = RlweParameters::new(degree, modulus, 2, 0);

                let mut rng = ChaCha20Rng::seed_from_u64(seed ^ (index as u64 * 0x9E37));

                encrypt_raw_with_rng(params, &secret_key, &message, &mut rng)
            })
            .collect();

        RnsCkksCiphertext::new(
            RnsRlweCiphertext::from_limbs(limbs),
            CkksChainState::new(chain, level, scale),
            chain,
        )
    }

    fn decrypt_decode_rns(ciphertext: &RnsCkksCiphertext, secret: &[i8]) -> Vec<f64> {
        use crate::grafting::decrypt_rns_raw;

        let polynomial = decrypt_rns_raw(ciphertext.rlwe(), secret);

        let modulus = polynomial.composite_modulus();

        polynomial
            .reconstruct_coefficients()
            .into_iter()
            .map(|value| centered(value, modulus) as f64 / ciphertext.scale())
            .collect()
    }

    fn encrypt_rns_slots(
        chain: &ModulusChain,
        level: usize,
        slots: &[Complex64],
        scale: f64,
        secret: &[i8],
        seed: u64,
    ) -> RnsCkksCiphertext {
        use crate::rlwe::{encrypt_raw_with_rng, RlweParameters, SecretKey};

        let basis = chain.level(level);

        let degree = slots.len() * 2;

        /*
         * Use the canonical slot encoder to perform the inverse
         * embedding and quantization. The temporary modulus only
         * provides a canonical container for the quantized signed
         * coefficients; the resulting centered integers are then
         * projected into every active RNS limb.
         *
         * The selected validation vectors are intentionally small,
         * so no temporary-modulus wrap occurs.
         */
        let temporary_modulus = Modulus::new(2_147_483_647);

        let encoder = CkksSlotEncoder::new(degree, temporary_modulus, scale);

        let encoded = encoder.encode_slots(slots);

        let signed_coefficients: Vec<i128> = encoded
            .coefficients()
            .iter()
            .map(|&value| {
                let value = i128::from(value);

                let modulus = i128::from(temporary_modulus.value());

                if value > modulus / 2 {
                    value - modulus
                } else {
                    value
                }
            })
            .collect();

        let limbs = basis
            .moduli()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, modulus)| {
                let secret_polynomial = Polynomial::new(
                    modulus,
                    secret
                        .iter()
                        .map(|&value| match value {
                            -1 => modulus.value() - 1,
                            0 => 0,
                            1 => 1,
                            _ => {
                                panic!("secret must be ternary")
                            }
                        })
                        .collect(),
                );

                let secret_key = SecretKey::from_polynomial(secret_polynomial);

                let q = i128::from(modulus.value());

                let message = Polynomial::new(
                    modulus,
                    signed_coefficients
                        .iter()
                        .map(|&value| ((value % q + q) % q) as u64)
                        .collect(),
                );

                let params = RlweParameters::new(degree, modulus, 2, 0);

                let mut rng = ChaCha20Rng::seed_from_u64(seed ^ (index as u64 * 0x9E37));

                encrypt_raw_with_rng(params, &secret_key, &message, &mut rng)
            })
            .collect();

        RnsCkksCiphertext::new(
            RnsRlweCiphertext::from_limbs(limbs),
            CkksChainState::new(chain, level, scale),
            chain,
        )
    }

    fn decrypt_decode_rns_slots(ciphertext: &RnsCkksCiphertext, secret: &[i8]) -> Vec<Complex64> {
        let coefficients = decrypt_decode_rns(ciphertext, secret);

        CkksCanonicalEmbedding::new(coefficients.len()).coefficients_to_slots(&coefficients)
    }

    fn reference_slot_product(lhs: &[Complex64], rhs: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(lhs.len(), rhs.len(),);

        lhs.iter().zip(rhs).map(|(&lhs, &rhs)| lhs * rhs).collect()
    }

    fn reference_negacyclic_product(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
        assert_eq!(lhs.len(), rhs.len());

        let degree = lhs.len();

        let mut output = vec![0.0; degree];

        for i in 0..degree {
            for j in 0..degree {
                let product = lhs[i] * rhs[j];

                if i + j < degree {
                    output[i + j] += product;
                } else {
                    output[i + j - degree] -= product;
                }
            }
        }

        output
    }

    #[test]
    fn encrypted_rns_ckks_product_decodes_correctly() {
        let chain = chain();

        let degree = 8;

        let scale = 65_537.0;

        let secret = ternary(degree);

        let lhs_values = [0.25, -0.5, 0.75, 0.125, -0.25, 0.5, 0.0, 0.125];

        let rhs_values = [-0.25, 0.25, 0.5, -0.125, 0.25, 0.0, -0.25, 0.5];

        let lhs = encrypt_rns_values(&chain, 0, &lhs_values, scale, &secret, 0xD410);

        let rhs = encrypt_rns_values(&chain, 0, &rhs_values, scale, &secret, 0xD411);

        let layout = RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]);

        let mut eval_rng = ChaCha20Rng::seed_from_u64(0xD412);

        let key =
            RnsMultiplicationKey::generate_with_rng(degree, 2, 0, &secret, layout, &mut eval_rng);

        let result = multiply_relinearize_rescale_rns_ckks(&lhs, &rhs, &key, &chain);

        assert_eq!(result.level(), 1);

        assert_eq!(result.basis(), chain.level(1));

        let actual = decrypt_decode_rns(&result, &secret);

        let expected = reference_negacyclic_product(&lhs_values, &rhs_values);

        let tolerance = 0.03;

        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "coefficient {index}: \
                 actual={actual}, \
                 expected={expected}, \
                 error={}, \
                 tolerance={tolerance}",
                (actual - expected).abs()
            );
        }
    }

    #[test]
    fn encrypted_rns_ckks_product_campaign_is_stable() {
        let chain = chain();

        let degree = 8;

        let scale = 65_537.0;

        let secret = ternary(degree);

        for seed in 0_u64..32 {
            let lhs_values: Vec<f64> = (0..degree)
                .map(|index| (((seed + 3 * index as u64) % 9) as i64 - 4) as f64 / 8.0)
                .collect();

            let rhs_values: Vec<f64> = (0..degree)
                .map(|index| (((2 * seed + 5 * index as u64 + 1) % 9) as i64 - 4) as f64 / 8.0)
                .collect();

            let lhs = encrypt_rns_values(&chain, 0, &lhs_values, scale, &secret, seed ^ 0xD420);

            let rhs = encrypt_rns_values(&chain, 0, &rhs_values, scale, &secret, seed ^ 0xD421);

            let layout = RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]);

            let mut eval_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xD422);

            let key = RnsMultiplicationKey::generate_with_rng(
                degree,
                2,
                0,
                &secret,
                layout,
                &mut eval_rng,
            );

            let result = multiply_relinearize_rescale_rns_ckks(&lhs, &rhs, &key, &chain);

            let actual = decrypt_decode_rns(&result, &secret);

            let expected = reference_negacyclic_product(&lhs_values, &rhs_values);

            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 0.05,
                    "seed={seed}, \
                     coefficient={index}, \
                     actual={actual}, \
                     expected={expected}, \
                     error={}",
                    (actual - expected).abs()
                );
            }
        }
    }

    #[test]
    fn diagnose_encrypted_rns_ckks_pipeline() {
        use crate::grafting::{
            decrypt_rns_quadratic_raw, decrypt_rns_raw, rns_relinearize, RnsQuadraticCiphertext,
        };
        use crate::ring::RnsPolynomial;
        use crate::rlwe::tensor;

        let chain = chain();
        let degree = 8;
        let scale = 65_537.0;
        let secret = ternary(degree);

        let lhs_values = [0.25, -0.5, 0.75, 0.125, -0.25, 0.5, 0.0, 0.125];

        let rhs_values = [-0.25, 0.25, 0.5, -0.125, 0.25, 0.0, -0.25, 0.5];

        let lhs = encrypt_rns_values(&chain, 0, &lhs_values, scale, &secret, 0xD510);

        let rhs = encrypt_rns_values(&chain, 0, &rhs_values, scale, &secret, 0xD511);

        /*
         * Gate 1: encryption/decryption must preserve the
         * encoded coefficient vectors exactly (noise = 0).
         */
        let lhs_dec = decrypt_rns_raw(lhs.rlwe(), &secret);

        let rhs_dec = decrypt_rns_raw(rhs.rlwe(), &secret);

        let lhs_encoded =
            encode_coefficients(&lhs_values, scale, chain.level(0).composite_modulus());

        let rhs_encoded =
            encode_coefficients(&rhs_values, scale, chain.level(0).composite_modulus());

        assert_eq!(
            lhs_dec.reconstruct_coefficients(),
            lhs_encoded,
            "GATE1_LHS_ENCRYPT_DECRYPT"
        );

        assert_eq!(
            rhs_dec.reconstruct_coefficients(),
            rhs_encoded,
            "GATE1_RHS_ENCRYPT_DECRYPT"
        );

        /*
         * Build the exact limbwise quadratic ciphertext manually,
         * before relinearization or rescale.
         */
        let mut c0 = Vec::new();
        let mut c1 = Vec::new();
        let mut c2 = Vec::new();

        for limb_index in 0..lhs.basis().len() {
            let q = tensor(lhs.rlwe().limb(limb_index), rhs.rlwe().limb(limb_index));

            c0.push(q.c0().clone());
            c1.push(q.c1().clone());
            c2.push(q.c2().clone());
        }

        let quadratic = RnsQuadraticCiphertext::from_rns_polynomials(
            RnsPolynomial::from_residues(c0),
            RnsPolynomial::from_residues(c1),
            RnsPolynomial::from_residues(c2),
        );

        /*
         * Gate 2: decrypted quadratic product must equal the
         * polynomial product of the decrypted operands modulo Q.
         */
        let quadratic_dec = decrypt_rns_quadratic_raw(&quadratic, &secret);

        let expected_product = RnsPolynomial::from_residues(
            lhs_dec
                .residues()
                .iter()
                .zip(rhs_dec.residues())
                .map(|(lhs, rhs)| lhs.negacyclic_mul(rhs))
                .collect(),
        );

        assert_eq!(quadratic_dec, expected_product, "GATE2_TENSOR_PRODUCT");

        let layout = RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]);

        let mut eval_rng = ChaCha20Rng::seed_from_u64(0xD512);

        let key =
            RnsMultiplicationKey::generate_with_rng(degree, 2, 0, &secret, layout, &mut eval_rng);

        /*
         * Gate 3: zero-noise relinearization must preserve the
         * decrypted quadratic polynomial exactly.
         */
        let relinearized = rns_relinearize(&quadratic, &key);

        let relinearized_dec = decrypt_rns_raw(&relinearized, &secret);

        assert_eq!(relinearized_dec, quadratic_dec, "GATE3_RELINEARIZATION");

        /*
         * Gate 4: compare ciphertext rescale against direct
         * rescaling of the decrypted polynomial.
         */
        let product_state = lhs.state().after_multiply(rhs.state(), &chain);

        let wrapped = RnsCkksCiphertext::new(relinearized, product_state, &chain);

        let rescaled = rescale_rns_ckks_to_next(&wrapped, &chain);

        let actual = decrypt_rns_raw(rescaled.rlwe(), &secret);

        println!(
            "DIAG_PRE_RESCALE={:?}",
            quadratic_dec.reconstruct_coefficients()
        );

        println!("DIAG_POST_RESCALE={:?}", actual.reconstruct_coefficients());

        println!("DIAG_SCALE={}", rescaled.scale());

        println!("DIAG_DECODED={:?}", decrypt_decode_rns(&rescaled, &secret,));

        println!(
            "DIAG_EXPECTED={:?}",
            reference_negacyclic_product(&lhs_values, &rhs_values,)
        );
    }

    #[test]
    fn encrypted_rns_ckks_two_depths_decode_correctly() {
        let chain = chain();

        let degree = 8;

        let secret = ternary(degree);

        /*
         * First level:
         *
         * Delta_0 = 65537, which is also the first dropped
         * modulus. Therefore
         *
         *     Delta_0^2 / 65537 = 65537.
         */
        let level0_scale = 65_537.0;

        /*
         * Keep the plaintext values deliberately small so the
         * depth-2 scaled plaintext remains comfortably inside the
         * centered modulus interval at every level.
         */
        let lhs_values = [0.125, -0.0625, 0.03125, 0.0, 0.0, 0.0, 0.0, 0.0];

        let rhs_values = [0.0625, 0.03125, -0.0625, 0.0, 0.0, 0.0, 0.0, 0.0];

        let third_values = [0.125, -0.0625, 0.03125, 0.0, 0.0, 0.0, 0.0, 0.0];

        let lhs = encrypt_rns_values(&chain, 0, &lhs_values, level0_scale, &secret, 0xD600);

        let rhs = encrypt_rns_values(&chain, 0, &rhs_values, level0_scale, &secret, 0xD601);

        let level0_layout = RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]);

        let mut level0_eval_rng = ChaCha20Rng::seed_from_u64(0xD602);

        let level0_key = RnsMultiplicationKey::generate_with_rng(
            degree,
            2,
            0,
            &secret,
            level0_layout,
            &mut level0_eval_rng,
        );

        let level1 = multiply_relinearize_rescale_rns_ckks(&lhs, &rhs, &level0_key, &chain);

        assert_eq!(level1.level(), 1);

        assert_eq!(level1.basis(), chain.level(1));

        assert!((level1.scale() - 65_537.0).abs() < 1.0e-9);

        /*
         * Second level.
         *
         * Choose the third operand scale equal to the second
         * dropped modulus:
         *
         *     65537 * 40961 / 40961 = 65537.
         */
        let level1_operand_scale = 40_961.0;

        let third = encrypt_rns_values(
            &chain,
            1,
            &third_values,
            level1_operand_scale,
            &secret,
            0xD603,
        );

        let level1_layout = RnsGadgetLayout::new(chain.level(1).clone(), vec![1, 1]);

        let mut level1_eval_rng = ChaCha20Rng::seed_from_u64(0xD604);

        let level1_key = RnsMultiplicationKey::generate_with_rng(
            degree,
            2,
            0,
            &secret,
            level1_layout,
            &mut level1_eval_rng,
        );

        let level2 = multiply_relinearize_rescale_rns_ckks(&level1, &third, &level1_key, &chain);

        assert_eq!(level2.level(), 2);

        assert_eq!(level2.basis(), chain.level(2));

        assert_eq!(level2.basis(), chain.bottom());

        assert!(
            (level2.scale() - 65_537.0).abs() < 1.0e-9,
            "unexpected final scale: {}",
            level2.scale()
        );

        /*
         * Independent floating-point depth-2 reference:
         *
         *     (lhs * rhs) * third
         *
         * using negacyclic polynomial multiplication.
         */
        let reference_level1 = reference_negacyclic_product(&lhs_values, &rhs_values);

        let expected = reference_negacyclic_product(&reference_level1, &third_values);

        let actual = decrypt_decode_rns(&level2, &secret);

        let tolerance = 0.002;

        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "depth-2 coefficient {index}: \
                 actual={actual}, \
                 expected={expected}, \
                 error={}, \
                 tolerance={tolerance}",
                (actual - expected).abs()
            );
        }
    }

    fn depth_chain() -> ModulusChain {
        ModulusChain::from_top_basis(ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
            Modulus::new(114_689),
            Modulus::new(147_457),
        ]))
    }

    fn multiplication_key_for_level(
        chain: &ModulusChain,
        level: usize,
        degree: usize,
        secret: &[i8],
        seed: u64,
    ) -> RnsMultiplicationKey {
        let basis = chain.level(level);

        /*
         * A single block spanning the complete active basis is
         * sufficient for this depth-generalization campaign.
         */
        let layout = RnsGadgetLayout::new(basis.clone(), vec![basis.len()]);

        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        RnsMultiplicationKey::generate_with_rng(degree, 2, 0, secret, layout, &mut rng)
    }

    #[test]
    fn depth_parametric_encrypted_ckks_campaign() {
        let chain = depth_chain();

        let degree = 8;

        let secret = ternary(degree);

        assert_eq!(chain.max_level(), 4);

        /*
         * Small sparse coefficient vectors keep the depth-4
         * plaintext comfortably inside the final centered modulus.
         */
        let lhs_values = [0.125, -0.0625, 0.03125, 0.0, 0.0, 0.0, 0.0, 0.0];

        let rhs_values = [0.0625, 0.03125, -0.0625, 0.0, 0.0, 0.0, 0.0, 0.0];

        let continuation_values = [0.125, -0.0625, 0.03125, 0.0, 0.0, 0.0, 0.0, 0.0];

        /*
         * The first dropped modulus is also the initial scale:
         *
         *     Delta^2 / q_drop = Delta.
         */
        let stable_scale = chain.dropped_modulus(0).unwrap().value() as f64;

        for depth in 1_usize..=4 {
            let lhs = encrypt_rns_values(
                &chain,
                0,
                &lhs_values,
                stable_scale,
                &secret,
                0xE000 + depth as u64,
            );

            let rhs = encrypt_rns_values(
                &chain,
                0,
                &rhs_values,
                stable_scale,
                &secret,
                0xE100 + depth as u64,
            );

            let mut operands = Vec::with_capacity(depth.saturating_sub(1));

            /*
             * After the first multiplication, the accumulator is
             * at level 1. At each later level choose the new
             * operand scale equal to that level's dropped modulus:
             *
             *     stable_scale * q_drop / q_drop
             *       = stable_scale.
             */
            for level in 1..depth {
                let operand_scale = chain.dropped_modulus(level).unwrap().value() as f64;

                operands.push(encrypt_rns_values(
                    &chain,
                    level,
                    &continuation_values,
                    operand_scale,
                    &secret,
                    0xE200 + (depth as u64 * 16) + level as u64,
                ));
            }

            let keys: Vec<_> = (0..depth)
                .map(|level| {
                    multiplication_key_for_level(
                        &chain,
                        level,
                        degree,
                        &secret,
                        0xE300 + (depth as u64 * 16) + level as u64,
                    )
                })
                .collect();

            let result = evaluate_rns_ckks_product_chain(&lhs, &rhs, &operands, &keys, &chain);

            assert_eq!(
                result.level(),
                depth,
                "depth {depth} ended at wrong chain level"
            );

            assert_eq!(
                result.basis(),
                chain.level(depth),
                "depth {depth} ended on wrong RNS basis"
            );

            assert!(
                (result.scale() - stable_scale).abs() < 1.0e-9,
                "depth {depth}: unexpected scale {}",
                result.scale()
            );

            let mut expected = reference_negacyclic_product(&lhs_values, &rhs_values);

            for _ in 1..depth {
                expected = reference_negacyclic_product(&expected, &continuation_values);
            }

            let actual = decrypt_decode_rns(&result, &secret);

            /*
             * Errors remain very small for these validation
             * parameters, but allow accumulation with depth.
             */
            let tolerance = 0.0005 * depth as f64;

            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "depth={depth}, \
                     coefficient={index}, \
                     actual={actual}, \
                     expected={expected}, \
                     error={}, \
                     tolerance={tolerance}",
                    (actual - expected).abs()
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "RNS CKKS multiply-rescale requires a next chain level")]
    fn depth_parametric_execution_rejects_chain_exhaustion() {
        let chain = depth_chain();

        let degree = 8;

        let secret = ternary(degree);

        let level = chain.max_level();

        let lhs = zero_ciphertext(&chain, level, degree, 1.0);

        let rhs = zero_ciphertext(&chain, level, degree, 1.0);

        let key = multiplication_key_for_level(&chain, level, degree, &secret, 0xEFFF);

        let _ = multiply_relinearize_rescale_rns_ckks(&lhs, &rhs, &key, &chain);
    }

    #[test]
    fn encrypted_ckks_simd_multiplication_matches_slotwise_reference() {
        let chain = chain();

        let degree = 8;

        let scale = 65_537.0;

        let secret = ternary(degree);

        let lhs_slots = [
            Complex64::new(0.25, 0.125),
            Complex64::new(-0.5, 0.25),
            Complex64::new(0.75, -0.125),
            Complex64::new(0.125, 0.5),
        ];

        let rhs_slots = [
            Complex64::new(-0.25, 0.5),
            Complex64::new(0.25, -0.125),
            Complex64::new(0.5, 0.25),
            Complex64::new(-0.125, 0.25),
        ];

        let lhs = encrypt_rns_slots(&chain, 0, &lhs_slots, scale, &secret, 0xF100);

        let rhs = encrypt_rns_slots(&chain, 0, &rhs_slots, scale, &secret, 0xF101);

        /*
         * First verify that canonical slot semantics survive
         * encode -> encrypt -> decrypt -> decode before testing
         * homomorphic multiplication.
         */
        let lhs_roundtrip = decrypt_decode_rns_slots(&lhs, &secret);

        let rhs_roundtrip = decrypt_decode_rns_slots(&rhs, &secret);

        let input_tolerance = degree as f64 / (2.0 * scale) + 1.0e-10;

        for (index, (&actual, &expected)) in lhs_roundtrip.iter().zip(&lhs_slots).enumerate() {
            assert!(
                (actual - expected).norm() <= input_tolerance,
                "lhs slot {index}: \
                 actual={actual:?}, \
                 expected={expected:?}"
            );
        }

        for (index, (&actual, &expected)) in rhs_roundtrip.iter().zip(&rhs_slots).enumerate() {
            assert!(
                (actual - expected).norm() <= input_tolerance,
                "rhs slot {index}: \
                 actual={actual:?}, \
                 expected={expected:?}"
            );
        }

        let layout = RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]);

        let mut rng = ChaCha20Rng::seed_from_u64(0xF102);

        let multiplication_key =
            RnsMultiplicationKey::generate_with_rng(degree, 2, 0, &secret, layout, &mut rng);

        let result = multiply_relinearize_rescale_rns_ckks(&lhs, &rhs, &multiplication_key, &chain);

        assert_eq!(result.level(), 1);

        assert_eq!(result.basis(), chain.level(1));

        assert!(
            (result.scale() - scale).abs() < 1.0e-9,
            "unexpected SIMD output scale: {}",
            result.scale()
        );

        let actual = decrypt_decode_rns_slots(&result, &secret);

        let expected = reference_slot_product(&lhs_slots, &rhs_slots);

        /*
         * This bound includes:
         *
         * - input canonical-embedding quantization;
         * - polynomial multiplication of quantized inputs;
         * - nearest CKKS rescaling.
         *
         * It remains intentionally conservative for the small
         * correctness parameters used here.
         */
        let tolerance = 0.001;

        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let error = (actual - expected).norm();

            assert!(
                error <= tolerance,
                "SIMD slot {index}: \
                 actual={actual:?}, \
                 expected={expected:?}, \
                 error={error}, \
                 tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn encrypted_ckks_simd_two_depths_match_slotwise_reference() {
        let chain = chain();

        let degree = 8;

        let secret = ternary(degree);

        /*
         * Keep the post-rescale accumulator scale stable:
         *
         * first multiplication:
         *
         *     65537^2 / 65537 = 65537
         *
         * second multiplication:
         *
         *     65537 * 40961 / 40961 = 65537
         */
        let level0_scale = 65_537.0;

        let level1_operand_scale = 40_961.0;

        let lhs_slots = [
            Complex64::new(0.25, 0.125),
            Complex64::new(-0.5, 0.25),
            Complex64::new(0.75, -0.125),
            Complex64::new(0.125, 0.5),
        ];

        let rhs_slots = [
            Complex64::new(-0.25, 0.5),
            Complex64::new(0.25, -0.125),
            Complex64::new(0.5, 0.25),
            Complex64::new(-0.125, 0.25),
        ];

        let third_slots = [
            Complex64::new(0.5, -0.25),
            Complex64::new(-0.25, 0.5),
            Complex64::new(0.125, 0.25),
            Complex64::new(0.25, -0.125),
        ];

        let lhs = encrypt_rns_slots(&chain, 0, &lhs_slots, level0_scale, &secret, 0xF200);

        let rhs = encrypt_rns_slots(&chain, 0, &rhs_slots, level0_scale, &secret, 0xF201);

        let level0_layout = RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]);

        let mut level0_rng = ChaCha20Rng::seed_from_u64(0xF202);

        let level0_key = RnsMultiplicationKey::generate_with_rng(
            degree,
            2,
            0,
            &secret,
            level0_layout,
            &mut level0_rng,
        );

        let level1 = multiply_relinearize_rescale_rns_ckks(&lhs, &rhs, &level0_key, &chain);

        assert_eq!(level1.level(), 1);

        assert!((level1.scale() - 65_537.0).abs() < 1.0e-9);

        let third = encrypt_rns_slots(
            &chain,
            1,
            &third_slots,
            level1_operand_scale,
            &secret,
            0xF203,
        );

        let level1_layout = RnsGadgetLayout::new(chain.level(1).clone(), vec![1, 1]);

        let mut level1_rng = ChaCha20Rng::seed_from_u64(0xF204);

        let level1_key = RnsMultiplicationKey::generate_with_rng(
            degree,
            2,
            0,
            &secret,
            level1_layout,
            &mut level1_rng,
        );

        let level2 = multiply_relinearize_rescale_rns_ckks(&level1, &third, &level1_key, &chain);

        assert_eq!(level2.level(), 2);

        assert_eq!(level2.basis(), chain.level(2));

        assert!(
            (level2.scale() - 65_537.0).abs() < 1.0e-9,
            "unexpected final SIMD scale: {}",
            level2.scale()
        );

        let actual = decrypt_decode_rns_slots(&level2, &secret);

        let expected_level1 = reference_slot_product(&lhs_slots, &rhs_slots);

        let expected = reference_slot_product(&expected_level1, &third_slots);

        /*
         * Conservative depth-2 tolerance including:
         *
         * - first canonical-embedding quantization;
         * - first encrypted multiply/rescale;
         * - level-1 operand quantization;
         * - second multiply/rescale.
         */
        let tolerance = 0.003;

        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let error = (actual - expected).norm();

            assert!(
                error <= tolerance,
                "depth-2 SIMD slot {index}: \
                 actual={actual:?}, \
                 expected={expected:?}, \
                 error={error}, \
                 tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn research_4096_bounded_gaussian_multiply_rescale_slot_sweep() {
        use std::time::Instant;

        use crate::ckks::research_profile_4096;
        use crate::grafting::decrypt_rns_raw_with_ntt;

        use crate::grafting::{
            bounded_rns_relinearize_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng,
            rns_tensor_with_ntt, BoundedGadgetLayout, BoundedRnsKeygenConfig,
            BoundedRnsMultiplicationKey,
        };
        use crate::rlwe::ErrorDistribution;

        let profile = research_profile_4096();
        let chain = profile.modulus_chain();
        let degree = profile.degree();
        let scale = profile.initial_scale();
        let basis = chain.top().clone();
        let top_plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
        let embedding = CkksCanonicalEmbedding::new(degree);

        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 1,
            })
            .collect();

        let slot_count = degree / 2;

        let lhs_slots: Vec<Complex64> = (0..slot_count)
            .map(|index| {
                let x = index as f64;
                Complex64::new(0.10 + 0.00002 * x, -0.08 + 0.00001 * x)
            })
            .collect();

        let rhs_slots: Vec<Complex64> = (0..slot_count)
            .map(|index| {
                let x = index as f64;
                Complex64::new(-0.15 + 0.000015 * x, 0.12 - 0.000008 * x)
            })
            .collect();

        let encode_rns = |slots: &[Complex64]| {
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
                            .map(|&value| {
                                let signed = (value * scale).round() as i128;

                                signed.rem_euclid(q) as u64
                            })
                            .collect(),
                    )
                })
                .collect();

            RnsPolynomial::from_residues(residues)
        };

        let lhs_plaintext = encode_rns(&lhs_slots);
        let rhs_plaintext = encode_rns(&rhs_slots);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(0x31B3_0001);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(0x31B3_0002);

        let lhs_rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
            &lhs_plaintext,
            2,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            &secret,
            &top_plan,
            &mut lhs_rng,
        );

        let rhs_rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
            &rhs_plaintext,
            2,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            &secret,
            &top_plan,
            &mut rhs_rng,
        );

        let lhs = RnsCkksCiphertext::new(lhs_rlwe, CkksChainState::top(&chain, scale), &chain);

        let rhs = RnsCkksCiphertext::new(rhs_rlwe, CkksChainState::top(&chain, scale), &chain);

        let quadratic = rns_tensor_with_ntt(lhs.rlwe(), rhs.rlwe(), &top_plan);

        let expected = reference_slot_product(&lhs_slots, &rhs_slots);

        let tolerance = 1.0e-3_f64;

        println!("R3_1B3_PROFILE={}", profile.name());
        println!("R3_1B3_RING_DEGREE={degree}");
        println!("R3_1B3_SLOT_COUNT={slot_count}");
        println!("R3_1B3_INPUT_SCALE={scale:.17e}");
        println!("R3_1B3_TOLERANCE={tolerance:.12e}");

        for base_log in [4_u32, 8, 12, 16, 20] {
            let layout = BoundedGadgetLayout::new(basis.clone(), base_log);

            let key_start = Instant::now();

            let mut key_rng = ChaCha20Rng::seed_from_u64(0x31B3_1000 + u64::from(base_log));

            let multiplication_key =
                BoundedRnsMultiplicationKey::generate_with_distribution_ntt_rng(
                    BoundedRnsKeygenConfig {
                        plaintext_modulus: 2,
                        layout: layout.clone(),
                        plan: &top_plan,
                    },
                    &secret,
                    ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                    &mut key_rng,
                );

            let keygen_us = key_start.elapsed().as_secs_f64() * 1.0e6;

            let relin_start = Instant::now();

            let relinearized =
                bounded_rns_relinearize_with_ntt(&quadratic, &multiplication_key, &top_plan);

            let relinearize_us = relin_start.elapsed().as_secs_f64() * 1.0e6;

            let product_state = lhs.state().after_multiply(rhs.state(), &chain);

            let product = RnsCkksCiphertext::new(relinearized, product_state, &chain);

            let rescaled = rescale_rns_ckks_to_next(&product, &chain);

            let result_plan = RnsNttPlan::new(rescaled.rlwe().basis().moduli().to_vec(), degree);

            let decrypted = decrypt_rns_raw_with_ntt(rescaled.rlwe(), &secret, &result_plan);

            let modulus = decrypted.composite_modulus();
            let output_scale = rescaled.scale();

            let decoded_coefficients: Vec<f64> = decrypted
                .reconstruct_coefficients()
                .into_iter()
                .map(|value| centered(value, modulus) as f64 / output_scale)
                .collect();

            let observed = embedding.coefficients_to_slots(&decoded_coefficients);

            let mut max_error = 0.0_f64;
            let mut sum_error = 0.0_f64;
            let mut sum_squared_error = 0.0_f64;

            for (&actual, &reference) in observed.iter().zip(&expected) {
                let error = (actual - reference).norm();

                max_error = max_error.max(error);
                sum_error += error;
                sum_squared_error += error * error;
            }

            let mean_error = sum_error / observed.len() as f64;

            let rms_error = (sum_squared_error / observed.len() as f64).sqrt();

            let status = if max_error <= tolerance {
                "PASS"
            } else {
                "FAIL"
            };

            println!(
                "R3_1B3_BASE_LOG={base_log} \
                 BASE={} \
                 DIGITS={} \
                 KEYGEN_US={keygen_us:.3} \
                 RELINEARIZE_US={relinearize_us:.3} \
                 OUTPUT_SCALE={output_scale:.17e} \
                 MAX_SLOT_ERROR={max_error:.12e} \
                 MEAN_SLOT_ERROR={mean_error:.12e} \
                 RMS_SLOT_ERROR={rms_error:.12e} \
                 STATUS={status}",
                layout.base(),
                layout.digit_count(),
            );

            assert!(
                max_error <= tolerance,
                "bounded Gaussian CKKS multiplication exceeded \
                 tolerance for base_log={base_log}: \
                 error={max_error:e}"
            );
        }

        println!("R3_1B3_BOUNDED_GAUSSIAN_CKKS_SWEEP=PASS");
    }

    #[test]
    fn research_4096_bounded_gaussian_radix_statistical_characterization() {
        use std::time::Instant;

        use crate::ckks::research_profile_4096;
        use crate::grafting::{
            bounded_rns_relinearize_with_ntt, decrypt_rns_raw_with_ntt,
            encrypt_rns_raw_with_distribution_ntt_rng, rns_tensor_with_ntt, BoundedGadgetLayout,
            BoundedRnsKeygenConfig, BoundedRnsMultiplicationKey,
        };
        use crate::rlwe::ErrorDistribution;

        #[derive(Debug, Clone, Copy)]
        struct TrialResult {
            keygen_us: f64,
            relinearize_us: f64,
            max_slot_error: f64,
            mean_slot_error: f64,
            rms_slot_error: f64,
        }

        fn min(values: &[f64]) -> f64 {
            values.iter().copied().fold(f64::INFINITY, f64::min)
        }

        fn max(values: &[f64]) -> f64 {
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        }

        fn mean(values: &[f64]) -> f64 {
            values.iter().sum::<f64>() / values.len() as f64
        }

        fn median(values: &[f64]) -> f64 {
            let mut values = values.to_vec();
            values.sort_by(f64::total_cmp);

            let middle = values.len() / 2;

            if values.len() % 2 == 0 {
                (values[middle - 1] + values[middle]) / 2.0
            } else {
                values[middle]
            }
        }

        let profile = research_profile_4096();
        let chain = profile.modulus_chain();
        let degree = profile.degree();

        /*
         * Keep the statistical campaign on the profile-defined initial scale.
         * This is 2^35 for research-4096.
         */
        let scale = profile.initial_scale();

        let basis = chain.top().clone();
        let top_plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
        let embedding = CkksCanonicalEmbedding::new(degree);

        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 1,
            })
            .collect();

        let slot_count = degree / 2;

        let lhs_slots: Vec<Complex64> = (0..slot_count)
            .map(|index| {
                let x = index as f64;
                Complex64::new(0.10 + 0.00002 * x, -0.08 + 0.00001 * x)
            })
            .collect();

        let rhs_slots: Vec<Complex64> = (0..slot_count)
            .map(|index| {
                let x = index as f64;
                Complex64::new(-0.15 + 0.000015 * x, 0.12 - 0.000008 * x)
            })
            .collect();

        let encode_rns = |slots: &[Complex64]| {
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
                            .map(|&value| {
                                let signed = (value * scale).round() as i128;
                                signed.rem_euclid(q) as u64
                            })
                            .collect(),
                    )
                })
                .collect();

            RnsPolynomial::from_residues(residues)
        };

        let lhs_plaintext = encode_rns(&lhs_slots);
        let rhs_plaintext = encode_rns(&rhs_slots);

        let expected = reference_slot_product(&lhs_slots, &rhs_slots);

        let tolerance = 1.0e-3_f64;
        let trial_count = 10_usize;
        let base_logs = [8_u32, 12, 16, 20];

        println!("R3_1B4_PROFILE={}", profile.name());
        println!("R3_1B4_RING_DEGREE={degree}");
        println!("R3_1B4_SLOT_COUNT={slot_count}");
        println!("R3_1B4_TRIALS={trial_count}");
        println!("R3_1B4_INPUT_SCALE={scale:.17e}");
        println!("R3_1B4_TOLERANCE={tolerance:.12e}");

        for base_log in base_logs {
            let layout = BoundedGadgetLayout::new(basis.clone(), base_log);
            let mut results = Vec::with_capacity(trial_count);
            let mut pass_count = 0_usize;

            for trial in 0..trial_count {
                let trial_u64 = u64::try_from(trial).expect("trial index must fit u64");

                let mut lhs_rng = ChaCha20Rng::seed_from_u64(
                    0x31B4_0000 ^ (trial_u64 << 8) ^ u64::from(base_log),
                );

                let mut rhs_rng = ChaCha20Rng::seed_from_u64(
                    0x31B4_1000 ^ (trial_u64 << 8) ^ u64::from(base_log),
                );

                let lhs_rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                    &lhs_plaintext,
                    2,
                    ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                    &secret,
                    &top_plan,
                    &mut lhs_rng,
                );

                let rhs_rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                    &rhs_plaintext,
                    2,
                    ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                    &secret,
                    &top_plan,
                    &mut rhs_rng,
                );

                let lhs =
                    RnsCkksCiphertext::new(lhs_rlwe, CkksChainState::top(&chain, scale), &chain);

                let rhs =
                    RnsCkksCiphertext::new(rhs_rlwe, CkksChainState::top(&chain, scale), &chain);

                let quadratic = rns_tensor_with_ntt(lhs.rlwe(), rhs.rlwe(), &top_plan);

                let mut key_rng = ChaCha20Rng::seed_from_u64(
                    0x31B4_2000 ^ (trial_u64 << 8) ^ u64::from(base_log),
                );

                let keygen_start = Instant::now();

                let multiplication_key =
                    BoundedRnsMultiplicationKey::generate_with_distribution_ntt_rng(
                        BoundedRnsKeygenConfig {
                            plaintext_modulus: 2,
                            layout: layout.clone(),
                            plan: &top_plan,
                        },
                        &secret,
                        ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                        &mut key_rng,
                    );

                let keygen_us = keygen_start.elapsed().as_secs_f64() * 1.0e6;

                let relinearize_start = Instant::now();

                let relinearized =
                    bounded_rns_relinearize_with_ntt(&quadratic, &multiplication_key, &top_plan);

                let relinearize_us = relinearize_start.elapsed().as_secs_f64() * 1.0e6;

                let product_state = lhs.state().after_multiply(rhs.state(), &chain);

                let product = RnsCkksCiphertext::new(relinearized, product_state, &chain);

                let rescaled = rescale_rns_ckks_to_next(&product, &chain);

                let result_plan =
                    RnsNttPlan::new(rescaled.rlwe().basis().moduli().to_vec(), degree);

                let decrypted = decrypt_rns_raw_with_ntt(rescaled.rlwe(), &secret, &result_plan);

                let modulus = decrypted.composite_modulus();
                let output_scale = rescaled.scale();

                let decoded_coefficients: Vec<f64> = decrypted
                    .reconstruct_coefficients()
                    .into_iter()
                    .map(|value| centered(value, modulus) as f64 / output_scale)
                    .collect();

                let observed = embedding.coefficients_to_slots(&decoded_coefficients);

                let mut max_slot_error = 0.0_f64;
                let mut sum_error = 0.0_f64;
                let mut sum_squared_error = 0.0_f64;

                for (&actual, &reference) in observed.iter().zip(&expected) {
                    let error = (actual - reference).norm();

                    max_slot_error = max_slot_error.max(error);

                    sum_error += error;
                    sum_squared_error += error * error;
                }

                let mean_slot_error = sum_error / observed.len() as f64;

                let rms_slot_error = (sum_squared_error / observed.len() as f64).sqrt();

                let passed = max_slot_error <= tolerance;

                if passed {
                    pass_count += 1;
                }

                println!(
                    "R3_1B4_TRIAL={} \
                     BASE_LOG={} \
                     BASE={} \
                     DIGITS={} \
                     KEYGEN_US={:.3} \
                     RELINEARIZE_US={:.3} \
                     MAX_SLOT_ERROR={:.12e} \
                     MEAN_SLOT_ERROR={:.12e} \
                     RMS_SLOT_ERROR={:.12e} \
                     STATUS={}",
                    trial,
                    base_log,
                    layout.base(),
                    layout.digit_count(),
                    keygen_us,
                    relinearize_us,
                    max_slot_error,
                    mean_slot_error,
                    rms_slot_error,
                    if passed { "PASS" } else { "FAIL" },
                );

                results.push(TrialResult {
                    keygen_us,
                    relinearize_us,
                    max_slot_error,
                    mean_slot_error,
                    rms_slot_error,
                });
            }

            let keygen: Vec<f64> = results.iter().map(|r| r.keygen_us).collect();

            let relinearize: Vec<f64> = results.iter().map(|r| r.relinearize_us).collect();

            let max_slot_error: Vec<f64> = results.iter().map(|r| r.max_slot_error).collect();

            let mean_slot_error: Vec<f64> = results.iter().map(|r| r.mean_slot_error).collect();

            let rms_slot_error: Vec<f64> = results.iter().map(|r| r.rms_slot_error).collect();

            println!(
                "R3_1B4_SUMMARY_BASE_LOG={} \
                 BASE={} \
                 DIGITS={} \
                 PASS_COUNT={} \
                 TRIALS={} \
                 KEYGEN_US_MIN={:.3} \
                 KEYGEN_US_MEAN={:.3} \
                 KEYGEN_US_MEDIAN={:.3} \
                 KEYGEN_US_MAX={:.3} \
                 RELINEARIZE_US_MIN={:.3} \
                 RELINEARIZE_US_MEAN={:.3} \
                 RELINEARIZE_US_MEDIAN={:.3} \
                 RELINEARIZE_US_MAX={:.3} \
                 MAX_SLOT_ERROR_MIN={:.12e} \
                 MAX_SLOT_ERROR_MEAN={:.12e} \
                 MAX_SLOT_ERROR_MEDIAN={:.12e} \
                 MAX_SLOT_ERROR_MAX={:.12e} \
                 MEAN_SLOT_ERROR_MIN={:.12e} \
                 MEAN_SLOT_ERROR_MEAN={:.12e} \
                 MEAN_SLOT_ERROR_MEDIAN={:.12e} \
                 MEAN_SLOT_ERROR_MAX={:.12e} \
                 RMS_SLOT_ERROR_MIN={:.12e} \
                 RMS_SLOT_ERROR_MEAN={:.12e} \
                 RMS_SLOT_ERROR_MEDIAN={:.12e} \
                 RMS_SLOT_ERROR_MAX={:.12e}",
                base_log,
                layout.base(),
                layout.digit_count(),
                pass_count,
                trial_count,
                min(&keygen),
                mean(&keygen),
                median(&keygen),
                max(&keygen),
                min(&relinearize),
                mean(&relinearize),
                median(&relinearize),
                max(&relinearize),
                min(&max_slot_error),
                mean(&max_slot_error),
                median(&max_slot_error),
                max(&max_slot_error),
                min(&mean_slot_error),
                mean(&mean_slot_error),
                median(&mean_slot_error),
                max(&mean_slot_error),
                min(&rms_slot_error),
                mean(&rms_slot_error),
                median(&rms_slot_error),
                max(&rms_slot_error),
            );

            assert_eq!(
                pass_count, trial_count,
                "bounded Gaussian CKKS statistical campaign failed for base_log={base_log}"
            );
        }

        println!("R3_1B4_STATISTICAL_CHARACTERIZATION=PASS");
    }
}
