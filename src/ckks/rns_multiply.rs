use crate::grafting::{rns_relinearize, RnsMultiplicationKey, RnsQuadraticCiphertext};
use crate::ring::{ModulusChain, RnsPolynomial};
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

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ckks::CkksChainState;
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
}
