use rand::{CryptoRng, RngCore};

use crate::grafting::{RnsGadgetLayout, RnsKeySwitchKey, RnsRlweCiphertext};

use super::{apply_automorphism, apply_rns_automorphism};

/// RNS Galois evaluation key for one CKKS automorphism.
///
/// For exponent `k`, the transformed ciphertext is naturally under
///
/// `sigma_k(s)`.
///
/// The contained RNS key-switch key maps
///
/// `sigma_k(s) -> s`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsGaloisKey {
    exponent: usize,
    key_switch_key: RnsKeySwitchKey,
}

impl RnsGaloisKey {
    pub fn generate_with_rng<R>(
        degree: usize,
        plaintext_modulus: u64,
        noise_bound: i64,
        secret_coefficients: &[i8],
        exponent: usize,
        layout: RnsGadgetLayout,
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        assert_eq!(
            secret_coefficients.len(),
            degree,
            "secret coefficient count must match RLWE degree"
        );

        assert!(
            secret_coefficients
                .iter()
                .all(|&value| matches!(value, -1..=1)),
            "RNS Galois secret coefficients must be ternary"
        );

        let two_n = 2 * degree;

        let exponent = exponent % two_n;

        assert!(
            exponent % 2 == 1,
            "CKKS Galois exponent must be odd modulo 2N"
        );

        /*
         * Construct sigma_k(s) in signed ternary form.
         *
         * We may use any modulus > 2 because automorphisms preserve
         * coefficients in {-1,0,1}; modulus 3 is sufficient to carry
         * that representation exactly.
         */
        let modulus = crate::ring::Modulus::new(3);

        let polynomial = crate::ring::Polynomial::new(
            modulus,
            secret_coefficients
                .iter()
                .map(|&value| match value {
                    -1 => modulus.value() - 1,
                    0 => 0,
                    1 => 1,
                    _ => unreachable!(),
                })
                .collect(),
        );

        let transformed = apply_automorphism(&polynomial, exponent);

        let transformed_secret: Vec<i8> = transformed
            .coefficients()
            .iter()
            .map(|&value| match value {
                0 => 0,
                1 => 1,
                value if value == modulus.value() - 1 => -1,
                _ => panic!("automorphism of ternary secret must remain ternary"),
            })
            .collect();

        let key_switch_key = RnsKeySwitchKey::generate_with_rng(
            degree,
            plaintext_modulus,
            noise_bound,
            &transformed_secret,
            secret_coefficients,
            layout,
            rng,
        );

        Self {
            exponent,
            key_switch_key,
        }
    }

    pub fn exponent(&self) -> usize {
        self.exponent
    }

    pub fn key_switch_key(&self) -> &RnsKeySwitchKey {
        &self.key_switch_key
    }

    pub fn layout(&self) -> &RnsGadgetLayout {
        self.key_switch_key.layout()
    }
}

/// Applies an RNS CKKS Galois automorphism and switches the transformed
/// ciphertext back to the original secret.
pub fn apply_rns_galois_automorphism(
    ciphertext: &RnsRlweCiphertext,
    galois_key: &RnsGaloisKey,
) -> RnsRlweCiphertext {
    assert_eq!(
        ciphertext.basis(),
        galois_key.layout().full_basis(),
        "RNS ciphertext basis must match RNS Galois key"
    );

    let transformed = apply_rns_automorphism(ciphertext, galois_key.exponent());

    crate::grafting::rns_key_switch::rns_key_switch(&transformed, galois_key.key_switch_key())
}

/// Applies an RNS Galois automorphism to a leveled CKKS ciphertext.
///
/// The active CKKS level, basis, and scale are preserved. Only the
/// encrypted polynomial is transformed and key-switched back to the
/// original secret.
pub fn apply_rns_ckks_galois_automorphism(
    ciphertext: &crate::ckks::RnsCkksCiphertext,
    galois_key: &RnsGaloisKey,
    chain: &crate::ring::ModulusChain,
) -> crate::ckks::RnsCkksCiphertext {
    ciphertext.assert_matches_chain(chain);

    assert_eq!(
        galois_key.layout().full_basis(),
        ciphertext.basis(),
        "RNS Galois key basis must match active CKKS level"
    );

    let transformed = apply_rns_galois_automorphism(ciphertext.rlwe(), galois_key);

    crate::ckks::RnsCkksCiphertext::new(transformed, ciphertext.state().clone(), chain)
}

/// Rotates logical CKKS slots left by `steps` at the current RNS level.
pub fn rotate_left_rns_ckks(
    ciphertext: &crate::ckks::RnsCkksCiphertext,
    steps: usize,
    galois_key: &RnsGaloisKey,
    chain: &crate::ring::ModulusChain,
) -> crate::ckks::RnsCkksCiphertext {
    let expected = crate::ckks::rotation_exponent_left(ciphertext.rlwe().degree(), steps);

    assert_eq!(
        galois_key.exponent(),
        expected,
        "RNS Galois key exponent does not match requested left rotation"
    );

    apply_rns_ckks_galois_automorphism(ciphertext, galois_key, chain)
}

/// Rotates logical CKKS slots right by `steps` at the current RNS level.
pub fn rotate_right_rns_ckks(
    ciphertext: &crate::ckks::RnsCkksCiphertext,
    steps: usize,
    galois_key: &RnsGaloisKey,
    chain: &crate::ring::ModulusChain,
) -> crate::ckks::RnsCkksCiphertext {
    let expected = crate::ckks::rotation_exponent_right(ciphertext.rlwe().degree(), steps);

    assert_eq!(
        galois_key.exponent(),
        expected,
        "RNS Galois key exponent does not match requested right rotation"
    );

    apply_rns_ckks_galois_automorphism(ciphertext, galois_key, chain)
}

/// Applies logical CKKS complex conjugation at the current RNS level.
///
/// The supplied Galois key must correspond to exponent `-1 mod 2N`.
/// Level, active basis, and CKKS scale are preserved.
pub fn conjugate_rns_ckks(
    ciphertext: &crate::ckks::RnsCkksCiphertext,
    galois_key: &RnsGaloisKey,
    chain: &crate::ring::ModulusChain,
) -> crate::ckks::RnsCkksCiphertext {
    let expected = crate::ckks::conjugation_exponent(ciphertext.rlwe().degree());

    assert_eq!(
        galois_key.exponent(),
        expected,
        "RNS Galois key exponent does not match CKKS complex conjugation"
    );

    apply_rns_ckks_galois_automorphism(ciphertext, galois_key, chain)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::grafting::{decrypt_rns_raw, RnsGadgetLayout, RnsRlweCiphertext};
    use crate::ring::{Modulus, ModulusBasis, Polynomial};
    use crate::rlwe::{encrypt_raw_with_rng, RlweParameters, SecretKey};

    use super::*;

    fn basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ])
    }

    fn secret() -> Vec<i8> {
        vec![-1, 0, 1, 1, 0, -1, 1, 0]
    }

    fn project_secret(modulus: Modulus, coefficients: &[i8]) -> SecretKey {
        SecretKey::from_polynomial(Polynomial::new(
            modulus,
            coefficients
                .iter()
                .map(|&value| match value {
                    -1 => modulus.value() - 1,
                    0 => 0,
                    1 => 1,
                    _ => panic!("secret must be ternary"),
                })
                .collect(),
        ))
    }

    fn encrypt_message(
        basis: &ModulusBasis,
        secret: &[i8],
        message: &[u128],
        seed: u64,
    ) -> RnsRlweCiphertext {
        let degree = secret.len();

        let limbs = basis
            .moduli()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, modulus)| {
                let params = RlweParameters::new(degree, modulus, 2, 0);

                let secret_key = project_secret(modulus, secret);

                let plaintext = Polynomial::new(
                    modulus,
                    message
                        .iter()
                        .map(|&value| (value % u128::from(modulus.value())) as u64)
                        .collect(),
                );

                let mut rng = ChaCha20Rng::seed_from_u64(seed ^ index as u64);

                encrypt_raw_with_rng(params, &secret_key, &plaintext, &mut rng)
            })
            .collect();

        RnsRlweCiphertext::from_limbs(limbs)
    }

    #[test]
    fn rns_galois_key_normalizes_exponent() {
        let basis = basis();

        let secret = secret();

        let layout = RnsGadgetLayout::new(basis, vec![1, 2]);

        let mut rng = ChaCha20Rng::seed_from_u64(0xB100);

        let key = RnsGaloisKey::generate_with_rng(8, 2, 0, &secret, 19, layout, &mut rng);

        assert_eq!(key.exponent(), 3);
    }

    #[test]
    fn encrypted_rns_galois_automorphism_matches_plaintext_automorphism() {
        let basis = basis();

        let secret = secret();

        let message = [3_u128, 1, 4, 1, 5, 9, 2, 6];

        let ciphertext = encrypt_message(&basis, &secret, &message, 0xB200);

        let exponent = 5;

        let layout = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xB201);

        let key = RnsGaloisKey::generate_with_rng(
            secret.len(),
            2,
            0,
            &secret,
            exponent,
            layout,
            &mut key_rng,
        );

        let transformed = apply_rns_galois_automorphism(&ciphertext, &key);

        let observed = decrypt_rns_raw(&transformed, &secret);

        let plaintext =
            crate::ring::RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &message);

        let expected_residues = plaintext
            .residues()
            .iter()
            .map(|residue| apply_automorphism(residue, exponent))
            .collect();

        let expected = crate::ring::RnsPolynomial::from_residues(expected_residues);

        assert_eq!(observed, expected);
    }

    #[test]
    fn inverse_rns_galois_automorphism_recovers_plaintext() {
        let basis = basis();

        let secret = secret();

        let message = [2_u128, 7, 1, 8, 2, 8, 1, 8];

        let ciphertext = encrypt_message(&basis, &secret, &message, 0xB300);

        let exponent = 5;

        let inverse = crate::ckks::inverse_automorphism_exponent(secret.len(), exponent);

        let mut forward_rng = ChaCha20Rng::seed_from_u64(0xB301);

        let forward_key = RnsGaloisKey::generate_with_rng(
            secret.len(),
            2,
            0,
            &secret,
            exponent,
            RnsGadgetLayout::new(basis.clone(), vec![1, 2]),
            &mut forward_rng,
        );

        let mut inverse_rng = ChaCha20Rng::seed_from_u64(0xB302);

        let inverse_key = RnsGaloisKey::generate_with_rng(
            secret.len(),
            2,
            0,
            &secret,
            inverse,
            RnsGadgetLayout::new(basis, vec![1, 2]),
            &mut inverse_rng,
        );

        let transformed = apply_rns_galois_automorphism(&ciphertext, &forward_key);

        let recovered = apply_rns_galois_automorphism(&transformed, &inverse_key);

        let observed = decrypt_rns_raw(&recovered, &secret);

        let expected = decrypt_rns_raw(&ciphertext, &secret);

        assert_eq!(observed, expected);
    }

    #[test]
    fn rns_galois_automorphism_preserves_basis() {
        let basis = basis();

        let secret = secret();

        let ciphertext = encrypt_message(&basis, &secret, &[1_u128, 2, 3, 4, 5, 6, 7, 8], 0xB400);

        let mut rng = ChaCha20Rng::seed_from_u64(0xB401);

        let key = RnsGaloisKey::generate_with_rng(
            secret.len(),
            2,
            0,
            &secret,
            5,
            RnsGadgetLayout::new(basis.clone(), vec![1, 2]),
            &mut rng,
        );

        let transformed = apply_rns_galois_automorphism(&ciphertext, &key);

        assert_eq!(transformed.basis(), &basis);
    }

    fn ckks_chain() -> crate::ring::ModulusChain {
        crate::ring::ModulusChain::from_top_basis(crate::ring::ModulusBasis::new(vec![
            crate::ring::Modulus::new(12_289),
            crate::ring::Modulus::new(40_961),
            crate::ring::Modulus::new(65_537),
        ]))
    }

    fn encrypt_ckks_slots(
        chain: &crate::ring::ModulusChain,
        level: usize,
        slots: &[num_complex::Complex64],
        scale: f64,
        secret: &[i8],
        seed: u64,
    ) -> crate::ckks::RnsCkksCiphertext {
        let basis = chain.level(level);

        let composite = basis.composite_modulus();

        assert!(
            composite <= u128::from(u64::MAX),
            "test CKKS composite modulus must fit in u64"
        );

        let encoder = crate::ckks::CkksSlotEncoder::new(
            secret.len(),
            crate::ring::Modulus::new(composite as u64),
            scale,
        );

        let encoded = encoder.encode_slots(slots);

        let limbs = basis
            .moduli()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, modulus)| {
                let params = crate::rlwe::RlweParameters::new(secret.len(), modulus, 2, 0);

                let secret_key = project_secret(modulus, secret);

                let plaintext = crate::ring::Polynomial::new(
                    modulus,
                    encoded
                        .coefficients()
                        .iter()
                        .map(|&value| value % modulus.value())
                        .collect(),
                );

                let mut rng = ChaCha20Rng::seed_from_u64(seed ^ (index as u64 * 0x9E37));

                crate::rlwe::encrypt_raw_with_rng(params, &secret_key, &plaintext, &mut rng)
            })
            .collect();

        crate::ckks::RnsCkksCiphertext::new(
            RnsRlweCiphertext::from_limbs(limbs),
            crate::ckks::CkksChainState::new(chain, level, scale),
            chain,
        )
    }

    fn decrypt_ckks_slots(
        ciphertext: &crate::ckks::RnsCkksCiphertext,
        secret: &[i8],
    ) -> Vec<num_complex::Complex64> {
        let plaintext = decrypt_rns_raw(ciphertext.rlwe(), secret);

        let composite = plaintext.composite_modulus();

        assert!(
            composite <= u128::from(u64::MAX),
            "test CKKS composite modulus must fit in u64"
        );

        let modulus = crate::ring::Modulus::new(composite as u64);

        let coefficients: Vec<u64> = plaintext
            .reconstruct_coefficients()
            .into_iter()
            .map(|value| value as u64)
            .collect();

        let polynomial = crate::ring::Polynomial::new(modulus, coefficients);

        crate::ckks::CkksSlotEncoder::new(ciphertext.rlwe().degree(), modulus, ciphertext.scale())
            .decode_slots(&polynomial)
    }

    fn assert_complex_vectors_close(
        actual: &[num_complex::Complex64],
        expected: &[num_complex::Complex64],
        tolerance: f64,
    ) {
        assert_eq!(actual.len(), expected.len());

        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let error = (actual - expected).norm();

            assert!(
                error <= tolerance,
                "slot {index}: actual={actual:?}, \
                 expected={expected:?}, \
                 error={error}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn level_zero_rns_ckks_left_rotation_matches_logical_slots() {
        use num_complex::Complex64;

        let chain = ckks_chain();

        let secret = secret();

        let slots = [
            Complex64::new(0.25, 0.50),
            Complex64::new(-0.75, 0.125),
            Complex64::new(1.25, -0.375),
            Complex64::new(-0.50, -0.625),
        ];

        let scale = 65_537.0;

        let ciphertext = encrypt_ckks_slots(&chain, 0, &slots, scale, &secret, 0xC100);

        let steps = 1;

        let exponent = crate::ckks::rotation_exponent_left(secret.len(), steps);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xC101);

        let key = RnsGaloisKey::generate_with_rng(
            secret.len(),
            2,
            0,
            &secret,
            exponent,
            RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]),
            &mut key_rng,
        );

        let rotated = rotate_left_rns_ckks(&ciphertext, steps, &key, &chain);

        assert_eq!(rotated.level(), ciphertext.level());

        assert_eq!(rotated.scale(), ciphertext.scale());

        assert_eq!(rotated.basis(), ciphertext.basis());

        let actual = decrypt_ckks_slots(&rotated, &secret);

        let expected = [slots[1], slots[2], slots[3], slots[0]];

        assert_complex_vectors_close(&actual, &expected, 5.0e-4);
    }

    #[test]
    fn level_zero_rns_ckks_right_rotation_matches_logical_slots() {
        use num_complex::Complex64;

        let chain = ckks_chain();

        let secret = secret();

        let slots = [
            Complex64::new(0.50, -0.25),
            Complex64::new(1.00, 0.50),
            Complex64::new(-0.25, 0.75),
            Complex64::new(0.125, -1.00),
        ];

        let scale = 65_537.0;

        let ciphertext = encrypt_ckks_slots(&chain, 0, &slots, scale, &secret, 0xC200);

        let steps = 1;

        let exponent = crate::ckks::rotation_exponent_right(secret.len(), steps);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xC201);

        let key = RnsGaloisKey::generate_with_rng(
            secret.len(),
            2,
            0,
            &secret,
            exponent,
            RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]),
            &mut key_rng,
        );

        let rotated = rotate_right_rns_ckks(&ciphertext, steps, &key, &chain);

        let actual = decrypt_ckks_slots(&rotated, &secret);

        let expected = [slots[3], slots[0], slots[1], slots[2]];

        assert_complex_vectors_close(&actual, &expected, 5.0e-4);
    }

    #[test]
    fn level_zero_rns_ckks_left_then_right_recovers_slots() {
        use num_complex::Complex64;

        let chain = ckks_chain();

        let secret = secret();

        let slots = [
            Complex64::new(0.25, 0.125),
            Complex64::new(-0.50, 0.25),
            Complex64::new(0.75, -0.125),
            Complex64::new(0.125, 0.50),
        ];

        let ciphertext = encrypt_ckks_slots(&chain, 0, &slots, 65_537.0, &secret, 0xC300);

        let steps = 2;

        let mut left_rng = ChaCha20Rng::seed_from_u64(0xC301);

        let left_key = RnsGaloisKey::generate_with_rng(
            secret.len(),
            2,
            0,
            &secret,
            crate::ckks::rotation_exponent_left(secret.len(), steps),
            RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]),
            &mut left_rng,
        );

        let mut right_rng = ChaCha20Rng::seed_from_u64(0xC302);

        let right_key = RnsGaloisKey::generate_with_rng(
            secret.len(),
            2,
            0,
            &secret,
            crate::ckks::rotation_exponent_right(secret.len(), steps),
            RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]),
            &mut right_rng,
        );

        let left = rotate_left_rns_ckks(&ciphertext, steps, &left_key, &chain);

        let recovered = rotate_right_rns_ckks(&left, steps, &right_key, &chain);

        let actual = decrypt_ckks_slots(&recovered, &secret);

        assert_complex_vectors_close(&actual, &slots, 5.0e-4);
    }

    fn multiplication_key_for_level(
        chain: &crate::ring::ModulusChain,
        level: usize,
        secret: &[i8],
        seed: u64,
    ) -> crate::grafting::RnsMultiplicationKey {
        let basis = chain.level(level);

        let block_sizes = if basis.len() == 3 {
            vec![1, 2]
        } else if basis.len() == 2 {
            vec![1, 1]
        } else {
            vec![1]
        };

        let layout = RnsGadgetLayout::new(basis.clone(), block_sizes);

        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        crate::grafting::RnsMultiplicationKey::generate_with_rng(
            secret.len(),
            2,
            0,
            secret,
            layout,
            &mut rng,
        )
    }

    #[test]
    fn level_zero_rns_ckks_conjugation_matches_logical_slots() {
        use num_complex::Complex64;

        let chain = ckks_chain();

        let secret = secret();

        let slots = [
            Complex64::new(0.25, 0.50),
            Complex64::new(-0.75, 0.125),
            Complex64::new(1.25, -0.375),
            Complex64::new(-0.50, -0.625),
        ];

        let ciphertext = encrypt_ckks_slots(&chain, 0, &slots, 65_537.0, &secret, 0xD100);

        let exponent = crate::ckks::conjugation_exponent(secret.len());

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xD101);

        let key = RnsGaloisKey::generate_with_rng(
            secret.len(),
            2,
            0,
            &secret,
            exponent,
            RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]),
            &mut key_rng,
        );

        let conjugated = conjugate_rns_ckks(&ciphertext, &key, &chain);

        assert_eq!(conjugated.level(), 0);

        assert_eq!(conjugated.scale(), ciphertext.scale());

        let actual = decrypt_ckks_slots(&conjugated, &secret);

        let expected: Vec<_> = slots.iter().map(|slot| slot.conj()).collect();

        assert_complex_vectors_close(&actual, &expected, 5.0e-4);
    }

    #[test]
    fn post_rescale_rns_ckks_rotation_matches_slotwise_product() {
        use num_complex::Complex64;

        let chain = ckks_chain();

        let secret = secret();

        let lhs_slots = [
            Complex64::new(0.25, 0.125),
            Complex64::new(-0.50, 0.25),
            Complex64::new(0.75, -0.125),
            Complex64::new(0.125, 0.50),
        ];

        let rhs_slots = [
            Complex64::new(0.50, -0.25),
            Complex64::new(0.25, 0.50),
            Complex64::new(-0.125, 0.25),
            Complex64::new(0.75, -0.125),
        ];

        let scale = 65_537.0;

        let lhs = encrypt_ckks_slots(&chain, 0, &lhs_slots, scale, &secret, 0xD200);

        let rhs = encrypt_ckks_slots(&chain, 0, &rhs_slots, scale, &secret, 0xD201);

        let multiplication_key = multiplication_key_for_level(&chain, 0, &secret, 0xD202);

        let product = crate::ckks::multiply_relinearize_rescale_rns_ckks(
            &lhs,
            &rhs,
            &multiplication_key,
            &chain,
        );

        assert_eq!(product.level(), 1);

        assert_eq!(product.basis(), chain.level(1));

        let steps = 1;

        let exponent = crate::ckks::rotation_exponent_left(secret.len(), steps);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xD203);

        let rotation_key = RnsGaloisKey::generate_with_rng(
            secret.len(),
            2,
            0,
            &secret,
            exponent,
            RnsGadgetLayout::new(chain.level(1).clone(), vec![1, 1]),
            &mut key_rng,
        );

        let rotated = rotate_left_rns_ckks(&product, steps, &rotation_key, &chain);

        assert_eq!(rotated.level(), 1);

        assert_eq!(rotated.basis(), chain.level(1));

        assert_eq!(rotated.scale(), product.scale());

        let actual = decrypt_ckks_slots(&rotated, &secret);

        let product_slots: Vec<_> = lhs_slots
            .iter()
            .zip(&rhs_slots)
            .map(|(&lhs, &rhs)| lhs * rhs)
            .collect();

        let expected = [
            product_slots[1],
            product_slots[2],
            product_slots[3],
            product_slots[0],
        ];

        assert_complex_vectors_close(&actual, &expected, 3.0e-3);
    }

    #[test]
    fn post_rescale_rns_ckks_conjugation_matches_slotwise_product() {
        use num_complex::Complex64;

        let chain = ckks_chain();

        let secret = secret();

        let lhs_slots = [
            Complex64::new(0.125, 0.25),
            Complex64::new(-0.25, 0.125),
            Complex64::new(0.50, -0.125),
            Complex64::new(0.25, 0.375),
        ];

        let rhs_slots = [
            Complex64::new(0.50, 0.125),
            Complex64::new(0.25, -0.25),
            Complex64::new(-0.25, 0.125),
            Complex64::new(0.50, -0.125),
        ];

        let lhs = encrypt_ckks_slots(&chain, 0, &lhs_slots, 65_537.0, &secret, 0xD300);

        let rhs = encrypt_ckks_slots(&chain, 0, &rhs_slots, 65_537.0, &secret, 0xD301);

        let multiplication_key = multiplication_key_for_level(&chain, 0, &secret, 0xD302);

        let product = crate::ckks::multiply_relinearize_rescale_rns_ckks(
            &lhs,
            &rhs,
            &multiplication_key,
            &chain,
        );

        assert_eq!(product.level(), 1);

        let exponent = crate::ckks::conjugation_exponent(secret.len());

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xD303);

        let conjugation_key = RnsGaloisKey::generate_with_rng(
            secret.len(),
            2,
            0,
            &secret,
            exponent,
            RnsGadgetLayout::new(chain.level(1).clone(), vec![1, 1]),
            &mut key_rng,
        );

        let conjugated = conjugate_rns_ckks(&product, &conjugation_key, &chain);

        assert_eq!(conjugated.level(), 1);

        assert_eq!(conjugated.basis(), chain.level(1));

        assert_eq!(conjugated.scale(), product.scale());

        let actual = decrypt_ckks_slots(&conjugated, &secret);

        let expected: Vec<_> = lhs_slots
            .iter()
            .zip(&rhs_slots)
            .map(|(&lhs, &rhs)| (lhs * rhs).conj())
            .collect();

        assert_complex_vectors_close(&actual, &expected, 3.0e-3);
    }

    #[test]
    #[should_panic(expected = "RNS Galois key basis must match active CKKS level")]
    fn post_rescale_operation_rejects_top_level_galois_key() {
        use num_complex::Complex64;

        let chain = ckks_chain();

        let secret = secret();

        let lhs = encrypt_ckks_slots(
            &chain,
            0,
            &[
                Complex64::new(0.25, 0.0),
                Complex64::new(0.50, 0.0),
                Complex64::new(0.75, 0.0),
                Complex64::new(1.00, 0.0),
            ],
            65_537.0,
            &secret,
            0xD400,
        );

        let rhs = encrypt_ckks_slots(
            &chain,
            0,
            &[
                Complex64::new(0.50, 0.0),
                Complex64::new(0.25, 0.0),
                Complex64::new(0.125, 0.0),
                Complex64::new(0.0625, 0.0),
            ],
            65_537.0,
            &secret,
            0xD401,
        );

        let multiplication_key = multiplication_key_for_level(&chain, 0, &secret, 0xD402);

        let product = crate::ckks::multiply_relinearize_rescale_rns_ckks(
            &lhs,
            &rhs,
            &multiplication_key,
            &chain,
        );

        assert_eq!(product.level(), 1);

        let mut wrong_key_rng = ChaCha20Rng::seed_from_u64(0xD403);

        let wrong_key = RnsGaloisKey::generate_with_rng(
            secret.len(),
            2,
            0,
            &secret,
            crate::ckks::rotation_exponent_left(secret.len(), 1),
            RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]),
            &mut wrong_key_rng,
        );

        let _ = rotate_left_rns_ckks(&product, 1, &wrong_key, &chain);
    }

    #[test]
    fn randomized_rns_ckks_galois_differential_campaign() {
        use num_complex::Complex64;
        use rand::Rng;

        let chain = ckks_chain();

        let secret = secret();

        let degree = secret.len();

        let slot_count = degree / 2;

        let tolerance_level0 = 7.5e-4;

        let tolerance_level1 = 4.0e-3;

        for seed in 0_u64..24 {
            let mut value_rng = ChaCha20Rng::seed_from_u64(0xE000 ^ seed);

            let lhs_slots: Vec<_> = (0..slot_count)
                .map(|_| {
                    Complex64::new(
                        value_rng.gen_range(-0.75..=0.75),
                        value_rng.gen_range(-0.75..=0.75),
                    )
                })
                .collect();

            let rhs_slots: Vec<_> = (0..slot_count)
                .map(|_| {
                    Complex64::new(
                        value_rng.gen_range(-0.50..=0.50),
                        value_rng.gen_range(-0.50..=0.50),
                    )
                })
                .collect();

            let lhs = encrypt_ckks_slots(&chain, 0, &lhs_slots, 65_537.0, &secret, 0xE100 ^ seed);

            let rhs = encrypt_ckks_slots(&chain, 0, &rhs_slots, 65_537.0, &secret, 0xE200 ^ seed);

            for steps in 0..slot_count {
                /*
                 * Level-0 left rotation.
                 */
                let left_exponent = crate::ckks::rotation_exponent_left(degree, steps);

                let mut left_key_rng =
                    ChaCha20Rng::seed_from_u64(0xE300 ^ seed ^ ((steps as u64) << 8));

                let left_key = RnsGaloisKey::generate_with_rng(
                    degree,
                    2,
                    0,
                    &secret,
                    left_exponent,
                    RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]),
                    &mut left_key_rng,
                );

                let left = rotate_left_rns_ckks(&lhs, steps, &left_key, &chain);

                let observed_left = decrypt_ckks_slots(&left, &secret);

                let expected_left: Vec<_> = (0..slot_count)
                    .map(|destination| lhs_slots[(destination + steps) % slot_count])
                    .collect();

                assert_complex_vectors_close(&observed_left, &expected_left, tolerance_level0);

                /*
                 * Level-0 right rotation.
                 */
                let right_exponent = crate::ckks::rotation_exponent_right(degree, steps);

                let mut right_key_rng =
                    ChaCha20Rng::seed_from_u64(0xE400 ^ seed ^ ((steps as u64) << 8));

                let right_key = RnsGaloisKey::generate_with_rng(
                    degree,
                    2,
                    0,
                    &secret,
                    right_exponent,
                    RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]),
                    &mut right_key_rng,
                );

                let right = rotate_right_rns_ckks(&lhs, steps, &right_key, &chain);

                let observed_right = decrypt_ckks_slots(&right, &secret);

                let expected_right: Vec<_> = (0..slot_count)
                    .map(|destination| lhs_slots[(destination + slot_count - steps) % slot_count])
                    .collect();

                assert_complex_vectors_close(&observed_right, &expected_right, tolerance_level0);
            }

            /*
             * Level-0 conjugation.
             */
            let mut conjugation_rng = ChaCha20Rng::seed_from_u64(0xE500 ^ seed);

            let conjugation_key = RnsGaloisKey::generate_with_rng(
                degree,
                2,
                0,
                &secret,
                crate::ckks::conjugation_exponent(degree),
                RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]),
                &mut conjugation_rng,
            );

            let conjugated = conjugate_rns_ckks(&lhs, &conjugation_key, &chain);

            let observed_conjugation = decrypt_ckks_slots(&conjugated, &secret);

            let expected_conjugation: Vec<_> = lhs_slots.iter().map(|slot| slot.conj()).collect();

            assert_complex_vectors_close(
                &observed_conjugation,
                &expected_conjugation,
                tolerance_level0,
            );

            /*
             * Move to level 1 through a genuine encrypted multiply,
             * relinearization, and rescale.
             */
            let multiplication_key =
                multiplication_key_for_level(&chain, 0, &secret, 0xE600 ^ seed);

            let product = crate::ckks::multiply_relinearize_rescale_rns_ckks(
                &lhs,
                &rhs,
                &multiplication_key,
                &chain,
            );

            assert_eq!(product.level(), 1);

            let expected_product: Vec<_> = lhs_slots
                .iter()
                .zip(&rhs_slots)
                .map(|(&lhs, &rhs)| lhs * rhs)
                .collect();

            /*
             * Exercise all logical rotations at level 1.
             */
            for steps in 0..slot_count {
                let exponent = crate::ckks::rotation_exponent_left(degree, steps);

                let mut rotation_rng =
                    ChaCha20Rng::seed_from_u64(0xE700 ^ seed ^ ((steps as u64) << 8));

                let key = RnsGaloisKey::generate_with_rng(
                    degree,
                    2,
                    0,
                    &secret,
                    exponent,
                    RnsGadgetLayout::new(chain.level(1).clone(), vec![1, 1]),
                    &mut rotation_rng,
                );

                let rotated = rotate_left_rns_ckks(&product, steps, &key, &chain);

                let observed = decrypt_ckks_slots(&rotated, &secret);

                let expected: Vec<_> = (0..slot_count)
                    .map(|destination| expected_product[(destination + steps) % slot_count])
                    .collect();

                assert_complex_vectors_close(&observed, &expected, tolerance_level1);
            }

            /*
             * Level-1 conjugation after multiply/rescale.
             */
            let mut level1_conjugation_rng = ChaCha20Rng::seed_from_u64(0xE800 ^ seed);

            let level1_conjugation_key = RnsGaloisKey::generate_with_rng(
                degree,
                2,
                0,
                &secret,
                crate::ckks::conjugation_exponent(degree),
                RnsGadgetLayout::new(chain.level(1).clone(), vec![1, 1]),
                &mut level1_conjugation_rng,
            );

            let level1_conjugated = conjugate_rns_ckks(&product, &level1_conjugation_key, &chain);

            let observed = decrypt_ckks_slots(&level1_conjugated, &secret);

            let expected: Vec<_> = expected_product.iter().map(|slot| slot.conj()).collect();

            assert_complex_vectors_close(&observed, &expected, tolerance_level1);
        }
    }
}
