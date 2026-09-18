use rand::{CryptoRng, RngCore};

use crate::grafting::RnsGadgetLayout;
use crate::ring::{ModulusBasis, Polynomial, RnsPolynomial};
use crate::rlwe::{
    encrypt_raw_with_rng, RlweCiphertext, RlweParameters, RlweQuadraticCiphertext, SecretKey,
};

/// One RLWE ciphertext per RNS modulus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsRlweCiphertext {
    basis: ModulusBasis,
    limbs: Vec<RlweCiphertext>,
}

impl RnsRlweCiphertext {
    pub fn from_limbs(limbs: Vec<RlweCiphertext>) -> Self {
        assert!(
            !limbs.is_empty(),
            "RNS RLWE ciphertext requires at least one limb"
        );

        let degree = limbs[0].b().degree();

        for limb in &limbs {
            assert_eq!(
                limb.b().degree(),
                degree,
                "RNS RLWE ciphertext limbs must have the same degree"
            );
        }

        let basis = ModulusBasis::new(limbs.iter().map(|limb| limb.b().modulus()).collect());

        Self { basis, limbs }
    }

    pub fn basis(&self) -> &ModulusBasis {
        &self.basis
    }

    pub fn limbs(&self) -> &[RlweCiphertext] {
        &self.limbs
    }

    pub fn limb(&self, index: usize) -> &RlweCiphertext {
        &self.limbs[index]
    }

    pub fn degree(&self) -> usize {
        self.limbs[0].b().degree()
    }
}

/// One evaluation-key ciphertext for every gadget block and RNS limb.
///
/// Block `i` encrypts
///
/// ```text
/// E_i * s^2
/// ```
///
/// where E_i is the CRT idempotent of gadget block i.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsMultiplicationKey {
    layout: RnsGadgetLayout,
    entries: Vec<RnsRlweCiphertext>,
}

impl RnsMultiplicationKey {
    pub fn generate_with_rng<R>(
        degree: usize,
        plaintext_modulus: u64,
        noise_bound: i64,
        secret_coefficients: &[i8],
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
            "RNS secret coefficients must be ternary"
        );

        assert!(
            secret_coefficients.iter().any(|&value| value != 0),
            "RNS secret must be nonzero"
        );

        let basis = layout.full_basis().clone();

        let mut entries = Vec::with_capacity(layout.block_count());

        for block_index in 0..layout.block_count() {
            let idempotent = layout.crt_idempotent(block_index);

            let mut limbs = Vec::with_capacity(basis.len());

            for &modulus in basis.moduli() {
                let params = RlweParameters::new(degree, modulus, plaintext_modulus, noise_bound);

                let secret = project_secret(modulus, secret_coefficients);

                let secret_squared = secret.polynomial().negacyclic_mul(secret.polynomial());

                let factor = (idempotent % u128::from(modulus.value())) as u64;

                let target = secret_squared.scalar_mul(factor);

                limbs.push(encrypt_raw_with_rng(params, &secret, &target, rng));
            }

            entries.push(RnsRlweCiphertext::from_limbs(limbs));
        }

        Self { layout, entries }
    }

    pub fn layout(&self) -> &RnsGadgetLayout {
        &self.layout
    }

    pub fn entries(&self) -> &[RnsRlweCiphertext] {
        &self.entries
    }

    pub fn entry(&self, index: usize) -> &RnsRlweCiphertext {
        &self.entries[index]
    }
}

/// Projects one logical ternary secret into one coefficient modulus.
fn project_secret(modulus: crate::ring::Modulus, coefficients: &[i8]) -> SecretKey {
    let values = coefficients
        .iter()
        .map(|&value| match value {
            -1 => modulus.value() - 1,
            0 => 0,
            1 => 1,
            _ => unreachable!("secret coefficients validated as ternary"),
        })
        .collect();

    SecretKey::from_polynomial(Polynomial::new(modulus, values))
}

/// Projects an existing ternary secret into signed {-1,0,1} form.
pub fn ternary_secret_coefficients(secret: &SecretKey) -> Vec<i8> {
    let q = secret.polynomial().modulus().value();

    secret
        .polynomial()
        .coefficients()
        .iter()
        .map(|&value| {
            if value == 0 {
                0
            } else if value == 1 {
                1
            } else if value == q - 1 {
                -1
            } else {
                panic!("source secret is not ternary");
            }
        })
        .collect()
}

/// RNS representation of a degree-2 ciphertext.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsQuadraticCiphertext {
    c0: RnsPolynomial,
    c1: RnsPolynomial,
    c2: RnsPolynomial,
}

impl RnsQuadraticCiphertext {
    pub fn from_coefficient_ciphertext(
        product: &RlweQuadraticCiphertext,
        basis: &ModulusBasis,
    ) -> Self {
        let c0 = product
            .c0()
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect::<Vec<_>>();

        let c1 = product
            .c1()
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect::<Vec<_>>();

        let c2 = product
            .c2()
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect::<Vec<_>>();

        Self {
            c0: RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &c0),
            c1: RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &c1),
            c2: RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &c2),
        }
    }

    pub fn c0(&self) -> &RnsPolynomial {
        &self.c0
    }

    pub fn c1(&self) -> &RnsPolynomial {
        &self.c1
    }

    pub fn c2(&self) -> &RnsPolynomial {
        &self.c2
    }
}

/// Relinearizes a degree-2 RNS ciphertext using CRT gadget blocks.
pub fn rns_relinearize(
    product: &RnsQuadraticCiphertext,
    multiplication_key: &RnsMultiplicationKey,
) -> RnsRlweCiphertext {
    let layout = multiplication_key.layout();

    assert_eq!(
        product.c0().basis(),
        layout.full_basis(),
        "quadratic ciphertext basis must match RNS multiplication key"
    );

    assert_eq!(
        product.c1().basis(),
        layout.full_basis(),
        "quadratic ciphertext basis must match RNS multiplication key"
    );

    assert_eq!(
        product.c2().basis(),
        layout.full_basis(),
        "quadratic ciphertext basis must match RNS multiplication key"
    );

    let decomposition = layout.decompose(product.c2());

    let mut output_limbs = Vec::with_capacity(layout.full_basis().len());

    for limb_index in 0..layout.full_basis().len() {
        let modulus = layout.full_basis().modulus(limb_index);

        let mut b = product.c0().residue(limb_index).clone();

        let mut a = product.c1().residue(limb_index).clone();

        for block_index in 0..layout.block_count() {
            let digit = Polynomial::new(
                modulus,
                decomposition
                    .digit(block_index)
                    .iter()
                    .map(|&value| (value % u128::from(modulus.value())) as u64)
                    .collect(),
            );

            let evaluation_key = multiplication_key.entry(block_index).limb(limb_index);

            b = b.add(&digit.negacyclic_mul(evaluation_key.b()));

            a = a.add(&digit.negacyclic_mul(evaluation_key.a()));
        }

        output_limbs.push(RlweCiphertext::new(b, a));
    }

    RnsRlweCiphertext::from_limbs(output_limbs)
}

/// Raw RNS decryption using one logical ternary secret.
pub fn decrypt_rns_raw(
    ciphertext: &RnsRlweCiphertext,
    secret_coefficients: &[i8],
) -> RnsPolynomial {
    assert_eq!(
        secret_coefficients.len(),
        ciphertext.degree(),
        "secret coefficient count must match ciphertext degree"
    );

    let residues = ciphertext
        .basis()
        .moduli()
        .iter()
        .copied()
        .zip(ciphertext.limbs())
        .map(|(modulus, limb)| {
            let secret = project_secret(modulus, secret_coefficients);

            limb.b().add(&limb.a().negacyclic_mul(secret.polynomial()))
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

/// Raw degree-2 RNS decryption.
pub fn decrypt_rns_quadratic_raw(
    ciphertext: &RnsQuadraticCiphertext,
    secret_coefficients: &[i8],
) -> RnsPolynomial {
    let residues = ciphertext
        .c0()
        .basis()
        .moduli()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, modulus)| {
            let secret = project_secret(modulus, secret_coefficients);

            let s = secret.polynomial();

            let s_squared = s.negacyclic_mul(s);

            ciphertext
                .c0()
                .residue(index)
                .add(&ciphertext.c1().residue(index).negacyclic_mul(s))
                .add(&ciphertext.c2().residue(index).negacyclic_mul(&s_squared))
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ring::{Modulus, ModulusBasis};
    use crate::rlwe::{encrypt_with_rng, tensor, RlweParameters, RlwePlaintext, SecretKey};

    use super::*;

    fn basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ])
    }

    fn source_params() -> RlweParameters {
        RlweParameters::new(8, Modulus::new(12_289), 16, 0)
    }

    #[test]
    fn ternary_secret_projects_identically_across_moduli() {
        let params = source_params();

        let mut rng = ChaCha20Rng::seed_from_u64(1);

        let secret = SecretKey::generate_with_rng(params, &mut rng);

        let ternary = ternary_secret_coefficients(&secret);

        for &modulus in basis().moduli() {
            let projected = project_secret(modulus, &ternary);

            let recovered = ternary_secret_coefficients(&projected);

            assert_eq!(recovered, ternary);
        }
    }

    #[test]
    fn evaluation_key_targets_crt_idempotent_times_s_squared() {
        let params = source_params();

        let mut secret_rng = ChaCha20Rng::seed_from_u64(10);

        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let ternary = ternary_secret_coefficients(&secret);

        let layout = RnsGadgetLayout::new(basis(), vec![1, 2]);

        let mut eval_rng = ChaCha20Rng::seed_from_u64(11);

        let key = RnsMultiplicationKey::generate_with_rng(
            params.degree(),
            params.plaintext_modulus(),
            0,
            &ternary,
            layout.clone(),
            &mut eval_rng,
        );

        for block_index in 0..layout.block_count() {
            let e = layout.crt_idempotent(block_index);

            for (limb_index, &modulus) in layout.full_basis().moduli().iter().enumerate() {
                let projected = project_secret(modulus, &ternary);

                let s_squared = projected
                    .polynomial()
                    .negacyclic_mul(projected.polynomial());

                let factor = (e % u128::from(modulus.value())) as u64;

                let expected = s_squared.scalar_mul(factor);

                let actual = key.entry(block_index).limb(limb_index).b().add(
                    &key.entry(block_index)
                        .limb(limb_index)
                        .a()
                        .negacyclic_mul(projected.polynomial()),
                );

                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn rns_relinearization_preserves_quadratic_semantics_without_noise() {
        let params = source_params();

        let mut secret_rng = ChaCha20Rng::seed_from_u64(20);

        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let ternary = ternary_secret_coefficients(&secret);

        let lhs_plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let rhs_plaintext = RlwePlaintext::new(params, vec![8, 7, 6, 5, 4, 3, 2, 1]);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(21);

        let mut rhs_rng = ChaCha20Rng::seed_from_u64(22);

        let lhs = encrypt_with_rng(params, &secret, &lhs_plaintext, &mut lhs_rng);

        let rhs = encrypt_with_rng(params, &secret, &rhs_plaintext, &mut rhs_rng);

        let product = tensor(&lhs, &rhs);

        let layout = RnsGadgetLayout::new(basis(), vec![1, 2]);

        let rns_product =
            RnsQuadraticCiphertext::from_coefficient_ciphertext(&product, layout.full_basis());

        let mut eval_rng = ChaCha20Rng::seed_from_u64(23);

        let key = RnsMultiplicationKey::generate_with_rng(
            params.degree(),
            params.plaintext_modulus(),
            0,
            &ternary,
            layout,
            &mut eval_rng,
        );

        let relinearized = rns_relinearize(&rns_product, &key);

        assert_eq!(
            decrypt_rns_raw(&relinearized, &ternary,),
            decrypt_rns_quadratic_raw(&rns_product, &ternary,)
        );
    }

    #[test]
    fn rns_relinearization_campaign_is_exact_without_noise() {
        let params = source_params();

        for seed in 0_u64..32 {
            let mut secret_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x1111);

            let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

            let ternary = ternary_secret_coefficients(&secret);

            let lhs_plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);

            let rhs_plaintext = RlwePlaintext::new(params, vec![8, 7, 6, 5, 4, 3, 2, 1]);

            let mut lhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x2222);

            let mut rhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x3333);

            let product = tensor(
                &encrypt_with_rng(params, &secret, &lhs_plaintext, &mut lhs_rng),
                &encrypt_with_rng(params, &secret, &rhs_plaintext, &mut rhs_rng),
            );

            for block_sizes in [vec![3], vec![1, 2], vec![2, 1], vec![1, 1, 1]] {
                let layout = RnsGadgetLayout::new(basis(), block_sizes);

                let rns_product = RnsQuadraticCiphertext::from_coefficient_ciphertext(
                    &product,
                    layout.full_basis(),
                );

                let mut eval_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x4444);

                let key = RnsMultiplicationKey::generate_with_rng(
                    params.degree(),
                    params.plaintext_modulus(),
                    0,
                    &ternary,
                    layout,
                    &mut eval_rng,
                );

                let actual = decrypt_rns_raw(&rns_relinearize(&rns_product, &key), &ternary);

                let expected = decrypt_rns_quadratic_raw(&rns_product, &ternary);

                assert_eq!(
                    actual, expected,
                    "RNS relinearization mismatch for seed {seed}"
                );
            }
        }
    }
}
