use rand::{CryptoRng, RngCore};

use crate::ring::{Polynomial, RnsPolynomial};
use crate::rlwe::{encrypt_raw_with_rng, RlweCiphertext, RlweParameters, SecretKey};

use super::{
    decrypt_pow2_raw, encrypt_pow2_raw_with_rng, project_ternary_secret_pow2, HybridRlweCiphertext,
    MixedGadgetLayout, Pow2Polynomial, Pow2RlweCiphertext, Pow2RnsPolynomial, RnsRlweCiphertext,
};

/// Evaluation-key entry for one mixed gadget block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridEvaluationKeyEntry {
    ordinary: RnsRlweCiphertext,
    sprout: Pow2RlweCiphertext,
}

impl HybridEvaluationKeyEntry {
    pub fn ordinary(&self) -> &RnsRlweCiphertext {
        &self.ordinary
    }

    pub fn sprout(&self) -> &Pow2RlweCiphertext {
        &self.sprout
    }
}

/// Evaluation key for mixed odd-RNS / power-of-two gadget
/// relinearization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridMultiplicationKey {
    layout: MixedGadgetLayout,
    entries: Vec<HybridEvaluationKeyEntry>,
}

impl HybridMultiplicationKey {
    pub fn generate_with_rng<R>(
        degree: usize,
        plaintext_modulus: u64,
        secret_coefficients: &[i8],
        layout: MixedGadgetLayout,
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        assert_eq!(
            secret_coefficients.len(),
            degree,
            "hybrid secret coefficient count must match degree"
        );

        assert!(
            secret_coefficients
                .iter()
                .all(|&value| matches!(value, -1..=1)),
            "hybrid secret coefficients must be ternary"
        );

        let ordinary_basis = layout.ordinary_basis().clone();

        let sprout_bits = layout.sprout_bits();

        let ordinary_block_count = layout.ordinary_layout().block_count();

        let mut entries = Vec::with_capacity(layout.block_count());

        // Ordinary gadget blocks.
        for block_index in 0..ordinary_block_count {
            let idempotent = layout.ordinary_idempotent(block_index);

            entries.push(generate_entry(
                degree,
                plaintext_modulus,
                secret_coefficients,
                &ordinary_basis,
                sprout_bits,
                idempotent,
                rng,
            ));
        }

        // Terminal power-of-two sprout block.
        entries.push(generate_entry(
            degree,
            plaintext_modulus,
            secret_coefficients,
            &ordinary_basis,
            sprout_bits,
            layout.sprout_idempotent(),
            rng,
        ));

        Self { layout, entries }
    }

    pub fn layout(&self) -> &MixedGadgetLayout {
        &self.layout
    }

    pub fn entries(&self) -> &[HybridEvaluationKeyEntry] {
        &self.entries
    }

    pub fn entry(&self, index: usize) -> &HybridEvaluationKeyEntry {
        &self.entries[index]
    }
}

/// Degree-two ciphertext over the mixed hybrid basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridQuadraticCiphertext {
    c0: Pow2RnsPolynomial,
    c1: Pow2RnsPolynomial,
    c2: Pow2RnsPolynomial,
}

impl HybridQuadraticCiphertext {
    pub fn new(c0: Pow2RnsPolynomial, c1: Pow2RnsPolynomial, c2: Pow2RnsPolynomial) -> Self {
        assert_eq!(
            c0.ordinary_basis(),
            c1.ordinary_basis(),
            "hybrid quadratic c0/c1 ordinary bases must match"
        );

        assert_eq!(
            c0.ordinary_basis(),
            c2.ordinary_basis(),
            "hybrid quadratic c0/c2 ordinary bases must match"
        );

        assert_eq!(
            c0.sprout_bits(),
            c1.sprout_bits(),
            "hybrid quadratic c0/c1 sprout sizes must match"
        );

        assert_eq!(
            c0.sprout_bits(),
            c2.sprout_bits(),
            "hybrid quadratic c0/c2 sprout sizes must match"
        );

        assert_eq!(
            c0.degree(),
            c1.degree(),
            "hybrid quadratic c0/c1 degrees must match"
        );

        assert_eq!(
            c0.degree(),
            c2.degree(),
            "hybrid quadratic c0/c2 degrees must match"
        );

        Self { c0, c1, c2 }
    }

    pub fn c0(&self) -> &Pow2RnsPolynomial {
        &self.c0
    }

    pub fn c1(&self) -> &Pow2RnsPolynomial {
        &self.c1
    }

    pub fn c2(&self) -> &Pow2RnsPolynomial {
        &self.c2
    }
}

/// Relinearizes a mixed hybrid degree-two ciphertext.
pub fn hybrid_relinearize(
    product: &HybridQuadraticCiphertext,
    multiplication_key: &HybridMultiplicationKey,
) -> HybridRlweCiphertext {
    let layout = multiplication_key.layout();

    assert_eq!(
        product.c0().ordinary_basis(),
        layout.ordinary_basis(),
        "hybrid quadratic basis must match multiplication key"
    );

    assert_eq!(
        product.c0().sprout_bits(),
        layout.sprout_bits(),
        "hybrid quadratic sprout size must match multiplication key"
    );

    let decomposition = layout.decompose(product.c2());

    let ordinary_basis = layout.ordinary_basis();

    let ordinary_block_count = layout.ordinary_layout().block_count();

    let mut ordinary_output = Vec::with_capacity(ordinary_basis.len());

    // Odd-prime RNS limbs.
    for limb_index in 0..ordinary_basis.len() {
        let modulus = ordinary_basis.modulus(limb_index);

        let mut b = product.c0().ordinary().residue(limb_index).clone();

        let mut a = product.c1().ordinary().residue(limb_index).clone();

        for block_index in 0..ordinary_block_count {
            let digit = Polynomial::new(
                modulus,
                decomposition
                    .ordinary_digit(block_index)
                    .iter()
                    .map(|&value| (value % u128::from(modulus.value())) as u64)
                    .collect(),
            );

            let evaluation_key = multiplication_key
                .entry(block_index)
                .ordinary()
                .limb(limb_index);

            b = b.add(&digit.negacyclic_mul(evaluation_key.b()));

            a = a.add(&digit.negacyclic_mul(evaluation_key.a()));
        }

        // Terminal sprout gadget digit also contributes to every
        // ordinary modulus through its full CRT idempotent.
        let sprout_digit = Polynomial::new(
            modulus,
            decomposition
                .sprout_digit()
                .iter()
                .map(|&value| value % modulus.value())
                .collect(),
        );

        let sprout_entry = multiplication_key
            .entry(ordinary_block_count)
            .ordinary()
            .limb(limb_index);

        b = b.add(&sprout_digit.negacyclic_mul(sprout_entry.b()));

        a = a.add(&sprout_digit.negacyclic_mul(sprout_entry.a()));

        ordinary_output.push(RlweCiphertext::new(b, a));
    }

    let ordinary = RnsRlweCiphertext::from_limbs(ordinary_output);

    // Power-of-two sprout limb.
    let bits = layout.sprout_bits();

    let mut sprout_b = product.c0().sprout().clone();

    let mut sprout_a = product.c1().sprout().clone();

    for block_index in 0..ordinary_block_count {
        let digit = Pow2Polynomial::new(
            bits,
            decomposition
                .ordinary_digit(block_index)
                .iter()
                .map(|&value| (value % (1_u128 << bits)) as u64)
                .collect(),
        );

        let evaluation_key = multiplication_key.entry(block_index).sprout();

        sprout_b = sprout_b.add(&digit.negacyclic_mul(evaluation_key.b()));

        sprout_a = sprout_a.add(&digit.negacyclic_mul(evaluation_key.a()));
    }

    let sprout_digit = Pow2Polynomial::new(bits, decomposition.sprout_digit().to_vec());

    let evaluation_key = multiplication_key.entry(ordinary_block_count).sprout();

    sprout_b = sprout_b.add(&sprout_digit.negacyclic_mul(evaluation_key.b()));

    sprout_a = sprout_a.add(&sprout_digit.negacyclic_mul(evaluation_key.a()));

    HybridRlweCiphertext::new(ordinary, Pow2RlweCiphertext::new(sprout_b, sprout_a))
}

/// Raw hybrid RLWE decryption.
pub fn decrypt_hybrid_raw(
    ciphertext: &HybridRlweCiphertext,
    secret_coefficients: &[i8],
) -> Pow2RnsPolynomial {
    let ordinary_residues = ciphertext
        .ordinary()
        .basis()
        .moduli()
        .iter()
        .copied()
        .zip(ciphertext.ordinary().limbs())
        .map(|(modulus, limb)| {
            let secret = project_secret_odd(modulus, secret_coefficients);

            limb.b().add(&limb.a().negacyclic_mul(secret.polynomial()))
        })
        .collect();

    let ordinary = RnsPolynomial::from_residues(ordinary_residues);

    let sprout_secret =
        project_ternary_secret_pow2(ciphertext.sprout().bits(), secret_coefficients);

    let sprout = decrypt_pow2_raw(&sprout_secret, ciphertext.sprout());

    Pow2RnsPolynomial::from_parts(ordinary, sprout)
}

/// Raw hybrid quadratic decryption.
pub fn decrypt_hybrid_quadratic_raw(
    ciphertext: &HybridQuadraticCiphertext,
    secret_coefficients: &[i8],
) -> Pow2RnsPolynomial {
    let ordinary_residues = ciphertext
        .c0()
        .ordinary_basis()
        .moduli()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, modulus)| {
            let secret = project_secret_odd(modulus, secret_coefficients);

            let s = secret.polynomial();

            let s_squared = s.negacyclic_mul(s);

            ciphertext
                .c0()
                .ordinary()
                .residue(index)
                .add(&ciphertext.c1().ordinary().residue(index).negacyclic_mul(s))
                .add(
                    &ciphertext
                        .c2()
                        .ordinary()
                        .residue(index)
                        .negacyclic_mul(&s_squared),
                )
        })
        .collect();

    let ordinary = RnsPolynomial::from_residues(ordinary_residues);

    let bits = ciphertext.c0().sprout_bits();

    let secret = project_ternary_secret_pow2(bits, secret_coefficients);

    let s_squared = secret.negacyclic_mul(&secret);

    let sprout = ciphertext
        .c0()
        .sprout()
        .add(&ciphertext.c1().sprout().negacyclic_mul(&secret))
        .add(&ciphertext.c2().sprout().negacyclic_mul(&s_squared));

    Pow2RnsPolynomial::from_parts(ordinary, sprout)
}

fn generate_entry<R>(
    degree: usize,
    plaintext_modulus: u64,
    secret_coefficients: &[i8],
    ordinary_basis: &crate::ring::ModulusBasis,
    sprout_bits: u32,
    idempotent: u128,
    rng: &mut R,
) -> HybridEvaluationKeyEntry
where
    R: RngCore + CryptoRng,
{
    let ordinary_limbs = ordinary_basis
        .moduli()
        .iter()
        .copied()
        .map(|modulus| {
            let params = RlweParameters::new(degree, modulus, plaintext_modulus, 0);

            let secret = project_secret_odd(modulus, secret_coefficients);

            let s_squared = secret.polynomial().negacyclic_mul(secret.polynomial());

            let factor = (idempotent % u128::from(modulus.value())) as u64;

            let target = s_squared.scalar_mul(factor);

            encrypt_raw_with_rng(params, &secret, &target, rng)
        })
        .collect();

    let ordinary = RnsRlweCiphertext::from_limbs(ordinary_limbs);

    let sprout_secret = project_ternary_secret_pow2(sprout_bits, secret_coefficients);

    let sprout_squared = sprout_secret.negacyclic_mul(&sprout_secret);

    let factor = (idempotent % (1_u128 << sprout_bits)) as u64;

    let sprout_target = Pow2Polynomial::new(
        sprout_bits,
        sprout_squared
            .coefficients()
            .iter()
            .map(|&value| value.wrapping_mul(factor))
            .collect(),
    );

    let sprout = encrypt_pow2_raw_with_rng(&sprout_secret, &sprout_target, rng);

    HybridEvaluationKeyEntry { ordinary, sprout }
}

fn project_secret_odd(modulus: crate::ring::Modulus, coefficients: &[i8]) -> SecretKey {
    let values = coefficients
        .iter()
        .map(|&value| match value {
            -1 => modulus.value() - 1,
            0 => 0,
            1 => 1,
            _ => panic!("hybrid secret is not ternary"),
        })
        .collect();

    SecretKey::from_polynomial(Polynomial::new(modulus, values))
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ring::{Modulus, ModulusBasis};

    use super::*;

    fn basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ])
    }

    fn ternary() -> Vec<i8> {
        vec![-1, 0, 1, 1, -1, 0, 1, 0]
    }

    fn quadratic(bits: u32) -> HybridQuadraticCiphertext {
        let ordinary_basis = basis();

        let q = ordinary_basis.composite_modulus() * (1_u128 << bits);

        let make = |offset: u128| {
            Pow2RnsPolynomial::from_coefficients(
                &ordinary_basis,
                bits,
                &(0..8_u128)
                    .map(|index| (offset + 17 * index + 5 * index * index) % q)
                    .collect::<Vec<_>>(),
            )
        };

        HybridQuadraticCiphertext::new(make(3), make(29), make(71))
    }

    #[test]
    fn hybrid_evaluation_key_has_one_entry_per_mixed_block() {
        let layout = MixedGadgetLayout::new(basis(), vec![1, 2], 12);

        let mut rng = ChaCha20Rng::seed_from_u64(1);

        let key =
            HybridMultiplicationKey::generate_with_rng(8, 16, &ternary(), layout.clone(), &mut rng);

        assert_eq!(key.entries().len(), layout.block_count());
    }

    #[test]
    fn hybrid_relinearization_preserves_quadratic_semantics() {
        let layout = MixedGadgetLayout::new(basis(), vec![1, 2], 12);

        let product = quadratic(12);

        let mut rng = ChaCha20Rng::seed_from_u64(2);

        let key = HybridMultiplicationKey::generate_with_rng(8, 16, &ternary(), layout, &mut rng);

        let linear = hybrid_relinearize(&product, &key);

        assert_eq!(
            decrypt_hybrid_raw(&linear, &ternary(),),
            decrypt_hybrid_quadratic_raw(&product, &ternary(),)
        );
    }

    #[test]
    fn hybrid_relinearization_campaign_is_exact() {
        for bits in [4_u32, 8, 12, 16] {
            for block_sizes in [vec![3], vec![1, 2], vec![2, 1], vec![1, 1, 1]] {
                for seed in 0_u64..32 {
                    let layout = MixedGadgetLayout::new(basis(), block_sizes.clone(), bits);

                    let product = quadratic(bits);

                    let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0xABCD_1234);

                    let key = HybridMultiplicationKey::generate_with_rng(
                        8,
                        16,
                        &ternary(),
                        layout,
                        &mut rng,
                    );

                    let linear = hybrid_relinearize(&product, &key);

                    assert_eq!(
                        decrypt_hybrid_raw(&linear, &ternary(),),
                        decrypt_hybrid_quadratic_raw(&product, &ternary(),),
                        "hybrid relinearization mismatch: bits={bits}, seed={seed}"
                    );
                }
            }
        }
    }
}
