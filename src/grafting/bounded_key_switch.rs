use rand::{CryptoRng, RngCore};

use crate::grafting::{
    encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout, RnsQuadraticCiphertext,
    RnsRlweCiphertext,
};
use crate::ring::{Polynomial, RnsNttPlan, RnsPolynomial};
use crate::rlwe::{ErrorDistribution, RlweCiphertext, SecretKey};

/// Configuration for bounded-base RNS multiplication-key generation.
pub struct BoundedRnsKeygenConfig<'a> {
    pub plaintext_modulus: u64,
    pub layout: BoundedGadgetLayout,
    pub plan: &'a RnsNttPlan,
}

/// Multiplication key using balanced signed power-of-two gadget digits.
///
/// Entry `j` encrypts `B^j * s^2` under the same secret, where `B` is the
/// bounded-gadget radix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedRnsMultiplicationKey {
    layout: BoundedGadgetLayout,
    entries: Vec<RnsRlweCiphertext>,
}

impl BoundedRnsMultiplicationKey {
    pub fn generate_with_distribution_ntt_rng<R>(
        config: BoundedRnsKeygenConfig<'_>,
        secret_coefficients: &[i8],
        distribution: ErrorDistribution,
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        assert_eq!(
            secret_coefficients.len(),
            config.plan.degree(),
            "secret coefficient count must match NTT plan degree"
        );

        assert_eq!(
            config.plan.moduli(),
            config.layout.full_basis().moduli(),
            "NTT plan basis must match bounded gadget basis"
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

        distribution.validate();

        let basis = config.layout.full_basis().clone();
        let mut entries = Vec::with_capacity(config.layout.digit_count());

        for digit_index in 0..config.layout.digit_count() {
            let target_residues = basis
                .moduli()
                .iter()
                .copied()
                .enumerate()
                .map(|(limb_index, modulus)| {
                    let secret = project_secret(modulus, secret_coefficients);
                    let limb_plan = config.plan.plan(limb_index);

                    let secret_squared =
                        limb_plan.negacyclic_mul(secret.polynomial(), secret.polynomial());

                    let mut factor = 1_u64;
                    let base_mod = (config.layout.base() % u128::from(modulus.value())) as u64;

                    for _ in 0..digit_index {
                        factor = modulus.mul(factor, base_mod);
                    }

                    secret_squared.scalar_mul(factor)
                })
                .collect();

            let target = RnsPolynomial::from_residues(target_residues);

            entries.push(encrypt_rns_raw_with_distribution_ntt_rng(
                &target,
                config.plaintext_modulus,
                distribution,
                secret_coefficients,
                config.plan,
                rng,
            ));
        }

        Self {
            layout: config.layout,
            entries,
        }
    }

    pub fn layout(&self) -> &BoundedGadgetLayout {
        &self.layout
    }

    pub fn entries(&self) -> &[RnsRlweCiphertext] {
        &self.entries
    }

    pub fn entry(&self, index: usize) -> &RnsRlweCiphertext {
        &self.entries[index]
    }
}

/// NTT-backed bounded-base relinearization.
///
/// If `c2 = sum_j d_j B^j`, and evaluation-key entry `j` decrypts to
/// `B^j s^2 + e_j`, then the output decrypts to
/// `c0 + c1 s + c2 s^2 + sum_j d_j e_j`.
pub fn bounded_rns_relinearize_with_ntt(
    product: &RnsQuadraticCiphertext,
    multiplication_key: &BoundedRnsMultiplicationKey,
    plan: &RnsNttPlan,
) -> RnsRlweCiphertext {
    let layout = multiplication_key.layout();

    assert_eq!(
        product.c0().basis(),
        layout.full_basis(),
        "quadratic ciphertext basis must match bounded multiplication key"
    );

    assert_eq!(
        product.c1().basis(),
        layout.full_basis(),
        "quadratic ciphertext basis must match bounded multiplication key"
    );

    assert_eq!(
        product.c2().basis(),
        layout.full_basis(),
        "quadratic ciphertext basis must match bounded multiplication key"
    );

    assert_eq!(
        plan.moduli(),
        layout.full_basis().moduli(),
        "NTT plan basis must match bounded multiplication-key basis"
    );

    assert_eq!(
        plan.degree(),
        product.c0().degree(),
        "NTT plan degree must match quadratic ciphertext degree"
    );

    let decomposition = layout.decompose(product.c2());

    let mut output_limbs = Vec::with_capacity(layout.full_basis().len());

    for limb_index in 0..layout.full_basis().len() {
        let modulus = layout.full_basis().modulus(limb_index);
        let limb_plan = plan.plan(limb_index);

        let mut b = product.c0().residue(limb_index).clone();
        let mut a = product.c1().residue(limb_index).clone();

        for digit_index in 0..layout.digit_count() {
            let digit = Polynomial::new(
                modulus,
                decomposition
                    .digit(digit_index)
                    .iter()
                    .map(|&value| signed_mod_u64(value, modulus.value()))
                    .collect(),
            );

            let evaluation_key = multiplication_key.entry(digit_index).limb(limb_index);

            b = b.add(&limb_plan.negacyclic_mul(&digit, evaluation_key.b()));
            a = a.add(&limb_plan.negacyclic_mul(&digit, evaluation_key.a()));
        }

        output_limbs.push(RlweCiphertext::new(b, a));
    }

    RnsRlweCiphertext::from_limbs(output_limbs)
}

fn signed_mod_u64(value: i128, modulus: u64) -> u64 {
    value.rem_euclid(i128::from(modulus)) as u64
}

fn project_secret(modulus: crate::ring::Modulus, secret_coefficients: &[i8]) -> SecretKey {
    let coefficients = secret_coefficients
        .iter()
        .map(|&value| match value {
            -1 => modulus.value() - 1,
            0 => 0,
            1 => 1,
            _ => unreachable!("validated ternary secret"),
        })
        .collect();

    SecretKey::from_polynomial(Polynomial::new(modulus, coefficients))
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ckks::research_profile_4096;
    use crate::grafting::{
        decrypt_rns_quadratic_raw, decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_ntt_rng,
        rns_tensor_with_ntt,
    };

    fn centered(value: u128, modulus: u128) -> i128 {
        if value > modulus / 2 {
            value as i128 - modulus as i128
        } else {
            value as i128
        }
    }

    #[test]
    fn bounded_gaussian_relinearization_noise_sweep() {
        let profile = research_profile_4096();
        let chain = profile.modulus_chain();
        let degree = profile.degree();
        let basis = chain.top().clone();
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 1,
            })
            .collect();

        let zero = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &vec![0_u128; degree]);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(0x31B2_0001);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(0x31B2_0002);

        let lhs = encrypt_rns_raw_with_ntt_rng(&zero, 2, 0, &secret, &plan, &mut lhs_rng);

        let rhs = encrypt_rns_raw_with_ntt_rng(&zero, 2, 0, &secret, &plan, &mut rhs_rng);

        let quadratic = rns_tensor_with_ntt(&lhs, &rhs, &plan);

        let expected = decrypt_rns_quadratic_raw(&quadratic, &secret);

        println!("R3_1B_PROFILE={}", profile.name());
        println!("R3_1B_RING_DEGREE={degree}");

        for base_log in [4_u32, 8, 12, 16, 20] {
            let layout = BoundedGadgetLayout::new(basis.clone(), base_log);

            let decomposition = layout.decompose(quadratic.c2());

            let mut key_rng = ChaCha20Rng::seed_from_u64(0x31B2_1000 + u64::from(base_log));

            let key = BoundedRnsMultiplicationKey::generate_with_distribution_ntt_rng(
                BoundedRnsKeygenConfig {
                    plaintext_modulus: 2,
                    layout: layout.clone(),
                    plan: &plan,
                },
                &secret,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                &mut key_rng,
            );

            let actual_ciphertext = bounded_rns_relinearize_with_ntt(&quadratic, &key, &plan);

            let actual = decrypt_rns_raw_with_ntt(&actual_ciphertext, &secret, &plan);

            let delta = actual.sub(&expected);
            let q = delta.composite_modulus();

            let max_composite_noise = delta
                .reconstruct_coefficients()
                .into_iter()
                .map(|value| centered(value, q).unsigned_abs())
                .max()
                .unwrap_or(0);

            let mut max_limb_noise = 0_u128;

            for limb_index in 0..basis.len() {
                let modulus = basis.modulus(limb_index);

                let limb_max = delta
                    .residue(limb_index)
                    .coefficients()
                    .iter()
                    .map(|&value| {
                        let centered = if value > modulus.value() / 2 {
                            i128::from(value) - i128::from(modulus.value())
                        } else {
                            i128::from(value)
                        };

                        centered.unsigned_abs()
                    })
                    .max()
                    .unwrap_or(0);

                max_limb_noise = max_limb_noise.max(limb_max);
            }

            println!(
                "R3_1B_BASE_LOG={base_log} BASE={} DIGITS={} MAX_DIGIT={} MAX_LIMB_NOISE={} MAX_COMPOSITE_NOISE={}",
                layout.base(),
                layout.digit_count(),
                decomposition.maximum_observed_digit_magnitude(),
                max_limb_noise,
                max_composite_noise,
            );
        }

        println!("R3_1B_NOISY_RELINEARIZATION_SWEEP=PASS");
    }
}
