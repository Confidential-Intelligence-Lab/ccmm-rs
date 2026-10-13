use rand::{CryptoRng, RngCore};

use crate::grafting::{
    encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout, BoundedRnsKeygenConfig,
    RnsRlweCiphertext,
};
use crate::ring::{ModulusChain, NttPolynomial, Polynomial, RnsNttPlan, RnsPolynomial};
use crate::rlwe::{ErrorDistribution, RlweCiphertext};

use super::{
    apply_automorphism, apply_rns_automorphism, rotation_exponent_left, RnsCkksCiphertext,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedRnsGaloisKey {
    exponent: usize,
    layout: BoundedGadgetLayout,
    entries: Vec<RnsRlweCiphertext>,
}

impl BoundedRnsGaloisKey {
    pub fn generate_with_distribution_ntt_rng<R>(
        config: BoundedRnsKeygenConfig<'_>,
        secret_coefficients: &[i8],
        exponent: usize,
        distribution: ErrorDistribution,
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        let degree = config.plan.degree();

        assert_eq!(secret_coefficients.len(), degree);
        assert_eq!(config.plan.moduli(), config.layout.full_basis().moduli());
        assert!(secret_coefficients.iter().all(|&x| (-1..=1).contains(&x)));
        assert!(secret_coefficients.iter().any(|&x| x != 0));

        distribution.validate();

        let exponent = exponent % (2 * degree);
        assert_eq!(exponent % 2, 1, "Galois exponent must be odd");

        let modulus = crate::ring::Modulus::new(3);

        let secret_poly = Polynomial::new(
            modulus,
            secret_coefficients
                .iter()
                .map(|&x| match x {
                    -1 => 2,
                    0 => 0,
                    1 => 1,
                    _ => unreachable!(),
                })
                .collect(),
        );

        let transformed = apply_automorphism(&secret_poly, exponent);

        let transformed_secret: Vec<i8> = transformed
            .coefficients()
            .iter()
            .map(|&x| match x {
                0 => 0,
                1 => 1,
                2 => -1,
                _ => unreachable!(),
            })
            .collect();

        let layout = config.layout;
        let basis = layout.full_basis().clone();

        let mut entries = Vec::with_capacity(layout.digit_count());

        let mut gadget_factors = vec![1_u64; basis.len()];

        for _digit_index in 0..layout.digit_count() {
            let residues = basis
                .moduli()
                .iter()
                .copied()
                .enumerate()
                .map(|(limb_index, modulus)| {
                    let q = modulus.value();
                    let factor = gadget_factors[limb_index];

                    Polynomial::new(
                        modulus,
                        transformed_secret
                            .iter()
                            .map(|&x| {
                                let coefficient = match x {
                                    -1 => q - 1,
                                    0 => 0,
                                    1 => 1,
                                    _ => unreachable!(),
                                };

                                modulus.mul(coefficient, factor)
                            })
                            .collect(),
                    )
                })
                .collect();

            for (limb_index, modulus) in basis.moduli().iter().enumerate() {
                let base = (layout.base() % u128::from(modulus.value())) as u64;
                gadget_factors[limb_index] = modulus.mul(gadget_factors[limb_index], base);
            }

            let target = RnsPolynomial::from_residues(residues);

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
            exponent,
            layout,
            entries,
        }
    }

    pub fn exponent(&self) -> usize {
        self.exponent
    }

    pub fn layout(&self) -> &BoundedGadgetLayout {
        &self.layout
    }

    pub fn entries(&self) -> &[RnsRlweCiphertext] {
        &self.entries
    }
}

/// NTT-domain representation of a bounded Gaussian Galois key.
///
/// The prepared key owns its transformed entries. The original
/// coefficient-domain key may be dropped after preparation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedBoundedRnsGaloisKey {
    exponent: usize,
    layout: BoundedGadgetLayout,
    degree: usize,
    entries: Vec<Vec<(crate::ring::NttPolynomial, crate::ring::NttPolynomial)>>,
}

impl PreparedBoundedRnsGaloisKey {
    /// Generates a bounded Gaussian Galois key directly in prepared NTT form.
    ///
    /// Preserves the reference generator's gadget targets and RNG ordering.
    pub fn generate_with_distribution_ntt_rng<R>(
        config: BoundedRnsKeygenConfig<'_>,
        secret_coefficients: &[i8],
        exponent: usize,
        distribution: ErrorDistribution,
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        Self::generate_with_distribution_ntt_rng_impl(
            config,
            secret_coefficients,
            exponent,
            distribution,
            rng,
            None,
        )
    }

    pub fn generate_profiled_with_distribution_ntt_rng<R>(
        config: BoundedRnsKeygenConfig<'_>,
        secret_coefficients: &[i8],
        exponent: usize,
        distribution: ErrorDistribution,
        rng: &mut R,

        profile: &mut crate::grafting::PreparedRlweEncryptionProfile,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        Self::generate_with_distribution_ntt_rng_impl(
            config,
            secret_coefficients,
            exponent,
            distribution,
            rng,
            Some(profile),
        )
    }

    fn generate_with_distribution_ntt_rng_impl<R>(
        config: BoundedRnsKeygenConfig<'_>,
        secret_coefficients: &[i8],
        exponent: usize,
        distribution: ErrorDistribution,
        rng: &mut R,

        mut profile: Option<&mut crate::grafting::PreparedRlweEncryptionProfile>,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        let degree = config.plan.degree();

        assert_eq!(secret_coefficients.len(), degree);
        assert_eq!(config.plan.moduli(), config.layout.full_basis().moduli());
        assert!(secret_coefficients
            .iter()
            .all(|&value| (-1..=1).contains(&value)));
        assert!(secret_coefficients.iter().any(|&value| value != 0));

        distribution.validate();

        let exponent = exponent % (2 * degree);
        assert_eq!(exponent % 2, 1, "Galois exponent must be odd");

        let modulus = crate::ring::Modulus::new(3);

        let secret_poly = Polynomial::new(
            modulus,
            secret_coefficients
                .iter()
                .map(|&x| match x {
                    -1 => 2,
                    0 => 0,
                    1 => 1,
                    _ => unreachable!(),
                })
                .collect(),
        );

        let transformed = apply_automorphism(&secret_poly, exponent);

        let transformed_secret: Vec<i8> = transformed
            .coefficients()
            .iter()
            .map(|&x| match x {
                0 => 0,
                1 => 1,
                2 => -1,
                _ => unreachable!(),
            })
            .collect();

        let layout = config.layout;
        let basis = layout.full_basis().clone();

        // Cache the secret transform once per RNS limb for all gadget digits.
        let secret_ntt: Vec<NttPolynomial> = basis
            .moduli()
            .iter()
            .copied()
            .enumerate()
            .map(|(limb_index, modulus)| {
                let q = modulus.value();
                let secret_poly = Polynomial::new(
                    modulus,
                    secret_coefficients
                        .iter()
                        .map(|&coefficient| match coefficient {
                            -1 => q - 1,
                            0 => 0,
                            1 => 1,
                            _ => unreachable!(),
                        })
                        .collect(),
                );

                NttPolynomial::from_polynomial(config.plan.plan(limb_index), &secret_poly)
            })
            .collect();

        let mut entries = Vec::with_capacity(layout.digit_count());
        let mut gadget_factors = vec![1_u64; basis.len()];

        for _digit_index in 0..layout.digit_count() {
            let residues = basis
                .moduli()
                .iter()
                .copied()
                .enumerate()
                .map(|(limb_index, modulus)| {
                    let q = modulus.value();
                    let factor = gadget_factors[limb_index];

                    Polynomial::new(
                        modulus,
                        transformed_secret
                            .iter()
                            .map(|&x| {
                                let coefficient = match x {
                                    -1 => q - 1,
                                    0 => 0,
                                    1 => 1,
                                    _ => unreachable!(),
                                };

                                modulus.mul(coefficient, factor)
                            })
                            .collect(),
                    )
                })
                .collect();

            for (limb_index, modulus) in basis.moduli().iter().enumerate() {
                let base = (layout.base() % u128::from(modulus.value())) as u64;

                gadget_factors[limb_index] = modulus.mul(gadget_factors[limb_index], base);
            }

            let target = RnsPolynomial::from_residues(residues);

            entries.push(
                crate::grafting::encrypt_rns_raw_prepared_with_cached_secret_ntt_profiled_rng(
                    &target,
                    config.plaintext_modulus,
                    distribution,
                    &secret_ntt,
                    config.plan,
                    rng,
                    profile.as_deref_mut(),
                ),
            );
        }

        Self {
            exponent,
            layout,
            degree,
            entries,
        }
    }

    pub fn new(key: &BoundedRnsGaloisKey, plan: &RnsNttPlan) -> Self {
        assert_eq!(plan.moduli(), key.layout().full_basis().moduli());

        let degree = plan.degree();
        assert_eq!(key.entries().len(), key.layout().digit_count());

        let entries = key
            .entries()
            .iter()
            .map(|entry| {
                assert_eq!(entry.degree(), degree);
                assert_eq!(entry.basis(), key.layout().full_basis());

                entry
                    .limbs()
                    .iter()
                    .enumerate()
                    .map(|(limb_index, limb)| {
                        let limb_plan = plan.plan(limb_index);
                        (
                            crate::ring::NttPolynomial::from_polynomial(limb_plan, limb.b()),
                            crate::ring::NttPolynomial::from_polynomial(limb_plan, limb.a()),
                        )
                    })
                    .collect()
            })
            .collect();

        Self {
            exponent: key.exponent(),
            layout: key.layout().clone(),
            degree,
            entries,
        }
    }

    pub fn exponent(&self) -> usize {
        self.exponent
    }

    pub fn layout(&self) -> &BoundedGadgetLayout {
        &self.layout
    }

    pub fn degree(&self) -> usize {
        self.degree
    }
}

/// Applies a bounded Galois automorphism with a prepared evaluation key.
pub fn apply_bounded_rns_galois_automorphism_with_prepared_ntt(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedBoundedRnsGaloisKey,
    plan: &RnsNttPlan,
) -> RnsRlweCiphertext {
    assert_eq!(ciphertext.basis(), key.layout.full_basis());
    assert_eq!(plan.moduli(), ciphertext.basis().moduli());
    assert_eq!(plan.degree(), ciphertext.degree());
    assert_eq!(key.degree, ciphertext.degree());

    let transformed = apply_rns_automorphism(ciphertext, key.exponent);

    let transformed_a = RnsPolynomial::from_residues(
        transformed
            .limbs()
            .iter()
            .map(|limb| limb.a().clone())
            .collect(),
    );

    let decomposition = key.layout.decompose(&transformed_a);
    let mut output_limbs = Vec::with_capacity(ciphertext.basis().len());

    for limb_index in 0..ciphertext.basis().len() {
        let modulus = ciphertext.basis().modulus(limb_index);
        let limb_plan = plan.plan(limb_index);

        let mut b_ntt = crate::ring::NttPolynomial::from_polynomial(
            limb_plan,
            transformed.limb(limb_index).b(),
        );

        let zero = Polynomial::zero(modulus, plan.degree());
        let mut a_ntt = crate::ring::NttPolynomial::from_polynomial(limb_plan, &zero);

        for digit_index in 0..key.layout.digit_count() {
            let digit = Polynomial::new(
                modulus,
                decomposition
                    .digit(digit_index)
                    .iter()
                    .map(|&value| signed_mod(value, modulus.value()))
                    .collect(),
            );

            let digit_ntt = crate::ring::NttPolynomial::from_polynomial(limb_plan, &digit);

            let (key_b_ntt, key_a_ntt) = &key.entries[digit_index][limb_index];

            b_ntt.pointwise_mul_add_assign(&digit_ntt, key_b_ntt);
            a_ntt.pointwise_mul_add_assign(&digit_ntt, key_a_ntt);
        }

        output_limbs.push(RlweCiphertext::new(
            b_ntt.to_polynomial(limb_plan),
            a_ntt.to_polynomial(limb_plan),
        ));
    }

    RnsRlweCiphertext::from_limbs(output_limbs)
}

/// Phase timings for a prepared bounded Galois rotation.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoundedGaloisRotationProfile {
    pub automorphism_seconds: f64,
    pub decomposition_seconds: f64,
    pub base_forward_seconds: f64,
    pub digit_prepare_seconds: f64,
    pub digit_forward_seconds: f64,
    pub mac_seconds: f64,
    pub inverse_seconds: f64,
}

/// Profiling variant of the prepared bounded Galois automorphism.
///
/// The arithmetic follows the ordinary prepared path exactly.
pub fn apply_bounded_rns_galois_automorphism_with_prepared_ntt_profiled(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedBoundedRnsGaloisKey,
    plan: &RnsNttPlan,
) -> (RnsRlweCiphertext, BoundedGaloisRotationProfile) {
    use std::time::Instant;

    assert_eq!(ciphertext.basis(), key.layout.full_basis());
    assert_eq!(plan.moduli(), ciphertext.basis().moduli());
    assert_eq!(plan.degree(), ciphertext.degree());
    assert_eq!(key.degree, ciphertext.degree());

    let mut profile = BoundedGaloisRotationProfile::default();

    let start = Instant::now();
    let transformed = apply_rns_automorphism(ciphertext, key.exponent);
    profile.automorphism_seconds = start.elapsed().as_secs_f64();

    let transformed_a = RnsPolynomial::from_residues(
        transformed
            .limbs()
            .iter()
            .map(|limb| limb.a().clone())
            .collect(),
    );

    let start = Instant::now();
    let decomposition = key.layout.decompose(&transformed_a);
    profile.decomposition_seconds = start.elapsed().as_secs_f64();

    let mut output_limbs = Vec::with_capacity(ciphertext.basis().len());

    for limb_index in 0..ciphertext.basis().len() {
        let modulus = ciphertext.basis().modulus(limb_index);
        let limb_plan = plan.plan(limb_index);

        let start = Instant::now();
        let mut b_ntt = crate::ring::NttPolynomial::from_polynomial(
            limb_plan,
            transformed.limb(limb_index).b(),
        );
        let zero = Polynomial::zero(modulus, plan.degree());
        let mut a_ntt = crate::ring::NttPolynomial::from_polynomial(limb_plan, &zero);
        profile.base_forward_seconds += start.elapsed().as_secs_f64();

        for digit_index in 0..key.layout.digit_count() {
            let start = Instant::now();
            let digit = Polynomial::new(
                modulus,
                decomposition
                    .digit(digit_index)
                    .iter()
                    .map(|&value| signed_mod(value, modulus.value()))
                    .collect(),
            );
            profile.digit_prepare_seconds += start.elapsed().as_secs_f64();

            let start = Instant::now();
            let digit_ntt = crate::ring::NttPolynomial::from_polynomial(limb_plan, &digit);
            profile.digit_forward_seconds += start.elapsed().as_secs_f64();

            let (key_b_ntt, key_a_ntt) = &key.entries[digit_index][limb_index];

            let start = Instant::now();
            b_ntt.pointwise_mul_add_assign(&digit_ntt, key_b_ntt);
            a_ntt.pointwise_mul_add_assign(&digit_ntt, key_a_ntt);
            profile.mac_seconds += start.elapsed().as_secs_f64();
        }

        let start = Instant::now();
        output_limbs.push(RlweCiphertext::new(
            b_ntt.to_polynomial(limb_plan),
            a_ntt.to_polynomial(limb_plan),
        ));
        profile.inverse_seconds += start.elapsed().as_secs_f64();
    }

    (RnsRlweCiphertext::from_limbs(output_limbs), profile)
}

/// Applies a prepared bounded Galois key using prepared NTT execution.
///
/// The evaluation key, decomposition, and key-switch arithmetic are
/// unchanged. Only the NTT execution path is specialized.
pub fn apply_bounded_rns_galois_automorphism_with_prepared_execution(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedBoundedRnsGaloisKey,
    plan: &RnsNttPlan,
    prepared_plan: &crate::ring::PreparedRnsNttPlan,
) -> RnsRlweCiphertext {
    assert_eq!(ciphertext.basis(), key.layout.full_basis());
    assert_eq!(plan.moduli(), ciphertext.basis().moduli());
    assert_eq!(plan.degree(), ciphertext.degree());
    assert_eq!(key.degree, ciphertext.degree());
    assert_eq!(prepared_plan.moduli(), plan.moduli());
    assert_eq!(prepared_plan.degree(), plan.degree());

    let transformed = apply_rns_automorphism(ciphertext, key.exponent);

    let transformed_a = RnsPolynomial::from_residues(
        transformed
            .limbs()
            .iter()
            .map(|limb| limb.a().clone())
            .collect(),
    );

    let decomposition = key.layout.decompose(&transformed_a);
    let mut output_limbs = Vec::with_capacity(ciphertext.basis().len());

    for limb_index in 0..ciphertext.basis().len() {
        let modulus = ciphertext.basis().modulus(limb_index);
        let limb_plan = plan.plan(limb_index);
        let prepared_limb_plan = prepared_plan.plan(limb_index);

        let mut b_ntt = crate::ring::NttPolynomial::from_prepared_values(
            prepared_limb_plan,
            prepared_limb_plan.forward(transformed.limb(limb_index).b()),
        );

        let mut a_ntt = crate::ring::NttPolynomial::zero(limb_plan);

        for digit_index in 0..key.layout.digit_count() {
            let digit = Polynomial::new(
                modulus,
                decomposition
                    .digit(digit_index)
                    .iter()
                    .map(|&value| signed_mod(value, modulus.value()))
                    .collect(),
            );

            let digit_ntt = crate::ring::NttPolynomial::from_prepared_values(
                prepared_limb_plan,
                prepared_limb_plan.forward(&digit),
            );

            let (key_b_ntt, key_a_ntt) = &key.entries[digit_index][limb_index];

            b_ntt.pointwise_mul_add_assign(&digit_ntt, key_b_ntt);
            a_ntt.pointwise_mul_add_assign(&digit_ntt, key_a_ntt);
        }

        output_limbs.push(RlweCiphertext::new(
            prepared_limb_plan.inverse(b_ntt.values()),
            prepared_limb_plan.inverse(a_ntt.values()),
        ));
    }

    RnsRlweCiphertext::from_limbs(output_limbs)
}

/// Rotates CKKS slots using a prepared bounded Gaussian Galois key.
pub fn rotate_left_bounded_rns_ckks_with_prepared_ntt(
    ciphertext: &RnsCkksCiphertext,
    steps: usize,
    key: &PreparedBoundedRnsGaloisKey,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertext {
    ciphertext.assert_matches_chain(chain);

    let expected = rotation_exponent_left(ciphertext.rlwe().degree(), steps);
    assert_eq!(
        key.exponent(),
        expected,
        "prepared bounded Galois key exponent does not match rotation"
    );

    let rotated =
        apply_bounded_rns_galois_automorphism_with_prepared_ntt(ciphertext.rlwe(), key, plan);

    RnsCkksCiphertext::new(rotated, ciphertext.state().clone(), chain)
}

/// Rotates CKKS slots with a Gaussian-noisy bounded Galois key and
/// reusable prepared NTT execution plan.
fn rotate_bounded_rns_ckks_with_prepared_execution(
    ciphertext: &RnsCkksCiphertext,
    expected_exponent: usize,
    key: &PreparedBoundedRnsGaloisKey,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
    prepared_plan: &crate::ring::PreparedRnsNttPlan,
) -> RnsCkksCiphertext {
    ciphertext.assert_matches_chain(chain);

    assert_eq!(
        key.exponent(),
        expected_exponent,
        "prepared bounded Galois key exponent does not match rotation"
    );

    let rotated = apply_bounded_rns_galois_automorphism_with_prepared_execution(
        ciphertext.rlwe(),
        key,
        plan,
        prepared_plan,
    );

    RnsCkksCiphertext::new(rotated, ciphertext.state().clone(), chain)
}

/// Accelerated left CKKS rotation with Gaussian-noisy bounded key switching.
pub fn rotate_left_bounded_rns_ckks_with_prepared_execution(
    ciphertext: &RnsCkksCiphertext,
    steps: usize,
    key: &PreparedBoundedRnsGaloisKey,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
    prepared_plan: &crate::ring::PreparedRnsNttPlan,
) -> RnsCkksCiphertext {
    let exponent = rotation_exponent_left(ciphertext.rlwe().degree(), steps);
    rotate_bounded_rns_ckks_with_prepared_execution(
        ciphertext,
        exponent,
        key,
        chain,
        plan,
        prepared_plan,
    )
}

/// Accelerated right CKKS rotation with Gaussian-noisy bounded key switching.
pub fn rotate_right_bounded_rns_ckks_with_prepared_execution(
    ciphertext: &RnsCkksCiphertext,
    steps: usize,
    key: &PreparedBoundedRnsGaloisKey,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
    prepared_plan: &crate::ring::PreparedRnsNttPlan,
) -> RnsCkksCiphertext {
    let exponent = crate::ckks::rotation_exponent_right(ciphertext.rlwe().degree(), steps);
    rotate_bounded_rns_ckks_with_prepared_execution(
        ciphertext,
        exponent,
        key,
        chain,
        plan,
        prepared_plan,
    )
}

fn signed_mod(value: i128, modulus: u64) -> u64 {
    value.rem_euclid(i128::from(modulus)) as u64
}

pub fn apply_bounded_rns_galois_automorphism_with_ntt(
    ciphertext: &RnsRlweCiphertext,
    key: &BoundedRnsGaloisKey,
    plan: &RnsNttPlan,
) -> RnsRlweCiphertext {
    assert_eq!(ciphertext.basis(), key.layout.full_basis());
    assert_eq!(plan.moduli(), ciphertext.basis().moduli());
    assert_eq!(plan.degree(), ciphertext.degree());

    let transformed = apply_rns_automorphism(ciphertext, key.exponent);

    let transformed_a = RnsPolynomial::from_residues(
        (0..ciphertext.basis().len())
            .map(|limb_index| transformed.limb(limb_index).a().clone())
            .collect(),
    );

    let decomposition = key.layout.decompose(&transformed_a);

    let mut output_limbs = Vec::with_capacity(ciphertext.basis().len());

    for limb_index in 0..ciphertext.basis().len() {
        let modulus = ciphertext.basis().modulus(limb_index);
        let limb_plan = plan.plan(limb_index);

        let mut b = transformed.limb(limb_index).b().clone();
        let mut a = Polynomial::zero(modulus, plan.degree());

        for digit_index in 0..key.layout.digit_count() {
            let digit = Polynomial::new(
                modulus,
                decomposition
                    .digit(digit_index)
                    .iter()
                    .map(|&value| signed_mod(value, modulus.value()))
                    .collect(),
            );

            let evaluation_key = key.entries[digit_index].limb(limb_index);

            b = b.add(&limb_plan.negacyclic_mul(&digit, evaluation_key.b()));
            a = a.add(&limb_plan.negacyclic_mul(&digit, evaluation_key.a()));
        }

        output_limbs.push(RlweCiphertext::new(b, a));
    }

    RnsRlweCiphertext::from_limbs(output_limbs)
}

pub fn rotate_left_bounded_rns_ckks_with_ntt(
    ciphertext: &RnsCkksCiphertext,
    steps: usize,
    key: &BoundedRnsGaloisKey,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertext {
    ciphertext.assert_matches_chain(chain);

    let expected = rotation_exponent_left(ciphertext.rlwe().degree(), steps);

    assert_eq!(
        key.exponent(),
        expected,
        "bounded Galois key exponent does not match rotation"
    );

    let rotated = apply_bounded_rns_galois_automorphism_with_ntt(ciphertext.rlwe(), key, plan);

    RnsCkksCiphertext::new(rotated, ciphertext.state().clone(), chain)
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ckks::research_profile_4096;
    use crate::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_ntt_rng};

    #[test]
    fn r15_g13b_direct_prepared_galois_matches_reference_exactly() {
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
                _ => 0,
            })
            .collect();

        let exponent = rotation_exponent_left(degree, 32);
        let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };

        let mut reference_rng = ChaCha20Rng::seed_from_u64(0x1531_1302);

        let mut direct_rng = ChaCha20Rng::seed_from_u64(0x1531_1302);

        let reference_key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
            BoundedRnsKeygenConfig {
                plaintext_modulus: 2,
                layout: BoundedGadgetLayout::new(basis.clone(), 8),
                plan: &plan,
            },
            &secret,
            exponent,
            distribution,
            &mut reference_rng,
        );

        let reference = PreparedBoundedRnsGaloisKey::new(&reference_key, &plan);

        let direct = PreparedBoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
            BoundedRnsKeygenConfig {
                plaintext_modulus: 2,
                layout: BoundedGadgetLayout::new(basis, 8),
                plan: &plan,
            },
            &secret,
            exponent,
            distribution,
            &mut direct_rng,
        );

        assert_eq!(
            direct, reference,
            "direct prepared Galois key must match reference exactly"
        );

        println!("R15_G13B_DIRECT_PREPARED_EXACT_EQUIVALENCE=PASS");
    }

    #[test]
    fn r22_4c_prepared_bounded_galois_matches_reference_exactly() {
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
                _ => 0,
            })
            .collect();

        let plaintext = RnsPolynomial::from_residues(
            basis
                .moduli()
                .iter()
                .copied()
                .map(|modulus| {
                    Polynomial::new(
                        modulus,
                        (0..degree)
                            .map(|index| (index as u64 + 17) % modulus.value())
                            .collect(),
                    )
                })
                .collect(),
        );

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x224C_0001);
        let ciphertext = encrypt_rns_raw_with_distribution_ntt_rng(
            &plaintext,
            2,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            &secret,
            &plan,
            &mut encryption_rng,
        );

        let steps = 32;
        let exponent = rotation_exponent_left(degree, steps);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x224C_0002);
        let key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
            BoundedRnsKeygenConfig {
                plaintext_modulus: 2,
                layout: BoundedGadgetLayout::new(basis, 8),
                plan: &plan,
            },
            &secret,
            exponent,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            &mut key_rng,
        );

        let prepared = PreparedBoundedRnsGaloisKey::new(&key, &plan);

        let reference = apply_bounded_rns_galois_automorphism_with_ntt(&ciphertext, &key, &plan);

        let optimized =
            apply_bounded_rns_galois_automorphism_with_prepared_ntt(&ciphertext, &prepared, &plan);

        assert_eq!(
            optimized, reference,
            "prepared bounded Galois rotation must match reference ciphertext exactly"
        );

        println!(
            "R22_4C_EXACT_EQUIVALENCE=PASS degree={} limbs={} digits={} rotation={}",
            degree,
            chain.top().len(),
            prepared.layout().digit_count(),
            steps,
        );

        let prepared_plan = crate::ring::PreparedRnsNttPlan::new(&plan);

        let accelerated = apply_bounded_rns_galois_automorphism_with_prepared_execution(
            &ciphertext,
            &prepared,
            &plan,
            &prepared_plan,
        );

        assert_eq!(
            accelerated, reference,
            "prepared NTT execution must match reference ciphertext exactly"
        );

        assert_eq!(
            accelerated, optimized,
            "prepared NTT execution must match prepared-key rotation exactly"
        );

        // R23.2B.1: Validate the CKKS wrapper against the raw-RLWE
        // reference without changing the underlying ciphertext.
        let ckks_input = RnsCkksCiphertext::new(
            ciphertext.clone(),
            crate::ckks::CkksChainState::top(&chain, 2.0_f64.powi(35)),
            &chain,
        );

        let ckks_left = rotate_left_bounded_rns_ckks_with_prepared_execution(
            &ckks_input,
            steps,
            &prepared,
            &chain,
            &plan,
            &prepared_plan,
        );

        assert_eq!(
            ckks_left.rlwe(),
            &reference,
            "left CKKS bounded wrapper must match raw-RLWE reference"
        );

        assert_eq!(ckks_left.level(), ckks_input.level());
        assert_eq!(ckks_left.scale(), ckks_input.scale());

        println!("R23_2B_LEFT_WRAPPER=PASS");

        let right_exponent = crate::ckks::rotation_exponent_right(degree, steps);

        let mut right_rng = ChaCha20Rng::seed_from_u64(0x232B_0002);

        let right_key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
            BoundedRnsKeygenConfig {
                plaintext_modulus: 2,
                layout: BoundedGadgetLayout::new(chain.top().clone(), 8),
                plan: &plan,
            },
            &secret,
            right_exponent,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            &mut right_rng,
        );

        let prepared_right = PreparedBoundedRnsGaloisKey::new(&right_key, &plan);

        let right_reference =
            apply_bounded_rns_galois_automorphism_with_ntt(&ciphertext, &right_key, &plan);

        let ckks_right = rotate_right_bounded_rns_ckks_with_prepared_execution(
            &ckks_input,
            steps,
            &prepared_right,
            &chain,
            &plan,
            &prepared_plan,
        );

        assert_eq!(
            ckks_right.rlwe(),
            &right_reference,
            "right CKKS bounded wrapper must match raw-RLWE reference"
        );

        assert_eq!(ckks_right.level(), ckks_input.level());
        assert_eq!(ckks_right.scale(), ckks_input.scale());

        println!("R23_2B_RIGHT_WRAPPER=PASS");

        println!(
            "R22_6B_EXACT_EQUIVALENCE=PASS degree={} limbs={} digits={} rotation={}",
            degree,
            chain.top().len(),
            prepared.layout().digit_count(),
            steps
        );
    }

    #[test]
    fn r22_3b_exact_galois_noise_identity() {
        let profile = research_profile_4096();
        let chain = profile.modulus_chain();
        let degree = profile.degree();
        let basis = chain.top().clone();
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        let secret: Vec<i8> = (0..degree)
            .map(|i| match i % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 1,
            })
            .collect();

        let zero = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &vec![0_u128; degree]);

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x223B_0001);
        let ciphertext =
            encrypt_rns_raw_with_ntt_rng(&zero, 2, 0, &secret, &plan, &mut encryption_rng);

        let exponent = rotation_exponent_left(degree, 32);

        let secret_poly = Polynomial::new(
            crate::ring::Modulus::new(3),
            secret
                .iter()
                .map(|&x| match x {
                    -1 => 2,
                    0 => 0,
                    1 => 1,
                    _ => unreachable!(),
                })
                .collect(),
        );

        let transformed_secret_poly = apply_automorphism(&secret_poly, exponent);

        let transformed_secret: Vec<i8> = transformed_secret_poly
            .coefficients()
            .iter()
            .map(|&x| match x {
                0 => 0,
                1 => 1,
                2 => -1,
                _ => unreachable!(),
            })
            .collect();

        let transformed = apply_rns_automorphism(&ciphertext, exponent);

        let transformed_a = RnsPolynomial::from_residues(
            (0..basis.len())
                .map(|i| transformed.limb(i).a().clone())
                .collect(),
        );

        let expected = decrypt_rns_raw_with_ntt(&transformed, &transformed_secret, &plan);

        for base_log in [4_u32, 8, 12] {
            let layout = BoundedGadgetLayout::new(basis.clone(), base_log);
            let digits = layout.decompose(&transformed_a);

            let mut key_rng = ChaCha20Rng::seed_from_u64(0x223B_1000 + u64::from(base_log));

            let key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedRnsKeygenConfig {
                    plaintext_modulus: 2,
                    layout: layout.clone(),
                    plan: &plan,
                },
                &secret,
                exponent,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                &mut key_rng,
            );

            let output = apply_bounded_rns_galois_automorphism_with_ntt(&ciphertext, &key, &plan);

            let actual = decrypt_rns_raw_with_ntt(&output, &secret, &plan);

            let observed = actual.sub(&expected);

            let mut predicted_limbs = Vec::with_capacity(basis.len());

            for limb_index in 0..basis.len() {
                let modulus = basis.modulus(limb_index);
                let limb_plan = plan.plan(limb_index);

                let mut predicted = Polynomial::zero(modulus, degree);

                let base = (layout.base() % u128::from(modulus.value())) as u64;
                let mut factor = 1_u64;

                for digit_index in 0..layout.digit_count() {
                    let decrypted_key =
                        decrypt_rns_raw_with_ntt(&key.entries()[digit_index], &secret, &plan);

                    let target = Polynomial::new(
                        modulus,
                        transformed_secret
                            .iter()
                            .map(|&x| {
                                let signed = i128::from(x);
                                let residue = signed.rem_euclid(i128::from(modulus.value())) as u64;
                                modulus.mul(residue, factor)
                            })
                            .collect(),
                    );

                    let key_error = decrypted_key.residue(limb_index).sub(&target);

                    let digit = Polynomial::new(
                        modulus,
                        digits
                            .digit(digit_index)
                            .iter()
                            .map(|&value| signed_mod(value, modulus.value()))
                            .collect(),
                    );

                    predicted = predicted.add(&limb_plan.negacyclic_mul(&digit, &key_error));

                    factor = modulus.mul(factor, base);
                }

                predicted_limbs.push(predicted);
            }

            let predicted = RnsPolynomial::from_residues(predicted_limbs);

            assert_eq!(
                observed, predicted,
                "bounded Galois noise identity failed at base_log={base_log}"
            );

            let max_limb_noise = (0..basis.len())
                .flat_map(|limb_index| {
                    let q = basis.modulus(limb_index).value();
                    observed
                        .residue(limb_index)
                        .coefficients()
                        .iter()
                        .map(move |&value| if value > q / 2 { q - value } else { value })
                })
                .max()
                .unwrap_or(0);

            println!(
                "R22_3B_BASE_LOG={} DIGITS={} MAX_LIMB_NOISE={} IDENTITY=PASS",
                base_log,
                layout.digit_count(),
                max_limb_noise,
            );
        }

        println!("R22_3B_EXACT_GALOIS_NOISE_IDENTITY=PASS");
    }

    #[test]
    fn r22_3a_bounded_gaussian_galois_smoke() {
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

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x223A_0001);

        let ciphertext =
            encrypt_rns_raw_with_ntt_rng(&zero, 2, 0, &secret, &plan, &mut encryption_rng);

        let exponent = rotation_exponent_left(degree, 32);

        for base_log in [4_u32, 8, 12] {
            let layout = BoundedGadgetLayout::new(basis.clone(), base_log);

            let mut key_rng = ChaCha20Rng::seed_from_u64(0x223A_1000 + u64::from(base_log));

            let key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedRnsKeygenConfig {
                    plaintext_modulus: 2,
                    layout,
                    plan: &plan,
                },
                &secret,
                exponent,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                &mut key_rng,
            );

            let rotated = apply_bounded_rns_galois_automorphism_with_ntt(&ciphertext, &key, &plan);

            let decrypted = decrypt_rns_raw_with_ntt(&rotated, &secret, &plan);

            assert_eq!(decrypted.degree(), degree);
            assert_eq!(decrypted.basis(), &basis);

            println!(
                "R22_3A_BASE_LOG={} DIGITS={} STATUS=PASS",
                base_log,
                key.entries().len(),
            );
        }
    }
}
