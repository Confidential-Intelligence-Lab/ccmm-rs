//! R16.5.18d.1: auxiliary-P prepared Galois switching.
//!
//! Correctness-oriented implementation using exact wide-CRT modulus-down.
//! Not a production-performance or security-parameter validation.

use rand::{CryptoRng, RngCore};

use crate::grafting::{
    encrypt_rns_raw_prepared_with_distribution_ntt_rng, BoundedGadgetLayout, RnsRlweCiphertext,
};
use crate::ring::{
    auxiliary_mod_down_rns_candidate, ModulusBasis, NttPolynomial, Polynomial, RnsNttPlan,
    RnsPolynomial,
};
use crate::rlwe::{ErrorDistribution, RlweCiphertext};

use super::{apply_automorphism, apply_rns_automorphism};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedHybridRnsGaloisKey {
    exponent: usize,
    layout: BoundedGadgetLayout,
    auxiliary_basis: ModulusBasis,
    degree: usize,
    entries: Vec<Vec<(NttPolynomial, NttPolynomial)>>,
}

fn transformed_ternary_secret(secret: &[i8], exponent: usize) -> Vec<i8> {
    let poly = Polynomial::new(
        crate::ring::Modulus::new(3),
        secret
            .iter()
            .map(|&x| match x {
                -1 => 2,
                0 => 0,
                1 => 1,
                _ => panic!("secret must be ternary"),
            })
            .collect(),
    );

    apply_automorphism(&poly, exponent)
        .coefficients()
        .iter()
        .map(|&x| match x {
            0 => 0,
            1 => 1,
            2 => -1,
            _ => unreachable!(),
        })
        .collect()
}

fn signed_mod(value: i128, modulus: u64) -> u64 {
    value.rem_euclid(i128::from(modulus)) as u64
}

impl PreparedHybridRnsGaloisKey {
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_distribution_ntt_rng<R>(
        layout: BoundedGadgetLayout,
        auxiliary_basis: ModulusBasis,
        secret: &[i8],
        exponent: usize,
        distribution: ErrorDistribution,
        plaintext_modulus: u64,
        qp_plan: &RnsNttPlan,
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        distribution.validate();

        let degree = qp_plan.degree();
        assert_eq!(secret.len(), degree);
        assert!(secret.iter().all(|&x| (-1..=1).contains(&x)));
        assert!(secret.iter().any(|&x| x != 0));
        assert!(!auxiliary_basis.is_empty());

        let exponent = exponent % (2 * degree);
        assert_eq!(exponent % 2, 1, "Galois exponent must be odd");

        let q_basis = layout.full_basis();

        let mut moduli = q_basis.moduli().to_vec();
        moduli.extend_from_slice(auxiliary_basis.moduli());

        assert_eq!(qp_plan.moduli(), moduli.as_slice());

        let transformed = transformed_ternary_secret(secret, exponent);

        // P is represented modulo each QP prime.
        let mut p_factors = vec![1_u64; moduli.len()];
        for (index, modulus) in moduli.iter().enumerate() {
            for auxiliary in auxiliary_basis.moduli() {
                p_factors[index] =
                    modulus.mul(p_factors[index], auxiliary.value() % modulus.value());
            }
        }

        let mut gadget_factors = vec![1_u64; moduli.len()];
        let mut entries = Vec::with_capacity(layout.digit_count());

        for _ in 0..layout.digit_count() {
            let residues = moduli
                .iter()
                .enumerate()
                .map(|(index, &modulus)| {
                    let multiplier = modulus.mul(p_factors[index], gadget_factors[index]);

                    Polynomial::new(
                        modulus,
                        transformed
                            .iter()
                            .map(|&coefficient| {
                                modulus.mul(
                                    signed_mod(i128::from(coefficient), modulus.value()),
                                    multiplier,
                                )
                            })
                            .collect(),
                    )
                })
                .collect();

            let target = RnsPolynomial::from_residues(residues);

            entries.push(encrypt_rns_raw_prepared_with_distribution_ntt_rng(
                &target,
                plaintext_modulus,
                distribution,
                secret,
                qp_plan,
                rng,
            ));

            for (index, modulus) in moduli.iter().enumerate() {
                let base = (layout.base() % u128::from(modulus.value())) as u64;
                gadget_factors[index] = modulus.mul(gadget_factors[index], base);
            }
        }

        Self {
            exponent,
            layout,
            auxiliary_basis,
            degree,
            entries,
        }
    }

    pub fn exponent(&self) -> usize {
        self.exponent
    }

    pub fn digit_count(&self) -> usize {
        self.layout.digit_count()
    }

    pub fn auxiliary_basis(&self) -> &ModulusBasis {
        &self.auxiliary_basis
    }
}

/// Applies a prepared hybrid Galois automorphism while preserving CKKS state.
///
/// Hybrid key switching does not consume a CKKS modulus-chain level.
/// The evaluation key must correspond to the requested automorphism.
pub fn rotate_hybrid_rns_ckks_with_prepared_ntt(
    ciphertext: &crate::ckks::RnsCkksCiphertext,
    exponent: usize,
    key: &PreparedHybridRnsGaloisKey,
    chain: &crate::ring::ModulusChain,
    qp_plan: &RnsNttPlan,
) -> crate::ckks::RnsCkksCiphertext {
    ciphertext.assert_matches_chain(chain);

    let degree = ciphertext.rlwe().degree();

    assert_eq!(degree, key.degree, "hybrid CKKS rotation degree mismatch");

    let exponent = exponent % (2 * degree);

    assert_eq!(
        exponent, key.exponent,
        "hybrid CKKS rotation exponent does not match evaluation key"
    );

    assert_eq!(
        ciphertext.basis(),
        key.layout.full_basis(),
        "hybrid CKKS rotation Q basis mismatch"
    );

    let mut expected_moduli = ciphertext.basis().moduli().to_vec();
    expected_moduli.extend_from_slice(key.auxiliary_basis.moduli());

    assert_eq!(
        qp_plan.degree(),
        degree,
        "hybrid CKKS rotation QP plan degree mismatch"
    );

    assert_eq!(
        qp_plan.moduli(),
        expected_moduli.as_slice(),
        "hybrid CKKS rotation QP plan basis mismatch"
    );

    let rotated =
        apply_hybrid_rns_galois_automorphism_with_prepared_ntt(ciphertext.rlwe(), key, qp_plan);

    crate::ckks::RnsCkksCiphertext::new(rotated, ciphertext.state().clone(), chain)
}

/// Rotates CKKS slots left using a prepared hybrid QP Galois key.
pub fn rotate_left_hybrid_rns_ckks_with_prepared_ntt(
    ciphertext: &crate::ckks::RnsCkksCiphertext,
    steps: usize,
    key: &PreparedHybridRnsGaloisKey,
    chain: &crate::ring::ModulusChain,
    qp_plan: &RnsNttPlan,
) -> crate::ckks::RnsCkksCiphertext {
    let exponent = crate::ckks::rotation_exponent_left(ciphertext.rlwe().degree(), steps);

    rotate_hybrid_rns_ckks_with_prepared_ntt(ciphertext, exponent, key, chain, qp_plan)
}

/// Rotates CKKS slots right using a prepared hybrid QP Galois key.
pub fn rotate_right_hybrid_rns_ckks_with_prepared_ntt(
    ciphertext: &crate::ckks::RnsCkksCiphertext,
    steps: usize,
    key: &PreparedHybridRnsGaloisKey,
    chain: &crate::ring::ModulusChain,
    qp_plan: &RnsNttPlan,
) -> crate::ckks::RnsCkksCiphertext {
    let exponent = crate::ckks::rotation_exponent_right(ciphertext.rlwe().degree(), steps);

    rotate_hybrid_rns_ckks_with_prepared_ntt(ciphertext, exponent, key, chain, qp_plan)
}

/// Apply an automorphism, switch from the transformed secret back to
/// the original secret, then divide both ciphertext components by P.
pub fn apply_hybrid_rns_galois_automorphism_with_prepared_ntt(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
) -> RnsRlweCiphertext {
    let q_basis = key.layout.full_basis();

    assert_eq!(ciphertext.basis(), q_basis);
    assert_eq!(ciphertext.degree(), key.degree);
    assert_eq!(qp_plan.degree(), key.degree);

    let mut expected_moduli = q_basis.moduli().to_vec();
    expected_moduli.extend_from_slice(key.auxiliary_basis.moduli());
    assert_eq!(qp_plan.moduli(), expected_moduli.as_slice());

    let transformed = apply_rns_automorphism(ciphertext, key.exponent);

    let transformed_a = RnsPolynomial::from_residues(
        transformed
            .limbs()
            .iter()
            .map(|limb| limb.a().clone())
            .collect(),
    );

    let decomposition = key.layout.decompose(&transformed_a);

    // P*b' is the base term, including on the auxiliary limbs.
    // The canonical Q lift of b' is sufficient because multiplication
    // by P makes lifts differing by Q identical modulo QP.
    let transformed_b = RnsPolynomial::from_residues(
        transformed
            .limbs()
            .iter()
            .map(|limb| limb.b().clone())
            .collect(),
    );

    let canonical_b = crate::ring::reconstruct_coefficients_big(&transformed_b);
    let p = crate::ring::composite_modulus_big(&key.auxiliary_basis);

    let base_b = crate::ring::rns_from_big_coefficients(
        expected_moduli.clone(),
        &canonical_b.iter().map(|x| x * &p).collect::<Vec<_>>(),
    );

    let mut qp_b_limbs = Vec::with_capacity(expected_moduli.len());
    let mut qp_a_limbs = Vec::with_capacity(expected_moduli.len());

    for (limb_index, &modulus) in expected_moduli.iter().enumerate() {
        let plan = qp_plan.plan(limb_index);

        let mut b_ntt = NttPolynomial::from_polynomial(plan, base_b.residue(limb_index));
        let mut a_ntt =
            NttPolynomial::from_polynomial(plan, &Polynomial::zero(modulus, key.degree));

        for digit_index in 0..key.layout.digit_count() {
            let digit = Polynomial::new(
                modulus,
                decomposition
                    .digit(digit_index)
                    .iter()
                    .map(|&value| signed_mod(value, modulus.value()))
                    .collect(),
            );

            let digit_ntt = NttPolynomial::from_polynomial(plan, &digit);
            let (key_b, key_a) = &key.entries[digit_index][limb_index];

            b_ntt.pointwise_mul_add_assign(&digit_ntt, key_b);
            a_ntt.pointwise_mul_add_assign(&digit_ntt, key_a);
        }

        qp_b_limbs.push(b_ntt.to_polynomial(plan));
        qp_a_limbs.push(a_ntt.to_polynomial(plan));
    }

    let qp_b = RnsPolynomial::from_residues(qp_b_limbs);
    let qp_a = RnsPolynomial::from_residues(qp_a_limbs);

    let output_b = auxiliary_mod_down_rns_candidate(&qp_b, q_basis, &key.auxiliary_basis);
    let output_a = auxiliary_mod_down_rns_candidate(&qp_a, q_basis, &key.auxiliary_basis);

    RnsRlweCiphertext::from_limbs(
        output_b
            .moduli()
            .iter()
            .enumerate()
            .map(|(index, _)| {
                RlweCiphertext::new(
                    output_b.residue(index).clone(),
                    output_a.residue(index).clone(),
                )
            })
            .collect(),
    )
}
