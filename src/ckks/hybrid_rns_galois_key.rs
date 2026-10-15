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

/// Ciphertext-specific canonical CRT preparation reusable across
/// multiple hybrid-QP Galois rotations.
/// Compatibility context for ciphertext-bound hybrid-QP rotations.
///
/// R16.7b.3: direct-RNS base-term construction no longer needs CRT
/// preprocessing. This context retains the originating ciphertext.
pub struct PreparedHybridRotationInput<'a> {
    ciphertext: &'a RnsRlweCiphertext,
}

impl<'a> PreparedHybridRotationInput<'a> {
    pub fn new(ciphertext: &'a RnsRlweCiphertext) -> Self {
        Self { ciphertext }
    }
}

/// Apply an automorphism, switch from the transformed secret back to
/// the original secret, then divide both ciphertext components by P.
pub fn apply_hybrid_rns_galois_automorphism_with_prepared_ntt(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
) -> RnsRlweCiphertext {
    apply_hybrid_rns_galois_automorphism_internal(ciphertext, key, qp_plan, false)
}

/// Hybrid-QP automorphism using a shared canonical CRT input.
pub fn apply_hybrid_rns_galois_automorphism_with_shared_crt(
    prepared: &PreparedHybridRotationInput<'_>,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
) -> RnsRlweCiphertext {
    apply_hybrid_rns_galois_automorphism_internal(prepared.ciphertext, key, qp_plan, false)
}

/// R16.7b.4 validation-only reference using the original CRT base term.
///
/// This entry point is intended for exact arithmetic regression tests.
pub fn apply_hybrid_rns_galois_automorphism_crt_reference(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
) -> RnsRlweCiphertext {
    // R16_7B4_CRT_REFERENCE
    apply_hybrid_rns_galois_automorphism_internal(ciphertext, key, qp_plan, true)
}

// R16_7C2_ROTATION_PROFILE
#[derive(Debug, Clone, Copy, Default)]
pub struct HybridRotationTiming {
    pub automorphism_ms: f64,
    pub decomposition_ms: f64,
    pub base_term_ms: f64,
    pub key_switch_ntt_ms: f64,
    pub modulus_down_ms: f64,
    // R16_7C3A_SUBKERNEL_PROFILE
    pub ntt_initialization_ms: f64,
    pub digit_construction_ms: f64,
    pub digit_forward_ntt_ms: f64,
    pub pointwise_mac_ms: f64,
    pub inverse_ntt_ms: f64,
}

pub fn profile_hybrid_rns_galois_rotation(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
) -> (RnsRlweCiphertext, HybridRotationTiming) {
    let mut timing = HybridRotationTiming::default();

    let result = apply_hybrid_rns_galois_automorphism_internal_profiled(
        ciphertext,
        key,
        qp_plan,
        false,
        Some(&mut timing),
        None,
    );

    (result, timing)
}

fn apply_hybrid_rns_galois_automorphism_internal(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
    use_crt_reference: bool,
) -> RnsRlweCiphertext {
    apply_hybrid_rns_galois_automorphism_internal_profiled(
        ciphertext,
        key,
        qp_plan,
        use_crt_reference,
        None,
        None,
    )
}

/// Applies hybrid-QP rotation with prepared digit-forward NTT execution.
///
/// The prepared plan must correspond to the supplied canonical QP plan.
/// The canonical rotation entry point remains unchanged.
pub fn apply_hybrid_rns_galois_automorphism_prepared_qp(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
    prepared_qp: &crate::ring::PreparedRnsNttPlan,
) -> RnsRlweCiphertext {
    apply_hybrid_rns_galois_automorphism_internal_profiled(
        ciphertext,
        key,
        qp_plan,
        false,
        None,
        Some(prepared_qp),
    )
}

fn apply_hybrid_rns_galois_automorphism_internal_profiled(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
    use_crt_reference: bool,
    mut timing: Option<&mut HybridRotationTiming>,
    prepared_qp: Option<&crate::ring::PreparedRnsNttPlan>,
) -> RnsRlweCiphertext {
    let q_basis = key.layout.full_basis();

    assert_eq!(ciphertext.basis(), q_basis);
    assert_eq!(ciphertext.degree(), key.degree);
    assert_eq!(qp_plan.degree(), key.degree);

    let mut expected_moduli = q_basis.moduli().to_vec();
    expected_moduli.extend_from_slice(key.auxiliary_basis.moduli());
    assert_eq!(qp_plan.moduli(), expected_moduli.as_slice());

    let started = std::time::Instant::now();
    // R16_7C3B2_PREPARED_QP_ROTATION
    if let Some(prepared) = prepared_qp {
        assert_eq!(prepared.degree(), qp_plan.degree());
        assert_eq!(prepared.moduli(), qp_plan.moduli());
    }

    let transformed = apply_rns_automorphism(ciphertext, key.exponent);
    if let Some(t) = timing.as_deref_mut() {
        t.automorphism_ms = started.elapsed().as_secs_f64() * 1000.0;
    }

    let transformed_a = RnsPolynomial::from_residues(
        transformed
            .limbs()
            .iter()
            .map(|limb| limb.a().clone())
            .collect(),
    );

    let started = std::time::Instant::now();
    let decomposition = key.layout.decompose(&transformed_a);
    if let Some(t) = timing.as_deref_mut() {
        t.decomposition_ms = started.elapsed().as_secs_f64() * 1000.0;
    }

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

    let started = std::time::Instant::now();
    let base_b = if use_crt_reference {
        // Original hybrid-QP CRT construction, retained independently.
        let canonical_b = crate::ring::reconstruct_coefficients_big(&transformed_b);
        let p = crate::ring::composite_modulus_big(&key.auxiliary_basis);

        crate::ring::rns_from_big_coefficients(
            expected_moduli.clone(),
            &canonical_b
                .iter()
                .map(|coefficient| coefficient * &p)
                .collect::<Vec<_>>(),
        )
    } else {
        // R16.7b.2: construct P*b' directly in the QP residue basis.
        //
        // Q limbs: (P*b') mod q_i = (P mod q_i)*(b' mod q_i).
        // P limbs: zero, because every auxiliary modulus divides P.
        let mut base_b_residues = Vec::with_capacity(expected_moduli.len());

        for (limb_index, &modulus) in q_basis.moduli().iter().enumerate() {
            let p_factor = key
                .auxiliary_basis
                .moduli()
                .iter()
                .fold(1_u64, |product, auxiliary| {
                    modulus.mul(product, auxiliary.value() % modulus.value())
                });

            let coefficients = transformed_b
                .residue(limb_index)
                .coefficients()
                .iter()
                .map(|&coefficient| modulus.mul(coefficient, p_factor))
                .collect();

            base_b_residues.push(Polynomial::new(modulus, coefficients));
        }

        for &modulus in key.auxiliary_basis.moduli() {
            base_b_residues.push(Polynomial::zero(modulus, key.degree));
        }

        RnsPolynomial::from_residues(base_b_residues)
    };

    if let Some(t) = timing.as_deref_mut() {
        t.base_term_ms = started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = std::time::Instant::now();
    let mut qp_b_limbs = Vec::with_capacity(expected_moduli.len());
    let mut qp_a_limbs = Vec::with_capacity(expected_moduli.len());

    for (limb_index, &modulus) in expected_moduli.iter().enumerate() {
        let plan = qp_plan.plan(limb_index);

        let sub_started = std::time::Instant::now();
        let mut b_ntt = NttPolynomial::from_polynomial(plan, base_b.residue(limb_index));
        let mut a_ntt =
            NttPolynomial::from_polynomial(plan, &Polynomial::zero(modulus, key.degree));
        if let Some(t) = timing.as_deref_mut() {
            t.ntt_initialization_ms += sub_started.elapsed().as_secs_f64() * 1000.0;
        }

        for digit_index in 0..key.layout.digit_count() {
            let sub_started = std::time::Instant::now();
            let digit = Polynomial::new(
                modulus,
                decomposition
                    .digit(digit_index)
                    .iter()
                    .map(|&value| signed_mod(value, modulus.value()))
                    .collect(),
            );

            if let Some(t) = timing.as_deref_mut() {
                t.digit_construction_ms += sub_started.elapsed().as_secs_f64() * 1000.0;
            }

            let sub_started = std::time::Instant::now();
            let digit_ntt = if let Some(prepared) = prepared_qp {
                let prepared_plan = prepared.plan(limb_index);
                NttPolynomial::from_prepared_values(prepared_plan, prepared_plan.forward(&digit))
            } else {
                NttPolynomial::from_polynomial(plan, &digit)
            };
            if let Some(t) = timing.as_deref_mut() {
                t.digit_forward_ntt_ms += sub_started.elapsed().as_secs_f64() * 1000.0;
            }
            let (key_b, key_a) = &key.entries[digit_index][limb_index];

            let sub_started = std::time::Instant::now();
            b_ntt.pointwise_mul_add_assign(&digit_ntt, key_b);
            a_ntt.pointwise_mul_add_assign(&digit_ntt, key_a);
            if let Some(t) = timing.as_deref_mut() {
                t.pointwise_mac_ms += sub_started.elapsed().as_secs_f64() * 1000.0;
            }
        }

        let sub_started = std::time::Instant::now();
        qp_b_limbs.push(b_ntt.to_polynomial(plan));
        qp_a_limbs.push(a_ntt.to_polynomial(plan));
        if let Some(t) = timing.as_deref_mut() {
            t.inverse_ntt_ms += sub_started.elapsed().as_secs_f64() * 1000.0;
        }
    }

    if let Some(t) = timing.as_deref_mut() {
        t.key_switch_ntt_ms = started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = std::time::Instant::now();
    let qp_b = RnsPolynomial::from_residues(qp_b_limbs);
    let qp_a = RnsPolynomial::from_residues(qp_a_limbs);

    let output_b = auxiliary_mod_down_rns_candidate(&qp_b, q_basis, &key.auxiliary_basis);
    let output_a = auxiliary_mod_down_rns_candidate(&qp_a, q_basis, &key.auxiliary_basis);

    let result = RnsRlweCiphertext::from_limbs(
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
    );
    if let Some(t) = timing.as_mut() {
        t.modulus_down_ms = started.elapsed().as_secs_f64() * 1000.0;
    }
    result
}
