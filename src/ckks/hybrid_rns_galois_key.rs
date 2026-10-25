//! R16.5.18d.1: auxiliary-P prepared Galois switching.
//!
//! Correctness-oriented implementation using exact wide-CRT modulus-down.
//! Not a production-performance or security-parameter validation.

use rand::{CryptoRng, RngCore};

use crate::grafting::{
    encrypt_rns_raw_prepared_with_cached_secret_ntt_rng, BoundedGadgetLayout,
    PreparedCanonicalGaloisSource, RnsRlweCiphertext,
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

        // R35_9E_1_CACHED_SECRET_NTT
        // The secret and QP basis are invariant across gadget digits.
        let secret_ntt: Vec<NttPolynomial> = moduli
            .iter()
            .copied()
            .enumerate()
            .map(|(index, modulus)| {
                let projected = Polynomial::new(
                    modulus,
                    secret
                        .iter()
                        .map(|&coefficient| signed_mod(i128::from(coefficient), modulus.value()))
                        .collect(),
                );
                NttPolynomial::from_polynomial(qp_plan.plan(index), &projected)
            })
            .collect();

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

            entries.push(encrypt_rns_raw_prepared_with_cached_secret_ntt_rng(
                &target,
                plaintext_modulus,
                distribution,
                &secret_ntt,
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

    // R36_8G_2C_1_KEY_PAYLOAD_ACCOUNTING
    /// Logical NTT coefficient payload retained by this prepared key.
    ///
    /// Excludes Vec capacity, allocation metadata, and other structures.
    /// Does not allocate or modify cryptographic state.
    pub fn coefficient_payload_bytes(&self) -> usize {
        self.entries
            .iter()
            .flat_map(|digit| digit.iter())
            .map(|(a, b)| (a.values().len() + b.values().len()) * std::mem::size_of::<u64>())
            .sum()
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
        None,
    );

    (result, timing)
}

// R36_5D1_PREPARED_ROTATION_PROFILE
/// Profiles a hybrid-QP rotation using a prepared QP NTT plan.
///
/// `canonical_source = None` selects legacy gadget decomposition.
/// `Some(source)` selects prepared-canonical decomposition.
pub fn profile_hybrid_rns_galois_rotation_prepared_qp(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
    prepared_qp: &crate::ring::PreparedRnsNttPlan,
    canonical_source: Option<&PreparedCanonicalGaloisSource>,
) -> (RnsRlweCiphertext, HybridRotationTiming) {
    let mut timing = HybridRotationTiming::default();

    let result = apply_hybrid_rns_galois_automorphism_internal_profiled(
        ciphertext,
        key,
        qp_plan,
        false,
        Some(&mut timing),
        Some(prepared_qp),
        canonical_source,
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
        None,
    )
}

// R36_4C_PREPARED_CANONICAL_ROTATION
/// Hybrid-QP rotation using a previously reconstructed canonical source.
pub fn apply_hybrid_rns_galois_automorphism_prepared_canonical_qp(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
    prepared_qp: &crate::ring::PreparedRnsNttPlan,
    canonical_source: &PreparedCanonicalGaloisSource,
) -> RnsRlweCiphertext {
    apply_hybrid_rns_galois_automorphism_internal_profiled_owned(
        ciphertext,
        key,
        qp_plan,
        false,
        None,
        Some(prepared_qp),
        Some(canonical_source),
        // R36_5F6_PRODUCTION_CANONICAL_FORWARD
        true,
        false,
        // R36_5G4_PRODUCTION_PREPARED_INVERSE
        true,
    )
}

// R36_5F4_PREPARED_BUTTERFLY_ROTATION
/// Profiles canonical hybrid-QP rotation using prepared butterfly multiplication.
pub fn profile_hybrid_rns_galois_rotation_prepared_butterfly_qp(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
    prepared_qp: &crate::ring::PreparedRnsNttPlan,
    canonical_source: &PreparedCanonicalGaloisSource,
) -> (RnsRlweCiphertext, HybridRotationTiming) {
    let mut timing = HybridRotationTiming::default();
    let result = apply_hybrid_rns_galois_automorphism_internal_profiled_owned(
        ciphertext,
        key,
        qp_plan,
        false,
        Some(&mut timing),
        Some(prepared_qp),
        Some(canonical_source),
        true,
        true,
        false,
    );
    (result, timing)
}

/// Profiles a canonical hybrid-QP rotation using owned-vector forward NTT.
pub fn profile_hybrid_rns_galois_rotation_owned_forward_qp(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
    prepared_qp: &crate::ring::PreparedRnsNttPlan,
    canonical_source: &PreparedCanonicalGaloisSource,
) -> (RnsRlweCiphertext, HybridRotationTiming) {
    let mut timing = HybridRotationTiming::default();
    let result = apply_hybrid_rns_galois_automorphism_internal_profiled_owned(
        ciphertext,
        key,
        qp_plan,
        false,
        Some(&mut timing),
        Some(prepared_qp),
        Some(canonical_source),
        true,
        false,
        false,
    );
    (result, timing)
}

// R36_5G3_PREPARED_INVERSE_PROFILE
/// Experimental canonical hybrid-QP rotation with prepared forward
/// and inverse NTT arithmetic. Production dispatch is unchanged.
pub fn profile_hybrid_rns_galois_rotation_prepared_inverse_qp(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
    prepared_qp: &crate::ring::PreparedRnsNttPlan,
    canonical_source: &PreparedCanonicalGaloisSource,
) -> (RnsRlweCiphertext, HybridRotationTiming) {
    let mut timing = HybridRotationTiming::default();

    let result = apply_hybrid_rns_galois_automorphism_internal_profiled_owned(
        ciphertext,
        key,
        qp_plan,
        false,
        Some(&mut timing),
        Some(prepared_qp),
        Some(canonical_source),
        true,
        false,
        true,
    );

    (result, timing)
}

fn apply_hybrid_rns_galois_automorphism_internal_profiled(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
    use_crt_reference: bool,
    timing: Option<&mut HybridRotationTiming>,
    prepared_qp: Option<&crate::ring::PreparedRnsNttPlan>,
    canonical_source: Option<&PreparedCanonicalGaloisSource>,
) -> RnsRlweCiphertext {
    apply_hybrid_rns_galois_automorphism_internal_profiled_owned(
        ciphertext,
        key,
        qp_plan,
        use_crt_reference,
        timing,
        prepared_qp,
        canonical_source,
        false,
        false,
        false,
    )
}

// R36_5E3_OWNED_NTT_ROTATION
#[allow(clippy::too_many_arguments)]
fn apply_hybrid_rns_galois_automorphism_internal_profiled_owned(
    ciphertext: &RnsRlweCiphertext,
    key: &PreparedHybridRnsGaloisKey,
    qp_plan: &RnsNttPlan,
    use_crt_reference: bool,
    mut timing: Option<&mut HybridRotationTiming>,
    prepared_qp: Option<&crate::ring::PreparedRnsNttPlan>,
    canonical_source: Option<&PreparedCanonicalGaloisSource>,
    owned_forward_ntt: bool,
    prepared_butterfly: bool,
    prepared_inverse: bool,
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
    let decomposition = if let Some(source) = canonical_source {
        assert_eq!(source.basis(), q_basis, "prepared source basis mismatch");
        assert_eq!(
            source.degree(),
            key.degree,
            "prepared source degree mismatch"
        );
        source.decompose_automorphism(&key.layout, key.exponent)
    } else {
        key.layout.decompose(&transformed_a)
    };
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
            // R36_5E4_DIRECT_OWNED_DIGIT
            let digit_values: Vec<u64> = decomposition
                .digit(digit_index)
                .iter()
                .map(|&value| signed_mod(value, modulus.value()))
                .collect();

            if let Some(t) = timing.as_deref_mut() {
                t.digit_construction_ms += sub_started.elapsed().as_secs_f64() * 1000.0;
            }

            let sub_started = std::time::Instant::now();
            let digit_ntt = if let Some(prepared) = prepared_qp {
                let prepared_plan = prepared.plan(limb_index);
                if prepared_butterfly {
                    NttPolynomial::from_prepared_values(
                        prepared_plan,
                        prepared_plan.forward_owned_prepared_mul(digit_values),
                    )
                } else if owned_forward_ntt {
                    NttPolynomial::from_prepared_values(
                        prepared_plan,
                        prepared_plan.forward_owned(digit_values),
                    )
                } else {
                    let digit = Polynomial::new(modulus, digit_values);
                    NttPolynomial::from_prepared_values(
                        prepared_plan,
                        prepared_plan.forward(&digit),
                    )
                }
            } else {
                let digit = Polynomial::new(modulus, digit_values);
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
        // R36_5G3_PREPARED_INVERSE_DISPATCH
        if prepared_inverse {
            let prepared = prepared_qp
                .expect("prepared inverse requires a prepared QP plan")
                .plan(limb_index);

            qp_b_limbs.push(prepared.inverse_prepared_mul(b_ntt.values()));
            qp_a_limbs.push(prepared.inverse_prepared_mul(a_ntt.values()));
        } else {
            qp_b_limbs.push(b_ntt.to_polynomial(plan));
            qp_a_limbs.push(a_ntt.to_polynomial(plan));
        }
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

#[cfg(test)]
mod r35_9e_exact_key_equivalence {
    // R35_9E_2_EXACT_KEY_EQUIVALENCE
    use super::*;
    use crate::grafting::encrypt_rns_raw_prepared_with_distribution_ntt_rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    // R35_9E_3_FULL_KEY_EQUIVALENCE
    #[allow(clippy::too_many_arguments)]
    fn reference_key<R: RngCore + CryptoRng>(
        layout: BoundedGadgetLayout,
        auxiliary_basis: ModulusBasis,
        secret: &[i8],
        exponent: usize,
        distribution: ErrorDistribution,
        plaintext_modulus: u64,
        qp_plan: &RnsNttPlan,
        rng: &mut R,
    ) -> PreparedHybridRnsGaloisKey {
        let degree = qp_plan.degree();
        let exponent = exponent % (2 * degree);
        let q_basis = layout.full_basis();

        let mut moduli = q_basis.moduli().to_vec();
        moduli.extend_from_slice(auxiliary_basis.moduli());

        assert_eq!(qp_plan.moduli(), moduli.as_slice());

        let transformed = transformed_ternary_secret(secret, exponent);

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

        PreparedHybridRnsGaloisKey {
            exponent,
            layout,
            auxiliary_basis,
            degree,
            entries,
        }
    }

    // R36_4C_EXACT_RLWE_EQUIVALENCE
    #[test]
    fn r36_4c_prepared_canonical_rotation_matches_legacy_rlwe_exactly() {
        use crate::ring::{Modulus, ModulusBasis, PreparedRnsNttPlan};

        const N: usize = 16;

        let secret: [i8; N] = [1, 0, -1, 1, 0, -1, 1, 0, -1, 0, 1, 0, 1, -1, 0, 1];

        let distribution = ErrorDistribution::BoundedUniform { bound: 0 };

        for q_count in [2_usize, 3] {
            let q = ModulusBasis::new(
                [12_289_u64, 40_961, 65_537]
                    .into_iter()
                    .take(q_count)
                    .map(Modulus::new)
                    .collect(),
            );

            let p = ModulusBasis::new(vec![Modulus::new(114_689), Modulus::new(786_433)]);

            let mut qp_moduli = q.moduli().to_vec();
            qp_moduli.extend_from_slice(p.moduli());

            let qp_plan = RnsNttPlan::new(qp_moduli, N);
            let prepared_qp = PreparedRnsNttPlan::new(&qp_plan);

            let message = RnsPolynomial::from_residues(
                q.moduli()
                    .iter()
                    .map(|&modulus| {
                        Polynomial::new(
                            modulus,
                            (0..N)
                                .map(|i| (i as u64 * 37 + 11) % modulus.value())
                                .collect(),
                        )
                    })
                    .collect(),
            );

            let q_plan = RnsNttPlan::new(q.moduli().to_vec(), N);

            let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x364c_1000 + q_count as u64);

            let encrypted_ntt = encrypt_rns_raw_prepared_with_distribution_ntt_rng(
                &message,
                2,
                distribution,
                &secret,
                &q_plan,
                &mut encryption_rng,
            );

            let ciphertext = RnsRlweCiphertext::from_limbs(
                encrypted_ntt
                    .into_iter()
                    .enumerate()
                    .map(|(index, (b_ntt, a_ntt))| {
                        RlweCiphertext::new(
                            b_ntt.to_polynomial(q_plan.plan(index)),
                            a_ntt.to_polynomial(q_plan.plan(index)),
                        )
                    })
                    .collect(),
            );

            let original_a = RnsPolynomial::from_residues(
                ciphertext
                    .limbs()
                    .iter()
                    .map(|limb| limb.a().clone())
                    .collect(),
            );

            let canonical_source = PreparedCanonicalGaloisSource::new(&original_a);

            for exponent in [3_usize, 5, 7, 11, 15] {
                let mut key_rng = ChaCha20Rng::seed_from_u64(
                    0x364c_2000 + q_count as u64 * 100 + exponent as u64,
                );

                let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                    BoundedGadgetLayout::new(q.clone(), 8),
                    p.clone(),
                    &secret,
                    exponent,
                    distribution,
                    2,
                    &qp_plan,
                    &mut key_rng,
                );

                let legacy = apply_hybrid_rns_galois_automorphism_prepared_qp(
                    &ciphertext,
                    &key,
                    &qp_plan,
                    &prepared_qp,
                );

                let optimized = apply_hybrid_rns_galois_automorphism_prepared_canonical_qp(
                    &ciphertext,
                    &key,
                    &qp_plan,
                    &prepared_qp,
                    &canonical_source,
                );

                assert_eq!(
                    legacy, optimized,
                    "R36.4C RLWE mismatch: q_count={q_count}, exponent={exponent}"
                );

                println!(
                    "R36_4C_CASE=PASS q_count={} exponent={} degree={}",
                    q_count, exponent, N,
                );
            }
        }

        println!("R36_4C_EXACT_RLWE_EQUIVALENCE_STATUS=PASS");
    }

    // R36_5D2_PRODUCTION_ROTATION_AB_BENCHMARK
    #[test]
    #[ignore = "opt-in UHD-scale single-key rotation benchmark"]
    fn r36_5d2_production_rotation_ab_benchmark() {
        use crate::ckks::research_profile_65536_for_levels;
        use crate::ring::{ModulusBasis, ModulusChain, PreparedRnsNttPlan};
        use std::time::Instant;

        const N: usize = 65_536;
        const EXPONENT: usize = 3;
        const BASE_LOG: u32 = 32;

        let profile = research_profile_65536_for_levels(13, 3);
        assert_eq!(profile.degree(), N);

        let extended = research_profile_65536_for_levels(15, 3);

        let top = profile.modulus_chain().top().clone();
        let extended_chain = extended.modulus_chain();
        let extended_moduli = extended_chain.top().moduli();

        assert_eq!(top.moduli().len(), 16);
        assert_eq!(extended_moduli.len(), 18);
        assert_eq!(&extended_moduli[..16], top.moduli());

        let p = ModulusBasis::new(extended_moduli[16..18].to_vec());
        let chain = ModulusChain::from_top_basis(top);
        let q = chain.level(3).clone();

        assert_eq!(q.moduli().len(), 13);
        assert_eq!(p.moduli().len(), 2);

        let layout = BoundedGadgetLayout::new(q.clone(), BASE_LOG);
        assert_eq!(layout.digit_count(), 23);

        let mut qp_moduli = q.moduli().to_vec();
        qp_moduli.extend_from_slice(p.moduli());

        let qp_plan = RnsNttPlan::new(qp_moduli, N);
        let prepared_qp = PreparedRnsNttPlan::new(&qp_plan);
        let q_plan = RnsNttPlan::new(q.moduli().to_vec(), N);

        let secret: Vec<i8> = (0..N)
            .map(|i| match i % 3 {
                0 => 1,
                1 => 0,
                _ => -1,
            })
            .collect();

        let distribution = ErrorDistribution::BoundedUniform { bound: 0 };

        let message = RnsPolynomial::from_residues(
            q.moduli()
                .iter()
                .map(|&modulus| {
                    Polynomial::new(
                        modulus,
                        (0..N)
                            .map(|i| (i as u64 * 37 + 11) % modulus.value())
                            .collect(),
                    )
                })
                .collect(),
        );

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x365d_2001);

        let encrypted_ntt = encrypt_rns_raw_prepared_with_distribution_ntt_rng(
            &message,
            2,
            distribution,
            &secret,
            &q_plan,
            &mut encryption_rng,
        );

        let ciphertext = RnsRlweCiphertext::from_limbs(
            encrypted_ntt
                .into_iter()
                .enumerate()
                .map(|(index, (b_ntt, a_ntt))| {
                    RlweCiphertext::new(
                        b_ntt.to_polynomial(q_plan.plan(index)),
                        a_ntt.to_polynomial(q_plan.plan(index)),
                    )
                })
                .collect(),
        );

        let original_a = RnsPolynomial::from_residues(
            ciphertext
                .limbs()
                .iter()
                .map(|limb| limb.a().clone())
                .collect(),
        );

        let started = Instant::now();
        let canonical_source = PreparedCanonicalGaloisSource::new(&original_a);
        let preparation_seconds = started.elapsed().as_secs_f64();

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x365d_2002);

        let started = Instant::now();
        let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
            layout,
            p,
            &secret,
            EXPONENT,
            distribution,
            2,
            &qp_plan,
            &mut key_rng,
        );
        let keygen_seconds = started.elapsed().as_secs_f64();

        println!("R36_5D_KEYGEN_SECONDS={:.6}", keygen_seconds);
        println!(
            "R36_5D_CANONICAL_PREPARATION_SECONDS={:.6}",
            preparation_seconds
        );

        let warmups = 1usize;
        let repeats = 3usize;

        let mut legacy_seconds = Vec::with_capacity(repeats);
        let mut canonical_seconds = Vec::with_capacity(repeats);
        let mut legacy_decomposition_ms = Vec::with_capacity(repeats);
        let mut canonical_decomposition_ms = Vec::with_capacity(repeats);

        for iteration in 0..(warmups + repeats) {
            let canonical_first = iteration % 2 == 1;

            let run = |canonical: bool| {
                let started = Instant::now();

                let (result, timing) = profile_hybrid_rns_galois_rotation_prepared_qp(
                    &ciphertext,
                    &key,
                    &qp_plan,
                    &prepared_qp,
                    if canonical {
                        Some(&canonical_source)
                    } else {
                        None
                    },
                );

                let elapsed = started.elapsed().as_secs_f64();

                (result, timing, elapsed)
            };

            let (legacy, canonical) = if canonical_first {
                let canonical = run(true);
                let legacy = run(false);
                (legacy, canonical)
            } else {
                let legacy = run(false);
                let canonical = run(true);
                (legacy, canonical)
            };

            assert_eq!(
                legacy.0, canonical.0,
                "R36.5D.2 exact RLWE mismatch iteration={iteration}"
            );

            // R36_5E1_KEYSWITCH_SUBKERNEL_PROFILE
            for (mode, timing) in [("legacy", &legacy.1), ("canonical", &canonical.1)] {
                let measured_sum_ms = timing.ntt_initialization_ms
                    + timing.digit_construction_ms
                    + timing.digit_forward_ntt_ms
                    + timing.pointwise_mac_ms
                    + timing.inverse_ntt_ms;

                println!(
                    "R36_5E1_SUBKERNEL iteration={} warmup={} mode={} ntt_initialization_ms={:.6} digit_construction_ms={:.6} forward_ntt_ms={:.6} pointwise_mac_ms={:.6} inverse_ntt_ms={:.6} measured_sum_ms={:.6} key_switch_ntt_ms={:.6}",
                    iteration,
                    iteration < warmups,
                    mode,
                    timing.ntt_initialization_ms,
                    timing.digit_construction_ms,
                    timing.digit_forward_ntt_ms,
                    timing.pointwise_mac_ms,
                    timing.inverse_ntt_ms,
                    measured_sum_ms,
                    timing.key_switch_ntt_ms,
                );
            }

            if iteration >= warmups {
                legacy_seconds.push(legacy.2);
                canonical_seconds.push(canonical.2);
                legacy_decomposition_ms.push(legacy.1.decomposition_ms);
                canonical_decomposition_ms.push(canonical.1.decomposition_ms);
            }

            println!(
                "R36_5D_ITERATION={} warmup={} legacy_seconds={:.6} canonical_seconds={:.6} legacy_decomposition_ms={:.6} canonical_decomposition_ms={:.6} legacy_key_switch_ntt_ms={:.6} canonical_key_switch_ntt_ms={:.6} legacy_modulus_down_ms={:.6} canonical_modulus_down_ms={:.6}",
                iteration,
                iteration < warmups,
                legacy.2,
                canonical.2,
                legacy.1.decomposition_ms,
                canonical.1.decomposition_ms,
                legacy.1.key_switch_ntt_ms,
                canonical.1.key_switch_ntt_ms,
                legacy.1.modulus_down_ms,
                canonical.1.modulus_down_ms,
            );
        }

        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;

        let legacy_mean = mean(&legacy_seconds);
        let canonical_mean = mean(&canonical_seconds);
        let legacy_decomposition_mean = mean(&legacy_decomposition_ms);
        let canonical_decomposition_mean = mean(&canonical_decomposition_ms);

        println!(
            "R36_5D_ROTATION_SPEEDUP={:.6}",
            legacy_mean / canonical_mean
        );
        println!(
            "R36_5D_DECOMPOSITION_SPEEDUP={:.6}",
            legacy_decomposition_mean / canonical_decomposition_mean
        );
        println!(
            "R36_5D_AMORTIZED_ROTATION_SPEEDUP={:.6}",
            legacy_mean * repeats as f64 / (preparation_seconds + canonical_mean * repeats as f64)
        );

        // R36_5E5_OWNED_FORWARD_AB
        let mut baseline_samples = Vec::new();
        let mut optimized_samples = Vec::new();
        let mut baseline_forward_samples = Vec::new();
        let mut optimized_forward_samples = Vec::new();
        let mut baseline_keyswitch_samples = Vec::new();
        let mut optimized_keyswitch_samples = Vec::new();

        for iteration in 0..4 {
            let warmup = iteration == 0;

            let baseline_started = std::time::Instant::now();
            let (baseline, baseline_timing) = profile_hybrid_rns_galois_rotation_prepared_qp(
                &ciphertext,
                &key,
                &qp_plan,
                &prepared_qp,
                Some(&canonical_source),
            );
            let baseline_seconds = baseline_started.elapsed().as_secs_f64();

            let optimized_started = std::time::Instant::now();
            let (optimized, optimized_timing) = profile_hybrid_rns_galois_rotation_owned_forward_qp(
                &ciphertext,
                &key,
                &qp_plan,
                &prepared_qp,
                &canonical_source,
            );
            let optimized_seconds = optimized_started.elapsed().as_secs_f64();

            assert_eq!(
                baseline, optimized,
                "R36.5E.5 exact RLWE mismatch iteration={iteration}"
            );

            println!(
                "R36_5E5_ITERATION={} warmup={} baseline_seconds={:.6} optimized_seconds={:.6} baseline_forward_ms={:.6} optimized_forward_ms={:.6} baseline_keyswitch_ms={:.6} optimized_keyswitch_ms={:.6}",
                iteration,
                warmup,
                baseline_seconds,
                optimized_seconds,
                baseline_timing.digit_forward_ntt_ms,
                optimized_timing.digit_forward_ntt_ms,
                baseline_timing.key_switch_ntt_ms,
                optimized_timing.key_switch_ntt_ms,
            );

            if !warmup {
                baseline_samples.push(baseline_seconds);
                optimized_samples.push(optimized_seconds);
                baseline_forward_samples.push(baseline_timing.digit_forward_ntt_ms);
                optimized_forward_samples.push(optimized_timing.digit_forward_ntt_ms);
                baseline_keyswitch_samples.push(baseline_timing.key_switch_ntt_ms);
                optimized_keyswitch_samples.push(optimized_timing.key_switch_ntt_ms);
            }
        }

        let mean = |samples: &[f64]| -> f64 { samples.iter().sum::<f64>() / samples.len() as f64 };

        println!(
            "R36_5E5_ROTATION_SPEEDUP={:.6}",
            mean(&baseline_samples) / mean(&optimized_samples)
        );
        println!(
            "R36_5E5_FORWARD_NTT_SPEEDUP={:.6}",
            mean(&baseline_forward_samples) / mean(&optimized_forward_samples)
        );
        println!(
            "R36_5E5_KEYSWITCH_SPEEDUP={:.6}",
            mean(&baseline_keyswitch_samples) / mean(&optimized_keyswitch_samples)
        );
        println!("R36_5E5_EXACT_RLWE_STATUS=PASS");

        // R36_5F4_PREPARED_BUTTERFLY_AB
        let mut owned_rotation_samples = Vec::new();
        let mut prepared_rotation_samples = Vec::new();
        let mut owned_forward_samples = Vec::new();
        let mut prepared_forward_samples = Vec::new();
        let mut owned_keyswitch_samples = Vec::new();
        let mut prepared_keyswitch_samples = Vec::new();

        for iteration in 0..5 {
            let warmup = iteration == 0;

            let run_owned = || {
                let started = Instant::now();
                let (result, timing) = profile_hybrid_rns_galois_rotation_owned_forward_qp(
                    &ciphertext,
                    &key,
                    &qp_plan,
                    &prepared_qp,
                    &canonical_source,
                );
                (result, timing, started.elapsed().as_secs_f64())
            };

            let run_prepared = || {
                let started = Instant::now();
                let (result, timing) = profile_hybrid_rns_galois_rotation_prepared_butterfly_qp(
                    &ciphertext,
                    &key,
                    &qp_plan,
                    &prepared_qp,
                    &canonical_source,
                );
                (result, timing, started.elapsed().as_secs_f64())
            };

            // Alternate execution order to reduce systematic order bias.
            let (owned, prepared) = if iteration % 2 == 0 {
                (run_owned(), run_prepared())
            } else {
                let prepared = run_prepared();
                let owned = run_owned();
                (owned, prepared)
            };

            assert_eq!(
                owned.0, prepared.0,
                "R36.5F.4 exact RLWE mismatch iteration={iteration}"
            );

            println!(
                "R36_5F4_ITERATION={} warmup={} owned_seconds={:.6} prepared_seconds={:.6} owned_forward_ms={:.6} prepared_forward_ms={:.6} owned_keyswitch_ms={:.6} prepared_keyswitch_ms={:.6}",
                iteration,
                warmup,
                owned.2,
                prepared.2,
                owned.1.digit_forward_ntt_ms,
                prepared.1.digit_forward_ntt_ms,
                owned.1.key_switch_ntt_ms,
                prepared.1.key_switch_ntt_ms,
            );

            if !warmup {
                owned_rotation_samples.push(owned.2);
                prepared_rotation_samples.push(prepared.2);
                owned_forward_samples.push(owned.1.digit_forward_ntt_ms);
                prepared_forward_samples.push(prepared.1.digit_forward_ntt_ms);
                owned_keyswitch_samples.push(owned.1.key_switch_ntt_ms);
                prepared_keyswitch_samples.push(prepared.1.key_switch_ntt_ms);
            }
        }

        let mean = |samples: &[f64]| -> f64 { samples.iter().sum::<f64>() / samples.len() as f64 };

        println!(
            "R36_5F4_ROTATION_SPEEDUP={:.6}",
            mean(&owned_rotation_samples) / mean(&prepared_rotation_samples)
        );
        println!(
            "R36_5F4_FORWARD_NTT_SPEEDUP={:.6}",
            mean(&owned_forward_samples) / mean(&prepared_forward_samples)
        );
        println!(
            "R36_5F4_KEYSWITCH_SPEEDUP={:.6}",
            mean(&owned_keyswitch_samples) / mean(&prepared_keyswitch_samples)
        );
        println!("R36_5F4_EXACT_RLWE_STATUS=PASS");

        // R36_5G3_INVERSE_AB
        let mut g3_baseline_rotation = Vec::new();
        let mut g3_candidate_rotation = Vec::new();
        let mut g3_baseline_inverse = Vec::new();
        let mut g3_candidate_inverse = Vec::new();
        let mut g3_baseline_keyswitch = Vec::new();
        let mut g3_candidate_keyswitch = Vec::new();

        for iteration in 0..5 {
            let warmup = iteration == 0;

            let run_baseline = || {
                let started = Instant::now();
                let (result, timing) = profile_hybrid_rns_galois_rotation_owned_forward_qp(
                    &ciphertext,
                    &key,
                    &qp_plan,
                    &prepared_qp,
                    &canonical_source,
                );
                (result, timing, started.elapsed().as_secs_f64())
            };

            let run_candidate = || {
                let started = Instant::now();
                let (result, timing) = profile_hybrid_rns_galois_rotation_prepared_inverse_qp(
                    &ciphertext,
                    &key,
                    &qp_plan,
                    &prepared_qp,
                    &canonical_source,
                );
                (result, timing, started.elapsed().as_secs_f64())
            };

            let (baseline, candidate) = if iteration % 2 == 0 {
                (run_baseline(), run_candidate())
            } else {
                let candidate = run_candidate();
                let baseline = run_baseline();
                (baseline, candidate)
            };

            assert_eq!(
                baseline.0, candidate.0,
                "R36.5G.3 exact RLWE mismatch iteration={iteration}"
            );

            println!(
                "R36_5G3_ITERATION={} warmup={} baseline_seconds={:.6} candidate_seconds={:.6} baseline_inverse_ms={:.6} candidate_inverse_ms={:.6} baseline_keyswitch_ms={:.6} candidate_keyswitch_ms={:.6}",
                iteration,
                warmup,
                baseline.2,
                candidate.2,
                baseline.1.inverse_ntt_ms,
                candidate.1.inverse_ntt_ms,
                baseline.1.key_switch_ntt_ms,
                candidate.1.key_switch_ntt_ms,
            );

            if !warmup {
                g3_baseline_rotation.push(baseline.2);
                g3_candidate_rotation.push(candidate.2);
                g3_baseline_inverse.push(baseline.1.inverse_ntt_ms);
                g3_candidate_inverse.push(candidate.1.inverse_ntt_ms);
                g3_baseline_keyswitch.push(baseline.1.key_switch_ntt_ms);
                g3_candidate_keyswitch.push(candidate.1.key_switch_ntt_ms);
            }
        }

        let g3_mean = |values: &[f64]| -> f64 { values.iter().sum::<f64>() / values.len() as f64 };

        println!(
            "R36_5G3_INVERSE_SPEEDUP={:.6}",
            g3_mean(&g3_baseline_inverse) / g3_mean(&g3_candidate_inverse)
        );
        println!(
            "R36_5G3_KEYSWITCH_SPEEDUP={:.6}",
            g3_mean(&g3_baseline_keyswitch) / g3_mean(&g3_candidate_keyswitch)
        );
        println!(
            "R36_5G3_ROTATION_SPEEDUP={:.6}",
            g3_mean(&g3_baseline_rotation) / g3_mean(&g3_candidate_rotation)
        );
        println!("R36_5G3_EXACT_RLWE_STATUS=PASS");

        // R36_5F6_PRODUCTION_ENTRY_EQUIVALENCE
        let production_result = apply_hybrid_rns_galois_automorphism_prepared_canonical_qp(
            &ciphertext,
            &key,
            &qp_plan,
            &prepared_qp,
            &canonical_source,
        );

        let (reference_result, _) = profile_hybrid_rns_galois_rotation_prepared_qp(
            &ciphertext,
            &key,
            &qp_plan,
            &prepared_qp,
            Some(&canonical_source),
        );

        assert_eq!(
            production_result, reference_result,
            "R36.5F.6 production/reference RLWE mismatch"
        );

        println!("R36_5F6_PRODUCTION_EXACT_RLWE_STATUS=PASS");

        println!("R36_5D_EXACT_RLWE_STATUS=PASS");
    }

    // R35_9E_4_KEYGEN_BENCHMARK
    #[test]
    #[ignore = "Run explicitly for key-generation performance characterization"]
    fn r35_9e_keygen_benchmark() {
        use crate::ring::{Modulus, ModulusBasis};
        use std::time::Instant;

        const N: usize = 16;

        let q = ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]);

        let p = ModulusBasis::new(vec![Modulus::new(114_689), Modulus::new(786_433)]);

        let mut qp_moduli = q.moduli().to_vec();
        qp_moduli.extend_from_slice(p.moduli());
        let qp_plan = RnsNttPlan::new(qp_moduli, N);

        let secret: [i8; N] = [1, 0, -1, 1, 0, -1, 1, 0, -1, 0, 1, 0, 1, -1, 0, 1];

        let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };

        for repetition in 0..5_u64 {
            let seed = 0x359e_4000 + repetition;
            let mut reference_rng = ChaCha20Rng::seed_from_u64(seed);
            let mut optimized_rng = ChaCha20Rng::seed_from_u64(seed);

            let start = Instant::now();
            let reference = reference_key(
                BoundedGadgetLayout::new(q.clone(), 8),
                p.clone(),
                &secret,
                3,
                distribution,
                2,
                &qp_plan,
                &mut reference_rng,
            );
            let reference_seconds = start.elapsed().as_secs_f64();

            let start = Instant::now();
            let optimized = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(q.clone(), 8),
                p.clone(),
                &secret,
                3,
                distribution,
                2,
                &qp_plan,
                &mut optimized_rng,
            );
            let optimized_seconds = start.elapsed().as_secs_f64();

            assert_eq!(reference, optimized);

            println!(
                "R35_9E_4_SAMPLE={} REFERENCE_SECONDS={:.9} OPTIMIZED_SECONDS={:.9} SPEEDUP={:.4}",
                repetition,
                reference_seconds,
                optimized_seconds,
                reference_seconds / optimized_seconds
            );
        }

        println!("R35_9E_4_BENCHMARK_STATUS=PASS");
    }

    #[test]
    fn r35_9e_full_prepared_galois_key_exact_equivalence() {
        use crate::ring::{Modulus, ModulusBasis};

        const N: usize = 16;

        let q = ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]);

        let p = ModulusBasis::new(vec![Modulus::new(114_689), Modulus::new(786_433)]);

        let mut qp_moduli = q.moduli().to_vec();
        qp_moduli.extend_from_slice(p.moduli());

        let qp_plan = RnsNttPlan::new(qp_moduli, N);

        let secret: [i8; N] = [1, 0, -1, 1, 0, -1, 1, 0, -1, 0, 1, 0, 1, -1, 0, 1];

        for exponent in [3_usize, 5] {
            for (mode, distribution) in [
                ("ZERO_NOISE", ErrorDistribution::BoundedUniform { bound: 0 }),
                (
                    "GAUSSIAN",
                    ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                ),
            ] {
                let seed =
                    0x359e_3000_u64 + exponent as u64 + if mode == "GAUSSIAN" { 100 } else { 0 };

                let mut reference_rng = ChaCha20Rng::seed_from_u64(seed);
                let mut optimized_rng = ChaCha20Rng::seed_from_u64(seed);

                let reference = reference_key(
                    BoundedGadgetLayout::new(q.clone(), 8),
                    p.clone(),
                    &secret,
                    exponent,
                    distribution,
                    2,
                    &qp_plan,
                    &mut reference_rng,
                );

                let optimized = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                    BoundedGadgetLayout::new(q.clone(), 8),
                    p.clone(),
                    &secret,
                    exponent,
                    distribution,
                    2,
                    &qp_plan,
                    &mut optimized_rng,
                );

                assert_eq!(
                    reference, optimized,
                    "R35.9E full-key mismatch: exponent={exponent}, mode={mode}"
                );

                println!(
                    "R35_9E_FULL_KEY_CASE=PASS exponent={} mode={} digits={}",
                    exponent,
                    mode,
                    optimized.digit_count()
                );
            }
        }

        println!("R35_9E_FULL_KEY_EQUIVALENCE_STATUS=PASS");
    }

    #[test]
    fn r35_9e_cached_secret_ntt_matches_original_encryption_exactly() {
        let degree = 16usize;
        let moduli = [97_u64, 193_u64];
        let secret: Vec<i8> = (0..degree)
            .map(|i| match i % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();

        let message = RnsPolynomial::from_residues(
            moduli
                .iter()
                .map(|&q| {
                    let modulus = crate::ring::Modulus::new(q);
                    Polynomial::new(
                        modulus,
                        (0..degree).map(|i| (i as u64 * 7 + 3) % q).collect(),
                    )
                })
                .collect(),
        );

        let plan = RnsNttPlan::new(
            moduli
                .iter()
                .copied()
                .map(crate::ring::Modulus::new)
                .collect(),
            degree,
        );
        let secret_ntt: Vec<NttPolynomial> = moduli
            .iter()
            .enumerate()
            .map(|(index, &q)| {
                let modulus = crate::ring::Modulus::new(q);
                let projected = Polynomial::new(
                    modulus,
                    secret
                        .iter()
                        .map(|&x| signed_mod(i128::from(x), q))
                        .collect(),
                );
                NttPolynomial::from_polynomial(plan.plan(index), &projected)
            })
            .collect();

        let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };

        for seed_byte in [7_u8, 19_u8, 83_u8] {
            let mut reference_rng = ChaCha20Rng::from_seed([seed_byte; 32]);
            let mut optimized_rng = ChaCha20Rng::from_seed([seed_byte; 32]);

            let reference = encrypt_rns_raw_prepared_with_distribution_ntt_rng(
                &message,
                2,
                distribution,
                &secret,
                &plan,
                &mut reference_rng,
            );

            let optimized = crate::grafting::encrypt_rns_raw_prepared_with_cached_secret_ntt_rng(
                &message,
                2,
                distribution,
                &secret_ntt,
                &plan,
                &mut optimized_rng,
            );

            assert_eq!(
                reference, optimized,
                "R35.9E exact encryption mismatch for seed {seed_byte}"
            );
        }

        println!("R35_9E_EXACT_KEY_EQUIVALENCE_STATUS=PASS");
    }
}
