use crate::grafting::RnsRlweCiphertext;
use crate::ring::{ModulusChain, PreparedRnsNttPlan, RnsNttPlan, RnsNttPolynomial, RnsPolynomial};
use crate::rlwe::RlweCiphertext;

use super::{CkksChainState, RnsCkksCiphertext};

/// Multiplies an RNS CKKS ciphertext by an encoded RNS plaintext polynomial.
///
/// The result remains an RLWE ciphertext at the same chain level. Its scale is
/// `ciphertext.scale() * plaintext_scale`. No evaluation key or relinearization
/// is required because ciphertext-plaintext multiplication preserves RLWE
/// degree one.
pub fn multiply_plain_rns_ckks_with_ntt(
    ciphertext: &RnsCkksCiphertext,
    plaintext: &RnsPolynomial,
    plaintext_scale: f64,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertext {
    ciphertext.assert_matches_chain(chain);

    assert!(
        plaintext_scale.is_finite() && plaintext_scale > 0.0,
        "CKKS plaintext scale must be finite and positive"
    );
    assert_eq!(
        plaintext.basis(),
        ciphertext.basis(),
        "CKKS plaintext basis must match ciphertext basis"
    );
    assert_eq!(
        plaintext.degree(),
        ciphertext.rlwe().degree(),
        "CKKS plaintext degree must match ciphertext degree"
    );
    assert_eq!(
        plan.degree(),
        ciphertext.rlwe().degree(),
        "RNS NTT plan degree must match ciphertext degree"
    );
    assert_eq!(
        plan.moduli(),
        ciphertext.basis().moduli(),
        "RNS NTT plan basis must match ciphertext basis"
    );

    let limbs = ciphertext
        .rlwe()
        .limbs()
        .iter()
        .enumerate()
        .map(|(index, limb)| {
            let limb_plan = plan.plan(index);
            let plain = plaintext.residue(index);

            RlweCiphertext::new(
                limb_plan.negacyclic_mul(limb.b(), plain),
                limb_plan.negacyclic_mul(limb.a(), plain),
            )
        })
        .collect();

    let scale = ciphertext.scale() * plaintext_scale;
    let state = CkksChainState::new(chain, ciphertext.level(), scale);

    RnsCkksCiphertext::new(RnsRlweCiphertext::from_limbs(limbs), state, chain)
}

/// Reusable NTT-domain representation of an encoded RNS plaintext polynomial.
///
/// This representation is intended for public plaintext operands that are
/// multiplied into multiple ciphertexts under the same RNS basis. Preparing
/// the plaintext once avoids repeating its forward NTT for every
/// ciphertext-plaintext multiplication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedRnsPlaintextNtt {
    polynomial: RnsNttPolynomial,
}

impl PreparedRnsPlaintextNtt {
    /// Prepares an encoded RNS plaintext polynomial for repeated NTT-domain
    /// ciphertext-plaintext multiplication.
    pub fn new(plaintext: &RnsPolynomial, plan: &PreparedRnsNttPlan) -> Self {
        assert_eq!(
            plaintext.degree(),
            plan.degree(),
            "CKKS plaintext degree must match prepared RNS NTT plan"
        );
        assert_eq!(
            plaintext.basis().moduli(),
            plan.moduli(),
            "CKKS plaintext basis must match prepared RNS NTT plan"
        );

        Self {
            polynomial: plan.forward(plaintext),
        }
    }

    pub fn degree(&self) -> usize {
        self.polynomial.degree()
    }

    pub fn moduli(&self) -> &[crate::ring::Modulus] {
        self.polynomial.moduli()
    }

    pub fn polynomial(&self) -> &RnsNttPolynomial {
        &self.polynomial
    }
}

/// Multiplies an RNS CKKS ciphertext by a reusable NTT-prepared plaintext.
///
/// This is mathematically equivalent to `multiply_plain_rns_ckks_with_ntt`,
/// but the plaintext forward NTT is performed once when constructing
/// `PreparedRnsPlaintextNtt` rather than once for every polynomial product.
///
/// The result remains at the ciphertext's current chain level and has scale
/// `ciphertext.scale() * plaintext_scale`.
pub fn multiply_plain_rns_ckks_with_prepared_ntt(
    ciphertext: &RnsCkksCiphertext,
    plaintext: &PreparedRnsPlaintextNtt,
    plaintext_scale: f64,
    chain: &ModulusChain,
    plan: &PreparedRnsNttPlan,
) -> RnsCkksCiphertext {
    ciphertext.assert_matches_chain(chain);

    assert!(
        plaintext_scale.is_finite() && plaintext_scale > 0.0,
        "CKKS plaintext scale must be finite and positive"
    );
    assert_eq!(
        plaintext.moduli(),
        ciphertext.basis().moduli(),
        "CKKS plaintext basis must match ciphertext basis"
    );
    assert_eq!(
        plaintext.degree(),
        ciphertext.rlwe().degree(),
        "CKKS plaintext degree must match ciphertext degree"
    );
    assert_eq!(
        plan.degree(),
        ciphertext.rlwe().degree(),
        "prepared RNS NTT plan degree must match ciphertext degree"
    );
    assert_eq!(
        plan.moduli(),
        ciphertext.basis().moduli(),
        "prepared RNS NTT plan basis must match ciphertext basis"
    );

    let limbs = ciphertext
        .rlwe()
        .limbs()
        .iter()
        .enumerate()
        .map(|(index, limb)| {
            let limb_plan = plan.plan(index);
            let plain_ntt = plaintext.polynomial().residue(index);

            let b_ntt = crate::ring::NttPolynomial::from_prepared_values(
                limb_plan,
                limb_plan.forward(limb.b()),
            );
            let a_ntt = crate::ring::NttPolynomial::from_prepared_values(
                limb_plan,
                limb_plan.forward(limb.a()),
            );

            let b_product = b_ntt.pointwise_mul(plain_ntt);
            let a_product = a_ntt.pointwise_mul(plain_ntt);

            RlweCiphertext::new(
                limb_plan.inverse(b_product.values()),
                limb_plan.inverse(a_product.values()),
            )
        })
        .collect();

    let scale = ciphertext.scale() * plaintext_scale;
    let state = CkksChainState::new(chain, ciphertext.level(), scale);

    RnsCkksCiphertext::new(RnsRlweCiphertext::from_limbs(limbs), state, chain)
}

#[cfg(test)]
mod tests {
    use crate::grafting::RnsRlweCiphertext;
    use crate::ring::{
        Modulus, ModulusBasis, ModulusChain, Polynomial, PreparedRnsNttPlan, RnsNttPlan,
        RnsPolynomial,
    };
    use crate::rlwe::RlweCiphertext;

    use super::*;

    #[test]
    fn prepared_plain_multiply_matches_legacy_exactly() {
        let degree = 16;
        let level = 0;
        let ciphertext_scale = 1024.0;
        let plaintext_scale = 4096.0;

        let chain = ModulusChain::from_top_basis(ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]));
        let basis = chain.level(level);

        let ciphertext_limbs = basis
            .moduli()
            .iter()
            .copied()
            .enumerate()
            .map(|(limb_index, modulus)| {
                let q = modulus.value();

                let b = Polynomial::new(
                    modulus,
                    (0..degree)
                        .map(|index| ((17 * index as u64 + 11 * limb_index as u64 + 3) % q) as u64)
                        .collect(),
                );
                let a = Polynomial::new(
                    modulus,
                    (0..degree)
                        .map(|index| ((29 * index as u64 + 7 * limb_index as u64 + 5) % q) as u64)
                        .collect(),
                );

                RlweCiphertext::new(b, a)
            })
            .collect();

        let ciphertext = RnsCkksCiphertext::new(
            RnsRlweCiphertext::from_limbs(ciphertext_limbs),
            CkksChainState::new(&chain, level, ciphertext_scale),
            &chain,
        );

        let plaintext = RnsPolynomial::from_residues(
            basis
                .moduli()
                .iter()
                .copied()
                .enumerate()
                .map(|(limb_index, modulus)| {
                    let q = modulus.value();

                    Polynomial::new(
                        modulus,
                        (0..degree)
                            .map(|index| {
                                ((13 * index as u64 + 19 * limb_index as u64 + 1) % q) as u64
                            })
                            .collect(),
                    )
                })
                .collect(),
        );

        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
        let prepared_plan = PreparedRnsNttPlan::new(&plan);
        let prepared_plaintext = PreparedRnsPlaintextNtt::new(&plaintext, &prepared_plan);

        let legacy = multiply_plain_rns_ckks_with_ntt(
            &ciphertext,
            &plaintext,
            plaintext_scale,
            &chain,
            &plan,
        );

        let prepared = multiply_plain_rns_ckks_with_prepared_ntt(
            &ciphertext,
            &prepared_plaintext,
            plaintext_scale,
            &chain,
            &prepared_plan,
        );

        assert_eq!(prepared.level(), legacy.level());
        assert_eq!(prepared.basis(), legacy.basis());
        assert_eq!(prepared.scale(), legacy.scale());
        assert_eq!(prepared.rlwe(), legacy.rlwe());
    }
}
