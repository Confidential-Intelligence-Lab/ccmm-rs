use crate::grafting::RnsRlweCiphertext;
use crate::ring::{ModulusChain, RnsNttPlan, RnsPolynomial};
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
