//! Experimental ordinary-RNS NTT relinearization backend.

use super::{
    hybrid_key_switch::relinearize_prepared_sprout, HelperPrimeNttPlan, HybridMultiplicationKey,
    HybridQuadraticCiphertext, HybridRlweCiphertext, PreparedHybridMultiplicationKey,
};
use crate::ring::{make_ntt_plan, NttPolynomial, Polynomial};

/// Prepared ordinary-RNS evaluation-key operands.
pub struct PreparedOrdinaryNttKey {
    degree: usize,
    entries: Vec<Vec<(NttPolynomial, NttPolynomial)>>,
}

impl PreparedOrdinaryNttKey {
    pub fn prepare(key: &HybridMultiplicationKey, degree: usize) -> Self {
        let basis = key.layout().ordinary_basis();

        let entries = (0..key.layout().block_count())
            .map(|block| {
                (0..basis.len())
                    .map(|limb| {
                        let plan = make_ntt_plan(basis.modulus(limb), degree);
                        let entry = key.entry(block).ordinary().limb(limb);
                        (
                            NttPolynomial::from_polynomial(&plan, entry.b()),
                            NttPolynomial::from_polynomial(&plan, entry.a()),
                        )
                    })
                    .collect()
            })
            .collect();

        Self { degree, entries }
    }
}

/// Experimental hybrid relinearization using prepared ordinary-RNS NTT keys.
///
/// The existing prepared helper-prime backend is retained for the sprout limb.
pub fn hybrid_relinearize_ordinary_ntt_prepared(
    product: &HybridQuadraticCiphertext,
    key: &HybridMultiplicationKey,
    prepared_ordinary: &PreparedOrdinaryNttKey,
    prepared_sprout: &PreparedHybridMultiplicationKey,
    helper: &HelperPrimeNttPlan,
) -> HybridRlweCiphertext {
    assert_eq!(product.c0().degree(), prepared_ordinary.degree);

    let layout = key.layout();
    let decomposition = layout.decompose(product.c2());
    let basis = layout.ordinary_basis();
    let ordinary_blocks = layout.ordinary_layout().block_count();

    let mut ordinary_limbs = Vec::with_capacity(basis.len());

    for limb in 0..basis.len() {
        let modulus = basis.modulus(limb);
        let plan = make_ntt_plan(modulus, prepared_ordinary.degree);

        let mut b = NttPolynomial::from_polynomial(&plan, product.c0().ordinary().residue(limb));
        let mut a = NttPolynomial::from_polynomial(&plan, product.c1().ordinary().residue(limb));

        for block in 0..=ordinary_blocks {
            let coefficients: Vec<u64> = if block == ordinary_blocks {
                decomposition
                    .sprout_digit()
                    .iter()
                    .map(|&value| value % modulus.value())
                    .collect()
            } else {
                decomposition
                    .ordinary_digit(block)
                    .iter()
                    .map(|&value| (value % u128::from(modulus.value())) as u64)
                    .collect()
            };

            let digit = Polynomial::new(modulus, coefficients);
            let digit_ntt = NttPolynomial::from_polynomial(&plan, &digit);
            let (key_b, key_a) = &prepared_ordinary.entries[block][limb];

            b.pointwise_mul_add_assign(&digit_ntt, key_b);
            a.pointwise_mul_add_assign(&digit_ntt, key_a);
        }

        ordinary_limbs.push(crate::rlwe::RlweCiphertext::new(
            b.to_polynomial(&plan),
            a.to_polynomial(&plan),
        ));
    }

    let sprout = relinearize_prepared_sprout(product, prepared_sprout, helper, &decomposition);

    let ordinary = crate::grafting::RnsRlweCiphertext::from_limbs(ordinary_limbs);

    HybridRlweCiphertext::new(ordinary, sprout)
}
