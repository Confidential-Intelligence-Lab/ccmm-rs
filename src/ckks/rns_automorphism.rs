use crate::grafting::RnsRlweCiphertext;
use crate::rlwe::RlweCiphertext;

use super::apply_automorphism;

/// Applies the ring automorphism `X -> X^k` independently to every
/// RNS limb of an RLWE ciphertext.
///
/// This operation preserves the RNS basis and ciphertext degree. It does
/// not perform key switching: after the automorphism, the ciphertext
/// decrypts under the transformed secret `sigma_k(s)`.
pub fn apply_rns_automorphism(
    ciphertext: &RnsRlweCiphertext,
    exponent: usize,
) -> RnsRlweCiphertext {
    let limbs = ciphertext
        .limbs()
        .iter()
        .map(|limb| {
            RlweCiphertext::new(
                apply_automorphism(limb.b(), exponent),
                apply_automorphism(limb.a(), exponent),
            )
        })
        .collect();

    RnsRlweCiphertext::from_limbs(limbs)
}

#[cfg(test)]
mod tests {
    use crate::ring::{Modulus, Polynomial};

    use super::*;

    fn ciphertext() -> RnsRlweCiphertext {
        let limbs = [
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]
        .into_iter()
        .map(|modulus| {
            RlweCiphertext::new(
                Polynomial::new(modulus, vec![1, 2, 3, 4, 5, 6, 7, 8]),
                Polynomial::new(modulus, vec![8, 7, 6, 5, 4, 3, 2, 1]),
            )
        })
        .collect();

        RnsRlweCiphertext::from_limbs(limbs)
    }

    #[test]
    fn identity_preserves_rns_ciphertext_exactly() {
        let ciphertext = ciphertext();

        assert_eq!(apply_rns_automorphism(&ciphertext, 1), ciphertext);
    }

    #[test]
    fn rns_automorphism_matches_independent_limb_application() {
        let ciphertext = ciphertext();

        for exponent in (1..16).step_by(2) {
            let transformed = apply_rns_automorphism(&ciphertext, exponent);

            assert_eq!(transformed.basis(), ciphertext.basis());

            assert_eq!(transformed.degree(), ciphertext.degree());

            for limb_index in 0..ciphertext.basis().len() {
                let source = ciphertext.limb(limb_index);

                let observed = transformed.limb(limb_index);

                assert_eq!(observed.b(), &apply_automorphism(source.b(), exponent,));

                assert_eq!(observed.a(), &apply_automorphism(source.a(), exponent,));
            }
        }
    }

    #[test]
    fn inverse_rns_automorphism_recovers_ciphertext() {
        let ciphertext = ciphertext();
        let degree = ciphertext.degree();

        for exponent in (1..2 * degree).step_by(2) {
            let inverse = crate::ckks::inverse_automorphism_exponent(degree, exponent);

            let transformed = apply_rns_automorphism(&ciphertext, exponent);

            let recovered = apply_rns_automorphism(&transformed, inverse);

            assert_eq!(
                recovered, ciphertext,
                "exponent={exponent}, inverse={inverse}"
            );
        }
    }

    #[test]
    fn rns_automorphism_preserves_basis_order() {
        let ciphertext = ciphertext();

        let transformed = apply_rns_automorphism(&ciphertext, 5);

        assert_eq!(transformed.basis().moduli(), ciphertext.basis().moduli());
    }
}
