use crate::grafting::RnsRlweCiphertext;
use crate::ring::ModulusChain;

use super::{CkksChainState, RnsCkksCiphertext};

/// Projects one RNS CKKS ciphertext to the next modulus-chain level
/// without changing its CKKS scale.
///
/// This operation retains exactly the RNS RLWE limbs belonging to the
/// next chain level. Unlike CKKS rescaling, it performs no coefficient
/// division and leaves the scale unchanged.
///
/// The caller must ensure that the ciphertext has a next chain level.
pub fn mod_switch_rns_ckks_to_next(
    ciphertext: &RnsCkksCiphertext,
    chain: &ModulusChain,
) -> RnsCkksCiphertext {
    ciphertext.assert_matches_chain(chain);

    let next_level = ciphertext.level() + 1;
    assert!(
        next_level <= chain.max_level(),
        "cannot modulus-switch the final CKKS chain level"
    );

    let target_basis = chain.level(next_level);
    let retained_limbs = ciphertext.rlwe().limbs()[..target_basis.len()].to_vec();
    let inner = RnsRlweCiphertext::from_limbs(retained_limbs);

    let state = CkksChainState::new(chain, next_level, ciphertext.scale());

    RnsCkksCiphertext::new(inner, state, chain)
}

#[cfg(test)]
mod tests {
    use crate::ckks::CkksChainState;
    use crate::grafting::RnsRlweCiphertext;
    use crate::ring::{Modulus, ModulusBasis, ModulusChain, Polynomial};
    use crate::rlwe::RlweCiphertext;

    use super::*;

    fn chain() -> ModulusChain {
        ModulusChain::from_top_basis(ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]))
    }

    fn ciphertext(chain: &ModulusChain) -> RnsCkksCiphertext {
        let degree = 8;
        let limbs = chain
            .top()
            .moduli()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, modulus)| {
                let b = Polynomial::new(
                    modulus,
                    (0..degree)
                        .map(|coefficient| {
                            (100 + 17 * index + 3 * coefficient) as u64 % modulus.value()
                        })
                        .collect(),
                );
                let a = Polynomial::new(
                    modulus,
                    (0..degree)
                        .map(|coefficient| {
                            (200 + 19 * index + 5 * coefficient) as u64 % modulus.value()
                        })
                        .collect(),
                );
                RlweCiphertext::new(b, a)
            })
            .collect();

        RnsCkksCiphertext::new(
            RnsRlweCiphertext::from_limbs(limbs),
            CkksChainState::top(chain, 65_537.0),
            chain,
        )
    }

    #[test]
    fn modulus_switch_preserves_retained_limbs_and_scale() {
        let chain = chain();
        let input = ciphertext(&chain);

        let output = mod_switch_rns_ckks_to_next(&input, &chain);

        assert_eq!(output.level(), input.level() + 1);
        assert_eq!(output.basis(), chain.level(1));
        assert_eq!(output.scale(), input.scale());

        assert_eq!(
            output.rlwe().limbs(),
            &input.rlwe().limbs()[..chain.level(1).len()]
        );
    }

    #[test]
    #[should_panic(expected = "cannot modulus-switch the final CKKS chain level")]
    fn modulus_switch_rejects_final_level() {
        let chain = chain();
        let top = ciphertext(&chain);
        let level1 = mod_switch_rns_ckks_to_next(&top, &chain);
        let level2 = mod_switch_rns_ckks_to_next(&level1, &chain);

        let _ = mod_switch_rns_ckks_to_next(&level2, &chain);
    }
}
