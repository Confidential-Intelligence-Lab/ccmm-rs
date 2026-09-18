use crate::grafting::RnsRlweCiphertext;
use crate::ring::{ModulusChain, RnsNttPlan};
use crate::rlwe::RlweCiphertext;

use super::{
    conjugate_with_evaluation_keys, multiply_relinearize_rescale_rns_ckks_with_ntt,
    multiply_with_evaluation_keys, rotate_left_with_evaluation_keys,
    rotate_right_with_evaluation_keys, RnsCkksCiphertext, RnsCkksEvaluationKeys,
};

/// High-level evaluator for leveled RNS CKKS operations.
///
/// The evaluator owns no cryptographic secrets. It borrows the modulus
/// chain and the level-indexed evaluation-key set and automatically
/// selects the correct key material from each ciphertext's active state.
#[derive(Debug)]
pub struct RnsCkksEvaluator<'a> {
    chain: &'a ModulusChain,
    keys: &'a RnsCkksEvaluationKeys,
}

impl<'a> RnsCkksEvaluator<'a> {
    pub fn new(chain: &'a ModulusChain, keys: &'a RnsCkksEvaluationKeys) -> Self {
        Self { chain, keys }
    }

    pub fn chain(&self) -> &'a ModulusChain {
        self.chain
    }

    pub fn keys(&self) -> &'a RnsCkksEvaluationKeys {
        self.keys
    }

    pub fn add(&self, lhs: &RnsCkksCiphertext, rhs: &RnsCkksCiphertext) -> RnsCkksCiphertext {
        lhs.assert_matches_chain(self.chain);
        rhs.assert_matches_chain(self.chain);

        assert_eq!(
            lhs.level(),
            rhs.level(),
            "RNS CKKS addition requires matching levels"
        );
        assert_eq!(
            lhs.basis(),
            rhs.basis(),
            "RNS CKKS addition requires matching bases"
        );
        assert_eq!(
            lhs.scale(),
            rhs.scale(),
            "RNS CKKS addition requires matching scales"
        );

        let limbs = lhs
            .rlwe()
            .limbs()
            .iter()
            .zip(rhs.rlwe().limbs())
            .map(|(lhs_limb, rhs_limb)| {
                RlweCiphertext::new(
                    lhs_limb.b().add(rhs_limb.b()),
                    lhs_limb.a().add(rhs_limb.a()),
                )
            })
            .collect();

        RnsCkksCiphertext::new(
            RnsRlweCiphertext::from_limbs(limbs),
            lhs.state().clone(),
            self.chain,
        )
    }

    pub fn multiply(&self, lhs: &RnsCkksCiphertext, rhs: &RnsCkksCiphertext) -> RnsCkksCiphertext {
        multiply_with_evaluation_keys(lhs, rhs, self.keys, self.chain)
    }

    /// Multiplies, relinearizes, and rescales using the NTT-backed
    /// RNS CKKS multiplication path while selecting the evaluation key
    /// from the operands' active chain level.
    pub fn multiply_with_ntt(
        &self,
        lhs: &RnsCkksCiphertext,
        rhs: &RnsCkksCiphertext,
        plan: &RnsNttPlan,
    ) -> RnsCkksCiphertext {
        lhs.assert_matches_chain(self.chain);
        rhs.assert_matches_chain(self.chain);

        assert_eq!(
            lhs.level(),
            rhs.level(),
            "RNS CKKS multiplication requires matching levels"
        );
        assert_eq!(
            lhs.basis(),
            rhs.basis(),
            "RNS CKKS multiplication requires matching bases"
        );

        let multiplication_key = self.keys.multiplication_for(lhs.state());

        multiply_relinearize_rescale_rns_ckks_with_ntt(
            lhs,
            rhs,
            multiplication_key,
            self.chain,
            plan,
        )
    }

    pub fn rotate_left(&self, ciphertext: &RnsCkksCiphertext, steps: usize) -> RnsCkksCiphertext {
        rotate_left_with_evaluation_keys(ciphertext, steps, self.keys, self.chain)
    }

    pub fn rotate_right(&self, ciphertext: &RnsCkksCiphertext, steps: usize) -> RnsCkksCiphertext {
        rotate_right_with_evaluation_keys(ciphertext, steps, self.keys, self.chain)
    }

    pub fn conjugate(&self, ciphertext: &RnsCkksCiphertext) -> RnsCkksCiphertext {
        conjugate_with_evaluation_keys(ciphertext, self.keys, self.chain)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ckks::{
        conjugation_exponent, rotation_exponent_left, rotation_exponent_right, CkksChainState,
        RnsCkksEvaluationKeys, RnsCkksLevelKeys, RnsGaloisKey,
    };
    use crate::grafting::{RnsGadgetLayout, RnsMultiplicationKey, RnsRlweCiphertext};
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

    fn secret() -> Vec<i8> {
        vec![-1, 0, 1, 1, 0, -1, 1, 0]
    }

    fn zero_ciphertext(chain: &ModulusChain, level: usize, scale: f64) -> RnsCkksCiphertext {
        let degree = 8;

        let limbs = chain
            .level(level)
            .moduli()
            .iter()
            .copied()
            .map(|modulus| {
                RlweCiphertext::new(
                    Polynomial::zero(modulus, degree),
                    Polynomial::zero(modulus, degree),
                )
            })
            .collect();

        RnsCkksCiphertext::new(
            RnsRlweCiphertext::from_limbs(limbs),
            CkksChainState::new(chain, level, scale),
            chain,
        )
    }

    fn block_sizes(basis_len: usize) -> Vec<usize> {
        match basis_len {
            3 => vec![1, 2],
            2 => vec![1, 1],
            1 => vec![1],
            _ => panic!("unexpected test basis size"),
        }
    }

    fn level_keys(
        chain: &ModulusChain,
        level: usize,
        secret: &[i8],
        seed: u64,
    ) -> RnsCkksLevelKeys {
        let basis = chain.level(level).clone();

        let mut keys = RnsCkksLevelKeys::new(level, basis.clone());

        if chain.has_next_level(level) {
            let mut multiplication_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x1000);

            let multiplication_key = RnsMultiplicationKey::generate_with_rng(
                secret.len(),
                2,
                0,
                secret,
                RnsGadgetLayout::new(basis.clone(), block_sizes(basis.len())),
                &mut multiplication_rng,
            );

            keys.set_multiplication_key(multiplication_key);
        }

        let exponents = [
            rotation_exponent_left(secret.len(), 1),
            rotation_exponent_right(secret.len(), 1),
            conjugation_exponent(secret.len()),
        ];

        for (index, exponent) in exponents.into_iter().enumerate() {
            /*
             * For degree 8, left-1, right-1, and conjugation are distinct.
             * Future parameter sets may deduplicate before insertion.
             */
            let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0x2000 ^ index as u64);

            let key = RnsGaloisKey::generate_with_rng(
                secret.len(),
                2,
                0,
                secret,
                exponent,
                RnsGadgetLayout::new(basis.clone(), block_sizes(basis.len())),
                &mut rng,
            );

            keys.insert_galois_key(key);
        }

        keys
    }

    fn evaluation_keys(chain: &ModulusChain) -> RnsCkksEvaluationKeys {
        let secret = secret();

        let mut keys = RnsCkksEvaluationKeys::new();

        for level in 0..=chain.max_level() {
            keys.insert_level(level_keys(chain, level, &secret, 0xCB00 ^ level as u64));
        }

        keys
    }

    #[test]
    fn evaluator_preserves_configuration() {
        let chain = chain();

        let keys = evaluation_keys(&chain);

        let evaluator = RnsCkksEvaluator::new(&chain, &keys);

        assert_eq!(evaluator.chain(), &chain);

        assert_eq!(evaluator.keys(), &keys);
    }

    #[test]
    fn evaluator_add_preserves_active_state() {
        let chain = chain();
        let keys = evaluation_keys(&chain);
        let evaluator = RnsCkksEvaluator::new(&chain, &keys);

        let lhs = zero_ciphertext(&chain, 1, 65_537.0);
        let rhs = zero_ciphertext(&chain, 1, 65_537.0);
        let result = evaluator.add(&lhs, &rhs);

        assert_eq!(result.level(), lhs.level());
        assert_eq!(result.basis(), lhs.basis());
        assert_eq!(result.scale(), lhs.scale());
    }

    #[test]
    #[should_panic(expected = "RNS CKKS addition requires matching levels")]
    fn evaluator_add_rejects_mismatched_levels() {
        let chain = chain();
        let keys = evaluation_keys(&chain);
        let evaluator = RnsCkksEvaluator::new(&chain, &keys);

        let lhs = zero_ciphertext(&chain, 0, 65_537.0);
        let rhs = zero_ciphertext(&chain, 1, 65_537.0);
        let _ = evaluator.add(&lhs, &rhs);
    }

    #[test]
    #[should_panic(expected = "RNS CKKS addition requires matching scales")]
    fn evaluator_add_rejects_mismatched_scales() {
        let chain = chain();
        let keys = evaluation_keys(&chain);
        let evaluator = RnsCkksEvaluator::new(&chain, &keys);

        let lhs = zero_ciphertext(&chain, 1, 65_537.0);
        let rhs = zero_ciphertext(&chain, 1, 32_768.0);
        let _ = evaluator.add(&lhs, &rhs);
    }

    #[test]
    fn evaluator_ntt_multiply_selects_active_level_key() {
        let chain = chain();
        let keys = evaluation_keys(&chain);
        let evaluator = RnsCkksEvaluator::new(&chain, &keys);

        let lhs = zero_ciphertext(&chain, 0, 256.0);
        let rhs = zero_ciphertext(&chain, 0, 512.0);
        let plan = RnsNttPlan::new(chain.level(0).moduli().to_vec(), 8);

        let result = evaluator.multiply_with_ntt(&lhs, &rhs, &plan);

        assert_eq!(result.level(), 1);
        assert_eq!(result.basis(), chain.level(1));
        assert_eq!(
            result.scale(),
            256.0 * 512.0 / chain.dropped_modulus(0).unwrap().value() as f64
        );
    }

    #[test]
    fn evaluator_multiply_selects_active_level_key() {
        let chain = chain();

        let keys = evaluation_keys(&chain);

        let evaluator = RnsCkksEvaluator::new(&chain, &keys);

        let lhs = zero_ciphertext(&chain, 0, 256.0);

        let rhs = zero_ciphertext(&chain, 0, 512.0);

        let result = evaluator.multiply(&lhs, &rhs);

        assert_eq!(result.level(), 1);

        assert_eq!(result.basis(), chain.level(1));
    }

    #[test]
    fn evaluator_rotations_preserve_state() {
        let chain = chain();

        let keys = evaluation_keys(&chain);

        let evaluator = RnsCkksEvaluator::new(&chain, &keys);

        let ciphertext = zero_ciphertext(&chain, 1, 65_537.0);

        let left = evaluator.rotate_left(&ciphertext, 1);

        let right = evaluator.rotate_right(&ciphertext, 1);

        for output in [left, right] {
            assert_eq!(output.level(), ciphertext.level());

            assert_eq!(output.basis(), ciphertext.basis());

            assert_eq!(output.scale(), ciphertext.scale());
        }
    }

    #[test]
    fn evaluator_conjugation_preserves_state() {
        let chain = chain();

        let keys = evaluation_keys(&chain);

        let evaluator = RnsCkksEvaluator::new(&chain, &keys);

        let ciphertext = zero_ciphertext(&chain, 0, 65_537.0);

        let output = evaluator.conjugate(&ciphertext);

        assert_eq!(output.level(), ciphertext.level());

        assert_eq!(output.basis(), ciphertext.basis());

        assert_eq!(output.scale(), ciphertext.scale());
    }

    #[test]
    fn evaluator_can_continue_after_multiplication() {
        let chain = chain();

        let keys = evaluation_keys(&chain);

        let evaluator = RnsCkksEvaluator::new(&chain, &keys);

        let lhs = zero_ciphertext(&chain, 0, 256.0);

        let rhs = zero_ciphertext(&chain, 0, 256.0);

        let product = evaluator.multiply(&lhs, &rhs);

        assert_eq!(product.level(), 1);

        let rotated = evaluator.rotate_left(&product, 1);

        assert_eq!(rotated.level(), 1);

        let conjugated = evaluator.conjugate(&rotated);

        assert_eq!(conjugated.level(), 1);
    }
}
