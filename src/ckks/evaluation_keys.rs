use std::collections::BTreeMap;

use crate::grafting::RnsMultiplicationKey;
use crate::ring::ModulusBasis;

use super::{CkksChainState, RnsGaloisKey};

/// Evaluation keys associated with one CKKS modulus-chain level.
///
/// The basis is retained explicitly so that key lookup can validate both
/// the numerical level and the active RNS basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsCkksLevelKeys {
    level: usize,
    basis: ModulusBasis,
    multiplication_key: Option<RnsMultiplicationKey>,
    galois_keys: BTreeMap<usize, RnsGaloisKey>,
}

impl RnsCkksLevelKeys {
    pub fn new(level: usize, basis: ModulusBasis) -> Self {
        Self {
            level,
            basis,
            multiplication_key: None,
            galois_keys: BTreeMap::new(),
        }
    }

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn basis(&self) -> &ModulusBasis {
        &self.basis
    }

    pub fn multiplication_key(&self) -> Option<&RnsMultiplicationKey> {
        self.multiplication_key.as_ref()
    }

    pub fn galois_key(&self, exponent: usize) -> Option<&RnsGaloisKey> {
        self.galois_keys.get(&exponent)
    }

    pub fn galois_keys(&self) -> &BTreeMap<usize, RnsGaloisKey> {
        &self.galois_keys
    }

    pub fn set_multiplication_key(&mut self, key: RnsMultiplicationKey) {
        assert_eq!(
            key.layout().full_basis(),
            &self.basis,
            "RNS multiplication key basis must match level-key basis"
        );
        self.multiplication_key = Some(key);
    }

    pub fn insert_galois_key(&mut self, key: RnsGaloisKey) {
        assert_eq!(
            key.layout().full_basis(),
            &self.basis,
            "RNS Galois key basis must match level-key basis"
        );

        let exponent = key.exponent();
        let previous = self.galois_keys.insert(exponent, key);
        assert!(
            previous.is_none(),
            "RNS Galois key exponent already exists at this level"
        );
    }

    pub fn assert_matches_state(&self, state: &CkksChainState) {
        assert_eq!(
            self.level,
            state.level(),
            "evaluation-key level must match CKKS ciphertext level"
        );
        assert_eq!(
            &self.basis,
            state.basis(),
            "evaluation-key basis must match CKKS ciphertext basis"
        );
    }
}

/// Level-indexed RNS CKKS evaluation-key set.
///
/// Cryptographic key types remain distinct. This container is responsible
/// only for organizing them by active CKKS level and automorphism exponent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RnsCkksEvaluationKeys {
    levels: BTreeMap<usize, RnsCkksLevelKeys>,
}

impl RnsCkksEvaluationKeys {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_level(&mut self, level_keys: RnsCkksLevelKeys) {
        let level = level_keys.level();
        let previous = self.levels.insert(level, level_keys);
        assert!(
            previous.is_none(),
            "evaluation keys already exist for this CKKS level"
        );
    }

    pub fn level(&self, level: usize) -> Option<&RnsCkksLevelKeys> {
        self.levels.get(&level)
    }

    pub fn level_mut(&mut self, level: usize) -> Option<&mut RnsCkksLevelKeys> {
        self.levels.get_mut(&level)
    }

    pub fn for_state(&self, state: &CkksChainState) -> &RnsCkksLevelKeys {
        let keys = self
            .levels
            .get(&state.level())
            .expect("evaluation keys are missing for active CKKS level");
        keys.assert_matches_state(state);
        keys
    }

    pub fn multiplication_for(&self, state: &CkksChainState) -> &RnsMultiplicationKey {
        self.for_state(state)
            .multiplication_key()
            .expect("multiplication key is missing for active CKKS level")
    }

    pub fn galois_for(&self, state: &CkksChainState, exponent: usize) -> &RnsGaloisKey {
        self.for_state(state)
            .galois_key(exponent)
            .expect("Galois key is missing for requested exponent at active CKKS level")
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    pub fn levels(&self) -> &BTreeMap<usize, RnsCkksLevelKeys> {
        &self.levels
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }
}

/// Multiplies two RNS CKKS ciphertexts using the multiplication key
/// selected automatically from their active CKKS level.
///
/// The underlying multiply/relinearize/rescale primitive is unchanged;
/// this wrapper provides level-aware evaluation-key orchestration.
pub fn multiply_with_evaluation_keys(
    lhs: &crate::ckks::RnsCkksCiphertext,
    rhs: &crate::ckks::RnsCkksCiphertext,
    keys: &RnsCkksEvaluationKeys,
    chain: &crate::ring::ModulusChain,
) -> crate::ckks::RnsCkksCiphertext {
    lhs.assert_matches_chain(chain);
    rhs.assert_matches_chain(chain);

    assert_eq!(
        lhs.level(),
        rhs.level(),
        "automatic CKKS multiplication requires matching levels"
    );

    assert_eq!(
        lhs.basis(),
        rhs.basis(),
        "automatic CKKS multiplication requires matching bases"
    );

    let multiplication_key = keys.multiplication_for(lhs.state());

    crate::ckks::multiply_relinearize_rescale_rns_ckks(lhs, rhs, multiplication_key, chain)
}

/// Rotates logical CKKS slots left using the Galois key selected
/// automatically from the ciphertext's active level and requested
/// rotation exponent.
pub fn rotate_left_with_evaluation_keys(
    ciphertext: &crate::ckks::RnsCkksCiphertext,
    steps: usize,
    keys: &RnsCkksEvaluationKeys,
    chain: &crate::ring::ModulusChain,
) -> crate::ckks::RnsCkksCiphertext {
    ciphertext.assert_matches_chain(chain);

    let exponent = crate::ckks::rotation_exponent_left(ciphertext.rlwe().degree(), steps);

    let galois_key = keys.galois_for(ciphertext.state(), exponent);

    crate::ckks::rotate_left_rns_ckks(ciphertext, steps, galois_key, chain)
}

/// Rotates logical CKKS slots right using the Galois key selected
/// automatically from the ciphertext's active level and requested
/// rotation exponent.
pub fn rotate_right_with_evaluation_keys(
    ciphertext: &crate::ckks::RnsCkksCiphertext,
    steps: usize,
    keys: &RnsCkksEvaluationKeys,
    chain: &crate::ring::ModulusChain,
) -> crate::ckks::RnsCkksCiphertext {
    ciphertext.assert_matches_chain(chain);

    let exponent = crate::ckks::rotation_exponent_right(ciphertext.rlwe().degree(), steps);

    let galois_key = keys.galois_for(ciphertext.state(), exponent);

    crate::ckks::rotate_right_rns_ckks(ciphertext, steps, galois_key, chain)
}

/// Applies logical CKKS complex conjugation using the Galois key
/// selected automatically from the ciphertext's active level.
pub fn conjugate_with_evaluation_keys(
    ciphertext: &crate::ckks::RnsCkksCiphertext,
    keys: &RnsCkksEvaluationKeys,
    chain: &crate::ring::ModulusChain,
) -> crate::ckks::RnsCkksCiphertext {
    ciphertext.assert_matches_chain(chain);

    let exponent = crate::ckks::conjugation_exponent(ciphertext.rlwe().degree());

    let galois_key = keys.galois_for(ciphertext.state(), exponent);

    crate::ckks::conjugate_rns_ckks(ciphertext, galois_key, chain)
}

/// Multiplication/relinearization backend registered for one CKKS level.
///
/// `OrdinaryRns` uses the conventional RNS multiplication key.
/// Hybrid variants use mixed odd-RNS / power-of-two Grafting material.
/// Helper-prime variants accelerate only the power-of-two sprout path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RnsCkksMultiplicationBackend {
    OrdinaryRns,
    HybridDirect,
    HybridHelperPrime,
    HybridPrepared,
}

/// Hybrid/Grafting evaluation material associated with one CKKS level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsCkksHybridLevelKeys {
    level: usize,
    ordinary_basis: ModulusBasis,
    key: crate::grafting::HybridMultiplicationKey,
    prepared_key: Option<crate::grafting::PreparedHybridMultiplicationKey>,
}

impl RnsCkksHybridLevelKeys {
    pub fn new(
        level: usize,
        ordinary_basis: ModulusBasis,
        key: crate::grafting::HybridMultiplicationKey,
    ) -> Self {
        assert_eq!(
            key.layout().ordinary_basis(),
            &ordinary_basis,
            "hybrid multiplication-key ordinary basis must match CKKS level basis"
        );

        Self {
            level,
            ordinary_basis,
            key,
            prepared_key: None,
        }
    }

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn ordinary_basis(&self) -> &ModulusBasis {
        &self.ordinary_basis
    }

    pub fn key(&self) -> &crate::grafting::HybridMultiplicationKey {
        &self.key
    }

    pub fn prepared_key(&self) -> Option<&crate::grafting::PreparedHybridMultiplicationKey> {
        self.prepared_key.as_ref()
    }

    pub fn prepare(&mut self, helper_plan: &crate::grafting::HelperPrimeNttPlan) {
        self.prepared_key = Some(crate::grafting::PreparedHybridMultiplicationKey::prepare(
            &self.key,
            helper_plan,
        ));
    }

    pub fn assert_matches_state(&self, state: &CkksChainState) {
        assert_eq!(
            self.level,
            state.level(),
            "hybrid evaluation-key level must match CKKS ciphertext level"
        );
        assert_eq!(
            &self.ordinary_basis,
            state.basis(),
            "hybrid evaluation-key ordinary basis must match CKKS ciphertext basis"
        );
    }
}

/// Level-aware multiplication policy.
///
/// Ordinary and Grafting material remain cryptographically distinct.
/// This object decides which registered backend is active at each level.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RnsCkksMultiplicationPolicy {
    backends: BTreeMap<usize, RnsCkksMultiplicationBackend>,
    hybrid_levels: BTreeMap<usize, RnsCkksHybridLevelKeys>,
}

impl RnsCkksMultiplicationPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_backend(&mut self, level: usize, backend: RnsCkksMultiplicationBackend) {
        self.backends.insert(level, backend);
    }

    pub fn backend_for(&self, state: &CkksChainState) -> RnsCkksMultiplicationBackend {
        self.backends
            .get(&state.level())
            .copied()
            .unwrap_or(RnsCkksMultiplicationBackend::OrdinaryRns)
    }

    pub fn insert_hybrid_level(&mut self, level_keys: RnsCkksHybridLevelKeys) {
        let level = level_keys.level();
        let previous = self.hybrid_levels.insert(level, level_keys);
        assert!(
            previous.is_none(),
            "hybrid evaluation keys already exist for this CKKS level"
        );
    }

    pub fn hybrid_levels(&self) -> &BTreeMap<usize, RnsCkksHybridLevelKeys> {
        &self.hybrid_levels
    }

    pub fn hybrid_for(&self, state: &CkksChainState) -> &RnsCkksHybridLevelKeys {
        let keys = self
            .hybrid_levels
            .get(&state.level())
            .expect("hybrid evaluation keys are missing for active CKKS level");

        keys.assert_matches_state(state);
        keys
    }

    pub fn assert_backend_available(&self, state: &CkksChainState) {
        match self.backend_for(state) {
            RnsCkksMultiplicationBackend::OrdinaryRns => {}

            RnsCkksMultiplicationBackend::HybridDirect
            | RnsCkksMultiplicationBackend::HybridHelperPrime => {
                let _ = self.hybrid_for(state);
            }

            RnsCkksMultiplicationBackend::HybridPrepared => {
                let keys = self.hybrid_for(state);
                assert!(
                    keys.prepared_key().is_some(),
                    "prepared hybrid evaluation key is missing for active CKKS level"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::grafting::{RnsGadgetLayout, RnsMultiplicationKey};
    use crate::ring::{Modulus, ModulusBasis, ModulusChain};

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

    #[test]
    fn level_keys_match_ckks_state() {
        let chain = chain();
        let state = CkksChainState::top(&chain, 65_537.0);
        let keys = RnsCkksLevelKeys::new(0, chain.level(0).clone());

        keys.assert_matches_state(&state);
        assert_eq!(keys.level(), 0);
        assert_eq!(keys.basis(), chain.level(0));
    }

    #[test]
    fn evaluation_keys_select_level_from_state() {
        let chain = chain();
        let mut keys = RnsCkksEvaluationKeys::new();

        keys.insert_level(RnsCkksLevelKeys::new(0, chain.level(0).clone()));
        keys.insert_level(RnsCkksLevelKeys::new(1, chain.level(1).clone()));

        let state = CkksChainState::new(&chain, 1, 65_537.0);
        let selected = keys.for_state(&state);

        assert_eq!(selected.level(), 1);
        assert_eq!(selected.basis(), chain.level(1));
        assert_eq!(keys.level_count(), 2);
    }

    #[test]
    fn multiplication_key_is_selected_by_active_level() {
        let chain = chain();
        let secret = secret();
        let state = CkksChainState::top(&chain, 65_537.0);

        let layout = RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]);
        let mut rng = ChaCha20Rng::seed_from_u64(0xC800);
        let multiplication_key =
            RnsMultiplicationKey::generate_with_rng(8, 2, 0, &secret, layout, &mut rng);

        let mut level_keys = RnsCkksLevelKeys::new(0, chain.level(0).clone());
        level_keys.set_multiplication_key(multiplication_key);

        let mut keys = RnsCkksEvaluationKeys::new();
        keys.insert_level(level_keys);

        assert_eq!(
            keys.multiplication_for(&state).layout().full_basis(),
            chain.level(0)
        );
    }

    #[test]
    fn galois_key_is_selected_by_level_and_exponent() {
        let chain = chain();
        let secret = secret();
        let state = CkksChainState::top(&chain, 65_537.0);
        let exponent = crate::ckks::rotation_exponent_left(8, 1);

        let layout = RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]);
        let mut rng = ChaCha20Rng::seed_from_u64(0xC801);
        let galois_key =
            RnsGaloisKey::generate_with_rng(8, 2, 0, &secret, exponent, layout, &mut rng);

        let mut level_keys = RnsCkksLevelKeys::new(0, chain.level(0).clone());
        level_keys.insert_galois_key(galois_key);

        let mut keys = RnsCkksEvaluationKeys::new();
        keys.insert_level(level_keys);

        assert_eq!(keys.galois_for(&state, exponent).exponent(), exponent);
    }

    #[test]
    #[should_panic(expected = "evaluation-key basis must match CKKS ciphertext basis")]
    fn state_with_wrong_basis_is_rejected() {
        let chain = chain();

        let keys = RnsCkksLevelKeys::new(0, chain.level(1).clone());
        let state = CkksChainState::top(&chain, 65_537.0);

        keys.assert_matches_state(&state);
    }

    #[test]
    #[should_panic(expected = "evaluation keys are missing for active CKKS level")]
    fn missing_level_is_rejected() {
        let chain = chain();
        let keys = RnsCkksEvaluationKeys::new();
        let state = CkksChainState::top(&chain, 65_537.0);

        let _ = keys.for_state(&state);
    }

    fn zero_ciphertext(
        chain: &ModulusChain,
        level: usize,
        degree: usize,
        scale: f64,
    ) -> crate::ckks::RnsCkksCiphertext {
        use crate::grafting::RnsRlweCiphertext;
        use crate::ring::Polynomial;
        use crate::rlwe::RlweCiphertext;

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

        crate::ckks::RnsCkksCiphertext::new(
            RnsRlweCiphertext::from_limbs(limbs),
            CkksChainState::new(chain, level, scale),
            chain,
        )
    }

    fn multiplication_key(
        chain: &ModulusChain,
        level: usize,
        secret: &[i8],
        seed: u64,
    ) -> RnsMultiplicationKey {
        let basis = chain.level(level);

        let blocks = match basis.len() {
            3 => vec![1, 2],
            2 => vec![1, 1],
            1 => vec![1],
            _ => panic!("unexpected test basis size"),
        };

        let layout = RnsGadgetLayout::new(basis.clone(), blocks);

        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        RnsMultiplicationKey::generate_with_rng(secret.len(), 2, 0, secret, layout, &mut rng)
    }

    #[test]
    fn automatic_multiplication_selects_level_zero_key() {
        let chain = chain();

        let secret = secret();

        let lhs = zero_ciphertext(&chain, 0, 8, 256.0);

        let rhs = zero_ciphertext(&chain, 0, 8, 512.0);

        let mut level0 = RnsCkksLevelKeys::new(0, chain.level(0).clone());

        level0.set_multiplication_key(multiplication_key(&chain, 0, &secret, 0xC900));

        let mut keys = RnsCkksEvaluationKeys::new();

        keys.insert_level(level0);

        let result = multiply_with_evaluation_keys(&lhs, &rhs, &keys, &chain);

        assert_eq!(result.level(), 1);

        assert_eq!(result.basis(), chain.level(1));
    }

    #[test]
    fn automatic_multiplication_selects_middle_level_key() {
        let chain = chain();

        let secret = secret();

        let lhs = zero_ciphertext(&chain, 1, 8, 128.0);

        let rhs = zero_ciphertext(&chain, 1, 8, 256.0);

        let mut level1 = RnsCkksLevelKeys::new(1, chain.level(1).clone());

        level1.set_multiplication_key(multiplication_key(&chain, 1, &secret, 0xC901));

        let mut keys = RnsCkksEvaluationKeys::new();

        keys.insert_level(level1);

        let result = multiply_with_evaluation_keys(&lhs, &rhs, &keys, &chain);

        assert_eq!(result.level(), 2);

        assert_eq!(result.basis(), chain.level(2));
    }

    #[test]
    #[should_panic(expected = "multiplication key is missing for active CKKS level")]
    fn automatic_multiplication_rejects_missing_key() {
        let chain = chain();

        let lhs = zero_ciphertext(&chain, 0, 8, 256.0);

        let rhs = zero_ciphertext(&chain, 0, 8, 256.0);

        let mut keys = RnsCkksEvaluationKeys::new();

        keys.insert_level(RnsCkksLevelKeys::new(0, chain.level(0).clone()));

        let _ = multiply_with_evaluation_keys(&lhs, &rhs, &keys, &chain);
    }

    #[test]
    #[should_panic(expected = "automatic CKKS multiplication requires matching levels")]
    fn automatic_multiplication_rejects_mismatched_levels() {
        let chain = chain();

        let lhs = zero_ciphertext(&chain, 0, 8, 256.0);

        let rhs = zero_ciphertext(&chain, 1, 8, 256.0);

        let keys = RnsCkksEvaluationKeys::new();

        let _ = multiply_with_evaluation_keys(&lhs, &rhs, &keys, &chain);
    }

    fn galois_key(
        chain: &ModulusChain,
        level: usize,
        secret: &[i8],
        exponent: usize,
        seed: u64,
    ) -> RnsGaloisKey {
        let basis = chain.level(level);

        let blocks = match basis.len() {
            3 => vec![1, 2],
            2 => vec![1, 1],
            1 => vec![1],
            _ => panic!("unexpected test basis size"),
        };

        let layout = RnsGadgetLayout::new(basis.clone(), blocks);

        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        RnsGaloisKey::generate_with_rng(secret.len(), 2, 0, secret, exponent, layout, &mut rng)
    }

    #[test]
    fn automatic_left_rotation_selects_key_by_level_and_exponent() {
        let chain = chain();

        let secret = secret();

        let ciphertext = zero_ciphertext(&chain, 0, 8, 65_537.0);

        let steps = 1;

        let exponent = crate::ckks::rotation_exponent_left(8, steps);

        let mut level0 = RnsCkksLevelKeys::new(0, chain.level(0).clone());

        level0.insert_galois_key(galois_key(&chain, 0, &secret, exponent, 0xCA00));

        let mut keys = RnsCkksEvaluationKeys::new();

        keys.insert_level(level0);

        let rotated = rotate_left_with_evaluation_keys(&ciphertext, steps, &keys, &chain);

        assert_eq!(rotated.level(), 0);

        assert_eq!(rotated.basis(), chain.level(0));

        assert_eq!(rotated.scale(), ciphertext.scale());
    }

    #[test]
    fn automatic_right_rotation_selects_middle_level_key() {
        let chain = chain();

        let secret = secret();

        let ciphertext = zero_ciphertext(&chain, 1, 8, 65_537.0);

        let steps = 1;

        let exponent = crate::ckks::rotation_exponent_right(8, steps);

        let mut level1 = RnsCkksLevelKeys::new(1, chain.level(1).clone());

        level1.insert_galois_key(galois_key(&chain, 1, &secret, exponent, 0xCA01));

        let mut keys = RnsCkksEvaluationKeys::new();

        keys.insert_level(level1);

        let rotated = rotate_right_with_evaluation_keys(&ciphertext, steps, &keys, &chain);

        assert_eq!(rotated.level(), 1);

        assert_eq!(rotated.basis(), chain.level(1));
    }

    #[test]
    fn automatic_conjugation_selects_key_by_level() {
        let chain = chain();

        let secret = secret();

        let ciphertext = zero_ciphertext(&chain, 0, 8, 65_537.0);

        let exponent = crate::ckks::conjugation_exponent(8);

        let mut level0 = RnsCkksLevelKeys::new(0, chain.level(0).clone());

        level0.insert_galois_key(galois_key(&chain, 0, &secret, exponent, 0xCA02));

        let mut keys = RnsCkksEvaluationKeys::new();

        keys.insert_level(level0);

        let conjugated = conjugate_with_evaluation_keys(&ciphertext, &keys, &chain);

        assert_eq!(conjugated.level(), ciphertext.level());

        assert_eq!(conjugated.scale(), ciphertext.scale());
    }

    #[test]
    #[should_panic(expected = "Galois key is missing for requested exponent at active CKKS level")]
    fn automatic_rotation_rejects_missing_exponent_key() {
        let chain = chain();

        let ciphertext = zero_ciphertext(&chain, 0, 8, 65_537.0);

        let mut keys = RnsCkksEvaluationKeys::new();

        keys.insert_level(RnsCkksLevelKeys::new(0, chain.level(0).clone()));

        let _ = rotate_left_with_evaluation_keys(&ciphertext, 1, &keys, &chain);
    }

    #[test]
    #[should_panic(expected = "evaluation keys are missing for active CKKS level")]
    fn automatic_galois_operation_rejects_missing_level() {
        let chain = chain();

        let ciphertext = zero_ciphertext(&chain, 1, 8, 65_537.0);

        let keys = RnsCkksEvaluationKeys::new();

        let _ = conjugate_with_evaluation_keys(&ciphertext, &keys, &chain);
    }

    fn hybrid_key(
        chain: &ModulusChain,
        level: usize,
        secret: &[i8],
        seed: u64,
    ) -> crate::grafting::HybridMultiplicationKey {
        use crate::grafting::MixedGadgetLayout;

        let basis = chain.level(level).clone();

        let blocks = match basis.len() {
            3 => vec![1, 2],
            2 => vec![1, 1],
            1 => vec![1],
            _ => panic!("unexpected test basis size"),
        };

        let layout = MixedGadgetLayout::new(basis, blocks, 8);

        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        crate::grafting::HybridMultiplicationKey::generate_with_rng(
            secret.len(),
            2,
            0,
            secret,
            layout,
            &mut rng,
        )
    }

    #[test]
    fn multiplication_policy_defaults_to_ordinary_rns() {
        let chain = chain();
        let state = CkksChainState::top(&chain, 65_537.0);
        let policy = RnsCkksMultiplicationPolicy::new();

        assert_eq!(
            policy.backend_for(&state),
            RnsCkksMultiplicationBackend::OrdinaryRns
        );

        policy.assert_backend_available(&state);
    }

    #[test]
    fn hybrid_backend_is_selected_by_active_level() {
        let chain = chain();
        let secret = secret();
        let state = CkksChainState::new(&chain, 1, 65_537.0);

        let level_keys = RnsCkksHybridLevelKeys::new(
            1,
            chain.level(1).clone(),
            hybrid_key(&chain, 1, &secret, 0xCC00),
        );

        let mut policy = RnsCkksMultiplicationPolicy::new();

        policy.insert_hybrid_level(level_keys);
        policy.set_backend(1, RnsCkksMultiplicationBackend::HybridDirect);

        assert_eq!(
            policy.backend_for(&state),
            RnsCkksMultiplicationBackend::HybridDirect
        );

        assert_eq!(policy.hybrid_for(&state).ordinary_basis(), chain.level(1));

        policy.assert_backend_available(&state);
    }

    #[test]
    #[should_panic(expected = "hybrid evaluation keys are missing for active CKKS level")]
    fn hybrid_backend_rejects_missing_level_material() {
        let chain = chain();
        let state = CkksChainState::top(&chain, 65_537.0);

        let mut policy = RnsCkksMultiplicationPolicy::new();

        policy.set_backend(0, RnsCkksMultiplicationBackend::HybridDirect);

        policy.assert_backend_available(&state);
    }

    #[test]
    #[should_panic(expected = "prepared hybrid evaluation key is missing for active CKKS level")]
    fn prepared_backend_requires_prepared_material() {
        let chain = chain();
        let secret = secret();
        let state = CkksChainState::top(&chain, 65_537.0);

        let level_keys = RnsCkksHybridLevelKeys::new(
            0,
            chain.level(0).clone(),
            hybrid_key(&chain, 0, &secret, 0xCC01),
        );

        let mut policy = RnsCkksMultiplicationPolicy::new();

        policy.insert_hybrid_level(level_keys);
        policy.set_backend(0, RnsCkksMultiplicationBackend::HybridPrepared);

        policy.assert_backend_available(&state);
    }

    #[test]
    #[should_panic(expected = "evaluation keys already exist for this CKKS level")]
    fn duplicate_level_registration_is_rejected() {
        let chain = chain();

        let mut keys = RnsCkksEvaluationKeys::new();

        keys.insert_level(RnsCkksLevelKeys::new(0, chain.level(0).clone()));

        keys.insert_level(RnsCkksLevelKeys::new(0, chain.level(0).clone()));
    }

    #[test]
    #[should_panic(expected = "RNS Galois key exponent already exists at this level")]
    fn duplicate_galois_exponent_is_rejected() {
        let chain = chain();
        let secret = secret();

        let exponent = crate::ckks::rotation_exponent_left(secret.len(), 1);

        let mut level = RnsCkksLevelKeys::new(0, chain.level(0).clone());

        level.insert_galois_key(galois_key(&chain, 0, &secret, exponent, 0xD000));

        level.insert_galois_key(galois_key(&chain, 0, &secret, exponent, 0xD001));
    }

    #[test]
    #[should_panic(expected = "RNS multiplication key basis must match level-key basis")]
    fn multiplication_key_with_wrong_basis_is_rejected() {
        let chain = chain();
        let secret = secret();

        let wrong = multiplication_key(&chain, 1, &secret, 0xD010);

        let mut level = RnsCkksLevelKeys::new(0, chain.level(0).clone());

        level.set_multiplication_key(wrong);
    }

    #[test]
    #[should_panic(expected = "RNS Galois key basis must match level-key basis")]
    fn galois_key_with_wrong_basis_is_rejected() {
        let chain = chain();
        let secret = secret();

        let exponent = crate::ckks::rotation_exponent_left(secret.len(), 1);

        let wrong = galois_key(&chain, 1, &secret, exponent, 0xD020);

        let mut level = RnsCkksLevelKeys::new(0, chain.level(0).clone());

        level.insert_galois_key(wrong);
    }

    #[test]
    #[should_panic(
        expected = "hybrid multiplication-key ordinary basis must match CKKS level basis"
    )]
    fn hybrid_key_with_wrong_basis_is_rejected() {
        let chain = chain();
        let secret = secret();

        let key = hybrid_key(&chain, 1, &secret, 0xD030);

        let _ = RnsCkksHybridLevelKeys::new(0, chain.level(0).clone(), key);
    }

    #[test]
    #[should_panic(expected = "hybrid evaluation keys already exist for this CKKS level")]
    fn duplicate_hybrid_level_registration_is_rejected() {
        let chain = chain();
        let secret = secret();

        let mut policy = RnsCkksMultiplicationPolicy::new();

        policy.insert_hybrid_level(RnsCkksHybridLevelKeys::new(
            0,
            chain.level(0).clone(),
            hybrid_key(&chain, 0, &secret, 0xD040),
        ));

        policy.insert_hybrid_level(RnsCkksHybridLevelKeys::new(
            0,
            chain.level(0).clone(),
            hybrid_key(&chain, 0, &secret, 0xD041),
        ));
    }

    #[test]
    #[should_panic(expected = "multiplication key is missing for active CKKS level")]
    fn automatic_multiply_rejects_level_without_multiplication_key() {
        let chain = chain();

        let lhs = zero_ciphertext(&chain, 0, 8, 256.0);

        let rhs = zero_ciphertext(&chain, 0, 8, 256.0);

        let mut keys = RnsCkksEvaluationKeys::new();

        keys.insert_level(RnsCkksLevelKeys::new(0, chain.level(0).clone()));

        let _ = multiply_with_evaluation_keys(&lhs, &rhs, &keys, &chain);
    }

    #[test]
    #[should_panic(expected = "Galois key is missing for requested exponent at active CKKS level")]
    fn automatic_rotation_rejects_unregistered_exponent() {
        let chain = chain();
        let secret = secret();

        let ciphertext = zero_ciphertext(&chain, 0, 8, 65_537.0);

        let registered = crate::ckks::rotation_exponent_left(secret.len(), 1);

        let mut level = RnsCkksLevelKeys::new(0, chain.level(0).clone());

        level.insert_galois_key(galois_key(&chain, 0, &secret, registered, 0xD050));

        let mut keys = RnsCkksEvaluationKeys::new();

        keys.insert_level(level);

        /*
         * Request a different exponent than the one registered above.
         */
        let _ = rotate_left_with_evaluation_keys(&ciphertext, 2, &keys, &chain);
    }

    #[test]
    #[should_panic(expected = "evaluation-key basis must match CKKS ciphertext basis")]
    fn state_level_with_wrong_basis_is_rejected_by_lookup() {
        let chain = chain();

        let mut keys = RnsCkksEvaluationKeys::new();

        keys.insert_level(RnsCkksLevelKeys::new(0, chain.level(1).clone()));

        let state = CkksChainState::top(&chain, 65_537.0);

        let _ = keys.for_state(&state);
    }
}
