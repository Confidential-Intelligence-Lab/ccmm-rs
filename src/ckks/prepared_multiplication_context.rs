use std::collections::BTreeMap;

use crate::grafting::{PreparedRnsMultiplicationKey, RnsMultiplicationKey};
use crate::ring::{ModulusChain, PreparedRnsNttPlan, RnsNttPlan};

use super::RnsCkksEvaluationKeys;

/// Prepared multiplication resources for selected CKKS chain levels.
///
/// Preparation is explicit and performed outside online evaluation.
#[derive(Debug)]
pub struct PreparedRnsCkksMultiplicationContext<'a> {
    chain: &'a ModulusChain,
    keys: &'a RnsCkksEvaluationKeys,
    levels: BTreeMap<usize, PreparedMultiplicationLevel>,
}

#[derive(Debug)]
pub struct PreparedMultiplicationLevel {
    plan: PreparedRnsNttPlan,
    multiplication_key: PreparedRnsMultiplicationKey,
}

impl<'a> PreparedRnsCkksMultiplicationContext<'a> {
    pub fn new(chain: &'a ModulusChain, keys: &'a RnsCkksEvaluationKeys) -> Self {
        Self {
            chain,
            keys,
            levels: BTreeMap::new(),
        }
    }

    pub fn matches_source(&self, chain: &ModulusChain, keys: &RnsCkksEvaluationKeys) -> bool {
        std::ptr::eq(self.chain, chain) && std::ptr::eq(self.keys, keys)
    }

    pub fn prepare_level(&mut self, level: usize, degree: usize) {
        let chain = self.chain;
        let keys = self.keys;

        assert!(
            chain.has_next_level(level),
            "prepared CKKS multiplication requires a next chain level"
        );

        assert!(
            !self.levels.contains_key(&level),
            "prepared CKKS multiplication level already registered"
        );

        let basis = chain.level(level);
        let level_keys = keys
            .level(level)
            .expect("evaluation keys missing for requested CKKS level");

        assert_eq!(
            level_keys.basis(),
            basis,
            "evaluation-key basis must match CKKS chain level"
        );

        let multiplication_key: &RnsMultiplicationKey = level_keys
            .multiplication_key()
            .expect("multiplication key missing for requested CKKS level");

        let ntt_plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        let prepared_key = PreparedRnsMultiplicationKey::new(multiplication_key, &ntt_plan);

        let prepared_plan = PreparedRnsNttPlan::new(&ntt_plan);

        self.levels.insert(
            level,
            PreparedMultiplicationLevel {
                plan: prepared_plan,
                multiplication_key: prepared_key,
            },
        );
    }

    pub fn level(&self, level: usize) -> Option<&PreparedMultiplicationLevel> {
        self.levels.get(&level)
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }
}

impl PreparedMultiplicationLevel {
    pub fn plan(&self) -> &PreparedRnsNttPlan {
        &self.plan
    }

    pub fn multiplication_key(&self) -> &PreparedRnsMultiplicationKey {
        &self.multiplication_key
    }
}
