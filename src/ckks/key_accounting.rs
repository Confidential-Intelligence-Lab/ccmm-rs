use crate::grafting::{
    HybridMultiplicationKey, PreparedHybridMultiplicationKey, RnsKeySwitchKey,
    RnsMultiplicationKey, RnsRlweCiphertext,
};
use crate::rlwe::RlweCiphertext;

use super::RnsGaloisKey;

/// Logical coefficient payload size.
///
/// This deliberately excludes Rust object headers, Vec metadata,
/// allocator slack, and other implementation-dependent overhead.
/// Every stored coefficient is currently represented as `u64`.
const COEFFICIENT_BYTES: usize = core::mem::size_of::<u64>();

pub trait LogicalPayloadBytes {
    fn logical_payload_bytes(&self) -> usize;
}

impl LogicalPayloadBytes for RlweCiphertext {
    fn logical_payload_bytes(&self) -> usize {
        (self.b().degree() + self.a().degree()) * COEFFICIENT_BYTES
    }
}

impl LogicalPayloadBytes for RnsRlweCiphertext {
    fn logical_payload_bytes(&self) -> usize {
        self.limbs()
            .iter()
            .map(LogicalPayloadBytes::logical_payload_bytes)
            .sum()
    }
}

impl LogicalPayloadBytes for RnsKeySwitchKey {
    fn logical_payload_bytes(&self) -> usize {
        self.entries()
            .iter()
            .map(LogicalPayloadBytes::logical_payload_bytes)
            .sum()
    }
}

impl LogicalPayloadBytes for RnsMultiplicationKey {
    fn logical_payload_bytes(&self) -> usize {
        self.entries()
            .iter()
            .map(LogicalPayloadBytes::logical_payload_bytes)
            .sum()
    }
}

impl LogicalPayloadBytes for RnsGaloisKey {
    fn logical_payload_bytes(&self) -> usize {
        self.key_switch_key().logical_payload_bytes()
    }
}

impl LogicalPayloadBytes for HybridMultiplicationKey {
    fn logical_payload_bytes(&self) -> usize {
        self.entries()
            .iter()
            .map(|entry| {
                entry.ordinary().logical_payload_bytes()
                    + (entry.sprout().b().degree() + entry.sprout().a().degree())
                        * COEFFICIENT_BYTES
            })
            .sum()
    }
}

/// Incremental NTT-domain cache added by a prepared hybrid key.
///
/// Each prepared entry stores two helper-prime NTT polynomials (`b`, `a`);
/// each NTT polynomial stores one `u64` value per ring coefficient.
pub fn prepared_hybrid_cache_bytes(key: &PreparedHybridMultiplicationKey) -> usize {
    key.key()
        .entries()
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let entry = key.sprout_entry(index);

            (entry.b().degree() + entry.a().degree()) * COEFFICIENT_BYTES
        })
        .sum()
}

impl LogicalPayloadBytes for PreparedHybridMultiplicationKey {
    fn logical_payload_bytes(&self) -> usize {
        self.key().logical_payload_bytes() + prepared_hybrid_cache_bytes(self)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::grafting::{
        HelperPrimeNttPlan, HybridMultiplicationKey, MixedGadgetLayout,
        PreparedHybridMultiplicationKey, RnsGadgetLayout, RnsMultiplicationKey,
    };
    use crate::ring::{Modulus, ModulusBasis};

    use super::*;

    fn secret(degree: usize) -> Vec<i8> {
        (0..degree)
            .map(|index| match index % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect()
    }

    #[test]
    fn rns_multiplication_key_payload_matches_closed_form() {
        let degree = 8;

        let basis = ModulusBasis::new(vec![Modulus::new(12_289), Modulus::new(40_961)]);

        let layout = RnsGadgetLayout::new(basis.clone(), vec![1, 1]);

        let mut rng = ChaCha20Rng::seed_from_u64(0xAC00);

        let key = RnsMultiplicationKey::generate_with_rng(
            degree,
            2,
            0,
            &secret(degree),
            layout,
            &mut rng,
        );

        let expected = 2 * basis.len() * 2 * degree * COEFFICIENT_BYTES;

        assert_eq!(key.logical_payload_bytes(), expected);
    }

    #[test]
    fn hybrid_key_payload_counts_ordinary_and_sprout_components() {
        let degree = 8;

        let basis = ModulusBasis::new(vec![Modulus::new(12_289), Modulus::new(40_961)]);

        let layout = MixedGadgetLayout::new(basis.clone(), vec![1, 1], 8);

        let mut rng = ChaCha20Rng::seed_from_u64(0xAC01);

        let key = HybridMultiplicationKey::generate_with_rng(
            degree,
            2,
            0,
            &secret(degree),
            layout,
            &mut rng,
        );

        let entry_count = key.entries().len();

        let bytes_per_entry = (basis.len() * 2 * degree + 2 * degree) * COEFFICIENT_BYTES;

        assert_eq!(key.logical_payload_bytes(), entry_count * bytes_per_entry);
    }

    #[test]
    fn prepared_hybrid_payload_separates_base_key_and_cache() {
        let degree = 8;

        let basis = ModulusBasis::new(vec![Modulus::new(12_289), Modulus::new(40_961)]);

        let layout = MixedGadgetLayout::new(basis, vec![1, 1], 8);

        let mut rng = ChaCha20Rng::seed_from_u64(0xAC02);

        let key = HybridMultiplicationKey::generate_with_rng(
            degree,
            2,
            0,
            &secret(degree),
            layout,
            &mut rng,
        );

        let helper = HelperPrimeNttPlan::new(8, degree, Modulus::new(2_013_265_921));

        let prepared = PreparedHybridMultiplicationKey::prepare(&key, &helper);

        let expected_cache = key.entries().len() * 2 * degree * COEFFICIENT_BYTES;

        assert_eq!(prepared_hybrid_cache_bytes(&prepared), expected_cache);

        assert_eq!(
            prepared.logical_payload_bytes(),
            key.logical_payload_bytes() + expected_cache
        );
    }
}

/// Logical resident evaluation-key storage associated with one CKKS level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RnsCkksLevelKeyStorage {
    pub level: usize,
    pub multiplication_bytes: usize,
    pub galois_key_count: usize,
    pub galois_bytes: usize,
    pub hybrid_base_bytes: usize,
    pub prepared_cache_bytes: usize,
    pub hybrid_prepared_total_bytes: usize,
    pub total_resident_bytes: usize,
}

/// Aggregate logical resident evaluation-key storage.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RnsCkksEvaluationKeyStorage {
    pub levels: Vec<RnsCkksLevelKeyStorage>,
    pub multiplication_bytes: usize,
    pub galois_key_count: usize,
    pub galois_bytes: usize,
    pub hybrid_base_bytes: usize,
    pub prepared_cache_bytes: usize,
    pub hybrid_prepared_total_bytes: usize,
    pub total_resident_bytes: usize,
}

/// Accounts for the material physically owned by the ordinary evaluation-key
/// set and the optional hybrid multiplication policy.
///
/// A prepared hybrid level owns both:
///
/// - the standalone `HybridMultiplicationKey` stored by the level; and
/// - a `PreparedHybridMultiplicationKey`, which itself owns a clone of that
///   base key plus its helper-prime NTT cache.
///
/// Consequently `total_resident_bytes` intentionally counts both base-key
/// copies when prepared material is present.
pub fn evaluation_key_storage(
    keys: &super::RnsCkksEvaluationKeys,
    policy: Option<&super::RnsCkksMultiplicationPolicy>,
) -> RnsCkksEvaluationKeyStorage {
    use std::collections::BTreeSet;

    let mut level_numbers = BTreeSet::new();

    level_numbers.extend(keys.levels().keys().copied());

    if let Some(policy) = policy {
        level_numbers.extend(policy.hybrid_levels().keys().copied());
    }

    let mut report = RnsCkksEvaluationKeyStorage::default();

    for level in level_numbers {
        let ordinary = keys.level(level);

        let multiplication_bytes = ordinary
            .and_then(|level_keys| level_keys.multiplication_key())
            .map_or(0, LogicalPayloadBytes::logical_payload_bytes);

        let (galois_key_count, galois_bytes) = ordinary.map_or((0, 0), |level_keys| {
            (
                level_keys.galois_keys().len(),
                level_keys
                    .galois_keys()
                    .values()
                    .map(LogicalPayloadBytes::logical_payload_bytes)
                    .sum(),
            )
        });

        let hybrid = policy.and_then(|policy| policy.hybrid_levels().get(&level));

        let hybrid_base_bytes =
            hybrid.map_or(0, |level_keys| level_keys.key().logical_payload_bytes());

        let prepared_cache_bytes = hybrid
            .and_then(|level_keys| level_keys.prepared_key())
            .map_or(0, prepared_hybrid_cache_bytes);

        let hybrid_prepared_total_bytes = hybrid
            .and_then(|level_keys| level_keys.prepared_key())
            .map_or(0, LogicalPayloadBytes::logical_payload_bytes);

        let total_resident_bytes =
            multiplication_bytes + galois_bytes + hybrid_base_bytes + hybrid_prepared_total_bytes;

        let level_report = RnsCkksLevelKeyStorage {
            level,
            multiplication_bytes,
            galois_key_count,
            galois_bytes,
            hybrid_base_bytes,
            prepared_cache_bytes,
            hybrid_prepared_total_bytes,
            total_resident_bytes,
        };

        report.multiplication_bytes += multiplication_bytes;

        report.galois_key_count += galois_key_count;

        report.galois_bytes += galois_bytes;

        report.hybrid_base_bytes += hybrid_base_bytes;

        report.prepared_cache_bytes += prepared_cache_bytes;

        report.hybrid_prepared_total_bytes += hybrid_prepared_total_bytes;

        report.total_resident_bytes += total_resident_bytes;

        report.levels.push(level_report);
    }

    report
}

impl RnsCkksEvaluationKeyStorage {
    /// Compact deterministic text report suitable for logs and benchmarks.
    pub fn to_report_string(&self) -> String {
        let mut out = String::new();

        out.push_str("CKKS_EVALUATION_KEY_STORAGE\n");

        for level in &self.levels {
            out.push_str(&format!(
                "level={} multiplication_bytes={} galois_key_count={} \
galois_bytes={} hybrid_base_bytes={} prepared_cache_bytes={} \
hybrid_prepared_total_bytes={} total_resident_bytes={}\n",
                level.level,
                level.multiplication_bytes,
                level.galois_key_count,
                level.galois_bytes,
                level.hybrid_base_bytes,
                level.prepared_cache_bytes,
                level.hybrid_prepared_total_bytes,
                level.total_resident_bytes,
            ));
        }

        out.push_str(&format!(
            "TOTAL multiplication_bytes={} galois_key_count={} \
galois_bytes={} hybrid_base_bytes={} prepared_cache_bytes={} \
hybrid_prepared_total_bytes={} total_resident_bytes={}\n",
            self.multiplication_bytes,
            self.galois_key_count,
            self.galois_bytes,
            self.hybrid_base_bytes,
            self.prepared_cache_bytes,
            self.hybrid_prepared_total_bytes,
            self.total_resident_bytes,
        ));

        out
    }
}

#[cfg(test)]
mod storage_report_tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ckks::{
        conjugation_exponent, rotation_exponent_left, RnsCkksEvaluationKeys,
        RnsCkksHybridLevelKeys, RnsCkksLevelKeys, RnsCkksMultiplicationBackend,
        RnsCkksMultiplicationPolicy, RnsGaloisKey,
    };
    use crate::grafting::{
        HelperPrimeNttPlan, HybridMultiplicationKey, MixedGadgetLayout, RnsGadgetLayout,
        RnsMultiplicationKey,
    };
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

    fn ordinary_level(chain: &ModulusChain, level: usize, seed: u64) -> RnsCkksLevelKeys {
        let secret = secret();
        let basis = chain.level(level).clone();

        let blocks = match basis.len() {
            3 => vec![1, 2],
            2 => vec![1, 1],
            1 => vec![1],
            _ => panic!("unexpected test basis size"),
        };

        let mut level_keys = RnsCkksLevelKeys::new(level, basis.clone());

        if chain.has_next_level(level) {
            let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0x1000);

            level_keys.set_multiplication_key(RnsMultiplicationKey::generate_with_rng(
                secret.len(),
                2,
                0,
                &secret,
                RnsGadgetLayout::new(basis.clone(), blocks.clone()),
                &mut rng,
            ));
        }

        for (index, exponent) in [
            rotation_exponent_left(secret.len(), 1),
            conjugation_exponent(secret.len()),
        ]
        .into_iter()
        .enumerate()
        {
            let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0x2000 ^ index as u64);

            level_keys.insert_galois_key(RnsGaloisKey::generate_with_rng(
                secret.len(),
                2,
                0,
                &secret,
                exponent,
                RnsGadgetLayout::new(basis.clone(), blocks.clone()),
                &mut rng,
            ));
        }

        level_keys
    }

    fn hybrid_level(
        chain: &ModulusChain,
        level: usize,
        seed: u64,
        prepared: bool,
    ) -> RnsCkksHybridLevelKeys {
        let secret = secret();
        let basis = chain.level(level).clone();

        let blocks = match basis.len() {
            3 => vec![1, 2],
            2 => vec![1, 1],
            1 => vec![1],
            _ => panic!("unexpected test basis size"),
        };

        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        let key = HybridMultiplicationKey::generate_with_rng(
            secret.len(),
            2,
            0,
            &secret,
            MixedGadgetLayout::new(basis.clone(), blocks, 8),
            &mut rng,
        );

        let mut level_keys = RnsCkksHybridLevelKeys::new(level, basis, key);

        if prepared {
            let helper = HelperPrimeNttPlan::new(8, secret.len(), Modulus::new(2_013_265_921));

            level_keys.prepare(&helper);
        }

        level_keys
    }

    #[test]
    fn per_level_accounting_matches_owned_material_exactly() {
        let chain = chain();

        let mut ordinary = RnsCkksEvaluationKeys::new();

        ordinary.insert_level(ordinary_level(&chain, 0, 0xAD00));

        let mut policy = RnsCkksMultiplicationPolicy::new();

        let hybrid = hybrid_level(&chain, 0, 0xAD01, true);

        let hybrid_base_expected = hybrid.key().logical_payload_bytes();

        let prepared_expected = hybrid.prepared_key().unwrap().logical_payload_bytes();

        let cache_expected = prepared_hybrid_cache_bytes(hybrid.prepared_key().unwrap());

        policy.insert_hybrid_level(hybrid);

        policy.set_backend(0, RnsCkksMultiplicationBackend::HybridPrepared);

        let report = evaluation_key_storage(&ordinary, Some(&policy));

        assert_eq!(report.levels.len(), 1);

        let level = &report.levels[0];

        let ordinary_level = ordinary.level(0).unwrap();

        let multiplication_expected = ordinary_level
            .multiplication_key()
            .unwrap()
            .logical_payload_bytes();

        let galois_expected: usize = ordinary_level
            .galois_keys()
            .values()
            .map(LogicalPayloadBytes::logical_payload_bytes)
            .sum();

        assert_eq!(level.multiplication_bytes, multiplication_expected);

        assert_eq!(level.galois_key_count, ordinary_level.galois_keys().len());

        assert_eq!(level.galois_bytes, galois_expected);

        assert_eq!(level.hybrid_base_bytes, hybrid_base_expected);

        assert_eq!(level.prepared_cache_bytes, cache_expected);

        assert_eq!(level.hybrid_prepared_total_bytes, prepared_expected);

        assert_eq!(
            level.total_resident_bytes,
            multiplication_expected + galois_expected + hybrid_base_expected + prepared_expected
        );
    }

    #[test]
    fn aggregate_accounting_is_sum_of_level_reports() {
        let chain = chain();

        let mut ordinary = RnsCkksEvaluationKeys::new();

        ordinary.insert_level(ordinary_level(&chain, 0, 0xAD10));

        ordinary.insert_level(ordinary_level(&chain, 1, 0xAD11));

        let mut policy = RnsCkksMultiplicationPolicy::new();

        policy.insert_hybrid_level(hybrid_level(&chain, 0, 0xAD12, true));

        policy.insert_hybrid_level(hybrid_level(&chain, 1, 0xAD13, false));

        let report = evaluation_key_storage(&ordinary, Some(&policy));

        assert_eq!(report.levels.len(), 2);

        assert_eq!(
            report.total_resident_bytes,
            report
                .levels
                .iter()
                .map(|level| { level.total_resident_bytes })
                .sum::<usize>()
        );

        assert_eq!(
            report.multiplication_bytes,
            report
                .levels
                .iter()
                .map(|level| { level.multiplication_bytes })
                .sum::<usize>()
        );

        assert_eq!(
            report.galois_bytes,
            report
                .levels
                .iter()
                .map(|level| { level.galois_bytes })
                .sum::<usize>()
        );

        assert_eq!(
            report.hybrid_base_bytes,
            report
                .levels
                .iter()
                .map(|level| { level.hybrid_base_bytes })
                .sum::<usize>()
        );

        assert_eq!(
            report.prepared_cache_bytes,
            report
                .levels
                .iter()
                .map(|level| { level.prepared_cache_bytes })
                .sum::<usize>()
        );
    }

    #[test]
    fn report_string_contains_level_and_total_records() {
        let chain = chain();

        let mut ordinary = RnsCkksEvaluationKeys::new();

        ordinary.insert_level(ordinary_level(&chain, 0, 0xAD20));

        let report = evaluation_key_storage(&ordinary, None);

        let text = report.to_report_string();

        assert!(text.starts_with("CKKS_EVALUATION_KEY_STORAGE\n"));

        assert!(text.contains("level=0"));

        assert!(text.contains("TOTAL "));

        assert!(text.contains("total_resident_bytes="));
    }
}
