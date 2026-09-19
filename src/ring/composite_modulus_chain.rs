use num_bigint::BigUint;

use super::{composite_modulus_big, ModulusBasis};

/// One logical CKKS rescale unit.
///
/// A logical level may contain one or more physical RNS moduli. Consuming the
/// level means removing the complete physical group as one CKKS transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalModulusLevel {
    physical_start: usize,
    physical_end: usize,
    basis: ModulusBasis,
}

impl LogicalModulusLevel {
    fn new(physical_start: usize, physical_end: usize, basis: ModulusBasis) -> Self {
        assert!(
            physical_start < physical_end,
            "logical CKKS level must contain at least one physical modulus"
        );

        Self {
            physical_start,
            physical_end,
            basis,
        }
    }

    pub fn physical_range(&self) -> core::ops::Range<usize> {
        self.physical_start..self.physical_end
    }

    pub fn physical_limb_count(&self) -> usize {
        self.physical_end - self.physical_start
    }

    pub fn basis(&self) -> &ModulusBasis {
        &self.basis
    }

    pub fn composite_modulus_big(&self) -> BigUint {
        composite_modulus_big(&self.basis)
    }
}

/// Physical RNS basis plus explicit logical CKKS level boundaries.
///
/// The physical basis is ordered from the persistent low-end basis toward the
/// trailing rescale groups. Level 0 uses the complete physical basis.
/// Advancing one logical level removes the complete trailing logical group.
///
/// Example:
///
/// ```text
/// physical basis: [q0 q1 | q2 q3 | q4 q5]
/// logical groups:    L0      L1      L2
///
/// chain state 0:  q0 q1 q2 q3 q4 q5
/// chain state 1:  q0 q1 q2 q3
/// chain state 2:  q0 q1
/// ```
///
/// This generalizes the legacy `ModulusChain`, whose special case has exactly
/// one physical modulus per logical group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeModulusChain {
    physical_basis: ModulusBasis,
    logical_levels: Vec<LogicalModulusLevel>,
}

impl CompositeModulusChain {
    /// Creates a logical chain from physical group sizes.
    ///
    /// `group_sizes` is ordered in the same direction as the physical basis.
    /// Every physical modulus must belong to exactly one nonempty logical
    /// group.
    pub fn new(physical_basis: ModulusBasis, group_sizes: Vec<usize>) -> Self {
        assert!(
            !group_sizes.is_empty(),
            "composite modulus chain requires at least one logical level"
        );

        assert!(
            group_sizes.iter().all(|&size| size > 0),
            "logical CKKS level groups must be nonempty"
        );

        let grouped_limb_count = group_sizes
            .iter()
            .try_fold(0_usize, |total, &size| total.checked_add(size))
            .expect("logical CKKS group sizes overflow usize");

        assert_eq!(
            grouped_limb_count,
            physical_basis.len(),
            "logical CKKS level groups must cover the physical basis exactly"
        );

        let mut physical_start = 0_usize;
        let mut logical_levels = Vec::with_capacity(group_sizes.len());

        for size in group_sizes {
            let physical_end = physical_start + size;
            let basis =
                ModulusBasis::new(physical_basis.moduli()[physical_start..physical_end].to_vec());

            logical_levels.push(LogicalModulusLevel::new(
                physical_start,
                physical_end,
                basis,
            ));

            physical_start = physical_end;
        }

        Self {
            physical_basis,
            logical_levels,
        }
    }

    /// Builds the conventional special case: one physical modulus per logical
    /// CKKS level.
    pub fn single_limb_levels(physical_basis: ModulusBasis) -> Self {
        let count = physical_basis.len();
        Self::new(physical_basis, vec![1; count])
    }

    pub fn physical_basis(&self) -> &ModulusBasis {
        &self.physical_basis
    }

    pub fn physical_limb_count(&self) -> usize {
        self.physical_basis.len()
    }

    pub fn logical_levels(&self) -> &[LogicalModulusLevel] {
        &self.logical_levels
    }

    pub fn logical_level_count(&self) -> usize {
        self.logical_levels.len()
    }

    /// Highest valid CKKS chain-state index.
    ///
    /// With `k` logical groups, the chain has `k` active states and therefore
    /// `max_level() == k - 1`.
    pub fn max_level(&self) -> usize {
        self.logical_levels.len() - 1
    }

    pub fn has_next_level(&self, level: usize) -> bool {
        assert!(
            level <= self.max_level(),
            "logical CKKS level must exist in the composite chain"
        );

        level < self.max_level()
    }

    /// Physical basis active at one logical CKKS chain state.
    pub fn active_basis(&self, level: usize) -> ModulusBasis {
        assert!(
            level <= self.max_level(),
            "logical CKKS level must exist in the composite chain"
        );

        let remaining_groups = self.logical_levels.len() - level;
        let physical_end = self.logical_levels[remaining_groups - 1].physical_end;

        self.physical_basis.prefix(physical_end)
    }

    /// Logical group consumed by the transition `level -> level + 1`.
    pub fn dropped_logical_level(&self, level: usize) -> Option<&LogicalModulusLevel> {
        assert!(
            level <= self.max_level(),
            "logical CKKS level must exist in the composite chain"
        );

        if !self.has_next_level(level) {
            return None;
        }

        let dropped_index = self.logical_levels.len() - 1 - level;
        Some(&self.logical_levels[dropped_index])
    }

    /// Exact composite divisor consumed by the transition
    /// `level -> level + 1`.
    pub fn rescale_divisor_big(&self, level: usize) -> Option<BigUint> {
        self.dropped_logical_level(level)
            .map(LogicalModulusLevel::composite_modulus_big)
    }

    /// Exact active composite modulus for one logical CKKS chain state.
    pub fn active_composite_modulus_big(&self, level: usize) -> BigUint {
        composite_modulus_big(&self.active_basis(level))
    }

    /// Returns the logical group containing one physical modulus index.
    pub fn logical_group_for_physical_index(&self, physical_index: usize) -> usize {
        assert!(
            physical_index < self.physical_limb_count(),
            "physical modulus index out of bounds"
        );

        self.logical_levels
            .iter()
            .position(|group| group.physical_range().contains(&physical_index))
            .expect("validated logical groups cover the physical basis")
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;

    use crate::ring::{Modulus, ModulusBasis};

    use super::CompositeModulusChain;

    fn six_limb_basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(97),
            Modulus::new(193),
            Modulus::new(257),
            Modulus::new(449),
            Modulus::new(577),
            Modulus::new(641),
        ])
    }

    #[test]
    fn two_limb_groups_define_three_logical_levels() {
        let chain = CompositeModulusChain::new(six_limb_basis(), vec![2, 2, 2]);

        assert_eq!(chain.physical_limb_count(), 6);
        assert_eq!(chain.logical_level_count(), 3);
        assert_eq!(chain.max_level(), 2);

        assert_eq!(chain.logical_levels()[0].physical_range(), 0..2);
        assert_eq!(chain.logical_levels()[1].physical_range(), 2..4);
        assert_eq!(chain.logical_levels()[2].physical_range(), 4..6);

        assert_eq!(chain.logical_levels()[0].physical_limb_count(), 2);
        assert_eq!(chain.logical_levels()[1].physical_limb_count(), 2);
        assert_eq!(chain.logical_levels()[2].physical_limb_count(), 2);
    }

    #[test]
    fn logical_rescale_drops_complete_trailing_groups() {
        let basis = six_limb_basis();
        let chain = CompositeModulusChain::new(basis.clone(), vec![2, 2, 2]);

        assert_eq!(chain.active_basis(0), basis);
        assert_eq!(chain.active_basis(1).moduli(), &basis.moduli()[..4]);
        assert_eq!(chain.active_basis(2).moduli(), &basis.moduli()[..2]);

        assert!(chain.has_next_level(0));
        assert!(chain.has_next_level(1));
        assert!(!chain.has_next_level(2));
    }

    #[test]
    fn rescale_divisor_is_product_of_entire_logical_group() {
        let basis = six_limb_basis();
        let chain = CompositeModulusChain::new(basis.clone(), vec![2, 2, 2]);

        assert_eq!(
            chain.rescale_divisor_big(0),
            Some(BigUint::from(577_u64) * BigUint::from(641_u64))
        );

        assert_eq!(
            chain.rescale_divisor_big(1),
            Some(BigUint::from(257_u64) * BigUint::from(449_u64))
        );

        assert_eq!(chain.rescale_divisor_big(2), None);
    }

    #[test]
    fn active_composite_modulus_tracks_logical_state() {
        let basis = six_limb_basis();
        let chain = CompositeModulusChain::new(basis, vec![2, 2, 2]);

        let q0 = chain.active_composite_modulus_big(0);
        let q1 = chain.active_composite_modulus_big(1);
        let q2 = chain.active_composite_modulus_big(2);

        assert_eq!(q0, &q1 * chain.rescale_divisor_big(0).unwrap());
        assert_eq!(q1, &q2 * chain.rescale_divisor_big(1).unwrap());
    }

    #[test]
    fn single_limb_levels_preserve_legacy_chain_shape() {
        let basis = six_limb_basis();
        let chain = CompositeModulusChain::single_limb_levels(basis.clone());

        assert_eq!(chain.logical_level_count(), basis.len());
        assert_eq!(chain.max_level(), basis.len() - 1);

        for level in 0..=chain.max_level() {
            assert_eq!(chain.active_basis(level).len(), basis.len() - level);
        }

        for level in 0..chain.max_level() {
            assert_eq!(
                chain
                    .dropped_logical_level(level)
                    .unwrap()
                    .physical_limb_count(),
                1
            );
        }
    }

    #[test]
    fn physical_indices_map_to_logical_groups() {
        let chain = CompositeModulusChain::new(six_limb_basis(), vec![2, 1, 3]);

        assert_eq!(chain.logical_group_for_physical_index(0), 0);
        assert_eq!(chain.logical_group_for_physical_index(1), 0);
        assert_eq!(chain.logical_group_for_physical_index(2), 1);
        assert_eq!(chain.logical_group_for_physical_index(3), 2);
        assert_eq!(chain.logical_group_for_physical_index(4), 2);
        assert_eq!(chain.logical_group_for_physical_index(5), 2);
    }

    #[test]
    #[should_panic(expected = "must cover the physical basis exactly")]
    fn rejects_incomplete_grouping() {
        let _ = CompositeModulusChain::new(six_limb_basis(), vec![2, 2]);
    }

    #[test]
    #[should_panic(expected = "must be nonempty")]
    fn rejects_empty_logical_group() {
        let _ = CompositeModulusChain::new(six_limb_basis(), vec![2, 0, 4]);
    }
}
