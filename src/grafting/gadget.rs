use crate::ring::{Modulus, ModulusBasis};

/// One gadget-decomposition block.
///
/// A block groups one or more RNS moduli used together by the
/// key-switching decomposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GadgetBlock {
    basis: ModulusBasis,
    contains_sprout: bool,
    consumed_limbs: usize,
}

impl GadgetBlock {
    pub fn new(basis: ModulusBasis, contains_sprout: bool) -> Self {
        Self {
            basis,
            contains_sprout,
            consumed_limbs: 0,
        }
    }

    pub fn basis(&self) -> &ModulusBasis {
        &self.basis
    }

    pub fn contains_sprout(&self) -> bool {
        self.contains_sprout
    }

    pub fn consumed_limbs(&self) -> usize {
        self.consumed_limbs
    }

    pub fn remaining_limbs(&self) -> usize {
        self.basis.len() - self.consumed_limbs
    }

    pub fn is_fresh(&self) -> bool {
        self.consumed_limbs == 0
    }

    pub fn is_exhausted(&self) -> bool {
        self.consumed_limbs == self.basis.len()
    }

    pub fn is_partially_consumed(&self) -> bool {
        self.consumed_limbs > 0 && self.consumed_limbs < self.basis.len()
    }

    pub fn consume_one(&mut self) -> Modulus {
        assert!(
            !self.is_exhausted(),
            "cannot consume exhausted gadget block"
        );

        let index = self.consumed_limbs;
        let modulus = self.basis.modulus(index);

        self.consumed_limbs += 1;

        modulus
    }

    /// Restores a partially-consumed sprout block to its original state.
    pub fn resurrect(&mut self) {
        assert!(
            self.contains_sprout,
            "only the sprout gadget block may be resurrected"
        );

        assert!(
            self.is_partially_consumed(),
            "gadget resurrection requires a partially-consumed block"
        );

        self.consumed_limbs = 0;
    }
}

/// Ordered gadget decomposition together with Grafting-specific
/// resurrection policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GadgetLayout {
    blocks: Vec<GadgetBlock>,
}

impl GadgetLayout {
    pub fn new(blocks: Vec<GadgetBlock>) -> Self {
        assert!(
            !blocks.is_empty(),
            "gadget layout must contain at least one block"
        );

        assert_eq!(
            blocks
                .iter()
                .filter(|block| block.contains_sprout())
                .count(),
            1,
            "gadget layout must contain exactly one sprout block"
        );

        Self { blocks }
    }

    pub fn blocks(&self) -> &[GadgetBlock] {
        &self.blocks
    }

    pub fn block(&self, index: usize) -> &GadgetBlock {
        &self.blocks[index]
    }

    pub fn partially_consumed_count(&self) -> usize {
        self.blocks
            .iter()
            .filter(|block| block.is_partially_consumed())
            .count()
    }

    pub fn sprout_block_index(&self) -> usize {
        self.blocks
            .iter()
            .position(|block| block.contains_sprout())
            .expect("validated gadget layout has a sprout block")
    }

    /// Consumes one modulus from a block while enforcing the Grafting
    /// invariant that at most one gadget block is partially consumed.
    pub fn consume_one(&mut self, block_index: usize) -> Modulus {
        let sprout_index = self.sprout_block_index();

        if block_index != sprout_index {
            let would_partial = self.blocks[block_index].remaining_limbs() > 1;

            if would_partial {
                assert_eq!(
                    self.partially_consumed_count(),
                    0,
                    "cannot create a second partially-consumed gadget block"
                );
            }
        }

        let modulus = self.blocks[block_index].consume_one();

        assert!(
            self.partially_consumed_count() <= 1,
            "Grafting permits at most one partially-consumed gadget block"
        );

        modulus
    }

    /// Resurrects the unique sprout-containing gadget block.
    pub fn resurrect_sprout_block(&mut self) {
        let index = self.sprout_block_index();

        self.blocks[index].resurrect();

        assert!(
            self.partially_consumed_count() <= 1,
            "gadget resurrection violated partial-block invariant"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> GadgetLayout {
        GadgetLayout::new(vec![
            GadgetBlock::new(
                ModulusBasis::new(vec![Modulus::new(12_289), Modulus::new(40_961)]),
                false,
            ),
            GadgetBlock::new(
                ModulusBasis::new(vec![Modulus::new(65_537), Modulus::new(114_689)]),
                true,
            ),
            GadgetBlock::new(
                ModulusBasis::new(vec![Modulus::new(147_457), Modulus::new(163_841)]),
                false,
            ),
        ])
    }

    #[test]
    fn layout_has_exactly_one_sprout_block() {
        let layout = layout();

        assert_eq!(layout.sprout_block_index(), 1);
    }

    #[test]
    fn partial_consumption_is_detected() {
        let mut layout = layout();

        layout.consume_one(1);

        assert_eq!(layout.partially_consumed_count(), 1);

        assert!(layout.block(1).is_partially_consumed());
    }

    #[test]
    fn sprout_block_can_be_resurrected() {
        let mut layout = layout();

        layout.consume_one(1);

        assert!(layout.block(1).is_partially_consumed());

        layout.resurrect_sprout_block();

        assert!(layout.block(1).is_fresh());

        assert_eq!(layout.partially_consumed_count(), 0);
    }

    #[test]
    fn repeated_resurrection_never_accumulates_partial_blocks() {
        let mut layout = layout();

        for _ in 0..16 {
            layout.consume_one(1);

            assert_eq!(layout.partially_consumed_count(), 1);

            layout.resurrect_sprout_block();

            assert_eq!(layout.partially_consumed_count(), 0);
        }
    }

    #[test]
    fn consuming_complete_block_does_not_leave_partial_block() {
        let mut layout = layout();

        layout.consume_one(0);
        assert_eq!(layout.partially_consumed_count(), 1);

        layout.consume_one(0);

        assert!(layout.block(0).is_exhausted());

        assert_eq!(layout.partially_consumed_count(), 0);
    }

    #[test]
    #[should_panic(expected = "second partially-consumed")]
    fn rejects_two_partially_consumed_blocks() {
        let mut layout = layout();

        layout.consume_one(0);

        // Block zero is now partial. Partially consuming another
        // non-sprout block is forbidden.
        let _ = layout.consume_one(2);
    }

    #[test]
    #[should_panic(expected = "only the sprout")]
    fn rejects_resurrection_of_non_sprout_block() {
        let mut block = GadgetBlock::new(
            ModulusBasis::new(vec![Modulus::new(12_289), Modulus::new(40_961)]),
            false,
        );

        block.consume_one();
        block.resurrect();
    }

    #[test]
    #[should_panic(expected = "exactly one sprout")]
    fn rejects_layout_without_sprout_block() {
        let _ = GadgetLayout::new(vec![GadgetBlock::new(
            ModulusBasis::new(vec![Modulus::new(12_289)]),
            false,
        )]);
    }

    #[test]
    #[should_panic(expected = "exactly one sprout")]
    fn rejects_multiple_sprout_blocks() {
        let _ = GadgetLayout::new(vec![
            GadgetBlock::new(ModulusBasis::new(vec![Modulus::new(12_289)]), true),
            GadgetBlock::new(ModulusBasis::new(vec![Modulus::new(40_961)]), true),
        ]);
    }
}
