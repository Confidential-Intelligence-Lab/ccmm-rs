//! Mechanism-aware scheduling of decomposed eBLAS GEMM work.
//!
//! This module groups deterministic native GEMM tile products into the lane
//! capacity exposed by a packed Batch GEMM geometry.
//!
//! It performs no matrix materialization, SinC packing, CKKS encoding,
//! encryption, or cryptographic execution. Unused physical lanes in the final
//! group remain a later representation-layer responsibility.

use crate::eblas::{
    BatchGemmGeometry, BatchGemmTileBinding, GemmDecompositionPlan, GemmTileProduct,
    NativeGemmTileMapping, PrivacyMode,
};

/// One logical native GEMM product assigned to one active Batch GEMM lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchGemmScheduledProduct {
    ordinal: usize,
    lane: usize,
    binding: BatchGemmTileBinding,
}

impl BatchGemmScheduledProduct {
    /// Stable ordinal in `GemmDecompositionPlan::products()` order.
    pub const fn ordinal(self) -> usize {
        self.ordinal
    }

    /// Active lane within this Batch GEMM work group.
    pub const fn lane(self) -> usize {
        self.lane
    }

    /// Validated native-tile to Batch-geometry binding.
    pub const fn binding(self) -> BatchGemmTileBinding {
        self.binding
    }

    /// Logical GEMM tile product carried by this lane.
    pub const fn product(self) -> GemmTileProduct {
        self.binding.mapping().product()
    }
}

/// One packed-execution work group containing at most one geometry batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchGemmWorkGroup {
    group_index: usize,
    capacity: usize,
    products: Vec<BatchGemmScheduledProduct>,
}

impl BatchGemmWorkGroup {
    /// Zero-based work-group index.
    pub const fn group_index(&self) -> usize {
        self.group_index
    }

    /// Physical Batch GEMM lane capacity.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of active logical products in this group.
    pub fn len(&self) -> usize {
        self.products.len()
    }

    /// Whether this group contains no active products.
    pub fn is_empty(&self) -> bool {
        self.products.is_empty()
    }

    /// Whether every physical lane carries an active logical product.
    pub fn is_full(&self) -> bool {
        self.len() == self.capacity
    }

    /// Scheduled products in deterministic lane order.
    pub fn products(&self) -> &[BatchGemmScheduledProduct] {
        &self.products
    }

    /// Number of physical lanes left for representation-layer zero filling.
    pub fn unused_lanes(&self) -> usize {
        self.capacity - self.len()
    }
}

/// Groups one deterministic GEMM decomposition into packed Batch GEMM work.
///
/// Products retain the canonical decomposition order:
/// output row, output column, reduction.
///
/// Each full group contains `geometry.batch_count()` active products. The final
/// group may contain fewer active products; physical zero-filling of unused
/// SinC lanes belongs to the representation layer.
pub fn batch_gemm_work_groups(
    plan: GemmDecompositionPlan,
    geometry: BatchGemmGeometry,
    privacy: PrivacyMode,
) -> Vec<BatchGemmWorkGroup> {
    assert_eq!(
        plan.tile_dimension(),
        geometry.dimension(),
        "eBLAS decomposition tile dimension must match Batch GEMM geometry"
    );

    let capacity = geometry.batch_count();
    assert!(
        capacity > 0,
        "Batch GEMM geometry must expose positive lane capacity"
    );

    let product_count = plan.native_products();
    let group_count = product_count
        .checked_add(capacity - 1)
        .expect("eBLAS Batch GEMM work-group count overflow")
        / capacity;

    let mut groups = Vec::with_capacity(group_count);

    for (ordinal, product) in plan.products().enumerate() {
        let group_index = ordinal / capacity;
        let lane = ordinal % capacity;

        if lane == 0 {
            groups.push(BatchGemmWorkGroup {
                group_index,
                capacity,
                products: Vec::with_capacity(capacity.min(product_count - ordinal)),
            });
        }

        let mapping = NativeGemmTileMapping::new(product, plan.tile_dimension());
        let binding = BatchGemmTileBinding::new(mapping, geometry, privacy);

        groups
            .last_mut()
            .expect("eBLAS Batch GEMM scheduler must create a group before inserting work")
            .products
            .push(BatchGemmScheduledProduct {
                ordinal,
                lane,
                binding,
            });
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eblas::{BatchGemmMechanism, GemmShape, MatrixShape};

    fn shape(m: usize, k: usize, n: usize) -> GemmShape {
        GemmShape::new(MatrixShape::new(m, k), MatrixShape::new(k, n))
    }

    fn cpmm_d128() -> BatchGemmGeometry {
        BatchGemmGeometry::new(BatchGemmMechanism::Cpmm, 128, 128, 8192)
    }

    fn ccmm_d128() -> BatchGemmGeometry {
        BatchGemmGeometry::new(BatchGemmMechanism::Ccmm, 128, 64, 8192)
    }

    #[test]
    fn single_product_uses_one_active_lane() {
        let plan = GemmDecompositionPlan::new(shape(128, 128, 128), 128);
        let groups = batch_gemm_work_groups(plan, cpmm_d128(), PrivacyMode::Cp);

        assert_eq!(groups.len(), 1);

        let group = &groups[0];
        assert_eq!(group.group_index(), 0);
        assert_eq!(group.capacity(), 64);
        assert_eq!(group.len(), 1);
        assert!(!group.is_empty());
        assert!(!group.is_full());
        assert_eq!(group.unused_lanes(), 63);

        let scheduled = group.products()[0];
        assert_eq!(scheduled.ordinal(), 0);
        assert_eq!(scheduled.lane(), 0);
        assert_eq!(scheduled.product(), plan.product(0, 0, 0));
    }

    #[test]
    fn exact_capacity_produces_one_full_group() {
        // 4 row tiles * 4 column tiles * 4 reduction tiles = 64 products.
        let plan = GemmDecompositionPlan::new(shape(512, 512, 512), 128);
        assert_eq!(plan.native_products(), 64);

        let groups = batch_gemm_work_groups(plan, cpmm_d128(), PrivacyMode::Cp);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 64);
        assert!(groups[0].is_full());
        assert_eq!(groups[0].unused_lanes(), 0);

        for (lane, scheduled) in groups[0].products().iter().enumerate() {
            assert_eq!(scheduled.ordinal(), lane);
            assert_eq!(scheduled.lane(), lane);
        }
    }

    #[test]
    fn partial_tail_preserves_capacity_and_lane_numbering() {
        // 1 row tile * 13 column tiles * 5 reduction tiles = 65 products.
        let plan = GemmDecompositionPlan::new(shape(128, 640, 1664), 128);
        assert_eq!(plan.native_products(), 65);

        let groups = batch_gemm_work_groups(plan, cpmm_d128(), PrivacyMode::Cp);

        assert_eq!(groups.len(), 2);

        assert_eq!(groups[0].group_index(), 0);
        assert_eq!(groups[0].len(), 64);
        assert!(groups[0].is_full());

        assert_eq!(groups[1].group_index(), 1);
        assert_eq!(groups[1].capacity(), 64);
        assert_eq!(groups[1].len(), 1);
        assert_eq!(groups[1].unused_lanes(), 63);

        let tail = groups[1].products()[0];
        assert_eq!(tail.ordinal(), 64);
        assert_eq!(tail.lane(), 0);
    }

    #[test]
    fn flattened_schedule_preserves_decomposition_order() {
        let plan = GemmDecompositionPlan::new(shape(384, 640, 512), 128);
        let expected: Vec<_> = plan.products().collect();

        let groups = batch_gemm_work_groups(plan, cpmm_d128(), PrivacyMode::Cp);
        let actual: Vec<_> = groups
            .iter()
            .flat_map(|group| group.products().iter())
            .map(|scheduled| scheduled.product())
            .collect();

        assert_eq!(actual, expected);

        for (ordinal, scheduled) in groups
            .iter()
            .flat_map(|group| group.products().iter())
            .enumerate()
        {
            assert_eq!(scheduled.ordinal(), ordinal);
            assert_eq!(scheduled.lane(), ordinal % cpmm_d128().batch_count());
        }
    }

    #[test]
    fn ccmm_uses_geometry_specific_batch_capacity() {
        // 64 products: two full CCMM d128 groups because B = 32.
        let plan = GemmDecompositionPlan::new(shape(512, 512, 512), 128);
        assert_eq!(plan.native_products(), 64);

        let groups = batch_gemm_work_groups(plan, ccmm_d128(), PrivacyMode::Cc);

        assert_eq!(groups.len(), 2);

        for (group_index, group) in groups.iter().enumerate() {
            assert_eq!(group.group_index(), group_index);
            assert_eq!(group.capacity(), 32);
            assert_eq!(group.len(), 32);
            assert!(group.is_full());
            assert_eq!(group.unused_lanes(), 0);
        }

        assert_eq!(groups[1].products()[0].ordinal(), 32);
        assert_eq!(groups[1].products()[0].lane(), 0);
    }

    #[test]
    #[should_panic(expected = "decomposition tile dimension must match Batch GEMM geometry")]
    fn scheduler_rejects_geometry_dimension_mismatch() {
        let plan = GemmDecompositionPlan::new(shape(64, 64, 64), 64);

        let _ = batch_gemm_work_groups(plan, cpmm_d128(), PrivacyMode::Cp);
    }
}
