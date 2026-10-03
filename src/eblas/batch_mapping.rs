//! Binding between generic native GEMM tile placement and packed Batch GEMM
//! execution geometry.
//!
//! This layer validates that a shape-only native tile mapping can be realized
//! by an explicitly selected Batch CPMM or Batch CCMM geometry. It performs no
//! packing, encryption, key preparation, or cryptographic execution.

use crate::eblas::{BatchGemmGeometry, NativeGemmTileMapping, PrivacyMode};

/// Validated binding of one native GEMM tile mapping to one packed Batch GEMM
/// execution geometry.
///
/// Privacy remains explicit: the binding verifies that the logical operation's
/// privacy contract agrees with the selected CPMM/CCMM mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchGemmTileBinding {
    mapping: NativeGemmTileMapping,
    geometry: BatchGemmGeometry,
    privacy: PrivacyMode,
}

impl BatchGemmTileBinding {
    /// Creates a valid packed-execution binding.
    pub fn new(
        mapping: NativeGemmTileMapping,
        geometry: BatchGemmGeometry,
        privacy: PrivacyMode,
    ) -> Self {
        assert_eq!(
            mapping.native_dimension(),
            geometry.dimension(),
            "eBLAS native GEMM tile dimension must match Batch GEMM geometry"
        );

        assert_eq!(
            privacy,
            geometry.privacy(),
            "eBLAS GEMM tile privacy must match Batch GEMM mechanism"
        );

        Self {
            mapping,
            geometry,
            privacy,
        }
    }

    /// Shape-only native tile mapping.
    pub const fn mapping(self) -> NativeGemmTileMapping {
        self.mapping
    }

    /// Packed execution geometry selected for this tile.
    pub const fn geometry(self) -> BatchGemmGeometry {
        self.geometry
    }

    /// Operand privacy contract validated for this binding.
    pub const fn privacy(self) -> PrivacyMode {
        self.privacy
    }

    /// Returns whether the native execution requires logical boundary padding.
    pub fn requires_padding(self) -> bool {
        self.mapping.requires_padding()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eblas::{BatchGemmMechanism, GemmDecompositionPlan, GemmShape, MatrixShape};

    fn product(
        m: usize,
        k: usize,
        n: usize,
        tile_dimension: usize,
        output_row: usize,
        reduction: usize,
        output_col: usize,
    ) -> crate::eblas::GemmTileProduct {
        let shape = GemmShape::new(MatrixShape::new(m, k), MatrixShape::new(k, n));

        GemmDecompositionPlan::new(shape, tile_dimension).product(output_row, reduction, output_col)
    }

    #[test]
    fn cp_mapping_binds_to_cpmm_d128() {
        let product = product(128, 128, 128, 128, 0, 0, 0);

        let mapping = NativeGemmTileMapping::new(product, 128);

        let geometry = BatchGemmGeometry::new(BatchGemmMechanism::Cpmm, 128, 128, 8192);

        let binding = BatchGemmTileBinding::new(mapping, geometry, PrivacyMode::Cp);

        assert_eq!(binding.mapping(), mapping);
        assert_eq!(binding.geometry(), geometry);
        assert_eq!(binding.privacy(), PrivacyMode::Cp);
        assert!(!binding.requires_padding());
    }

    #[test]
    fn cc_mapping_binds_to_ccmm_d128() {
        let product = product(128, 128, 128, 128, 0, 0, 0);

        let mapping = NativeGemmTileMapping::new(product, 128);

        let geometry = BatchGemmGeometry::new(BatchGemmMechanism::Ccmm, 128, 64, 8192);

        let binding = BatchGemmTileBinding::new(mapping, geometry, PrivacyMode::Cc);

        assert_eq!(binding.privacy(), PrivacyMode::Cc);
        assert_eq!(binding.geometry().mechanism(), BatchGemmMechanism::Ccmm);
    }

    #[test]
    fn boundary_mapping_retains_padding_state() {
        let product = product(200, 300, 100, 128, 1, 2, 0);

        let mapping = NativeGemmTileMapping::new(product, 128);

        let geometry = BatchGemmGeometry::new(BatchGemmMechanism::Cpmm, 128, 128, 8192);

        let binding = BatchGemmTileBinding::new(mapping, geometry, PrivacyMode::Cp);

        assert!(binding.requires_padding());

        assert_eq!(binding.mapping().lhs_logical(), MatrixShape::new(72, 44));
        assert_eq!(binding.mapping().rhs_logical(), MatrixShape::new(44, 100));
        assert_eq!(
            binding.mapping().output_logical(),
            MatrixShape::new(72, 100)
        );
    }

    #[test]
    #[should_panic(expected = "native GEMM tile dimension must match Batch GEMM geometry")]
    fn binding_rejects_geometry_dimension_mismatch() {
        let product = product(64, 64, 64, 64, 0, 0, 0);

        let mapping = NativeGemmTileMapping::new(product, 64);

        let geometry = BatchGemmGeometry::new(BatchGemmMechanism::Cpmm, 128, 128, 8192);

        let _ = BatchGemmTileBinding::new(mapping, geometry, PrivacyMode::Cp);
    }

    #[test]
    #[should_panic(expected = "GEMM tile privacy must match Batch GEMM mechanism")]
    fn binding_rejects_privacy_mismatch() {
        let product = product(128, 128, 128, 128, 0, 0, 0);

        let mapping = NativeGemmTileMapping::new(product, 128);

        let geometry = BatchGemmGeometry::new(BatchGemmMechanism::Cpmm, 128, 128, 8192);

        let _ = BatchGemmTileBinding::new(mapping, geometry, PrivacyMode::Cc);
    }
}
