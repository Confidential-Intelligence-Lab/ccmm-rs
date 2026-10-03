//! Deterministic shape decomposition for eBLAS GEMM.
//!
//! This module decomposes one logical GEMM into square native execution
//! regions without selecting a cryptographic backend or representation.
//! Boundary products retain their logical extents; padding into a native
//! square representation belongs to a later execution layer.

use crate::eblas::{GemmShape, MatrixShape};

fn ceil_div(value: usize, divisor: usize) -> usize {
    value
        .checked_add(divisor - 1)
        .expect("eBLAS GEMM decomposition dimension overflow")
        / divisor
}

/// One logical tile product contributing to one output tile.
///
/// For `C = A B`, this represents one `A[i,p] * B[p,j]` contribution.
/// `output_row` and `output_col` identify the destination output tile;
/// `reduction` identifies the tile along the GEMM inner dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmTileProduct {
    output_row: usize,
    output_col: usize,
    reduction: usize,
    lhs: MatrixShape,
    rhs: MatrixShape,
    output: MatrixShape,
}

impl GemmTileProduct {
    /// Output-tile row index.
    pub const fn output_row(self) -> usize {
        self.output_row
    }

    /// Output-tile column index.
    pub const fn output_col(self) -> usize {
        self.output_col
    }

    /// Reduction-tile index along logical GEMM dimension K.
    pub const fn reduction(self) -> usize {
        self.reduction
    }

    /// Logical shape of this left-hand tile.
    pub const fn lhs(self) -> MatrixShape {
        self.lhs
    }

    /// Logical shape of this right-hand tile.
    pub const fn rhs(self) -> MatrixShape {
        self.rhs
    }

    /// Logical shape of this product's output contribution.
    pub const fn output(self) -> MatrixShape {
        self.output
    }

    /// Shape-only GEMM contract for this tile product.
    pub fn gemm(self) -> GemmShape {
        GemmShape::new(self.lhs, self.rhs)
    }
}

/// Deterministic square-tile decomposition of one logical GEMM.
///
/// The tile dimension describes a native square execution extent. Logical
/// boundary tiles may be smaller and are represented with their exact shapes.
/// This type does not select CPMM, CCMM, a backend, or a padding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmDecompositionPlan {
    logical: GemmShape,
    tile_dimension: usize,
    row_tiles: usize,
    reduction_tiles: usize,
    col_tiles: usize,
}

impl GemmDecompositionPlan {
    /// Creates a deterministic square-tile decomposition.
    pub fn new(logical: GemmShape, tile_dimension: usize) -> Self {
        assert!(
            tile_dimension > 0,
            "eBLAS GEMM decomposition tile dimension must be positive"
        );

        let row_tiles = ceil_div(logical.lhs().rows(), tile_dimension);
        let reduction_tiles = ceil_div(logical.inner_dimension(), tile_dimension);
        let col_tiles = ceil_div(logical.rhs().cols(), tile_dimension);

        Self {
            logical,
            tile_dimension,
            row_tiles,
            reduction_tiles,
            col_tiles,
        }
    }

    /// Original logical GEMM shape.
    pub const fn logical(self) -> GemmShape {
        self.logical
    }

    /// Native square tile dimension requested by the plan.
    pub const fn tile_dimension(self) -> usize {
        self.tile_dimension
    }

    /// Number of tiles along logical output rows M.
    pub const fn row_tiles(self) -> usize {
        self.row_tiles
    }

    /// Number of tiles along reduction dimension K.
    pub const fn reduction_tiles(self) -> usize {
        self.reduction_tiles
    }

    /// Number of tiles along logical output columns N.
    pub const fn col_tiles(self) -> usize {
        self.col_tiles
    }

    /// Number of logical output tiles.
    pub fn output_tiles(self) -> usize {
        self.row_tiles
            .checked_mul(self.col_tiles)
            .expect("eBLAS GEMM decomposition output-tile count overflow")
    }

    /// Number of native tile products in the decomposition.
    pub fn native_products(self) -> usize {
        self.output_tiles()
            .checked_mul(self.reduction_tiles)
            .expect("eBLAS GEMM decomposition product count overflow")
    }

    /// Returns one tile product by `(output_row, reduction, output_col)`.
    pub fn product(
        self,
        output_row: usize,
        reduction: usize,
        output_col: usize,
    ) -> GemmTileProduct {
        assert!(
            output_row < self.row_tiles,
            "eBLAS GEMM decomposition output-row tile index out of range"
        );
        assert!(
            reduction < self.reduction_tiles,
            "eBLAS GEMM decomposition reduction tile index out of range"
        );
        assert!(
            output_col < self.col_tiles,
            "eBLAS GEMM decomposition output-column tile index out of range"
        );

        let d = self.tile_dimension;
        let logical = self.logical;

        let row_start = output_row
            .checked_mul(d)
            .expect("eBLAS GEMM decomposition row offset overflow");
        let reduction_start = reduction
            .checked_mul(d)
            .expect("eBLAS GEMM decomposition reduction offset overflow");
        let col_start = output_col
            .checked_mul(d)
            .expect("eBLAS GEMM decomposition column offset overflow");

        let rows = (logical.lhs().rows() - row_start).min(d);
        let inner = (logical.inner_dimension() - reduction_start).min(d);
        let cols = (logical.rhs().cols() - col_start).min(d);

        GemmTileProduct {
            output_row,
            output_col,
            reduction,
            lhs: MatrixShape::new(rows, inner),
            rhs: MatrixShape::new(inner, cols),
            output: MatrixShape::new(rows, cols),
        }
    }

    /// Iterates products in deterministic output-row, output-column,
    /// reduction order.
    pub fn products(self) -> impl Iterator<Item = GemmTileProduct> {
        (0..self.row_tiles).flat_map(move |output_row| {
            (0..self.col_tiles).flat_map(move |output_col| {
                (0..self.reduction_tiles)
                    .map(move |reduction| self.product(output_row, reduction, output_col))
            })
        })
    }

    /// Compact static accounting for this decomposition.
    pub fn count(self) -> GemmDecompositionCount {
        GemmDecompositionCount {
            output_tiles: self.output_tiles(),
            native_products: self.native_products(),
            tile_accumulations: self
                .output_tiles()
                .checked_mul(self.reduction_tiles.saturating_sub(1))
                .expect("eBLAS GEMM decomposition accumulation count overflow"),
        }
    }
}

/// Shape-only placement of one logical tile product inside a square native
/// execution extent.
///
/// Logical data occupies the upper-left region of each native operand/output.
/// Elements outside the logical region are padding owned by a later
/// representation/execution layer. This type does not allocate or materialize
/// padded matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeGemmTileMapping {
    product: GemmTileProduct,
    native_dimension: usize,
}

impl NativeGemmTileMapping {
    /// Creates a mapping into a square `native_dimension x native_dimension`
    /// execution extent.
    pub fn new(product: GemmTileProduct, native_dimension: usize) -> Self {
        assert!(
            native_dimension > 0,
            "eBLAS native GEMM tile dimension must be positive"
        );

        let lhs = product.lhs();
        let rhs = product.rhs();
        let output = product.output();

        assert!(
            lhs.rows() <= native_dimension
                && lhs.cols() <= native_dimension
                && rhs.rows() <= native_dimension
                && rhs.cols() <= native_dimension
                && output.rows() <= native_dimension
                && output.cols() <= native_dimension,
            "eBLAS logical GEMM tile does not fit native square extent"
        );

        Self {
            product,
            native_dimension,
        }
    }

    /// Logical tile product represented by this mapping.
    pub const fn product(self) -> GemmTileProduct {
        self.product
    }

    /// Square native execution dimension.
    pub const fn native_dimension(self) -> usize {
        self.native_dimension
    }

    /// Logical left-hand operand shape.
    pub const fn lhs_logical(self) -> MatrixShape {
        self.product.lhs()
    }

    /// Logical right-hand operand shape.
    pub const fn rhs_logical(self) -> MatrixShape {
        self.product.rhs()
    }

    /// Logical output contribution shape.
    pub const fn output_logical(self) -> MatrixShape {
        self.product.output()
    }

    /// Square native operand/output shape.
    pub fn native_shape(self) -> MatrixShape {
        MatrixShape::new(self.native_dimension, self.native_dimension)
    }

    fn native_elements(self) -> usize {
        self.native_dimension
            .checked_mul(self.native_dimension)
            .expect("eBLAS native GEMM tile element count overflow")
    }

    /// Number of zero-padding elements required by the left operand.
    pub fn lhs_padding_elements(self) -> usize {
        self.native_elements()
            .checked_sub(self.lhs_logical().elements())
            .expect("eBLAS native GEMM lhs padding underflow")
    }

    /// Number of zero-padding elements required by the right operand.
    pub fn rhs_padding_elements(self) -> usize {
        self.native_elements()
            .checked_sub(self.rhs_logical().elements())
            .expect("eBLAS native GEMM rhs padding underflow")
    }

    /// Number of non-logical elements in the native output extent.
    ///
    /// A later execution layer may discard these elements rather than
    /// materializing them as an assembled logical output.
    pub fn output_padding_elements(self) -> usize {
        self.native_elements()
            .checked_sub(self.output_logical().elements())
            .expect("eBLAS native GEMM output padding underflow")
    }

    /// Returns whether any operand or output requires padding.
    pub fn requires_padding(self) -> bool {
        self.lhs_padding_elements() != 0
            || self.rhs_padding_elements() != 0
            || self.output_padding_elements() != 0
    }
}

/// Static accounting for one GEMM decomposition.
///
/// These counts describe decomposition structure only. They do not estimate
/// runtime or lower-level cryptographic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmDecompositionCount {
    /// Number of logical output tiles.
    pub output_tiles: usize,
    /// Number of native tile products.
    pub native_products: usize,
    /// Number of inter-product tile accumulations.
    pub tile_accumulations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(m: usize, k: usize, n: usize) -> GemmShape {
        GemmShape::new(MatrixShape::new(m, k), MatrixShape::new(k, n))
    }

    #[test]
    fn exact_native_shape_is_one_product() {
        let plan = GemmDecompositionPlan::new(shape(128, 128, 128), 128);

        assert_eq!(plan.row_tiles(), 1);
        assert_eq!(plan.reduction_tiles(), 1);
        assert_eq!(plan.col_tiles(), 1);
        assert_eq!(
            plan.count(),
            GemmDecompositionCount {
                output_tiles: 1,
                native_products: 1,
                tile_accumulations: 0,
            }
        );

        let product = plan.product(0, 0, 0);
        assert_eq!(product.lhs(), MatrixShape::new(128, 128));
        assert_eq!(product.rhs(), MatrixShape::new(128, 128));
        assert_eq!(product.output(), MatrixShape::new(128, 128));
    }

    #[test]
    fn rectangular_shape_preserves_boundary_extents() {
        let plan = GemmDecompositionPlan::new(shape(200, 300, 100), 128);

        assert_eq!(plan.row_tiles(), 2);
        assert_eq!(plan.reduction_tiles(), 3);
        assert_eq!(plan.col_tiles(), 1);
        assert_eq!(
            plan.count(),
            GemmDecompositionCount {
                output_tiles: 2,
                native_products: 6,
                tile_accumulations: 4,
            }
        );

        let first = plan.product(0, 0, 0);
        assert_eq!(first.lhs(), MatrixShape::new(128, 128));
        assert_eq!(first.rhs(), MatrixShape::new(128, 100));
        assert_eq!(first.output(), MatrixShape::new(128, 100));

        let reduction_edge = plan.product(0, 2, 0);
        assert_eq!(reduction_edge.lhs(), MatrixShape::new(128, 44));
        assert_eq!(reduction_edge.rhs(), MatrixShape::new(44, 100));
        assert_eq!(reduction_edge.output(), MatrixShape::new(128, 100));

        let row_edge = plan.product(1, 0, 0);
        assert_eq!(row_edge.lhs(), MatrixShape::new(72, 128));
        assert_eq!(row_edge.rhs(), MatrixShape::new(128, 100));
        assert_eq!(row_edge.output(), MatrixShape::new(72, 100));

        let corner = plan.product(1, 2, 0);
        assert_eq!(corner.lhs(), MatrixShape::new(72, 44));
        assert_eq!(corner.rhs(), MatrixShape::new(44, 100));
        assert_eq!(corner.output(), MatrixShape::new(72, 100));
    }

    #[test]
    fn product_iteration_is_deterministic_and_complete() {
        let plan = GemmDecompositionPlan::new(shape(129, 257, 257), 128);
        let products: Vec<_> = plan.products().collect();

        assert_eq!(plan.row_tiles(), 2);
        assert_eq!(plan.reduction_tiles(), 3);
        assert_eq!(plan.col_tiles(), 3);
        assert_eq!(products.len(), 18);

        assert_eq!(
            products
                .iter()
                .map(|p| (p.output_row(), p.output_col(), p.reduction()))
                .collect::<Vec<_>>(),
            vec![
                (0, 0, 0),
                (0, 0, 1),
                (0, 0, 2),
                (0, 1, 0),
                (0, 1, 1),
                (0, 1, 2),
                (0, 2, 0),
                (0, 2, 1),
                (0, 2, 2),
                (1, 0, 0),
                (1, 0, 1),
                (1, 0, 2),
                (1, 1, 0),
                (1, 1, 1),
                (1, 1, 2),
                (1, 2, 0),
                (1, 2, 1),
                (1, 2, 2),
            ]
        );
    }

    #[test]
    fn decomposition_preserves_logical_scalar_product_count() {
        let logical = shape(200, 300, 100);
        let plan = GemmDecompositionPlan::new(logical, 128);

        let decomposed_products = plan
            .products()
            .map(|product| product.gemm().scalar_products())
            .sum::<usize>();

        assert_eq!(decomposed_products, logical.scalar_products());
    }

    #[test]
    fn exact_native_mapping_requires_no_padding() {
        let plan = GemmDecompositionPlan::new(shape(128, 128, 128), 128);
        let product = plan.product(0, 0, 0);

        let mapping = NativeGemmTileMapping::new(product, 128);

        assert_eq!(mapping.product(), product);
        assert_eq!(mapping.native_dimension(), 128);
        assert_eq!(mapping.native_shape(), MatrixShape::new(128, 128));
        assert_eq!(mapping.lhs_logical(), MatrixShape::new(128, 128));
        assert_eq!(mapping.rhs_logical(), MatrixShape::new(128, 128));
        assert_eq!(mapping.output_logical(), MatrixShape::new(128, 128));
        assert_eq!(mapping.lhs_padding_elements(), 0);
        assert_eq!(mapping.rhs_padding_elements(), 0);
        assert_eq!(mapping.output_padding_elements(), 0);
        assert!(!mapping.requires_padding());
    }

    #[test]
    fn boundary_mapping_accounts_for_padding_exactly() {
        let plan = GemmDecompositionPlan::new(shape(200, 300, 100), 128);
        let product = plan.product(1, 2, 0);

        assert_eq!(product.lhs(), MatrixShape::new(72, 44));
        assert_eq!(product.rhs(), MatrixShape::new(44, 100));
        assert_eq!(product.output(), MatrixShape::new(72, 100));

        let mapping = NativeGemmTileMapping::new(product, 128);

        assert_eq!(mapping.lhs_padding_elements(), 128 * 128 - 72 * 44);
        assert_eq!(mapping.rhs_padding_elements(), 128 * 128 - 44 * 100);
        assert_eq!(mapping.output_padding_elements(), 128 * 128 - 72 * 100);
        assert!(mapping.requires_padding());
    }

    #[test]
    fn native_mapping_preserves_logical_shapes() {
        let plan = GemmDecompositionPlan::new(shape(129, 257, 257), 128);

        for product in plan.products() {
            let mapping = NativeGemmTileMapping::new(product, 128);

            assert_eq!(mapping.lhs_logical(), product.lhs());
            assert_eq!(mapping.rhs_logical(), product.rhs());
            assert_eq!(mapping.output_logical(), product.output());
        }
    }

    #[test]
    #[should_panic(expected = "does not fit native square extent")]
    fn native_mapping_rejects_oversized_logical_tile() {
        let product = GemmDecompositionPlan::new(shape(128, 128, 128), 128).product(0, 0, 0);

        let _ = NativeGemmTileMapping::new(product, 64);
    }

    #[test]
    #[should_panic(expected = "native GEMM tile dimension must be positive")]
    fn native_mapping_rejects_zero_dimension() {
        let product = GemmDecompositionPlan::new(shape(1, 1, 1), 1).product(0, 0, 0);

        let _ = NativeGemmTileMapping::new(product, 0);
    }

    #[test]
    #[should_panic(expected = "tile dimension must be positive")]
    fn decomposition_rejects_zero_tile_dimension() {
        let _ = GemmDecompositionPlan::new(shape(2, 3, 4), 0);
    }
}
