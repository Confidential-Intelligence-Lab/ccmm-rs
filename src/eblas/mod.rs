//! Encrypted BLAS (eBLAS) semantic contracts.
//!
//! eBLAS separates three concerns:
//!
//! 1. the mathematical operation (for example GEMM),
//! 2. the privacy mode of the operands (PP, CP, PC, or CC), and
//! 3. the execution backend used to realize the operation.
//!
//! R3.5a intentionally introduces only contracts and validation.  Existing
//! cryptographic kernels are wired behind these contracts in later work
//! packages.

pub mod gemm;
pub use gemm::{gemm_cc, gemm_cp, gemm_pp};

/// Operand privacy for one matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandPrivacy {
    /// Plaintext matrix.
    Plaintext,
    /// Ciphertext matrix.
    Ciphertext,
}

/// Privacy mode for a binary linear-algebra operation.
///
/// The first letter describes the left operand and the second letter describes
/// the right operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivacyMode {
    /// Plaintext × plaintext.
    Pp,
    /// Ciphertext × plaintext.
    Cp,
    /// Plaintext × ciphertext.
    Pc,
    /// Ciphertext × ciphertext.
    Cc,
}

impl PrivacyMode {
    /// Returns the privacy of the left operand.
    pub const fn lhs(self) -> OperandPrivacy {
        match self {
            Self::Pp | Self::Pc => OperandPrivacy::Plaintext,
            Self::Cp | Self::Cc => OperandPrivacy::Ciphertext,
        }
    }

    /// Returns the privacy of the right operand.
    pub const fn rhs(self) -> OperandPrivacy {
        match self {
            Self::Pp | Self::Cp => OperandPrivacy::Plaintext,
            Self::Pc | Self::Cc => OperandPrivacy::Ciphertext,
        }
    }
}

/// Matrix storage convention used by the current eBLAS layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    /// Column-major storage: `index = row + column * rows`.
    ColumnMajor,
}

/// Mathematical eBLAS operation class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EblasOperation {
    /// Matrix-matrix multiplication.
    Gemm,
    /// Matrix-vector multiplication.
    Gemv,
    /// Vector dot product.
    Dot,
    /// `y <- alpha * x + y`.
    Axpy,
    /// Matrix transpose/layout operation.
    Transpose,
    /// Batched matrix-matrix multiplication.
    BatchedGemm,
}

/// Execution backend for GEMM.
///
/// Backends are deliberately distinct from [`PrivacyMode`].  A privacy mode
/// states what is public or encrypted; a backend states how the operation is
/// executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmBackend {
    /// Cleartext/reference execution.
    Reference,
    /// Existing realistic ciphertext-plaintext RNS/NTT path.
    CpDirect,
    /// Existing scalar ciphertext-ciphertext CKKS path.
    CcScalar,
    /// Matrix-structured ciphertext-ciphertext CCMM path.
    CcStructured,
}

/// Dense matrix dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    rows: usize,
    cols: usize,
}

impl MatrixShape {
    /// Creates a non-empty matrix shape.
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0, "eBLAS matrix row count must be positive");
        assert!(cols > 0, "eBLAS matrix column count must be positive");
        rows.checked_mul(cols)
            .expect("eBLAS matrix dimensions overflow");
        Self { rows, cols }
    }

    /// Number of rows.
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub const fn cols(self) -> usize {
        self.cols
    }

    /// Number of logical matrix elements.
    pub fn elements(self) -> usize {
        self.rows
            .checked_mul(self.cols)
            .expect("eBLAS matrix dimensions overflow")
    }
}

/// Shape-only GEMM contract.
///
/// For `C = A B`, `lhs = M x K`, `rhs = K x N`, and `output = M x N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmShape {
    lhs: MatrixShape,
    rhs: MatrixShape,
    output: MatrixShape,
}

impl GemmShape {
    /// Creates a valid GEMM shape.
    pub fn new(lhs: MatrixShape, rhs: MatrixShape) -> Self {
        assert_eq!(
            lhs.cols(),
            rhs.rows(),
            "eBLAS GEMM inner dimensions must match"
        );

        Self {
            lhs,
            rhs,
            output: MatrixShape::new(lhs.rows(), rhs.cols()),
        }
    }

    /// Left-hand matrix shape.
    pub const fn lhs(self) -> MatrixShape {
        self.lhs
    }

    /// Right-hand matrix shape.
    pub const fn rhs(self) -> MatrixShape {
        self.rhs
    }

    /// Output matrix shape.
    pub const fn output(self) -> MatrixShape {
        self.output
    }

    /// GEMM inner dimension `K`.
    pub const fn inner_dimension(self) -> usize {
        self.lhs.cols()
    }

    /// Number of logical scalar products in a direct GEMM schedule.
    pub fn scalar_products(self) -> usize {
        self.output
            .elements()
            .checked_mul(self.inner_dimension())
            .expect("eBLAS GEMM scalar-product count overflow")
    }

    /// Number of additions in a conventional dot-product GEMM schedule.
    pub fn scalar_additions(self) -> usize {
        self.output
            .elements()
            .checked_mul(self.inner_dimension().saturating_sub(1))
            .expect("eBLAS GEMM addition count overflow")
    }
}

/// Full GEMM semantic contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmSpec {
    shape: GemmShape,
    privacy: PrivacyMode,
    layout: MatrixLayout,
}

impl GemmSpec {
    /// Creates a column-major GEMM specification.
    pub fn new(shape: GemmShape, privacy: PrivacyMode) -> Self {
        Self {
            shape,
            privacy,
            layout: MatrixLayout::ColumnMajor,
        }
    }

    /// GEMM dimensions.
    pub const fn shape(self) -> GemmShape {
        self.shape
    }

    /// Operand privacy mode.
    pub const fn privacy(self) -> PrivacyMode {
        self.privacy
    }

    /// Matrix storage convention.
    pub const fn layout(self) -> MatrixLayout {
        self.layout
    }

    /// Returns whether a backend is currently supported for this privacy mode.
    ///
    /// R3.5a only exposes already-demonstrated execution paths:
    ///
    /// - PP + Reference
    /// - CP + CpDirect
    /// - CC + CcScalar
    /// - CC + CcStructured
    ///
    /// PC is part of the semantic contract but intentionally has no execution
    /// backend yet.
    pub const fn supports_backend(self, backend: GemmBackend) -> bool {
        matches!(
            (self.privacy, backend),
            (PrivacyMode::Pp, GemmBackend::Reference)
                | (PrivacyMode::Cp, GemmBackend::CpDirect)
                | (PrivacyMode::Cc, GemmBackend::CcScalar)
                | (PrivacyMode::Cc, GemmBackend::CcStructured)
        )
    }
}

/// Static operation accounting for a GEMM backend.
///
/// These counts describe the algorithmic schedule.  They do not estimate wall
/// time and they intentionally exclude lower-level NTT/RNS operation counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmOperationCount {
    /// Logical scalar products `M*K*N`.
    pub scalar_products: usize,
    /// Dot-product additions `M*N*(K-1)`.
    pub additions: usize,
    /// Relinearization count.
    pub relinearizations: usize,
    /// Rescale count.
    pub rescales: usize,
}

impl GemmOperationCount {
    /// Returns the static schedule for a currently supported backend.
    pub fn for_backend(spec: GemmSpec, backend: GemmBackend) -> Self {
        assert!(
            spec.supports_backend(backend),
            "eBLAS GEMM backend is not supported for this privacy mode"
        );

        let shape = spec.shape();
        let products = shape.scalar_products();
        let outputs = shape.output().elements();
        let additions = shape.scalar_additions();

        match backend {
            GemmBackend::Reference => Self {
                scalar_products: products,
                additions,
                relinearizations: 0,
                rescales: 0,
            },
            GemmBackend::CpDirect => Self {
                scalar_products: products,
                additions,
                relinearizations: 0,
                rescales: outputs,
            },
            GemmBackend::CcScalar => Self {
                scalar_products: products,
                additions,
                relinearizations: products,
                rescales: products,
            },
            GemmBackend::CcStructured => Self {
                scalar_products: products,
                additions,
                relinearizations: outputs,
                rescales: outputs,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GemmBackend, GemmOperationCount, GemmShape, GemmSpec, MatrixLayout, MatrixShape,
        OperandPrivacy, PrivacyMode,
    };

    #[test]
    fn privacy_modes_report_operand_privacy() {
        assert_eq!(PrivacyMode::Pp.lhs(), OperandPrivacy::Plaintext);
        assert_eq!(PrivacyMode::Pp.rhs(), OperandPrivacy::Plaintext);

        assert_eq!(PrivacyMode::Cp.lhs(), OperandPrivacy::Ciphertext);
        assert_eq!(PrivacyMode::Cp.rhs(), OperandPrivacy::Plaintext);

        assert_eq!(PrivacyMode::Pc.lhs(), OperandPrivacy::Plaintext);
        assert_eq!(PrivacyMode::Pc.rhs(), OperandPrivacy::Ciphertext);

        assert_eq!(PrivacyMode::Cc.lhs(), OperandPrivacy::Ciphertext);
        assert_eq!(PrivacyMode::Cc.rhs(), OperandPrivacy::Ciphertext);
    }

    #[test]
    fn gemm_shape_preserves_rectangular_dimensions() {
        let shape = GemmShape::new(MatrixShape::new(2, 4), MatrixShape::new(4, 3));

        assert_eq!(shape.lhs(), MatrixShape::new(2, 4));
        assert_eq!(shape.rhs(), MatrixShape::new(4, 3));
        assert_eq!(shape.output(), MatrixShape::new(2, 3));
        assert_eq!(shape.inner_dimension(), 4);
        assert_eq!(shape.scalar_products(), 24);
        assert_eq!(shape.scalar_additions(), 18);
    }

    #[test]
    #[should_panic(expected = "inner dimensions must match")]
    fn gemm_shape_rejects_incompatible_dimensions() {
        let _ = GemmShape::new(MatrixShape::new(2, 3), MatrixShape::new(4, 2));
    }

    #[test]
    fn gemm_contract_is_column_major() {
        let spec = GemmSpec::new(
            GemmShape::new(MatrixShape::new(2, 4), MatrixShape::new(4, 2)),
            PrivacyMode::Cc,
        );

        assert_eq!(spec.layout(), MatrixLayout::ColumnMajor);
    }

    #[test]
    fn current_backend_support_matrix_is_explicit() {
        let shape = GemmShape::new(MatrixShape::new(2, 4), MatrixShape::new(4, 2));

        assert!(GemmSpec::new(shape, PrivacyMode::Pp).supports_backend(GemmBackend::Reference));
        assert!(GemmSpec::new(shape, PrivacyMode::Cp).supports_backend(GemmBackend::CpDirect));
        assert!(GemmSpec::new(shape, PrivacyMode::Cc).supports_backend(GemmBackend::CcScalar));
        assert!(GemmSpec::new(shape, PrivacyMode::Cc).supports_backend(GemmBackend::CcStructured));

        for backend in [
            GemmBackend::Reference,
            GemmBackend::CpDirect,
            GemmBackend::CcScalar,
            GemmBackend::CcStructured,
        ] {
            assert!(!GemmSpec::new(shape, PrivacyMode::Pc).supports_backend(backend));
        }
    }

    #[test]
    fn operation_accounting_matches_r34_schedules() {
        let shape = GemmShape::new(MatrixShape::new(2, 4), MatrixShape::new(4, 2));

        let cp = GemmOperationCount::for_backend(
            GemmSpec::new(shape, PrivacyMode::Cp),
            GemmBackend::CpDirect,
        );
        assert_eq!(cp.scalar_products, 16);
        assert_eq!(cp.additions, 12);
        assert_eq!(cp.relinearizations, 0);
        assert_eq!(cp.rescales, 4);

        let scalar_cc = GemmOperationCount::for_backend(
            GemmSpec::new(shape, PrivacyMode::Cc),
            GemmBackend::CcScalar,
        );
        assert_eq!(scalar_cc.relinearizations, 16);
        assert_eq!(scalar_cc.rescales, 16);

        let structured_cc = GemmOperationCount::for_backend(
            GemmSpec::new(shape, PrivacyMode::Cc),
            GemmBackend::CcStructured,
        );
        assert_eq!(structured_cc.relinearizations, 4);
        assert_eq!(structured_cc.rescales, 4);
    }
}
