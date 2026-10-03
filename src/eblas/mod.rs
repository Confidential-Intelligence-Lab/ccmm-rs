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

pub mod backend_policy;
pub mod correlation;

pub mod batched_gemm;

pub mod batch_gemm;
pub mod batch_mapping;
pub mod batch_representation;
pub mod batch_schedule;

pub mod decomposed_ccmm;
pub mod decomposed_cpmm;
pub mod decomposed_gemm;
pub mod decomposition;
pub mod gemm;
pub mod native_representation;
pub use backend_policy::select_cc_backend;
pub use batch_schedule::{batch_gemm_work_groups, BatchGemmScheduledProduct, BatchGemmWorkGroup};
pub use decomposed_ccmm::{gemm_cc_decomposed, DecomposedCcmmExecutionContext};
pub use decomposed_cpmm::{gemm_cp_decomposed, DecomposedCpmmExecutionContext};
pub use decomposed_gemm::gemm_pp_decomposed;
pub use decomposition::{
    GemmDecompositionCount, GemmDecompositionPlan, GemmTileProduct, NativeGemmTileMapping,
};

pub use batched_gemm::{
    batched_gemm_cc, batched_gemm_cc_auto, batched_gemm_cp, batched_gemm_pc, batched_gemm_pp,
    BatchedGemmOperationCount, BatchedGemmShape, BatchedGemmSpec,
};

pub use batch_gemm::{
    batch_dot_ccmm, batch_dot_cpmm, batch_gemm_ccmm, batch_gemm_cpmm, batch_gemv_ccmm,
    batch_gemv_cpmm,
};
pub use batch_mapping::BatchGemmTileBinding;
pub use batch_representation::{represent_batch_gemm_work_group, BatchGemmRepresentation};

pub use gemm::{gemm_cc, gemm_cc_auto, gemm_cc_structured_observed, gemm_cp, gemm_pc, gemm_pp};
pub use native_representation::{
    crop_native_output_tile, pad_native_lhs_tile, pad_native_rhs_tile,
};
pub mod gemv;
pub use gemv::{
    dot_cc, dot_cc_auto, dot_cp, dot_pc, dot_pp, gemv_cc, gemv_cc_auto, gemv_cp, gemv_pc, gemv_pp,
    DotShape, GemvShape,
};

pub mod level1;

pub use level1::{add_cc, axpy_cp, scale_cp};

pub mod transpose;

pub mod tensor_mapping;
pub use transpose::{transpose_cipher, transpose_plain, transpose_pp};

pub use tensor_mapping::{
    batch_matrix_to_tensor3, flatten_nhwc_to_matrix, tensor3_to_batch_matrix, tensor_gemm_spec,
    unflatten_matrix_to_nhwc, Im2ColShape, NhwcShape, TensorBatchShape, TensorGemmShape,
};

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
    /// Valid correlation.
    Correlation,
    /// Element-wise addition.
    Add,
    /// Public-scalar scaling.
    Scale,
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

/// Execution mechanism for packed/batched encrypted GEMM.
///
/// These mechanisms implement the same logical batched matrix-multiplication
/// operation while using the SinC representation and the large-ring/scalar-ring
/// decomposition employed by the Batch CPMM and Batch CCMM algorithms.
///
/// The mechanism is deliberately separate from [`PrivacyMode`]:
///
/// - [`BatchGemmMechanism::Cpmm`] realizes ciphertext/plaintext GEMM;
/// - [`BatchGemmMechanism::Ccmm`] realizes ciphertext/ciphertext GEMM.
///
/// Keeping mechanism selection explicit makes characterization reproducible
/// and allows the implementations to evolve without changing the eBLAS
/// operation contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchGemmMechanism {
    /// Packed ciphertext/plaintext matrix multiplication.
    Cpmm,
    /// Packed ciphertext/ciphertext matrix multiplication.
    Ccmm,
}

impl BatchGemmMechanism {
    /// Operand-privacy contract required by this mechanism.
    pub const fn privacy(self) -> PrivacyMode {
        match self {
            Self::Cpmm => PrivacyMode::Cp,
            Self::Ccmm => PrivacyMode::Cc,
        }
    }
}

/// Geometry of one packed Batch GEMM execution.
///
/// Batch CPMM and Batch CCMM use related SinC representations but different
/// large-ring decomposition geometries:
///
/// - CPMM: `(dimension / 2) * scalar_degree == large_degree`;
/// - CCMM: `dimension * scalar_degree == large_degree`.
///
/// The number of simultaneously represented real matrix products is
/// `scalar_degree / 2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchGemmGeometry {
    mechanism: BatchGemmMechanism,
    dimension: usize,
    scalar_degree: usize,
    large_degree: usize,
}

impl BatchGemmGeometry {
    /// Creates a valid Batch GEMM geometry for the selected mechanism.
    pub fn new(
        mechanism: BatchGemmMechanism,
        dimension: usize,
        scalar_degree: usize,
        large_degree: usize,
    ) -> Self {
        assert!(dimension > 0, "Batch GEMM dimension must be positive");
        assert!(
            scalar_degree > 0 && scalar_degree % 2 == 0,
            "Batch GEMM scalar degree must be positive and even"
        );

        let decomposition_rows = match mechanism {
            BatchGemmMechanism::Cpmm => {
                assert!(dimension % 2 == 0, "Batch CPMM dimension must be even");
                dimension / 2
            }
            BatchGemmMechanism::Ccmm => dimension,
        };

        assert_eq!(
            decomposition_rows
                .checked_mul(scalar_degree)
                .expect("Batch GEMM geometry overflow"),
            large_degree,
            "Batch GEMM decomposition must match the large-ring degree"
        );

        Self {
            mechanism,
            dimension,
            scalar_degree,
            large_degree,
        }
    }

    /// Selected packed matrix-multiplication mechanism.
    pub const fn mechanism(self) -> BatchGemmMechanism {
        self.mechanism
    }

    /// Operand-privacy contract implied by the selected mechanism.
    pub const fn privacy(self) -> PrivacyMode {
        self.mechanism.privacy()
    }

    /// Logical square-matrix dimension.
    pub const fn dimension(self) -> usize {
        self.dimension
    }

    /// Degree of each scalar-ring component.
    pub const fn scalar_degree(self) -> usize {
        self.scalar_degree
    }

    /// Degree of the packed large ring.
    pub const fn large_degree(self) -> usize {
        self.large_degree
    }

    /// Number of real matrix products represented simultaneously.
    pub const fn batch_count(self) -> usize {
        self.scalar_degree / 2
    }
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
    /// PC reuses the CP direct backend through transpose reduction.
    pub const fn supports_backend(self, backend: GemmBackend) -> bool {
        matches!(
            (self.privacy, backend),
            (PrivacyMode::Pp, GemmBackend::Reference)
                | (PrivacyMode::Cp, GemmBackend::CpDirect)
                | (PrivacyMode::Pc, GemmBackend::CpDirect)
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

/// Returns the logical representation profile for one GEMM operation.
///
/// The current eBLAS RNS-CKKS matrix representation stores one logical matrix
/// element per ciphertext/plaintext object. `slots_available` records the
/// underlying CKKS SIMD capacity so that future packed implementations can be
/// compared against this scalar-per-object baseline.
pub fn gemm_representation_profile(
    spec: GemmSpec,
    slots_available: usize,
) -> crate::execution::RepresentationProfile {
    use crate::execution::{BatchingStrategy, RepresentationProfile};

    assert!(
        slots_available > 0,
        "eBLAS representation profile requires positive CKKS slot capacity"
    );

    let lhs = spec.shape().lhs().elements();
    let rhs = spec.shape().rhs().elements();
    let output = spec.shape().output().elements();

    let (input_ciphertexts, input_plaintexts, output_ciphertexts, output_plaintexts) =
        match spec.privacy() {
            PrivacyMode::Pp => (0, lhs + rhs, 0, output),
            PrivacyMode::Cp => (lhs, rhs, output, 0),
            PrivacyMode::Pc => (rhs, lhs, output, 0),
            PrivacyMode::Cc => (lhs + rhs, 0, output, 0),
        };

    RepresentationProfile {
        batching: BatchingStrategy::ScalarPerCiphertext,
        input_ciphertexts,
        input_plaintexts,
        output_ciphertexts,
        output_plaintexts,
        slots_used_per_ciphertext: 1,
        slots_available,
    }
}

/// Returns the compact device-neutral execution profile for one GEMM schedule.
///
/// `GemmOperationCount` remains the authoritative eBLAS algorithmic schedule.
/// This function translates that schedule into the common execution vocabulary
/// without expanding individual events.
pub fn gemm_execution_profile(
    spec: GemmSpec,
    backend: GemmBackend,
) -> crate::execution::ExecutionProfile {
    use crate::execution::ExecutionProfile;

    let count = GemmOperationCount::for_backend(spec, backend);

    match backend {
        GemmBackend::Reference => ExecutionProfile {
            additions: count.additions,
            ..ExecutionProfile::default()
        },

        GemmBackend::CpDirect => ExecutionProfile {
            ciphertext_plaintext_multiplies: count.scalar_products,
            additions: count.additions,
            rescales: count.rescales,
            ..ExecutionProfile::default()
        },

        GemmBackend::CcScalar | GemmBackend::CcStructured => ExecutionProfile {
            ciphertext_ciphertext_multiplies: count.scalar_products,
            additions: count.additions,
            relinearizations: count.relinearizations,
            rescales: count.rescales,
            ..ExecutionProfile::default()
        },
    }
}

/// Expands one GEMM static schedule into a device-neutral semantic trace.
///
/// This helper is intended for bounded characterization and validation. It
/// materializes one event per predicted logical operation and therefore must
/// not be used as the scalable cost representation for large workloads.
/// `GemmOperationCount` remains the compact analytical schedule.
///
/// This trace is predicted rather than observed: it is derived from the eBLAS
/// contract and selected backend, does not observe runtime execution, and does
/// not include lower-level NTT/RNS events.
pub fn predicted_gemm_execution_trace(
    spec: GemmSpec,
    backend: GemmBackend,
    device: crate::execution::ExecutionDevice,
) -> crate::execution::ExecutionTrace {
    use crate::execution::{ExecutionEvent, ExecutionEventKind, ExecutionTrace};

    let count = GemmOperationCount::for_backend(spec, backend);
    let mut trace = ExecutionTrace::new(device);

    trace.record(ExecutionEvent::new(ExecutionEventKind::EblasGemm));

    match backend {
        GemmBackend::Reference => {}

        GemmBackend::CpDirect => {
            trace.record(ExecutionEvent::new(ExecutionEventKind::Cpmm));

            for _ in 0..count.scalar_products {
                trace.record(ExecutionEvent::new(
                    ExecutionEventKind::CiphertextPlaintextMultiply,
                ));
            }

            for _ in 0..count.additions {
                trace.record(ExecutionEvent::new(ExecutionEventKind::Add));
            }

            for _ in 0..count.rescales {
                trace.record(ExecutionEvent::new(ExecutionEventKind::Rescale));
            }
        }

        GemmBackend::CcScalar => {
            trace.record(ExecutionEvent::new(ExecutionEventKind::CcmmScalar));

            for _ in 0..count.scalar_products {
                trace.record(ExecutionEvent::new(
                    ExecutionEventKind::CiphertextCiphertextMultiply,
                ));
                trace.record(ExecutionEvent::new(ExecutionEventKind::Relinearize));
                trace.record(ExecutionEvent::new(ExecutionEventKind::Rescale));
            }

            for _ in 0..count.additions {
                trace.record(ExecutionEvent::new(ExecutionEventKind::Add));
            }
        }

        GemmBackend::CcStructured => {
            trace.record(ExecutionEvent::new(ExecutionEventKind::CcmmStructured));

            for _ in 0..count.scalar_products {
                trace.record(ExecutionEvent::new(
                    ExecutionEventKind::CiphertextCiphertextMultiply,
                ));
            }

            for _ in 0..count.additions {
                trace.record(ExecutionEvent::new(ExecutionEventKind::Add));
            }

            for _ in 0..count.relinearizations {
                trace.record(ExecutionEvent::new(ExecutionEventKind::Relinearize));
            }

            for _ in 0..count.rescales {
                trace.record(ExecutionEvent::new(ExecutionEventKind::Rescale));
            }
        }
    }

    trace
}

#[cfg(test)]
mod tests {

    #[test]
    fn batch_gemm_authors_geometries_are_explicit() {
        let cpmm_d64 = BatchGemmGeometry::new(BatchGemmMechanism::Cpmm, 64, 256, 8192);
        assert_eq!(cpmm_d64.batch_count(), 128);
        assert_eq!(cpmm_d64.privacy(), PrivacyMode::Cp);

        let cpmm_d128 = BatchGemmGeometry::new(BatchGemmMechanism::Cpmm, 128, 128, 8192);
        assert_eq!(cpmm_d128.batch_count(), 64);
        assert_eq!(cpmm_d128.privacy(), PrivacyMode::Cp);

        let ccmm_d64 = BatchGemmGeometry::new(BatchGemmMechanism::Ccmm, 64, 128, 8192);
        assert_eq!(ccmm_d64.batch_count(), 64);
        assert_eq!(ccmm_d64.privacy(), PrivacyMode::Cc);

        let ccmm_d128 = BatchGemmGeometry::new(BatchGemmMechanism::Ccmm, 128, 64, 8192);
        assert_eq!(ccmm_d128.batch_count(), 32);
        assert_eq!(ccmm_d128.privacy(), PrivacyMode::Cc);
    }

    #[test]
    #[should_panic(expected = "Batch GEMM decomposition must match the large-ring degree")]
    fn batch_cpmm_rejects_ccmm_geometry() {
        let _ = BatchGemmGeometry::new(BatchGemmMechanism::Cpmm, 64, 128, 8192);
    }

    #[test]
    #[should_panic(expected = "Batch GEMM decomposition must match the large-ring degree")]
    fn batch_ccmm_rejects_cpmm_geometry() {
        let _ = BatchGemmGeometry::new(BatchGemmMechanism::Ccmm, 64, 256, 8192);
    }

    use super::gemm_representation_profile;
    use super::{
        gemm_execution_profile, predicted_gemm_execution_trace, BatchGemmGeometry,
        BatchGemmMechanism, GemmBackend, GemmOperationCount, GemmShape, GemmSpec, MatrixLayout,
        MatrixShape, OperandPrivacy, PrivacyMode,
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

        assert!(!GemmSpec::new(shape, PrivacyMode::Pc).supports_backend(GemmBackend::Reference));
        assert!(GemmSpec::new(shape, PrivacyMode::Pc).supports_backend(GemmBackend::CpDirect));
        assert!(!GemmSpec::new(shape, PrivacyMode::Pc).supports_backend(GemmBackend::CcScalar));
        assert!(!GemmSpec::new(shape, PrivacyMode::Pc).supports_backend(GemmBackend::CcStructured));
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

    #[test]
    fn predicted_cp_trace_matches_static_operation_count() {
        use crate::execution::{ExecutionDevice, ExecutionEventKind};

        let spec = GemmSpec::new(
            GemmShape::new(MatrixShape::new(2, 3), MatrixShape::new(3, 4)),
            PrivacyMode::Cp,
        );

        let trace =
            predicted_gemm_execution_trace(spec, GemmBackend::CpDirect, ExecutionDevice::Cpu);
        let count = GemmOperationCount::for_backend(spec, GemmBackend::CpDirect);

        assert_eq!(trace.count(ExecutionEventKind::EblasGemm), 1);
        assert_eq!(trace.count(ExecutionEventKind::Cpmm), 1);
        assert_eq!(
            trace.count(ExecutionEventKind::CiphertextPlaintextMultiply),
            count.scalar_products
        );
        assert_eq!(trace.count(ExecutionEventKind::Add), count.additions);
        assert_eq!(trace.count(ExecutionEventKind::Relinearize), 0);
        assert_eq!(trace.count(ExecutionEventKind::Rescale), count.rescales);
    }

    #[test]
    fn predicted_scalar_cc_trace_matches_static_operation_count() {
        use crate::execution::{ExecutionDevice, ExecutionEventKind};

        let spec = GemmSpec::new(
            GemmShape::new(MatrixShape::new(2, 3), MatrixShape::new(3, 4)),
            PrivacyMode::Cc,
        );

        let trace =
            predicted_gemm_execution_trace(spec, GemmBackend::CcScalar, ExecutionDevice::Cpu);
        let count = GemmOperationCount::for_backend(spec, GemmBackend::CcScalar);

        assert_eq!(trace.count(ExecutionEventKind::EblasGemm), 1);
        assert_eq!(trace.count(ExecutionEventKind::CcmmScalar), 1);
        assert_eq!(
            trace.count(ExecutionEventKind::CiphertextCiphertextMultiply),
            count.scalar_products
        );
        assert_eq!(trace.count(ExecutionEventKind::Add), count.additions);
        assert_eq!(
            trace.count(ExecutionEventKind::Relinearize),
            count.relinearizations
        );
        assert_eq!(trace.count(ExecutionEventKind::Rescale), count.rescales);
    }

    #[test]
    fn predicted_structured_cc_trace_matches_static_operation_count() {
        use crate::execution::{ExecutionDevice, ExecutionEventKind};

        let spec = GemmSpec::new(
            GemmShape::new(MatrixShape::new(2, 3), MatrixShape::new(3, 4)),
            PrivacyMode::Cc,
        );

        let trace =
            predicted_gemm_execution_trace(spec, GemmBackend::CcStructured, ExecutionDevice::Cpu);
        let count = GemmOperationCount::for_backend(spec, GemmBackend::CcStructured);

        assert_eq!(trace.count(ExecutionEventKind::EblasGemm), 1);
        assert_eq!(trace.count(ExecutionEventKind::CcmmStructured), 1);
        assert_eq!(
            trace.count(ExecutionEventKind::CiphertextCiphertextMultiply),
            count.scalar_products
        );
        assert_eq!(trace.count(ExecutionEventKind::Add), count.additions);
        assert_eq!(
            trace.count(ExecutionEventKind::Relinearize),
            count.relinearizations
        );
        assert_eq!(trace.count(ExecutionEventKind::Rescale), count.rescales);
    }

    #[test]
    fn structured_cc_trace_reduces_expensive_post_product_operations() {
        use crate::execution::{ExecutionDevice, ExecutionEventKind};

        let spec = GemmSpec::new(
            GemmShape::new(MatrixShape::new(4, 8), MatrixShape::new(8, 4)),
            PrivacyMode::Cc,
        );

        let scalar =
            predicted_gemm_execution_trace(spec, GemmBackend::CcScalar, ExecutionDevice::Cpu);
        let structured =
            predicted_gemm_execution_trace(spec, GemmBackend::CcStructured, ExecutionDevice::Cpu);

        assert_eq!(
            scalar.count(ExecutionEventKind::CiphertextCiphertextMultiply),
            structured.count(ExecutionEventKind::CiphertextCiphertextMultiply)
        );

        assert_eq!(
            scalar.count(ExecutionEventKind::Relinearize),
            8 * structured.count(ExecutionEventKind::Relinearize)
        );

        assert_eq!(
            scalar.count(ExecutionEventKind::Rescale),
            8 * structured.count(ExecutionEventKind::Rescale)
        );
    }

    #[test]
    fn gemm_execution_profiles_match_static_schedules() {
        let shape = GemmShape::new(MatrixShape::new(4, 8), MatrixShape::new(8, 4));

        let cp_spec = GemmSpec::new(shape, PrivacyMode::Cp);
        let cc_spec = GemmSpec::new(shape, PrivacyMode::Cc);

        let cp_count = GemmOperationCount::for_backend(cp_spec, GemmBackend::CpDirect);
        let cp = gemm_execution_profile(cp_spec, GemmBackend::CpDirect);

        assert_eq!(cp.ciphertext_plaintext_multiplies, cp_count.scalar_products);
        assert_eq!(cp.additions, cp_count.additions);
        assert_eq!(cp.relinearizations, 0);
        assert_eq!(cp.rescales, cp_count.rescales);

        let scalar_count = GemmOperationCount::for_backend(cc_spec, GemmBackend::CcScalar);
        let scalar = gemm_execution_profile(cc_spec, GemmBackend::CcScalar);

        assert_eq!(
            scalar.ciphertext_ciphertext_multiplies,
            scalar_count.scalar_products
        );
        assert_eq!(scalar.additions, scalar_count.additions);
        assert_eq!(scalar.relinearizations, scalar_count.relinearizations);
        assert_eq!(scalar.rescales, scalar_count.rescales);

        let structured_count = GemmOperationCount::for_backend(cc_spec, GemmBackend::CcStructured);
        let structured = gemm_execution_profile(cc_spec, GemmBackend::CcStructured);

        assert_eq!(
            structured.ciphertext_ciphertext_multiplies,
            structured_count.scalar_products
        );
        assert_eq!(structured.additions, structured_count.additions);
        assert_eq!(
            structured.relinearizations,
            structured_count.relinearizations
        );
        assert_eq!(structured.rescales, structured_count.rescales);

        assert_eq!(
            scalar.ciphertext_ciphertext_multiplies,
            structured.ciphertext_ciphertext_multiplies
        );
        assert_eq!(scalar.relinearizations, 8 * structured.relinearizations);
        assert_eq!(scalar.rescales, 8 * structured.rescales);
    }

    #[test]
    fn gemm_representation_profiles_match_privacy_modes() {
        use crate::execution::BatchingStrategy;

        let shape = GemmShape::new(MatrixShape::new(2, 3), MatrixShape::new(3, 4));

        let pp = gemm_representation_profile(GemmSpec::new(shape, PrivacyMode::Pp), 2048);
        assert_eq!(pp.batching, BatchingStrategy::ScalarPerCiphertext);
        assert_eq!(pp.input_ciphertexts, 0);
        assert_eq!(pp.input_plaintexts, 6 + 12);
        assert_eq!(pp.output_ciphertexts, 0);
        assert_eq!(pp.output_plaintexts, 8);
        assert_eq!(pp.slots_used_per_ciphertext, 1);
        assert_eq!(pp.slots_available, 2048);

        let cp = gemm_representation_profile(GemmSpec::new(shape, PrivacyMode::Cp), 2048);
        assert_eq!(cp.input_ciphertexts, 6);
        assert_eq!(cp.input_plaintexts, 12);
        assert_eq!(cp.output_ciphertexts, 8);
        assert_eq!(cp.output_plaintexts, 0);

        let pc = gemm_representation_profile(GemmSpec::new(shape, PrivacyMode::Pc), 2048);
        assert_eq!(pc.input_ciphertexts, 12);
        assert_eq!(pc.input_plaintexts, 6);
        assert_eq!(pc.output_ciphertexts, 8);
        assert_eq!(pc.output_plaintexts, 0);

        let cc = gemm_representation_profile(GemmSpec::new(shape, PrivacyMode::Cc), 2048);
        assert_eq!(cc.input_ciphertexts, 6 + 12);
        assert_eq!(cc.input_plaintexts, 0);
        assert_eq!(cc.output_ciphertexts, 8);
        assert_eq!(cc.output_plaintexts, 0);

        assert!((cc.packing_utilization() - (1.0 / 2048.0)).abs() < 1.0e-15);
    }
}
