//! Reusable encrypted FFT3840 execution service.
//!
//! This layer borrows prepared public operands and evaluation contexts.
//! It does not generate keys, encrypt inputs, or decrypt outputs.

// R29_1_PREPARED_ENCRYPTED_FFT3840_SERVICE

use crate::ckks::RnsCkksCiphertext;
use crate::eblas::fused_fft256_handoff::{
    apply_fft3840_packed_fft256_with_context, PreparedFusedFft256Source,
};
use crate::eblas::prepared_radix4_fft256::{
    PreparedFft256ExecutionContext, PreparedFft256Stage, StageContext,
};

/// A reusable FFT3840 service operating on encrypted FFT15 branches.
///
/// Each call accepts fifteen level-3 ciphertexts and returns two
/// level-7 ciphertexts containing the packed FFT3840 spectrum.
///
/// Prepared operands, evaluation keys, and Q/P plans are borrowed.
/// The caller retains ownership and controls their lifetime.
pub struct PreparedEncryptedFft3840<'a, 'ctx> {
    prepared_sources: &'a [PreparedFusedFft256Source; 15],
    later_stages: &'a [PreparedFft256Stage; 3],
    contexts: [&'a StageContext<'ctx>; 4],
    prepared_context: &'a PreparedFft256ExecutionContext,
}

impl<'a, 'ctx> PreparedEncryptedFft3840<'a, 'ctx> {
    /// Construct a service from existing prepared execution state.
    ///
    /// No cryptographic setup or key generation is performed.
    pub fn new(
        prepared_sources: &'a [PreparedFusedFft256Source; 15],
        later_stages: &'a [PreparedFft256Stage; 3],
        contexts: [&'a StageContext<'ctx>; 4],
        prepared_context: &'a PreparedFft256ExecutionContext,
    ) -> Self {
        Self {
            prepared_sources,
            later_stages,
            contexts,
            prepared_context,
        }
    }

    /// Evaluate one encrypted FFT3840 transform.
    ///
    /// The underlying production kernel enforces input and output levels.
    pub fn execute(&self, branches: &[RnsCkksCiphertext; 15]) -> [RnsCkksCiphertext; 2] {
        apply_fft3840_packed_fft256_with_context(
            branches,
            self.prepared_sources,
            self.later_stages,
            self.contexts,
            self.prepared_context,
        )
    }
}
