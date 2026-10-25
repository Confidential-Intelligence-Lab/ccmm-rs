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

// R29_2B1_CIPHERTEXT_ONLY_FFT15

use crate::ckks::{
    rotate_left_bounded_rns_ckks_with_prepared_execution, CkksCanonicalEmbedding,
    PreparedBoundedRnsGaloisKey, RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use crate::eblas::level1::{multiply_complex_slots_cp_prepared, PreparedComplexSlotsCp};
use crate::eblas::mixed_radix_ckks::{
    packed_fft15_twiddled_radix3_cp_prepared, packed_radix5_dft_cp_prepared,
    PreparedFft15TwiddledRadix3Cp, PreparedRadix5Cp,
};
use crate::ring::{ModulusBasis, ModulusChain, PreparedRnsNttPlan, RnsNttPlan};
use num_complex::Complex64;

// R29_5_PARALLEL_FFT3840
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::OnceLock;

pub(crate) fn fft3840_workers() -> usize {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let requested = std::env::var("FHE_R29_FFT_WORKERS")
        .or_else(|_| std::env::var("FHE_RS_WORKERS"))
        .ok()
        .map(|value| value.parse::<usize>().expect("invalid FFT worker count"))
        .unwrap_or(available);

    assert!(requested > 0, "FFT worker count must be positive");
    requested.min(available)
}

pub(crate) fn fft3840_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();

    POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(fft3840_workers())
            .build()
            .expect("FFT3840 pool initialization failed")
    })
}

/// Evaluate the encrypted FFT15 front end on eight packed ciphertexts.
///
/// The caller performs encoding and encryption. Each ciphertext contains
/// two 4096-slot tiles, with the final high tile unused.
///
/// The resulting 15 ciphertexts are at level 3 and can be consumed by
/// `PreparedEncryptedFft3840`.
#[allow(clippy::too_many_arguments)]
pub fn execute_encrypted_fft15(
    physical: &[RnsCkksCiphertext; 8],
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    plan0: &RnsNttPlan,
    prepared0: &PreparedRnsNttPlan,
    plan1: &RnsNttPlan,
    prepared1: &PreparedRnsNttPlan,
    plan2: &RnsNttPlan,
    prepared2: &PreparedRnsNttPlan,
    prepared_left: &PreparedBoundedRnsGaloisKey,
) -> [RnsCkksCiphertext; 15] {
    const HALF: usize = 4096;
    const TILES: usize = 15;

    let slots = embedding.slot_count();
    assert!(slots >= 2 * HALF);
    assert!(physical.iter().all(|ct| ct.level() == 0));

    let low_mask: Vec<Complex64> = (0..slots)
        .map(|s| Complex64::new(if s < HALF { 1.0 } else { 0.0 }, 0.0))
        .collect();

    let prepared_mask =
        PreparedComplexSlotsCp::new(&physical[0], &low_mask, embedding, chain, plan0);

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(chain, &evaluation_keys);

    let aligned: [RnsCkksCiphertext; TILES] = std::array::from_fn(|tile| {
        let source = &physical[tile / 2];

        if tile % 2 == 0 {
            multiply_complex_slots_cp_prepared(source, &prepared_mask, chain, prepared0)
        } else {
            let shifted = rotate_left_bounded_rns_ckks_with_prepared_execution(
                source,
                HALF,
                prepared_left,
                chain,
                plan0,
                prepared0,
            );

            multiply_complex_slots_cp_prepared(&shifted, &prepared_mask, chain, prepared0)
        }
    });

    assert!(aligned.iter().all(|ct| ct.level() == 1));

    let radix5 = PreparedRadix5Cp::new(&aligned[0], embedding, chain, plan1);

    let branches: [[RnsCkksCiphertext; 5]; 3] = std::array::from_fn(|j| {
        packed_radix5_dft_cp_prepared(
            &evaluator,
            [
                &aligned[j],
                &aligned[j + 3],
                &aligned[j + 6],
                &aligned[j + 9],
                &aligned[j + 12],
            ],
            &radix5,
            chain,
            prepared1,
        )
    });

    let radix3 = PreparedFft15TwiddledRadix3Cp::new(&branches[0][0], embedding, chain, plan2);

    let groups: [[RnsCkksCiphertext; 3]; 5] = std::array::from_fn(|k| {
        packed_fft15_twiddled_radix3_cp_prepared(
            &evaluator,
            [&branches[0][k], &branches[1][k], &branches[2][k]],
            k,
            &radix3,
            chain,
            prepared2,
        )
    });

    let sources = std::array::from_fn(|h| groups[h % 5][h / 5].clone());

    assert!(sources.iter().all(|ct: &RnsCkksCiphertext| ct.level() == 3));

    sources
}

// R29_2B3_REUSABLE_PREPARED_FFT15

/// Public FFT15 operands prepared from the first compatible encrypted input.
///
/// The coefficients contain no secret key or encrypted image payload.
/// The extraction key and execution plans remain caller-owned.
pub struct PreparedEncryptedFft15 {
    mask: PreparedComplexSlotsCp,
    radix5: PreparedRadix5Cp,
    radix3: PreparedFft15TwiddledRadix3Cp,
    input_scale: f64,
    input_basis: ModulusBasis,
}

impl PreparedEncryptedFft15 {
    /// Prepare all three FFT15 operand sets during the first evaluation.
    ///
    /// Returns the prepared context and the first image's 15 branches.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_and_execute_first(
        physical: &[RnsCkksCiphertext; 8],
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        plan0: &RnsNttPlan,
        prepared0: &PreparedRnsNttPlan,
        plan1: &RnsNttPlan,
        prepared1: &PreparedRnsNttPlan,
        plan2: &RnsNttPlan,
        prepared2: &PreparedRnsNttPlan,
        prepared_left: &PreparedBoundedRnsGaloisKey,
    ) -> (Self, [RnsCkksCiphertext; 15]) {
        const HALF: usize = 4096;

        assert_eq!(embedding.slot_count(), 32768);
        for ct in physical {
            ct.assert_matches_chain(chain);
            assert_eq!(ct.level(), 0);
            assert_eq!(ct.scale(), physical[0].scale());
            assert_eq!(ct.basis(), physical[0].basis());
        }

        let low_mask: Vec<Complex64> = (0..embedding.slot_count())
            .map(|s| Complex64::new(if s < HALF { 1.0 } else { 0.0 }, 0.0))
            .collect();

        let mask = PreparedComplexSlotsCp::new(&physical[0], &low_mask, embedding, chain, plan0);

        let aligned = Self::extract(physical, &mask, chain, plan0, prepared0, prepared_left);

        let radix5 = PreparedRadix5Cp::new(&aligned[0], embedding, chain, plan1);

        let branches = Self::radix5_stage(&aligned, &radix5, chain, prepared1);

        let radix3 = PreparedFft15TwiddledRadix3Cp::new(&branches[0][0], embedding, chain, plan2);

        let sources = Self::radix3_stage(&branches, &radix3, chain, prepared2);

        (
            Self {
                mask,
                radix5,
                radix3,
                input_scale: physical[0].scale(),
                input_basis: physical[0].basis().clone(),
            },
            sources,
        )
    }

    /// Evaluate another compatible image without preparing public operands.
    // R29_2B3B_PREPARED_BASIS_GUARD
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        physical: &[RnsCkksCiphertext; 8],
        chain: &ModulusChain,
        plan0: &RnsNttPlan,
        prepared0: &PreparedRnsNttPlan,
        prepared1: &PreparedRnsNttPlan,
        prepared2: &PreparedRnsNttPlan,
        prepared_left: &PreparedBoundedRnsGaloisKey,
    ) -> [RnsCkksCiphertext; 15] {
        for ct in physical {
            ct.assert_matches_chain(chain);
            assert_eq!(ct.level(), 0, "FFT15 requires level-0 input");
            assert_eq!(ct.scale(), self.input_scale, "FFT15 scale mismatch");
            assert_eq!(
                ct.basis(),
                &self.input_basis,
                "FFT15 prepared basis mismatch"
            );
        }

        let aligned = Self::extract(physical, &self.mask, chain, plan0, prepared0, prepared_left);
        let branches = Self::radix5_stage(&aligned, &self.radix5, chain, prepared1);

        Self::radix3_stage(&branches, &self.radix3, chain, prepared2)
    }

    fn extract(
        physical: &[RnsCkksCiphertext; 8],
        mask: &PreparedComplexSlotsCp,
        chain: &ModulusChain,
        plan0: &RnsNttPlan,
        prepared0: &PreparedRnsNttPlan,
        prepared_left: &PreparedBoundedRnsGaloisKey,
    ) -> [RnsCkksCiphertext; 15] {
        const HALF: usize = 4096;

        let aligned: [RnsCkksCiphertext; 15] = fft3840_pool().install(|| {
            (0..15)
                .into_par_iter()
                .map(|tile| {
                    let source = &physical[tile / 2];

                    if tile % 2 == 0 {
                        multiply_complex_slots_cp_prepared(source, mask, chain, prepared0)
                    } else {
                        let shifted = rotate_left_bounded_rns_ckks_with_prepared_execution(
                            source,
                            HALF,
                            prepared_left,
                            chain,
                            plan0,
                            prepared0,
                        );

                        multiply_complex_slots_cp_prepared(&shifted, mask, chain, prepared0)
                    }
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap_or_else(|_| panic!("FFT15 extraction length mismatch"))
        });

        assert!(aligned.iter().all(|ct: &RnsCkksCiphertext| ct.level() == 1));
        aligned
    }

    fn radix5_stage(
        aligned: &[RnsCkksCiphertext; 15],
        radix5: &PreparedRadix5Cp,
        chain: &ModulusChain,
        prepared1: &PreparedRnsNttPlan,
    ) -> [[RnsCkksCiphertext; 5]; 3] {
        let keys = RnsCkksEvaluationKeys::new();
        let evaluator = RnsCkksEvaluator::new(chain, &keys);

        let branches: [[RnsCkksCiphertext; 5]; 3] = fft3840_pool().install(|| {
            (0..3)
                .into_par_iter()
                .map(|j| {
                    packed_radix5_dft_cp_prepared(
                        &evaluator,
                        [
                            &aligned[j],
                            &aligned[j + 3],
                            &aligned[j + 6],
                            &aligned[j + 9],
                            &aligned[j + 12],
                        ],
                        radix5,
                        chain,
                        prepared1,
                    )
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap_or_else(|_| panic!("FFT15 radix5 group count mismatch"))
        });

        assert!(branches
            .iter()
            .flatten()
            .all(|ct: &RnsCkksCiphertext| ct.level() == 2));
        branches
    }

    fn radix3_stage(
        branches: &[[RnsCkksCiphertext; 5]; 3],
        radix3: &PreparedFft15TwiddledRadix3Cp,
        chain: &ModulusChain,
        prepared2: &PreparedRnsNttPlan,
    ) -> [RnsCkksCiphertext; 15] {
        let keys = RnsCkksEvaluationKeys::new();
        let evaluator = RnsCkksEvaluator::new(chain, &keys);

        let groups: [[RnsCkksCiphertext; 3]; 5] = fft3840_pool().install(|| {
            (0..5)
                .into_par_iter()
                .map(|k| {
                    packed_fft15_twiddled_radix3_cp_prepared(
                        &evaluator,
                        [&branches[0][k], &branches[1][k], &branches[2][k]],
                        k,
                        radix3,
                        chain,
                        prepared2,
                    )
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap_or_else(|_| panic!("FFT15 radix3 group count mismatch"))
        });

        let sources: [RnsCkksCiphertext; 15] =
            std::array::from_fn(|h| groups[h % 5][h / 5].clone());

        assert!(sources.iter().all(|ct| ct.level() == 3));
        sources
    }
}

// R29_3A_UNIFIED_ENCRYPTED_FFT3840_PIPELINE

/// Borrowed resources required to evaluate the FFT15 front end.
///
/// Cryptographic plans and evaluation keys remain caller-owned.
pub struct EncryptedFft15ExecutionResources<'a> {
    pub chain: &'a ModulusChain,
    pub plan0: &'a RnsNttPlan,
    pub prepared0: &'a PreparedRnsNttPlan,
    pub prepared1: &'a PreparedRnsNttPlan,
    pub prepared2: &'a PreparedRnsNttPlan,
    pub prepared_left: &'a PreparedBoundedRnsGaloisKey,
}

/// A reusable encrypted FFT3840 pipeline.
///
/// The first execution prepares FFT15 public operands. Subsequent
/// executions reuse them together with the borrowed FFT256 service.
///
/// This pipeline neither encrypts nor decrypts ciphertexts.
pub struct EncryptedFft3840Pipeline<'a, 'ctx> {
    fft15: PreparedEncryptedFft15,
    fft256: PreparedEncryptedFft3840<'a, 'ctx>,
    resources: EncryptedFft15ExecutionResources<'a>,
}

impl<'a, 'ctx> EncryptedFft3840Pipeline<'a, 'ctx> {
    // R29_3B1_FROM_PREPARED_COMPONENTS
    /// Compose already-prepared FFT15 and FFT256 services.
    ///
    /// No ciphertext evaluation or preparation is performed here.
    pub fn from_prepared(
        fft15: PreparedEncryptedFft15,
        fft256: PreparedEncryptedFft3840<'a, 'ctx>,
        resources: EncryptedFft15ExecutionResources<'a>,
    ) -> Self {
        Self {
            fft15,
            fft256,
            resources,
        }
    }
}

impl<'a, 'ctx> EncryptedFft3840Pipeline<'a, 'ctx> {
    /// Prepare the FFT15 front end and execute the first encrypted input.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_and_execute_first(
        physical: &[RnsCkksCiphertext; 8],
        embedding: &CkksCanonicalEmbedding,
        plan1: &RnsNttPlan,
        plan2: &RnsNttPlan,
        resources: EncryptedFft15ExecutionResources<'a>,
        fft256: PreparedEncryptedFft3840<'a, 'ctx>,
    ) -> (Self, [RnsCkksCiphertext; 2]) {
        let (fft15, branches) = PreparedEncryptedFft15::prepare_and_execute_first(
            physical,
            embedding,
            resources.chain,
            resources.plan0,
            resources.prepared0,
            plan1,
            resources.prepared1,
            plan2,
            resources.prepared2,
            resources.prepared_left,
        );

        let spectrum = fft256.execute(&branches);

        (
            Self {
                fft15,
                fft256,
                resources,
            },
            spectrum,
        )
    }

    /// Execute another compatible encrypted input using prepared operands.
    pub fn execute(&self, physical: &[RnsCkksCiphertext; 8]) -> [RnsCkksCiphertext; 2] {
        let branches = self.fft15.execute(
            physical,
            self.resources.chain,
            self.resources.plan0,
            self.resources.prepared0,
            self.resources.prepared1,
            self.resources.prepared2,
            self.resources.prepared_left,
        );

        self.fft256.execute(&branches)
    }
}

// R29_4A_MEMORY_BOUNDED_STREAMING

/// Progress information for one completed encrypted FFT3840 batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncryptedFft3840BatchProgress {
    /// Zero-based index of the completed batch.
    pub batch_index: usize,

    /// Number of batches completed, including this batch.
    pub completed_batches: usize,

    /// Time spent executing the encrypted FFT3840 pipeline.
    pub execution_duration: std::time::Duration,
}

/// Failure reported by an encrypted FFT3840 streaming source or sink.
#[derive(Debug)]
pub enum EncryptedFft3840StreamError<E> {
    Source(E),
    Sink(E),
}

impl<'a, 'ctx> EncryptedFft3840Pipeline<'a, 'ctx> {
    /// Execute successive encrypted FFT3840 batches.
    ///
    /// Each source invocation returns one batch of eight level-0
    /// ciphertexts, or None when the stream is exhausted.
    ///
    /// Each completed transform produces two level-7 ciphertexts,
    /// which are transferred immediately to the output sink.
    ///
    /// The controller does not retain ciphertexts from previous
    /// batches, and reuses the pipeline's prepared operands.
    ///
    /// The caller is responsible for compatible ciphertext
    /// parameters and for enforcing any expected image geometry.
    pub fn execute_stream<E, Source, Sink>(
        &self,
        mut source: Source,
        mut sink: Sink,
    ) -> Result<usize, EncryptedFft3840StreamError<E>>
    where
        Source: FnMut() -> Result<Option<[RnsCkksCiphertext; 8]>, E>,
        Sink: FnMut([RnsCkksCiphertext; 2], EncryptedFft3840BatchProgress) -> Result<(), E>,
    {
        let mut completed_batches = 0usize;

        loop {
            let Some(physical) = source().map_err(EncryptedFft3840StreamError::Source)? else {
                break;
            };

            let started = std::time::Instant::now();

            let spectrum = self.execute(&physical);

            let execution_duration = started.elapsed();

            let next_completed = completed_batches
                .checked_add(1)
                .expect("FFT3840 batch count overflow");

            let progress = EncryptedFft3840BatchProgress {
                batch_index: completed_batches,
                completed_batches: next_completed,
                execution_duration,
            };

            sink(spectrum, progress).map_err(EncryptedFft3840StreamError::Sink)?;

            completed_batches = next_completed;
        }

        Ok(completed_batches)
    }
}

// R29_10B_TESTABLE_BOUNDED_SCHEDULER
fn execute_stream_bounded_with<I, O, E, Source, Sink, Execute>(
    mut source: Source,
    mut sink: Sink,
    execute: Execute,
) -> Result<usize, EncryptedFft3840StreamError<E>>
where
    I: Send + Sync,
    O: Send,
    Source: FnMut() -> Result<Option<I>, E>,
    Sink: FnMut(O, EncryptedFft3840BatchProgress) -> Result<(), E>,
    Execute: Fn(&I) -> O + Sync,
{
    let mut completed = 0usize;

    loop {
        let Some(first) = source().map_err(EncryptedFft3840StreamError::Source)? else {
            break;
        };

        // Both inputs are acquired before either evaluation begins.
        // A failure here leaves the first input consumed but undelivered.
        let second = source().map_err(EncryptedFft3840StreamError::Source)?;

        let run = |input: &I| {
            let started = std::time::Instant::now();
            let output = execute(input);
            (output, started.elapsed())
        };

        let (first_result, second_result) = if let Some(second) = second {
            let (a, b) = fft3840_pool().install(|| rayon::join(|| run(&first), || run(&second)));
            (a, Some(b))
        } else {
            (run(&first), None)
        };

        let next = completed
            .checked_add(1)
            .expect("FFT3840 batch count overflow");

        sink(
            first_result.0,
            EncryptedFft3840BatchProgress {
                batch_index: completed,
                completed_batches: next,
                execution_duration: first_result.1,
            },
        )
        .map_err(EncryptedFft3840StreamError::Sink)?;

        completed = next;

        if let Some((output, duration)) = second_result {
            let next = completed
                .checked_add(1)
                .expect("FFT3840 batch count overflow");

            sink(
                output,
                EncryptedFft3840BatchProgress {
                    batch_index: completed,
                    completed_batches: next,
                    execution_duration: duration,
                },
            )
            .map_err(EncryptedFft3840StreamError::Sink)?;

            completed = next;
        } else {
            break;
        }
    }

    Ok(completed)
}

impl<'a, 'ctx> EncryptedFft3840Pipeline<'a, 'ctx> {
    /// Execute at most two encrypted FFT3840 batches concurrently.
    ///
    /// Inputs are acquired and outputs delivered sequentially.
    /// Output order matches input order. Prepared operands are reused.
    ///
    /// Source failures during a two-batch fetch can leave an earlier
    /// input consumed without producing an output.
    pub fn execute_stream_bounded<E, Source, Sink>(
        &self,
        source: Source,
        sink: Sink,
    ) -> Result<usize, EncryptedFft3840StreamError<E>>
    where
        Source: FnMut() -> Result<Option<[RnsCkksCiphertext; 8]>, E>,
        Sink: FnMut([RnsCkksCiphertext; 2], EncryptedFft3840BatchProgress) -> Result<(), E>,
    {
        execute_stream_bounded_with(source, sink, |physical| self.execute(physical))
    }
}

// R29_3B2A_UNIFIED_PIPELINE_MIGRATION
impl<'a, 'ctx> EncryptedFft3840Pipeline<'a, 'ctx> {
    /// Execute already-computed FFT15 branches through FFT256.
    ///
    /// No FFT15 computation or preparation is repeated.
    pub fn execute_prepared_branches(
        &self,
        branches: &[RnsCkksCiphertext; 15],
    ) -> [RnsCkksCiphertext; 2] {
        self.fft256.execute(branches)
    }
}

// R29_9B_PIPELINE_SYNC_PROBE
#[allow(dead_code)]
fn r29_9b_assert_pipeline_sync<'a, 'ctx>()
where
    'ctx: 'a,
{
    fn assert_sync<T: Sync>() {}
    assert_sync::<EncryptedFft3840Pipeline<'a, 'ctx>>();
}

// R29_10C_BOUNDED_SCHEDULER_TESTS

// R31_2B_BATCHED_HORIZONTAL_EXECUTOR

/// Execute multiple encrypted FFT3840 horizontal batches.
///
/// Each batch contains 15 FFT15 ciphertexts at level 3.
/// Each output contains two packed FFT3840 ciphertexts at level 7.
///
/// Stage zero uses operand-major streaming key generation.
/// Later stages reuse the caller's level-specific rotation keys.
///
/// The caller controls batch-window size and resource lifetimes.
#[allow(clippy::too_many_arguments)]
pub fn execute_encrypted_fft3840_batched<F>(
    batches: &[[RnsCkksCiphertext; 15]],
    prepared_sources: &[PreparedFusedFft256Source; 15],
    later_stages: &[PreparedFft256Stage; 3],
    evaluator: &RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
    prepared_stage0: &PreparedRnsNttPlan,
    qp_stage0: &RnsNttPlan,
    prepared_qp_stage0: &PreparedRnsNttPlan,
    later_contexts: [&StageContext<'_>; 3],
    make_stage0_key: &mut F,
) -> Vec<[RnsCkksCiphertext; 2]>
where
    F: FnMut(isize) -> crate::ckks::PreparedHybridRnsGaloisKey,
{
    assert!(!batches.is_empty(), "FFT3840 batch window is empty");
    assert!(batches
        .iter()
        .all(|batch| { batch.iter().all(|ct| ct.level() == 3) }));

    let mut outputs =
        crate::eblas::fused_fft256_handoff::apply_fft3840_packed_fft256_stage0_batched_streaming(
            batches,
            prepared_sources,
            evaluator,
            chain,
            prepared_stage0,
            qp_stage0,
            prepared_qp_stage0,
            make_stage0_key,
        );

    assert_eq!(outputs.len(), batches.len());

    for (stage_index, context) in later_contexts.iter().enumerate() {
        let expected_input_level = stage_index + 4;

        for pair in &mut outputs {
            for ciphertext in pair {
                assert_eq!(ciphertext.level(), expected_input_level);

                *ciphertext = crate::eblas::prepared_radix4_fft256::apply_stage(
                    ciphertext,
                    &later_stages[stage_index],
                    context,
                );

                assert_eq!(ciphertext.level(), expected_input_level + 1);
            }
        }
    }

    assert!(outputs
        .iter()
        .all(|pair| { pair.iter().all(|ct| ct.level() == 7) }));

    outputs
}

#[cfg(test)]
mod r29_10c_bounded_scheduler_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn zero_through_four_batches_preserve_order() {
        for count in 0usize..=4 {
            let mut inputs = 0usize..count;
            let mut outputs = Vec::new();

            let completed = execute_stream_bounded_with(
                || Ok::<_, ()>(inputs.next()),
                |output, progress| {
                    outputs.push((output, progress.batch_index, progress.completed_batches));
                    Ok::<(), ()>(())
                },
                |input: &usize| input * 7,
            )
            .expect("bounded execution failed");

            assert_eq!(completed, count);
            assert_eq!(outputs.len(), count);

            for (index, (value, batch_index, completed_batches)) in outputs.into_iter().enumerate()
            {
                assert_eq!(value, index * 7);
                assert_eq!(batch_index, index);
                assert_eq!(completed_batches, index + 1);
            }
        }
    }

    #[test]
    fn second_source_failure_does_not_execute_first() {
        let mut calls = 0usize;
        let executions = AtomicUsize::new(0);

        let result = execute_stream_bounded_with(
            || {
                calls += 1;
                match calls {
                    1 => Ok::<_, &'static str>(Some(10usize)),
                    2 => Err("source failure"),
                    _ => panic!("unexpected source call"),
                }
            },
            |_: usize, _| Ok::<(), &'static str>(()),
            |input: &usize| {
                executions.fetch_add(1, Ordering::SeqCst);
                *input
            },
        );

        assert!(matches!(
            result,
            Err(EncryptedFft3840StreamError::Source("source failure"))
        ));
        assert_eq!(executions.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn first_source_failure_is_propagated() {
        let result = execute_stream_bounded_with(
            || Err::<Option<usize>, _>("first source failure"),
            |_: usize, _| Ok::<(), &'static str>(()),
            |input: &usize| *input,
        );

        assert!(matches!(
            result,
            Err(EncryptedFft3840StreamError::Source("first source failure"))
        ));
    }

    #[test]
    fn sink_failure_stops_delivery() {
        let mut inputs = 0usize..4;
        let mut deliveries = 0usize;

        let result = execute_stream_bounded_with(
            || Ok::<_, &'static str>(inputs.next()),
            |_: usize, _| {
                deliveries += 1;
                Err::<(), _>("sink failure")
            },
            |input: &usize| *input,
        );

        assert!(matches!(
            result,
            Err(EncryptedFft3840StreamError::Sink("sink failure"))
        ));
        assert_eq!(deliveries, 1);
    }

    #[test]
    fn two_batches_can_overlap() {
        if fft3840_pool().current_num_threads() < 2 {
            return;
        }

        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let mut inputs = 0usize..2;

        let completed = execute_stream_bounded_with(
            || Ok::<_, ()>(inputs.next()),
            |_: usize, _| Ok::<(), ()>(()),
            |input: &usize| {
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(current, Ordering::SeqCst);

                std::thread::sleep(Duration::from_millis(100));

                active.fetch_sub(1, Ordering::SeqCst);
                *input
            },
        )
        .expect("parallel execution failed");

        assert_eq!(completed, 2);
        assert_eq!(peak.load(Ordering::SeqCst), 2);
    }
}
