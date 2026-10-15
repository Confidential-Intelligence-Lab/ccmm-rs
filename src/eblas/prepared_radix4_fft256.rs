//! Prepared radix-4 FFT256 execution for encrypted eBLAS.
//!
//! Public FFT diagonals are encoded and NTT-prepared once per CKKS
//! level, then reused across ciphertext inputs.
//!
//! The arithmetic sequence follows the validated R16.6b.3 regression.

use rayon::prelude::*;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Instant;

use num_complex::Complex64;

use crate::application_support::ckks::encode_rns;
use crate::ckks::{
    apply_hybrid_rns_galois_automorphism_prepared_qp,
    apply_hybrid_rns_galois_automorphism_with_prepared_ntt,
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next, CkksCanonicalEmbedding,
    PreparedHybridRnsGaloisKey, PreparedRnsPlaintextNtt, RnsCkksCiphertext, RnsCkksEvaluator,
};
use crate::eblas::mixed_radix_ckks::{
    packed_fft15_all_branches_cp_prepared, PreparedFft15TwiddledRadix3Cp,
};
use crate::ring::{ModulusChain, PreparedRnsNttPlan, RnsNttPlan};

/// Public signed-offset FFT stage diagonals.
pub type Diagonals = BTreeMap<isize, Vec<Complex64>>;

/// Prepared hybrid-QP rotation keys indexed by signed offset.
pub type RotationKeys = BTreeMap<isize, PreparedHybridRnsGaloisKey>;

/// Assemble independent FFT256 diagonal sets into contiguous SIMD lanes.
///
/// All lanes must use the same rotation offsets. The caller is responsible
/// for constructing each lane's coefficients using the appropriate branch
/// and FFT stage. Lane sizes must be multiples of 256 CKKS slots.
pub fn pack_fft256_branch_diagonals(
    branch_diagonals: &[&Diagonals],
    lane_slots: usize,
) -> Diagonals {
    assert!(
        !branch_diagonals.is_empty(),
        "at least one FFT256 lane required"
    );
    assert!(
        lane_slots > 0 && lane_slots % 256 == 0,
        "FFT256 lane size must be a nonzero multiple of 256"
    );

    let total_slots = lane_slots
        .checked_mul(branch_diagonals.len())
        .expect("packed FFT256 slot count overflow");

    let expected_offsets: Vec<isize> = branch_diagonals[0].keys().copied().collect();

    assert!(!expected_offsets.is_empty(), "empty FFT256 diagonal set");

    for (lane, diagonals) in branch_diagonals.iter().enumerate() {
        let offsets: Vec<isize> = diagonals.keys().copied().collect();
        assert_eq!(
            offsets, expected_offsets,
            "FFT256 rotation offsets differ for lane {lane}"
        );

        for values in diagonals.values() {
            assert_eq!(
                values.len(),
                lane_slots,
                "FFT256 diagonal length mismatch for lane {lane}"
            );
        }
    }

    let mut packed = Diagonals::new();

    for offset in expected_offsets {
        let mut values = Vec::with_capacity(total_slots);

        for diagonals in branch_diagonals {
            values.extend_from_slice(diagonals.get(&offset).expect("validated FFT256 offset"));
        }

        assert_eq!(values.len(), total_slots);
        packed.insert(offset, values);
    }

    packed
}

pub struct StageContext<'a> {
    pub keys: &'a RotationKeys,
    pub evaluator: &'a RnsCkksEvaluator<'a>,
    pub chain: &'a ModulusChain,
    pub qp_plan: &'a RnsNttPlan,
    pub prepared: &'a PreparedRnsNttPlan,
}

pub struct PreparedFft256Stage {
    level: usize,
    plain_scale: f64,
    operands: Vec<(isize, PreparedRnsPlaintextNtt)>,
}

impl PreparedFft256Stage {
    pub fn new(
        level: usize,
        diagonals: &Diagonals,
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        prepared: &PreparedRnsNttPlan,
    ) -> Self {
        let basis = chain.level(level);
        let plain_scale = chain
            .dropped_modulus(level)
            .expect("rescale level available")
            .value() as f64;

        assert_eq!(basis.moduli(), prepared.moduli());

        let operands = diagonals
            .iter()
            .map(|(&offset, diagonal)| {
                let plain = encode_rns(diagonal, embedding, basis, plain_scale);

                (offset, PreparedRnsPlaintextNtt::new(&plain, prepared))
            })
            .collect();

        Self {
            level,
            plain_scale,
            operands,
        }
    }
}

/// Four prepared FFT256 stages for one selected horizontal branch.
///
/// This is a bounded-cost validation path before constructing the
/// complete 15-branch operand bank.
pub struct PreparedFft256SelectedBranch {
    branch: usize,
    stages: [PreparedFft256Stage; 4],
}

impl PreparedFft256SelectedBranch {
    pub fn new<F>(
        branch: usize,
        mut diagonals: F,
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        prepared: [&PreparedRnsNttPlan; 4],
    ) -> Self
    where
        F: FnMut(usize, usize) -> Diagonals,
    {
        assert!(branch < 15, "FFT256 horizontal branch out of range");

        let spans = [4, 16, 64, 256];
        let stages = std::array::from_fn(|stage| {
            let values = diagonals(spans[stage], branch);

            PreparedFft256Stage::new(stage + 3, &values, embedding, chain, prepared[stage])
        });

        Self { branch, stages }
    }

    pub fn branch(&self) -> usize {
        self.branch
    }

    pub fn stage(&self, stage: usize) -> &PreparedFft256Stage {
        &self.stages[stage]
    }
}

/// Prepared FFT256 operands for all 15 horizontal FFT15 branches.
///
/// Only stage 0 depends on the horizontal branch. Stages 1..4
/// reuse their prepared public operands across all branches.
///
/// Rotation keys are supplied separately through StageContext.
pub struct PreparedFft256BranchBank {
    first: Vec<PreparedFft256Stage>,
    shared: [PreparedFft256Stage; 3],
}

impl PreparedFft256BranchBank {
    /// Prepare the complete horizontal FFT256 operand bank.
    ///
    /// Stage 0 has 15 branch-specific coefficient sets.
    /// Stages 1..3 are shared across all horizontal branches.
    ///
    /// Rotation-key offset equivalence and shared-stage coefficient
    /// equivalence are covered by the R23.3D.4 regression.
    pub fn new<F>(
        mut diagonals: F,
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        prepared: [&PreparedRnsNttPlan; 4],
    ) -> Self
    where
        F: FnMut(usize, usize) -> Diagonals,
    {
        const BRANCHES: usize = 15;
        const SPANS: [usize; 4] = [4, 16, 64, 256];

        let first = (0..BRANCHES)
            .map(|branch| {
                let values = diagonals(SPANS[0], branch);

                assert!(
                    !values.is_empty(),
                    "empty FFT256 first-stage diagonals for branch {branch}"
                );

                PreparedFft256Stage::new(3, &values, embedding, chain, prepared[0])
            })
            .collect();

        let shared = std::array::from_fn(|index| {
            let stage = index + 1;
            let values = diagonals(SPANS[stage], 0);

            assert!(
                !values.is_empty(),
                "empty FFT256 shared-stage diagonals at stage {stage}"
            );

            PreparedFft256Stage::new(stage + 3, &values, embedding, chain, prepared[stage])
        });

        Self { first, shared }
    }

    /// Parallel construction of the 15 branch-specific FFT256 operand sets.
    ///
    /// The callback must be safe to invoke concurrently. Prepared operands
    /// are collected in deterministic branch order.
    pub fn new_parallel<F>(
        diagonals: F,
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        prepared: [&PreparedRnsNttPlan; 4],
    ) -> Self
    where
        F: Fn(usize, usize) -> Diagonals + Send + Sync,
    {
        const BRANCHES: usize = 15;
        const SPANS: [usize; 4] = [4, 16, 64, 256];

        let workers = fft256_branch_worker_count();
        let pool = fft256_pool(workers);

        let first: Vec<PreparedFft256Stage> = pool.install(|| {
            (0..BRANCHES)
                .into_par_iter()
                .map(|branch| {
                    let values = diagonals(SPANS[0], branch);
                    assert!(
                        !values.is_empty(),
                        "empty FFT256 first-stage diagonals for branch {branch}"
                    );
                    PreparedFft256Stage::new(3, &values, embedding, chain, prepared[0])
                })
                .collect()
        });

        let shared = std::array::from_fn(|index| {
            let stage = index + 1;
            let values = diagonals(SPANS[stage], 0);
            assert!(
                !values.is_empty(),
                "empty FFT256 shared-stage diagonals at stage {stage}"
            );
            PreparedFft256Stage::new(stage + 3, &values, embedding, chain, prepared[stage])
        });

        assert_eq!(first.len(), BRANCHES);

        Self { first, shared }
    }

    pub fn stage(&self, branch: usize, stage: usize) -> &PreparedFft256Stage {
        assert!(branch < 15, "FFT256 horizontal branch out of range");
        assert!(stage < 4, "FFT256 stage out of range");

        if stage == 0 {
            &self.first[branch]
        } else {
            &self.shared[stage - 1]
        }
    }

    pub fn prepared_stage_count(&self) -> usize {
        self.first.len() + self.shared.len()
    }
}

/// Apply the four prepared FFT256 stages to every horizontal FFT15 branch.
///
/// Inputs must be at CKKS level 3. Outputs are at level 7.
/// Prepared operands and rotation keys are supplied by the caller.
///
/// Execution is sequential to bound transient ciphertext allocations.
/// Complete the horizontal FFT3840 after its three radix-5 groups.
///
/// The input groups contain the 15 radix-5 ciphertext outputs at CKKS
/// level 2, organized as three groups of five.
///
/// FFT15 combines them into natural frequency branches h = k + 5*q.
/// The prepared FFT256 bank then transforms each branch from level 3
/// to level 7, preserving natural horizontal branch ordering.
///
/// No keys or plaintext operands are generated by this function.
pub fn apply_fft3840_horizontal_from_radix5_prepared_with_context(
    evaluator: &RnsCkksEvaluator<'_>,
    radix5_groups: &[[RnsCkksCiphertext; 5]; 3],
    fft15_coefficients: &PreparedFft15TwiddledRadix3Cp,
    chain: &ModulusChain,
    fft15_plan: &PreparedRnsNttPlan,
    fft256_bank: &PreparedFft256BranchBank,
    fft256_execution: ([&StageContext<'_>; 4], &PreparedFft256ExecutionContext),
) -> [RnsCkksCiphertext; 15] {
    let (fft256_contexts, prepared_context) = fft256_execution;

    let branches = packed_fft15_all_branches_cp_prepared(
        evaluator,
        radix5_groups,
        fft15_coefficients,
        chain,
        fft15_plan,
    );

    assert!(
        branches.iter().all(|ciphertext| ciphertext.level() == 3),
        "horizontal FFT15 must produce level-3 ciphertexts"
    );

    // R24.3: production execution uses the bounded Rayon branch engine.
    // Preserve the existing array-returning API and natural branch order.
    let mut outputs: [Option<RnsCkksCiphertext>; 15] = std::array::from_fn(|_| None);

    apply_fft256_all_branches_streaming_with_context(
        branches,
        fft256_bank,
        fft256_contexts,
        prepared_context,
        |branch, ciphertext| {
            assert!(outputs[branch].is_none(), "duplicate FFT256 branch");
            outputs[branch] = Some(ciphertext);
        },
    );

    let outputs = outputs.map(|output| output.expect("missing FFT256 production branch"));

    assert!(
        outputs.iter().all(|ciphertext| ciphertext.level() == 7),
        "horizontal FFT3840 must produce level-7 ciphertexts"
    );

    outputs
}

pub fn apply_fft3840_horizontal_from_radix5_prepared(
    evaluator: &RnsCkksEvaluator<'_>,
    radix5_groups: &[[RnsCkksCiphertext; 5]; 3],
    fft15_coefficients: &PreparedFft15TwiddledRadix3Cp,
    chain: &ModulusChain,
    fft15_plan: &PreparedRnsNttPlan,
    fft256_bank: &PreparedFft256BranchBank,
    fft256_contexts: [&StageContext<'_>; 4],
) -> [RnsCkksCiphertext; 15] {
    let prepared_context = PreparedFft256ExecutionContext::new(fft256_contexts);

    apply_fft3840_horizontal_from_radix5_prepared_with_context(
        evaluator,
        radix5_groups,
        fft15_coefficients,
        chain,
        fft15_plan,
        fft256_bank,
        (fft256_contexts, &prepared_context),
    )
}

/// Borrowed FFT256 resources for horizontal FFT3840 streaming.
pub struct Fft3840StreamingExecution<'a, 'ctx> {
    pub bank: &'a PreparedFft256BranchBank,
    pub contexts: [&'a StageContext<'ctx>; 4],
}

/// Execute the horizontal FFT3840 and transfer each completed
/// frequency branch to the consumer without collecting the outputs.
pub fn apply_fft3840_horizontal_from_radix5_streaming<F>(
    evaluator: &RnsCkksEvaluator<'_>,
    radix5_groups: &[[RnsCkksCiphertext; 5]; 3],
    fft15_coefficients: &PreparedFft15TwiddledRadix3Cp,
    chain: &ModulusChain,
    fft15_plan: &PreparedRnsNttPlan,
    execution: Fft3840StreamingExecution<'_, '_>,
    consume: F,
) where
    F: FnMut(usize, RnsCkksCiphertext),
{
    let branches = packed_fft15_all_branches_cp_prepared(
        evaluator,
        radix5_groups,
        fft15_coefficients,
        chain,
        fft15_plan,
    );

    assert!(
        branches.iter().all(|ciphertext| ciphertext.level() == 3),
        "horizontal FFT15 must produce level-3 ciphertexts"
    );

    apply_fft256_all_branches_streaming(branches, execution.bank, execution.contexts, consume);
}

/// Number of FFT256 branch workers selected for this execution.
///
/// The available CPU parallelism is the default. The environment variables
/// FHE_RS_WORKERS and FHE_RS_MAX_INFLIGHT_BRANCHES can reduce or override
/// the branch concurrency. The maximum is the 15 independent branches.
pub fn fft256_branch_worker_count() -> usize {
    fn positive_env(name: &str) -> Option<usize> {
        std::env::var(name).ok().map(|value| {
            let count = value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{name} must be a positive integer"));
            assert!(count > 0, "{name} must be positive");
            count
        })
    }

    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);

    let requested = positive_env("FHE_RS_WORKERS").unwrap_or(available);
    let inflight = positive_env("FHE_RS_MAX_INFLIGHT_BRANCHES").unwrap_or(15);

    requested.min(inflight).clamp(1, 15)
}

/// Apply all four prepared FFT256 stages to a single horizontal branch.
fn apply_fft256_prepared_branch_shared_qp(
    branch: usize,
    mut current: RnsCkksCiphertext,
    bank: &PreparedFft256BranchBank,
    contexts: [&StageContext<'_>; 4],
    shared_qp: &[PreparedRnsNttPlan; 4],
) -> RnsCkksCiphertext {
    assert_eq!(current.level(), 3);

    for stage in 0..4 {
        current = apply_stage_internal(
            &current,
            bank.stage(branch, stage),
            contexts[stage],
            true,
            Some(&shared_qp[stage]),
        );
        assert_eq!(current.level(), stage + 4);
    }

    current
}

/// Persistent Rayon pools indexed by worker count.
///
/// Pools are initialized lazily and reused across repeated FFT executions.
fn fft256_pool(workers: usize) -> &'static ThreadPool {
    static POOLS: OnceLock<std::sync::Mutex<BTreeMap<usize, &'static ThreadPool>>> =
        OnceLock::new();

    let pools = POOLS.get_or_init(|| std::sync::Mutex::new(BTreeMap::new()));
    let mut guard = pools.lock().expect("FFT256 pool registry poisoned");

    guard.entry(workers).or_insert_with(|| {
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|index| format!("fhe-rs-fft256-{index}"))
            .build()
            .expect("FFT256 Rayon pool initialization");

        Box::leak(Box::new(pool))
    })
}

/// Reusable prepared Q/P resources for the four FFT256 stages.
///
/// Construct once and share across all horizontal UHD tile rows.
/// The resources contain public NTT precomputation, not secret keys.
pub struct PreparedFft256ExecutionContext {
    qp: [PreparedRnsNttPlan; 4],
}

impl PreparedFft256ExecutionContext {
    pub fn new(contexts: [&StageContext<'_>; 4]) -> Self {
        Self {
            qp: std::array::from_fn(|stage| PreparedRnsNttPlan::new(contexts[stage].qp_plan)),
        }
    }
}

/// Execute FFT256 branches using a persistent, bounded Rayon pool.
///
/// Prepared operands and evaluation keys are borrowed, never duplicated.
/// Results are consumed in deterministic branch order.
pub fn apply_fft256_all_branches_streaming_with_context<F>(
    inputs: [RnsCkksCiphertext; 15],
    bank: &PreparedFft256BranchBank,
    contexts: [&StageContext<'_>; 4],
    prepared_context: &PreparedFft256ExecutionContext,
    mut consume: F,
) where
    F: FnMut(usize, RnsCkksCiphertext),
{
    assert_eq!(bank.prepared_stage_count(), 18);

    // R25.1C: one immutable prepared Q/P plan per FFT256 stage.
    // Rayon workers borrow these plans without rebuilding them.

    let workers = fft256_branch_worker_count();
    let benchmark = std::env::var("FHE_RS_BENCH").is_ok_and(|value| value == "1");
    let started = Instant::now();

    println!("R24_FFT256_WORKERS={workers}");
    println!(
        "R24_FFT256_EXECUTION_MODE={}",
        if workers == 1 {
            "SEQUENTIAL"
        } else {
            "RAYON_BOUNDED"
        }
    );

    let pool = if workers > 1 {
        Some(fft256_pool(workers))
    } else {
        None
    };

    // R25_3A_CONTINUOUS_SCHEDULER
    //
    // Submit all 15 independent FFT256 branches to one bounded Rayon
    // pool. Workers take subsequent branches as they become available.
    //
    // The indexed parallel iterator preserves branch ordering.
    // Prepared operands, Q/P plans, and evaluation keys are borrowed.
    //
    // No nested pool is constructed and no cryptographic arithmetic
    // changes are introduced.

    let results: Vec<RnsCkksCiphertext> = if let Some(pool) = pool {
        pool.install(|| {
            inputs
                .into_par_iter()
                .enumerate()
                .map(|(branch, input)| {
                    apply_fft256_prepared_branch_shared_qp(
                        branch,
                        input,
                        bank,
                        contexts,
                        &prepared_context.qp,
                    )
                })
                .collect()
        })
    } else {
        inputs
            .into_iter()
            .enumerate()
            .map(|(branch, input)| {
                apply_fft256_prepared_branch_shared_qp(
                    branch,
                    input,
                    bank,
                    contexts,
                    &prepared_context.qp,
                )
            })
            .collect()
    };

    assert_eq!(results.len(), 15);

    // Indexed collection guarantees natural frequency-branch order.
    // Deliver outputs sequentially to preserve the callback contract.
    for (branch, output) in results.into_iter().enumerate() {
        consume(branch, output);
    }

    if benchmark {
        println!(
            "R24_FFT256_WARM_EXECUTION_MS={:.3}",
            started.elapsed().as_secs_f64() * 1000.0
        );
        println!("R24_FFT256_BRANCHES=15");
        println!("R24_FFT256_BENCH_STATUS=COMPLETE");
    }
}

/// R25_3B_1_GLOBAL_MULTIROW_SCHEDULER
///
/// Execute multiple independently prepared FFT256 rows through one
/// persistent, bounded Rayon pool.
///
/// Each row contains 15 ciphertext branches at CKKS level 3.
/// All rows share the same prepared public operands, evaluation keys,
/// and persistent Q/P NTT plans.
///
/// Output ordering is deterministic: row index, then branch index.
///
/// This API does not perform encryption, FFT15 preparation, or
/// vertical FFT execution.
pub fn apply_fft256_multirow_prepared_with_context(
    rows: Vec<[RnsCkksCiphertext; 15]>,
    bank: &PreparedFft256BranchBank,
    contexts: [&StageContext<'_>; 4],
    prepared_context: &PreparedFft256ExecutionContext,
) -> Vec<[RnsCkksCiphertext; 15]> {
    assert_eq!(bank.prepared_stage_count(), 18);

    if rows.is_empty() {
        return Vec::new();
    }

    let workers = fft256_branch_worker_count();

    println!("R25_3B_GLOBAL_WORKERS={workers}");
    println!("R25_3B_TILE_ROWS={}", rows.len());
    println!("R25_3B_SCHEDULER=GLOBAL_BOUNDED");

    // Flatten the row-major input sequence. Rayon indexed collection
    // preserves this order independently of worker completion order.
    let indexed: Vec<(usize, RnsCkksCiphertext)> = rows
        .into_iter()
        .flat_map(|row| row.into_iter().enumerate())
        .collect();

    let outputs: Vec<RnsCkksCiphertext> = if workers == 1 {
        indexed
            .into_iter()
            .map(|(branch, input)| {
                apply_fft256_prepared_branch_shared_qp(
                    branch,
                    input,
                    bank,
                    contexts,
                    &prepared_context.qp,
                )
            })
            .collect()
    } else {
        fft256_pool(workers).install(|| {
            indexed
                .into_par_iter()
                .map(|(branch, input)| {
                    apply_fft256_prepared_branch_shared_qp(
                        branch,
                        input,
                        bank,
                        contexts,
                        &prepared_context.qp,
                    )
                })
                .collect()
        })
    };

    assert_eq!(outputs.len() % 15, 0);

    let mut outputs = outputs.into_iter();
    let row_count = outputs.len() / 15;

    (0..row_count)
        .map(|_| std::array::from_fn(|_| outputs.next().expect("complete FFT256 output row")))
        .collect()
}

pub fn apply_fft256_all_branches_streaming<F>(
    inputs: [RnsCkksCiphertext; 15],
    bank: &PreparedFft256BranchBank,
    contexts: [&StageContext<'_>; 4],
    consume: F,
) where
    F: FnMut(usize, RnsCkksCiphertext),
{
    let prepared_context = PreparedFft256ExecutionContext::new(contexts);
    apply_fft256_all_branches_streaming_with_context(
        inputs,
        bank,
        contexts,
        &prepared_context,
        consume,
    );
}

pub fn apply_fft256_all_branches_prepared(
    inputs: [RnsCkksCiphertext; 15],
    bank: &PreparedFft256BranchBank,
    contexts: [&StageContext<'_>; 4],
) -> [RnsCkksCiphertext; 15] {
    assert_eq!(bank.prepared_stage_count(), 18);

    let mut input_iter = inputs.into_iter();

    std::array::from_fn(|branch| {
        let mut current = input_iter.next().expect("FFT256 branch input must exist");
        assert_eq!(
            current.level(),
            3,
            "FFT256 branch {branch} must start at level 3"
        );

        for (stage, context) in contexts.iter().enumerate() {
            current = apply_stage(&current, bank.stage(branch, stage), context);

            assert_eq!(
                current.level(),
                stage + 4,
                "FFT256 branch {branch} level mismatch at stage {stage}"
            );
        }

        current
    })
}

// R16_7C3C_PRODUCTION_PREPARED_QP
pub fn apply_stage(
    input: &RnsCkksCiphertext,
    stage: &PreparedFft256Stage,
    ctx: &StageContext<'_>,
) -> RnsCkksCiphertext {
    apply_stage_internal(input, stage, ctx, true, None)
}

/// Canonical rotation reference retained for exact FFT regression.
pub fn apply_stage_canonical(
    input: &RnsCkksCiphertext,
    stage: &PreparedFft256Stage,
    ctx: &StageContext<'_>,
) -> RnsCkksCiphertext {
    apply_stage_internal(input, stage, ctx, false, None)
}

fn apply_stage_internal(
    input: &RnsCkksCiphertext,
    stage: &PreparedFft256Stage,
    ctx: &StageContext<'_>,
    use_prepared_qp: bool,
    shared_qp: Option<&PreparedRnsNttPlan>,
) -> RnsCkksCiphertext {
    input.assert_matches_chain(ctx.chain);
    assert_eq!(input.level(), stage.level);
    assert_eq!(input.basis().moduli(), ctx.prepared.moduli());
    // R16.7a.6b: reconstruct the input b component once per FFT stage.
    // All nonzero diagonals rotate the same originating ciphertext.

    // Construct once per stage; reuse across its nonzero rotations.
    let owned_qp =
        (use_prepared_qp && shared_qp.is_none()).then(|| PreparedRnsNttPlan::new(ctx.qp_plan));
    let prepared_qp = shared_qp.or(owned_qp.as_ref());

    let mut accumulator: Option<RnsCkksCiphertext> = None;
    for (offset, prepared_plain) in &stage.operands {
        let offset = *offset;
        let rotated = if offset != 0 {
            let key = ctx.keys.get(&offset).expect("FFT256 rotation key");

            let rotated_rlwe = if let Some(prepared_qp) = &prepared_qp {
                apply_hybrid_rns_galois_automorphism_prepared_qp(
                    input.rlwe(),
                    key,
                    ctx.qp_plan,
                    prepared_qp,
                )
            } else {
                apply_hybrid_rns_galois_automorphism_with_prepared_ntt(
                    input.rlwe(),
                    key,
                    ctx.qp_plan,
                )
            };

            Some(RnsCkksCiphertext::new(
                rotated_rlwe,
                input.state().clone(),
                ctx.chain,
            ))
        } else {
            None
        };
        let source = rotated.as_ref().unwrap_or(input);
        let term = multiply_plain_rns_ckks_with_prepared_ntt(
            source,
            prepared_plain,
            stage.plain_scale,
            ctx.chain,
            ctx.prepared,
        );
        accumulator = Some(match accumulator {
            Some(previous) => ctx.evaluator.add(&previous, &term),
            None => term,
        });
    }
    rescale_rns_ckks_to_next(&accumulator.expect("nonempty stage"), ctx.chain)
}

#[cfg(test)]
mod r24_parallel_tests {
    use super::*;

    #[test]
    fn r24_worker_pool_initializes() {
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let workers = available.min(15);
        let pool = fft256_pool(workers);

        assert_eq!(pool.current_num_threads(), workers);

        let result: Vec<usize> =
            pool.install(|| (0..15_usize).into_par_iter().map(|x| x * x).collect());

        assert_eq!(result, (0..15).map(|x| x * x).collect::<Vec<_>>());

        println!("R24_WORKERS={workers}");
        println!("R24_RAYON_POOL_STATUS=PASS");
    }
}
