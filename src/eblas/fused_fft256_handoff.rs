//! Source-specific diagonal placement for dense FFT256 handoff.
//!
//! This module constructs the linear operator that combines source-lane
//! extraction, destination-lane placement, and an existing FFT256 stage.
//!
//! No ciphertext operations or key generation occur during construction.

use rayon::prelude::*;
use std::collections::BTreeMap;

use num_complex::Complex64;

use super::prepared_radix4_fft256::Diagonals;

/// Prepared diagonal coefficients for one source ciphertext.
///
/// Offsets follow the convention:
///
/// output[i] += diagonal[offset][i] * input[(i + offset) % slots].
///
/// The coefficients are zero outside the selected destination lane.
pub type FusedSourceDiagonals = Diagonals;

/// Build the contribution of one source branch to a dense FFT256 stage.
///
/// `source_lane` and `destination_lane` identify the source and destination
/// 4096-slot regions, respectively.
///
/// `stage_diagonals` describe the existing lane-local FFT256 stage.
/// The returned diagonals operate over the entire CKKS slot vector.
pub fn fuse_fft256_source_lane(
    stage_diagonals: &Diagonals,
    slots: usize,
    lane_slots: usize,
    source_lane: usize,
    destination_lane: usize,
) -> FusedSourceDiagonals {
    assert!(slots > 0, "slot count must be positive");
    assert!(lane_slots > 0, "lane size must be positive");
    assert_eq!(slots % lane_slots, 0, "lane geometry mismatch");

    let lanes = slots / lane_slots;

    assert!(source_lane < lanes, "source lane out of range");
    assert!(destination_lane < lanes, "destination lane out of range");

    let mut fused = BTreeMap::<isize, Vec<Complex64>>::new();

    for (&offset, coefficients) in stage_diagonals {
        assert_eq!(
            coefficients.len(),
            lane_slots,
            "stage diagonal must describe exactly one lane"
        );

        for (local, &coefficient) in coefficients.iter().enumerate() {
            if coefficient == Complex64::new(0.0, 0.0) {
                continue;
            }

            let local_source = (local as isize + offset).rem_euclid(lane_slots as isize) as usize;

            let destination = destination_lane * lane_slots + local;
            let source = source_lane * lane_slots + local_source;

            let global_offset = (source as isize - destination as isize).rem_euclid(slots as isize);

            fused
                .entry(global_offset)
                .or_insert_with(|| vec![Complex64::new(0.0, 0.0); slots])[destination] +=
                coefficient;
        }
    }

    fused
}

// R26_H11_14B_PRODUCTION_ENCRYPTED_EVALUATOR

use crate::application_support::ckks::encode_rns;
use crate::ckks::{
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next, CkksCanonicalEmbedding,
    PreparedRnsPlaintextNtt, RnsCkksCiphertext,
};
use crate::ring::{ModulusChain, PreparedRnsNttPlan};

use super::prepared_radix4_fft256::StageContext;

/// Prepared operands for one source ciphertext's contribution to
/// the fused placement and first FFT256 stage.
pub struct PreparedFusedFft256Source {
    level: usize,
    plain_scale: f64,
    operands: Vec<(isize, PreparedRnsPlaintextNtt)>,
}

impl PreparedFusedFft256Source {
    pub fn new(
        level: usize,
        diagonals: &FusedSourceDiagonals,
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        prepared: &PreparedRnsNttPlan,
    ) -> Self {
        assert!(!diagonals.is_empty(), "empty fused diagonal set");

        let basis = chain.level(level);
        assert_eq!(basis.moduli(), prepared.moduli());

        let plain_scale = chain
            .dropped_modulus(level)
            .expect("fused stage requires a rescale level")
            .value() as f64;

        let operands = diagonals
            .iter()
            .map(|(&offset, coefficients)| {
                let plaintext = encode_rns(coefficients, embedding, basis, plain_scale);

                (offset, PreparedRnsPlaintextNtt::new(&plaintext, prepared))
            })
            .collect();

        Self {
            level,
            plain_scale,
            operands,
        }
    }

    pub fn rotation_offsets(&self) -> impl Iterator<Item = isize> + '_ {
        self.operands.iter().map(|(offset, _)| *offset)
    }
}

/// Evaluate multiple prepared source contributions into one ciphertext.
///
/// Every source must use the same CKKS level and scale. Rotation keys
/// are supplied through the existing FFT256 StageContext.
///
/// The complete sum is rescaled exactly once.
pub fn apply_fused_fft256_sources(
    sources: &[(&RnsCkksCiphertext, &PreparedFusedFft256Source)],
    ctx: &StageContext<'_>,
) -> RnsCkksCiphertext {
    let prepared_qp = crate::ring::PreparedRnsNttPlan::new(ctx.qp_plan);
    apply_fused_fft256_sources_with_qp(sources, ctx, &prepared_qp)
}

fn apply_fused_fft256_sources_with_qp(
    sources: &[(&RnsCkksCiphertext, &PreparedFusedFft256Source)],
    ctx: &StageContext<'_>,
    prepared_qp: &crate::ring::PreparedRnsNttPlan,
) -> RnsCkksCiphertext {
    assert!(!sources.is_empty(), "at least one fused source required");

    let expected_level = sources[0].1.level;
    let expected_scale = sources[0].0.scale();

    // R28_FUSED_OPERATION_PROFILE: opt-in operation-level profiling.
    let profile_enabled = std::env::var("FHE_R28_FUSED_PROFILE").is_ok_and(|value| value == "1");

    let mut rotation_ms = 0.0_f64;
    let mut multiply_ms = 0.0_f64;
    let mut addition_ms = 0.0_f64;
    let mut rotation_count = 0usize;
    let mut multiply_count = 0usize;
    let mut addition_count = 0usize;

    let mut accumulator: Option<RnsCkksCiphertext> = None;

    // R29_7A_BOUNDED_FUSED_TERMS
    // Compute independent contributions concurrently, but accumulate
    // in the original source/operand order before the single rescale.
    use rayon::prelude::*;

    let jobs: Vec<_> = sources
        .iter()
        .flat_map(|&(input, prepared_source)| {
            input.assert_matches_chain(ctx.chain);
            assert_eq!(input.level(), expected_level);
            assert_eq!(prepared_source.level, expected_level);
            assert_eq!(input.scale(), expected_scale);
            assert_eq!(input.basis().moduli(), ctx.prepared.moduli());

            prepared_source
                .operands
                .iter()
                .map(move |(offset, plaintext)| {
                    (input, prepared_source.plain_scale, *offset, plaintext)
                })
        })
        .collect();

    // Four temporary terms bound memory consumption. The shared
    // FFT3840 pool remains limited by FHE_R29_FFT_WORKERS.
    let max_inflight_terms = std::env::var("FHE_R29_FUSED_INFLIGHT")
        .ok()
        .map(|value| {
            let count = value
                .parse::<usize>()
                .expect("FHE_R29_FUSED_INFLIGHT must be an integer");
            assert!(count > 0, "FHE_R29_FUSED_INFLIGHT must be positive");
            count
        })
        .unwrap_or(4)
        .min(8);

    for chunk in jobs.chunks(max_inflight_terms) {
        let terms: Vec<_> = crate::eblas::encrypted_fft3840::fft3840_pool().install(|| {
            chunk
                .par_iter()
                .map(|&(input, plain_scale, offset, prepared_plaintext)| {
                    let rotation_started = profile_enabled.then(std::time::Instant::now);

                    let rotated = if offset == 0 {
                        None
                    } else {
                        let key = ctx
                            .keys
                            .get(&offset)
                            .expect("missing fused FFT256 rotation key");

                        let rotated_rlwe =
                            crate::ckks::apply_hybrid_rns_galois_automorphism_prepared_qp(
                                input.rlwe(),
                                key,
                                ctx.qp_plan,
                                prepared_qp,
                            );

                        Some(RnsCkksCiphertext::new(
                            rotated_rlwe,
                            input.state().clone(),
                            ctx.chain,
                        ))
                    };

                    let local_rotation_ms = if offset == 0 {
                        0.0
                    } else {
                        rotation_started
                            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                            .unwrap_or(0.0)
                    };

                    let operand = rotated.as_ref().unwrap_or(input);

                    let multiply_started = profile_enabled.then(std::time::Instant::now);

                    let term = multiply_plain_rns_ckks_with_prepared_ntt(
                        operand,
                        prepared_plaintext,
                        plain_scale,
                        ctx.chain,
                        ctx.prepared,
                    );

                    let local_multiply_ms = multiply_started
                        .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);

                    (term, local_rotation_ms, local_multiply_ms, offset != 0)
                })
                .collect()
        });

        // Indexed Rayon collection preserves the original job order.
        for (term, local_rotation_ms, local_multiply_ms, rotated) in terms {
            rotation_ms += local_rotation_ms;
            multiply_ms += local_multiply_ms;

            if rotated {
                rotation_count += 1;
            }
            multiply_count += 1;

            let started = profile_enabled.then(std::time::Instant::now);
            let had_accumulator = accumulator.is_some();

            accumulator = Some(match accumulator {
                Some(previous) => ctx.evaluator.add(&previous, &term),
                None => term,
            });

            if let Some(started) = started {
                addition_ms += started.elapsed().as_secs_f64() * 1000.0;
                if had_accumulator {
                    addition_count += 1;
                }
            }
        }
    }

    let started = profile_enabled.then(std::time::Instant::now);

    let result = rescale_rns_ckks_to_next(
        &accumulator.expect("nonempty fused FFT256 accumulator"),
        ctx.chain,
    );

    if let Some(started) = started {
        let rescale_ms = started.elapsed().as_secs_f64() * 1000.0;

        println!(
            "R28_FUSED_PROFILE rotation_ms={rotation_ms:.3} multiply_ms={multiply_ms:.3} addition_ms={addition_ms:.3} rescale_ms={rescale_ms:.3} rotations={rotation_count} multiplications={multiply_count} additions={addition_count}"
        );
    }

    result
}

// R30.9A.21A: caller-managed rotation interface.
//
// The callback owns rotation-key lookup and key lifetime. The FFT kernel
// owns only plaintext multiplication, accumulation, and rescaling.
// This supports persistent or key-major evaluation without copying keys.
#[allow(clippy::too_many_arguments)]
pub fn apply_fused_fft256_sources_with_rotation<F>(
    sources: &[(&RnsCkksCiphertext, &PreparedFusedFft256Source)],
    evaluator: &crate::ckks::RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
    prepared: &PreparedRnsNttPlan,
    rotate: &mut F,
) -> RnsCkksCiphertext
where
    F: FnMut(isize, &RnsCkksCiphertext) -> RnsCkksCiphertext,
{
    assert!(!sources.is_empty(), "at least one fused source required");

    let expected_level = sources[0].1.level;
    let expected_scale = sources[0].0.scale();
    let mut accumulator: Option<RnsCkksCiphertext> = None;

    for &(input, source) in sources {
        input.assert_matches_chain(chain);
        assert_eq!(input.level(), expected_level);
        assert_eq!(source.level, expected_level);
        assert_eq!(input.scale(), expected_scale);
        assert_eq!(input.basis().moduli(), prepared.moduli());

        for (offset, plaintext) in &source.operands {
            let rotated = if *offset == 0 {
                None
            } else {
                let rotated = rotate(*offset, input);
                rotated.assert_matches_chain(chain);
                assert_eq!(rotated.level(), expected_level);
                assert_eq!(rotated.scale(), expected_scale);
                Some(rotated)
            };

            let operand = rotated.as_ref().unwrap_or(input);
            let term = multiply_plain_rns_ckks_with_prepared_ntt(
                operand,
                plaintext,
                source.plain_scale,
                chain,
                prepared,
            );

            accumulator = Some(match accumulator {
                Some(previous) => evaluator.add(&previous, &term),
                None => term,
            });
        }
    }

    rescale_rns_ckks_to_next(
        &accumulator.expect("nonempty fused FFT256 accumulator"),
        chain,
    )
}

// R30.9A.18: sequential fused evaluation with a caller-owned key factory.
// The factory is invoked once per nonzero term; each key is dropped before
// the next term. Source/operand order and the single rescale are preserved.
pub fn apply_fused_fft256_sources_streaming<F>(
    sources: &[(&RnsCkksCiphertext, &PreparedFusedFft256Source)],
    evaluator: &crate::ckks::RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
    prepared: &PreparedRnsNttPlan,
    qp_plan: &crate::ring::RnsNttPlan,
    prepared_qp: &PreparedRnsNttPlan,
    make_key: &mut F,
) -> RnsCkksCiphertext
where
    F: FnMut(isize) -> crate::ckks::PreparedHybridRnsGaloisKey,
{
    assert!(!sources.is_empty(), "at least one fused source required");
    let expected_level = sources[0].1.level;
    let expected_scale = sources[0].0.scale();
    let mut accumulator: Option<RnsCkksCiphertext> = None;

    for &(input, source) in sources {
        input.assert_matches_chain(chain);
        assert_eq!(input.level(), expected_level);
        assert_eq!(source.level, expected_level);
        assert_eq!(input.scale(), expected_scale);
        assert_eq!(input.basis().moduli(), prepared.moduli());
        for (offset, plaintext) in &source.operands {
            let rotated = if *offset == 0 {
                None
            } else {
                let key = make_key(*offset);
                let rlwe = crate::ckks::apply_hybrid_rns_galois_automorphism_prepared_qp(
                    input.rlwe(),
                    &key,
                    qp_plan,
                    prepared_qp,
                );
                Some(RnsCkksCiphertext::new(rlwe, input.state().clone(), chain))
            };
            let operand = rotated.as_ref().unwrap_or(input);
            let term = multiply_plain_rns_ckks_with_prepared_ntt(
                operand,
                plaintext,
                source.plain_scale,
                chain,
                prepared,
            );
            accumulator = Some(match accumulator {
                Some(previous) => evaluator.add(&previous, &term),
                None => term,
            });
        }
    }
    rescale_rns_ckks_to_next(
        &accumulator.expect("nonempty fused FFT256 accumulator"),
        chain,
    )
}

// R30.9A.21C: stage-0 key-generation schedule.
//
// One entry per nonzero operand occurrence, in the exact order
// consumed by the original streaming evaluator.
// The schedule is independent of the number of ciphertext batches.
pub fn fft3840_stage0_key_schedule(
    prepared_sources: &[PreparedFusedFft256Source; 15],
) -> Vec<isize> {
    prepared_sources
        .iter()
        .flat_map(|source| source.rotation_offsets())
        .filter(|&offset| offset != 0)
        .collect()
}

// R30.9A.21B: bounded cross-batch stage-0 key reuse.
//
// Batches are evaluated in operand order. Each nonzero operand's
// prepared Galois key is generated once and borrowed across every
// batch in the window, then released.
//
// The caller bounds memory by controlling batches.len().
#[allow(clippy::too_many_arguments)]
// R35_9F_2_GROUP_AWARE_KEY_CACHE
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Fft3840Stage0KeyCacheStats {
    pub hits: usize,
    pub misses: usize,
    pub generations: usize,
    pub peak_retained_keys: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn apply_fft3840_packed_fft256_stage0_batched_streaming_cached<F>(
    batches: &[[RnsCkksCiphertext; 15]],
    prepared_sources: &[PreparedFusedFft256Source; 15],
    evaluator: &crate::ckks::RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
    prepared: &PreparedRnsNttPlan,
    qp_plan: &crate::ring::RnsNttPlan,
    prepared_qp: &PreparedRnsNttPlan,
    make_key: &mut F,
    eval_mode: &str,
    eval_workers: usize,
    capacity: usize,
) -> (Vec<[RnsCkksCiphertext; 2]>, Fft3840Stage0KeyCacheStats)
where
    F: FnMut(isize) -> crate::ckks::PreparedHybridRnsGaloisKey,
{
    use std::collections::BTreeSet;

    assert!(capacity <= 48, "R36.7D supports at most 48 retained keys");

    let group_offsets = |range: std::ops::Range<usize>| -> BTreeSet<isize> {
        range
            .flat_map(|branch| prepared_sources[branch].rotation_offsets())
            .filter(|&offset| offset != 0)
            .collect()
    };

    let group0 = group_offsets(0..8);
    let group1 = group_offsets(8..15);

    let retained_offsets: BTreeSet<isize> = group0
        .intersection(&group1)
        .copied()
        .take(capacity)
        .collect();

    let mut cache =
        std::collections::BTreeMap::<isize, crate::ckks::PreparedHybridRnsGaloisKey>::new();

    let mut stats = Fft3840Stage0KeyCacheStats::default();

    let mut provider =
        |offset: isize, consume: &mut dyn FnMut(&crate::ckks::PreparedHybridRnsGaloisKey)| {
            // R35_9F_4_RELEASE_AFTER_FINAL_USE
            if let Some(key) = cache.remove(&offset) {
                stats.hits += 1;
                consume(&key);
                // The second occurrence is the final scheduled use.
                // Release the key immediately after evaluation.
                return;
            }

            stats.misses += 1;
            stats.generations += 1;

            let key = make_key(offset);

            if retained_offsets.contains(&offset) {
                assert!(cache.len() < capacity);
                cache.insert(offset, key);
                stats.peak_retained_keys = stats.peak_retained_keys.max(cache.len());
                consume(cache.get(&offset).expect("inserted retained key"));
            } else {
                consume(&key);
            }
        };

    let outputs = apply_fft3840_packed_fft256_stage0_batched_streaming_with_provider(
        batches,
        prepared_sources,
        evaluator,
        chain,
        prepared,
        qp_plan,
        prepared_qp,
        &mut provider,
        eval_mode,
        eval_workers,
    );

    assert!(stats.peak_retained_keys <= capacity);

    (outputs, stats)
}

#[allow(clippy::too_many_arguments)]
pub fn apply_fft3840_packed_fft256_stage0_batched_streaming_configured<F>(
    batches: &[[RnsCkksCiphertext; 15]],
    prepared_sources: &[PreparedFusedFft256Source; 15],
    evaluator: &crate::ckks::RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
    prepared: &PreparedRnsNttPlan,
    qp_plan: &crate::ring::RnsNttPlan,
    prepared_qp: &PreparedRnsNttPlan,
    make_key: &mut F,
    eval_mode: &str,
    eval_workers: usize,
) -> Vec<[RnsCkksCiphertext; 2]>
where
    F: FnMut(isize) -> crate::ckks::PreparedHybridRnsGaloisKey,
{
    let mut provider =
        |offset: isize, consume: &mut dyn FnMut(&crate::ckks::PreparedHybridRnsGaloisKey)| {
            let key = make_key(offset);
            consume(&key);
        };

    apply_fft3840_packed_fft256_stage0_batched_streaming_with_provider(
        batches,
        prepared_sources,
        evaluator,
        chain,
        prepared,
        qp_plan,
        prepared_qp,
        &mut provider,
        eval_mode,
        eval_workers,
    )
}

// R36_8C_PERSISTENT_STAGE0_KEY_CACHE
// Fixed-offset, bounded, cross-window cache. The caller owns this
// object and must preserve it between stage-0 evaluation windows.
pub struct Fft3840PersistentStage0KeyCache {
    retained_offsets: std::collections::BTreeSet<isize>,
    keys: std::collections::BTreeMap<isize, crate::ckks::PreparedHybridRnsGaloisKey>,
    pub hits: usize,
    pub generations: usize,
    pub peak_keys: usize,
}

impl Fft3840PersistentStage0KeyCache {
    pub fn new(prepared_sources: &[PreparedFusedFft256Source; 15], capacity: usize) -> Self {
        use std::collections::BTreeSet;

        assert!(capacity <= 48, "R36.8C capacity must not exceed 48");

        let offsets = |range: std::ops::Range<usize>| -> BTreeSet<isize> {
            range
                .flat_map(|branch| prepared_sources[branch].rotation_offsets())
                .filter(|&offset| offset != 0)
                .collect()
        };

        let group0 = offsets(0..8);
        let group1 = offsets(8..15);

        let retained_offsets = group0
            .intersection(&group1)
            .copied()
            .take(capacity)
            .collect();

        Self {
            retained_offsets,
            keys: std::collections::BTreeMap::new(),
            hits: 0,
            generations: 0,
            peak_keys: 0,
        }
    }

    pub fn with_key<F>(
        &mut self,
        offset: isize,
        make_key: &mut F,
        consume: &mut dyn FnMut(&crate::ckks::PreparedHybridRnsGaloisKey),
    ) where
        F: FnMut(isize) -> crate::ckks::PreparedHybridRnsGaloisKey,
    {
        if let Some(key) = self.keys.get(&offset) {
            self.hits += 1;
            consume(key);
            return;
        }

        self.generations += 1;
        let key = make_key(offset);

        if self.retained_offsets.contains(&offset) {
            assert!(self.keys.len() < self.retained_offsets.len());
            assert!(self.keys.insert(offset, key).is_none());
            self.peak_keys = self.peak_keys.max(self.keys.len());
            consume(self.keys.get(&offset).expect("retained key must exist"));
        } else {
            consume(&key);
        }
    }

    // R36_8E_1_EXTERNAL_PREPARED_KEY_API
    /// Whether a rotation key is already retained by this cache.
    // R36_8E_2_TEST_RETENTION_POLICY
    #[cfg(test)]
    fn for_test_offsets(offsets: &[isize]) -> Self {
        Self {
            retained_offsets: offsets.iter().copied().collect(),
            keys: std::collections::BTreeMap::new(),
            hits: 0,
            generations: 0,
            peak_keys: 0,
        }
    }

    pub fn contains(&self, offset: isize) -> bool {
        self.keys.contains_key(&offset)
    }

    /// Whether this offset belongs to the fixed retention set.
    pub fn should_retain(&self, offset: isize) -> bool {
        self.retained_offsets.contains(&offset)
    }

    /// Consume a key generated outside the cache, typically by a bounded
    /// parallel worker batch. A cache hit must not receive a generated key.
    ///
    /// This preserves the accounting and retention behavior of `with_key`.
    pub fn with_prepared_key(
        &mut self,
        offset: isize,
        prepared_key: Option<crate::ckks::PreparedHybridRnsGaloisKey>,
        consume: &mut dyn FnMut(&crate::ckks::PreparedHybridRnsGaloisKey),
    ) {
        if let Some(key) = self.keys.get(&offset) {
            assert!(
                prepared_key.is_none(),
                "R36.8E: prefetched key supplied for a cache hit"
            );
            self.hits += 1;
            consume(key);
            return;
        }

        let key = prepared_key.expect("R36.8E: missing prefetched key for cache miss");
        self.generations += 1;

        if self.retained_offsets.contains(&offset) {
            assert!(
                self.keys.len() < self.retained_offsets.len(),
                "R36.8E: persistent cache capacity exceeded"
            );
            assert!(
                self.keys.insert(offset, key).is_none(),
                "R36.8E: duplicate retained key"
            );
            self.peak_keys = self.peak_keys.max(self.keys.len());
            consume(
                self.keys
                    .get(&offset)
                    .expect("R36.8E: retained key must exist"),
            );
        } else {
            consume(&key);
        }
    }

    pub fn retained_keys(&self) -> usize {
        self.keys.len()
    }

    pub fn capacity(&self) -> usize {
        self.retained_offsets.len()
    }
}

// R36_8G_1A_SHARED_READ_ONLY_KEY_REGISTRY
/// Prepared rotation keys that can be borrowed concurrently by FFT workers.
///
/// Keys are inserted during preparation and then accessed immutably.
/// The caller controls the number of resident keys.
#[derive(Debug, Default)]
pub struct Fft3840SharedKeyRegistry {
    keys: std::collections::BTreeMap<isize, crate::ckks::PreparedHybridRnsGaloisKey>,
}

impl Fft3840SharedKeyRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, offset: isize, key: crate::ckks::PreparedHybridRnsGaloisKey) {
        assert_ne!(offset, 0, "identity rotation does not require a key");
        assert!(
            !self.keys.contains_key(&offset),
            "duplicate prepared rotation key"
        );
        self.keys.insert(offset, key);
    }

    pub fn get(&self, offset: isize) -> &crate::ckks::PreparedHybridRnsGaloisKey {
        self.keys
            .get(&offset)
            .expect("missing shared prepared rotation key")
    }

    // R36_8G_1B_SHARED_PROVIDER
    /// Supply a prepared key by immutable reference to an FFT evaluator.
    pub fn with_key(
        &self,
        offset: isize,
        consume: &mut dyn FnMut(&crate::ckks::PreparedHybridRnsGaloisKey),
    ) {
        consume(self.get(offset));
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

// R35_9F_1_BORROWED_KEY_PROVIDER
#[allow(clippy::too_many_arguments)]
pub fn apply_fft3840_packed_fft256_stage0_batched_streaming_with_provider<F>(
    batches: &[[RnsCkksCiphertext; 15]],
    prepared_sources: &[PreparedFusedFft256Source; 15],
    evaluator: &crate::ckks::RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
    prepared: &PreparedRnsNttPlan,
    qp_plan: &crate::ring::RnsNttPlan,
    prepared_qp: &PreparedRnsNttPlan,
    with_key: &mut F,
    eval_mode: &str,
    eval_workers: usize,
) -> Vec<[RnsCkksCiphertext; 2]>
where
    F: FnMut(isize, &mut dyn FnMut(&crate::ckks::PreparedHybridRnsGaloisKey)),
{
    // R36_4F_CONST_GENERIC_SPECIALIZATION
    apply_fft3840_stage0_batched_with_rotation_mode::<true, F>(
        batches,
        prepared_sources,
        evaluator,
        chain,
        prepared,
        qp_plan,
        prepared_qp,
        with_key,
        eval_mode,
        eval_workers,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_fft3840_stage0_batched_with_rotation_mode<const CANONICAL: bool, F>(
    batches: &[[RnsCkksCiphertext; 15]],
    prepared_sources: &[PreparedFusedFft256Source; 15],
    evaluator: &crate::ckks::RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
    prepared: &PreparedRnsNttPlan,
    qp_plan: &crate::ring::RnsNttPlan,
    prepared_qp: &PreparedRnsNttPlan,
    with_key: &mut F,
    eval_mode: &str,
    eval_workers: usize,
) -> Vec<[RnsCkksCiphertext; 2]>
where
    F: FnMut(isize, &mut dyn FnMut(&crate::ckks::PreparedHybridRnsGaloisKey)),
{
    // R35_8C_EXPLICIT_EVAL_CONFIGURATION

    assert!(
        eval_mode == "serial" || eval_mode == "parallel",
        "invalid R35.8 stage-zero evaluation mode"
    );

    let parallel_evaluation = eval_mode == "parallel";

    assert!(eval_workers >= 1);

    let eval_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(eval_workers)
        .build()
        .expect("stage-zero evaluation thread pool");

    println!("R35_8B_STAGE0_EVAL_MODE={eval_mode}");
    println!("R35_8B_STAGE0_EVAL_WORKERS={eval_workers}");

    assert!(!batches.is_empty(), "empty FFT3840 batch window");

    for batch in batches {
        assert!(batch.iter().all(|ct| ct.level() == 3));
    }

    // Output groups contain branches 0..8 and 8..15.
    // Preserve the existing source-major, operand-major ordering
    // independently within each batch accumulator.
    let mut outputs: Vec<Vec<RnsCkksCiphertext>> =
        (0..batches.len()).map(|_| Vec::with_capacity(2)).collect();

    for group in 0..2 {
        let range = if group == 0 { 0..8 } else { 8..15 };

        let mut accumulators: Vec<Option<RnsCkksCiphertext>> =
            (0..batches.len()).map(|_| None).collect();

        for branch in range {
            let source = &prepared_sources[branch];

            assert_eq!(source.level, 3);
            assert_eq!(chain.level(3).moduli(), prepared.moduli());

            for batch in batches {
                let input = &batch[branch];
                input.assert_matches_chain(chain);
                assert_eq!(input.level(), source.level);
                assert_eq!(input.basis().moduli(), prepared.moduli());
            }

            // R36_5B2_INDEXED_OPTIONAL_CANONICAL_SOURCE
            // R36_5B_CONDITIONAL_CANONICAL_PREPARATION
            // R36_4D_STAGE0_CANONICAL_SOURCE_HOISTING
            // Reconstruct each input a-polynomial once per source branch.
            let canonical_sources: Vec<crate::grafting::PreparedCanonicalGaloisSource> =
                if CANONICAL {
                    batches
                        .iter()
                        .map(|batch| {
                            let input = &batch[branch];
                            let original_a = crate::ring::RnsPolynomial::from_residues(
                                input
                                    .rlwe()
                                    .limbs()
                                    .iter()
                                    .map(|limb| limb.a().clone())
                                    .collect(),
                            );
                            crate::grafting::PreparedCanonicalGaloisSource::new(&original_a)
                        })
                        .collect()
                } else {
                    Vec::new()
                };

            for (offset, plaintext) in &source.operands {
                // R35_9F_1: Borrow key material for this operand only.
                // The provider may own a temporary key or retain a cached key.
                // R35_8A_PARALLEL_STAGE0
                let mut run_evaluation = |key: Option<&crate::ckks::PreparedHybridRnsGaloisKey>| {
                    let evaluate_batch = |((accumulator, batch), canonical_source): (
                        (&mut Option<RnsCkksCiphertext>, &[RnsCkksCiphertext; 15]),
                        Option<&crate::grafting::PreparedCanonicalGaloisSource>,
                    )| {
                        let input = &batch[branch];
                        let rotated = key.map(|key| {
                            let rlwe = if CANONICAL {
                                crate::ckks::apply_hybrid_rns_galois_automorphism_prepared_canonical_qp(
                                    input.rlwe(),
                                    key,
                                    qp_plan,
                                    prepared_qp,
                                    canonical_source.expect(
                                        "missing canonical source in canonical specialization",
                                    ),
                                )
                            } else {
                                crate::ckks::apply_hybrid_rns_galois_automorphism_prepared_qp(
                                    input.rlwe(),
                                    key,
                                    qp_plan,
                                    prepared_qp,
                                )
                            };
                            RnsCkksCiphertext::new(rlwe, input.state().clone(), chain)
                        });
                        let operand = rotated.as_ref().unwrap_or(input);
                        let term = multiply_plain_rns_ckks_with_prepared_ntt(
                            operand,
                            plaintext,
                            source.plain_scale,
                            chain,
                            prepared,
                        );
                        let previous = accumulator.take();
                        *accumulator = Some(match previous {
                            Some(previous) => evaluator.add(&previous, &term),
                            None => term,
                        });
                    };

                    if parallel_evaluation {
                        eval_pool.install(|| {
                            accumulators
                                .par_iter_mut()
                                .zip(batches.par_iter())
                                .enumerate()
                                .map(|(index, pair)| (pair, canonical_sources.get(index)))
                                .for_each(evaluate_batch);
                        });
                    } else {
                        accumulators
                            .iter_mut()
                            .zip(batches.iter())
                            .enumerate()
                            .map(|(index, pair)| (pair, canonical_sources.get(index)))
                            .for_each(evaluate_batch);
                    }
                };

                if *offset == 0 {
                    run_evaluation(None);
                } else {
                    with_key(*offset, &mut |key| run_evaluation(Some(key)));
                }
            }
        }

        for (batch_index, accumulator) in accumulators.into_iter().enumerate() {
            let result = rescale_rns_ckks_to_next(
                &accumulator.expect("empty FFT3840 batch accumulator"),
                chain,
            );

            assert_eq!(result.level(), 4);
            outputs[batch_index].push(result);
        }
    }

    outputs
        .into_iter()
        .map(|mut group_outputs| {
            assert_eq!(group_outputs.len(), 2);
            let second = group_outputs.pop().expect("missing group 1");
            let first = group_outputs.pop().expect("missing group 0");
            [first, second]
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn apply_fft3840_packed_fft256_stage0_batched_streaming<F>(
    batches: &[[RnsCkksCiphertext; 15]],
    prepared_sources: &[PreparedFusedFft256Source; 15],
    evaluator: &crate::ckks::RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
    prepared: &PreparedRnsNttPlan,
    qp_plan: &crate::ring::RnsNttPlan,
    prepared_qp: &PreparedRnsNttPlan,
    make_key: &mut F,
) -> Vec<[RnsCkksCiphertext; 2]>
where
    F: FnMut(isize) -> crate::ckks::PreparedHybridRnsGaloisKey,
{
    let eval_mode =
        std::env::var("FHE_R35_8_STAGE0_EVAL_MODE").unwrap_or_else(|_| "parallel".to_string());
    let eval_workers = std::env::var("FHE_R35_8_STAGE0_EVAL_WORKERS")
        .ok()
        .map(|v| v.parse::<usize>().expect("invalid stage-zero worker count"))
        .unwrap_or(4);

    apply_fft3840_packed_fft256_stage0_batched_streaming_configured(
        batches,
        prepared_sources,
        evaluator,
        chain,
        prepared,
        qp_plan,
        prepared_qp,
        make_key,
        &eval_mode,
        eval_workers,
    )
}

/// Streaming first-stage FFT3840 evaluation. Groups are evaluated sequentially
/// to guarantee at most one hybrid rotation key is resident at a time.
#[allow(clippy::too_many_arguments)]
pub fn apply_fft3840_packed_fft256_stage0_streaming<F>(
    branches: &[RnsCkksCiphertext; 15],
    prepared_sources: &[PreparedFusedFft256Source; 15],
    evaluator: &crate::ckks::RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
    prepared: &PreparedRnsNttPlan,
    qp_plan: &crate::ring::RnsNttPlan,
    prepared_qp: &PreparedRnsNttPlan,
    make_key: &mut F,
) -> [RnsCkksCiphertext; 2]
where
    F: FnMut(isize) -> crate::ckks::PreparedHybridRnsGaloisKey,
{
    assert!(branches.iter().all(|ct| ct.level() == 3));
    std::array::from_fn(|group| {
        let range = if group == 0 { 0..8 } else { 8..15 };
        let inputs: Vec<_> = range
            .map(|i| (&branches[i], &prepared_sources[i]))
            .collect();
        let output = apply_fused_fft256_sources_streaming(
            &inputs,
            evaluator,
            chain,
            prepared,
            qp_plan,
            prepared_qp,
            make_key,
        );
        assert_eq!(output.level(), 4);
        output
    })
}

/// Apply a full-slot diagonal operator to plaintext values.
///
/// This is a mathematical reference, not a ciphertext evaluator.
pub fn apply_fused_source_diagonals_plain(
    input: &[Complex64],
    diagonals: &FusedSourceDiagonals,
) -> Vec<Complex64> {
    let slots = input.len();
    assert!(slots > 0);

    let mut output = vec![Complex64::new(0.0, 0.0); slots];

    for (&offset, coefficients) in diagonals {
        assert_eq!(coefficients.len(), slots);

        for (destination, &coefficient) in coefficients.iter().enumerate() {
            if coefficient == Complex64::new(0.0, 0.0) {
                continue;
            }

            let source = (destination as isize + offset).rem_euclid(slots as isize) as usize;

            output[destination] += coefficient * input[source];
        }
    }

    output
}

// R26_H11_15_PACKED_FFT3840_HANDOFF

/// Execute the prepared, packed FFT256 continuation of all 15
/// horizontal FFT15 frequency branches.
///
/// Inputs are natural-order FFT15 branches at CKKS level 3.
/// The first stage fuses branch-dependent FFT3840 twiddles,
/// radix-4 butterflies, and placement into two packed outputs.
///
/// Group A contains branches 0..8; group B contains 8..15.
/// Each output contains eight 4096-slot lanes.
///
/// All prepared operands and rotation keys are caller-owned.
/// This function performs no encryption or key generation.
pub fn apply_fft3840_packed_fft256_from_fft15(
    branches: &[RnsCkksCiphertext; 15],
    prepared_sources: &[PreparedFusedFft256Source; 15],
    later_stages: &[crate::eblas::prepared_radix4_fft256::PreparedFft256Stage; 3],
    contexts: [&crate::eblas::prepared_radix4_fft256::StageContext<'_>; 4],
) -> [RnsCkksCiphertext; 2] {
    let prepared_context =
        crate::eblas::prepared_radix4_fft256::PreparedFft256ExecutionContext::new(contexts);

    apply_fft3840_packed_fft256_with_context(
        branches,
        prepared_sources,
        later_stages,
        contexts,
        &prepared_context,
    )
}

// R30_9A_16A_STAGEWISE_FFT3840
/// Execute the fused first FFT256 stage for both packed FFT3840 groups.
///
/// The caller owns the stage-0 rotation keys and may release them
/// after this function returns.
pub fn apply_fft3840_packed_fft256_stage0(
    branches: &[RnsCkksCiphertext; 15],
    prepared_sources: &[PreparedFusedFft256Source; 15],
    ctx: &crate::eblas::prepared_radix4_fft256::StageContext<'_>,
    prepared_qp: &crate::ring::PreparedRnsNttPlan,
) -> [RnsCkksCiphertext; 2] {
    assert!(
        branches.iter().all(|ciphertext| ciphertext.level() == 3),
        "packed FFT3840 requires level-3 FFT15 branches"
    );

    let evaluate_group = |group: usize| {
        let range = if group == 0 { 0..8 } else { 8..15 };
        let inputs: Vec<(&RnsCkksCiphertext, &PreparedFusedFft256Source)> = range
            .map(|branch| (&branches[branch], &prepared_sources[branch]))
            .collect();

        let output = apply_fused_fft256_sources_with_qp(&inputs, ctx, prepared_qp);
        assert_eq!(output.level(), 4);
        output
    };

    let (output0, output1) = crate::eblas::encrypted_fft3840::fft3840_pool()
        .install(|| rayon::join(|| evaluate_group(0), || evaluate_group(1)));

    [output0, output1]
}

pub fn apply_fft3840_packed_fft256_with_context(
    branches: &[RnsCkksCiphertext; 15],
    prepared_sources: &[PreparedFusedFft256Source; 15],
    later_stages: &[crate::eblas::prepared_radix4_fft256::PreparedFft256Stage; 3],
    contexts: [&crate::eblas::prepared_radix4_fft256::StageContext<'_>; 4],
    prepared_context: &crate::eblas::prepared_radix4_fft256::PreparedFft256ExecutionContext,
) -> [RnsCkksCiphertext; 2] {
    // R28_4A_REUSABLE_FUSED_QP
    let shared_qp = prepared_context.qp_plans();
    use crate::eblas::prepared_radix4_fft256::apply_stage;

    assert!(
        branches.iter().all(|ciphertext| ciphertext.level() == 3),
        "packed FFT3840 requires level-3 FFT15 branches"
    );

    // R28_FFT_STAGE_TIMING: opt-in, non-invasive stage instrumentation.
    let timing_enabled = std::env::var("FHE_R28_FFT_STAGE_TIMING").is_ok_and(|value| value == "1");

    // R29_5_PARALLEL_FFT3840
    let evaluate_group = |group: usize| {
        let mut stage_totals_ms = [0.0_f64; 4];
        let range = if group == 0 { 0..8 } else { 8..15 };

        let inputs: Vec<(&RnsCkksCiphertext, &PreparedFusedFft256Source)> = range
            .map(|branch| (&branches[branch], &prepared_sources[branch]))
            .collect();

        let started = timing_enabled.then(std::time::Instant::now);
        let mut output = apply_fused_fft256_sources_with_qp(&inputs, contexts[0], &shared_qp[0]);

        if let Some(started) = started {
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            stage_totals_ms[0] += elapsed_ms;
            println!(
                "R28_STAGE_TIMING group={group} stage=0 elapsed_ms={elapsed_ms:.3} output_level={}",
                output.level()
            );
        }

        assert_eq!(output.level(), 4);

        for stage in 0..3 {
            let started = timing_enabled.then(std::time::Instant::now);
            output = apply_stage(&output, &later_stages[stage], contexts[stage + 1]);

            if let Some(started) = started {
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                stage_totals_ms[stage + 1] += elapsed_ms;
                println!(
                    "R28_STAGE_TIMING group={group} stage={} elapsed_ms={elapsed_ms:.3} output_level={}",
                    stage + 1,
                    output.level()
                );
            }

            assert_eq!(output.level(), stage + 5);
        }

        (output, stage_totals_ms)
    };

    let ((output0, timing0), (output1, timing1)) = crate::eblas::encrypted_fft3840::fft3840_pool()
        .install(|| rayon::join(|| evaluate_group(0), || evaluate_group(1)));

    let outputs = [output0, output1];
    let stage_totals_ms: [f64; 4] = std::array::from_fn(|stage| timing0[stage] + timing1[stage]);

    if timing_enabled {
        for (stage, elapsed_ms) in stage_totals_ms.iter().enumerate() {
            println!("R28_STAGE_TOTAL stage={stage} elapsed_ms={elapsed_ms:.3}");
        }
        println!(
            "R28_STAGE_TOTAL_SUM_MS={:.3}",
            stage_totals_ms.iter().sum::<f64>()
        );
    }

    outputs
}

#[cfg(test)]
mod tests {
    use super::*;

    // R36_4E_EXACT_BATCHED_STAGE0_EQUIVALENCE
    #[test]
    fn r36_4e_stage0_canonical_hoisting_matches_legacy_exactly() {
        use crate::ckks::hybrid_rns_galois_key::PreparedHybridRnsGaloisKey;
        use crate::ckks::{CkksChainState, RnsCkksEvaluationKeys, RnsCkksEvaluator};
        use crate::grafting::{BoundedGadgetLayout, RnsRlweCiphertext};
        use crate::ring::{Modulus, ModulusBasis, Polynomial, RnsNttPlan};
        use crate::rlwe::ErrorDistribution;
        use crate::rlwe::RlweCiphertext;
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        const N: usize = 16;

        let q = ModulusBasis::new(
            [12_289_u64, 40_961, 65_537, 114_689, 786_433]
                .into_iter()
                .map(Modulus::new)
                .collect(),
        );

        let p = ModulusBasis::new(vec![Modulus::new(147_457), Modulus::new(163_841)]);

        let chain = ModulusChain::from_top_basis(q);

        assert_eq!(chain.max_level(), 4);

        let level = 3;
        let active_q = chain.level(level).clone();

        assert_eq!(active_q.moduli().len(), 2);

        let q_plan = RnsNttPlan::new(active_q.moduli().to_vec(), N);
        let prepared_q = PreparedRnsNttPlan::new(&q_plan);

        let mut qp_moduli = active_q.moduli().to_vec();
        qp_moduli.extend_from_slice(p.moduli());

        let qp_plan = RnsNttPlan::new(qp_moduli, N);
        let prepared_qp = PreparedRnsNttPlan::new(&qp_plan);

        let embedding = CkksCanonicalEmbedding::new(N);
        let slots = embedding.slot_count();

        let plain_scale = chain
            .dropped_modulus(level)
            .expect("level 3 must permit rescaling")
            .value() as f64;

        let make_diagonals = |branch: usize| {
            let mut diagonals = FusedSourceDiagonals::new();

            diagonals.insert(0, vec![Complex64::new(0.25, 0.0); slots]);

            if branch == 0 || branch == 8 {
                diagonals.insert(1, vec![Complex64::new(0.125, 0.0); slots]);
            }

            diagonals
        };

        let prepared_sources: [PreparedFusedFft256Source; 15] = std::array::from_fn(|branch| {
            PreparedFusedFft256Source::new(
                level,
                &make_diagonals(branch),
                &embedding,
                &chain,
                &prepared_q,
            )
        });

        let secret: [i8; N] = [1, 0, -1, 1, 0, -1, 1, 0, -1, 0, 1, 0, 1, -1, 0, 1];

        let make_ciphertext = |batch_index: usize, branch: usize| {
            let limbs = active_q
                .moduli()
                .iter()
                .copied()
                .map(|modulus| {
                    let b = Polynomial::new(
                        modulus,
                        (0..N)
                            .map(|i| {
                                (11 + 17 * i as u64 + 29 * branch as u64 + 37 * batch_index as u64)
                                    % modulus.value()
                            })
                            .collect(),
                    );

                    let a = Polynomial::new(
                        modulus,
                        (0..N)
                            .map(|i| {
                                (7 + 23 * i as u64 + 31 * branch as u64 + 41 * batch_index as u64)
                                    % modulus.value()
                            })
                            .collect(),
                    );

                    RlweCiphertext::new(b, a)
                })
                .collect();

            let rlwe = RnsRlweCiphertext::from_limbs(limbs);

            RnsCkksCiphertext::new(
                rlwe,
                CkksChainState::new(&chain, level, plain_scale),
                &chain,
            )
        };

        let batches: Vec<[RnsCkksCiphertext; 15]> = (0..2)
            .map(|batch_index| std::array::from_fn(|branch| make_ciphertext(batch_index, branch)))
            .collect();

        let exponent = crate::ckks::rotation_exponent_left(N, 1);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x364e_2026);

        let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
            BoundedGadgetLayout::new(active_q.clone(), 8),
            p,
            &secret,
            exponent,
            ErrorDistribution::BoundedUniform { bound: 0 },
            2,
            &qp_plan,
            &mut key_rng,
        );

        let evaluation_keys = RnsCkksEvaluationKeys::new();
        let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

        for (batch_count, mode, workers) in [
            (1_usize, "serial", 1_usize),
            (1, "parallel", 2),
            (2, "parallel", 2),
        ] {
            let run = |use_canonical: bool| {
                let mut provider =
                    |offset: isize, consume: &mut dyn FnMut(&PreparedHybridRnsGaloisKey)| {
                        assert_eq!(offset, 1);
                        consume(&key);
                    };

                if use_canonical {
                    apply_fft3840_stage0_batched_with_rotation_mode::<true, _>(
                        &batches[..batch_count],
                        &prepared_sources,
                        &evaluator,
                        &chain,
                        &prepared_q,
                        &qp_plan,
                        &prepared_qp,
                        &mut provider,
                        mode,
                        workers,
                    )
                } else {
                    apply_fft3840_stage0_batched_with_rotation_mode::<false, _>(
                        &batches[..batch_count],
                        &prepared_sources,
                        &evaluator,
                        &chain,
                        &prepared_q,
                        &qp_plan,
                        &prepared_qp,
                        &mut provider,
                        mode,
                        workers,
                    )
                }
            };

            let legacy = run(false);
            let canonical = run(true);

            // R36_5C1_CONTROLLED_STAGE0_BENCHMARK
            if std::env::var("FHE_R36_5C_BENCH").as_deref() == Ok("1") {
                let warmups = 2usize;
                let repeats = 7usize;

                for use_canonical in [false, true] {
                    let implementation = if use_canonical { "canonical" } else { "legacy" };

                    for _ in 0..warmups {
                        let _ = run(use_canonical);
                    }

                    let mut measurements = Vec::with_capacity(repeats);

                    for _ in 0..repeats {
                        let started = std::time::Instant::now();
                        let result = run(use_canonical);
                        let seconds = started.elapsed().as_secs_f64();

                        assert_eq!(result.len(), batch_count);
                        measurements.push(seconds);
                    }

                    measurements.sort_by(f64::total_cmp);

                    println!(
                        "R36_5C_BENCH implementation={} batches={} mode={} workers={} warmups={} repeats={} median_seconds={:.9} min_seconds={:.9} max_seconds={:.9}",
                        implementation,
                        batch_count,
                        mode,
                        workers,
                        warmups,
                        repeats,
                        measurements[repeats / 2],
                        measurements[0],
                        measurements[repeats - 1],
                    );
                }
            }

            assert_eq!(legacy.len(), canonical.len());

            for (batch_index, (reference, optimized)) in
                legacy.iter().zip(canonical.iter()).enumerate()
            {
                for group in 0..2 {
                    assert_eq!(
                        reference[group].rlwe(),
                        optimized[group].rlwe(),
                        "R36.4E exact RLWE mismatch: \
                         batches={batch_count} mode={mode} \
                         batch={batch_index} group={group}"
                    );

                    assert_eq!(reference[group].state(), optimized[group].state());

                    assert_eq!(reference[group].level(), 4);
                }
            }

            println!(
                "R36_4E_CASE=PASS batches={batch_count} \
                 mode={mode} workers={workers}"
            );
        }

        println!("R36_4E_EXACT_STAGE0_EQUIVALENCE_STATUS=PASS");
    }

    #[test]
    fn fused_source_lane_ignores_unselected_source_slots() {
        let slots = 32;
        let lane_slots = 8;

        let mut stage = Diagonals::new();
        stage.insert(0, vec![Complex64::new(1.0, 0.0); lane_slots]);

        let fused = fuse_fft256_source_lane(&stage, slots, lane_slots, 2, 1);

        let input: Vec<Complex64> = (0..slots)
            .map(|index| Complex64::new(index as f64 + 1.0, 0.0))
            .collect();

        let output = apply_fused_source_diagonals_plain(&input, &fused);

        for (index, &value) in output.iter().enumerate() {
            let expected = if (8..16).contains(&index) {
                input[16 + index - 8]
            } else {
                Complex64::new(0.0, 0.0)
            };

            assert_eq!(value, expected);
        }
    }

    #[test]
    fn fused_source_lane_handles_wraparound() {
        let slots = 32;
        let lane_slots = 8;

        let mut coefficients = vec![Complex64::new(0.0, 0.0); lane_slots];

        coefficients[7] = Complex64::new(2.0, 0.0);

        let mut stage = Diagonals::new();
        stage.insert(1, coefficients);

        let fused = fuse_fft256_source_lane(&stage, slots, lane_slots, 3, 0);

        let input: Vec<Complex64> = (0..slots)
            .map(|index| Complex64::new(index as f64, 0.0))
            .collect();

        let output = apply_fused_source_diagonals_plain(&input, &fused);

        assert_eq!(output[7], input[24] * Complex64::new(2.0, 0.0));

        for (index, value) in output.iter().enumerate() {
            if index != 7 {
                assert_eq!(*value, Complex64::new(0.0, 0.0));
            }
        }
    }
}

// R36_8E_2_CACHE_POLICY_TESTS
#[cfg(test)]
mod r36_8e_2_cache_tests {
    use super::Fft3840PersistentStage0KeyCache;

    #[test]
    fn retained_offsets_are_selected_without_keys() {
        let cache = Fft3840PersistentStage0KeyCache::for_test_offsets(&[3, 7, 11, 13]);

        for offset in [3, 7, 11, 13] {
            assert!(cache.should_retain(offset));
            assert!(!cache.contains(offset));
        }

        for offset in [1, 5, 9, 15] {
            assert!(!cache.should_retain(offset));
            assert!(!cache.contains(offset));
        }

        assert_eq!(cache.capacity(), 4);
        assert_eq!(cache.retained_keys(), 0);
    }

    #[test]
    fn initial_accounting_is_zero() {
        let cache = Fft3840PersistentStage0KeyCache::for_test_offsets(&[2, 4]);

        assert_eq!(cache.hits, 0);
        assert_eq!(cache.generations, 0);
        assert_eq!(cache.peak_keys, 0);
        assert_eq!(cache.retained_keys(), 0);
    }

    #[test]
    fn duplicate_retention_offsets_do_not_increase_capacity() {
        let cache = Fft3840PersistentStage0KeyCache::for_test_offsets(&[3, 3, 7, 7]);

        assert_eq!(cache.capacity(), 2);
        assert!(cache.should_retain(3));
        assert!(cache.should_retain(7));
        assert!(!cache.should_retain(5));
    }
}
