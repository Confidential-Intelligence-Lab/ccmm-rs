// R30_6B_PRODUCTION_FFT16_DIAGONALS
//! Vertical encrypted FFT2160 components for the optimized UHD pipeline.
//!
//! FFT16 uses two radix-4 DIT stages with stride-256 rotations.
//! The first stage includes the FFT135-to-FFT16 twiddle factors.
//! Public diagonal generation is separate from CKKS preparation.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use num_complex::Complex64;

use crate::eblas::prepared_radix4_fft256::Diagonals;

pub const UHD_ROWS: usize = 2160;
pub const UHD_COLS: usize = 3840;
pub const FFT135_BRANCHES: usize = 135;
pub const FFT16_ROWS: usize = 16;
pub const ROW_STRIDE: usize = 256;
pub const TILE_SLOTS: usize = FFT16_ROWS * ROW_STRIDE;
pub const HORIZONTAL_BRANCHES: usize = 15;
pub const SLOTS_PER_CIPHERTEXT: usize = 32768;
pub const TILES_PER_CIPHERTEXT: usize = SLOTS_PER_CIPHERTEXT / TILE_SLOTS;
pub const CIPHERTEXTS_PER_BATCH: usize = 2;

fn root(order: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * (exponent % order) as f64 / order as f64)
}

fn reverse_base4_2(index: usize) -> usize {
    assert!(index < FFT16_ROWS);
    (index % 4) * 4 + index / 4
}

/// Construct FFT16 stage diagonals for one packed ciphertext.
///
/// `ciphertext_in_batch` is 0 or 1. The second ciphertext contains
/// seven active 4096-slot tiles and one padding tile.
///
/// `k_cross` is the natural FFT135 frequency index in 0..135.
pub fn fft16_stage_diagonals(
    slots: usize,
    span: usize,
    k_cross: usize,
    ciphertext_in_batch: usize,
) -> Diagonals {
    assert!(span == 4 || span == 16);
    assert!(k_cross < FFT135_BRANCHES);
    assert!(ciphertext_in_batch < CIPHERTEXTS_PER_BATCH);
    assert_eq!(slots, SLOTS_PER_CIPHERTEXT);

    let active_tiles = if ciphertext_in_batch == 0 { 8 } else { 7 };
    let quarter = span / 4;
    let mut diagonals: Diagonals = BTreeMap::new();

    for tile in 0..active_tiles {
        let tile_base = tile * TILE_SLOTS;

        for local_row in 0..FFT16_ROWS {
            let within = local_row % span;
            let p = within / quarter;
            let j = within % quarter;

            for q in 0..4 {
                let input_row = (local_row / span) * span + j + q * quarter;

                let offset = (input_row as isize - local_row as isize) * ROW_STRIDE as isize;

                let mut coefficient = root(span, q * j) * root(4, p * q);

                if span == 4 {
                    coefficient *= root(UHD_ROWS, k_cross * reverse_base4_2(input_row));
                }

                let values = diagonals
                    .entry(offset)
                    .or_insert_with(|| vec![Complex64::new(0.0, 0.0); slots]);

                for column in 0..ROW_STRIDE {
                    let target = tile_base + local_row * ROW_STRIDE + column;
                    let source = target as isize + offset;

                    assert!(source >= 0 && source < slots as isize);
                    assert_eq!(source as usize / TILE_SLOTS, tile);
                    assert_eq!((source as usize % TILE_SLOTS) / ROW_STRIDE, input_row);

                    values[target] = coefficient;
                }
            }
        }
    }

    diagonals
}

// R30_6C_PREPARED_FFT16_EXECUTION

use crate::ckks::{CkksCanonicalEmbedding, RnsCkksCiphertext};
use crate::eblas::prepared_radix4_fft256::{apply_stage, PreparedFft256Stage, StageContext};
use crate::ring::{ModulusChain, PreparedRnsNttPlan};

/// Prepared two-stage vertical FFT16 for one FFT135 output frequency.
///
/// The public operand sets are prepared at levels 11 and 12.
/// Rotation keys and evaluator resources remain caller-owned.
pub struct PreparedVerticalFft16 {
    k_cross: usize,
    ciphertext_in_batch: usize,
    stages: [PreparedFft256Stage; 2],
}

impl PreparedVerticalFft16 {
    pub fn new(
        k_cross: usize,
        ciphertext_in_batch: usize,
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        prepared: [&PreparedRnsNttPlan; 2],
    ) -> Self {
        assert!(k_cross < FFT135_BRANCHES);
        assert!(ciphertext_in_batch < CIPHERTEXTS_PER_BATCH);

        let stages = std::array::from_fn(|stage| {
            let span = [4, 16][stage];
            let diagonals =
                fft16_stage_diagonals(SLOTS_PER_CIPHERTEXT, span, k_cross, ciphertext_in_batch);

            PreparedFft256Stage::new(11 + stage, &diagonals, embedding, chain, prepared[stage])
        });

        Self {
            k_cross,
            ciphertext_in_batch,
            stages,
        }
    }

    pub fn cross_frequency(&self) -> usize {
        self.k_cross
    }

    pub fn ciphertext_in_batch(&self) -> usize {
        self.ciphertext_in_batch
    }

    /// Evaluate levels 11 -> 12 -> 13 without intermediate decryption.
    ///
    /// Contexts must supply hybrid-QP keys for the signed offsets
    /// used by the two stages.
    pub fn execute(
        &self,
        input: &RnsCkksCiphertext,
        contexts: [&StageContext<'_>; 2],
    ) -> RnsCkksCiphertext {
        assert_eq!(input.level(), 11);

        let intermediate = apply_stage(input, &self.stages[0], contexts[0]);
        assert_eq!(intermediate.level(), 12);

        let output = apply_stage(&intermediate, &self.stages[1], contexts[1]);
        assert_eq!(output.level(), 13);

        output
    }
}

// R30_7A_PRODUCTION_FFT135_PLAN

/// Radix decomposition of the cross-ciphertext FFT135.
pub const FFT135_RADICES: [usize; 4] = [5, 3, 3, 3];

/// One contribution to a cross-ciphertext FFT butterfly.
#[derive(Clone, Debug, PartialEq)]
pub struct Fft135Term {
    pub input: usize,
    pub coefficient: Complex64,
}

/// One output of a cross-ciphertext FFT butterfly.
#[derive(Clone, Debug, PartialEq)]
pub struct Fft135Butterfly {
    pub output: usize,
    pub terms: Vec<Fft135Term>,
}

/// One DIF stage of FFT135.
#[derive(Clone, Debug, PartialEq)]
pub struct Fft135Stage {
    pub level: usize,
    pub radix: usize,
    pub span: usize,
    pub butterflies: Vec<Fft135Butterfly>,
}

/// Reusable four-stage FFT135 execution schedule.
///
/// Input ciphertexts are in natural spatial order.
/// Outputs are in mixed-radix DIF order, matching the historical
/// R16 FFT135 permutation.
pub struct Fft135Plan {
    stages: Vec<Fft135Stage>,
}

impl Default for Fft135Plan {
    fn default() -> Self {
        Self::new()
    }
}

impl Fft135Plan {
    pub fn new() -> Self {
        let mut stages = Vec::with_capacity(FFT135_RADICES.len());
        let mut span = FFT135_BRANCHES;

        for (stage_index, radix) in FFT135_RADICES.into_iter().enumerate() {
            assert_eq!(span % radix, 0);
            let m = span / radix;
            let mut butterflies = Vec::with_capacity(FFT135_BRANCHES);

            for base in (0..FFT135_BRANCHES).step_by(span) {
                for j in 0..m {
                    for k in 0..radix {
                        let terms = (0..radix)
                            .map(|q| Fft135Term {
                                input: base + j + q * m,
                                coefficient: root(radix, q * k) * root(span, j * k),
                            })
                            .collect();

                        butterflies.push(Fft135Butterfly {
                            output: base + j + k * m,
                            terms,
                        });
                    }
                }
            }

            assert_eq!(butterflies.len(), FFT135_BRANCHES);

            stages.push(Fft135Stage {
                level: 7 + stage_index,
                radix,
                span,
                butterflies,
            });

            span = m;
        }

        assert_eq!(span, 1);
        Self { stages }
    }

    pub fn stages(&self) -> &[Fft135Stage] {
        &self.stages
    }

    /// Natural frequency represented by one DIF output position.
    pub fn natural_frequency(mut output: usize) -> usize {
        assert!(output < FFT135_BRANCHES);

        // DIF output storage uses reversed radix order: [3, 3, 3, 5].
        let d0 = output % 3;
        output /= 3;
        let d1 = output % 3;
        output /= 3;
        let d2 = output % 3;
        output /= 3;
        let d3 = output;

        assert!(d3 < 5);
        45 * d0 + 15 * d1 + 5 * d2 + d3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r30_6b_fft16_geometry_and_padding() {
        for span in [4, 16] {
            for ciphertext in 0..2 {
                let diagonals = fft16_stage_diagonals(32768, span, 37, ciphertext);

                assert!(!diagonals.is_empty());

                let active_tiles = if ciphertext == 0 { 8 } else { 7 };

                for (&offset, values) in &diagonals {
                    assert_eq!(values.len(), 32768);
                    assert_eq!(offset % 256, 0);
                    assert!(offset.abs() <= 3072);

                    for tile in active_tiles..8 {
                        let start = tile * 4096;
                        assert!(values[start..start + 4096]
                            .iter()
                            .all(|value| value.norm_sqr() == 0.0));
                    }
                }
            }
        }
    }

    #[test]
    fn r30_6b_fft16_twiddle_depends_on_cross_frequency() {
        let first = fft16_stage_diagonals(32768, 4, 0, 0);
        let second = fft16_stage_diagonals(32768, 4, 1, 0);

        assert!(first.iter().any(|(offset, values)| {
            values
                .iter()
                .zip(&second[offset])
                .any(|(a, b)| (*a - *b).norm() > 1e-12)
        }));

        let first = fft16_stage_diagonals(32768, 16, 0, 0);
        let second = fft16_stage_diagonals(32768, 16, 1, 0);

        assert_eq!(first, second);
    }

    #[test]
    fn r30_6b_fft16_matches_historical_diagonal_equations() {
        let span = 4;
        let k_cross = 19;
        let diagonals = fft16_stage_diagonals(32768, span, k_cross, 0);

        for row in 0..16 {
            let quarter = span / 4;
            let within = row % span;
            let p = within / quarter;
            let j = within % quarter;

            for q in 0..4 {
                let input_row = (row / span) * span + j + q * quarter;
                let offset = (input_row as isize - row as isize) * 256;
                let expected = root(span, q * j)
                    * root(4, p * q)
                    * root(2160, k_cross * reverse_base4_2(input_row));

                let actual = diagonals[&offset][row * 256];
                assert!((actual - expected).norm() < 1e-12);
            }
        }
    }
}

// R30_7B_FFT135_OPERAND_CATALOG

/// A reusable public scalar operand for an FFT135 stage.
#[derive(Clone, Debug)]
pub struct Fft135ScalarOperand {
    pub coefficient: Complex64,
    pub exponent: usize,
    pub order: usize,
}

/// A stage-local catalog of distinct scalar operands.
///
/// Unity is represented by None and requires no prepared plaintext.
pub struct Fft135OperandCatalog {
    pub level: usize,
    pub operands: Vec<Fft135ScalarOperand>,
    pub references: Vec<Vec<Option<usize>>>,
}

impl Fft135OperandCatalog {
    pub fn from_stage(stage: &Fft135Stage) -> Self {
        assert!(stage.span > 0);
        assert_eq!(stage.butterflies.len(), FFT135_BRANCHES);

        let mut operands = Vec::<Fft135ScalarOperand>::new();
        let mut lookup = BTreeMap::<usize, usize>::new();
        let mut references = Vec::with_capacity(stage.butterflies.len());

        for butterfly in &stage.butterflies {
            let mut row = Vec::with_capacity(butterfly.terms.len());

            for term in &butterfly.terms {
                let mut matched = None;

                for exponent in 0..stage.span {
                    if (root(stage.span, exponent) - term.coefficient).norm() < 1e-12 {
                        matched = Some(exponent);
                        break;
                    }
                }

                let exponent = matched.expect("FFT135 twiddle is a span root");

                if exponent == 0 {
                    row.push(None);
                } else {
                    let index = *lookup.entry(exponent).or_insert_with(|| {
                        let index = operands.len();
                        operands.push(Fft135ScalarOperand {
                            coefficient: root(stage.span, exponent),
                            exponent,
                            order: stage.span,
                        });
                        index
                    });

                    row.push(Some(index));
                }
            }

            references.push(row);
        }

        Self {
            level: stage.level,
            operands,
            references,
        }
    }
}

// R30_7A_FFT135_TESTS
#[cfg(test)]
mod r30_7a_tests {
    use super::*;

    #[test]
    fn r30_7a_stage_geometry() {
        let plan = Fft135Plan::new();

        assert_eq!(plan.stages().len(), 4);

        for (index, stage) in plan.stages().iter().enumerate() {
            assert_eq!(stage.level, 7 + index);
            assert_eq!(stage.radix, FFT135_RADICES[index]);
            assert_eq!(stage.butterflies.len(), 135);

            let mut outputs = stage
                .butterflies
                .iter()
                .map(|butterfly| butterfly.output)
                .collect::<Vec<_>>();

            outputs.sort_unstable();
            assert_eq!(outputs, (0..135).collect::<Vec<_>>());

            assert!(stage.butterflies.iter().all(|butterfly| {
                butterfly.terms.len() == stage.radix
                    && butterfly.terms.iter().all(|term| term.input < 135)
            }));
        }
    }

    #[test]
    fn r30_7a_frequency_permutation_is_bijective() {
        let mut frequencies = (0..135)
            .map(Fft135Plan::natural_frequency)
            .collect::<Vec<_>>();

        frequencies.sort_unstable();

        assert_eq!(frequencies, (0..135).collect::<Vec<_>>());
    }

    #[test]
    fn r30_7a_known_dif_frequency_order() {
        let expected = [0, 45, 90, 15, 60, 105, 30, 75, 120, 5, 50, 95];

        for (index, frequency) in expected.into_iter().enumerate() {
            assert_eq!(Fft135Plan::natural_frequency(index), frequency);
        }
    }

    #[test]
    fn r30_7a_fft135_matches_direct_dft() {
        use std::f64::consts::PI;

        let plan = Fft135Plan::new();

        let input = (0..135)
            .map(|index| Complex64::new((index as f64 * 0.17).sin(), (index as f64 * 0.11).cos()))
            .collect::<Vec<_>>();

        let mut values = input.clone();

        for stage in plan.stages() {
            let mut next = vec![Complex64::new(0.0, 0.0); 135];

            for butterfly in &stage.butterflies {
                next[butterfly.output] = butterfly
                    .terms
                    .iter()
                    .map(|term| values[term.input] * term.coefficient)
                    .sum();
            }

            values = next;
        }

        let mut max_error = 0.0_f64;

        for (output, actual) in values.iter().enumerate() {
            let frequency = Fft135Plan::natural_frequency(output);

            let expected: Complex64 = input
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let angle = -2.0 * PI * (frequency * index) as f64 / 135.0;

                    *value * Complex64::from_polar(1.0, angle)
                })
                .sum();

            max_error = max_error.max((*actual - expected).norm());
        }

        println!("R30_7A_FFT135_MAX_ERROR={max_error:.12e}");

        assert!(
            max_error < 1e-9,
            "FFT135 butterfly plan differs from direct DFT"
        );
    }
}

// R30_7C_ENCRYPTED_FFT135_EVALUATOR

use crate::ckks::{rescale_rns_ckks_to_next, RnsCkksEvaluator};
use crate::eblas::level1::PreparedComplexSlotsCp;
use crate::ring::RnsNttPlan;

/// Execute the 135-way cross-ciphertext DIF FFT.
///
/// The input consists of exactly 135 ciphertexts at CKKS level 7.
/// Each output ciphertext retains its original SIMD slot geometry.
/// No rotations or intermediate decryption are performed.
pub fn execute_encrypted_fft135(
    ciphertexts: Vec<RnsCkksCiphertext>,
    embedding: &CkksCanonicalEmbedding,
    evaluator: &RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
) -> Vec<RnsCkksCiphertext> {
    execute_encrypted_fft135_observed(ciphertexts, embedding, evaluator, chain, None)
}

fn execute_encrypted_fft135_observed(
    mut ciphertexts: Vec<RnsCkksCiphertext>,
    embedding: &CkksCanonicalEmbedding,
    evaluator: &RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
    observer: Option<&dyn Fn(&str)>,
) -> Vec<RnsCkksCiphertext> {
    assert_eq!(
        ciphertexts.len(),
        FFT135_BRANCHES,
        "FFT135 requires exactly 135 ciphertexts"
    );
    assert!(
        ciphertexts.iter().all(|ct| ct.level() == 7),
        "FFT135 inputs must be at CKKS level 7"
    );

    let plan = Fft135Plan::new();
    let slots = embedding.slot_count();

    for (stage_index, stage) in plan.stages().iter().enumerate() {
        if let Some(observer) = observer {
            observer(&format!("FFT135_STAGE{}_BEGIN", stage_index + 1));
        }
        let level = stage.level;

        assert!(
            ciphertexts.iter().all(|ct| ct.level() == level),
            "FFT135 stage input level mismatch"
        );

        let ntt_plan = RnsNttPlan::new(chain.level(level).moduli().to_vec(), embedding.degree());
        let prepared_ntt = PreparedRnsNttPlan::new(&ntt_plan);

        let catalog = Fft135OperandCatalog::from_stage(stage);

        // Every prepared scalar is shared by all butterflies at this level.
        let prepared_operands: Vec<PreparedComplexSlotsCp> = catalog
            .operands
            .iter()
            .map(|operand| {
                let public_slots = vec![operand.coefficient; slots];

                PreparedComplexSlotsCp::new(
                    &ciphertexts[0],
                    &public_slots,
                    embedding,
                    chain,
                    &ntt_plan,
                )
            })
            .collect();

        // Unity must undergo the same plaintext-scale multiplication
        // as non-unity terms before ciphertext accumulation.
        let identity_slots = vec![Complex64::new(1.0, 0.0); slots];
        let prepared_identity = PreparedComplexSlotsCp::new(
            &ciphertexts[0],
            &identity_slots,
            embedding,
            chain,
            &ntt_plan,
        );

        let mut next: Vec<Option<RnsCkksCiphertext>> = vec![None; FFT135_BRANCHES];

        for (butterfly, references) in stage.butterflies.iter().zip(&catalog.references) {
            let mut accumulator: Option<RnsCkksCiphertext> = None;

            for (term, reference) in butterfly.terms.iter().zip(references) {
                let source = &ciphertexts[term.input];

                let product = match reference {
                    None => crate::eblas::level1::multiply_complex_slots_cp_prepared_unrescaled(
                        source,
                        &prepared_identity,
                        chain,
                        &prepared_ntt,
                    ),
                    Some(index) => {
                        crate::eblas::level1::multiply_complex_slots_cp_prepared_unrescaled(
                            source,
                            &prepared_operands[*index],
                            chain,
                            &prepared_ntt,
                        )
                    }
                };

                accumulator = Some(match accumulator {
                    Some(previous) => evaluator.add(&previous, &product),
                    None => product,
                });
            }

            let accumulated = accumulator.expect("FFT135 butterfly must have terms");

            let output = rescale_rns_ckks_to_next(&accumulated, chain);

            assert_eq!(
                output.level(),
                level + 1,
                "FFT135 stage must consume exactly one CKKS level"
            );
            assert!(
                next[butterfly.output].is_none(),
                "FFT135 output destination assigned twice"
            );

            next[butterfly.output] = Some(output);
        }

        ciphertexts = next
            .into_iter()
            .map(|ct| ct.expect("FFT135 output destination missing"))
            .collect();

        if let Some(observer) = observer {
            observer(&format!("FFT135_STAGE{}_END", stage_index + 1));
        }
    }

    assert_eq!(ciphertexts.len(), FFT135_BRANCHES);
    assert!(ciphertexts.iter().all(|ct| ct.level() == 11));

    ciphertexts
}

// R30_8A_ENCRYPTED_FFT2160_COMPOSITION

/// Execute one complete encrypted vertical FFT2160 packing group.
///
/// The 135 input ciphertexts represent the 135 groups of 16 vertical
/// samples. Each ciphertext contains the 16 local rows in the
/// production UHD packing layout.
///
/// `ciphertext_in_batch` selects one of the two horizontal ciphertext
/// packing groups and determines the active FFT16 diagonal geometry.
///
/// FFT135 consumes levels 7..11 without rotations.
/// FFT16 consumes levels 11..13 using the supplied hybrid-QP contexts.
///
/// Output order preserves FFT135 DIF branch order. The corresponding
/// natural cross-frequency is `Fft135Plan::natural_frequency(index)`.
pub fn execute_encrypted_fft2160(
    ciphertexts: Vec<RnsCkksCiphertext>,
    ciphertext_in_batch: usize,
    embedding: &CkksCanonicalEmbedding,
    evaluator: &RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
    prepared: [&PreparedRnsNttPlan; 2],
    contexts: [&StageContext<'_>; 2],
) -> Vec<RnsCkksCiphertext> {
    execute_encrypted_fft2160_observed(
        ciphertexts,
        ciphertext_in_batch,
        embedding,
        evaluator,
        chain,
        prepared,
        contexts,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn execute_encrypted_fft2160_observed(
    ciphertexts: Vec<RnsCkksCiphertext>,
    ciphertext_in_batch: usize,
    embedding: &CkksCanonicalEmbedding,
    evaluator: &RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
    prepared: [&PreparedRnsNttPlan; 2],
    contexts: [&StageContext<'_>; 2],
    observer: Option<&dyn Fn(&str)>,
) -> Vec<RnsCkksCiphertext> {
    assert_eq!(
        ciphertexts.len(),
        FFT135_BRANCHES,
        "FFT2160 requires exactly 135 input ciphertexts"
    );
    assert!(
        ciphertexts.iter().all(|ct| ct.level() == 7),
        "FFT2160 inputs must be at CKKS level 7"
    );
    assert!(
        ciphertext_in_batch < CIPHERTEXTS_PER_BATCH,
        "invalid FFT2160 horizontal packing group"
    );

    let intermediate =
        execute_encrypted_fft135_observed(ciphertexts, embedding, evaluator, chain, observer);

    assert_eq!(intermediate.len(), FFT135_BRANCHES);
    assert!(intermediate.iter().all(|ct| ct.level() == 11));

    if let Some(observer) = observer {
        observer("FFT16_BEGIN");
    }

    let outputs = execute_streaming_vertical_fft16(
        &intermediate,
        ciphertext_in_batch,
        embedding,
        chain,
        prepared,
        contexts,
    );

    if let Some(observer) = observer {
        observer("FFT16_END");
    }

    assert_eq!(
        outputs.len(),
        FFT135_BRANCHES,
        "FFT2160 must preserve all 135 output branches"
    );
    assert!(
        outputs.iter().all(|ct| ct.level() == 13),
        "FFT2160 outputs must be at CKKS level 13"
    );

    outputs
}

// R30_8D_STREAMING_FFT16_EXECUTION

/// Execute FFT16 with bounded public-operand residency.
///
/// The frequency-independent span-16 operands are prepared once.
/// Each frequency-dependent span-4 operand set is prepared, executed,
/// and released before advancing to the next FFT135 output.
///
/// Rotation keys and NTT plans remain caller-owned and reusable.
fn execute_streaming_vertical_fft16(
    intermediate: &[RnsCkksCiphertext],
    ciphertext_in_batch: usize,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    prepared: [&PreparedRnsNttPlan; 2],
    contexts: [&StageContext<'_>; 2],
) -> Vec<RnsCkksCiphertext> {
    assert_eq!(intermediate.len(), FFT135_BRANCHES);
    assert!(intermediate.iter().all(|ct| ct.level() == 11));

    let slots = embedding.slot_count();

    let second_diagonals = fft16_stage_diagonals(slots, 16, 0, ciphertext_in_batch);

    let second = PreparedFft256Stage::new(12, &second_diagonals, embedding, chain, prepared[1]);

    let mut outputs = Vec::with_capacity(FFT135_BRANCHES);

    for (output_index, input) in intermediate.iter().enumerate() {
        let k_cross = Fft135Plan::natural_frequency(output_index);

        let first_diagonals = fft16_stage_diagonals(slots, 4, k_cross, ciphertext_in_batch);

        let first = PreparedFft256Stage::new(11, &first_diagonals, embedding, chain, prepared[0]);

        let stage12 = apply_stage(input, &first, contexts[0]);
        assert_eq!(stage12.level(), 12);

        let stage13 = apply_stage(&stage12, &second, contexts[1]);
        assert_eq!(stage13.level(), 13);

        outputs.push(stage13);
    }

    outputs
}

// R30_8B_SHARED_FFT16_BANK

/// Reusable FFT16 operands for one horizontal ciphertext packing group.
///
/// The first stage depends on the natural FFT135 frequency.
/// The second stage is frequency-independent and prepared only once.
///
/// The existing PreparedVerticalFft16 API remains available.
pub struct PreparedVerticalFft16Bank {
    ciphertext_in_batch: usize,
    first: Vec<PreparedFft256Stage>,
    second: PreparedFft256Stage,
}

impl PreparedVerticalFft16Bank {
    pub fn new(
        ciphertext_in_batch: usize,
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        prepared: [&PreparedRnsNttPlan; 2],
    ) -> Self {
        assert!(ciphertext_in_batch < 2);

        let slots = embedding.slot_count();

        let second_diagonals = fft16_stage_diagonals(slots, 16, 0, ciphertext_in_batch);

        let second = PreparedFft256Stage::new(12, &second_diagonals, embedding, chain, prepared[1]);

        let first = (0..FFT135_BRANCHES)
            .map(|k_cross| {
                let diagonals = fft16_stage_diagonals(slots, 4, k_cross, ciphertext_in_batch);

                PreparedFft256Stage::new(11, &diagonals, embedding, chain, prepared[0])
            })
            .collect();

        Self {
            ciphertext_in_batch,
            first,
            second,
        }
    }

    pub fn execute(
        &self,
        input: &RnsCkksCiphertext,
        k_cross: usize,
        contexts: [&StageContext<'_>; 2],
    ) -> RnsCkksCiphertext {
        assert!(k_cross < FFT135_BRANCHES);
        assert_eq!(input.level(), 11);

        let intermediate = apply_stage(input, &self.first[k_cross], contexts[0]);

        assert_eq!(intermediate.level(), 12);

        let output = apply_stage(&intermediate, &self.second, contexts[1]);

        assert_eq!(output.level(), 13);

        output
    }

    pub fn ciphertext_in_batch(&self) -> usize {
        self.ciphertext_in_batch
    }

    pub fn prepared_first_stage_count(&self) -> usize {
        self.first.len()
    }

    pub fn prepared_second_stage_count(&self) -> usize {
        1
    }
}

// R30_7B_FFT135_CATALOG_TESTS
#[cfg(test)]
mod r30_7b_tests {
    use super::*;

    #[test]
    fn r30_7b_catalog_reconstructs_all_twiddles() {
        let plan = Fft135Plan::new();

        for stage in plan.stages() {
            let catalog = Fft135OperandCatalog::from_stage(stage);

            assert_eq!(catalog.level, stage.level);
            assert_eq!(catalog.references.len(), FFT135_BRANCHES);
            assert!(catalog.operands.len() < stage.span);

            for (butterfly, references) in stage.butterflies.iter().zip(&catalog.references) {
                assert_eq!(butterfly.terms.len(), references.len());

                for (term, reference) in butterfly.terms.iter().zip(references) {
                    let reconstructed = match reference {
                        None => Complex64::new(1.0, 0.0),
                        Some(index) => catalog.operands[*index].coefficient,
                    };

                    assert!(
                        (reconstructed - term.coefficient).norm() < 1e-12,
                        "FFT135 scalar catalog reconstruction mismatch"
                    );
                }
            }
        }
    }

    #[test]
    fn r30_7b_catalog_excludes_unity() {
        let plan = Fft135Plan::new();

        for stage in plan.stages() {
            let catalog = Fft135OperandCatalog::from_stage(stage);

            assert!(catalog.operands.iter().all(|operand| {
                operand.exponent != 0
                    && operand.order == stage.span
                    && (operand.coefficient - Complex64::new(1.0, 0.0)).norm() > 1e-12
            }));
        }
    }
}
