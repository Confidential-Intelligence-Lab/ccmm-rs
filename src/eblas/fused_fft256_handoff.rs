//! Source-specific diagonal placement for dense FFT256 handoff.
//!
//! This module constructs the linear operator that combines source-lane
//! extraction, destination-lane placement, and an existing FFT256 stage.
//!
//! No ciphertext operations or key generation occur during construction.

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
    apply_hybrid_rns_galois_automorphism_with_prepared_ntt,
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
    assert!(!sources.is_empty(), "at least one fused source required");

    let expected_level = sources[0].1.level;
    let expected_scale = sources[0].0.scale();

    let mut accumulator: Option<RnsCkksCiphertext> = None;

    for &(input, prepared_source) in sources {
        input.assert_matches_chain(ctx.chain);

        assert_eq!(input.level(), expected_level);
        assert_eq!(prepared_source.level, expected_level);
        assert_eq!(input.scale(), expected_scale);
        assert_eq!(input.basis().moduli(), ctx.prepared.moduli());

        for (offset, prepared_plaintext) in &prepared_source.operands {
            let rotated = if *offset == 0 {
                None
            } else {
                let key = ctx
                    .keys
                    .get(offset)
                    .expect("missing fused FFT256 rotation key");

                let rotated_rlwe = apply_hybrid_rns_galois_automorphism_with_prepared_ntt(
                    input.rlwe(),
                    key,
                    ctx.qp_plan,
                );

                Some(RnsCkksCiphertext::new(
                    rotated_rlwe,
                    input.state().clone(),
                    ctx.chain,
                ))
            };

            let operand = rotated.as_ref().unwrap_or(input);

            let term = multiply_plain_rns_ckks_with_prepared_ntt(
                operand,
                prepared_plaintext,
                prepared_source.plain_scale,
                ctx.chain,
                ctx.prepared,
            );

            accumulator = Some(match accumulator {
                Some(previous) => ctx.evaluator.add(&previous, &term),
                None => term,
            });
        }
    }

    rescale_rns_ckks_to_next(
        &accumulator.expect("nonempty fused FFT256 accumulator"),
        ctx.chain,
    )
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
    use crate::eblas::prepared_radix4_fft256::apply_stage;

    assert!(
        branches.iter().all(|ciphertext| ciphertext.level() == 3),
        "packed FFT3840 requires level-3 FFT15 branches"
    );

    std::array::from_fn(|group| {
        let range = if group == 0 { 0..8 } else { 8..15 };

        let inputs: Vec<(&RnsCkksCiphertext, &PreparedFusedFft256Source)> = range
            .map(|branch| (&branches[branch], &prepared_sources[branch]))
            .collect();

        let mut output = apply_fused_fft256_sources(&inputs, contexts[0]);

        assert_eq!(output.level(), 4);

        for stage in 0..3 {
            output = apply_stage(&output, &later_stages[stage], contexts[stage + 1]);

            assert_eq!(output.level(), stage + 5);
        }

        output
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
