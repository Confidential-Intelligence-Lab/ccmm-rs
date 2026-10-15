//! Encrypted mixed-radix FFT butterflies.
//!
//! R16.4c.2: radix-3, public transform coefficients, one CKKS level.

use num_complex::Complex64;

use crate::ckks::{
    mod_switch_rns_ckks_to_next, CkksCanonicalEmbedding, RnsCkksCiphertext, RnsCkksEvaluator,
};
use crate::ring::{ModulusChain, PreparedRnsNttPlan, RnsNttPlan};

use super::level1::{multiply_complex_slots_cp_prepared, PreparedComplexSlotsCp};

/// Prepared coefficients for a forward radix-3 DFT.
///
/// Preparation is specific to the input ciphertext level and RNS basis.
pub struct PreparedRadix3Cp {
    minus_half: PreparedComplexSlotsCp,
    imaginary_weight: PreparedComplexSlotsCp,
}

impl PreparedRadix3Cp {
    pub fn new(
        input: &RnsCkksCiphertext,
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        plan: &RnsNttPlan,
    ) -> Self {
        let slots = embedding.slot_count();

        let minus_half = vec![Complex64::new(-0.5, 0.0); slots];
        let imaginary_weight = vec![Complex64::new(0.0, -(3.0_f64).sqrt() / 2.0); slots];

        Self {
            minus_half: PreparedComplexSlotsCp::new(input, &minus_half, embedding, chain, plan),
            imaginary_weight: PreparedComplexSlotsCp::new(
                input,
                &imaginary_weight,
                embedding,
                chain,
                plan,
            ),
        }
    }
}

/// Computes a forward radix-3 DFT on three aligned CKKS ciphertexts.
///
/// Each ciphertext contains a vector of complex slots. The transform
/// acts independently at every slot position.
///
/// The result consumes exactly one CKKS level. No ciphertext rotations
/// or ciphertext-ciphertext multiplications are required.
///
/// Input ordering: X0, X1, X2.
/// Output ordering: Y0, Y1, Y2.
pub fn packed_radix3_dft_cp_prepared(
    evaluator: &RnsCkksEvaluator<'_>,
    inputs: [&RnsCkksCiphertext; 3],
    coefficients: &PreparedRadix3Cp,
    chain: &ModulusChain,
    plan: &PreparedRnsNttPlan,
) -> [RnsCkksCiphertext; 3] {
    let [x0, x1, x2] = inputs;

    for input in inputs {
        input.assert_matches_chain(chain);
        assert_eq!(input.level(), x0.level(), "radix-3 level mismatch");
        assert_eq!(input.scale(), x0.scale(), "radix-3 scale mismatch");
        assert_eq!(input.basis(), x0.basis(), "radix-3 basis mismatch");
    }

    assert!(
        chain.has_next_level(x0.level()),
        "radix-3 requires one additional CKKS level"
    );

    let sum = evaluator.add(x1, x2);
    let difference = evaluator.sub(x1, x2);

    let base = evaluator.add(x0, &sum);

    let half_sum = multiply_complex_slots_cp_prepared(&sum, &coefficients.minus_half, chain, plan);

    let weighted_difference = multiply_complex_slots_cp_prepared(
        &difference,
        &coefficients.imaginary_weight,
        chain,
        plan,
    );

    let aligned_x0 = mod_switch_rns_ckks_to_next(x0, chain);
    let aligned_base = mod_switch_rns_ckks_to_next(&base, chain);

    let common = evaluator.add(&aligned_x0, &half_sum);
    let y1 = evaluator.add(&common, &weighted_difference);
    let y2 = evaluator.sub(&common, &weighted_difference);

    let outputs = [aligned_base, y1, y2];

    for output in &outputs {
        assert_eq!(output.level(), x0.level() + 1);
        assert_eq!(output.scale(), x0.scale());
        assert_eq!(output.basis(), outputs[0].basis());
    }

    outputs
}

/// Prepared public coefficients for a forward radix-5 DFT.
pub struct PreparedRadix5Cp {
    weights: Vec<Vec<Option<PreparedComplexSlotsCp>>>,
}

impl PreparedRadix5Cp {
    pub fn new(
        input: &RnsCkksCiphertext,
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        plan: &RnsNttPlan,
    ) -> Self {
        let slots = embedding.slot_count();

        let weights = (0..5)
            .map(|output| {
                (0..5)
                    .map(|input_index| {
                        if output == 0 || input_index == 0 {
                            None
                        } else {
                            let angle =
                                -2.0 * std::f64::consts::PI * (output * input_index) as f64 / 5.0;
                            let coefficient = Complex64::new(angle.cos(), angle.sin());
                            let values = vec![coefficient; slots];
                            Some(PreparedComplexSlotsCp::new(
                                input, &values, embedding, chain, plan,
                            ))
                        }
                    })
                    .collect()
            })
            .collect();

        Self { weights }
    }
}

/// Forward radix-5 DFT on five aligned encrypted slot vectors.
///
/// Consumes one CKKS level, with no ciphertext rotations or
/// ciphertext-ciphertext multiplications.
pub fn packed_radix5_dft_cp_prepared(
    evaluator: &RnsCkksEvaluator<'_>,
    inputs: [&RnsCkksCiphertext; 5],
    coefficients: &PreparedRadix5Cp,
    chain: &ModulusChain,
    plan: &PreparedRnsNttPlan,
) -> [RnsCkksCiphertext; 5] {
    let first = inputs[0];

    for input in inputs {
        input.assert_matches_chain(chain);
        assert_eq!(input.level(), first.level(), "radix-5 level mismatch");
        assert_eq!(input.scale(), first.scale(), "radix-5 scale mismatch");
        assert_eq!(input.basis(), first.basis(), "radix-5 basis mismatch");
    }

    assert!(
        chain.has_next_level(first.level()),
        "radix-5 requires another CKKS level"
    );

    std::array::from_fn(|output| {
        if output == 0 {
            let mut sum = evaluator.add(inputs[0], inputs[1]);
            for input in &inputs[2..] {
                sum = evaluator.add(&sum, input);
            }
            return mod_switch_rns_ckks_to_next(&sum, chain);
        }

        let mut sum = mod_switch_rns_ckks_to_next(inputs[0], chain);

        for (input_index, input) in inputs.iter().enumerate().skip(1) {
            let weight = coefficients.weights[output][input_index]
                .as_ref()
                .expect("missing prepared radix-5 coefficient");

            let weighted = multiply_complex_slots_cp_prepared(input, weight, chain, plan);

            sum = evaluator.add(&sum, &weighted);
        }

        sum
    })
}

/// Prepared, twiddle-fused second stage of a forward FFT15.
///
/// For each k in 0..5, this prepares coefficients
///
/// C[k][q][j] = exp(-2*pi*i*(j*k/15 + j*q/3)).
///
/// The three input ciphertexts must contain aligned slot vectors at
/// the same CKKS level, scale, and basis.
pub struct PreparedFft15TwiddledRadix3Cp {
    weights: Vec<Vec<Vec<Option<PreparedComplexSlotsCp>>>>,
}

impl PreparedFft15TwiddledRadix3Cp {
    pub fn new(
        input_level1: &RnsCkksCiphertext,
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        plan_level1: &RnsNttPlan,
    ) -> Self {
        assert!(
            chain.has_next_level(input_level1.level()),
            "FFT15 fused radix-3 requires one remaining CKKS level"
        );

        let slots = embedding.slot_count();

        let weights = (0..5)
            .map(|k| {
                (0..3)
                    .map(|q| {
                        (0..3)
                            .map(|j| {
                                if j == 0 {
                                    return None;
                                }

                                let angle = -2.0
                                    * std::f64::consts::PI
                                    * ((j * k) as f64 / 15.0 + (j * q) as f64 / 3.0);

                                let coefficient = Complex64::new(angle.cos(), angle.sin());

                                let values = vec![coefficient; slots];

                                Some(PreparedComplexSlotsCp::new(
                                    input_level1,
                                    &values,
                                    embedding,
                                    chain,
                                    plan_level1,
                                ))
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        Self { weights }
    }
}

/// Executes the twiddle-fused radix-3 combine for FFT15 position k.
///
/// Inputs are the kth frequency outputs from the three radix-5
/// branches. Outputs are FFT15 frequencies k, k+5, and k+10.
///
/// This consumes exactly one CKKS level and performs no rotations
/// or ciphertext-ciphertext multiplications.
pub fn packed_fft15_twiddled_radix3_cp_prepared(
    evaluator: &RnsCkksEvaluator<'_>,
    inputs: [&RnsCkksCiphertext; 3],
    k: usize,
    coefficients: &PreparedFft15TwiddledRadix3Cp,
    chain: &ModulusChain,
    plan_level1: &PreparedRnsNttPlan,
) -> [RnsCkksCiphertext; 3] {
    assert!(k < 5, "FFT15 second-stage position must be < 5");

    let first = inputs[0];

    for input in inputs {
        input.assert_matches_chain(chain);
        assert_eq!(input.level(), first.level(), "FFT15 level mismatch");
        assert_eq!(input.scale(), first.scale(), "FFT15 scale mismatch");
        assert_eq!(input.basis(), first.basis(), "FFT15 basis mismatch");
    }

    assert!(
        chain.has_next_level(first.level()),
        "FFT15 fused combine requires another CKKS level"
    );

    std::array::from_fn(|q| {
        // j=0 always has coefficient 1.
        let mut sum = mod_switch_rns_ckks_to_next(inputs[0], chain);

        for (j, input) in inputs.iter().enumerate().skip(1) {
            let weight = coefficients.weights[k][q][j]
                .as_ref()
                .expect("missing FFT15 fused coefficient");

            let weighted = multiply_complex_slots_cp_prepared(input, weight, chain, plan_level1);

            sum = evaluator.add(&sum, &weighted);
        }

        assert_eq!(sum.level(), first.level() + 1);
        assert_eq!(sum.scale(), first.scale());

        sum
    })
}

/// Computes all 15 FFT15 frequency branches from three prepared radix-5 groups.
///
/// Each input group contains the five radix-5 outputs for one of the
/// three twiddled radix-3 inputs. Output index h is natural frequency
/// order: h = k + 5*q, where k in 0..5 and q in 0..3.
///
/// This consumes one CKKS level, reuses prepared public coefficients,
/// and performs no ciphertext rotations.
pub fn packed_fft15_all_branches_cp_prepared(
    evaluator: &RnsCkksEvaluator<'_>,
    branches: &[[RnsCkksCiphertext; 5]; 3],
    coefficients: &PreparedFft15TwiddledRadix3Cp,
    chain: &ModulusChain,
    plan_level1: &PreparedRnsNttPlan,
) -> [RnsCkksCiphertext; 15] {
    let mut outputs: [Option<RnsCkksCiphertext>; 15] = std::array::from_fn(|_| None);

    for k in 0..5 {
        let group = packed_fft15_twiddled_radix3_cp_prepared(
            evaluator,
            [&branches[0][k], &branches[1][k], &branches[2][k]],
            k,
            coefficients,
            chain,
            plan_level1,
        );

        for (q, ciphertext) in group.into_iter().enumerate() {
            let h = k + 5 * q;
            assert!(outputs[h].is_none(), "duplicate FFT15 branch");
            outputs[h] = Some(ciphertext);
        }
    }

    outputs.map(|output| output.expect("missing FFT15 branch"))
}

/// Prepared public twiddles for a six-point forward DFT.
///
/// The first stage consists of two radix-3 transforms.
/// The second stage combines their outputs using radix-2 DIT.
pub struct PreparedDft6Cp {
    radix3: PreparedRadix3Cp,
    radix2_twiddles: [PreparedComplexSlotsCp; 3],
}

impl PreparedDft6Cp {
    pub fn new(
        input: &RnsCkksCiphertext,
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        plan: &RnsNttPlan,
    ) -> Self {
        assert!(
            input.level() + 2 <= chain.max_level(),
            "DFT6 requires two CKKS levels"
        );

        let radix3 = PreparedRadix3Cp::new(input, embedding, chain, plan);

        // Prepared plaintexts are level-specific. Construct a
        // level-1 ciphertext solely to establish the appropriate
        // RNS basis and chain state for the second stage.
        let level1 = mod_switch_rns_ckks_to_next(input, chain);

        let level1_plan = RnsNttPlan::new(level1.basis().moduli().to_vec(), embedding.degree());

        let slots = embedding.slot_count();

        let radix2_twiddles = std::array::from_fn(|k| {
            let angle = -2.0 * std::f64::consts::PI * k as f64 / 6.0;
            let weight = Complex64::new(angle.cos(), angle.sin());
            let values = vec![weight; slots];

            PreparedComplexSlotsCp::new(&level1, &values, embedding, chain, &level1_plan)
        });

        Self {
            radix3,
            radix2_twiddles,
        }
    }
}

/// Six-point forward DFT on independently encrypted slot vectors.
///
/// Output order is natural frequency order, X[0] through X[5].
///
/// Stage 1: two radix-3 transforms, consuming one CKKS level.
/// Stage 2: three radix-2 DIT combinations, consuming one level.
///
/// Total multiplicative depth: two CKKS levels.
pub fn packed_dft6_cp_prepared(
    evaluator: &RnsCkksEvaluator<'_>,
    inputs: [&RnsCkksCiphertext; 6],
    coefficients: &PreparedDft6Cp,
    chain: &ModulusChain,
    plan_level0: &PreparedRnsNttPlan,
    plan_level1: &PreparedRnsNttPlan,
) -> [RnsCkksCiphertext; 6] {
    let first = inputs[0];

    assert!(
        first.level() + 2 <= chain.max_level(),
        "DFT6 requires two remaining CKKS levels"
    );

    for input in inputs {
        input.assert_matches_chain(chain);
        assert_eq!(input.level(), first.level());
        assert_eq!(input.scale(), first.scale());
        assert_eq!(input.basis(), first.basis());
    }

    // Decimation in time: even and odd subsequences.
    let even = packed_radix3_dft_cp_prepared(
        evaluator,
        [inputs[0], inputs[2], inputs[4]],
        &coefficients.radix3,
        chain,
        plan_level0,
    );

    let odd = packed_radix3_dft_cp_prepared(
        evaluator,
        [inputs[1], inputs[3], inputs[5]],
        &coefficients.radix3,
        chain,
        plan_level0,
    );

    let mut output: [Option<RnsCkksCiphertext>; 6] = std::array::from_fn(|_| None);

    for k in 0..3 {
        let weighted_odd = multiply_complex_slots_cp_prepared(
            &odd[k],
            &coefficients.radix2_twiddles[k],
            chain,
            plan_level1,
        );

        let aligned_even = mod_switch_rns_ckks_to_next(&even[k], chain);

        output[k] = Some(evaluator.add(&aligned_even, &weighted_odd));
        output[k + 3] = Some(evaluator.sub(&aligned_even, &weighted_odd));
    }

    output.map(|value| value.expect("DFT6 output missing"))
}

/// Prepared public coefficients for a ten-point forward DFT.
pub struct PreparedDft10Cp {
    radix5: PreparedRadix5Cp,
    radix2_twiddles: [PreparedComplexSlotsCp; 5],
}

impl PreparedDft10Cp {
    pub fn new(
        input: &RnsCkksCiphertext,
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        plan: &RnsNttPlan,
    ) -> Self {
        assert!(
            input.level() + 2 <= chain.max_level(),
            "DFT10 requires two CKKS levels"
        );

        let radix5 = PreparedRadix5Cp::new(input, embedding, chain, plan);

        let level1 = mod_switch_rns_ckks_to_next(input, chain);

        let level1_plan = RnsNttPlan::new(level1.basis().moduli().to_vec(), embedding.degree());

        let slots = embedding.slot_count();

        let radix2_twiddles = std::array::from_fn(|k| {
            let angle = -2.0 * std::f64::consts::PI * k as f64 / 10.0;
            let weight = Complex64::new(angle.cos(), angle.sin());

            PreparedComplexSlotsCp::new(
                &level1,
                &vec![weight; slots],
                embedding,
                chain,
                &level1_plan,
            )
        });

        Self {
            radix5,
            radix2_twiddles,
        }
    }
}

/// Forward DFT10 on ten independently encrypted CKKS slot vectors.
///
/// Stage 1: radix-5 transforms on even and odd inputs.
/// Stage 2: radix-2 DIT combination with public twiddles.
///
/// Outputs are in natural frequency order.
pub fn packed_dft10_cp_prepared(
    evaluator: &RnsCkksEvaluator<'_>,
    inputs: [&RnsCkksCiphertext; 10],
    coefficients: &PreparedDft10Cp,
    chain: &ModulusChain,
    plan_level0: &PreparedRnsNttPlan,
    plan_level1: &PreparedRnsNttPlan,
) -> [RnsCkksCiphertext; 10] {
    let first = inputs[0];

    assert!(
        first.level() + 2 <= chain.max_level(),
        "DFT10 requires two remaining CKKS levels"
    );

    for input in inputs {
        input.assert_matches_chain(chain);
        assert_eq!(input.level(), first.level());
        assert_eq!(input.scale(), first.scale());
        assert_eq!(input.basis(), first.basis());
    }

    let even = packed_radix5_dft_cp_prepared(
        evaluator,
        [inputs[0], inputs[2], inputs[4], inputs[6], inputs[8]],
        &coefficients.radix5,
        chain,
        plan_level0,
    );

    let odd = packed_radix5_dft_cp_prepared(
        evaluator,
        [inputs[1], inputs[3], inputs[5], inputs[7], inputs[9]],
        &coefficients.radix5,
        chain,
        plan_level0,
    );

    let mut output: [Option<RnsCkksCiphertext>; 10] = std::array::from_fn(|_| None);

    for k in 0..5 {
        let weighted_odd = multiply_complex_slots_cp_prepared(
            &odd[k],
            &coefficients.radix2_twiddles[k],
            chain,
            plan_level1,
        );

        let aligned_even = mod_switch_rns_ckks_to_next(&even[k], chain);

        output[k] = Some(evaluator.add(&aligned_even, &weighted_odd));

        output[k + 5] = Some(evaluator.sub(&aligned_even, &weighted_odd));
    }

    output.map(|value| value.expect("DFT10 output missing"))
}

/// Prepared coefficients for the final radix-3 stage of DFT30.
///
/// The interstage twiddles and radix-3 roots are fused into
/// one plaintext multiplication per nontrivial matrix entry.
pub struct PreparedDft30CombineCp {
    weights: Vec<Vec<Vec<Option<PreparedComplexSlotsCp>>>>,
}

impl PreparedDft30CombineCp {
    pub fn new(
        input: &RnsCkksCiphertext,
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        plan: &RnsNttPlan,
    ) -> Self {
        assert!(
            chain.has_next_level(input.level()),
            "DFT30 combination requires one remaining CKKS level"
        );

        assert_eq!(plan.moduli(), input.basis().moduli());
        assert_eq!(plan.degree(), embedding.degree());

        let slots = embedding.slot_count();

        let weights = (0..10)
            .map(|k| {
                (0..3)
                    .map(|q| {
                        (0..3)
                            .map(|j| {
                                if j == 0 {
                                    return None;
                                }

                                let angle =
                                    -2.0 * std::f64::consts::PI * (j * (k + 10 * q)) as f64 / 30.0;

                                let coefficient = Complex64::new(angle.cos(), angle.sin());

                                Some(PreparedComplexSlotsCp::new(
                                    input,
                                    &vec![coefficient; slots],
                                    embedding,
                                    chain,
                                    plan,
                                ))
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        Self { weights }
    }
}

/// Combines three encrypted DFT10 branches into natural-order
/// DFT30 outputs. Consumes exactly one CKKS level.
///
/// `branches[j][k]` is the kth output of the jth DFT10.
pub fn packed_dft30_combine_cp_prepared(
    evaluator: &RnsCkksEvaluator<'_>,
    branches: [&[RnsCkksCiphertext; 10]; 3],
    coefficients: &PreparedDft30CombineCp,
    chain: &ModulusChain,
    plan: &PreparedRnsNttPlan,
) -> [RnsCkksCiphertext; 30] {
    let first = &branches[0][0];

    first.assert_matches_chain(chain);
    assert!(
        chain.has_next_level(first.level()),
        "DFT30 combination requires one remaining CKKS level"
    );

    for branch in branches {
        for input in branch {
            input.assert_matches_chain(chain);
            assert_eq!(input.level(), first.level());
            assert_eq!(input.scale(), first.scale());
            assert_eq!(input.basis(), first.basis());
        }
    }

    std::array::from_fn(|frequency| {
        let k = frequency % 10;
        let q = frequency / 10;

        let mut sum = mod_switch_rns_ckks_to_next(&branches[0][k], chain);

        for (j, branch) in branches.iter().enumerate().skip(1) {
            let coefficient = coefficients.weights[k][q][j]
                .as_ref()
                .expect("DFT30 coefficient missing");

            let weighted = multiply_complex_slots_cp_prepared(&branch[k], coefficient, chain, plan);

            sum = evaluator.add(&sum, &weighted);
        }

        sum
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::application_support::ckks::encode_rns;
    use crate::ckks::{research_profile_4096, CkksChainState, RnsCkksEvaluationKeys};
    use crate::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng};
    use crate::ring::{
        centered_representative_big, composite_modulus_big, reconstruct_coefficients_big,
    };
    use crate::rlwe::ErrorDistribution;

    use num_traits::ToPrimitive;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn r16_fft15_encrypted_two_level_numerical() {
        let start = std::time::Instant::now();

        let profile = research_profile_4096();
        let degree = profile.degree();
        let scale = profile.initial_scale();
        let chain = profile.modulus_chain();
        let basis0 = chain.top().clone();

        let embedding = CkksCanonicalEmbedding::new(degree);
        let slots = embedding.slot_count();

        let plan0 = RnsNttPlan::new(basis0.moduli().to_vec(), degree);
        let prepared0 = PreparedRnsNttPlan::new(&plan0);

        let basis1 = chain.level(1);
        let plan1 = RnsNttPlan::new(basis1.moduli().to_vec(), degree);
        let prepared1 = PreparedRnsNttPlan::new(&plan1);

        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 0,
            })
            .collect();

        let mut rng = ChaCha20Rng::seed_from_u64(0x164F_1510);

        // Small bounded values reduce the risk of exhausting
        // the final CKKS modulus during this correctness experiment.
        let inputs: [Vec<Complex64>; 15] = std::array::from_fn(|index| {
            (0..slots)
                .map(|slot| {
                    let x = index as f64;
                    let s = slot as f64;

                    Complex64::new(
                        0.005 * (0.31 * x + 0.013 * s).sin(),
                        0.005 * (0.19 * x - 0.017 * s).cos(),
                    )
                })
                .collect()
        });

        let ciphertexts: [RnsCkksCiphertext; 15] = std::array::from_fn(|index| {
            let plaintext = encode_rns(&inputs[index], &embedding, &basis0, scale);

            let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                &secret,
                &plan0,
                &mut rng,
            );

            RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain)
        });

        let evaluation_keys = RnsCkksEvaluationKeys::new();
        let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

        let radix5_coefficients =
            PreparedRadix5Cp::new(&ciphertexts[0], &embedding, &chain, &plan0);

        let preparation0_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Recursive DIT: the three branches contain indices
        // j, j+3, j+6, j+9, j+12.
        let execution_start = std::time::Instant::now();

        let branches: [[RnsCkksCiphertext; 5]; 3] = std::array::from_fn(|j| {
            packed_radix5_dft_cp_prepared(
                &evaluator,
                [
                    &ciphertexts[j],
                    &ciphertexts[j + 3],
                    &ciphertexts[j + 6],
                    &ciphertexts[j + 9],
                    &ciphertexts[j + 12],
                ],
                &radix5_coefficients,
                &chain,
                &prepared0,
            )
        });

        let stage1_ms = execution_start.elapsed().as_secs_f64() * 1000.0;

        for branch in &branches {
            for ciphertext in branch {
                assert_eq!(ciphertext.level(), 1);
                assert_eq!(ciphertext.scale(), scale);
            }
        }

        // All branch outputs have the same level, basis, and scale.
        // Prepare the fused twiddle/radix-3 coefficients at level 1.
        let preparation1_start = std::time::Instant::now();

        let fused_coefficients =
            PreparedFft15TwiddledRadix3Cp::new(&branches[0][0], &embedding, &chain, &plan1);

        let preparation1_ms = preparation1_start.elapsed().as_secs_f64() * 1000.0;

        let stage2_start = std::time::Instant::now();

        let groups: [[RnsCkksCiphertext; 3]; 5] = std::array::from_fn(|k| {
            packed_fft15_twiddled_radix3_cp_prepared(
                &evaluator,
                [&branches[0][k], &branches[1][k], &branches[2][k]],
                k,
                &fused_coefficients,
                &chain,
                &prepared1,
            )
        });

        // R23.3C: all-branch wrapper versus the established
        // individually computed prepared radix-3 outputs.
        let all = packed_fft15_all_branches_cp_prepared(
            &evaluator,
            &branches,
            &fused_coefficients,
            &chain,
            &prepared1,
        );

        for (frequency, actual) in all.iter().enumerate() {
            let k = frequency % 5;
            let q = frequency / 5;
            let expected = &groups[k][q];

            assert_eq!(actual.level(), expected.level());
            assert_eq!(actual.scale(), expected.scale());
            assert_eq!(actual.basis(), expected.basis());
            assert_eq!(
                format!("{:?}", actual.rlwe()),
                format!("{:?}", expected.rlwe()),
                "R23.3C RLWE mismatch at FFT15 branch {frequency}"
            );
        }

        println!("R23_3C_ALL_BRANCHES_EXACT_STATUS=PASS");

        let stage2_ms = stage2_start.elapsed().as_secs_f64() * 1000.0;

        // Natural-order FFT15 output: X[k + 5*q] = groups[k][q].
        let validation_start = std::time::Instant::now();

        let mut error_squared = 0.0_f64;
        let mut reference_squared = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for frequency in 0..15 {
            let k = frequency % 5;
            let q = frequency / 5;
            let output = &groups[k][q];

            assert_eq!(output.level(), 2);
            assert_eq!(output.scale(), scale);

            let result_plan = RnsNttPlan::new(output.basis().moduli().to_vec(), degree);

            let decrypted = decrypt_rns_raw_with_ntt(output.rlwe(), &secret, &result_plan);

            let modulus = composite_modulus_big(decrypted.basis());

            let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
                .into_iter()
                .map(|value| {
                    centered_representative_big(&value, &modulus)
                        .to_f64()
                        .expect("CKKS coefficient must fit f64")
                        / output.scale()
                })
                .collect();

            let actual = embedding.coefficients_to_slots(&coefficients);

            let mut frequency_max_abs = 0.0_f64;

            for (slot, actual_value) in actual.iter().enumerate().take(slots) {
                let expected = (0..15)
                    .map(|index| {
                        let angle = -2.0 * std::f64::consts::PI * (frequency * index) as f64 / 15.0;

                        let weight = Complex64::new(angle.cos(), angle.sin());

                        inputs[index][slot] * weight
                    })
                    .sum::<Complex64>();

                let error = ((*actual_value) - expected).norm();

                error_squared += error * error;
                reference_squared += expected.norm_sqr();
                max_abs = max_abs.max(error);
                frequency_max_abs = frequency_max_abs.max(error);
            }

            println!(
                "R16_FFT15_FREQUENCY={} MAX_ABS={:.12e}",
                frequency, frequency_max_abs
            );
        }

        let validation_ms = validation_start.elapsed().as_secs_f64() * 1000.0;

        assert!(reference_squared > 0.0);

        let rel_l2 = (error_squared / reference_squared).sqrt();

        let preparation_ms = preparation0_ms + preparation1_ms;
        let execution_ms = stage1_ms + stage2_ms;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!("R16_FFT15_ENCRYPTED_INPUTS=15");
        println!("R16_FFT15_ENCRYPTED_OUTPUTS=15");
        println!("R16_FFT15_ENCRYPTED_LEVELS_CONSUMED=2");
        println!("R16_FFT15_ENCRYPTED_SLOTS_PER_OUTPUT={slots}");
        println!("R16_FFT15_ENCRYPTED_REL_L2={rel_l2:.12e}");
        println!("R16_FFT15_ENCRYPTED_MAX_ABS={max_abs:.12e}");
        println!("R16_FFT15_PREPARATION_MS={preparation_ms:.3}");
        println!("R16_FFT15_STAGE1_MS={stage1_ms:.3}");
        println!("R16_FFT15_STAGE2_MS={stage2_ms:.3}");
        println!("R16_FFT15_EXECUTION_MS={execution_ms:.3}");
        println!("R16_FFT15_VALIDATION_MS={validation_ms:.3}");
        println!("R16_FFT15_TOTAL_MS={total_ms:.3}");

        let pass = rel_l2 < 1.0e-3 && max_abs < 1.0e-3;

        println!(
            "R16_FFT15_ENCRYPTED_STATUS={}",
            if pass { "PASS" } else { "FAIL" }
        );

        assert!(
            pass,
            "Encrypted FFT15 mismatch: rel_l2={rel_l2:e}, max_abs={max_abs:e}"
        );
    }

    #[test]
    fn r16_radix3_encrypted_numerical() {
        let profile = research_profile_4096();
        let degree = profile.degree();
        let scale = profile.initial_scale();
        let chain = profile.modulus_chain();
        let basis = chain.top().clone();

        let embedding = CkksCanonicalEmbedding::new(degree);
        let slots = embedding.slot_count();

        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
        let prepared_plan = PreparedRnsNttPlan::new(&plan);

        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 0,
            })
            .collect();

        let mut rng = ChaCha20Rng::seed_from_u64(0x164C_0202);

        let inputs: [Vec<Complex64>; 3] = std::array::from_fn(|branch| {
            (0..slots)
                .map(|slot| {
                    let x = (slot % 17) as f64 / 128.0;
                    let y = (slot % 11) as f64 / 256.0;
                    Complex64::new(0.125 * (branch + 1) as f64 + x, -0.0625 * branch as f64 + y)
                })
                .collect()
        });

        let ciphertexts: [RnsCkksCiphertext; 3] = std::array::from_fn(|branch| {
            let plaintext = encode_rns(&inputs[branch], &embedding, &basis, scale);

            let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                &secret,
                &plan,
                &mut rng,
            );

            RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain)
        });

        let evaluation_keys = RnsCkksEvaluationKeys::new();
        let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

        let coefficients = PreparedRadix3Cp::new(&ciphertexts[0], &embedding, &chain, &plan);

        let outputs = packed_radix3_dft_cp_prepared(
            &evaluator,
            [&ciphertexts[0], &ciphertexts[1], &ciphertexts[2]],
            &coefficients,
            &chain,
            &prepared_plan,
        );

        let omega = Complex64::new(-0.5, -(3.0_f64).sqrt() / 2.0);

        let mut error_squared = 0.0_f64;
        let mut reference_squared = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for (branch, output) in outputs.iter().enumerate() {
            assert_eq!(output.level(), ciphertexts[0].level() + 1);
            assert_eq!(output.scale(), ciphertexts[0].scale());

            let result_plan = RnsNttPlan::new(output.basis().moduli().to_vec(), degree);

            let decrypted = decrypt_rns_raw_with_ntt(output.rlwe(), &secret, &result_plan);

            let modulus = composite_modulus_big(decrypted.basis());

            let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
                .into_iter()
                .map(|value| {
                    centered_representative_big(&value, &modulus)
                        .to_f64()
                        .expect("CKKS coefficient must fit f64")
                        / output.scale()
                })
                .collect();

            let actual = embedding.coefficients_to_slots(&coefficients);

            for (slot, actual_value) in actual.iter().enumerate().take(slots) {
                let expected = (0..3)
                    .map(|index| inputs[index][slot] * omega.powu((branch * index) as u32))
                    .sum::<Complex64>();

                let error = ((*actual_value) - expected).norm();

                error_squared += error * error;
                reference_squared += expected.norm_sqr();
                max_abs = max_abs.max(error);
            }
        }

        let rel_l2 = (error_squared / reference_squared).sqrt();
        let pass = rel_l2 < 1.0e-4 && max_abs < 1.0e-3;

        println!("R16_RADIX3_ENCRYPTED_LEVELS_CONSUMED=1");
        println!("R16_RADIX3_ENCRYPTED_REL_L2={rel_l2:.12e}");
        println!("R16_RADIX3_ENCRYPTED_MAX_ABS={max_abs:.12e}");
        println!(
            "R16_RADIX3_ENCRYPTED_STATUS={}",
            if pass { "PASS" } else { "FAIL" }
        );

        assert!(
            pass,
            "Encrypted radix-3 numerical mismatch: \
             rel_l2={rel_l2:e}, max_abs={max_abs:e}"
        );
    }

    #[test]
    fn r16_radix5_encrypted_numerical() {
        let profile = research_profile_4096();
        let degree = profile.degree();
        let scale = profile.initial_scale();
        let chain = profile.modulus_chain();
        let basis = chain.top().clone();

        let embedding = CkksCanonicalEmbedding::new(degree);
        let slots = embedding.slot_count();

        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
        let prepared_plan = PreparedRnsNttPlan::new(&plan);

        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 0,
            })
            .collect();

        let mut rng = ChaCha20Rng::seed_from_u64(0x164C_0303);

        let inputs: [Vec<Complex64>; 5] = std::array::from_fn(|branch| {
            (0..slots)
                .map(|slot| {
                    let real = (slot % 17) as f64 / 128.0;
                    let imag = (slot % 11) as f64 / 256.0;
                    Complex64::new(
                        0.0625 * (branch + 1) as f64 + real,
                        -0.03125 * branch as f64 + imag,
                    )
                })
                .collect()
        });

        let ciphertexts: [RnsCkksCiphertext; 5] = std::array::from_fn(|branch| {
            let plaintext = encode_rns(&inputs[branch], &embedding, &basis, scale);

            let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                &secret,
                &plan,
                &mut rng,
            );

            RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain)
        });

        let evaluation_keys = RnsCkksEvaluationKeys::new();
        let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

        let coefficients = PreparedRadix5Cp::new(&ciphertexts[0], &embedding, &chain, &plan);

        let outputs = packed_radix5_dft_cp_prepared(
            &evaluator,
            [
                &ciphertexts[0],
                &ciphertexts[1],
                &ciphertexts[2],
                &ciphertexts[3],
                &ciphertexts[4],
            ],
            &coefficients,
            &chain,
            &prepared_plan,
        );

        let omega = Complex64::new(
            (2.0 * std::f64::consts::PI / 5.0).cos(),
            -(2.0 * std::f64::consts::PI / 5.0).sin(),
        );

        let mut error_squared = 0.0_f64;
        let mut reference_squared = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for (branch, output) in outputs.iter().enumerate() {
            assert_eq!(output.level(), ciphertexts[0].level() + 1);
            assert_eq!(output.scale(), ciphertexts[0].scale());

            let result_plan = RnsNttPlan::new(output.basis().moduli().to_vec(), degree);

            let decrypted = decrypt_rns_raw_with_ntt(output.rlwe(), &secret, &result_plan);

            let modulus = composite_modulus_big(decrypted.basis());

            let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
                .into_iter()
                .map(|value| {
                    centered_representative_big(&value, &modulus)
                        .to_f64()
                        .expect("CKKS coefficient must fit f64")
                        / output.scale()
                })
                .collect();

            let actual = embedding.coefficients_to_slots(&coefficients);

            for (slot, actual_value) in actual.iter().enumerate().take(slots) {
                let expected = (0..5)
                    .map(|index| inputs[index][slot] * omega.powu((branch * index) as u32))
                    .sum::<Complex64>();

                let error = ((*actual_value) - expected).norm();

                error_squared += error * error;
                reference_squared += expected.norm_sqr();
                max_abs = max_abs.max(error);
            }
        }

        let rel_l2 = (error_squared / reference_squared).sqrt();
        let pass = rel_l2 < 1.0e-4 && max_abs < 1.0e-3;

        println!("R16_RADIX5_ENCRYPTED_LEVELS_CONSUMED=1");
        println!("R16_RADIX5_ENCRYPTED_REL_L2={rel_l2:.12e}");
        println!("R16_RADIX5_ENCRYPTED_MAX_ABS={max_abs:.12e}");
        println!(
            "R16_RADIX5_ENCRYPTED_STATUS={}",
            if pass { "PASS" } else { "FAIL" }
        );

        assert!(
            pass,
            "Encrypted radix-5 mismatch: \
             rel_l2={rel_l2:e}, max_abs={max_abs:e}"
        );
    }

    #[test]
    fn r16_dft6_encrypted_numerical() {
        let profile = research_profile_4096();
        let degree = profile.degree();
        let scale = profile.initial_scale();
        let chain = profile.modulus_chain();
        let basis = chain.top().clone();

        let embedding = CkksCanonicalEmbedding::new(degree);
        let slots = embedding.slot_count();

        let plan0 = RnsNttPlan::new(basis.moduli().to_vec(), degree);
        let prepared0 = PreparedRnsNttPlan::new(&plan0);

        let basis1 = chain.level(1);
        let plan1 = RnsNttPlan::new(basis1.moduli().to_vec(), degree);
        let prepared1 = PreparedRnsNttPlan::new(&plan1);

        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 0,
            })
            .collect();

        let mut rng = ChaCha20Rng::seed_from_u64(0x164D_0601);

        let inputs: [Vec<Complex64>; 6] = std::array::from_fn(|branch| {
            (0..slots)
                .map(|slot| {
                    (Complex64::new(
                        0.03125 * (branch + 1) as f64 + (slot % 13) as f64 / 256.0,
                        -0.015625 * branch as f64 + (slot % 7) as f64 / 512.0,
                    )) * 0.25
                })
                .collect()
        });

        let ciphertexts: [RnsCkksCiphertext; 6] = std::array::from_fn(|branch| {
            let plaintext = encode_rns(&inputs[branch], &embedding, &basis, scale);

            let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                &secret,
                &plan0,
                &mut rng,
            );

            RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain)
        });

        let evaluation_keys = RnsCkksEvaluationKeys::new();
        let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

        let coefficients = PreparedDft6Cp::new(&ciphertexts[0], &embedding, &chain, &plan0);

        let outputs = packed_dft6_cp_prepared(
            &evaluator,
            [
                &ciphertexts[0],
                &ciphertexts[1],
                &ciphertexts[2],
                &ciphertexts[3],
                &ciphertexts[4],
                &ciphertexts[5],
            ],
            &coefficients,
            &chain,
            &prepared0,
            &prepared1,
        );

        let mut error_squared = 0.0_f64;
        let mut reference_squared = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for (frequency, output) in outputs.iter().enumerate() {
            let mut frequency_error_squared = 0.0_f64;
            let mut frequency_reference_squared = 0.0_f64;
            let mut frequency_max_abs = 0.0_f64;

            assert_eq!(output.level(), 2);
            assert_eq!(output.scale(), scale);

            let result_plan = RnsNttPlan::new(output.basis().moduli().to_vec(), degree);

            let decrypted = decrypt_rns_raw_with_ntt(output.rlwe(), &secret, &result_plan);

            let modulus = composite_modulus_big(decrypted.basis());

            let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
                .into_iter()
                .map(|value| {
                    centered_representative_big(&value, &modulus)
                        .to_f64()
                        .expect("CKKS coefficient must fit f64")
                        / output.scale()
                })
                .collect();

            let actual = embedding.coefficients_to_slots(&coefficients);

            for (slot, actual_value) in actual.iter().enumerate().take(slots) {
                let expected = (0..6)
                    .map(|index| {
                        let angle = -2.0 * std::f64::consts::PI * (frequency * index) as f64 / 6.0;

                        inputs[index][slot] * Complex64::new(angle.cos(), angle.sin())
                    })
                    .sum::<Complex64>();

                let error = ((*actual_value) - expected).norm();

                error_squared += error * error;
                reference_squared += expected.norm_sqr();
                max_abs = max_abs.max(error);

                frequency_error_squared += error * error;
                frequency_reference_squared += expected.norm_sqr();
                frequency_max_abs = frequency_max_abs.max(error);
            }

            let frequency_rel_l2 = (frequency_error_squared / frequency_reference_squared).sqrt();

            println!("R16_DFT6_FREQUENCY_{frequency}_REL_L2={frequency_rel_l2:.12e}");
            println!("R16_DFT6_FREQUENCY_{frequency}_MAX_ABS={frequency_max_abs:.12e}");
        }

        let rel_l2 = (error_squared / reference_squared).sqrt();

        let pass = rel_l2 < 1.0e-4 && max_abs < 1.0e-3;

        println!("R16_DFT6_ENCRYPTED_LEVELS_CONSUMED=2");
        println!("R16_DFT6_ENCRYPTED_REL_L2={rel_l2:.12e}");
        println!("R16_DFT6_ENCRYPTED_MAX_ABS={max_abs:.12e}");
        println!(
            "R16_DFT6_ENCRYPTED_STATUS={}",
            if pass { "PASS" } else { "FAIL" }
        );

        assert!(
            pass,
            "Encrypted DFT6 mismatch: \
             rel_l2={rel_l2:e}, max_abs={max_abs:e}"
        );
    }

    #[test]
    fn r16_dft10_encrypted_numerical() {
        let profile = research_profile_4096();
        let degree = profile.degree();
        let scale = profile.initial_scale();
        let chain = profile.modulus_chain();
        let basis = chain.top().clone();

        let embedding = CkksCanonicalEmbedding::new(degree);
        let slots = embedding.slot_count();

        let plan0 = RnsNttPlan::new(basis.moduli().to_vec(), degree);
        let prepared0 = PreparedRnsNttPlan::new(&plan0);

        let basis1 = chain.level(1);
        let plan1 = RnsNttPlan::new(basis1.moduli().to_vec(), degree);
        let prepared1 = PreparedRnsNttPlan::new(&plan1);

        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 0,
            })
            .collect();

        let mut rng = ChaCha20Rng::seed_from_u64(0x164D_0601);

        let inputs: [Vec<Complex64>; 10] = std::array::from_fn(|branch| {
            (0..slots)
                .map(|slot| {
                    (Complex64::new(
                        0.03125 * (branch + 1) as f64 + (slot % 13) as f64 / 256.0,
                        -0.015625 * branch as f64 + (slot % 7) as f64 / 512.0,
                    )) * 0.10
                })
                .collect()
        });

        let ciphertexts: [RnsCkksCiphertext; 10] = std::array::from_fn(|branch| {
            let plaintext = encode_rns(&inputs[branch], &embedding, &basis, scale);

            let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                &secret,
                &plan0,
                &mut rng,
            );

            RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain)
        });

        let evaluation_keys = RnsCkksEvaluationKeys::new();
        let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

        let coefficients = PreparedDft10Cp::new(&ciphertexts[0], &embedding, &chain, &plan0);

        let outputs = packed_dft10_cp_prepared(
            &evaluator,
            [
                &ciphertexts[0],
                &ciphertexts[1],
                &ciphertexts[2],
                &ciphertexts[3],
                &ciphertexts[4],
                &ciphertexts[5],
                &ciphertexts[6],
                &ciphertexts[7],
                &ciphertexts[8],
                &ciphertexts[9],
            ],
            &coefficients,
            &chain,
            &prepared0,
            &prepared1,
        );

        let mut error_squared = 0.0_f64;
        let mut reference_squared = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for (frequency, output) in outputs.iter().enumerate() {
            let mut frequency_error_squared = 0.0_f64;
            let mut frequency_reference_squared = 0.0_f64;
            let mut frequency_max_abs = 0.0_f64;

            assert_eq!(output.level(), 2);
            assert_eq!(output.scale(), scale);

            let result_plan = RnsNttPlan::new(output.basis().moduli().to_vec(), degree);

            let decrypted = decrypt_rns_raw_with_ntt(output.rlwe(), &secret, &result_plan);

            let modulus = composite_modulus_big(decrypted.basis());

            let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
                .into_iter()
                .map(|value| {
                    centered_representative_big(&value, &modulus)
                        .to_f64()
                        .expect("CKKS coefficient must fit f64")
                        / output.scale()
                })
                .collect();

            let actual = embedding.coefficients_to_slots(&coefficients);

            for (slot, actual_value) in actual.iter().enumerate().take(slots) {
                let expected = (0..10)
                    .map(|index| {
                        let angle = -2.0 * std::f64::consts::PI * (frequency * index) as f64 / 10.0;

                        inputs[index][slot] * Complex64::new(angle.cos(), angle.sin())
                    })
                    .sum::<Complex64>();

                let error = ((*actual_value) - expected).norm();

                error_squared += error * error;
                reference_squared += expected.norm_sqr();
                max_abs = max_abs.max(error);

                frequency_error_squared += error * error;
                frequency_reference_squared += expected.norm_sqr();
                frequency_max_abs = frequency_max_abs.max(error);
            }

            let frequency_rel_l2 = (frequency_error_squared / frequency_reference_squared).sqrt();

            println!("R16_DFT10_FREQUENCY_{frequency}_REL_L2={frequency_rel_l2:.12e}");
            println!("R16_DFT10_FREQUENCY_{frequency}_MAX_ABS={frequency_max_abs:.12e}");
        }

        let rel_l2 = (error_squared / reference_squared).sqrt();

        let pass = rel_l2 < 1.0e-4 && max_abs < 1.0e-3;

        println!("R16_DFT10_ENCRYPTED_LEVELS_CONSUMED=2");
        println!("R16_DFT10_ENCRYPTED_REL_L2={rel_l2:.12e}");
        println!("R16_DFT10_ENCRYPTED_MAX_ABS={max_abs:.12e}");
        println!(
            "R16_DFT10_ENCRYPTED_STATUS={}",
            if pass { "PASS" } else { "FAIL" }
        );

        assert!(
            pass,
            "Encrypted DFT10 mismatch: \
             rel_l2={rel_l2:e}, max_abs={max_abs:e}"
        );
    }

    #[test]
    fn r16_dft30_composition_oracle() {
        use std::f64::consts::PI;

        let input: [Complex64; 30] = std::array::from_fn(|n| {
            Complex64::new(
                ((n * 7 + 3) % 31) as f64 / 128.0,
                ((n * 11 + 5) % 29) as f64 / 256.0,
            )
        });

        let omega = |n: usize, exponent: usize| {
            let angle = -2.0 * PI * exponent as f64 / n as f64;
            Complex64::new(angle.cos(), angle.sin())
        };

        let branches: [[Complex64; 10]; 3] = std::array::from_fn(|j| {
            std::array::from_fn(|k| (0..10).map(|m| input[j + 3 * m] * omega(10, k * m)).sum())
        });

        let composed: [Complex64; 30] = std::array::from_fn(|frequency| {
            let k = frequency % 10;
            let q = frequency / 10;

            (0..3)
                .map(|j| branches[j][k] * omega(30, j * (k + 10 * q)))
                .sum()
        });

        let reference: [Complex64; 30] = std::array::from_fn(|frequency| {
            (0..30).map(|n| input[n] * omega(30, frequency * n)).sum()
        });

        let mut error_squared = 0.0_f64;
        let mut reference_squared = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for frequency in 0..30 {
            let error = (composed[frequency] - reference[frequency]).norm();
            error_squared += error * error;
            reference_squared += reference[frequency].norm_sqr();
            max_abs = max_abs.max(error);
        }

        let rel_l2 = (error_squared / reference_squared).sqrt();

        println!("R16_DFT30_COMPOSITION_REL_L2={rel_l2:.12e}");
        println!("R16_DFT30_COMPOSITION_MAX_ABS={max_abs:.12e}");

        assert!(
            rel_l2 < 1.0e-12,
            "DFT30 composition oracle mismatch: {rel_l2:e}"
        );
    }

    #[test]
    fn r16_dft30_encrypted_numerical() {
        use crate::ckks::research_profile_8192;
        use std::time::Instant;

        let started = Instant::now();
        let profile = research_profile_8192();
        let degree = profile.degree();
        let scale = profile.initial_scale();
        let chain = profile.modulus_chain();
        let basis = chain.top().clone();

        assert!(chain.max_level() >= 3);

        let embedding = CkksCanonicalEmbedding::new(degree);
        let slots = embedding.slot_count();

        let plans: Vec<RnsNttPlan> = (0..=3)
            .map(|level| RnsNttPlan::new(chain.level(level).moduli().to_vec(), degree))
            .collect();

        let prepared: Vec<PreparedRnsNttPlan> = plans.iter().map(PreparedRnsNttPlan::new).collect();

        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 0,
            })
            .collect();

        let mut rng = ChaCha20Rng::seed_from_u64(0x164D_3001);

        let inputs: [Vec<Complex64>; 30] = std::array::from_fn(|branch| {
            (0..slots)
                .map(|slot| {
                    let real = ((branch * 7 + slot * 3) % 31) as f64 - 15.0;
                    let imag = ((branch * 11 + slot * 5) % 29) as f64 - 14.0;

                    Complex64::new(real, imag) * 0.0001
                })
                .collect()
        });

        let ciphertexts: [RnsCkksCiphertext; 30] = std::array::from_fn(|branch| {
            let plaintext = encode_rns(&inputs[branch], &embedding, &basis, scale);

            let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                &secret,
                &plans[0],
                &mut rng,
            );

            RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain)
        });

        let evaluation_keys = RnsCkksEvaluationKeys::new();
        let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

        let dft10_coefficients =
            PreparedDft10Cp::new(&ciphertexts[0], &embedding, &chain, &plans[0]);

        let branches: [[RnsCkksCiphertext; 10]; 3] = std::array::from_fn(|j| {
            packed_dft10_cp_prepared(
                &evaluator,
                std::array::from_fn(|m| &ciphertexts[j + 3 * m]),
                &dft10_coefficients,
                &chain,
                &prepared[0],
                &prepared[1],
            )
        });

        let combine_coefficients =
            PreparedDft30CombineCp::new(&branches[0][0], &embedding, &chain, &plans[2]);

        let outputs = packed_dft30_combine_cp_prepared(
            &evaluator,
            [&branches[0], &branches[1], &branches[2]],
            &combine_coefficients,
            &chain,
            &prepared[2],
        );

        let mut error_squared = 0.0_f64;
        let mut reference_squared = 0.0_f64;
        let mut max_abs = 0.0_f64;
        let mut max_expected_abs = 0.0_f64;
        let mut max_centered_coefficient_abs = 0.0_f64;

        for (frequency, output) in outputs.iter().enumerate() {
            assert_eq!(output.level(), 3);
            assert_eq!(output.scale(), scale);

            let decrypted = decrypt_rns_raw_with_ntt(output.rlwe(), &secret, &plans[3]);

            let modulus = composite_modulus_big(decrypted.basis());

            let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
                .into_iter()
                .map(|value| {
                    let centered = centered_representative_big(&value, &modulus)
                        .to_f64()
                        .expect("CKKS coefficient must fit f64")
                        / output.scale();

                    max_centered_coefficient_abs = max_centered_coefficient_abs.max(centered.abs());

                    centered
                })
                .collect();

            let actual = embedding.coefficients_to_slots(&coefficients);

            let mut frequency_error_squared = 0.0_f64;
            let mut frequency_reference_squared = 0.0_f64;
            let mut frequency_max_abs = 0.0_f64;

            for (slot, actual_value) in actual.iter().enumerate().take(slots) {
                let expected: Complex64 = (0..30)
                    .map(|index| {
                        let angle = -2.0 * std::f64::consts::PI * (frequency * index) as f64 / 30.0;

                        inputs[index][slot] * Complex64::new(angle.cos(), angle.sin())
                    })
                    .sum();

                let error = ((*actual_value) - expected).norm();

                max_expected_abs = max_expected_abs.max(expected.norm());

                error_squared += error * error;
                reference_squared += expected.norm_sqr();
                max_abs = max_abs.max(error);

                frequency_error_squared += error * error;
                frequency_reference_squared += expected.norm_sqr();
                frequency_max_abs = frequency_max_abs.max(error);
            }

            let frequency_rel_l2 = (frequency_error_squared / frequency_reference_squared).sqrt();

            println!("R16_DFT30_FREQUENCY_{frequency}_REL_L2={frequency_rel_l2:.12e}");
            println!("R16_DFT30_FREQUENCY_{frequency}_MAX_ABS={frequency_max_abs:.12e}");
        }

        let rel_l2 = (error_squared / reference_squared).sqrt();

        let pass = rel_l2 < 1.0e-4 && max_abs < 1.0e-3;

        println!("R16_DFT30_ENCRYPTED_LEVELS_CONSUMED=3");
        println!("R16_DFT30_ENCRYPTED_REL_L2={rel_l2:.12e}");
        println!("R16_DFT30_ENCRYPTED_MAX_ABS={max_abs:.12e}");
        println!("R16_DFT30_MAX_EXPECTED_SLOT_ABS={max_expected_abs:.12e}");
        println!("R16_DFT30_MAX_CENTERED_COEFFICIENT_ABS={max_centered_coefficient_abs:.12e}");
        println!(
            "R16_DFT30_ELAPSED_SECONDS={:.6}",
            started.elapsed().as_secs_f64()
        );
        println!(
            "R16_DFT30_ENCRYPTED_STATUS={}",
            if pass { "PASS" } else { "FAIL" }
        );

        assert!(
            pass,
            "Encrypted DFT30 mismatch: rel_l2={rel_l2:e}, max_abs={max_abs:e}"
        );
    }
}
