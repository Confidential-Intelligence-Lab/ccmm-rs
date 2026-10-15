// R16.5.16c: actual ciphertext-continuous selected-spectrum UHD FFT2.
// Research-only; one vertical DIF branch and one horizontal branch.
// R16.5.13c: all 15 real ciphertext FFT15 -> fused twiddle FFT256 branches.
// Full 3840-frequency horizontal spectrum for 16 independent physical rows.
// Research-only parameters and deterministic test keys; no intermediate decryption.
use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next,
    research_profile_32768_for_levels,
    rotate_left_bounded_rns_ckks_with_prepared_execution as rotate_left_for_extraction,
    rotate_left_hybrid_rns_ckks_with_prepared_ntt, rotate_right_hybrid_rns_ckks_with_prepared_ntt,
    rotation_exponent_left, rotation_exponent_right, BoundedRnsGaloisKey, CkksCanonicalEmbedding,
    CkksChainState, PreparedBoundedRnsGaloisKey, PreparedHybridRnsGaloisKey,
    PreparedRnsPlaintextNtt, RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::level1::{multiply_complex_slots_cp_prepared, PreparedComplexSlotsCp};
use ccmm_rs::eblas::mixed_radix_ckks::{
    packed_fft15_twiddled_radix3_cp_prepared, packed_radix5_dft_cp_prepared,
    PreparedFft15TwiddledRadix3Cp, PreparedRadix5Cp,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    BoundedRnsKeygenConfig,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, Modulus,
    ModulusBasis, ModulusChain, PreparedRnsNttPlan, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::time::Instant;

const SIGMA: f64 = 3.19;
type Diagonals = BTreeMap<isize, Vec<Complex64>>;
type RotationKeys = BTreeMap<isize, PreparedHybridRnsGaloisKey>;
fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * exponent as f64 / n as f64)
}
fn reverse_base4_4(mut i: usize) -> usize {
    let mut reversed = 0;
    for _ in 0..4 {
        reversed = reversed * 4 + i % 4;
        i /= 4;
    }
    reversed
}

// Radix-4 DIT: spans 4, 16, 64, 256. Signed offsets prevent FFT256 block crossings.
fn stage_diagonals(slots: usize, span: usize, k_cross: usize) -> Diagonals {
    assert!([4, 16, 64, 256].contains(&span));
    let quarter = span / 4;
    let mut diagonals = Diagonals::new();
    for i in 0..slots {
        let within = i % span;
        let p = within / quarter;
        let j = within % quarter;
        for q in 0..4 {
            let input_within = j + q * quarter;
            let offset = input_within as isize - within as isize;
            let mut coefficient = root(span, q * j) * root(4, p * q);
            if span == 4 {
                // The input is base-4 digit-reversed. Twiddle exponents refer
                // to natural local sample positions, not packed slot indices.
                let packed_input = i as isize + offset;
                assert!(packed_input >= 0 && (packed_input as usize) < slots);
                let natural_local = reverse_base4_4((packed_input as usize) % 256);
                coefficient *= root(3840, k_cross * natural_local);
            }
            diagonals
                .entry(offset)
                .or_insert_with(|| vec![Complex64::new(0.0, 0.0); slots])[i] = coefficient;
        }
    }
    diagonals
}

struct StageContext<'a> {
    keys: &'a RotationKeys,
    evaluator: &'a RnsCkksEvaluator<'a>,
    embedding: &'a CkksCanonicalEmbedding,
    chain: &'a ModulusChain,
    qp_plan: &'a RnsNttPlan,
    prepared: &'a PreparedRnsNttPlan,
}
fn apply_stage(
    input: &RnsCkksCiphertext,
    diagonals: &Diagonals,
    ctx: &StageContext<'_>,
) -> RnsCkksCiphertext {
    let plain_scale = ctx
        .chain
        .dropped_modulus(input.level())
        .expect("rescale level available")
        .value() as f64;
    let mut accumulator: Option<RnsCkksCiphertext> = None;
    for (&offset, diagonal) in diagonals {
        let rotated = if offset > 0 {
            Some(rotate_left_hybrid_rns_ckks_with_prepared_ntt(
                input,
                offset as usize,
                ctx.keys.get(&offset).expect("left rotation key"),
                ctx.chain,
                ctx.qp_plan,
            ))
        } else if offset < 0 {
            Some(rotate_right_hybrid_rns_ckks_with_prepared_ntt(
                input,
                (-offset) as usize,
                ctx.keys.get(&offset).expect("right rotation key"),
                ctx.chain,
                ctx.qp_plan,
            ))
        } else {
            None
        };
        let source = rotated.as_ref().unwrap_or(input);
        let plain = encode_rns(diagonal, ctx.embedding, input.basis(), plain_scale);
        let prepared_plain = PreparedRnsPlaintextNtt::new(&plain, ctx.prepared);
        let term = multiply_plain_rns_ckks_with_prepared_ntt(
            source,
            &prepared_plain,
            plain_scale,
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

fn generate_keys(
    diagonals: &Diagonals,
    basis: &ModulusBasis,
    auxiliary_basis: &ModulusBasis,
    qp_plan: &RnsNttPlan,
    degree: usize,
    secret: &[i8],
    rng: &mut ChaCha20Rng,
) -> RotationKeys {
    diagonals
        .keys()
        .filter(|&&offset| offset != 0)
        .map(|&offset| {
            let exponent = if offset > 0 {
                rotation_exponent_left(degree, offset as usize)
            } else {
                rotation_exponent_right(degree, (-offset) as usize)
            };
            let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(basis.clone(), 8),
                auxiliary_basis.clone(),
                secret,
                exponent,
                ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                2,
                qp_plan,
                rng,
            );
            (offset, key)
        })
        .collect()
}

fn decode(
    ct: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let plan = RnsNttPlan::new(ct.basis().moduli().to_vec(), embedding.degree());
    let decrypted = decrypt_rns_raw_with_ntt(ct.rlwe(), secret, &plan);
    let modulus = composite_modulus_big(decrypted.basis());
    let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
        .into_iter()
        .map(|value| {
            centered_representative_big(&value, &modulus)
                .to_f64()
                .expect("coefficient conversion")
                / ct.scale()
        })
        .collect();
    embedding.coefficients_to_slots(&coefficients)
}

#[allow(unused_imports, dead_code)]
mod vertical {
    // R16.5.15d: stride-256 encrypted FFT16 at CKKS levels 11 -> 13.
    // Research-only, standalone integration test. No intermediate decryption.
    // R16.5.14d: encrypted FFT2160 = FFT135 DIF + fused-twiddle FFT16 DIT.
    // 135 ciphertexts, each carrying 16 local rows in CKKS slots 0..15.
    // Research-only deterministic keys. No intermediate decryption.
    use ccmm_rs::application_support::ckks::encode_rns;
    use ccmm_rs::ckks::{
        mod_switch_rns_ckks_to_next, multiply_plain_rns_ckks_with_prepared_ntt,
        rescale_rns_ckks_to_next, research_profile_32768_for_levels,
        rotate_left_bounded_rns_ckks_with_prepared_execution,
        rotate_right_bounded_rns_ckks_with_prepared_execution, rotation_exponent_left,
        rotation_exponent_right, BoundedRnsGaloisKey, CkksCanonicalEmbedding, CkksChainState,
        PreparedBoundedRnsGaloisKey, PreparedRnsPlaintextNtt, RnsCkksCiphertext,
        RnsCkksEvaluationKeys, RnsCkksEvaluator,
    };
    use ccmm_rs::grafting::{
        decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
        BoundedRnsKeygenConfig,
    };
    use ccmm_rs::ring::{
        centered_representative_big, composite_modulus_big, reconstruct_coefficients_big,
        ModulusBasis, ModulusChain, PreparedRnsNttPlan, RnsNttPlan,
    };
    use ccmm_rs::rlwe::ErrorDistribution;
    use num_complex::Complex64;
    use num_traits::ToPrimitive;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use std::collections::BTreeMap;
    use std::f64::consts::PI;
    use std::time::Instant;

    const SIGMA: f64 = 3.19;
    fn root(n: usize, exponent: usize) -> Complex64 {
        Complex64::from_polar(1.0, -2.0 * PI * (exponent % n) as f64 / n as f64)
    }

    type Diagonals = BTreeMap<isize, Vec<Complex64>>;
    type RotationKeys = BTreeMap<isize, PreparedBoundedRnsGaloisKey>;
    fn reverse_base4_2(i: usize) -> usize {
        (i % 4) * 4 + i / 4
    }

    // Radix-4 DIT over 16 rows, each row separated by 256 CKKS slots.
    // The diagonal masks restrict each rotation to its own 4096-slot tile.
    const ROW_STRIDE: usize = 256;
    const TILE_SLOTS: usize = 4096;
    pub(super) fn stage_diagonals(slots: usize, span: usize, k_cross: usize) -> Diagonals {
        assert!(span == 4 || span == 16);
        assert_eq!(slots % TILE_SLOTS, 0);
        let quarter = span / 4;
        let mut diagonals = Diagonals::new();
        for i in 0..slots {
            let local_row = (i % TILE_SLOTS) / ROW_STRIDE;
            let within = local_row % span;
            let p = within / quarter;
            let j = within % quarter;
            for q in 0..4 {
                let input_row = (local_row / span) * span + j + q * quarter;
                let offset = (input_row as isize - local_row as isize) * ROW_STRIDE as isize;
                let source = i as isize + offset;
                assert!(source >= 0 && source < slots as isize);
                assert_eq!(source as usize / TILE_SLOTS, i / TILE_SLOTS);
                let mut coefficient = root(span, q * j) * root(4, p * q);
                if span == 4 {
                    coefficient *= root(2160, k_cross * reverse_base4_2(input_row));
                }
                diagonals
                    .entry(offset)
                    .or_insert_with(|| vec![Complex64::new(0.0, 0.0); slots])[i] = coefficient;
            }
        }
        diagonals
    }

    pub(super) struct StageContext<'a> {
        pub(super) keys: &'a RotationKeys,
        pub(super) evaluator: &'a RnsCkksEvaluator<'a>,
        pub(super) embedding: &'a CkksCanonicalEmbedding,
        pub(super) chain: &'a ModulusChain,
        pub(super) plan: &'a RnsNttPlan,
        pub(super) prepared: &'a PreparedRnsNttPlan,
    }
    pub(super) fn apply_stage(
        input: &RnsCkksCiphertext,
        diagonals: &Diagonals,
        ctx: &StageContext<'_>,
    ) -> RnsCkksCiphertext {
        let plain_scale = ctx
            .chain
            .dropped_modulus(input.level())
            .expect("rescale level available")
            .value() as f64;
        let mut accumulator: Option<RnsCkksCiphertext> = None;
        for (&offset, diagonal) in diagonals {
            let rotated = if offset > 0 {
                Some(rotate_left_bounded_rns_ckks_with_prepared_execution(
                    input,
                    offset as usize,
                    ctx.keys.get(&offset).expect("left rotation key"),
                    ctx.chain,
                    ctx.plan,
                    ctx.prepared,
                ))
            } else if offset < 0 {
                Some(rotate_right_bounded_rns_ckks_with_prepared_execution(
                    input,
                    (-offset) as usize,
                    ctx.keys.get(&offset).expect("right rotation key"),
                    ctx.chain,
                    ctx.plan,
                    ctx.prepared,
                ))
            } else {
                None
            };
            let source = rotated.as_ref().unwrap_or(input);
            let plain = encode_rns(diagonal, ctx.embedding, input.basis(), plain_scale);
            let prepared_plain = PreparedRnsPlaintextNtt::new(&plain, ctx.prepared);
            let term = multiply_plain_rns_ckks_with_prepared_ntt(
                source,
                &prepared_plain,
                plain_scale,
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

    pub(super) fn generate_keys(
        diagonals: &Diagonals,
        basis: &ModulusBasis,
        plan: &RnsNttPlan,
        degree: usize,
        secret: &[i8],
        rng: &mut ChaCha20Rng,
    ) -> RotationKeys {
        diagonals
            .keys()
            .filter(|&&offset| offset != 0)
            .map(|&offset| {
                let exponent = if offset > 0 {
                    rotation_exponent_left(degree, offset as usize)
                } else {
                    rotation_exponent_right(degree, (-offset) as usize)
                };
                let key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
                    BoundedRnsKeygenConfig {
                        plaintext_modulus: 2,
                        layout: BoundedGadgetLayout::new(basis.clone(), 8),
                        plan,
                    },
                    secret,
                    exponent,
                    ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                    rng,
                );
                (offset, PreparedBoundedRnsGaloisKey::new(&key, plan))
            })
            .collect()
    }

    fn decode(
        ct: &RnsCkksCiphertext,
        secret: &[i8],
        embedding: &CkksCanonicalEmbedding,
    ) -> Vec<Complex64> {
        let plan = RnsNttPlan::new(ct.basis().moduli().to_vec(), embedding.degree());
        let decrypted = decrypt_rns_raw_with_ntt(ct.rlwe(), secret, &plan);
        let modulus = composite_modulus_big(decrypted.basis());
        let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
            .into_iter()
            .map(|value| {
                centered_representative_big(&value, &modulus)
                    .to_f64()
                    .expect("CKKS coefficient must fit f64")
                    / ct.scale()
            })
            .collect();
        embedding.coefficients_to_slots(&coefficients)
    }
}

mod fft135_permutation {
    const RADICES: [usize; 4] = [5, 3, 3, 3];
    pub(super) fn natural_index(index: usize) -> usize {
        let mut remaining = index;
        let mut digits = [0usize; 4];
        for (stage, &radix) in RADICES.iter().rev().enumerate() {
            digits[3 - stage] = remaining % radix;
            remaining /= radix;
        }
        assert_eq!(remaining, 0);
        digits[0] + RADICES[0] * (digits[1] + RADICES[1] * (digits[2] + RADICES[2] * digits[3]))
    }
}

#[test]
#[ignore = "full 3840x2160 ciphertext-continuous FFT2; expensive explicit run"]
fn r16_5_18i3_hybrid_uhd_fft2() {
    const TILE_ROWS: usize = 135;
    const TILES: usize = 15;
    const HALF: usize = 4096;
    const H_BRANCH: usize = 7;
    const V_OUTPUT_INDEX: usize = 0;
    const RADICES: [usize; 4] = [5, 3, 3, 3];
    let started = Instant::now();
    let profile = research_profile_32768_for_levels(13, 3);
    let degree = profile.degree();
    let chain = profile.modulus_chain();
    let scale = profile.initial_scale();
    assert_eq!(degree, 32768);
    assert!(chain.max_level() >= 13);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();
    assert_eq!(slots, 16384);
    let basis0 = chain.top().clone();
    let plan0 = RnsNttPlan::new(basis0.moduli().to_vec(), degree);
    let prepared0 = PreparedRnsNttPlan::new(&plan0);
    let plan1 = RnsNttPlan::new(chain.level(1).moduli().to_vec(), degree);
    let prepared1 = PreparedRnsNttPlan::new(&plan1);
    let plan2 = RnsNttPlan::new(chain.level(2).moduli().to_vec(), degree);
    let prepared2 = PreparedRnsNttPlan::new(&plan2);
    let bases: Vec<ModulusBasis> = (3..7).map(|level| chain.level(level).clone()).collect();
    let plans: Vec<RnsNttPlan> = bases
        .iter()
        .map(|basis| RnsNttPlan::new(basis.moduli().to_vec(), degree))
        .collect();
    let prepared_plans: Vec<PreparedRnsNttPlan> =
        plans.iter().map(PreparedRnsNttPlan::new).collect();
    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();
    let mut rng = ChaCha20Rng::seed_from_u64(0x1651_16C0);
    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);
    let low_mask: Vec<Complex64> = (0..slots)
        .map(|s| Complex64::new(if s < HALF { 1.0 } else { 0.0 }, 0.0))
        .collect();
    let key_started = Instant::now();
    let left_key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout: BoundedGadgetLayout::new(basis0.clone(), 8),
            plan: &plan0,
        },
        &secret,
        rotation_exponent_left(degree, HALF),
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &mut rng,
    );
    let prepared_left = PreparedBoundedRnsGaloisKey::new(&left_key, &plan0);
    let spans = [4usize, 16, 64, 256];
    let diagonals: Vec<Diagonals> = spans
        .iter()
        .map(|&span| stage_diagonals(slots, span, H_BRANCH))
        .collect();
    let auxiliary_basis = ModulusBasis::new(vec![
        Modulus::new(1_125_899_908_022_273),
        Modulus::new(1_125_899_908_612_097),
    ]);

    for basis in &bases {
        for modulus in auxiliary_basis.moduli() {
            assert!(
                !basis.moduli().contains(modulus),
                "hybrid Q/P modulus overlap"
            );
        }
    }

    let qp_plans: Vec<RnsNttPlan> = bases
        .iter()
        .map(|basis| {
            let mut moduli = basis.moduli().to_vec();
            moduli.extend_from_slice(auxiliary_basis.moduli());
            RnsNttPlan::new(moduli, degree)
        })
        .collect();

    let hybrid_key_started = Instant::now();
    let rotation_keys: Vec<RotationKeys> = (0..4)
        .map(|level| {
            generate_keys(
                &diagonals[level],
                &bases[level],
                &auxiliary_basis,
                &qp_plans[level],
                degree,
                &secret,
                &mut rng,
            )
        })
        .collect();
    println!(
        "R16_5_18I3_HYBRID_KEYGEN_MS={:.3}",
        hybrid_key_started.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "R16_16C_SHARED_HORIZONTAL_KEYGEN_MS={:.3}",
        key_started.elapsed().as_secs_f64() * 1000.0
    );
    // Keep one branch ciphertext per physical tile row. Each input tile row
    // consists of 15 independent 4096-slot tiles, representing 16x256 samples.
    let mut horizontal: Vec<RnsCkksCiphertext> = Vec::with_capacity(TILE_ROWS);
    for tile_row in 0..TILE_ROWS {
        let row_started = Instant::now();
        let mut physical = Vec::with_capacity(8);
        for pair in 0..8 {
            let mut values = vec![Complex64::new(0.0, 0.0); slots];
            for half in 0..2 {
                let tile = 2 * pair + half;
                if tile >= TILES {
                    continue;
                }
                for packed_row in 0..16 {
                    let natural_row = (packed_row % 4) * 4 + packed_row / 4;
                    for packed_col in 0..256 {
                        let natural_col = reverse_base4_4(packed_col);
                        let y = tile_row * 16 + natural_row;
                        let x = tile * 256 + natural_col;
                        values[half * HALF + packed_row * 256 + packed_col] = sample(y, x);
                    }
                }
            }
            let plaintext = encode_rns(&values, &embedding, &basis0, scale);
            let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                &secret,
                &plan0,
                &mut rng,
            );
            physical.push(RnsCkksCiphertext::new(
                rlwe,
                CkksChainState::top(&chain, scale),
                &chain,
            ));
        }
        let prepared_mask =
            PreparedComplexSlotsCp::new(&physical[0], &low_mask, &embedding, &chain, &plan0);
        let aligned: Vec<RnsCkksCiphertext> = (0..TILES)
            .map(|tile| {
                let source = &physical[tile / 2];
                if tile % 2 == 0 {
                    multiply_complex_slots_cp_prepared(source, &prepared_mask, &chain, &prepared0)
                } else {
                    let shifted = rotate_left_for_extraction(
                        source,
                        HALF,
                        &prepared_left,
                        &chain,
                        &plan0,
                        &prepared0,
                    );
                    multiply_complex_slots_cp_prepared(&shifted, &prepared_mask, &chain, &prepared0)
                }
            })
            .collect();
        drop(physical);
        let radix5 = PreparedRadix5Cp::new(&aligned[0], &embedding, &chain, &plan1);
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
                &chain,
                &prepared1,
            )
        });
        drop(aligned);
        let radix3 =
            PreparedFft15TwiddledRadix3Cp::new(&branches[0][0], &embedding, &chain, &plan2);
        // Compute only the radix-3 group that contains the selected horizontal branch.
        let k = H_BRANCH % 5;
        let groups = packed_fft15_twiddled_radix3_cp_prepared(
            &evaluator,
            [&branches[0][k], &branches[1][k], &branches[2][k]],
            k,
            &radix3,
            &chain,
            &prepared2,
        );
        let mut current = groups[H_BRANCH / 5].clone();
        assert_eq!(current.level(), 3);
        if tile_row == 0 {
            println!(
                "R16_5_18I3_EXTRACTION_FFT15_MS={:.3}",
                row_started.elapsed().as_secs_f64() * 1000.0
            );
        }
        for level in 0..4 {
            let stage_started = Instant::now();
            current = apply_stage(
                &current,
                &diagonals[level],
                &StageContext {
                    keys: &rotation_keys[level],
                    evaluator: &evaluator,
                    embedding: &embedding,
                    chain: &chain,
                    qp_plan: &qp_plans[level],
                    prepared: &prepared_plans[level],
                },
            );
            assert_eq!(current.level(), level + 4);
            if tile_row == 0 {
                println!(
                    "R16_5_18I3_HYBRID_FFT256_STAGE_{}_MS={:.3}",
                    level + 1,
                    stage_started.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        horizontal.push(current);

        if std::env::var_os("R16_5_18I3_HORIZONTAL_VALIDATE").is_some() {
            assert_eq!(tile_row, 0);
            assert_eq!(horizontal.len(), 1);
            let ct = &horizontal[0];
            assert_eq!(ct.level(), 7);
            assert_eq!(ct.basis(), chain.level(7));
            assert!(ct.scale().is_finite());

            let actual = decode(ct, &secret, &embedding);
            let mut expected = vec![Complex64::new(0.0, 0.0); HALF];

            // Independent horizontal Fourier oracle for the three
            // analytically generated UHD input modes. The vertical
            // frequency is a multiple of 135, so its phase within
            // each 16-row tile is a 16-point Fourier phase.
            //
            // Horizontal transform length = 3840, unnormalized.
            // Output layout = 16 packed physical rows x 256 bins.
            const MODES: [(usize, usize, f64); 3] =
                [(3, 2, 1.0e-7), (31, 11, 0.4e-7), (129, 6, 0.7e-7)];

            for packed_row in 0..16 {
                let natural_row = (packed_row % 4) * 4 + packed_row / 4;
                let y = tile_row * 16 + natural_row;

                for &(local_h, local_v, amplitude) in &MODES {
                    let phase =
                        Complex64::from_polar(1.0, 2.0 * PI * (135 * local_v * y) as f64 / 2160.0);
                    expected[packed_row * 256 + local_h] = phase * (amplitude * 3840.0);
                }
            }

            let mut err_sq = 0.0_f64;
            let mut ref_sq = 0.0_f64;
            let mut max_abs = 0.0_f64;

            for (&got, &want) in actual.iter().zip(expected.iter()).take(HALF) {
                let delta = (got - want).norm();
                err_sq += delta * delta;
                ref_sq += want.norm_sqr();
                max_abs = max_abs.max(delta);
            }

            assert!(ref_sq > 0.0);
            let rel_l2 = (err_sq / ref_sq).sqrt();

            println!("R16_5_18I3_HORIZONTAL_ROWS=16");
            println!("R16_5_18I3_HORIZONTAL_COLUMNS=3840");
            println!("R16_5_18I3_HORIZONTAL_BRANCH={H_BRANCH}");
            println!("R16_5_18I3_HORIZONTAL_OUTPUT_BINS={HALF}");
            println!("R16_5_18I3_HORIZONTAL_LEVEL={}", ct.level());
            println!("R16_5_18I3_HORIZONTAL_REL_L2={rel_l2:.12e}");
            println!("R16_5_18I3_HORIZONTAL_MAX_ABS={max_abs:.12e}");
            println!(
                "R16_5_18I3_TOTAL_MS={:.3}",
                started.elapsed().as_secs_f64() * 1000.0
            );

            // Initial diagnostic gate: the previously established
            // full UHD tolerance is 1e-2. We will assess the
            // horizontal-only precision separately after measuring.
            let pass = rel_l2.is_finite() && rel_l2 < 1e-2 && max_abs.is_finite() && max_abs < 1e-2;

            println!(
                "R16_5_18I3_HORIZONTAL_STATUS={}",
                if pass { "PASS" } else { "FAIL" }
            );
            println!("R16_5_18I3_FULL_UHD_VALIDATED=NO");
            assert!(pass, "hybrid UHD horizontal numerical oracle failed");
            return;
        }

        if tile_row % 15 == 14 || tile_row == 0 {
            println!("R16_16C_HORIZONTAL_TILE_ROWS_DONE={}", tile_row + 1);
        }
    }
    assert_eq!(horizontal.len(), TILE_ROWS);
    assert!(horizontal.iter().all(|ct| ct.level() == 7));
    println!("R16_16C_HORIZONTAL_HANDOFF_LEVEL=7");
    println!("R16_16C_HORIZONTAL_CIPHERTEXTS=135");
    // DIF FFT135 on the actual 135 level-7 horizontal outputs.
    let mut ciphertexts = horizontal;
    let mut span = TILE_ROWS;
    for (stage, &radix) in RADICES.iter().enumerate() {
        let stage_started = Instant::now();
        let m = span / radix;
        let level = 7 + stage;
        assert!(ciphertexts.iter().all(|ct| ct.level() == level));
        let plan = RnsNttPlan::new(chain.level(level).moduli().to_vec(), degree);
        let prepared_plan = PreparedRnsNttPlan::new(&plan);
        let mut next: Vec<Option<RnsCkksCiphertext>> = vec![None; TILE_ROWS];
        for base in (0..TILE_ROWS).step_by(span) {
            for j in 0..m {
                for k in 0..radix {
                    let mut output: Option<RnsCkksCiphertext> = None;
                    for q in 0..radix {
                        let input = &ciphertexts[base + j + q * m];
                        let coefficient = root(radix, q * k) * root(span, j * k);
                        let term = if (coefficient - Complex64::new(1.0, 0.0)).norm() < 1e-14 {
                            ccmm_rs::ckks::mod_switch_rns_ckks_to_next(input, &chain)
                        } else {
                            let coefficients = vec![coefficient; slots];
                            let prepared = PreparedComplexSlotsCp::new(
                                input,
                                &coefficients,
                                &embedding,
                                &chain,
                                &plan,
                            );
                            multiply_complex_slots_cp_prepared(
                                input,
                                &prepared,
                                &chain,
                                &prepared_plan,
                            )
                        };
                        output = Some(match output {
                            Some(acc) => evaluator.add(&acc, &term),
                            None => term,
                        });
                    }
                    let destination = base + j + k * m;
                    assert!(next[destination].is_none());
                    next[destination] = output;
                }
            }
        }
        ciphertexts = next
            .into_iter()
            .map(|ct| ct.expect("FFT135 butterfly output"))
            .collect();
        span = m;
        println!("R16_16C_FFT135_STAGE_{}_LEVEL={}", stage + 1, level + 1);
        println!(
            "R16_16C_FFT135_STAGE_{}_MS={:.3}",
            stage + 1,
            stage_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    assert_eq!(span, 1);
    let vertical_cross = fft135_permutation::natural_index(V_OUTPUT_INDEX);
    let mut current = ciphertexts.swap_remove(V_OUTPUT_INDEX);
    drop(ciphertexts);
    assert_eq!(current.level(), 11);
    println!("R16_16C_VERTICAL_CROSS_FREQUENCY={vertical_cross}");
    for (stage, span) in [4usize, 16].into_iter().enumerate() {
        let stage_started = Instant::now();
        let level = 11 + stage;
        let basis = chain.level(level).clone();
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
        let prepared = PreparedRnsNttPlan::new(&plan);
        let diagonals = vertical::stage_diagonals(slots, span, vertical_cross);
        let keys = vertical::generate_keys(&diagonals, &basis, &plan, degree, &secret, &mut rng);
        current = vertical::apply_stage(
            &current,
            &diagonals,
            &vertical::StageContext {
                keys: &keys,
                evaluator: &evaluator,
                embedding: &embedding,
                chain: &chain,
                plan: &plan,
                prepared: &prepared,
            },
        );
        assert_eq!(current.level(), level + 1);
        println!(
            "R16_16C_FFT16_STAGE_{}_MS={:.3}",
            stage + 1,
            stage_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    assert_eq!(current.level(), 13);
    let actual = decode(&current, &secret, &embedding);
    // Independent exact spectral oracle: the input is a superposition of
    // three UHD Fourier basis functions. Orthogonality gives exact expected
    // FFT coefficients, without reusing any FFT implementation under test.
    let mut expected = vec![Complex64::new(0.0, 0.0); HALF];
    for &(local_h, local_v, amplitude) in &[
        (3usize, 2usize, 1.0e-7_f64),
        (31, 11, 0.4e-7),
        (129, 6, 0.7e-7),
    ] {
        expected[local_v * 256 + local_h] = Complex64::new(amplitude * (2160 * 3840) as f64, 0.0);
    }
    let mut err_sq = 0.0;
    let mut ref_sq = 0.0;
    let mut max_abs = 0.0_f64;
    for (&got, &want) in actual.iter().zip(expected.iter()).take(HALF) {
        let delta = (got - want).norm();
        err_sq += delta * delta;
        ref_sq += want.norm_sqr();
        max_abs = max_abs.max(delta);
    }
    let rel_l2 = (err_sq / ref_sq).sqrt();
    println!("R16_16C_UHD_ROWS=2160");
    println!("R16_16C_UHD_COLUMNS=3840");
    println!("R16_16C_HORIZONTAL_BRANCH={H_BRANCH}");
    println!("R16_16C_OUTPUT_BINS={HALF}");
    println!("R16_16C_FINAL_LEVEL={}", current.level());
    println!("R16_16C_INTERMEDIATE_DECRYPTIONS=0");
    println!("R16_16C_REL_L2={rel_l2:.12e}");
    println!("R16_16C_MAX_ABS={max_abs:.12e}");
    println!(
        "R16_16C_TOTAL_MS={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    let pass = rel_l2.is_finite() && rel_l2 < 1e-2 && max_abs.is_finite() && max_abs < 1e-2;
    println!("R16_5_18I3_STATUS={}", if pass { "PASS" } else { "FAIL" });
    assert!(pass, "integrated UHD FFT2 numerical validation failed");
}
fn sample(y: usize, x: usize) -> Complex64 {
    const MODES: [(usize, usize, f64); 3] = [
        (7 + 15 * 3, 135 * 2, 1.0e-7),
        (7 + 15 * 31, 135 * 11, 0.4e-7),
        (7 + 15 * 129, 135 * 6, 0.7e-7),
    ];
    MODES
        .iter()
        .map(|&(kx, ky, a)| {
            Complex64::new(a, 0.0) * root(3840, kx * x).conj() * root(2160, ky * y).conj()
        })
        .sum()
}
