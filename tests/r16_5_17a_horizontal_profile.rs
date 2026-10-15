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
    rotate_left_bounded_rns_ckks_with_prepared_execution,
    rotate_right_bounded_rns_ckks_with_prepared_execution, rotation_exponent_left,
    rotation_exponent_right, BoundedRnsGaloisKey, CkksCanonicalEmbedding, CkksChainState,
    PreparedBoundedRnsGaloisKey, PreparedRnsPlaintextNtt, RnsCkksCiphertext, RnsCkksEvaluationKeys,
    RnsCkksEvaluator,
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
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, ModulusBasis,
    ModulusChain, PreparedRnsNttPlan, RnsNttPlan,
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
type RotationKeys = BTreeMap<isize, PreparedBoundedRnsGaloisKey>;
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
    plan: &'a RnsNttPlan,
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

fn generate_keys(
    diagonals: &Diagonals,
    basis: &ModulusBasis,
    plan: &RnsNttPlan,
    degree: usize,
    secret: &[i8],
    distribution: ErrorDistribution,
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
                distribution,
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

// Plaintext mathematical oracle for the selected horizontal FFT branch.
// No CKKS keys, encryption, or ciphertext evaluation.

#[test]
fn r16_5_17a_plaintext_fft15_oracle() {
    const BRANCH: usize = 7;
    const K: usize = BRANCH % 5;
    const Q: usize = BRANCH / 5;

    for y in [0usize, 1, 9] {
        let mut error_sq = 0.0_f64;
        let mut reference_sq = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for packed_col in 0..256 {
            let natural_col = reverse_base4_4(packed_col);

            let mut radix5 = [Complex64::new(0.0, 0.0); 3];

            for (j, output) in radix5.iter_mut().enumerate() {
                for m in 0..5 {
                    let tile = j + 3 * m;
                    let x = tile * 256 + natural_col;
                    *output += sample(y, x) * root(5, K * m);
                }
            }

            let mut actual = Complex64::new(0.0, 0.0);

            for (j, value) in radix5.iter().enumerate() {
                actual += *value * root(15, j * K) * root(3, j * Q);
            }

            let mut expected = Complex64::new(0.0, 0.0);

            for tile in 0..15 {
                let x = tile * 256 + natural_col;
                expected += sample(y, x) * root(15, BRANCH * tile);
            }

            let error = (actual - expected).norm();
            error_sq += error * error;
            reference_sq += expected.norm_sqr();
            max_abs = max_abs.max(error);
        }

        let rel_l2 = (error_sq / reference_sq).sqrt();

        println!("R16_17_FFT15_ORACLE_ROW_{y}_REL_L2={rel_l2:.12e}");
        println!("R16_17_FFT15_ORACLE_ROW_{y}_MAX_ABS={max_abs:.12e}");

        assert!(
            rel_l2.is_finite() && rel_l2 < 1.0e-10,
            "FFT15 decomposition mismatch at row {y}"
        );
    }

    println!("R16_17_FFT15_ORACLE_STATUS=PASS");
}

#[test]
fn r16_5_17a_plaintext_layout_oracle() {
    const N: usize = 3840;
    const BRANCH: usize = 7;
    const LOCAL: usize = 256;
    const TILES: usize = 15;
    const SLOTS: usize = 4096;

    for y in [0usize, 1, 9] {
        let mut current = vec![Complex64::new(0.0, 0.0); SLOTS];

        // Selected DFT15 branch, followed by the FFT3840 twiddle.
        for (packed_col, output) in current.iter_mut().enumerate().take(LOCAL) {
            let natural_col = reverse_base4_4(packed_col);
            let mut branch_value = Complex64::new(0.0, 0.0);

            for tile in 0..TILES {
                let x = tile * LOCAL + natural_col;
                branch_value += sample(y, x) * root(TILES, BRANCH * tile);
            }

            *output = branch_value;
        }

        // Apply production FFT256 diagonal operators in plaintext.
        for span in [4usize, 16, 64, 256] {
            let diagonals = stage_diagonals(SLOTS, span, BRANCH);
            let mut next = vec![Complex64::new(0.0, 0.0); SLOTS];

            for (&offset, diagonal) in &diagonals {
                for slot in 0..LOCAL {
                    let source = slot as isize + offset;

                    if (0..LOCAL as isize).contains(&source) {
                        next[slot] += diagonal[slot] * current[source as usize];
                    } else {
                        assert!(
                            diagonal[slot].norm() < 1.0e-14,
                            "FFT stage crosses row boundary"
                        );
                    }
                }
            }

            current = next;
        }

        // Independent direct DFT3840 reference.
        let mut error_sq = 0.0_f64;
        let mut reference_sq = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for (local_h, got) in current.iter().enumerate().take(LOCAL) {
            let kx = BRANCH + TILES * local_h;
            let mut expected = Complex64::new(0.0, 0.0);

            for x in 0..N {
                expected += sample(y, x) * root(N, kx * x);
            }

            let error = (*got - expected).norm();
            error_sq += error * error;
            reference_sq += expected.norm_sqr();
            max_abs = max_abs.max(error);
        }

        let rel_l2 = (error_sq / reference_sq).sqrt();

        println!("R16_17_ORACLE_ROW_{y}_REL_L2={rel_l2:.12e}");
        println!("R16_17_ORACLE_ROW_{y}_MAX_ABS={max_abs:.12e}");

        assert!(
            rel_l2.is_finite() && rel_l2 < 1.0e-9,
            "plaintext horizontal FFT layout mismatch at row {y}"
        );
    }

    println!("R16_17_PLAINTEXT_ORACLE_STATUS=PASS");
}

#[test]
#[ignore = "full 3840x2160 ciphertext-continuous FFT2; expensive explicit run"]
fn r16_5_17a_horizontal_profile() {
    const TILE_ROWS: usize = 135;
    const TILES: usize = 15;
    const HALF: usize = 4096;
    const H_BRANCH: usize = 7;
    let started = Instant::now();
    let row_limit: usize = std::env::var("R16_17_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    assert!((1..=135).contains(&row_limit));
    let diagnostic = std::env::var("R16_17_DIAGNOSTICS").ok().as_deref() == Some("1");
    println!("R16_17_ROWS_REQUESTED={row_limit}");
    println!("R16_17_DIAGNOSTICS={diagnostic}");
    let extraction_only = std::env::var("R16_17_EXTRACTION_ONLY").ok().as_deref() == Some("1");
    assert!(
        !extraction_only || (diagnostic && row_limit == 1),
        "extraction-only mode requires diagnostics and exactly one tile row"
    );
    println!("R16_17C_EXTRACTION_ONLY={extraction_only}");
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
    let zero_noise_key = std::env::var("R16_17F_ZERO_NOISE_KEY").ok().as_deref() == Some("1");

    assert!(
        !zero_noise_key || (diagnostic && row_limit == 1),
        "zero-noise keys require an explicit single-row diagnostic"
    );

    println!("R16_17F_ZERO_NOISE_KEY={zero_noise_key}");
    println!(
        "R16_17G_HORIZONTAL_DIAGNOSTIC={}",
        zero_noise_key && !extraction_only
    );
    if zero_noise_key {
        println!("R16_17G_SECURITY_STATUS=INSECURE_DIAGNOSTIC_ONLY");
    }

    let extraction_base_log: u32 = std::env::var("R16_17H_GADGET_BASE_LOG")
        .ok()
        .map(|value| {
            value
                .parse::<u32>()
                .expect("R16_17H_GADGET_BASE_LOG must be an integer")
        })
        .unwrap_or(8);

    assert!(
        matches!(extraction_base_log, 4 | 6 | 8),
        "R16.5.17h supports gadget base logs 4, 6, and 8"
    );
    assert!(
        extraction_base_log == 8 || (diagnostic && extraction_only && row_limit == 1),
        "nondefault gadget bases require a single-row extraction diagnostic"
    );
    println!("R16_17H_GADGET_BASE_LOG={extraction_base_log}");

    let extraction_key_distribution = if zero_noise_key {
        ErrorDistribution::BoundedUniform { bound: 0 }
    } else {
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA }
    };

    let left_key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout: BoundedGadgetLayout::new(basis0.clone(), extraction_base_log),
            plan: &plan0,
        },
        &secret,
        rotation_exponent_left(degree, HALF),
        extraction_key_distribution,
        &mut rng,
    );
    let prepared_left = PreparedBoundedRnsGaloisKey::new(&left_key, &plan0);
    let spans = [4usize, 16, 64, 256];
    let diagonals: Vec<Diagonals> = spans
        .iter()
        .map(|&span| stage_diagonals(slots, span, H_BRANCH))
        .collect();
    let rotation_keys: Vec<RotationKeys> = (0..if extraction_only { 0 } else { 4 })
        .map(|level| {
            generate_keys(
                &diagonals[level],
                &bases[level],
                &plans[level],
                degree,
                &secret,
                extraction_key_distribution,
                &mut rng,
            )
        })
        .collect();
    println!(
        "R16_16C_SHARED_HORIZONTAL_KEYGEN_MS={:.3}",
        key_started.elapsed().as_secs_f64() * 1000.0
    );
    // Keep one branch ciphertext per physical tile row. Each input tile row
    // consists of 15 independent 4096-slot tiles, representing 16x256 samples.
    let mut horizontal: Vec<RnsCkksCiphertext> = Vec::with_capacity(TILE_ROWS);
    let mut input_ms = 0.0;
    let mut diagnostic_ms = 0.0;
    let mut extract_ms = 0.0;
    let mut fft15_ms = 0.0;
    let mut fft256_ms = 0.0;
    for tile_row in 0..row_limit {
        let input_started = Instant::now();
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
        input_ms += input_started.elapsed().as_secs_f64() * 1000.0;

        if diagnostic {
            let diagnostic_started = Instant::now();
            let mut error_sq = 0.0_f64;
            let mut reference_sq = 0.0_f64;
            let mut max_abs = 0.0_f64;

            for (pair, ct) in physical.iter().enumerate() {
                let actual = decode(ct, &secret, &embedding);

                for half in 0..2 {
                    let tile = 2 * pair + half;

                    for packed_row in 0..16 {
                        let natural_row = (packed_row % 4) * 4 + packed_row / 4;
                        let y = tile_row * 16 + natural_row;

                        for packed_col in 0..256 {
                            let slot = half * HALF + packed_row * 256 + packed_col;
                            let expected = if tile < TILES {
                                let natural_col = reverse_base4_4(packed_col);
                                sample(y, tile * 256 + natural_col)
                            } else {
                                Complex64::new(0.0, 0.0)
                            };

                            let error = (actual[slot] - expected).norm();
                            error_sq += error * error;
                            reference_sq += expected.norm_sqr();
                            max_abs = max_abs.max(error);
                        }
                    }
                }
            }

            let rel_l2 = (error_sq / reference_sq).sqrt();
            println!("R16_17B_INPUT_REL_L2={rel_l2:.12e}");
            println!("R16_17B_INPUT_MAX_ABS={max_abs:.12e}");
            diagnostic_ms += diagnostic_started.elapsed().as_secs_f64() * 1000.0;
        }

        let extract_started = Instant::now();
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

                    if diagnostic && tile_row == 0 && tile == 1 {
                        let diagnostic_started = Instant::now();
                        let rotated = decode(&shifted, &secret, &embedding);
                        let original = decode(source, &secret, &embedding);

                        let exponent = rotation_exponent_left(degree, HALF);
                        let raw_rotated =
                            ccmm_rs::ckks::apply_rns_automorphism(source.rlwe(), exponent);

                        let ternary_secret = ccmm_rs::ring::Polynomial::new(
                            ccmm_rs::ring::Modulus::new(3),
                            secret
                                .iter()
                                .map(|&coefficient| match coefficient {
                                    -1 => 2,
                                    0 => 0,
                                    1 => 1,
                                    _ => panic!("secret must be ternary"),
                                })
                                .collect(),
                        );

                        let transformed_secret_poly =
                            ccmm_rs::ckks::apply_automorphism(&ternary_secret, exponent);

                        let transformed_secret: Vec<i8> = transformed_secret_poly
                            .coefficients()
                            .iter()
                            .map(|&coefficient| match coefficient {
                                0 => 0,
                                1 => 1,
                                2 => -1,
                                _ => unreachable!(),
                            })
                            .collect();

                        let raw_ckks =
                            RnsCkksCiphertext::new(raw_rotated, source.state().clone(), &chain);
                        let raw_decoded = decode(&raw_ckks, &transformed_secret, &embedding);

                        let mut raw_error_sq = 0.0_f64;
                        let mut raw_reference_sq = 0.0_f64;
                        let mut raw_max_abs = 0.0_f64;

                        for (slot, &actual) in raw_decoded.iter().enumerate() {
                            let expected = original[(slot + HALF) % slots];
                            let error = (actual - expected).norm();
                            raw_error_sq += error * error;
                            raw_reference_sq += expected.norm_sqr();
                            raw_max_abs = raw_max_abs.max(error);
                        }

                        println!(
                            "R16_17E_RAW_AUTOMORPHISM_REL_L2={:.12e}",
                            (raw_error_sq / raw_reference_sq).sqrt()
                        );
                        println!("R16_17E_RAW_AUTOMORPHISM_MAX_ABS={raw_max_abs:.12e}");

                        let mut expected_error_sq = 0.0_f64;
                        let mut unchanged_error_sq = 0.0_f64;
                        let mut expected_sq = 0.0_f64;
                        let mut max_abs = 0.0_f64;

                        for slot in 0..slots {
                            let expected_slot = (slot + HALF) % slots;
                            let expected = original[expected_slot];
                            let unchanged = original[slot];
                            let actual = rotated[slot];

                            let delta = (actual - expected).norm();
                            expected_error_sq += delta * delta;
                            unchanged_error_sq += (actual - unchanged).norm_sqr();
                            expected_sq += expected.norm_sqr();
                            max_abs = max_abs.max(delta);
                        }

                        println!(
                            "R16_17D_ROTATION_EXPECTED_REL_L2={:.12e}",
                            (expected_error_sq / expected_sq).sqrt()
                        );
                        println!(
                            "R16_17D_ROTATION_UNCHANGED_REL_L2={:.12e}",
                            (unchanged_error_sq / expected_sq).sqrt()
                        );
                        println!("R16_17D_ROTATION_EXPECTED_MAX_ABS={max_abs:.12e}");

                        diagnostic_ms += diagnostic_started.elapsed().as_secs_f64() * 1000.0;
                    }

                    multiply_complex_slots_cp_prepared(&shifted, &prepared_mask, &chain, &prepared0)
                }
            })
            .collect();

        if diagnostic && tile_row == 0 {
            let diagnostic_started = Instant::now();

            for (label, tile) in [("EVEN_MASK", 0usize), ("ODD_ROTATE_MASK", 1usize)] {
                let actual = decode(&aligned[tile], &secret, &embedding);

                let mut active_error_sq = 0.0_f64;
                let mut active_reference_sq = 0.0_f64;
                let mut active_max_abs = 0.0_f64;
                let mut inactive_sq = 0.0_f64;
                let mut inactive_max_abs = 0.0_f64;

                for (slot, value) in actual.iter().enumerate().take(slots) {
                    if slot < HALF {
                        let packed_row = slot / 256;
                        let packed_col = slot % 256;
                        let natural_row = (packed_row % 4) * 4 + packed_row / 4;
                        let natural_col = reverse_base4_4(packed_col);
                        let expected = sample(natural_row, tile * 256 + natural_col);

                        let error = (*value - expected).norm();
                        active_error_sq += error * error;
                        active_reference_sq += expected.norm_sqr();
                        active_max_abs = active_max_abs.max(error);
                    } else {
                        let magnitude = value.norm();
                        inactive_sq += magnitude * magnitude;
                        inactive_max_abs = inactive_max_abs.max(magnitude);
                    }
                }

                let rel_l2 = (active_error_sq / active_reference_sq).sqrt();
                let active_rms = (active_error_sq / HALF as f64).sqrt();
                let inactive_rms = (inactive_sq / (slots - HALF) as f64).sqrt();

                println!("R16_17C_{label}_REL_L2={rel_l2:.12e}");
                println!("R16_17C_{label}_ACTIVE_RMS={active_rms:.12e}");
                println!("R16_17C_{label}_MAX_ABS={active_max_abs:.12e}");
                println!("R16_17C_{label}_INACTIVE_RMS={inactive_rms:.12e}");
                println!("R16_17C_{label}_INACTIVE_MAX_ABS={inactive_max_abs:.12e}");
            }

            diagnostic_ms += diagnostic_started.elapsed().as_secs_f64() * 1000.0;
        }

        if extraction_only {
            println!("R16_17C_INPUT_ENCRYPT_MS={input_ms:.3}");
            println!("R16_17C_EXTRACT_MS={extract_ms:.3}");
            println!("R16_17C_DIAGNOSTIC_MS={diagnostic_ms:.3}");
            println!("R16_17C_EXTRACTION_ONLY_STATUS=PASS");
            return;
        }

        drop(physical);
        extract_ms += extract_started.elapsed().as_secs_f64() * 1000.0;

        if diagnostic {
            let diagnostic_started = Instant::now();
            let mut error_sq = 0.0_f64;
            let mut reference_sq = 0.0_f64;
            let mut max_abs = 0.0_f64;

            for (tile, ct) in aligned.iter().enumerate() {
                let actual = decode(ct, &secret, &embedding);

                for packed_row in 0..16 {
                    let natural_row = (packed_row % 4) * 4 + packed_row / 4;
                    let y = tile_row * 16 + natural_row;

                    for packed_col in 0..256 {
                        let natural_col = reverse_base4_4(packed_col);
                        let expected = sample(y, tile * 256 + natural_col);
                        let slot = packed_row * 256 + packed_col;

                        let error = (actual[slot] - expected).norm();
                        error_sq += error * error;
                        reference_sq += expected.norm_sqr();
                        max_abs = max_abs.max(error);
                    }
                }
            }

            let rel_l2 = (error_sq / reference_sq).sqrt();
            println!("R16_17B_EXTRACT_REL_L2={rel_l2:.12e}");
            println!("R16_17B_EXTRACT_MAX_ABS={max_abs:.12e}");
            diagnostic_ms += diagnostic_started.elapsed().as_secs_f64() * 1000.0;
        }

        let phase_started = Instant::now();
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

        if diagnostic {
            let diagnostic_started = Instant::now();
            let mut error_sq = 0.0_f64;
            let mut reference_sq = 0.0_f64;
            let mut max_abs = 0.0_f64;

            for (j, branch) in branches.iter().enumerate() {
                let actual = decode(&branch[H_BRANCH % 5], &secret, &embedding);

                for packed_row in 0..16 {
                    let natural_row = (packed_row % 4) * 4 + packed_row / 4;
                    let y = tile_row * 16 + natural_row;

                    for packed_col in 0..256 {
                        let natural_col = reverse_base4_4(packed_col);
                        let mut expected = Complex64::new(0.0, 0.0);

                        for m in 0..5 {
                            let tile = j + 3 * m;
                            expected +=
                                sample(y, tile * 256 + natural_col) * root(5, (H_BRANCH % 5) * m);
                        }

                        let slot = packed_row * 256 + packed_col;
                        let error = (actual[slot] - expected).norm();

                        error_sq += error * error;
                        reference_sq += expected.norm_sqr();
                        max_abs = max_abs.max(error);
                    }
                }
            }

            let rel_l2 = (error_sq / reference_sq).sqrt();
            println!("R16_17B_RADIX5_REL_L2={rel_l2:.12e}");
            println!("R16_17B_RADIX5_MAX_ABS={max_abs:.12e}");
            diagnostic_ms += diagnostic_started.elapsed().as_secs_f64() * 1000.0;
        }

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

        if diagnostic {
            let actual = decode(&current, &secret, &embedding);
            let mut error_sq = 0.0_f64;
            let mut reference_sq = 0.0_f64;
            let mut max_abs = 0.0_f64;

            for packed_row in 0..16 {
                let natural_row = (packed_row % 4) * 4 + packed_row / 4;
                let y = tile_row * 16 + natural_row;

                for packed_col in 0..256 {
                    let natural_col = reverse_base4_4(packed_col);

                    let mut expected = Complex64::new(0.0, 0.0);
                    for tile in 0..15 {
                        expected += sample(y, tile * 256 + natural_col) * root(15, H_BRANCH * tile);
                    }

                    let got = actual[packed_row * 256 + packed_col];
                    let error = (got - expected).norm();

                    error_sq += error * error;
                    reference_sq += expected.norm_sqr();
                    max_abs = max_abs.max(error);
                }
            }

            let rel_l2 = (error_sq / reference_sq).sqrt();

            println!("R16_17_FFT15_HANDOFF_REL_L2={rel_l2:.12e}");
            println!("R16_17_FFT15_HANDOFF_MAX_ABS={max_abs:.12e}");
        }

        fft15_ms += phase_started.elapsed().as_secs_f64() * 1000.0;
        let fft256_started = Instant::now();
        for level in 0..4 {
            current = apply_stage(
                &current,
                &diagonals[level],
                &StageContext {
                    keys: &rotation_keys[level],
                    evaluator: &evaluator,
                    embedding: &embedding,
                    chain: &chain,
                    plan: &plans[level],
                    prepared: &prepared_plans[level],
                },
            );
            assert_eq!(current.level(), level + 4);
        }
        fft256_ms += fft256_started.elapsed().as_secs_f64() * 1000.0;
        if diagnostic {
            let actual = decode(&current, &secret, &embedding);
            let mut error_sq = 0.0_f64;
            let mut signal_error_sq = 0.0_f64;
            let mut zero_error_sq = 0.0_f64;
            let mut signal_bins = 0_usize;
            let mut zero_bins = 0_usize;
            let mut ref_sq = 0.0_f64;
            let mut max_abs = 0.0_f64;
            for packed_row in 0..16 {
                let natural_row = (packed_row % 4) * 4 + packed_row / 4;
                let y = tile_row * 16 + natural_row;
                for local_h in 0..256 {
                    let kx = H_BRANCH + 15 * local_h;
                    let expected: Complex64 = [
                        (7 + 15 * 3, 135 * 2, 1.0e-7_f64),
                        (7 + 15 * 31, 135 * 11, 0.4e-7),
                        (7 + 15 * 129, 135 * 6, 0.7e-7),
                    ]
                    .iter()
                    .filter(|&&(mode_x, _, _)| mode_x == kx)
                    .map(|&(_, mode_y, amplitude)| {
                        Complex64::new(amplitude * 3840.0, 0.0) * root(2160, mode_y * y).conj()
                    })
                    .sum();
                    let got = actual[packed_row * 256 + local_h];
                    let delta = (got - expected).norm();
                    error_sq += delta * delta;
                    ref_sq += expected.norm_sqr();
                    if expected.norm_sqr() > 0.0 {
                        signal_error_sq += delta * delta;
                        signal_bins += 1;
                    } else {
                        zero_error_sq += delta * delta;
                        zero_bins += 1;
                    }
                    max_abs = max_abs.max(delta);
                }
            }
            let rel_l2 = (error_sq / ref_sq).sqrt();
            println!("R16_17_ROW_{tile_row}_REL_L2={rel_l2:.12e}");
            println!("R16_17_ROW_{tile_row}_MAX_ABS={max_abs:.12e}");
            println!("R16_17_ROW_{tile_row}_SIGNAL_BINS={signal_bins}");
            println!("R16_17_ROW_{tile_row}_ZERO_BINS={zero_bins}");
            println!(
                "R16_17_ROW_{tile_row}_SIGNAL_ERROR_L2={:.12e}",
                signal_error_sq.sqrt()
            );
            println!(
                "R16_17_ROW_{tile_row}_ZERO_ERROR_L2={:.12e}",
                zero_error_sq.sqrt()
            );
            println!(
                "R16_17_ROW_{tile_row}_SIGNAL_REL_L2={:.12e}",
                (signal_error_sq / ref_sq).sqrt()
            );
            println!("R16_17B_DIAGNOSTIC_MS={diagnostic_ms:.3}");
            assert!(
                rel_l2.is_finite() && rel_l2 < 1e-2,
                "horizontal checkpoint accuracy failed for tile row {tile_row}"
            );
        }
        horizontal.push(current);
        if tile_row % 15 == 14 || tile_row == 0 {
            println!("R16_16C_HORIZONTAL_TILE_ROWS_DONE={}", tile_row + 1);
        }
    }
    println!("R16_17_INPUT_ENCRYPT_MS={input_ms:.3}");
    println!("R16_17_EXTRACT_MS={extract_ms:.3}");
    println!("R16_17_FFT15_MS={fft15_ms:.3}");
    println!("R16_17_FFT256_MS={fft256_ms:.3}");
    println!(
        "R16_17_TOTAL_MS={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    println!("R16_17_HORIZONTAL_STATUS=PASS");
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
