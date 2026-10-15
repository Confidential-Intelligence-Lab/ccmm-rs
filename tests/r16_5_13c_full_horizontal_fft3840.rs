// R16.5.13c: all 15 real ciphertext FFT15 -> fused twiddle FFT256 branches.
// Full 3840-frequency horizontal spectrum for 16 independent physical rows.
// Research-only parameters and deterministic test keys; no intermediate decryption.
use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next, research_profile_16384,
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
                .expect("coefficient conversion")
                / ct.scale()
        })
        .collect();
    embedding.coefficients_to_slots(&coefficients)
}

#[test]
fn r16_5_13c_full_horizontal_fft3840() {
    let total = Instant::now();
    let profile = research_profile_16384();
    let degree = profile.degree();
    let chain = profile.modulus_chain();
    assert!(chain.max_level() >= 7);
    let scale = profile.initial_scale();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();
    assert_eq!(slots % 256, 0);
    const HALF: usize = 4096;
    const TILES: usize = 15;
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
    let mut rng = ChaCha20Rng::seed_from_u64(0x1651_130B);
    // One physical tile row: 15 horizontal tiles, 4096 samples per tile.
    // Each tile contains 16 independent 256-sample horizontal segments.
    // The encrypted FFT15 acts on the tile axis, preserving all 4096 slots.
    let natural: Vec<Vec<Complex64>> = (0..TILES)
        .map(|tile| {
            (0..HALF)
                .map(|slot| {
                    let x = (tile * HALF + slot) as f64;
                    Complex64::new(0.0003 * (0.013 * x).sin(), 0.0003 * (0.017 * x).cos())
                })
                .collect()
        })
        .collect();
    // The physical FFT15 kernel has one ciphertext per pair of 4096-slot tiles.
    // Pack local FFT256 inputs in base-4 digit-reversed order so the FFT256
    // diagonal kernel can consume the actual FFT15 ciphertext directly.
    let physical: Vec<RnsCkksCiphertext> = (0..8)
        .map(|index| {
            let mut values = vec![Complex64::new(0.0, 0.0); slots];
            for half in 0..2 {
                let tile = 2 * index + half;
                if tile < TILES {
                    for slot in 0..HALF {
                        let block = (slot / 256) * 256;
                        let reordered = block + reverse_base4_4(slot % 256);
                        values[half * HALF + slot] = natural[tile][reordered];
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
            RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain)
        })
        .collect();
    let keygen_start = Instant::now();
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
    let low_mask: Vec<Complex64> = (0..slots)
        .map(|s| Complex64::new(if s < HALF { 1.0 } else { 0.0 }, 0.0))
        .collect();
    let prepared_mask =
        PreparedComplexSlotsCp::new(&physical[0], &low_mask, &embedding, &chain, &plan0);
    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);
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
    assert!(aligned.iter().all(|ct| ct.level() == 1));
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
    let radix3 = PreparedFft15TwiddledRadix3Cp::new(&branches[0][0], &embedding, &chain, &plan2);
    let groups: [[RnsCkksCiphertext; 3]; 5] = std::array::from_fn(|k| {
        packed_fft15_twiddled_radix3_cp_prepared(
            &evaluator,
            [&branches[0][k], &branches[1][k], &branches[2][k]],
            k,
            &radix3,
            &chain,
            &prepared2,
        )
    });
    // Independent cleartext oracle: 15-point DFT on tile axis, public
    // W3840 twiddle, then an independently coded iterative radix-2 FFT256.
    // In addition, selected bins are checked by direct O(3840) DFT.
    let mut reference = vec![Complex64::new(0.0, 0.0); 16 * 3840];
    for row in 0..16 {
        for k_cross in 0..TILES {
            let mut local = vec![Complex64::new(0.0, 0.0); 256];
            for (s, value) in local.iter_mut().enumerate() {
                let cross: Complex64 = (0..TILES)
                    .map(|tile| natural[tile][row * 256 + s] * root(TILES, k_cross * tile))
                    .sum();
                *value = cross * root(3840, k_cross * s);
            }
            // Iterative radix-2 DIT FFT256, bit-reversed input permutation.
            for i in 0usize..256 {
                let j = i.reverse_bits() >> (usize::BITS - 8);
                if i < j {
                    local.swap(i, j);
                }
            }
            let mut len = 2;
            while len <= 256 {
                for base in (0..256).step_by(len) {
                    for j in 0..len / 2 {
                        let a = local[base + j];
                        let b = local[base + j + len / 2] * root(len, j);
                        local[base + j] = a + b;
                        local[base + j + len / 2] = a - b;
                    }
                }
                len *= 2;
            }
            for (k_local, &value) in local.iter().enumerate() {
                reference[row * 3840 + k_cross + TILES * k_local] = value;
            }
        }
    }
    for &row in &[0usize, 7, 15] {
        for &frequency in &[0usize, 7, 255, 1024, 3839] {
            let direct: Complex64 = (0..3840)
                .map(|n| {
                    let tile = n / 256;
                    let s = n % 256;
                    natural[tile][row * 256 + s] * root(3840, frequency * n)
                })
                .sum();
            let staged = reference[row * 3840 + frequency];
            assert!(
                (direct - staged).norm() < 1e-8,
                "independent DFT disagreement: row={row} frequency={frequency}"
            );
        }
    }
    let fft15_ms = keygen_start.elapsed().as_secs_f64() * 1000.0;
    let spans = [4usize, 16, 64, 256];
    // Rotation offsets depend on stage span, NOT cross-frequency. Generate
    // all 24 keys once and reuse them for every one of the 15 branches.
    let base_diagonals: Vec<Diagonals> = spans
        .iter()
        .map(|&span| stage_diagonals(slots, span, 0))
        .collect();
    let keygen_start = Instant::now();
    let keys: Vec<RotationKeys> = (0..4)
        .map(|level| {
            generate_keys(
                &base_diagonals[level],
                &bases[level],
                &plans[level],
                degree,
                &secret,
                &mut rng,
            )
        })
        .collect();
    let keygen_ms = keygen_start.elapsed().as_secs_f64() * 1000.0;
    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);
    let start = Instant::now();
    let mut measured = vec![Complex64::new(0.0, 0.0); 16 * 3840];
    let mut occupied = vec![false; 16 * 3840];
    let mut branch_relative = Vec::new();
    let mut branch_max_abs = Vec::new();
    let mut branch_times_ms = Vec::new();
    let mut error_sq = 0.0;
    let mut reference_sq = 0.0;
    let mut max_abs = 0.0_f64;
    for k_cross in 0..TILES {
        let branch_start = Instant::now();
        let mut current = groups[k_cross % 5][k_cross / 5].clone();
        assert_eq!(current.level(), 3);
        let diagonals: Vec<Diagonals> = spans
            .iter()
            .map(|&span| stage_diagonals(slots, span, k_cross))
            .collect();
        for level in 0..4 {
            current = apply_stage(
                &current,
                &diagonals[level],
                &StageContext {
                    keys: &keys[level],
                    evaluator: &evaluator,
                    embedding: &embedding,
                    chain: &chain,
                    plan: &plans[level],
                    prepared: &prepared_plans[level],
                },
            );
            assert_eq!(current.level(), level + 4, "unexpected stage level");
        }
        assert_eq!(current.level(), 7);
        let actual = decode(&current, &secret, &embedding);
        let mut branch_err_sq = 0.0;
        let mut branch_ref_sq = 0.0;
        let mut branch_max = 0.0_f64;
        for row in 0..16 {
            for k_local in 0..256 {
                let index = row * 3840 + k_cross + TILES * k_local;
                assert!(!occupied[index], "duplicate frequency output {index}");
                occupied[index] = true;
                measured[index] = actual[row * 256 + k_local];
                let expected = reference[index];
                let delta = (measured[index] - expected).norm();
                branch_err_sq += delta * delta;
                branch_ref_sq += expected.norm_sqr();
                branch_max = branch_max.max(delta);
            }
        }
        error_sq += branch_err_sq;
        reference_sq += branch_ref_sq;
        max_abs = max_abs.max(branch_max);
        let relative = (branch_err_sq / branch_ref_sq).sqrt();
        branch_relative.push(relative);
        branch_max_abs.push(branch_max);
        branch_times_ms.push(branch_start.elapsed().as_secs_f64() * 1000.0);
        println!("R16_FULL_BRANCH_{k_cross:02}_REL_L2={relative:.12e}");
        println!("R16_FULL_BRANCH_{k_cross:02}_MAX_ABS={branch_max:.12e}");
    }
    assert!(
        occupied.iter().all(|&filled| filled),
        "missing frequency bins"
    );
    let relative = (error_sq / reference_sq).sqrt();
    println!("R16_FULL_DEGREE={degree}");
    println!("R16_FULL_SLOTS={slots}");
    println!("R16_FULL_PHYSICAL_TILES={TILES}");
    println!("R16_FULL_ROWS=16");
    println!("R16_FULL_FREQUENCIES_PER_ROW=3840");
    println!("R16_FULL_OUTPUTS={}", measured.len());
    println!("R16_FULL_BRANCHES={TILES}");
    println!("R16_FULL_REAL_CIPHERTEXT_CONTINUITY=true");
    println!("R16_FULL_INTERMEDIATE_DECRYPTIONS=0");
    println!("R16_FULL_FINAL_LEVEL=7");
    println!("R16_FULL_TWIDDLE_EXTRA_LEVELS=0");
    println!(
        "R16_FULL_ROTATION_KEYS={}",
        keys.iter().map(|k| k.len()).sum::<usize>()
    );
    println!("R16_FULL_REL_L2={relative:.12e}");
    println!("R16_FULL_MAX_ABS={max_abs:.12e}");
    println!(
        "R16_FULL_WORST_BRANCH_REL_L2={:.12e}",
        branch_relative.iter().copied().fold(0.0_f64, f64::max)
    );
    println!(
        "R16_FULL_WORST_BRANCH_MAX_ABS={:.12e}",
        branch_max_abs.iter().copied().fold(0.0_f64, f64::max)
    );
    println!("R16_FULL_FFT15_MS={fft15_ms:.3}");
    println!("R16_FULL_KEYGEN_MS={keygen_ms:.3}");
    println!("R16_FULL_BRANCH_TIMES_MS={branch_times_ms:?}");
    println!(
        "R16_FULL_EXECUTION_MS={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "R16_FULL_TOTAL_MS={:.3}",
        total.elapsed().as_secs_f64() * 1000.0
    );
    // Numerical pass is distinct from the stricter 1e-3 research target.
    let pass = relative.is_finite() && relative < 5e-3 && max_abs < 5e-3;
    println!(
        "R16_FULL_ACCURACY_TARGET_1E_3={}",
        if relative < 1e-3 { "MET" } else { "NOT_MET" }
    );
    println!("R16_FULL_STATUS={}", if pass { "PASS" } else { "FAIL" });
    assert!(pass, "full horizontal encrypted FFT3840 validation failed");
}
