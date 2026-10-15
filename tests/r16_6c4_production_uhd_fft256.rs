// R16.5.16c: actual ciphertext-continuous selected-spectrum UHD FFT2.
// Research-only; one vertical DIF branch and one horizontal branch.
// R16.5.13c: all 15 real ciphertext FFT15 -> fused twiddle FFT256 branches.
// Full 3840-frequency horizontal spectrum for 16 independent physical rows.
// Research-only parameters and deterministic test keys; no intermediate decryption.
use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next,
    research_profile_32768_for_levels, rotate_left_hybrid_rns_ckks_with_prepared_ntt,
    rotate_right_hybrid_rns_ckks_with_prepared_ntt, rotation_exponent_left,
    rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState, PreparedHybridRnsGaloisKey,
    PreparedRnsPlaintextNtt, RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::level1::{multiply_complex_slots_cp_prepared, PreparedComplexSlotsCp};
use ccmm_rs::eblas::mixed_radix_ckks::{
    packed_fft15_all_branches_cp_prepared, packed_fft15_twiddled_radix3_cp_prepared,
    packed_radix5_dft_cp_prepared, PreparedFft15TwiddledRadix3Cp, PreparedRadix5Cp,
};
use ccmm_rs::eblas::prepared_radix4_fft256::{
    apply_fft256_multirow_prepared_with_context, apply_fft3840_horizontal_from_radix5_prepared,
    apply_fft3840_horizontal_from_radix5_prepared_with_context, apply_stage,
    PreparedFft256BranchBank, PreparedFft256ExecutionContext, PreparedFft256SelectedBranch,
    PreparedFft256Stage, StageContext,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, Modulus,
    ModulusBasis, PreparedRnsNttPlan, RnsNttPlan,
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

// R23.3D.4: Validate branch-specific FFT256 preparation geometry.
#[test]
fn r23_3d4_fft256_branch_sharing_invariants() {
    let slots = 256;
    let spans = [4, 16, 64, 256];
    let mut first_stage_varies = false;

    for (stage, &span) in spans.iter().enumerate() {
        let reference = stage_diagonals(slots, span, 0);
        let reference_offsets: Vec<_> = reference.keys().copied().collect();

        assert!(!reference_offsets.is_empty());

        for branch in 1..15 {
            let candidate = stage_diagonals(slots, span, branch);
            let candidate_offsets: Vec<_> = candidate.keys().copied().collect();

            assert_eq!(
                reference_offsets, candidate_offsets,
                "rotation offsets differ at stage {stage}, branch {branch}"
            );

            if stage == 0 {
                first_stage_varies |= candidate != reference;
            } else {
                assert_eq!(
                    reference, candidate,
                    "shared coefficients differ at stage {stage}, branch {branch}"
                );
            }
        }

        println!("R23_3D4_STAGE_{stage}_OFFSET_SHARING=PASS");

        if stage > 0 {
            println!("R23_3D4_STAGE_{stage}_COEFFICIENT_SHARING=PASS");
        }
    }

    assert!(
        first_stage_varies,
        "first-stage coefficients unexpectedly branch-independent"
    );

    println!("R23_3D4_FIRST_STAGE_BRANCH_DEPENDENCE=PASS");
    println!("R23_3D4_BRANCH_SHARING_STATUS=PASS");
}

#[cfg(target_os = "macos")]
fn r23_5_memory_checkpoint(label: &str) {
    use std::ffi::{c_int, c_void};

    #[repr(C)]
    #[derive(Default)]
    struct RusageInfoV0 {
        ri_uuid: [u8; 16],
        ri_user_time: u64,
        ri_system_time: u64,
        ri_pkg_idle_wkups: u64,
        ri_interrupt_wkups: u64,
        ri_pageins: u64,
        ri_wired_size: u64,
        ri_resident_size: u64,
        ri_phys_footprint: u64,
        ri_proc_start_abstime: u64,
        ri_proc_exit_abstime: u64,
    }

    unsafe extern "C" {
        fn getpid() -> c_int;
        fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *mut c_void) -> c_int;
    }

    const RUSAGE_INFO_V0: c_int = 0;

    let mut usage = RusageInfoV0::default();

    // SAFETY: usage is a writable C-compatible structure and the
    // declared flavor matches the RusageInfoV0 layout.
    let status =
        unsafe { proc_pid_rusage(getpid(), RUSAGE_INFO_V0, (&raw mut usage).cast::<c_void>()) };

    assert_eq!(status, 0, "proc_pid_rusage failed");

    println!(
        "R23_5_MEMORY_{}_CURRENT_RSS_BYTES={}",
        label, usage.ri_resident_size
    );

    println!(
        "R23_5_MEMORY_{}_PHYSICAL_FOOTPRINT_BYTES={}",
        label, usage.ri_phys_footprint
    );
}

#[cfg(not(target_os = "macos"))]
fn r23_5_memory_checkpoint(label: &str) {
    println!("R23_5_MEMORY_{}_CURRENT_RSS_BYTES=UNAVAILABLE", label);
    println!(
        "R23_5_MEMORY_{}_PHYSICAL_FOOTPRINT_BYTES=UNAVAILABLE",
        label
    );
}

/// R23.4C.4: verify that the existing FFT256 key sets
/// cover the rotation offsets required by all 15 branches.
#[test]
fn r23_4c4_rotation_key_coverage() {
    const BRANCHES: usize = 15;
    const SLOTS: usize = 16384;

    for span in [4usize, 16, 64, 256] {
        let reference = stage_diagonals(SLOTS, span, 7);

        let reference_offsets: Vec<isize> = reference.keys().copied().collect();

        for branch in 0..BRANCHES {
            let candidate = stage_diagonals(SLOTS, span, branch);

            let candidate_offsets: Vec<isize> = candidate.keys().copied().collect();

            assert_eq!(
                candidate_offsets, reference_offsets,
                "FFT256 rotation-key coverage mismatch: \
                 span={span}, branch={branch}"
            );
        }
    }

    println!("R23_4C4_ROTATION_KEY_BRANCHES=15");
    println!("R23_4C4_ROTATION_KEY_STAGES=4");
    println!("R23_4C4_ROTATION_KEY_COVERAGE=PASS");
}

/// R23.4C.3: Verify the production FFT256 diagonal schedule against
/// an independent direct DFT for every horizontal branch and frequency.
#[test]
fn r23_4c3_fft256_kernel_permutation_matches_direct_dft() {
    const LOCAL: usize = 256;
    const BRANCHES: usize = 15;
    const WIDTH: usize = LOCAL * BRANCHES;

    fn apply_plaintext_diagonals(input: &[Complex64], diagonals: &Diagonals) -> Vec<Complex64> {
        let slots = input.len();

        let mut output = vec![Complex64::new(0.0, 0.0); slots];

        for (&offset, diagonal) in diagonals {
            assert_eq!(diagonal.len(), slots);

            for index in 0..slots {
                let source = (index as isize + offset).rem_euclid(slots as isize) as usize;

                output[index] += diagonal[index] * input[source];
            }
        }

        output
    }

    let natural: Vec<Complex64> = (0..LOCAL)
        .map(|n| {
            Complex64::new(
                ((17 * n + 11) % 101) as f64 / 101.0 - 0.5,
                ((29 * n + 7) % 97) as f64 / 97.0 - 0.5,
            )
        })
        .collect();

    let packed: Vec<Complex64> = (0..LOCAL)
        .map(|index| natural[reverse_base4_4(index)])
        .collect();

    let mut maximum_error = 0.0_f64;

    for branch in 0..BRANCHES {
        let mut current = packed.clone();

        for span in [4, 16, 64, 256] {
            let diagonals = stage_diagonals(LOCAL, span, branch);

            current = apply_plaintext_diagonals(&current, &diagonals);
        }

        for (local_frequency, actual) in current.iter().enumerate() {
            let expected: Complex64 = (0..LOCAL)
                .map(|n| natural[n] * root(WIDTH, branch * n) * root(LOCAL, local_frequency * n))
                .sum();

            let error = (*actual - expected).norm();

            maximum_error = maximum_error.max(error);

            assert!(
                error < 1e-8,
                "FFT256 mismatch: branch={branch}, \
                 local={local_frequency}, global={}, error={error:e}",
                branch + BRANCHES * local_frequency
            );
        }
    }

    println!("R23_4C3_BRANCHES_VERIFIED={BRANCHES}");
    println!("R23_4C3_LOCAL_FREQUENCIES_VERIFIED={LOCAL}");
    println!("R23_4C3_TOTAL_FREQUENCIES_VERIFIED={WIDTH}");
    println!("R23_4C3_MAX_ABSOLUTE_ERROR={maximum_error:.12e}");
    println!("R23_4C3_KERNEL_PERMUTATION_STATUS=PASS");
}

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

fn plaintext_apply_stage(input: &[Complex64], diagonals: &Diagonals) -> Vec<Complex64> {
    let slots = input.len();
    let mut output = vec![Complex64::new(0.0, 0.0); slots];

    for (&offset, diagonal) in diagonals {
        assert_eq!(diagonal.len(), slots);
        for i in 0..slots {
            let source = (i as isize + offset).rem_euclid(slots as isize) as usize;
            output[i] += diagonal[i] * input[source];
        }
    }

    output
}

fn numerical_error(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());

    let mut error_sq = 0.0_f64;
    let mut reference_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (&got, &want) in actual.iter().zip(expected.iter()) {
        let delta = (got - want).norm();
        error_sq += delta * delta;
        reference_sq += want.norm_sqr();
        max_abs = max_abs.max(delta);
    }

    assert!(reference_sq > 0.0);
    ((error_sq / reference_sq).sqrt(), max_abs)
}

fn apply_stage_baseline(
    input: &RnsCkksCiphertext,
    diagonals: &Diagonals,
    ctx: &StageContext<'_>,
    embedding: &CkksCanonicalEmbedding,
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
        let plain = encode_rns(diagonal, embedding, input.basis(), plain_scale);
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
fn r16_6c4_production_uhd_fft256() {
    let r24_warm_bench = std::env::var("R24_WARM_BENCH").is_ok_and(|value| value == "1");
    if r24_warm_bench {
        println!("R24_6_MODE=WARM_BENCH");
        println!("R24_6_FULL_UHD=DISABLED");
        println!("R24_6_REFERENCE_COMPARISON=DISABLED");
        println!("R24_6_KEYGEN_REUSE=WITHIN_PROCESS");
    }

    // R25.2C.2: opt-in multi-row horizontal FFT3840 validation.
    // The default UHD test behavior remains unchanged.
    let r25_multirow = std::env::var("R25_2C_TILE_ROWS").ok().map(|value| {
        let count = value
            .parse::<usize>()
            .expect("R25_2C_TILE_ROWS must be an integer");
        assert!(
            (1..=135).contains(&count),
            "R25_2C_TILE_ROWS must be between 1 and 135"
        );
        count
    });

    // R25_3B_3_UHD_GLOBAL_BATCH
    // Opt-in two-row global scheduling experiment.
    let r25_3b_rows = std::env::var("R25_3B_TILE_ROWS").ok().map(|value| {
        let count = value
            .parse::<usize>()
            .expect("R25_3B_TILE_ROWS must be an integer");
        assert_eq!(
            count, 2,
            "R25.3B.3 currently supports exactly two tile rows"
        );
        count
    });

    assert!(
        r25_multirow.is_none() || r25_3b_rows.is_none(),
        "R25_2C_TILE_ROWS and R25_3B_TILE_ROWS are mutually exclusive"
    );

    let r25_multirow = r25_multirow.or(r25_3b_rows);

    if r25_multirow.is_some() {
        assert!(
            !r24_warm_bench
                && std::env::var_os("R23_4C4_COMPARE").is_none()
                && std::env::var_os("R25_1F_2A_COMPARE").is_none(),
            "R25.2C multi-row mode must run independently of comparison/benchmark modes"
        );
    }

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
    let extraction_auxiliary_basis = ModulusBasis::new(vec![
        Modulus::new(1_125_899_908_022_273),
        Modulus::new(1_125_899_908_612_097),
    ]);

    for modulus in extraction_auxiliary_basis.moduli() {
        assert!(
            !basis0.moduli().contains(modulus),
            "hybrid extraction Q/P modulus overlap"
        );
    }

    let mut extraction_qp_moduli = basis0.moduli().to_vec();
    extraction_qp_moduli.extend_from_slice(extraction_auxiliary_basis.moduli());
    let extraction_qp_plan = RnsNttPlan::new(extraction_qp_moduli, degree);

    let extraction_key_started = Instant::now();
    let prepared_left = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
        BoundedGadgetLayout::new(basis0.clone(), 8),
        extraction_auxiliary_basis,
        &secret,
        rotation_exponent_left(degree, HALF),
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        2,
        &extraction_qp_plan,
        &mut rng,
    );

    println!(
        "R16_5_18I9_EXTRACTION_KEYGEN_MS={:.3}",
        extraction_key_started.elapsed().as_secs_f64() * 1000.0
    );
    println!("R16_5_18I9_EXTRACTION_KEY_TYPE=HYBRID_QP");
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
    if std::env::var_os("R23_4C4_COMPARE").is_some() {
        r23_5_memory_checkpoint("M0");
    }
    let rotation_keys: Vec<RotationKeys> = {
        use rayon::prelude::*;

        let workers = std::env::var("FHE_RS_KEYGEN_WORKERS")
            .ok()
            .map(|value| {
                let count = value
                    .parse::<usize>()
                    .expect("FHE_RS_KEYGEN_WORKERS must be an integer");
                assert!(count > 0, "FHE_RS_KEYGEN_WORKERS must be positive");
                count
            })
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|count| count.get())
                    .unwrap_or(1)
            })
            .min(4);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("R25.1B key-generation Rayon pool");

        let started = Instant::now();

        let keys = pool.install(|| {
            (0..4usize)
                .into_par_iter()
                .map(|level| {
                    // Reproducible test-only streams, separated by CKKS level.
                    // Production key generation must use OS-backed entropy.
                    let mut level_rng =
                        ChaCha20Rng::seed_from_u64(0x5232_355F_4B45_5900 ^ (level as u64));

                    generate_keys(
                        &diagonals[level],
                        &bases[level],
                        &auxiliary_basis,
                        &qp_plans[level],
                        degree,
                        &secret,
                        &mut level_rng,
                    )
                })
                .collect::<Vec<RotationKeys>>()
        });

        println!("R25_1B_KEYGEN_WORKERS={workers}");
        println!(
            "R25_1B_PARALLEL_KEYGEN_MS={:.3}",
            started.elapsed().as_secs_f64() * 1000.0
        );
        println!("R25_1B_KEYGEN_LEVELS={}", keys.len());

        keys
    };
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
    // R16.6b.1: prepare FFT256 public operands once per CKKS level.
    if std::env::var_os("R23_4C4_COMPARE").is_some() {
        r23_5_memory_checkpoint("M1");
    }
    let prepared_fft256: Vec<PreparedFft256Stage> = (0..4)
        .map(|stage| {
            PreparedFft256Stage::new(
                stage + 3,
                &diagonals[stage],
                &embedding,
                &chain,
                &prepared_plans[stage],
            )
        })
        .collect();

    assert_eq!(prepared_fft256.len(), 4);

    let r23_selected_compare = std::env::var_os("R23_3D5B_COMPARE").is_some();
    let r23_integrated_compare = std::env::var_os("R23_4C4_COMPARE").is_some();

    let selected_branch = r23_selected_compare.then(|| {
        PreparedFft256SelectedBranch::new(
            H_BRANCH,
            |span, branch| stage_diagonals(slots, span, branch),
            &embedding,
            &chain,
            [
                &prepared_plans[0],
                &prepared_plans[1],
                &prepared_plans[2],
                &prepared_plans[3],
            ],
        )
    });

    if let Some(selected) = selected_branch.as_ref() {
        assert_eq!(selected.branch(), H_BRANCH);
        println!("R23_3D5B_SELECTED_BRANCH_PREPARATION=PASS");
    }

    let r25_bank_started = Instant::now();
    let r23_integrated_bank = (r23_integrated_compare || r24_warm_bench || r25_multirow.is_some())
        .then(|| {
            PreparedFft256BranchBank::new_parallel(
                |span, branch| stage_diagonals(slots, span, branch),
                &embedding,
                &chain,
                [
                    &prepared_plans[0],
                    &prepared_plans[1],
                    &prepared_plans[2],
                    &prepared_plans[3],
                ],
            )
        });

    if r23_integrated_bank.is_some() {
        println!(
            "R25_1A_PARALLEL_BANK_PREPARATION_MS={:.3}",
            r25_bank_started.elapsed().as_secs_f64() * 1000.0
        );
        println!("R25_1A_PREPARED_STAGE_COUNT=18");
    }

    if let Some(bank) = r23_integrated_bank.as_ref() {
        assert_eq!(bank.prepared_stage_count(), 18);
        println!("R23_4C4_PREPARED_STAGE_COUNT=18");
        r23_5_memory_checkpoint("M2");
    }

    // R25.2B: construct prepared Q/P execution resources once per UHD run.
    // Both diagnostic and warm-benchmark paths reuse these resources.
    let contexts = [
        StageContext {
            keys: &rotation_keys[0],
            evaluator: &evaluator,
            chain: &chain,
            qp_plan: &qp_plans[0],
            prepared: &prepared_plans[0],
        },
        StageContext {
            keys: &rotation_keys[1],
            evaluator: &evaluator,
            chain: &chain,
            qp_plan: &qp_plans[1],
            prepared: &prepared_plans[1],
        },
        StageContext {
            keys: &rotation_keys[2],
            evaluator: &evaluator,
            chain: &chain,
            qp_plan: &qp_plans[2],
            prepared: &prepared_plans[2],
        },
        StageContext {
            keys: &rotation_keys[3],
            evaluator: &evaluator,
            chain: &chain,
            qp_plan: &qp_plans[3],
            prepared: &prepared_plans[3],
        },
    ];

    let context_refs = [&contexts[0], &contexts[1], &contexts[2], &contexts[3]];

    let fft256_execution_context = PreparedFft256ExecutionContext::new(context_refs);
    println!("R25_2B_PERSISTENT_CONTEXT=READY");

    let mut r25_3b_pending: Vec<[RnsCkksCiphertext; 15]> = Vec::with_capacity(2);
    let mut r25_3b_started: Option<Instant> = None;

    let mut horizontal: Vec<RnsCkksCiphertext> = Vec::with_capacity(TILE_ROWS);
    for tile_row in 0..r25_multirow.unwrap_or(TILE_ROWS) {
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
                    let shifted = rotate_left_hybrid_rns_ckks_with_prepared_ntt(
                        source,
                        HALF,
                        &prepared_left,
                        &chain,
                        &extraction_qp_plan,
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

        if r25_3b_rows.is_some() {
            let bank = r23_integrated_bank
                .as_ref()
                .expect("R25.3B prepared FFT256 branch bank");

            if r25_3b_started.is_none() {
                r25_3b_started = Some(Instant::now());
            }

            // Preserve the existing FFT15 decomposition and
            // natural horizontal branch ordering.
            let fft15 = packed_fft15_all_branches_cp_prepared(
                &evaluator, &branches, &radix3, &chain, &prepared2,
            );

            assert!(
                fft15.iter().all(|ct| ct.level() == 3),
                "FFT15 outputs must be at CKKS level 3"
            );

            r25_3b_pending.push(fft15);

            println!("R25_3B_ROW_PREPARED={} BRANCHES=15 LEVEL=3", tile_row);

            if r25_3b_pending.len() < 2 {
                continue;
            }

            // R25_3B_4_EXACT_UHD_COMPARISON
            // Reference and global executions consume identical
            // encrypted FFT15 inputs.
            let exact_compare =
                std::env::var("R25_3B_EXACT_COMPARE").is_ok_and(|value| value == "1");

            let reference = if exact_compare {
                let reference_started = Instant::now();

                let reference: Vec<[RnsCkksCiphertext; 15]> = r25_3b_pending
                    .iter()
                    .cloned()
                    .map(|row| {
                        let mut collected = Vec::with_capacity(15);

                        ccmm_rs::eblas::prepared_radix4_fft256::
                                apply_fft256_all_branches_streaming_with_context(
                                    row,
                                    bank,
                                    context_refs,
                                    &fft256_execution_context,
                                    |branch, ciphertext| {
                                        assert_eq!(branch, collected.len());
                                        collected.push(ciphertext);
                                    },
                                );

                        collected
                            .try_into()
                            .unwrap_or_else(|_| panic!("reference FFT256 must return 15 branches"))
                    })
                    .collect();

                println!(
                    "R25_3B_4_REFERENCE_FFT256_MS={:.3}",
                    reference_started.elapsed().as_secs_f64() * 1000.0
                );

                Some(reference)
            } else {
                None
            };

            let fft256_started = Instant::now();

            let outputs = apply_fft256_multirow_prepared_with_context(
                std::mem::take(&mut r25_3b_pending),
                bank,
                context_refs,
                &fft256_execution_context,
            );

            let fft256_ms = fft256_started.elapsed().as_secs_f64() * 1000.0;

            assert_eq!(outputs.len(), 2);

            for (local_row, row) in outputs.iter().enumerate() {
                assert_eq!(row.len(), 15);
                assert!(
                    row.iter().all(|ct| ct.level() == 7),
                    "FFT256 outputs must reach CKKS level 7"
                );

                println!("R25_3B_ROW_COMPLETE={} BRANCHES=15 LEVEL=7", local_row);
            }

            if let Some(reference) = reference {
                assert_eq!(reference.len(), outputs.len());

                for (row_index, (expected_row, actual_row)) in
                    reference.iter().zip(outputs.iter()).enumerate()
                {
                    for (branch, (expected, actual)) in
                        expected_row.iter().zip(actual_row.iter()).enumerate()
                    {
                        assert_eq!(
                            expected.rlwe(),
                            actual.rlwe(),
                            "UHD RLWE mismatch: row={row_index} branch={branch}"
                        );

                        assert_eq!(
                            expected.state(),
                            actual.state(),
                            "UHD CKKS state mismatch: row={row_index} branch={branch}"
                        );

                        assert_eq!(actual.level(), 7);
                    }

                    println!("R25_3B_4_ROW_EXACT={} BRANCHES=15 LEVEL=7", row_index);
                }

                println!("R25_3B_4_EXACT_UHD_CIPHERTEXT_EQUIVALENCE=PASS");
            }

            println!("R25_3B_FFT256_EXECUTION_MS={fft256_ms:.3}");

            println!(
                "R25_3B_PREPARATION_AND_FFT256_MS={:.3}",
                r25_3b_started
                    .expect("R25.3B batch timer")
                    .elapsed()
                    .as_secs_f64()
                    * 1000.0
            );

            println!("R25_3B_HORIZONTAL_OUTPUTS=30");
            println!("R25_3B_HORIZONTAL_OUTPUT_LEVEL=7");
            println!("R25_3B_VERTICAL_FFT_EXECUTED=NO");
            println!("R25_3B_3_UHD_GLOBAL_BATCH=PASS");

            return;
        }

        if r23_integrated_compare || r24_warm_bench || r25_multirow.is_some() {
            if r25_multirow.is_none() {
                assert_eq!(tile_row, 0);
            }

            let bank = r23_integrated_bank
                .as_ref()
                .expect("integrated FFT256 bank");

            let integrated_started = Instant::now();

            let integrated = apply_fft3840_horizontal_from_radix5_prepared_with_context(
                &evaluator,
                &branches,
                &radix3,
                &chain,
                &prepared2,
                bank,
                (context_refs, &fft256_execution_context),
            );

            println!(
                "R23_4C4_INTEGRATED_EXECUTION_MS={:.3}",
                integrated_started.elapsed().as_secs_f64() * 1000.0
            );

            if let Some(row_limit) = r25_multirow {
                assert!(
                    integrated.iter().all(|ct| ct.level() == 7),
                    "R25.2C horizontal outputs must all reach CKKS level 7"
                );

                println!("R25_2C_ROW_COMPLETE={} BRANCHES=15 LEVEL=7", tile_row);

                // Release all 15 branch outputs before processing the next row.
                drop(integrated);

                if tile_row + 1 == row_limit {
                    println!("R25_2C_TILE_ROWS_COMPLETED={row_limit}");
                    println!("R25_2C_HORIZONTAL_OUTPUTS={}", row_limit * 15);
                    println!("R25_2C_HORIZONTAL_OUTPUT_LEVEL=7");
                    println!("R25_2C_MULTIROW_STATUS=PASS");
                    println!("R25_2C_VERTICAL_FFT_EXECUTED=NO");
                    return;
                }

                continue;
            }

            if std::env::var("R25_1F_2A_COMPARE").is_ok_and(|value| value == "1") {
                let compatibility = apply_fft3840_horizontal_from_radix5_prepared(
                    &evaluator,
                    &branches,
                    &radix3,
                    &chain,
                    &prepared2,
                    bank,
                    context_refs,
                );

                for branch in 0..15 {
                    assert_eq!(
                        compatibility[branch].rlwe(),
                        integrated[branch].rlwe(),
                        "R25.1F.2A RLWE mismatch at branch {branch}"
                    );
                    assert_eq!(
                        compatibility[branch].state(),
                        integrated[branch].state(),
                        "R25.1F.2A CKKS state mismatch at branch {branch}"
                    );
                }

                println!("R25_1F_2A_EXACT_CONTEXT_EQUIVALENCE=PASS");
                println!("R25_1F_2A_COMPARED_BRANCHES=15");
            }

            if r24_warm_bench {
                let parallel_workers = std::thread::available_parallelism()
                    .map(|count| count.get())
                    .unwrap_or(1)
                    .min(15);

                // The first integrated execution is a warm-up.
                // Benchmark subsequent calls with the same prepared resources.
                let mut measurements = Vec::new();

                for workers in [1, parallel_workers] {
                    std::env::set_var("FHE_RS_WORKERS", workers.to_string());

                    let started = Instant::now();

                    let output = apply_fft3840_horizontal_from_radix5_prepared_with_context(
                        &evaluator,
                        &branches,
                        &radix3,
                        &chain,
                        &prepared2,
                        bank,
                        (context_refs, &fft256_execution_context),
                    );

                    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

                    assert_eq!(
                        output, integrated,
                        "R24.6 ciphertext mismatch with {workers} workers"
                    );

                    println!("R24_6_WORKERS={workers}");
                    println!("R24_6_WARM_EXECUTION_MS={elapsed_ms:.3}");
                    println!("R24_6_EXACT_CIPHERTEXT_EQUALITY=PASS");

                    measurements.push((workers, elapsed_ms));
                }

                println!(
                    "R24_6_PARALLEL_SPEEDUP={:.4}",
                    measurements[0].1 / measurements[1].1
                );
                println!("R24_6_BENCH_STATUS=PASS");
                return;
            }

            // Independently orchestrate the FFT15 branches and
            // the four prepared FFT256 stages.
            r23_5_memory_checkpoint("M3");

            let reference_fft15 =
                ccmm_rs::eblas::mixed_radix_ckks::packed_fft15_all_branches_cp_prepared(
                    &evaluator, &branches, &radix3, &chain, &prepared2,
                );

            for (branch, mut reference) in reference_fft15.into_iter().enumerate() {
                assert_eq!(reference.level(), 3);

                for (stage, context) in context_refs.iter().enumerate() {
                    reference = apply_stage(&reference, bank.stage(branch, stage), context);
                }

                assert_eq!(reference.level(), 7);

                assert_eq!(
                    integrated[branch], reference,
                    "R23.4C.4 ciphertext mismatch at branch {branch}"
                );

                println!("R23_4C4_BRANCH_{}_EXACT_EQUALITY=PASS", branch);
            }

            r23_5_memory_checkpoint("M4");
            println!("R23_4C4_BRANCHES_VERIFIED=15");
            println!("R23_4C4_OUTPUT_LEVEL=7");
            println!("R23_4C4_INTEGRATED_CIPHERTEXT_STATUS=PASS");
            println!("R23_4C4_FULL_UHD_EXECUTED=NO");
            drop(integrated);
            r23_5_memory_checkpoint("M5");
            return;
        }

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
        let diagnostic = std::env::var_os("R16_5_18I10_DIAGNOSTIC").is_some();
        let mut plaintext_reference = if diagnostic && tile_row == 0 {
            Some(decode(&current, &secret, &embedding))
        } else {
            None
        };

        let compare = std::env::var_os("R16_6B2_COMPARE").is_some();
        let mut selected_current = if r23_selected_compare {
            assert_eq!(tile_row, 0);
            Some(current.clone())
        } else {
            None
        };

        let mut baseline_current = if compare {
            assert_eq!(tile_row, 0);
            Some(current.clone())
        } else {
            None
        };

        for level in 0..4 {
            if let Some(reference) = plaintext_reference.as_mut() {
                *reference = plaintext_apply_stage(reference, &diagonals[level]);
            }
            let stage_started = Instant::now();
            current = apply_stage(
                &current,
                &prepared_fft256[level],
                &StageContext {
                    keys: &rotation_keys[level],
                    evaluator: &evaluator,
                    chain: &chain,
                    qp_plan: &qp_plans[level],
                    prepared: &prepared_plans[level],
                },
            );
            assert_eq!(current.level(), level + 4);
            if let Some(selected_ct) = selected_current.as_mut() {
                let selected = selected_branch
                    .as_ref()
                    .expect("selected FFT256 preparation");

                let selected_next = apply_stage(
                    selected_ct,
                    selected.stage(level),
                    &StageContext {
                        keys: &rotation_keys[level],
                        evaluator: &evaluator,
                        chain: &chain,
                        qp_plan: &qp_plans[level],
                        prepared: &prepared_plans[level],
                    },
                );

                assert_eq!(
                    current,
                    selected_next,
                    "R23.3D.5B ciphertext mismatch at FFT256 stage {}",
                    level + 1
                );

                println!("R23_3D5B_STAGE_{}_EXACT_EQUALITY=PASS", level + 1);

                *selected_ct = selected_next;
            }

            if let Some(baseline) = baseline_current.as_mut() {
                let baseline_started = Instant::now();

                let baseline_next = apply_stage_baseline(
                    baseline,
                    &diagonals[level],
                    &StageContext {
                        keys: &rotation_keys[level],
                        evaluator: &evaluator,
                        chain: &chain,
                        qp_plan: &qp_plans[level],
                        prepared: &prepared_plans[level],
                    },
                    &embedding,
                );

                println!(
                    "R16_6B2_BASELINE_STAGE_{}_MS={:.3}",
                    level + 1,
                    baseline_started.elapsed().as_secs_f64() * 1000.0
                );

                assert_eq!(
                    current,
                    baseline_next,
                    "R16.6b.2 exact ciphertext mismatch at FFT256 stage {}",
                    level + 1
                );

                println!("R16_6B2_STAGE_{}_EXACT_EQUALITY=PASS", level + 1);

                *baseline = baseline_next;
            }

            if tile_row == 0 {
                println!(
                    "R16_5_18I3_HYBRID_FFT256_STAGE_{}_MS={:.3}",
                    level + 1,
                    stage_started.elapsed().as_secs_f64() * 1000.0
                );
            }
        }

        if r23_selected_compare {
            assert_eq!(tile_row, 0);
            assert_eq!(current.level(), 7);
            println!("R23_3D5B_SELECTED_BRANCH_EXACT_STATUS=PASS");
            println!("R23_3D5B_FULL_UHD_EXECUTED=NO");
            return;
        }

        if compare {
            assert_eq!(tile_row, 0);
            assert_eq!(current.level(), 7);
            println!("R16_6B2_EXACT_EQUIVALENCE_STATUS=PASS");
            println!("R16_6B2_FULL_UHD_EXECUTED=NO");
            return;
        }

        horizontal.push(current);

        if diagnostic {
            assert_eq!(tile_row, 0);
            let actual = decode(&horizontal[0], &secret, &embedding);
            let expected = plaintext_reference.expect("plaintext reference");

            let (rel_l2, max_abs) = numerical_error(&actual[..HALF], &expected[..HALF]);

            println!("R16_5_18I4_FFT256_INCREMENTAL_REL_L2={rel_l2:.12e}");
            println!("R16_5_18I4_FFT256_INCREMENTAL_MAX_ABS={max_abs:.12e}");
            println!("R16_5_18I4_HORIZONTAL_LEVEL={}", horizontal[0].level());
            println!("R16_5_18I4_DIAGNOSTIC_STATUS=COMPLETE");
            println!("R16_5_18I4_FULL_UHD_VALIDATED=NO");
            println!("R16_5_18I10_INCREMENTAL_STATUS=COMPLETE");
        }

        if std::env::var_os("R16_5_18I10_DIAGNOSTIC").is_some() {
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
            println!("R16_5_18I10_HORIZONTAL_REL_L2={rel_l2:.12e}");
            println!("R16_5_18I10_HORIZONTAL_MAX_ABS={max_abs:.12e}");
            println!(
                "R16_5_18I3_TOTAL_MS={:.3}",
                started.elapsed().as_secs_f64() * 1000.0
            );

            // Initial diagnostic gate: the previously established
            // full UHD tolerance is 1e-2. We will assess the
            // horizontal-only precision separately after measuring.
            let pass = rel_l2.is_finite() && rel_l2 < 1e-3 && max_abs.is_finite() && max_abs < 1e-2;

            println!(
                "R16_5_18I10_HORIZONTAL_STATUS={}",
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
    println!("R16_5_18I11_HORIZONTAL_HANDOFF_STATUS=PASS");
    println!("R16_5_18I11_HORIZONTAL_CIPHERTEXTS={}", horizontal.len());
    assert!(horizontal.iter().all(|ct| ct.level() == 7));
    println!("R16_5_18I11_HORIZONTAL_HANDOFF_LEVEL=7");
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
    println!("R16_5_18I11_FINAL_LEVEL=13");
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
    println!("R16_5_18I11_REL_L2={rel_l2:.12e}");
    println!("R16_5_18I11_MAX_ABS={max_abs:.12e}");
    println!(
        "R16_5_18I11_TOTAL_MS={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    let pass = rel_l2.is_finite() && rel_l2 < 1e-2 && max_abs.is_finite() && max_abs < 1e-2;
    println!("R16_5_18I11_STATUS={}", if pass { "PASS" } else { "FAIL" });
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
