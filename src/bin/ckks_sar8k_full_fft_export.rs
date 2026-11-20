//! Isolated 8K SAR 26-stage FFT2 integration (research correctness experiment).
//! DATE 2027 SAR 8K: retained-state two-row and stage-14 validation.
//!
//! 8x1 macrotile row strip: 3 global DIF stages and the existing cached 1K
//! row engine (10 local stages). This is experimental, non-security-bearing
//! validation; research-only and not a production encryption service.
//! Deterministic research-only keys MUST NOT be deployed.

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_65536_for_levels, rotation_exponent_left, rotation_exponent_right,
    CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext, RnsCkksEvaluationKeys,
    RnsCkksEvaluator, RnsCkksLevelKeys, RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{
    execute_packed_1k_fft_columns_cp, execute_packed_1k_fft_rows_cp_cached,
    packed_fft_dif_butterfly_cp_prepared, PackedRowPreparationContext,
};
use ccmm_rs::eblas::PreparedComplexSlotsCp;
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout,
    RnsKeygenConfig,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, ModulusChain,
    PreparedRnsNttPlan, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs;
use std::fs::OpenOptions;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Instant;

const DIM: usize = 8192;
const NATIVE_WIDTH: usize = 6144;
const NATIVE_HEIGHT: usize = 5120;
const MACRO_DIM: usize = 1024;
const MACROS_PER_AXIS: usize = 8;
const TILE_DIM: usize = 64;
const TILE_ELEMENTS: usize = TILE_DIM * TILE_DIM;
const TILES_PER_CT: usize = 8;
const CT_PER_MACRO: usize = 32;
const SLOT_COUNT: usize = TILE_ELEMENTS * TILES_PER_CT;
const REQUIRED_LEVELS: usize = 26;
const GUARD_LIMBS: usize = 2;
const SIGMA: f64 = 3.19;
const STRIP_HEIGHT: usize = MACRO_DIM;
const TOLERANCE: f64 = 5.0e-3;

// NumPy's ordinary little-endian, C-contiguous float64 .npy format.
// The file is loaded as bytes, avoiding duplicate 512 MiB float64 buffers.
struct NpyInput {
    bytes: Vec<u8>,
    payload_start: usize,
}

impl NpyInput {
    fn open(path: &Path) -> Self {
        let bytes = fs::read(path).expect("cannot read normalized NumPy input");
        assert!(bytes.starts_with(b"\x93NUMPY"), "expected NumPy .npy input");
        assert!(bytes.len() >= 12, "truncated NumPy header");
        let version = bytes[6];
        let (header_start, header_len) = match version {
            1 => (10, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
            2 | 3 => (
                12,
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            ),
            _ => panic!("unsupported NumPy .npy version"),
        };
        let payload_start = header_start + header_len;
        assert!(payload_start <= bytes.len(), "truncated NumPy header data");
        let header = std::str::from_utf8(&bytes[header_start..payload_start])
            .expect("NumPy header is not UTF-8");
        assert!(
            header.contains("<f8"),
            "expected little-endian float64 NumPy input"
        );
        assert!(
            header.contains("'fortran_order': False")
                || header.contains("\"fortran_order\": False"),
            "expected C-contiguous NumPy input"
        );
        let shape = header
            .split("shape")
            .nth(1)
            .expect("NumPy header lacks shape");
        let inside = shape
            .split_once('(')
            .expect("NumPy header has malformed shape")
            .1
            .split_once(')')
            .expect("NumPy header has unterminated shape")
            .0;
        let dims: Vec<usize> = inside
            .split(',')
            .filter(|part| !part.trim().is_empty())
            .map(|part| part.trim().parse().expect("invalid NumPy shape"))
            .collect();
        assert_eq!(dims, [DIM, DIM], "expected exactly 8192x8192 input");
        assert_eq!(
            bytes.len(),
            payload_start + DIM * DIM * std::mem::size_of::<f64>(),
            "unexpected NumPy payload length"
        );
        Self {
            bytes,
            payload_start,
        }
    }

    #[inline]
    fn at(&self, row: usize, col: usize) -> f64 {
        assert!(row < DIM && col < DIM);
        let offset = self.payload_start + (row * DIM + col) * 8;
        f64::from_le_bytes(self.bytes[offset..offset + 8].try_into().unwrap())
    }

    fn validate(&self) {
        let start = Instant::now();
        let mut native_nonzero = 0usize;
        for row in 0..DIM {
            for col in 0..DIM {
                let value = self.at(row, col);
                assert!(
                    value.is_finite() && (0.0..=1.0).contains(&value),
                    "normalized source has nonfinite/out-of-range sample"
                );
                if row >= NATIVE_HEIGHT || col >= NATIVE_WIDTH {
                    assert_eq!(value, 0.0, "nonzero padding at ({row},{col})");
                } else if value != 0.0 {
                    native_nonzero += 1;
                }
            }
        }
        assert!(native_nonzero > 0, "native SAR input is entirely zero");
        println!("DATE27_STRIP_INPUT_SHAPE={DIM}x{DIM}");
        println!("DATE27_STRIP_NATIVE_SHAPE={NATIVE_HEIGHT}x{NATIVE_WIDTH}");
        println!("DATE27_STRIP_NONZERO_NATIVE_PIXELS={native_nonzero}");
        println!("DATE27_STRIP_ZERO_PADDING=PASS");
        println!(
            "DATE27_STRIP_INPUT_CHECK_MS={}",
            start.elapsed().as_millis()
        );
    }
}

// Physical slot layout matches ckks_packed_uhd_fft_validation (R24.1):
// each CT holds eight vertically adjacent 64x64 tiles.
fn physical_position(macro_row: usize, macro_col: usize, ct: usize, slot: usize) -> (usize, usize) {
    assert!(macro_row < MACROS_PER_AXIS && macro_col < MACROS_PER_AXIS);
    assert!(ct < CT_PER_MACRO && slot < SLOT_COUNT);
    let row_block = ct / 16;
    let tile_col = ct % 16;
    let lane = slot / TILE_ELEMENTS;
    let slot_in_tile = slot % TILE_ELEMENTS;
    let local_row = slot_in_tile / TILE_DIM;
    let local_col = slot_in_tile % TILE_DIM;
    (
        macro_row * MACRO_DIM + (row_block * TILES_PER_CT + lane) * TILE_DIM + local_row,
        macro_col * MACRO_DIM + tile_col * TILE_DIM + local_col,
    )
}

fn fill_slots(image: &NpyInput, macro_row: usize, macro_col: usize, ct: usize) -> Vec<Complex64> {
    (0..SLOT_COUNT)
        .map(|slot| {
            let (r, c) = physical_position(macro_row, macro_col, ct, slot);
            Complex64::new(image.at(r, c), 0.0)
        })
        .collect()
}

fn decrypt_slots(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let plan = RnsNttPlan::new(
        ciphertext.basis().moduli().to_vec(),
        ciphertext.rlwe().degree(),
    );
    let plaintext = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);
    let modulus = composite_modulus_big(plaintext.basis());
    let coefficients: Vec<f64> = reconstruct_coefficients_big(&plaintext)
        .into_iter()
        .map(|value| {
            centered_representative_big(&value, &modulus)
                .to_f64()
                .expect("centered CKKS coefficient must fit f64")
                / ciphertext.scale()
        })
        .collect();
    embedding.coefficients_to_slots(&coefficients)
}

fn encrypt_slots(
    slots: &[Complex64],
    macro_index: usize,
    ct_index: usize,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    top_plan: &RnsNttPlan,
    scale: f64,
) -> RnsCkksCiphertext {
    let plaintext = encode_rns(slots, embedding, chain.top(), scale);
    let mut rng = ChaCha20Rng::seed_from_u64(
        0x5232_304b_454e_4300 ^ ((macro_index as u64) << 8) ^ ct_index as u64,
    );
    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        secret,
        top_plan,
        &mut rng,
    );
    RnsCkksCiphertext::new(rlwe, CkksChainState::top(chain, scale), chain)
}

#[derive(Default)]
struct Accuracy {
    error_sq: f64,
    reference_sq: f64,
    max_abs: f64,
    slots_checked: usize,
}

impl Accuracy {
    fn add(&mut self, actual: Complex64, expected: Complex64) {
        let delta = actual - expected;
        self.error_sq += delta.norm_sqr();
        self.reference_sq += expected.norm_sqr();
        self.max_abs = self.max_abs.max(delta.norm());
        self.slots_checked += 1;
    }
    fn rel_l2(&self) -> f64 {
        if self.reference_sq > 0.0 {
            (self.error_sq / self.reference_sq).sqrt()
        } else {
            self.error_sq.sqrt()
        }
    }
}

/// Return bytes as Linux reports them. VmHWM is an observed high-water mark,
/// not a prediction of future allocation peaks. Non-Linux reports no RSS.
fn rss_kib(field: &str) -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        if line.starts_with(field) && line.as_bytes().get(field.len()) == Some(&b':') {
            line.split_whitespace().nth(1)?.parse::<u64>().ok()
        } else {
            None
        }
    })
}

fn log_memory(label: &str) {
    match (rss_kib("VmRSS"), rss_kib("VmHWM")) {
        (Some(rss), Some(high)) => println!(
            "DATE27_STRIP_MEMORY checkpoint={label} rss_gib={:.4} peak_rss_gib={:.4}",
            rss as f64 / 1048576.0,
            high as f64 / 1048576.0
        ),
        _ => println!("DATE27_STRIP_MEMORY checkpoint={label} rss=UNAVAILABLE"),
    }
    io::stdout().flush().unwrap();
}

fn print_preflight() {
    assert_eq!(MACROS_PER_AXIS * MACRO_DIM, DIM);
    assert_eq!(CT_PER_MACRO * SLOT_COUNT, MACRO_DIM * MACRO_DIM);
    let total_ct = MACROS_PER_AXIS * CT_PER_MACRO;
    assert_eq!(total_ct, 256);
    assert_eq!(REQUIRED_LEVELS + GUARD_LIMBS, 28);
    let ciphertext_bytes = total_ct as u128 * 2 * 65536 * 28 * 8;
    let gib = 1024_f64.powi(3);
    println!("DATE27_STRIP_GLOBAL_DIM={DIM}");
    println!("DATE27_STRIP_SHAPE={}x{}", STRIP_HEIGHT, DIM);
    println!("DATE27_STRIP_MACRO_COUNT={MACROS_PER_AXIS}");
    println!("DATE27_STRIP_CIPHERTEXT_COUNT={total_ct}");
    println!(
        "DATE27_STRIP_RAW_INITIAL_CT_PAYLOAD_GIB={:.6}",
        ciphertext_bytes as f64 / gib
    );
    println!("DATE27_STRIP_PROFILE_TOP_LIMBS=28");
    println!("DATE27_STRIP_ROW_DEPTH=13");
    println!("DATE27_STRIP_FULL_FFT_DEPTH=26");
    println!("DATE27_STRIP_SECURITY_BEARING=false");
    println!("DATE27_STRIP_PREFLIGHT=PASS_GEOMETRY_ONLY");
}

fn global_twiddles(macro_offset: usize, tile_col: usize, span_macros: usize) -> Vec<Complex64> {
    assert!(span_macros == 8 || span_macros == 4 || span_macros == 2);
    assert!(macro_offset < span_macros / 2);
    assert!(tile_col < 16);
    let mut twiddles = Vec::with_capacity(SLOT_COUNT);
    for slot in 0..SLOT_COUNT {
        let local_col = slot % TILE_DIM;
        let j = macro_offset * MACRO_DIM + tile_col * TILE_DIM + local_col;
        let angle = -2.0 * PI * j as f64 / (span_macros * MACRO_DIM) as f64;
        twiddles.push(Complex64::new(angle.cos(), angle.sin()));
    }
    twiddles
}

/// The same packed DIF macro-row butterfly as R24.1, generalized from the
/// 4K 4x4 grid to one 8-macrotile strip. Reuses prepared public twiddles
/// across all equivalent macrotile pairs. Retains ciphertext output for the
/// next encrypted stage; no encrypt/decrypt cycle between stages.
fn execute_global_row_stage(
    macrotiles: &mut [Option<Vec<RnsCkksCiphertext>>],
    span_macros: usize,
    evaluator: &RnsCkksEvaluator<'_>,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    inter_ct_parallelism: usize,
) {
    assert_eq!(macrotiles.len(), MACROS_PER_AXIS);
    assert!(inter_ct_parallelism > 0);
    let half = span_macros / 2;
    let sample = &macrotiles
        .iter()
        .find_map(Option::as_ref)
        .expect("encrypted macrotiles missing")[0];
    let level = sample.level();
    let stage_plan = RnsNttPlan::new(chain.level(level).moduli().to_vec(), embedding.degree());
    let prepared_plan = PreparedRnsNttPlan::new(&stage_plan);
    let mut prepared: Vec<Vec<PreparedComplexSlotsCp>> = Vec::with_capacity(half);
    for macro_offset in 0..half {
        let mut by_tile_col = Vec::with_capacity(16);
        for tile_col in 0..16 {
            by_tile_col.push(PreparedComplexSlotsCp::new(
                sample,
                &global_twiddles(macro_offset, tile_col, span_macros),
                embedding,
                chain,
                &stage_plan,
            ));
        }
        prepared.push(by_tile_col);
    }
    for group_start in (0..MACROS_PER_AXIS).step_by(span_macros) {
        for (macro_offset, by_tile_col) in prepared.iter().enumerate() {
            let upper_index = group_start + macro_offset;
            let lower_index = upper_index + half;
            let upper = macrotiles[upper_index]
                .take()
                .expect("upper macrotile missing");
            let lower = macrotiles[lower_index]
                .take()
                .expect("lower macrotile missing");
            assert_eq!(upper.len(), CT_PER_MACRO);
            assert_eq!(lower.len(), CT_PER_MACRO);
            let mut upper_out: Vec<Option<RnsCkksCiphertext>> =
                (0..CT_PER_MACRO).map(|_| None).collect();
            let mut lower_out: Vec<Option<RnsCkksCiphertext>> =
                (0..CT_PER_MACRO).map(|_| None).collect();
            for batch_start in (0..CT_PER_MACRO).step_by(inter_ct_parallelism) {
                let batch_end = (batch_start + inter_ct_parallelism).min(CT_PER_MACRO);
                let batch = std::thread::scope(|scope| {
                    let mut workers = Vec::new();
                    for ct_index in batch_start..batch_end {
                        let a = &upper[ct_index];
                        let b = &lower[ct_index];
                        let twiddle = &by_tile_col[ct_index % 16];
                        workers.push((
                            ct_index,
                            scope.spawn(|| {
                                packed_fft_dif_butterfly_cp_prepared(
                                    evaluator,
                                    a,
                                    b,
                                    twiddle,
                                    chain,
                                    &prepared_plan,
                                )
                            }),
                        ));
                    }
                    workers
                        .into_iter()
                        .map(|(idx, worker)| {
                            let (a, b) = worker.join().expect("macro-row butterfly failed");
                            (idx, a, b)
                        })
                        .collect::<Vec<_>>()
                });
                for (idx, a, b) in batch {
                    upper_out[idx] = Some(a);
                    lower_out[idx] = Some(b);
                }
            }
            macrotiles[upper_index] = Some(upper_out.into_iter().map(|v| v.unwrap()).collect());
            macrotiles[lower_index] = Some(lower_out.into_iter().map(|v| v.unwrap()).collect());
        }
    }
}

fn advance_cleartext_row_stage(oracle: &mut [Complex64], stage: usize) {
    assert!((1..=13).contains(&stage));
    assert_eq!(oracle.len(), STRIP_HEIGHT * DIM);
    let span = DIM >> (stage - 1);
    let half = span / 2;
    let roots: Vec<Complex64> = (0..half)
        .map(|j| {
            let angle = -2.0 * PI * j as f64 / span as f64;
            Complex64::new(angle.cos(), angle.sin())
        })
        .collect();
    for y in 0..STRIP_HEIGHT {
        for group_start in (0..DIM).step_by(span) {
            let base = y * DIM + group_start;
            for j in 0..half {
                let a = oracle[base + j];
                let b = oracle[base + half + j];
                oracle[base + j] = a + b;
                oracle[base + half + j] = (a - b) * roots[j];
            }
        }
    }
}

fn load_strip_oracle(image: &NpyInput, macro_row: usize) -> Vec<Complex64> {
    let mut oracle = Vec::with_capacity(STRIP_HEIGHT * DIM);
    for y in macro_row * MACRO_DIM..(macro_row + 1) * MACRO_DIM {
        for x in 0..DIM {
            oracle.push(Complex64::new(image.at(y, x), 0.0));
        }
    }
    oracle
}

#[derive(Clone)]
struct Sample {
    slot: usize,
    expected: Complex64,
}

type SampleMap = BTreeMap<(usize, usize, usize), Vec<Sample>>;

/// Each non-comment TSV line: stage TAB global_y TAB global_x TAB real TAB imag.
/// Extracted from the existing cleartext checkpoint_dif.py JSON reference.
fn load_reference_samples(path: Option<&Path>, strip_row: usize) -> SampleMap {
    let mut samples: SampleMap = BTreeMap::new();
    let Some(path) = path else { return samples };
    let input = fs::read_to_string(path).expect("cannot read checkpoint TSV");
    for (line_num, line) in input.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(
            fields.len(),
            5,
            "checkpoint TSV line {} must have 5 fields",
            line_num + 1
        );
        let stage: usize = fields[0].parse().expect("invalid checkpoint stage");
        let y: usize = fields[1].parse().expect("invalid checkpoint row");
        let x: usize = fields[2].parse().expect("invalid checkpoint column");
        let re: f64 = fields[3].parse().expect("invalid checkpoint real part");
        let im: f64 = fields[4]
            .parse()
            .expect("invalid checkpoint imaginary part");
        assert!([1, 3, 13].contains(&stage), "unsupported checkpoint stage");
        assert!(y >= strip_row * MACRO_DIM && y < (strip_row + 1) * MACRO_DIM);
        assert!(x < DIM && re.is_finite() && im.is_finite());
        let local_y = y % MACRO_DIM;
        let local_x = x % MACRO_DIM;
        let macro_col = x / MACRO_DIM;
        let tile_row = local_y / TILE_DIM;
        let row_block = tile_row / TILES_PER_CT;
        let lane = tile_row % TILES_PER_CT;
        let ct_index = row_block * 16 + local_x / TILE_DIM;
        let slot = lane * TILE_ELEMENTS + (local_y % TILE_DIM) * TILE_DIM + local_x % TILE_DIM;
        assert_eq!(
            physical_position(strip_row, macro_col, ct_index, slot),
            (y, x)
        );
        samples
            .entry((stage, macro_col, ct_index))
            .or_default()
            .push(Sample {
                slot,
                expected: Complex64::new(re, im),
            });
    }
    assert!(
        !samples.is_empty(),
        "checkpoint TSV has no samples for selected strip"
    );
    samples
}

fn validate_checkpoint(
    stage: usize,
    macrotiles: &[Option<Vec<RnsCkksCiphertext>>],
    oracle: &[Complex64],
    macro_row: usize,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
    cts_per_macro: usize,
    reference_samples: &SampleMap,
) {
    assert!(stage == 1 || stage == 3 || stage == 13);
    assert_eq!(macrotiles.len(), MACROS_PER_AXIS);
    let full_check = cts_per_macro == CT_PER_MACRO;
    assert!(
        reference_samples.is_empty() || full_check,
        "--checkpoint-tsv requires --check-cts-per-macro 32"
    );
    let mut accuracy = Accuracy::default();
    let mut sample_accuracy = Accuracy::default();
    let started = Instant::now();
    for (macro_col, opt_cts) in macrotiles.iter().enumerate() {
        let cts = opt_cts
            .as_ref()
            .expect("missing macrotile checkpoint state");
        for k in 0..cts_per_macro {
            let ct_index = k * CT_PER_MACRO / cts_per_macro;
            let ct = &cts[ct_index];
            assert_eq!(ct.level(), stage, "unexpected CKKS checkpoint level");
            let slots = decrypt_slots(ct, secret, embedding);
            assert_eq!(slots.len(), SLOT_COUNT);
            for (slot, value) in slots.iter().enumerate() {
                let (global_y, global_x) = physical_position(macro_row, macro_col, ct_index, slot);
                let expected = oracle[(global_y % MACRO_DIM) * DIM + global_x];
                accuracy.add(*value, expected);
            }
            if let Some(samples) = reference_samples.get(&(stage, macro_col, ct_index)) {
                for sample in samples {
                    sample_accuracy.add(slots[sample.slot], sample.expected);
                }
            }
        }
        if full_check {
            println!(
                "DATE27_STRIP_VALIDATE_PROGRESS stage={stage} macrotiles={}/8",
                macro_col + 1
            );
            io::stdout().flush().unwrap();
        }
    }
    let rel_l2 = accuracy.rel_l2();
    let pass = rel_l2.is_finite() && accuracy.max_abs.is_finite() && rel_l2 <= TOLERANCE;
    println!(
        "DATE27_STRIP_CHECKPOINT stage={stage} level={stage} limbs={} slots_checked={} rel_l2={rel_l2:.12e} max_abs={:.12e} validation={} elapsed_ms={}",
        macrotiles[0].as_ref().unwrap()[0].basis().len(), accuracy.slots_checked,
        accuracy.max_abs, if full_check {"FULL_STRIP"} else {"SAMPLED_CIPHERTEXTS"},
        started.elapsed().as_millis()
    );
    println!(
        "DATE27_STRIP_CHECKPOINT_STATUS stage={stage} status={}",
        if pass { "PASS" } else { "FAIL" }
    );
    if !reference_samples.is_empty() {
        let expected_count = reference_samples
            .iter()
            .filter(|((s, _, _), _)| *s == stage)
            .map(|(_, v)| v.len())
            .sum::<usize>();
        assert_eq!(
            sample_accuracy.slots_checked, expected_count,
            "not all checkpoint TSV samples checked"
        );
        let sample_rel = sample_accuracy.rel_l2();
        let sample_pass = sample_rel.is_finite()
            && sample_accuracy.max_abs.is_finite()
            && sample_rel <= TOLERANCE;
        println!(
            "DATE27_STRIP_SAVED_REFERENCE stage={stage} samples={} rel_l2={sample_rel:.12e} max_abs={:.12e} status={}",
            sample_accuracy.slots_checked, sample_accuracy.max_abs,
            if sample_pass { "PASS" } else { "FAIL" }
        );
        assert!(
            sample_pass,
            "encrypted outputs differ from saved cleartext checkpoint"
        );
    }
    io::stdout().flush().unwrap();
    assert!(pass, "encrypted row-strip checkpoint {stage} failed");
}

fn block_sizes(limb_count: usize) -> Vec<usize> {
    vec![1; limb_count]
}

fn generate_row_keys(chain: &ModulusChain, degree: usize, secret: &[i8]) -> RnsCkksEvaluationKeys {
    let mut keys = RnsCkksEvaluationKeys::new();
    let mut rotation = TILE_DIM / 2;
    let mut count = 0usize;
    for level in 7..=12 {
        let basis = chain.level(level).clone();
        let layout = RnsGadgetLayout::new(basis.clone(), block_sizes(basis.len()));
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
        let left_exponent = rotation_exponent_left(degree, rotation);
        let right_exponent = rotation_exponent_right(degree, rotation);
        let mut level_keys = RnsCkksLevelKeys::new(level, basis);
        println!("DATE27_STRIP_KEYGEN_BEGIN level={level} rotation={rotation}");
        io::stdout().flush().unwrap();
        if left_exponent == right_exponent {
            let mut rng = ChaCha20Rng::seed_from_u64(0x5232_304b_4b45_5900 ^ ((level as u64) << 8));
            let key = RnsGaloisKey::generate_with_ntt_rng(
                RnsKeygenConfig {
                    degree,
                    plaintext_modulus: 2,
                    noise_bound: 0,
                    layout: layout.clone(),
                    plan: &plan,
                },
                secret,
                left_exponent,
                &mut rng,
            );
            level_keys.insert_galois_key(key);
            count += 1;
        } else {
            let (left, right) = std::thread::scope(|scope| {
                let left_worker = scope.spawn(|| {
                    let mut rng =
                        ChaCha20Rng::seed_from_u64(0x5232_304b_4b45_5900 ^ ((level as u64) << 8));
                    RnsGaloisKey::generate_with_ntt_rng(
                        RnsKeygenConfig {
                            degree,
                            plaintext_modulus: 2,
                            noise_bound: 0,
                            layout: layout.clone(),
                            plan: &plan,
                        },
                        secret,
                        left_exponent,
                        &mut rng,
                    )
                });
                let right_worker = scope.spawn(|| {
                    let mut rng =
                        ChaCha20Rng::seed_from_u64(0x5232_304b_4b45_5901 ^ ((level as u64) << 8));
                    RnsGaloisKey::generate_with_ntt_rng(
                        RnsKeygenConfig {
                            degree,
                            plaintext_modulus: 2,
                            noise_bound: 0,
                            layout: layout.clone(),
                            plan: &plan,
                        },
                        secret,
                        right_exponent,
                        &mut rng,
                    )
                });
                (
                    left_worker.join().expect("row keygen left failed"),
                    right_worker.join().expect("row keygen right failed"),
                )
            });
            level_keys.insert_galois_key(left);
            level_keys.insert_galois_key(right);
            count += 2;
        }
        keys.insert_level(level_keys);
        println!("DATE27_STRIP_KEYGEN_END level={level}");
        rotation /= 2;
    }
    assert_eq!(rotation, 0);
    println!("DATE27_STRIP_GALOIS_KEY_COUNT={count}");
    keys
}

fn usage() -> ! {
    eprintln!("usage:\n  date27_sar8k_row_strip preflight\n  date27_sar8k_row_strip run <input_8192.npy> [--macro-row 0..7] [--stop-stage 3|13] [--check-cts-per-macro 1..32] [--inter-ct 2] [--local 4] [--checkpoint-tsv <path>]");
    std::process::exit(2);
}

fn parse_number(args: &[String], i: &mut usize) -> usize {
    *i += 1;
    args.get(*i)
        .unwrap_or_else(|| usage())
        .parse()
        .expect("invalid numeric argument")
}

fn original_row_strip_main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() == 1 && args[0] == "preflight" {
        print_preflight();
        return;
    }
    if args.len() < 2 || args[0] != "run" {
        usage();
    }
    print_preflight();
    let mut macro_row = 0;
    let mut stop_stage = 3;
    let mut check_cts_per_macro = 2;
    let mut inter_ct = 2;
    let mut local_parallelism = 4;
    let mut checkpoint_path: Option<&Path> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--macro-row" => macro_row = parse_number(&args, &mut i),
            "--stop-stage" => stop_stage = parse_number(&args, &mut i),
            "--check-cts-per-macro" => check_cts_per_macro = parse_number(&args, &mut i),
            "--inter-ct" => inter_ct = parse_number(&args, &mut i),
            "--local" => local_parallelism = parse_number(&args, &mut i),
            "--checkpoint-tsv" => {
                i += 1;
                checkpoint_path = Some(Path::new(args.get(i).unwrap_or_else(|| usage())));
            }
            _ => usage(),
        }
        i += 1;
    }
    assert!(macro_row < MACROS_PER_AXIS);
    assert!(stop_stage == 3 || stop_stage == 13);
    assert!((1..=CT_PER_MACRO).contains(&check_cts_per_macro));
    assert!(inter_ct > 0 && local_parallelism > 0);
    if checkpoint_path.is_some() {
        assert_eq!(
            check_cts_per_macro, 32,
            "saved reference needs full validation"
        );
    }
    if stop_stage == 13 {
        assert_eq!(
            std::env::var("FHE_PACKED_1K_NTT_RESIDENT").as_deref(),
            Ok("1"),
            "R24.1 optimized row path requires FHE_PACKED_1K_NTT_RESIDENT=1"
        );
    }
    let samples = load_reference_samples(checkpoint_path, macro_row);
    let total_start = Instant::now();
    let image = NpyInput::open(Path::new(&args[1]));
    image.validate();
    let profile = research_profile_65536_for_levels(REQUIRED_LEVELS, GUARD_LIMBS);
    let chain = profile.modulus_chain();
    let embedding = CkksCanonicalEmbedding::new(profile.degree());
    let top_plan = profile.rns_ntt_plan();
    let scale = profile.initial_scale();
    assert_eq!(chain.top().len(), 28);
    assert_eq!(chain.level(13).len(), 15);
    println!("DATE27_STRIP_RUN row={macro_row} stop_stage={stop_stage} check_cts_per_macro={check_cts_per_macro} inter_ct={inter_ct} local={local_parallelism}");
    io::stdout().flush().unwrap();

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5232_304b_5345_4352);
    let mut secret: Vec<i8> = (0..embedding.degree())
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&v| v == 0) {
        secret[0] = 1;
    }

    let oracle_start = Instant::now();
    let mut oracle = load_strip_oracle(&image, macro_row);
    println!(
        "DATE27_STRIP_ORACLE_LOADED_MS={}",
        oracle_start.elapsed().as_millis()
    );
    let encrypt_start = Instant::now();
    let mut macrotiles: Vec<Option<Vec<RnsCkksCiphertext>>> = Vec::with_capacity(MACROS_PER_AXIS);
    for macro_col in 0..MACROS_PER_AXIS {
        let mut cts = Vec::with_capacity(CT_PER_MACRO);
        for ct in 0..CT_PER_MACRO {
            let slots = fill_slots(&image, macro_row, macro_col, ct);
            cts.push(encrypt_slots(
                &slots,
                macro_row * MACROS_PER_AXIS + macro_col,
                ct,
                &secret,
                &embedding,
                &chain,
                &top_plan,
                scale,
            ));
        }
        macrotiles.push(Some(cts));
        println!(
            "DATE27_STRIP_ENCRYPT_PROGRESS macrotiles={}/8 ciphertexts={}/256",
            macro_col + 1,
            (macro_col + 1) * CT_PER_MACRO
        );
        io::stdout().flush().unwrap();
        log_memory("encrypt");
    }
    println!(
        "DATE27_STRIP_ENCRYPT_MS={}",
        encrypt_start.elapsed().as_millis()
    );
    let keys_start = Instant::now();
    let keys = if stop_stage == 13 {
        generate_row_keys(&chain, embedding.degree(), &secret)
    } else {
        RnsCkksEvaluationKeys::new()
    };
    println!(
        "DATE27_STRIP_KEYGEN_MS={}",
        keys_start.elapsed().as_millis()
    );
    log_memory("keys");
    let evaluator = RnsCkksEvaluator::new(&chain, &keys);
    let fft_start = Instant::now();

    for (stage, span_macros) in [(1usize, 8usize), (2, 4), (3, 2)] {
        let stage_start = Instant::now();
        execute_global_row_stage(
            &mut macrotiles,
            span_macros,
            &evaluator,
            &embedding,
            &chain,
            inter_ct,
        );
        advance_cleartext_row_stage(&mut oracle, stage);
        let cts = macrotiles[0].as_ref().unwrap();
        assert_eq!(cts[0].level(), stage);
        println!(
            "DATE27_STRIP_STAGE stage={stage} span_macros={span_macros} limbs={} elapsed_ms={}",
            cts[0].basis().len(),
            stage_start.elapsed().as_millis()
        );
        io::stdout().flush().unwrap();
        log_memory("global_stage");
        if stage == 1 || stage == 3 {
            validate_checkpoint(
                stage,
                &macrotiles,
                &oracle,
                macro_row,
                &secret,
                &embedding,
                check_cts_per_macro,
                &samples,
            );
        }
    }
    if stop_stage == 13 {
        let mut cached = PackedRowPreparationContext::new();
        for macro_col in 0..MACROS_PER_AXIS {
            let start = Instant::now();
            let input = macrotiles[macro_col].take().expect("row macrotile missing");
            let output = execute_packed_1k_fft_rows_cp_cached(
                input,
                &evaluator,
                &embedding,
                &chain,
                inter_ct,
                local_parallelism,
                &mut cached,
            );
            assert_eq!(output.len(), CT_PER_MACRO);
            assert_eq!(output[0].level(), 13);
            macrotiles[macro_col] = Some(output);
            println!(
                "DATE27_STRIP_LOCAL_PROGRESS macrotiles={}/8 stage=13 limbs={} elapsed_ms={}",
                macro_col + 1,
                macrotiles[macro_col].as_ref().unwrap()[0].basis().len(),
                start.elapsed().as_millis()
            );
            io::stdout().flush().unwrap();
            log_memory("local_macro");
        }
        drop(cached);
        log_memory("row_cache_released");
        for stage in 4..=13 {
            advance_cleartext_row_stage(&mut oracle, stage);
        }
        validate_checkpoint(
            13,
            &macrotiles,
            &oracle,
            macro_row,
            &secret,
            &embedding,
            check_cts_per_macro,
            &samples,
        );
    }
    let level = macrotiles[0].as_ref().unwrap()[0].level();
    println!("DATE27_STRIP_FFT_MS={}", fft_start.elapsed().as_millis());
    println!("DATE27_STRIP_FINAL_LEVEL={level}");
    println!(
        "DATE27_STRIP_FINAL_LIMBS={}",
        macrotiles[0].as_ref().unwrap()[0].basis().len()
    );
    println!("DATE27_STRIP_IN_MEMORY_CIPHERTEXTS_RETAINED=256");
    println!("DATE27_STRIP_CIPHERTEXT_DISK_PERSISTENCE=NOT_IMPLEMENTED");
    println!("DATE27_STRIP_FULL_8K_FFT_STATUS=NOT_EXECUTED");
    println!(
        "DATE27_STRIP_TOTAL_MS={}",
        total_start.elapsed().as_millis()
    );
    log_memory("final");
    println!("DATE27_STRIP_STATUS=PASS");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn macrotile_slot_geometry() {
        let mut seen = vec![false; MACRO_DIM * MACRO_DIM];
        for ct in 0..CT_PER_MACRO {
            for slot in 0..SLOT_COUNT {
                let (y, x) = physical_position(0, 0, ct, slot);
                let idx = y * MACRO_DIM + x;
                assert!(!seen[idx]);
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|x| *x));
    }
    #[test]
    fn inter_macrotile_pairs_cover_each_column_once() {
        for span in [8usize, 4, 2] {
            let mut visited = [false; MACROS_PER_AXIS];
            for group in (0..MACROS_PER_AXIS).step_by(span) {
                for offset in 0..span / 2 {
                    let i = group + offset;
                    let j = i + span / 2;
                    assert!(!visited[i] && !visited[j]);
                    visited[i] = true;
                    visited[j] = true;
                }
            }
            assert!(visited.iter().all(|v| *v));
        }
    }
    #[test]
    fn global_twiddle_is_dif() {
        for span in [8usize, 4, 2] {
            for offset in 0..span / 2 {
                for tile_col in [0, 7, 15] {
                    let t = global_twiddles(offset, tile_col, span);
                    for slot in [0, 1, 4095, 16384, 32767] {
                        let x = offset * MACRO_DIM + tile_col * TILE_DIM + slot % TILE_DIM;
                        let angle = -2.0 * PI * x as f64 / (span * MACRO_DIM) as f64;
                        let expected = Complex64::new(angle.cos(), angle.sin());
                        assert!((expected - t[slot]).norm() < 1.0e-14);
                    }
                }
            }
        }
    }
}

// Extension experiment: complete encrypted row FFTs for row strips 0 and 4,
// followed by the very first 8K column DIF stage (stage 14).
// The existing single-row `run` mode remains available in this new binary,
// but the validated original date27_sar8k_row_strip binary is untouched.
fn column_pair_twiddles(row_offset: usize, row_block: usize, span_macros: usize) -> Vec<Complex64> {
    assert!(row_offset < span_macros / 2);
    assert!(row_block < 2);
    let span = span_macros * MACRO_DIM;
    let mut twiddles = Vec::with_capacity(SLOT_COUNT);
    for slot in 0..SLOT_COUNT {
        let lane = slot / TILE_ELEMENTS;
        let local_y = (slot % TILE_ELEMENTS) / TILE_DIM;
        let j = row_offset * MACRO_DIM
            + row_block * TILES_PER_CT * TILE_DIM
            + lane * TILE_DIM
            + local_y;
        let angle = -2.0 * PI * (j as f64) / (span as f64);
        twiddles.push(Complex64::new(angle.cos(), angle.sin()));
    }
    twiddles
}

fn encrypted_strip_to_row13<'a>(
    image: &NpyInput,
    macro_row: usize,
    secret: &[i8],
    embedding: &'a CkksCanonicalEmbedding,
    chain: &'a ModulusChain,
    top_plan: &RnsNttPlan,
    scale: f64,
    evaluator: &RnsCkksEvaluator<'a>,
    row_cache: &mut PackedRowPreparationContext<'a>,
    inter_ct: usize,
    local: usize,
    check_cts_per_macro: usize,
) -> (Vec<Option<Vec<RnsCkksCiphertext>>>, Vec<Complex64>) {
    let started = Instant::now();
    let mut oracle = load_strip_oracle(image, macro_row);
    let mut macrotiles = Vec::with_capacity(MACROS_PER_AXIS);
    for macro_col in 0..MACROS_PER_AXIS {
        let mut cts = Vec::with_capacity(CT_PER_MACRO);
        for ct_index in 0..CT_PER_MACRO {
            let slots = fill_slots(image, macro_row, macro_col, ct_index);
            cts.push(encrypt_slots(
                &slots,
                macro_row * MACROS_PER_AXIS + macro_col,
                ct_index,
                secret,
                embedding,
                chain,
                top_plan,
                scale,
            ));
        }
        macrotiles.push(Some(cts));
        println!(
            "DATE27_PAIR_ENCRYPT row={macro_row} macrotiles={}/8",
            macro_col + 1
        );
        io::stdout().flush().unwrap();
    }
    for (stage, span) in [(1usize, 8usize), (2, 4), (3, 2)] {
        execute_global_row_stage(&mut macrotiles, span, evaluator, embedding, chain, inter_ct);
        advance_cleartext_row_stage(&mut oracle, stage);
    }
    for macro_col in 0..MACROS_PER_AXIS {
        let input = macrotiles[macro_col]
            .take()
            .expect("row13 missing macrotile");
        let output = execute_packed_1k_fft_rows_cp_cached(
            input, evaluator, embedding, chain, inter_ct, local, row_cache,
        );
        assert_eq!(output.len(), CT_PER_MACRO);
        assert_eq!(output[0].level(), 13);
        macrotiles[macro_col] = Some(output);
        println!(
            "DATE27_PAIR_LOCAL_ROW row={macro_row} macrotiles={}/8",
            macro_col + 1
        );
        io::stdout().flush().unwrap();
    }
    for stage in 4..=13 {
        advance_cleartext_row_stage(&mut oracle, stage);
    }
    validate_checkpoint(
        13,
        &macrotiles,
        &oracle,
        macro_row,
        secret,
        embedding,
        check_cts_per_macro,
        &SampleMap::new(),
    );
    println!(
        "DATE27_PAIR_ROW13_DONE row={macro_row} elapsed_ms={}",
        started.elapsed().as_millis()
    );
    log_memory("row13_done");
    (macrotiles, oracle)
}

fn execute_stage14_pair(
    row0: &mut [Option<Vec<RnsCkksCiphertext>>],
    row4: &mut [Option<Vec<RnsCkksCiphertext>>],
    evaluator: &RnsCkksEvaluator<'_>,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    inter_ct: usize,
) {
    assert_eq!(row0.len(), MACROS_PER_AXIS);
    assert_eq!(row4.len(), MACROS_PER_AXIS);
    let sample = &row0[0].as_ref().expect("missing row0 macrotile")[0];
    assert_eq!(sample.level(), 13);
    let plan = RnsNttPlan::new(chain.level(13).moduli().to_vec(), embedding.degree());
    let prepared_plan = PreparedRnsNttPlan::new(&plan);
    // Pair (macro-row 0, macro-row 4): row_offset=0, span_macros=8.
    let prepared: Vec<PreparedComplexSlotsCp> = (0..2)
        .map(|row_block| {
            PreparedComplexSlotsCp::new(
                sample,
                &column_pair_twiddles(0, row_block, 8),
                embedding,
                chain,
                &plan,
            )
        })
        .collect();
    for macro_col in 0..MACROS_PER_AXIS {
        let upper = row0[macro_col]
            .take()
            .expect("missing upper column macrotile");
        let lower = row4[macro_col]
            .take()
            .expect("missing lower column macrotile");
        assert_eq!(upper.len(), CT_PER_MACRO);
        assert_eq!(lower.len(), CT_PER_MACRO);
        let mut upper_out: Vec<Option<RnsCkksCiphertext>> =
            (0..CT_PER_MACRO).map(|_| None).collect();
        let mut lower_out: Vec<Option<RnsCkksCiphertext>> =
            (0..CT_PER_MACRO).map(|_| None).collect();
        for batch_start in (0..CT_PER_MACRO).step_by(inter_ct) {
            let batch_end = (batch_start + inter_ct).min(CT_PER_MACRO);
            let batch = std::thread::scope(|scope| {
                let mut workers = Vec::new();
                for ct_index in batch_start..batch_end {
                    let a = &upper[ct_index];
                    let b = &lower[ct_index];
                    let twiddle = &prepared[ct_index / 16];
                    workers.push((
                        ct_index,
                        scope.spawn(|| {
                            packed_fft_dif_butterfly_cp_prepared(
                                evaluator,
                                a,
                                b,
                                twiddle,
                                chain,
                                &prepared_plan,
                            )
                        }),
                    ));
                }
                workers
                    .into_iter()
                    .map(|(idx, worker)| {
                        let (u, l) = worker.join().expect("stage14 butterfly failed");
                        (idx, u, l)
                    })
                    .collect::<Vec<_>>()
            });
            for (idx, a, b) in batch {
                upper_out[idx] = Some(a);
                lower_out[idx] = Some(b);
            }
        }
        row0[macro_col] = Some(upper_out.into_iter().map(|x| x.unwrap()).collect());
        row4[macro_col] = Some(lower_out.into_iter().map(|x| x.unwrap()).collect());
        println!(
            "DATE27_PAIR_STAGE14_PROGRESS macrotiles={}/8",
            macro_col + 1
        );
        io::stdout().flush().unwrap();
        log_memory("stage14_column_pair");
    }
}

fn clear_stage14_pair(upper: &mut [Complex64], lower: &mut [Complex64]) {
    assert_eq!(upper.len(), STRIP_HEIGHT * DIM);
    assert_eq!(lower.len(), STRIP_HEIGHT * DIM);
    let roots: Vec<Complex64> = (0..STRIP_HEIGHT)
        .map(|j| {
            let angle = -2.0 * PI * j as f64 / DIM as f64;
            Complex64::new(angle.cos(), angle.sin())
        })
        .collect();
    for y in 0..STRIP_HEIGHT {
        for x in 0..DIM {
            let index = y * DIM + x;
            let a = upper[index];
            let b = lower[index];
            upper[index] = a + b;
            lower[index] = (a - b) * roots[y];
        }
    }
}

fn validate_stage14_pair(
    row0: &[Option<Vec<RnsCkksCiphertext>>],
    row4: &[Option<Vec<RnsCkksCiphertext>>],
    oracle0: &[Complex64],
    oracle4: &[Complex64],
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
    check_cts_per_macro: usize,
) {
    let mut accuracy = Accuracy::default();
    for (macro_row, macrotile_row, oracle) in [(0usize, row0, oracle0), (4usize, row4, oracle4)] {
        for macro_col in 0..MACROS_PER_AXIS {
            let cts = macrotile_row[macro_col]
                .as_ref()
                .expect("missing stage14 CTs");
            for k in 0..check_cts_per_macro {
                let ct_index = k * CT_PER_MACRO / check_cts_per_macro;
                let ct = &cts[ct_index];
                assert_eq!(ct.level(), 14);
                assert_eq!(ct.basis().len(), 14);
                let values = decrypt_slots(ct, secret, embedding);
                for (slot, value) in values.iter().enumerate() {
                    let (y, x) = physical_position(macro_row, macro_col, ct_index, slot);
                    let expected = oracle[(y % MACRO_DIM) * DIM + x];
                    accuracy.add(*value, expected);
                }
            }
        }
    }
    let relative_error = accuracy.rel_l2();
    let pass =
        relative_error.is_finite() && accuracy.max_abs.is_finite() && relative_error <= TOLERANCE;
    println!(
        "DATE27_PAIR_STAGE14_CHECKPOINT level=14 limbs=14 slots_checked={} rel_l2={:.12e} max_abs={:.12e} status={}",
        accuracy.slots_checked, relative_error, accuracy.max_abs,
        if pass { "PASS" } else { "FAIL" }
    );
    assert!(pass, "stage14 pair numerical validation failed");
}

fn stage14_pair_main(args: &[String]) {
    // Only row strips 0 and 4 are processed. This is NOT a complete column stage.
    if args.len() < 2 {
        eprintln!("usage: date27_sar8k_four_strips pair-stage14 <input_8192.npy> [--check-cts-per-macro 1..32] [--inter-ct N] [--local N]");
        std::process::exit(2);
    }
    let mut check_cts_per_macro = 1usize;
    let mut inter_ct = 2usize;
    let mut local = 4usize;
    let mut idx = 2usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--check-cts-per-macro" => check_cts_per_macro = parse_number(args, &mut idx),
            "--inter-ct" => inter_ct = parse_number(args, &mut idx),
            "--local" => local = parse_number(args, &mut idx),
            unknown => panic!("unknown option: {unknown}"),
        }
        idx += 1;
    }
    assert!((1..=CT_PER_MACRO).contains(&check_cts_per_macro));
    assert!(inter_ct > 0 && local > 0);
    assert_eq!(
        std::env::var("FHE_PACKED_1K_NTT_RESIDENT").as_deref(),
        Ok("1"),
        "validated R24.1 row path requires NTT resident mode"
    );
    println!("DATE27_PAIR_SCOPE=ROW_STRIPS_0_AND_4_ONLY");
    println!("DATE27_PAIR_FULL_8K_FFT_STATUS=NOT_EXECUTED");
    println!("DATE27_PAIR_SECURITY_BEARING=false");
    print_preflight();
    io::stdout().flush().unwrap();
    let started = Instant::now();
    let image = NpyInput::open(Path::new(&args[1]));
    image.validate();
    let profile = research_profile_65536_for_levels(REQUIRED_LEVELS, GUARD_LIMBS);
    let chain = profile.modulus_chain();
    let embedding = CkksCanonicalEmbedding::new(profile.degree());
    let top_plan = profile.rns_ntt_plan();
    let scale = profile.initial_scale();
    assert_eq!(chain.top().len(), 28);
    assert_eq!(chain.level(14).len(), 14);
    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5232_304b_5345_4352);
    let mut secret: Vec<i8> = (0..embedding.degree())
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&v| v == 0) {
        secret[0] = 1;
    }
    // One secret key, one evaluation-key inventory for both strips.
    let keys = generate_row_keys(&chain, embedding.degree(), &secret);
    let evaluator = RnsCkksEvaluator::new(&chain, &keys);
    log_memory("keys");
    let mut cached_rows = PackedRowPreparationContext::new();
    let (mut row0, mut oracle0) = encrypted_strip_to_row13(
        &image,
        0,
        &secret,
        &embedding,
        &chain,
        &top_plan,
        scale,
        &evaluator,
        &mut cached_rows,
        inter_ct,
        local,
        check_cts_per_macro,
    );
    // Check available RAM before allocating a second strip.
    let available_kib = fs::read_to_string("/proc/meminfo").ok().and_then(|info| {
        info.lines()
            .find(|line| line.starts_with("MemAvailable:"))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u64>().ok())
    });
    if let Some(kib) = available_kib {
        println!(
            "DATE27_PAIR_MEM_AVAILABLE_GIB={:.3}",
            kib as f64 / 1048576.0
        );
        assert!(
            kib > 32 * 1024 * 1024,
            "requires more than 32 GiB available before second strip"
        );
    }
    let (mut row4, mut oracle4) = encrypted_strip_to_row13(
        &image,
        4,
        &secret,
        &embedding,
        &chain,
        &top_plan,
        scale,
        &evaluator,
        &mut cached_rows,
        inter_ct,
        local,
        check_cts_per_macro,
    );
    drop(cached_rows);
    log_memory("row_cache_released");
    let stage_start = Instant::now();
    execute_stage14_pair(
        &mut row0, &mut row4, &evaluator, &embedding, &chain, inter_ct,
    );
    clear_stage14_pair(&mut oracle0, &mut oracle4);
    println!(
        "DATE27_PAIR_STAGE14_EXEC_MS={}",
        stage_start.elapsed().as_millis()
    );
    validate_stage14_pair(
        &row0,
        &row4,
        &oracle0,
        &oracle4,
        &secret,
        &embedding,
        check_cts_per_macro,
    );
    println!("DATE27_PAIR_FINAL_LEVEL=14");
    println!("DATE27_PAIR_FINAL_LIMBS=14");
    println!("DATE27_PAIR_CIPHERTEXTS_RETAINED=512");
    println!("DATE27_PAIR_TOTAL_MS={}", started.elapsed().as_millis());
    log_memory("final");
    println!("DATE27_PAIR_STATUS=PASS");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("preflight") => {
            print_preflight();
            for span in [8usize, 4, 2] {
                assert_eq!(eight_expected_pairs(span).len(), 4);
            }
            println!("DATE27_EIGHT_PREFLIGHT=PASS_GEOMETRY_AND_PAIR_SCHEDULE_ONLY");
            println!("DATE27_EIGHT_TARGET_CIPHERTEXTS=2048");
            println!("DATE27_EIGHT_TARGET_LEVEL=16");
            println!("DATE27_EIGHT_TARGET_LIMBS=12");
            println!("DATE27_EIGHT_PEAK_RSS_CAP_GIB=90");
            println!("DATE27_EIGHT_SECURITY_BEARING=false");
            assert_eq!(full_column_key_schedule().len(), 9);
            assert_eq!(full_column_key_schedule()[0], (17, 16384));
            assert_eq!(full_column_key_schedule()[8], (25, 64));
            println!("DATE27_FULL_PREFLIGHT=PASS_GEOMETRY_KEY_AND_COLUMN_SCHEDULE_ONLY");
            println!("DATE27_FULL_TARGET_LEVEL=26");
            println!("DATE27_FULL_TARGET_LIMBS=2");
            println!("DATE27_FULL_TARGET_CIPHERTEXTS=2048");
            println!("DATE27_FULL_SECURITY_BEARING=false");
        }
        Some("eight-stage16") => eight_stage16_main(&args),
        Some("full-stage26") => full_stage26_main(&args),
        Some("four-stage15") => four_strip_stage15_main(&args),
        Some("pair-stage14") => stage14_pair_main(&args),
        _ => original_row_strip_main(),
    }
}

#[cfg(test)]
mod stage14_pair_tests {
    use super::*;
    #[test]
    fn column_twiddle_matches_scalar_dif() {
        for block in 0..2 {
            let twiddles = column_pair_twiddles(0, block, 8);
            for slot in [0, 1, 4095, 8192, 32767] {
                let lane = slot / TILE_ELEMENTS;
                let j = block * TILES_PER_CT * TILE_DIM
                    + lane * TILE_DIM
                    + (slot % TILE_ELEMENTS) / TILE_DIM;
                let angle = -2.0 * PI * (j as f64) / (DIM as f64);
                assert!((twiddles[slot] - Complex64::new(angle.cos(), angle.sin())).norm() < 1e-14);
            }
        }
    }
    #[test]
    fn cleartext_stage14_pair_is_dif() {
        let mut a = vec![Complex64::new(1.0, 0.0); STRIP_HEIGHT * DIM];
        let mut b = vec![Complex64::new(0.25, 0.0); STRIP_HEIGHT * DIM];
        clear_stage14_pair(&mut a, &mut b);
        for y in [0, 17, 510, 1023] {
            let j = y * DIM;
            assert!((a[j] - Complex64::new(1.25, 0.0)).norm() < 1e-12);
            let theta = -2.0 * PI * (y as f64) / (DIM as f64);
            let expected = Complex64::new(theta.cos(), theta.sin()) * 0.75;
            assert!((b[j] - expected).norm() < 1e-12);
        }
    }
}

// DATE 2027 extension: four row strips, first two global column stages.
// Separate experimental binary; does not alter the validated R24.1 FFT path.
// Row-strip physical packing, RNS key switching and butterfly primitives are
// inherited unchanged from the tested two-strip prototype above.

fn check_four_strip_memory(label: &str, minimum_available_gib: u64) {
    let available = fs::read_to_string("/proc/meminfo").ok().and_then(|info| {
        info.lines()
            .find(|line| line.starts_with("MemAvailable:"))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
    });
    let available = available.expect("Linux MemAvailable unavailable: refusing allocation");
    let rss = rss_kib("VmRSS").expect("Linux VmRSS unavailable: refusing allocation");
    println!(
        "DATE27_FOUR_MEM_GATE checkpoint={label} available_gib={:.4} rss_gib={:.4} min_available_gib={minimum_available_gib}",
        available as f64 / 1048576.0, rss as f64 / 1048576.0
    );
    io::stdout().flush().unwrap();
    assert!(
        available > minimum_available_gib * 1048576,
        "insufficient available memory before {label}"
    );
    assert!(
        rss < 90 * 1048576,
        "exceeded research process RSS cap of 90 GiB"
    );
}

// 'span_macros=8,row_offset=2' handles macro-rows (2,6), distinct from
// (0,4), because the global stage-14 DIF twiddle depends on row index.
// Stage 15 uses 'span_macros=4,row_offset=0' in both groups (0,2),(4,6).
fn execute_column_pair_general(
    upper_row: &mut [Option<Vec<RnsCkksCiphertext>>],
    lower_row: &mut [Option<Vec<RnsCkksCiphertext>>],
    span_macros: usize,
    row_offset: usize,
    input_level: usize,
    evaluator: &RnsCkksEvaluator<'_>,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    inter_ct: usize,
) {
    assert_eq!(upper_row.len(), MACROS_PER_AXIS);
    assert_eq!(lower_row.len(), MACROS_PER_AXIS);
    assert!([8, 4, 2].contains(&span_macros));
    assert!(row_offset < span_macros / 2);
    assert!(inter_ct > 0 && inter_ct <= CT_PER_MACRO);
    let sample = &upper_row[0].as_ref().expect("missing upper macrotile")[0];
    assert_eq!(sample.level(), input_level);
    let plan = RnsNttPlan::new(
        chain.level(input_level).moduli().to_vec(),
        embedding.degree(),
    );
    let prepared_plan = PreparedRnsNttPlan::new(&plan);
    let twiddles: Vec<PreparedComplexSlotsCp> = (0..2)
        .map(|row_block| {
            PreparedComplexSlotsCp::new(
                sample,
                &column_pair_twiddles(row_offset, row_block, span_macros),
                embedding,
                chain,
                &plan,
            )
        })
        .collect();
    for macro_col in 0..MACROS_PER_AXIS {
        let upper = upper_row[macro_col]
            .take()
            .expect("missing upper stage input");
        let lower = lower_row[macro_col]
            .take()
            .expect("missing lower stage input");
        assert_eq!(upper.len(), CT_PER_MACRO);
        assert_eq!(lower.len(), CT_PER_MACRO);
        let mut upper_out: Vec<Option<RnsCkksCiphertext>> =
            (0..CT_PER_MACRO).map(|_| None).collect();
        let mut lower_out: Vec<Option<RnsCkksCiphertext>> =
            (0..CT_PER_MACRO).map(|_| None).collect();
        for batch_start in (0..CT_PER_MACRO).step_by(inter_ct) {
            let batch_end = (batch_start + inter_ct).min(CT_PER_MACRO);
            let computed = std::thread::scope(|scope| {
                let mut workers = Vec::new();
                for index in batch_start..batch_end {
                    let a = &upper[index];
                    let b = &lower[index];
                    let twiddle = &twiddles[index / 16];
                    workers.push((
                        index,
                        scope.spawn(|| {
                            packed_fft_dif_butterfly_cp_prepared(
                                evaluator,
                                a,
                                b,
                                twiddle,
                                chain,
                                &prepared_plan,
                            )
                        }),
                    ));
                }
                workers
                    .into_iter()
                    .map(|(index, worker)| {
                        let (a, b) = worker.join().expect("encrypted column butterfly failed");
                        (index, a, b)
                    })
                    .collect::<Vec<_>>()
            });
            for (index, a, b) in computed {
                assert_eq!(a.level(), input_level + 1);
                assert_eq!(b.level(), input_level + 1);
                upper_out[index] = Some(a);
                lower_out[index] = Some(b);
            }
        }
        upper_row[macro_col] = Some(upper_out.into_iter().map(Option::unwrap).collect());
        lower_row[macro_col] = Some(lower_out.into_iter().map(Option::unwrap).collect());
        println!(
            "DATE27_FOUR_COLUMN_PROGRESS stage={} pair_offset={} span_macros={} macrotiles={}/8",
            input_level + 1,
            row_offset,
            span_macros,
            macro_col + 1
        );
        io::stdout().flush().unwrap();
        if macro_col % 2 == 1 {
            log_memory("four_column_pair");
        }
    }
}

fn clear_column_pair_general(
    upper: &mut [Complex64],
    lower: &mut [Complex64],
    row_offset: usize,
    span_macros: usize,
) {
    assert_eq!(upper.len(), STRIP_HEIGHT * DIM);
    assert_eq!(lower.len(), STRIP_HEIGHT * DIM);
    let span = span_macros * MACRO_DIM;
    let roots: Vec<Complex64> = (0..STRIP_HEIGHT)
        .map(|local_y| {
            let j = row_offset * MACRO_DIM + local_y;
            let theta = -2.0 * PI * (j as f64) / (span as f64);
            Complex64::new(theta.cos(), theta.sin())
        })
        .collect();
    for local_y in 0..STRIP_HEIGHT {
        for x in 0..DIM {
            let index = local_y * DIM + x;
            let a = upper[index];
            let b = lower[index];
            upper[index] = a + b;
            lower[index] = (a - b) * roots[local_y];
        }
    }
}

#[derive(Clone)]
struct ColumnSample {
    stage: usize,
    row: usize,
    col: usize,
    expected: Complex64,
}

fn read_column_samples(path: Option<&Path>) -> Vec<ColumnSample> {
    let Some(path) = path else { return Vec::new() };
    let contents = fs::read_to_string(path).expect("could not read stage14/15 reference TSV");
    let mut samples = Vec::new();
    for (line_number, line) in contents.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(
            parts.len(),
            5,
            "checkpoint TSV line {} must contain 5 fields",
            line_number + 1
        );
        let stage: usize = parts[0].parse().unwrap();
        let row: usize = parts[1].parse().unwrap();
        let col: usize = parts[2].parse().unwrap();
        let re: f64 = parts[3].parse().unwrap();
        let im: f64 = parts[4].parse().unwrap();
        assert!(stage == 14 || stage == 15);
        assert!([0, 2, 4, 6].contains(&(row / MACRO_DIM)));
        assert!(row < DIM && col < DIM && re.is_finite() && im.is_finite());
        samples.push(ColumnSample {
            stage,
            row,
            col,
            expected: Complex64::new(re, im),
        });
    }
    assert!(!samples.is_empty(), "empty checkpoint TSV");
    for stage in [14usize, 15] {
        assert!(
            samples.iter().any(|s| s.stage == stage),
            "missing stage {stage} reference samples"
        );
    }
    samples
}

fn sample_location(y: usize, x: usize) -> (usize, usize, usize) {
    assert!(y < DIM && x < DIM);
    let macro_col = x / MACRO_DIM;
    let local_y = y % MACRO_DIM;
    let local_x = x % MACRO_DIM;
    let tile_row = local_y / TILE_DIM;
    let row_block = tile_row / TILES_PER_CT;
    let lane = tile_row % TILES_PER_CT;
    let ct = row_block * 16 + local_x / TILE_DIM;
    let slot = lane * TILE_ELEMENTS + (local_y % TILE_DIM) * TILE_DIM + (local_x % TILE_DIM);
    assert_eq!(
        physical_position(y / MACRO_DIM, macro_col, ct, slot),
        (y, x)
    );
    (macro_col, ct, slot)
}

fn check_saved_column_samples(
    stage: usize,
    rows: [(usize, &[Option<Vec<RnsCkksCiphertext>>]); 4],
    samples: &[ColumnSample],
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) {
    if samples.is_empty() {
        return;
    }
    let mut grouped = BTreeMap::<(usize, usize, usize), Vec<(usize, Complex64)>>::new();
    for sample in samples.iter().filter(|s| s.stage == stage) {
        let strip_index = sample.row / MACRO_DIM;
        let (macro_col, ct, slot) = sample_location(sample.row, sample.col);
        grouped
            .entry((strip_index, macro_col, ct))
            .or_default()
            .push((slot, sample.expected));
    }
    let mut accuracy = Accuracy::default();
    for ((strip_index, macro_col, ct), entries) in grouped {
        let (_, row) = rows
            .iter()
            .find(|(r, _)| *r == strip_index)
            .expect("checkpoint sample row not held in memory");
        let ciphertext = &row[macro_col].as_ref().unwrap()[ct];
        assert_eq!(ciphertext.level(), stage);
        let decrypted = decrypt_slots(ciphertext, secret, embedding);
        for (slot, expected) in entries {
            accuracy.add(decrypted[slot], expected);
        }
    }
    let rel = accuracy.rel_l2();
    let ok = rel.is_finite() && accuracy.max_abs.is_finite() && rel <= TOLERANCE;
    println!("DATE27_FOUR_SAVED_REFERENCE stage={stage} samples={} rel_l2={rel:.12e} max_abs={:.12e} status={}",
        accuracy.slots_checked, accuracy.max_abs, if ok {"PASS"} else {"FAIL"});
    assert!(
        ok,
        "stage {stage} disagrees with saved independent DIF reference"
    );
}

fn validate_four_column_checkpoint(
    stage: usize,
    rows: [(usize, &[Option<Vec<RnsCkksCiphertext>>], &[Complex64]); 4],
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
    check_cts_per_macro: usize,
) {
    assert!(stage == 14 || stage == 15);
    let mut accuracy = Accuracy::default();
    for (macro_row, row, oracle) in rows {
        for macro_col in 0..MACROS_PER_AXIS {
            for k in 0..check_cts_per_macro {
                let ct_index = k * CT_PER_MACRO / check_cts_per_macro;
                let ct = &row[macro_col].as_ref().unwrap()[ct_index];
                assert_eq!(ct.level(), stage);
                assert_eq!(ct.basis().len(), 28 - stage);
                let slots = decrypt_slots(ct, secret, embedding);
                for (slot, actual) in slots.iter().enumerate() {
                    let (y, x) = physical_position(macro_row, macro_col, ct_index, slot);
                    let expected = oracle[(y % MACRO_DIM) * DIM + x];
                    accuracy.add(*actual, expected);
                }
            }
        }
    }
    let rel = accuracy.rel_l2();
    let ok = rel.is_finite() && accuracy.max_abs.is_finite() && rel <= TOLERANCE;
    println!("DATE27_FOUR_CHECKPOINT stage={stage} level={stage} limbs={} slots_checked={} rel_l2={rel:.12e} max_abs={:.12e} status={}",
        28-stage, accuracy.slots_checked, accuracy.max_abs, if ok {"PASS"} else {"FAIL"});
    assert!(
        ok,
        "stage {stage} encrypted four-strip numerical comparison failed"
    );
}

fn four_strip_stage15_main(args: &[String]) {
    if args.len() < 2 {
        eprintln!("usage: date27_sar8k_four_strips four-stage15 <input_8192.npy> [--check-cts-per-macro 1..32] [--inter-ct N] [--local N] [--checkpoint-tsv PATH]");
        std::process::exit(2);
    }
    let mut check_cts_per_macro = 1usize;
    let mut inter_ct = 2usize;
    let mut local = 4usize;
    let mut checkpoint_path: Option<&Path> = None;
    let mut idx = 2usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--check-cts-per-macro" => check_cts_per_macro = parse_number(args, &mut idx),
            "--inter-ct" => inter_ct = parse_number(args, &mut idx),
            "--local" => local = parse_number(args, &mut idx),
            "--checkpoint-tsv" => {
                idx += 1;
                checkpoint_path = Some(Path::new(args.get(idx).unwrap_or_else(|| usage())));
            }
            unknown => panic!("unknown four-stage15 option: {unknown}"),
        }
        idx += 1;
    }
    assert!([1, 2, 4, 8, 16, 32].contains(&check_cts_per_macro));
    assert!((1..=CT_PER_MACRO).contains(&inter_ct));
    assert!(local > 0 && local <= 16);
    assert_eq!(
        std::env::var("FHE_PACKED_1K_NTT_RESIDENT").as_deref(),
        Ok("1"),
        "R24.1 validated cached row path requires resident NTT mode"
    );
    let samples = read_column_samples(checkpoint_path);
    print_preflight();
    println!("DATE27_FOUR_SCOPE=ROWS_0_2_4_6_STAGE15_ONLY");
    println!("DATE27_FOUR_CHECK_CTS_PER_MACRO={check_cts_per_macro}");
    println!("DATE27_FOUR_SECURITY_BEARING=false");
    println!("DATE27_FOUR_FULL_8K_FFT_STATUS=NOT_EXECUTED");
    io::stdout().flush().unwrap();
    let started = Instant::now();
    let image = NpyInput::open(Path::new(&args[1]));
    image.validate();
    let profile = research_profile_65536_for_levels(REQUIRED_LEVELS, GUARD_LIMBS);
    let chain = profile.modulus_chain();
    let embedding = CkksCanonicalEmbedding::new(profile.degree());
    let top_plan = profile.rns_ntt_plan();
    let scale = profile.initial_scale();
    assert_eq!(chain.top().len(), 28);
    assert_eq!(chain.level(15).len(), 13);
    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5232_304b_5345_4352);
    let mut secret: Vec<i8> = (0..embedding.degree())
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&v| v == 0) {
        secret[0] = 1;
    }
    // All four strips share one secret and level-indexed evaluation keys.
    let keys = generate_row_keys(&chain, embedding.degree(), &secret);
    let evaluator = RnsCkksEvaluator::new(&chain, &keys);
    log_memory("four_keys");
    let mut cached = PackedRowPreparationContext::new();
    check_four_strip_memory("before_first_strip", 48);
    let (mut row0, mut oracle0) = encrypted_strip_to_row13(
        &image,
        0,
        &secret,
        &embedding,
        &chain,
        &top_plan,
        scale,
        &evaluator,
        &mut cached,
        inter_ct,
        local,
        check_cts_per_macro,
    );
    check_four_strip_memory("before_second_strip", 36);
    let (mut row4, mut oracle4) = encrypted_strip_to_row13(
        &image,
        4,
        &secret,
        &embedding,
        &chain,
        &top_plan,
        scale,
        &evaluator,
        &mut cached,
        inter_ct,
        local,
        check_cts_per_macro,
    );
    let start14a = Instant::now();
    execute_column_pair_general(
        &mut row0, &mut row4, 8, 0, 13, &evaluator, &embedding, &chain, inter_ct,
    );
    clear_column_pair_general(&mut oracle0, &mut oracle4, 0, 8);
    println!(
        "DATE27_FOUR_STAGE14_PAIR_DONE rows=0,4 elapsed_ms={}",
        start14a.elapsed().as_millis()
    );
    log_memory("four_stage14_first_pair");
    check_four_strip_memory("before_third_strip", 24);
    let (mut row2, mut oracle2) = encrypted_strip_to_row13(
        &image,
        2,
        &secret,
        &embedding,
        &chain,
        &top_plan,
        scale,
        &evaluator,
        &mut cached,
        inter_ct,
        local,
        check_cts_per_macro,
    );
    check_four_strip_memory("before_fourth_strip", 18);
    let (mut row6, mut oracle6) = encrypted_strip_to_row13(
        &image,
        6,
        &secret,
        &embedding,
        &chain,
        &top_plan,
        scale,
        &evaluator,
        &mut cached,
        inter_ct,
        local,
        check_cts_per_macro,
    );
    drop(cached);
    log_memory("four_row_cache_released");
    let start14b = Instant::now();
    execute_column_pair_general(
        &mut row2, &mut row6, 8, 2, 13, &evaluator, &embedding, &chain, inter_ct,
    );
    clear_column_pair_general(&mut oracle2, &mut oracle6, 2, 8);
    println!(
        "DATE27_FOUR_STAGE14_PAIR_DONE rows=2,6 elapsed_ms={}",
        start14b.elapsed().as_millis()
    );
    validate_four_column_checkpoint(
        14,
        [
            (0, &row0, &oracle0),
            (2, &row2, &oracle2),
            (4, &row4, &oracle4),
            (6, &row6, &oracle6),
        ],
        &secret,
        &embedding,
        check_cts_per_macro,
    );
    check_saved_column_samples(
        14,
        [(0, &row0), (2, &row2), (4, &row4), (6, &row6)],
        &samples,
        &secret,
        &embedding,
    );
    log_memory("four_stage14_done");
    check_four_strip_memory("before_stage15", 12);
    let start15 = Instant::now();
    execute_column_pair_general(
        &mut row0, &mut row2, 4, 0, 14, &evaluator, &embedding, &chain, inter_ct,
    );
    clear_column_pair_general(&mut oracle0, &mut oracle2, 0, 4);
    execute_column_pair_general(
        &mut row4, &mut row6, 4, 0, 14, &evaluator, &embedding, &chain, inter_ct,
    );
    clear_column_pair_general(&mut oracle4, &mut oracle6, 0, 4);
    println!(
        "DATE27_FOUR_STAGE15_EXEC_MS={}",
        start15.elapsed().as_millis()
    );
    validate_four_column_checkpoint(
        15,
        [
            (0, &row0, &oracle0),
            (2, &row2, &oracle2),
            (4, &row4, &oracle4),
            (6, &row6, &oracle6),
        ],
        &secret,
        &embedding,
        check_cts_per_macro,
    );
    check_saved_column_samples(
        15,
        [(0, &row0), (2, &row2), (4, &row4), (6, &row6)],
        &samples,
        &secret,
        &embedding,
    );
    println!("DATE27_FOUR_FINAL_LEVEL=15");
    println!("DATE27_FOUR_FINAL_LIMBS=13");
    println!("DATE27_FOUR_CIPHERTEXTS_RETAINED=1024");
    println!("DATE27_FOUR_TOTAL_MS={}", started.elapsed().as_millis());
    log_memory("four_final");
    println!("DATE27_FOUR_STATUS=PASS");
}

#[cfg(test)]
mod stage15_four_strip_tests {
    use super::*;
    #[test]
    fn stage14_offset_two_uses_correct_global_twiddle() {
        for block in 0..2 {
            let twiddles = column_pair_twiddles(2, block, 8);
            for slot in [0usize, 1, 4095, 8192, 32767] {
                let lane = slot / TILE_ELEMENTS;
                let local_y = block * TILES_PER_CT * TILE_DIM
                    + lane * TILE_DIM
                    + (slot % TILE_ELEMENTS) / TILE_DIM;
                let theta = -2.0 * PI * ((2 * MACRO_DIM + local_y) as f64) / (DIM as f64);
                assert!(
                    (twiddles[slot] - Complex64::new(theta.cos(), theta.sin())).norm() < 1.0e-14
                );
            }
        }
    }
    #[test]
    fn stage15_uses_4k_span() {
        for block in 0..2 {
            let twiddles = column_pair_twiddles(0, block, 4);
            for slot in [0usize, 1, 4095, 8192, 32767] {
                let lane = slot / TILE_ELEMENTS;
                let local_y = block * TILES_PER_CT * TILE_DIM
                    + lane * TILE_DIM
                    + (slot % TILE_ELEMENTS) / TILE_DIM;
                let theta = -2.0 * PI * (local_y as f64) / (4096.0);
                assert!(
                    (twiddles[slot] - Complex64::new(theta.cos(), theta.sin())).norm() < 1.0e-14
                );
            }
        }
    }
    #[test]
    fn sample_location_maps_all_tile_groups() {
        for macro_row in [0, 2, 4, 6] {
            for macro_col in 0..8 {
                for ct in [0usize, 1, 15, 16, 31] {
                    for slot in [0usize, 4095, 4096, 32767] {
                        let (y, x) = physical_position(macro_row, macro_col, ct, slot);
                        assert_eq!(sample_location(y, x), (macro_col, ct, slot));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod four_strip_full_dif_reference_tests {
    use super::*;

    #[test]
    fn four_strip_scalar_stage14_and_15_match_full_column_dif() {
        let mut full = (0..DIM)
            .map(|y| {
                Complex64::new(
                    ((y * 37 + 17) % 8191) as f64 / 8191.0,
                    ((y * 13 + 5) % 4093) as f64 / 4093.0,
                )
            })
            .collect::<Vec<_>>();
        let mut sections = BTreeMap::new();
        for macro_row in [0usize, 2, 4, 6] {
            sections.insert(
                macro_row,
                full[macro_row * MACRO_DIM..(macro_row + 1) * MACRO_DIM].to_vec(),
            );
        }
        // Independent full-column DIF reference for all eight 1K regions.
        for span in [DIM, DIM / 2] {
            let half = span / 2;
            for group_start in (0..DIM).step_by(span) {
                for j in 0..half {
                    let upper = group_start + j;
                    let lower = upper + half;
                    let a = full[upper];
                    let b = full[lower];
                    let theta = -2.0 * PI * (j as f64) / (span as f64);
                    full[upper] = a + b;
                    full[lower] = (a - b) * Complex64::new(theta.cos(), theta.sin());
                }
            }
        }
        let mut row0 = sections.remove(&0).unwrap();
        let mut row2 = sections.remove(&2).unwrap();
        let mut row4 = sections.remove(&4).unwrap();
        let mut row6 = sections.remove(&6).unwrap();
        // Same pair schedule as the encrypted four-strip implementation.
        let scalar_pair = |upper: &mut [Complex64],
                           lower: &mut [Complex64],
                           row_offset: usize,
                           span_macros: usize| {
            for j in 0..MACRO_DIM {
                let a = upper[j];
                let b = lower[j];
                let theta = -2.0 * PI * ((row_offset * MACRO_DIM + j) as f64)
                    / ((span_macros * MACRO_DIM) as f64);
                upper[j] = a + b;
                lower[j] = (a - b) * Complex64::new(theta.cos(), theta.sin());
            }
        };
        scalar_pair(&mut row0, &mut row4, 0, 8);
        scalar_pair(&mut row2, &mut row6, 2, 8);
        scalar_pair(&mut row0, &mut row2, 0, 4);
        scalar_pair(&mut row4, &mut row6, 0, 4);
        for (macro_row, row) in [(0usize, row0), (2, row2), (4, row4), (6, row6)] {
            for (local_y, actual) in row.iter().enumerate() {
                let expected = full[macro_row * MACRO_DIM + local_y];
                assert!(
                    (*actual - expected).norm() < 1.0e-12,
                    "stage-15 column DIF mismatch in macro-row {macro_row}, local_y {local_y}"
                );
            }
        }
    }
}

// Eight-strip orchestration extension. All row packing and encrypted butterfly
// arithmetic above is unchanged. This is NOT full FFT2 or a secure workload.
type EightStripState = (Vec<Option<Vec<RnsCkksCiphertext>>>, Vec<Complex64>);

fn eight_memory_gate(label: &str, minimum_available_gib: u64) {
    const RSS_CAP_GIB: u64 = 90;
    let available_kib = fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|line| line.starts_with("MemAvailable:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        })
        .expect("MemAvailable missing: refusing extended encrypted run");
    let rss_kib = rss_kib("VmRSS").expect("VmRSS missing: refusing extended encrypted run");
    let peak_kib = crate::rss_kib("VmHWM").expect("VmHWM missing: refusing extended encrypted run");
    println!(
        "DATE27_EIGHT_MEM_GATE checkpoint={label} available_gib={:.4} rss_gib={:.4} peak_gib={:.4} min_available_gib={minimum_available_gib} rss_cap_gib={RSS_CAP_GIB}",
        available_kib as f64 / 1048576.0,
        rss_kib as f64 / 1048576.0,
        peak_kib as f64 / 1048576.0,
    );
    io::stdout().flush().unwrap();
    assert!(
        available_kib > minimum_available_gib * 1048576,
        "memory gate: insufficient available RAM before {label}"
    );
    assert!(
        peak_kib < RSS_CAP_GIB * 1048576,
        "memory gate: observed peak RSS exceeds {RSS_CAP_GIB} GiB before {label}"
    );
}

// Physically pair owned strip objects so no ciphertext vectors are copied and
// no unsafe mutable aliasing is needed. row_offset is the offset within the
// current DIF group (NOT the absolute macro-row index).
fn eight_apply_pair(
    states: &mut BTreeMap<usize, EightStripState>,
    upper_id: usize,
    lower_id: usize,
    span_macros: usize,
    input_level: usize,
    evaluator: &RnsCkksEvaluator<'_>,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    inter_ct: usize,
) {
    assert!([8, 4, 2].contains(&span_macros));
    assert!(upper_id < MACROS_PER_AXIS && lower_id < MACROS_PER_AXIS);
    let half = span_macros / 2;
    assert_eq!(lower_id, upper_id + half);
    assert_eq!(upper_id / span_macros, lower_id / span_macros);
    let offset = upper_id % span_macros;
    assert!(offset < half);
    let mut upper = states
        .remove(&upper_id)
        .expect("upper strip missing from retained state");
    let mut lower = states
        .remove(&lower_id)
        .expect("lower strip missing from retained state");
    let started = Instant::now();
    execute_column_pair_general(
        &mut upper.0,
        &mut lower.0,
        span_macros,
        offset,
        input_level,
        evaluator,
        embedding,
        chain,
        inter_ct,
    );
    let encrypted_ms = started.elapsed().as_millis();
    clear_column_pair_general(&mut upper.1, &mut lower.1, offset, span_macros);
    assert!(states.insert(upper_id, upper).is_none());
    assert!(states.insert(lower_id, lower).is_none());
    println!(
        "DATE27_EIGHT_COLUMN_PAIR_DONE stage={} rows={},{} span_macros={} row_offset={} encrypted_ms={encrypted_ms}",
        input_level + 1, upper_id, lower_id, span_macros, offset,
    );
    eight_memory_gate("after_column_pair", 24);
}

fn eight_expected_pairs(span_macros: usize) -> Vec<(usize, usize)> {
    let half = span_macros / 2;
    let mut result = Vec::new();
    for group in (0..MACROS_PER_AXIS).step_by(span_macros) {
        for offset in 0..half {
            result.push((group + offset, group + offset + half));
        }
    }
    result
}

fn eight_read_independent_samples(path: &Path) -> Vec<ColumnSample> {
    let contents = fs::read_to_string(path).unwrap_or_else(|error| {
        panic!(
            "cannot read independent stages14-16 TSV {}: {error}",
            path.display()
        )
    });
    let mut samples = Vec::new();
    let mut coverage: BTreeMap<usize, [usize; 8]> = BTreeMap::new();
    for (index, line) in contents.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(
            parts.len(),
            5,
            "reference line {} must have five fields",
            index + 1
        );
        let stage: usize = parts[0].parse().expect("invalid reference stage");
        let y: usize = parts[1].parse().expect("invalid reference row");
        let x: usize = parts[2].parse().expect("invalid reference column");
        let re: f64 = parts[3].parse().expect("invalid reference real");
        let im: f64 = parts[4].parse().expect("invalid reference imaginary");
        assert!((14..=16).contains(&stage), "only stages14,15,16 permitted");
        assert!(
            y < DIM && x < DIM && re.is_finite() && im.is_finite(),
            "malformed reference coordinate/value"
        );
        coverage.entry(stage).or_insert([0; 8])[y / MACRO_DIM] += 1;
        samples.push(ColumnSample {
            stage,
            row: y,
            col: x,
            expected: Complex64::new(re, im),
        });
    }
    for stage in 14..=16 {
        let counts = coverage
            .get(&stage)
            .expect("missing stage in reference TSV");
        assert!(
            counts.iter().all(|&count| count > 0),
            "independent checkpoint missing at least one macro-row strip at stage {stage}"
        );
        println!(
            "DATE27_EIGHT_REFERENCE_PREFLIGHT stage={stage} samples={} strips_with_coverage={}",
            counts.iter().sum::<usize>(),
            counts.iter().filter(|&&n| n > 0).count()
        );
    }
    assert!(!samples.is_empty());
    samples
}

fn eight_validate_against_live_oracle(
    stage: usize,
    states: &BTreeMap<usize, EightStripState>,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
    check_cts_per_macro: usize,
) {
    assert!((14..=16).contains(&stage));
    assert_eq!(states.len(), 8);
    let start = Instant::now();
    let mut accuracy = Accuracy::default();
    for macro_row in 0..MACROS_PER_AXIS {
        let (ciphertexts, oracle) = states.get(&macro_row).expect("missing strip in checkpoint");
        for macro_col in 0..MACROS_PER_AXIS {
            let cts = ciphertexts[macro_col]
                .as_ref()
                .expect("missing encrypted macrotile");
            for k in 0..check_cts_per_macro {
                let ct_index = k * CT_PER_MACRO / check_cts_per_macro;
                let ct = &cts[ct_index];
                assert_eq!(ct.level(), stage);
                assert_eq!(ct.basis().len(), 28 - stage);
                let decoded = decrypt_slots(ct, secret, embedding);
                for (slot, actual) in decoded.iter().enumerate() {
                    let (y, x) = physical_position(macro_row, macro_col, ct_index, slot);
                    accuracy.add(*actual, oracle[(y % MACRO_DIM) * DIM + x]);
                }
            }
        }
        println!(
            "DATE27_EIGHT_VALIDATE_PROGRESS stage={stage} strips={}/8",
            macro_row + 1
        );
        io::stdout().flush().unwrap();
    }
    let rel = accuracy.rel_l2();
    let ok = rel.is_finite() && accuracy.max_abs.is_finite() && rel <= TOLERANCE;
    println!("DATE27_EIGHT_CHECKPOINT stage={stage} level={stage} limbs={} slots_checked={} rel_l2={rel:.12e} max_abs={:.12e} validation={} elapsed_ms={} status={}",
        28-stage,accuracy.slots_checked,accuracy.max_abs,
        if check_cts_per_macro==32 {"FULL_IMAGE"} else {"SAMPLED_CIPHERTEXTS"},
        start.elapsed().as_millis(),if ok {"PASS"}else{"FAIL"});
    io::stdout().flush().unwrap();
    assert!(
        ok,
        "stage {stage} encrypted checkpoint failed live DIF comparison"
    );
}

fn eight_validate_independent_samples(
    stage: usize,
    states: &BTreeMap<usize, EightStripState>,
    samples: &[ColumnSample],
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) {
    let start = Instant::now();
    let mut grouped: BTreeMap<(usize, usize, usize), Vec<(usize, Complex64)>> = BTreeMap::new();
    let mut expected_count = 0usize;
    for s in samples.iter().filter(|s| s.stage == stage) {
        let macro_row = s.row / MACRO_DIM;
        let (macro_col, ct, slot) = sample_location(s.row, s.col);
        grouped
            .entry((macro_row, macro_col, ct))
            .or_default()
            .push((slot, s.expected));
        expected_count += 1;
    }
    let mut accuracy = Accuracy::default();
    for ((macro_row, macro_col, index), entries) in grouped {
        let (row, _) = states
            .get(&macro_row)
            .expect("missing strip for independent sample");
        let ciphertext = &row[macro_col].as_ref().unwrap()[index];
        assert_eq!(ciphertext.level(), stage);
        let decoded = decrypt_slots(ciphertext, secret, embedding);
        for (slot, expected) in entries {
            accuracy.add(decoded[slot], expected);
        }
    }
    assert_eq!(accuracy.slots_checked, expected_count);
    let rel = accuracy.rel_l2();
    let ok = rel.is_finite() && accuracy.max_abs.is_finite() && rel <= TOLERANCE;
    println!("DATE27_EIGHT_SAVED_REFERENCE stage={stage} samples={} rel_l2={rel:.12e} max_abs={:.12e} elapsed_ms={} status={}",
        accuracy.slots_checked,accuracy.max_abs,start.elapsed().as_millis(),if ok {"PASS"}else{"FAIL"});
    io::stdout().flush().unwrap();
    assert!(
        ok,
        "stage {stage} does not match independent cleartext checkpoints"
    );
}

fn eight_stage16_main(args: &[String]) {
    if args.len() < 2 {
        eprintln!("usage: date27_sar8k_full_fft eight-stage16 <input_8192.npy> --checkpoint-tsv PATH [--check-cts-per-macro 1|2|4|8|16|32] [--inter-ct N] [--local N]");
        std::process::exit(2);
    }
    let mut check_cts_per_macro = 1usize;
    let mut inter_ct = 2usize;
    let mut local = 4usize;
    let mut checkpoint_path: Option<&Path> = None;
    let mut index = 2usize;
    while index < args.len() {
        match args[index].as_str() {
            "--check-cts-per-macro" => check_cts_per_macro = parse_number(args, &mut index),
            "--inter-ct" => inter_ct = parse_number(args, &mut index),
            "--local" => local = parse_number(args, &mut index),
            "--checkpoint-tsv" => {
                index += 1;
                checkpoint_path = Some(Path::new(args.get(index).unwrap_or_else(|| usage())));
            }
            arg => panic!("unknown eight-stage16 option: {arg}"),
        }
        index += 1;
    }
    assert!([1, 2, 4, 8, 16, 32].contains(&check_cts_per_macro));
    assert!((1..=CT_PER_MACRO).contains(&inter_ct));
    assert!((1..=16).contains(&local));
    assert_eq!(
        std::env::var("FHE_PACKED_1K_NTT_RESIDENT").as_deref(),
        Ok("1"),
        "R24.1 validated cached row implementation requires FHE_PACKED_1K_NTT_RESIDENT=1"
    );
    // Force independent reference existence and coverage before any expensive work.
    let samples = eight_read_independent_samples(
        checkpoint_path.expect("--checkpoint-tsv is mandatory for stage-16 independent validation"),
    );
    print_preflight();
    println!("DATE27_EIGHT_SCOPE=ALL_EIGHT_ROW_STRIPS_THROUGH_GLOBAL_STAGE16");
    println!("DATE27_EIGHT_CIPHERTEXT_TARGET=2048");
    println!("DATE27_EIGHT_CHECK_CTS_PER_MACRO={check_cts_per_macro}");
    println!("DATE27_EIGHT_SECURITY_BEARING=false");
    println!("DATE27_EIGHT_FULL_8K_FFT_STATUS=NOT_EXECUTED");
    io::stdout().flush().unwrap();
    let started = Instant::now();
    eight_memory_gate("pre_setup", 24);
    let image = NpyInput::open(Path::new(&args[1]));
    image.validate();
    let profile = research_profile_65536_for_levels(REQUIRED_LEVELS, GUARD_LIMBS);
    let chain = profile.modulus_chain();
    let embedding = CkksCanonicalEmbedding::new(profile.degree());
    let top_plan = profile.rns_ntt_plan();
    let scale = profile.initial_scale();
    assert_eq!(chain.top().len(), 28);
    assert_eq!(chain.level(16).len(), 12);
    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5232_304b_5345_4352);
    let mut secret: Vec<i8> = (0..embedding.degree())
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&v| v == 0) {
        secret[0] = 1;
    }
    let keygen_started = Instant::now();
    let keys = generate_row_keys(&chain, embedding.degree(), &secret);
    let evaluator = RnsCkksEvaluator::new(&chain, &keys);
    println!(
        "DATE27_EIGHT_KEYGEN_MS={}",
        keygen_started.elapsed().as_millis()
    );
    let mut cached = PackedRowPreparationContext::new();
    let mut states: BTreeMap<usize, EightStripState> = BTreeMap::new();
    // Each pair is reduced from 15 to 14 limbs before the next pair is
    // encrypted, lowering the peak live modulus payload.
    let stage14_started = Instant::now();
    for &(upper, lower) in &[(0usize, 4usize), (2, 6), (1, 5), (3, 7)] {
        for macro_row in [upper, lower] {
            eight_memory_gate("before_new_row_strip", 24);
            let (cts, oracle) = encrypted_strip_to_row13(
                &image,
                macro_row,
                &secret,
                &embedding,
                &chain,
                &top_plan,
                scale,
                &evaluator,
                &mut cached,
                inter_ct,
                local,
                check_cts_per_macro,
            );
            assert!(states.insert(macro_row, (cts, oracle)).is_none());
            eight_memory_gate("after_new_row_strip", 24);
        }
        eight_apply_pair(
            &mut states,
            upper,
            lower,
            8,
            13,
            &evaluator,
            &embedding,
            &chain,
            inter_ct,
        );
    }
    drop(cached);
    println!(
        "DATE27_EIGHT_STAGE14_PHASE_MS={}",
        stage14_started.elapsed().as_millis()
    );
    log_memory("eight_stage14_all_strips");
    assert_eq!(states.len(), 8);
    eight_validate_against_live_oracle(14, &states, &secret, &embedding, check_cts_per_macro);
    eight_validate_independent_samples(14, &states, &samples, &secret, &embedding);
    eight_memory_gate("before_stage15", 24);
    let stage15_started = Instant::now();
    for (upper, lower) in eight_expected_pairs(4) {
        eight_apply_pair(
            &mut states,
            upper,
            lower,
            4,
            14,
            &evaluator,
            &embedding,
            &chain,
            inter_ct,
        );
    }
    println!(
        "DATE27_EIGHT_STAGE15_EXEC_MS={}",
        stage15_started.elapsed().as_millis()
    );
    eight_validate_against_live_oracle(15, &states, &secret, &embedding, check_cts_per_macro);
    eight_validate_independent_samples(15, &states, &samples, &secret, &embedding);
    eight_memory_gate("before_stage16", 24);
    let stage16_started = Instant::now();
    for (upper, lower) in eight_expected_pairs(2) {
        eight_apply_pair(
            &mut states,
            upper,
            lower,
            2,
            15,
            &evaluator,
            &embedding,
            &chain,
            inter_ct,
        );
    }
    println!(
        "DATE27_EIGHT_STAGE16_EXEC_MS={}",
        stage16_started.elapsed().as_millis()
    );
    eight_validate_against_live_oracle(16, &states, &secret, &embedding, check_cts_per_macro);
    eight_validate_independent_samples(16, &states, &samples, &secret, &embedding);
    for row in states.values() {
        for ct_array in &row.0 {
            let cts = ct_array
                .as_ref()
                .expect("missing ciphertext array after stage16");
            assert_eq!(cts.len(), CT_PER_MACRO);
            assert!(cts
                .iter()
                .all(|ct| ct.level() == 16 && ct.basis().len() == 12));
        }
    }
    println!("DATE27_EIGHT_FINAL_LEVEL=16");
    println!("DATE27_EIGHT_FINAL_LIMBS=12");
    println!(
        "DATE27_EIGHT_CIPHERTEXTS_RETAINED={}",
        states
            .values()
            .map(|s| s.0.iter().map(|m| m.as_ref().unwrap().len()).sum::<usize>())
            .sum::<usize>()
    );
    println!("DATE27_EIGHT_TOTAL_MS={}", started.elapsed().as_millis());
    log_memory("eight_final");
    eight_memory_gate("final", 24);
    println!("DATE27_EIGHT_STATUS=PASS");
    io::stdout().flush().unwrap();
}

// Complete 26-stage SAR execution. All encrypted arithmetic is provided by
// unchanged ccmm-rs eBLAS kernels; the code here owns schedules, reference
// checks and memory guards only. The source stage16 executor remains present
// in this isolated binary as an independent baseline.
fn full_column_key_schedule() -> Vec<(usize, usize)> {
    let mut schedule = vec![(17, 16384), (18, 8192), (19, 4096)];
    let mut rotation = 2048;
    for level in 20..=25 {
        schedule.push((level, rotation));
        rotation /= 2;
    }
    assert_eq!(schedule.len(), 9);
    assert_eq!(rotation, 32);
    schedule
}

fn full_generate_column_keys(
    chain: &ModulusChain,
    degree: usize,
    secret: &[i8],
) -> RnsCkksEvaluationKeys {
    let mut keys = RnsCkksEvaluationKeys::new();
    let mut count = 0usize;
    for (level, rotation) in full_column_key_schedule() {
        eight_memory_gate("before_column_keygen_level", 24);
        let started = Instant::now();
        let basis = chain.level(level).clone();
        let layout = RnsGadgetLayout::new(basis.clone(), block_sizes(basis.len()));
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
        let left = rotation_exponent_left(degree, rotation);
        let right = rotation_exponent_right(degree, rotation);
        let mut level_keys = RnsCkksLevelKeys::new(level, basis);
        println!("DATE27_FULL_COLUMN_KEY_BEGIN level={level} rotation={rotation}");
        io::stdout().flush().unwrap();
        for (direction, exponent) in [(0u64, left), (1u64, right)] {
            if direction == 1 && left == right {
                continue;
            }
            let seed = 0x5232_304b_4b45_5900u64 ^ ((level as u64) << 8) ^ direction;
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let key = RnsGaloisKey::generate_with_ntt_rng(
                RnsKeygenConfig {
                    degree,
                    plaintext_modulus: 2,
                    noise_bound: 0,
                    layout: layout.clone(),
                    plan: &plan,
                },
                secret,
                exponent,
                &mut rng,
            );
            level_keys.insert_galois_key(key);
            count += 1;
        }
        keys.insert_level(level_keys);
        println!(
            "DATE27_FULL_COLUMN_KEY_DONE level={level} elapsed_ms={}",
            started.elapsed().as_millis()
        );
        io::stdout().flush().unwrap();
        eight_memory_gate("after_column_keygen_level", 24);
    }
    println!("DATE27_FULL_COLUMN_GALOIS_KEYS={count}");
    keys
}

// Apply ten successive 1024-point DIF butterflies to the corresponding
// 1024 x 1024 tile of the *independent cleartext oracle*. Column global
// stages 14..16 have already been applied to this oracle.
fn full_advance_cleartext_column_macrotile(oracle: &mut [Complex64], macro_col: usize) {
    assert_eq!(oracle.len(), MACRO_DIM * DIM);
    assert!(macro_col < MACROS_PER_AXIS);
    let start_x = macro_col * MACRO_DIM;
    for exponent in (1..=10usize).rev() {
        let span = 1usize << exponent;
        let half = span / 2;
        for group in (0..MACRO_DIM).step_by(span) {
            for j in 0..half {
                let theta = -2.0 * PI * j as f64 / span as f64;
                let twiddle = Complex64::new(theta.cos(), theta.sin());
                let top = (group + j) * DIM + start_x;
                let bottom = (group + j + half) * DIM + start_x;
                for dx in 0..MACRO_DIM {
                    let a = oracle[top + dx];
                    let b = oracle[bottom + dx];
                    oracle[top + dx] = a + b;
                    oracle[bottom + dx] = (a - b) * twiddle;
                }
            }
        }
    }
}

fn full_read_samples(path: &Path) -> Vec<ColumnSample> {
    let contents = fs::read_to_string(path).unwrap_or_else(|error| {
        panic!(
            "cannot read mandatory independent stages14/15/16/26 TSV {}: {error}",
            path.display()
        )
    });
    let mut samples = Vec::new();
    let mut coverage: BTreeMap<usize, [usize; MACROS_PER_AXIS]> = BTreeMap::new();
    let mut unique = std::collections::BTreeSet::new();
    for (line_number, line) in contents.lines().enumerate() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(
            parts.len(),
            5,
            "reference line {} must have five fields",
            line_number + 1
        );
        let stage: usize = parts[0].parse().expect("invalid reference stage");
        let y: usize = parts[1].parse().expect("invalid reference y");
        let x: usize = parts[2].parse().expect("invalid reference x");
        let re: f64 = parts[3].parse().expect("invalid reference real");
        let im: f64 = parts[4].parse().expect("invalid reference imaginary");
        assert!(
            [14, 15, 16, 26].contains(&stage),
            "unexpected reference stage"
        );
        assert!(
            y < DIM && x < DIM && re.is_finite() && im.is_finite(),
            "invalid reference sample"
        );
        assert!(
            unique.insert((stage, y, x)),
            "duplicate stage/coordinate in reference file"
        );
        coverage.entry(stage).or_insert([0; MACROS_PER_AXIS])[y / MACRO_DIM] += 1;
        samples.push(ColumnSample {
            stage,
            row: y,
            col: x,
            expected: Complex64::new(re, im),
        });
    }
    for stage in [14usize, 15, 16, 26] {
        let counts = coverage
            .get(&stage)
            .expect("missing mandatory reference stage");
        assert!(
            counts.iter().all(|&n| n > 0),
            "reference stage {stage} missing coverage in at least one strip"
        );
        println!(
            "DATE27_FULL_REF_PREFLIGHT stage={stage} samples={} covered_strips={}",
            counts.iter().sum::<usize>(),
            counts.iter().filter(|&&n| n > 0).count()
        );
    }
    samples
}

fn full_bit_reverse_13(index: usize) -> usize {
    assert!(index < DIM);
    index.reverse_bits() >> (usize::BITS as usize - 13)
}

// Read individual complex128 bins from the 1-GiB NumPy FFT2 reference with
// seeks, not by allocating another complete image. A physical DIF coordinate
// maps to a NumPy frequency coordinate after reversing 13 bits per axis.
struct FullNpyFft2 {
    file: fs::File,
    payload_offset: u64,
}
impl FullNpyFft2 {
    fn open(path: &Path) -> Self {
        let mut file = fs::File::open(path).unwrap_or_else(|err| {
            panic!("cannot open NumPy FFT2 reference {}: {err}", path.display())
        });
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .expect("truncated NumPy FFT2 reference");
        assert_eq!(&magic[..6], b"\x93NUMPY", "NumPy reference magic mismatch");
        let version = magic[6];
        let (header_len, offset) = match version {
            1 => {
                let mut bytes = [0u8; 2];
                file.read_exact(&mut bytes)
                    .expect("truncated v1 NumPy header");
                (u16::from_le_bytes(bytes) as usize, 10u64)
            }
            2 | 3 => {
                let mut bytes = [0u8; 4];
                file.read_exact(&mut bytes)
                    .expect("truncated v2/3 NumPy header");
                (u32::from_le_bytes(bytes) as usize, 12u64)
            }
            _ => panic!("unsupported NumPy reference version"),
        };
        assert!(
            header_len < 4096,
            "unexpected oversized NumPy reference header"
        );
        let mut raw = vec![0u8; header_len];
        file.read_exact(&mut raw)
            .expect("truncated NumPy reference header");
        let header = std::str::from_utf8(&raw).expect("NumPy reference header not ASCII");
        assert!(
            header.contains("<c16"),
            "expected little-endian complex128 NumPy reference"
        );
        assert!(
            header.contains("'fortran_order': False")
                || header.contains("\"fortran_order\": False"),
            "expected C-contiguous NumPy FFT2 output"
        );
        let shape_part = header
            .split("shape")
            .nth(1)
            .expect("missing FFT2 reference shape");
        let inside = shape_part
            .split_once('(')
            .unwrap()
            .1
            .split_once(')')
            .unwrap()
            .0;
        let dims: Vec<usize> = inside
            .split(',')
            .filter(|v| !v.trim().is_empty())
            .map(|v| v.trim().parse::<usize>().expect("invalid FFT2 dimension"))
            .collect();
        assert_eq!(dims, [DIM, DIM], "unexpected NumPy FFT2 reference shape");
        let payload_offset = offset + header_len as u64;
        assert_eq!(
            file.metadata().unwrap().len(),
            payload_offset + (DIM * DIM * 16) as u64,
            "unexpected NumPy FFT2 file length"
        );
        Self {
            file,
            payload_offset,
        }
    }
    fn at_physical_dif(&mut self, y: usize, x: usize) -> Complex64 {
        let ny = full_bit_reverse_13(y);
        let nx = full_bit_reverse_13(x);
        let byte_offset = self.payload_offset + ((ny * DIM + nx) * 16) as u64;
        self.file
            .seek(SeekFrom::Start(byte_offset))
            .expect("NumPy FFT2 seek failed");
        let mut buf = [0u8; 16];
        self.file
            .read_exact(&mut buf)
            .expect("NumPy FFT2 reference truncated");
        Complex64::new(
            f64::from_le_bytes(buf[..8].try_into().unwrap()),
            f64::from_le_bytes(buf[8..].try_into().unwrap()),
        )
    }
}

fn full_verify_cleartext_reference_against_numpy(samples: &[ColumnSample], npy_path: &Path) {
    let mut npy = FullNpyFft2::open(npy_path);
    let mut checked = 0usize;
    let mut max_delta = 0.0_f64;
    for sample in samples.iter().filter(|s| s.stage == 26) {
        let expected = npy.at_physical_dif(sample.row, sample.col);
        let err = (expected - sample.expected).norm();
        max_delta = max_delta.max(err);
        assert!(
            err <= 1.0e-8 + 1.0e-10 * expected.norm(),
            "stage26 cleartext checkpoint disagrees with independent NumPy FFT2 at ({},{})",
            sample.row,
            sample.col
        );
        checked += 1;
    }
    assert!(checked >= 8, "too few stage26 NumPy samples");
    println!("DATE27_FULL_NUMPY_REFERENCE_PREFLIGHT samples={checked} max_abs={max_delta:.12e} status=PASS");
    io::stdout().flush().unwrap();
}

fn full_validate_numpy_fft2(
    states: &BTreeMap<usize, EightStripState>,
    samples: &[ColumnSample],
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
    npy_path: &Path,
) {
    let started = Instant::now();
    let mut npy = FullNpyFft2::open(npy_path);
    let mut grouped: BTreeMap<(usize, usize, usize), Vec<(usize, Complex64)>> = BTreeMap::new();
    for sample in samples.iter().filter(|s| s.stage == 26) {
        let (macro_col, ct, slot) = sample_location(sample.row, sample.col);
        let bin = npy.at_physical_dif(sample.row, sample.col);
        grouped
            .entry((sample.row / MACRO_DIM, macro_col, ct))
            .or_default()
            .push((slot, bin));
    }
    let mut accuracy = Accuracy::default();
    for ((macro_row, macro_col, ct_index), items) in grouped {
        let state = states
            .get(&macro_row)
            .expect("missing final macrotile strip");
        let ct = &state.0[macro_col]
            .as_ref()
            .expect("missing final ciphertext")[ct_index];
        assert_eq!(ct.level(), 26);
        let decoded = decrypt_slots(ct, secret, embedding);
        for (slot, expected) in items {
            accuracy.add(decoded[slot], expected);
        }
    }
    let rel = accuracy.rel_l2();
    let pass = accuracy.slots_checked >= 8
        && rel.is_finite()
        && accuracy.max_abs.is_finite()
        && rel <= TOLERANCE;
    println!("DATE27_FULL_NUMPY_FFT2 samples={} rel_l2={rel:.12e} max_abs={:.12e} elapsed_ms={} status={}",
        accuracy.slots_checked,accuracy.max_abs,started.elapsed().as_millis(),
        if pass {"PASS"}else{"FAIL"});
    io::stdout().flush().unwrap();
    assert!(
        pass,
        "encrypted final FFT2 disagrees with NumPy sample bins"
    );
}

// Stream the authentic decrypted ciphertext outputs, one ciphertext at a time.
// The 4D array remains in the established physical ciphertext/slot order;
// a separate verified converter applies 13-bit DIF bit reversal on both axes.
// No extra 1-GiB memory allocation is required for the output image.
fn full_export_decrypted_packed(
    states: &BTreeMap<usize, EightStripState>,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
    output: &Path,
) {
    assert!(!output.exists(), "refusing to overwrite decrypted output");
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).expect("cannot make export directory");
    }
    let partial = output.with_extension("npy.partial");
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&partial)
        .expect("cannot create fresh partial decrypted output");
    // NumPy .npy v1.0, little-endian complex128 in C order.
    let mut header =
        "{'descr': '<c16', 'fortran_order': False, 'shape': (8, 8, 32, 32768), }".to_owned();
    let spaces = (16 - ((10 + header.len() + 1) % 16)) % 16;
    header.push_str(&" ".repeat(spaces));
    header.push('\n');
    let size = u16::try_from(header.len()).expect("NumPy header too long");
    file.write_all(b"\x93NUMPY").unwrap();
    file.write_all(&[1, 0]).unwrap();
    file.write_all(&size.to_le_bytes()).unwrap();
    file.write_all(header.as_bytes()).unwrap();
    let started = Instant::now();
    let mut written = 0usize;
    let mut bytes = vec![0u8; SLOT_COUNT * 16];
    for macro_row in 0..MACROS_PER_AXIS {
        let state = states
            .get(&macro_row)
            .expect("missing encrypted strip for export");
        for macro_col in 0..MACROS_PER_AXIS {
            let cts = state.0[macro_col]
                .as_ref()
                .expect("missing final macrotile for export");
            assert_eq!(cts.len(), CT_PER_MACRO);
            for ct in cts {
                assert_eq!(ct.level(), 26);
                let decoded = decrypt_slots(ct, secret, embedding);
                assert_eq!(decoded.len(), SLOT_COUNT);
                for (i, z) in decoded.iter().enumerate() {
                    assert!(
                        z.re.is_finite() && z.im.is_finite(),
                        "nonfinite decrypted FFT coefficient"
                    );
                    bytes[16 * i..16 * i + 8].copy_from_slice(&z.re.to_le_bytes());
                    bytes[16 * i + 8..16 * i + 16].copy_from_slice(&z.im.to_le_bytes());
                }
                file.write_all(&bytes)
                    .expect("decrypted spectrum export write failed");
                written += SLOT_COUNT;
            }
        }
        println!(
            "DATE27_DECRYPT_EXPORT_PROGRESS rows={}/8 coefficients={}",
            macro_row + 1,
            written
        );
        io::stdout().flush().unwrap();
    }
    assert_eq!(written, DIM * DIM);
    file.sync_all().expect("failed to sync decoded FFT array");
    let expected = 10u64 + header.len() as u64 + (DIM * DIM * 16) as u64;
    assert_eq!(
        file.metadata().unwrap().len(),
        expected,
        "incomplete decoded output file"
    );
    drop(file);
    fs::rename(&partial, output).expect("failed to finalize decoded FFT array");
    println!("DATE27_DECRYPT_EXPORT_STATUS=PASS coefficients={written} shape=(8,8,32,32768) format=numpy_complex128_physical_ct_slot_order bytes={} elapsed_ms={} path={}",
        expected,started.elapsed().as_millis(),output.display());
    io::stdout().flush().unwrap();
}

fn full_stage26_main(args: &[String]) {
    if args.len() < 2 {
        eprintln!("usage: date27_sar8k_full_fft full-stage26 <input_8192.npy> --checkpoint-tsv PATH --numpy-reference FFT2.npy [--check-cts-per-macro 1|2|4|8|16|32] [--inter-ct N] [--local N] [--export-packed-decrypted PATH]");
        std::process::exit(2);
    }
    let mut check_cts_per_macro = 1usize;
    let mut inter_ct = 2usize;
    let mut local = 4usize;
    let mut checkpoint_path: Option<&Path> = None;
    let mut numpy_path: Option<&Path> = None;
    let mut export_packed_path: Option<&Path> = None;
    let mut index = 2usize;
    while index < args.len() {
        match args[index].as_str() {
            "--check-cts-per-macro" => check_cts_per_macro = parse_number(args, &mut index),
            "--inter-ct" => inter_ct = parse_number(args, &mut index),
            "--local" => local = parse_number(args, &mut index),
            "--checkpoint-tsv" => {
                index += 1;
                checkpoint_path = Some(Path::new(args.get(index).unwrap_or_else(|| usage())));
            }
            "--numpy-reference" => {
                index += 1;
                numpy_path = Some(Path::new(args.get(index).unwrap_or_else(|| usage())));
            }
            "--export-packed-decrypted" => {
                index += 1;
                export_packed_path = Some(Path::new(args.get(index).unwrap_or_else(|| usage())));
            }
            other => panic!("unknown full-stage26 option: {other}"),
        }
        index += 1;
    }
    assert!([1, 2, 4, 8, 16, 32].contains(&check_cts_per_macro));
    assert!((1..=CT_PER_MACRO).contains(&inter_ct));
    assert!((1..=16).contains(&local));
    assert_eq!(
        std::env::var("FHE_PACKED_1K_NTT_RESIDENT").as_deref(),
        Ok("1"),
        "validated resident-NTT mode requires FHE_PACKED_1K_NTT_RESIDENT=1"
    );
    // Prohibit the multi-hour run if reference prerequisites fail. The final
    // checkpoint is independently tied to NumPy FFT2, accounting for DIF order.
    let samples = full_read_samples(checkpoint_path.expect("mandatory --checkpoint-tsv missing"));
    let numpy_path = numpy_path.expect("mandatory --numpy-reference missing");
    full_verify_cleartext_reference_against_numpy(&samples, numpy_path);
    print_preflight();
    println!("DATE27_FULL_SCOPE=ZARAGOZA_ALL_8K_STAGES_1_TO_26");
    println!("DATE27_FULL_CIPHERTEXTS=2048");
    println!("DATE27_FULL_SECURITY_BEARING=false");
    println!("DATE27_FULL_VALIDATION_CTS_PER_MACRO={check_cts_per_macro}");
    io::stdout().flush().unwrap();
    let total_started = Instant::now();
    eight_memory_gate("full_pre_setup", 24);
    let image = NpyInput::open(Path::new(&args[1]));
    image.validate();
    let profile = research_profile_65536_for_levels(REQUIRED_LEVELS, GUARD_LIMBS);
    let chain = profile.modulus_chain();
    let embedding = CkksCanonicalEmbedding::new(profile.degree());
    let top_plan = profile.rns_ntt_plan();
    let scale = profile.initial_scale();
    assert_eq!(chain.top().len(), 28);
    assert_eq!(chain.level(16).len(), 12);
    assert_eq!(chain.level(26).len(), 2);
    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5232_304b_5345_4352);
    let mut secret: Vec<i8> = (0..embedding.degree())
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&v| v == 0) {
        secret[0] = 1;
    }
    // Use one secret across all stages; release row keys before creating
    // independent level-17..25 column keys to limit resident memory.
    let keygen_started = Instant::now();
    let row_keys = generate_row_keys(&chain, embedding.degree(), &secret);
    let row_evaluator = RnsCkksEvaluator::new(&chain, &row_keys);
    println!(
        "DATE27_FULL_ROW_KEYGEN_MS={}",
        keygen_started.elapsed().as_millis()
    );
    let mut cached = PackedRowPreparationContext::new();
    let mut states: BTreeMap<usize, EightStripState> = BTreeMap::new();
    let stage14_started = Instant::now();
    for &(upper, lower) in &[(0usize, 4usize), (2, 6), (1, 5), (3, 7)] {
        for macro_row in [upper, lower] {
            eight_memory_gate("full_before_new_row_strip", 24);
            let (cts, oracle) = encrypted_strip_to_row13(
                &image,
                macro_row,
                &secret,
                &embedding,
                &chain,
                &top_plan,
                scale,
                &row_evaluator,
                &mut cached,
                inter_ct,
                local,
                check_cts_per_macro,
            );
            assert!(states.insert(macro_row, (cts, oracle)).is_none());
            eight_memory_gate("full_after_new_row_strip", 24);
        }
        eight_apply_pair(
            &mut states,
            upper,
            lower,
            8,
            13,
            &row_evaluator,
            &embedding,
            &chain,
            inter_ct,
        );
    }
    drop(cached);
    println!(
        "DATE27_FULL_STAGE14_PHASE_MS={}",
        stage14_started.elapsed().as_millis()
    );
    eight_validate_against_live_oracle(14, &states, &secret, &embedding, check_cts_per_macro);
    eight_validate_independent_samples(14, &states, &samples, &secret, &embedding);
    eight_memory_gate("full_before_stage15", 24);
    let stage15_started = Instant::now();
    for (upper, lower) in eight_expected_pairs(4) {
        eight_apply_pair(
            &mut states,
            upper,
            lower,
            4,
            14,
            &row_evaluator,
            &embedding,
            &chain,
            inter_ct,
        );
    }
    println!(
        "DATE27_FULL_STAGE15_EXEC_MS={}",
        stage15_started.elapsed().as_millis()
    );
    eight_validate_against_live_oracle(15, &states, &secret, &embedding, check_cts_per_macro);
    eight_validate_independent_samples(15, &states, &samples, &secret, &embedding);
    eight_memory_gate("full_before_stage16", 24);
    let stage16_started = Instant::now();
    for (upper, lower) in eight_expected_pairs(2) {
        eight_apply_pair(
            &mut states,
            upper,
            lower,
            2,
            15,
            &row_evaluator,
            &embedding,
            &chain,
            inter_ct,
        );
    }
    println!(
        "DATE27_FULL_STAGE16_EXEC_MS={}",
        stage16_started.elapsed().as_millis()
    );
    eight_validate_against_live_oracle(16, &states, &secret, &embedding, check_cts_per_macro);
    eight_validate_independent_samples(16, &states, &samples, &secret, &embedding);
    assert_eq!(states.len(), MACROS_PER_AXIS);
    for row in states.values() {
        for macrotile in &row.0 {
            let cts = macrotile.as_ref().expect("missing stage16 macrotile");
            assert_eq!(cts.len(), CT_PER_MACRO);
            assert!(cts
                .iter()
                .all(|ct| ct.level() == 16 && ct.basis().len() == 12));
        }
    }
    println!("DATE27_FULL_STAGE16_STATUS=PASS");
    io::stdout().flush().unwrap();
    eight_memory_gate("full_after_stage16", 24);
    drop(row_evaluator);
    drop(row_keys);
    eight_memory_gate("full_row_keys_released", 24);
    let started = Instant::now();
    let column_keys = full_generate_column_keys(&chain, embedding.degree(), &secret);
    let column_evaluator = RnsCkksEvaluator::new(&chain, &column_keys);
    println!(
        "DATE27_FULL_COLUMN_KEYGEN_MS={}",
        started.elapsed().as_millis()
    );
    eight_memory_gate("full_before_local_column_fft", 24);

    let local_started = Instant::now();
    let mut accuracy = Accuracy::default();
    let mut completed = 0usize;
    for macro_row in 0..MACROS_PER_AXIS {
        for macro_col in 0..MACROS_PER_AXIS {
            eight_memory_gate("full_before_local_column_macrotile", 24);
            let (row, oracle) = states
                .get_mut(&macro_row)
                .expect("missing strip in local column phase");
            let input = row[macro_col]
                .take()
                .expect("missing stage16 macrotile ciphertexts");
            assert_eq!(input.len(), CT_PER_MACRO);
            assert!(input
                .iter()
                .all(|ct| ct.level() == 16 && ct.basis().len() == 12));
            let started = Instant::now();
            let output = execute_packed_1k_fft_columns_cp(
                input,
                &column_evaluator,
                &embedding,
                &chain,
                inter_ct,
                local,
            );
            let encrypted_ms = started.elapsed().as_millis();
            assert_eq!(output.len(), CT_PER_MACRO);
            assert!(output
                .iter()
                .all(|ct| ct.level() == 26 && ct.basis().len() == 2));
            row[macro_col] = Some(output);
            full_advance_cleartext_column_macrotile(oracle, macro_col);
            let mut local_accuracy = Accuracy::default();
            for k in 0..check_cts_per_macro {
                let ct_index = k * CT_PER_MACRO / check_cts_per_macro;
                let ct = &row[macro_col].as_ref().unwrap()[ct_index];
                let decoded = decrypt_slots(ct, &secret, &embedding);
                for (slot, actual) in decoded.iter().enumerate() {
                    let (y, x) = physical_position(macro_row, macro_col, ct_index, slot);
                    let expected = oracle[(y % MACRO_DIM) * DIM + x];
                    accuracy.add(*actual, expected);
                    local_accuracy.add(*actual, expected);
                }
            }
            assert!(
                local_accuracy.max_abs.is_finite() && local_accuracy.rel_l2().is_finite(),
                "nonfinite local-column output detected"
            );
            completed += 1;
            println!("DATE27_FULL_LOCAL_MACRO row={macro_row} col={macro_col} done={completed}/64 level=26 limbs=2 encrypted_ms={encrypted_ms} checked_slots={} rel_l2={:.12e} max_abs={:.12e}",
                local_accuracy.slots_checked,local_accuracy.rel_l2(),local_accuracy.max_abs);
            io::stdout().flush().unwrap();
            eight_memory_gate("full_after_local_column_macrotile", 24);
        }
    }
    println!(
        "DATE27_FULL_LOCAL_COLUMN_EXEC_AND_VALIDATION_MS={}",
        local_started.elapsed().as_millis()
    );
    let rel = accuracy.rel_l2();
    let passed = accuracy.slots_checked
        == MACROS_PER_AXIS * MACROS_PER_AXIS * check_cts_per_macro * SLOT_COUNT
        && rel.is_finite()
        && accuracy.max_abs.is_finite()
        && rel <= TOLERANCE;
    println!("DATE27_FULL_CHECKPOINT stage=26 level=26 limbs=2 slots_checked={} rel_l2={rel:.12e} max_abs={:.12e} validation={} status={}",
        accuracy.slots_checked,accuracy.max_abs,
        if check_cts_per_macro==32 {"FULL_IMAGE"}else{"SAMPLED_CIPHERTEXTS"},
        if passed {"PASS"}else{"FAIL"});
    io::stdout().flush().unwrap();
    assert!(
        passed,
        "stage26 live-oracle encrypted FFT2 validation failed"
    );
    eight_validate_independent_samples(26, &states, &samples, &secret, &embedding);
    full_validate_numpy_fft2(&states, &samples, &secret, &embedding, numpy_path);
    let mut retained = 0usize;
    for state in states.values() {
        for macrotile in &state.0 {
            let cts = macrotile.as_ref().expect("missing final ciphertexts");
            retained += cts.len();
            assert!(cts
                .iter()
                .all(|ct| ct.level() == 26 && ct.basis().len() == 2));
        }
    }
    assert_eq!(retained, 2048);
    println!("DATE27_FULL_FINAL_LEVEL=26");
    println!("DATE27_FULL_FINAL_LIMBS=2");
    println!("DATE27_FULL_CIPHERTEXTS_RETAINED={retained}");
    println!(
        "DATE27_FULL_TOTAL_MS={}",
        total_started.elapsed().as_millis()
    );
    log_memory("full_final");
    eight_memory_gate("full_final", 24);
    println!("DATE27_FULL_STATUS=PASS");
    io::stdout().flush().unwrap();
    // Artifact export is intentionally excluded from the baseline timing.
    if let Some(export_path) = export_packed_path {
        full_export_decrypted_packed(&states, &secret, &embedding, export_path);
    }
}

#[cfg(test)]
mod stage26_full_integration_tests {
    use super::*;
    #[test]
    fn column_key_schedule_spans_only_galois_levels() {
        assert_eq!(
            full_column_key_schedule(),
            vec![
                (17, 16384),
                (18, 8192),
                (19, 4096),
                (20, 2048),
                (21, 1024),
                (22, 512),
                (23, 256),
                (24, 128),
                (25, 64)
            ]
        );
    }
    #[test]
    fn bit_reversal_is_involution() {
        for v in [0usize, 1, 2, 3, 127, 1023, 4095, 8191] {
            assert_eq!(full_bit_reverse_13(full_bit_reverse_13(v)), v);
        }
    }
    #[test]
    fn ten_local_column_stages_complete_independent_8k_dif() {
        // One 8192-element vertical column, split into eight physical strips.
        // Match a direct scalar 8192 DIF after all 13 column stages.
        let signal = |y: usize| {
            Complex64::new(
                ((y * 31 + 5) % 8179) as f64 / 8179.0,
                ((y * 17 + 7) % 4093) as f64 / 4093.0,
            )
        };
        let mut full = (0..DIM).map(signal).collect::<Vec<_>>();
        let mut strips = (0..MACROS_PER_AXIS)
            .map(|r| {
                (0..MACRO_DIM)
                    .map(|i| signal(r * MACRO_DIM + i))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for span in [DIM, DIM / 2, DIM / 4] {
            let half = span / 2;
            for group in (0..DIM).step_by(span) {
                for j in 0..half {
                    let (u, d) = (group + j, group + j + half);
                    let twiddle = Complex64::from_polar(1.0, -2.0 * PI * j as f64 / span as f64);
                    let (a, b) = (full[u], full[d]);
                    full[u] = a + b;
                    full[d] = (a - b) * twiddle;
                }
            }
        }
        for span in [8usize, 4, 2] {
            for (upper, lower) in eight_expected_pairs(span) {
                let offset = upper % span;
                for y in 0..MACRO_DIM {
                    let w = Complex64::from_polar(
                        1.0,
                        -2.0 * PI * (offset * MACRO_DIM + y) as f64 / (span * MACRO_DIM) as f64,
                    );
                    let (a, b) = (strips[upper][y], strips[lower][y]);
                    strips[upper][y] = a + b;
                    strips[lower][y] = (a - b) * w;
                }
            }
        }
        for span in [512usize, 256, 128, 64, 32, 16, 8, 4, 2, 1] {
            let full_span = span * 2;
            for group in (0..DIM).step_by(full_span) {
                for j in 0..span {
                    let a = full[group + j];
                    let b = full[group + j + span];
                    let w = Complex64::from_polar(1.0, -2.0 * PI * j as f64 / full_span as f64);
                    full[group + j] = a + b;
                    full[group + j + span] = (a - b) * w;
                }
            }
            for strip in &mut strips {
                for group in (0..MACRO_DIM).step_by(full_span) {
                    for j in 0..span {
                        let a = strip[group + j];
                        let b = strip[group + j + span];
                        let w = Complex64::from_polar(1.0, -2.0 * PI * j as f64 / full_span as f64);
                        strip[group + j] = a + b;
                        strip[group + j + span] = (a - b) * w;
                    }
                }
            }
        }
        for y in 0..DIM {
            assert!((full[y] - strips[y / MACRO_DIM][y % MACRO_DIM]).norm() < 1.0e-12);
        }
    }
}

#[cfg(test)]
mod stage16_eight_strips_tests {
    use super::*;

    #[test]
    fn all_column_pairs_cover_all_eight_strips_once_per_stage() {
        for span in [8usize, 4, 2] {
            let pairs = eight_expected_pairs(span);
            assert_eq!(pairs.len(), 4);
            let mut visited = [0usize; 8];
            for (upper, lower) in pairs {
                visited[upper] += 1;
                visited[lower] += 1;
                assert_eq!(lower - upper, span / 2);
                assert_eq!(upper / span, lower / span);
                assert!(upper % span < span / 2);
            }
            assert_eq!(visited, [1; 8]);
        }
    }

    #[test]
    fn odd_offset_column_twiddles_match_scalar_dif() {
        for (span, offsets) in [(8usize, vec![1, 3]), (4, vec![1]), (2, vec![0])] {
            for offset in offsets {
                for block in 0..2 {
                    let twiddles = column_pair_twiddles(offset, block, span);
                    for slot in [0usize, 1, 4095, 8192, 32767] {
                        let lane = slot / TILE_ELEMENTS;
                        let local_y = block * TILES_PER_CT * TILE_DIM
                            + lane * TILE_DIM
                            + (slot % TILE_ELEMENTS) / TILE_DIM;
                        let theta = -2.0 * PI * ((offset * MACRO_DIM + local_y) as f64)
                            / ((span * MACRO_DIM) as f64);
                        let expected = Complex64::new(theta.cos(), theta.sin());
                        assert!((twiddles[slot] - expected).norm() < 1.0e-14);
                    }
                }
            }
        }
    }

    #[test]
    fn complete_three_column_stages_match_independent_8192_point_dif() {
        let make_value = |y: usize| {
            Complex64::new(
                ((y * 37 + 17) % 8191) as f64 / 8191.0,
                ((y * 13 + 5) % 4093) as f64 / 4093.0,
            )
        };
        let mut independent = (0..DIM).map(make_value).collect::<Vec<_>>();
        let mut split: Vec<Vec<Complex64>> = (0..8)
            .map(|row| {
                (0..MACRO_DIM)
                    .map(|offset| make_value(row * MACRO_DIM + offset))
                    .collect()
            })
            .collect();
        for span in [DIM, DIM / 2, DIM / 4] {
            let half = span / 2;
            for group in (0..DIM).step_by(span) {
                for j in 0..half {
                    let (upper, lower) = (group + j, group + j + half);
                    let a = independent[upper];
                    let b = independent[lower];
                    let theta = -2.0 * PI * (j as f64) / (span as f64);
                    independent[upper] = a + b;
                    independent[lower] = (a - b) * Complex64::new(theta.cos(), theta.sin());
                }
            }
        }
        for span in [8usize, 4, 2] {
            for (upper, lower) in eight_expected_pairs(span) {
                let offset = upper % span;
                for y in 0..MACRO_DIM {
                    let a = split[upper][y];
                    let b = split[lower][y];
                    let theta =
                        -2.0 * PI * ((offset * MACRO_DIM + y) as f64) / ((span * MACRO_DIM) as f64);
                    split[upper][y] = a + b;
                    split[lower][y] = (a - b) * Complex64::new(theta.cos(), theta.sin());
                }
            }
        }
        for row in 0..8 {
            for y in 0..MACRO_DIM {
                let observed = split[row][y];
                let reference = independent[row * MACRO_DIM + y];
                assert!(
                    (observed - reference).norm() < 1.0e-12,
                    "stage16 DIF mismatch for strip {row}, row coordinate {y}"
                );
            }
        }
    }
}

#[cfg(test)]
mod full_export_layout_tests {
    use super::*;
    #[test]
    fn export_layout_preserves_bijective_macro_slot_geometry() {
        for mr in [0, 1, 3, 7] {
            for mc in [0, 2, 5, 7] {
                for ct in [0, 15, 16, 31] {
                    for slot in [0, 4095, 4096, 32767] {
                        let (y, x) = physical_position(mr, mc, ct, slot);
                        assert!(y >= mr * MACRO_DIM && y < (mr + 1) * MACRO_DIM);
                        assert!(x >= mc * MACRO_DIM && x < (mc + 1) * MACRO_DIM);
                        assert_eq!(full_bit_reverse_13(full_bit_reverse_13(y)), y);
                        assert_eq!(full_bit_reverse_13(full_bit_reverse_13(x)), x);
                    }
                }
            }
        }
    }
}
