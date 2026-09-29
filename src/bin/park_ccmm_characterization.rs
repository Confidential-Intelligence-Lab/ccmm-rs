//! Park Algorithm 8 quadratic-core characterization.
//!
//! Setup, encryption, and Galois-key generation are outside the timed region.
//! The measured Park path attributes execution time to:
//!
//!   * three logical C-MT operations;
//!   * four logical Mod-PP-MM operations;
//!   * four physical Mod-PP-MM kernels per active RNS limb;
//!   * residual representation/reconstruction work.
//!
//! In the current Park representation, matrix order equals RLWE ring degree.

use std::hint::black_box;

use ccmm_rs::ccmm::park::{
    rns_ccmm_quadratic_with_backend_measured, rns_transpose_measured, ParkCcmmMeasurement,
    ParkCmtMeasurement, ParkModPpMmBackend,
};
use ccmm_rs::ckks::RnsGaloisKey;
use ccmm_rs::grafting::{
    encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout, RnsRlweCiphertext,
};
use ccmm_rs::ring::{Modulus, ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const MODULI: [u64; 3] = [12_289, 40_961, 65_537];
const WARMUPS: usize = 1;
const REPEATS: usize = 5;
const DEGREES: &[usize] = &[8, 16, 32, 64];

#[derive(Debug, Clone, Copy)]
struct Summary {
    total_ns: u128,
    cmt_ns: u128,
    mod_pp_mm_ns: u128,
    residual_ns: u128,
}

#[derive(Debug, Clone, Copy)]
struct CmtSummary {
    total_ns: u128,
    input_shift_ns: u128,
    first_tweak_ns: u128,
    normalization_ns: u128,
    automorphism_ns: u128,
    key_switch_ns: u128,
    second_tweak_ns: u128,
    correction_ns: u128,
    residual_ns: u128,
}

fn moduli() -> Vec<Modulus> {
    MODULI.iter().copied().map(Modulus::new).collect()
}

fn secret_coefficients(degree: usize) -> Vec<i8> {
    const PATTERN: [i8; 8] = [-1, 0, 1, 1, 0, -1, 1, 0];

    (0..degree)
        .map(|index| PATTERN[index % PATTERN.len()])
        .collect()
}

fn deterministic_matrix(degree: usize, salt: u64) -> Vec<Vec<u64>> {
    (0..degree)
        .map(|row| {
            (0..degree)
                .map(|col| {
                    salt.wrapping_add(17 * row as u64)
                        .wrapping_add(29 * col as u64)
                        .wrapping_add(5 * row as u64 * col as u64)
                })
                .collect()
        })
        .collect()
}

fn encrypt_columns(
    matrix: &[Vec<u64>],
    basis: &ModulusBasis,
    secret: &[i8],
    plan: &RnsNttPlan,
    seed: u64,
) -> Vec<RnsRlweCiphertext> {
    let degree = matrix.len();

    assert!(matrix.iter().all(|row| row.len() == degree));

    (0..degree)
        .map(|column| {
            let residues = basis
                .moduli()
                .iter()
                .copied()
                .map(|modulus| {
                    Polynomial::new(
                        modulus,
                        (0..degree)
                            .map(|row| matrix[row][column] % modulus.value())
                            .collect(),
                    )
                })
                .collect();

            let plaintext = RnsPolynomial::from_residues(residues);

            let mut rng = ChaCha20Rng::seed_from_u64(seed ^ ((column as u64) << 16));

            encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                secret,
                plan,
                &mut rng,
            )
        })
        .collect()
}

fn encrypt_rows(
    matrix: &[Vec<u64>],
    basis: &ModulusBasis,
    secret: &[i8],
    plan: &RnsNttPlan,
    seed: u64,
) -> Vec<RnsRlweCiphertext> {
    let degree = matrix.len();

    assert!(matrix.iter().all(|row| row.len() == degree));

    matrix
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            let residues = basis
                .moduli()
                .iter()
                .copied()
                .map(|modulus| {
                    Polynomial::new(
                        modulus,
                        row.iter().map(|&value| value % modulus.value()).collect(),
                    )
                })
                .collect();

            let plaintext = RnsPolynomial::from_residues(residues);

            let mut rng = ChaCha20Rng::seed_from_u64(seed ^ ((row_index as u64) << 16));

            encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                secret,
                plan,
                &mut rng,
            )
        })
        .collect()
}

fn galois_keys(degree: usize, basis: &ModulusBasis, secret: &[i8], seed: u64) -> Vec<RnsGaloisKey> {
    let layout = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);

    (1..degree)
        .map(|j| 2 * j + 1)
        .map(|exponent| {
            let mut rng = ChaCha20Rng::seed_from_u64(seed ^ exponent as u64);

            RnsGaloisKey::generate_with_rng(
                degree,
                2,
                0,
                secret,
                exponent,
                layout.clone(),
                &mut rng,
            )
        })
        .collect()
}

fn median_u128(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn summarize(samples: &[ParkCcmmMeasurement]) -> Summary {
    let mut total: Vec<_> = samples
        .iter()
        .map(|measurement| measurement.total_ns)
        .collect();

    let mut cmt: Vec<_> = samples
        .iter()
        .map(|measurement| measurement.cmt_ns)
        .collect();

    let mut mod_pp_mm: Vec<_> = samples
        .iter()
        .map(|measurement| measurement.mod_pp_mm_ns)
        .collect();

    let mut residual: Vec<_> = samples
        .iter()
        .map(|measurement| measurement.residual_ns())
        .collect();

    Summary {
        total_ns: median_u128(&mut total),
        cmt_ns: median_u128(&mut cmt),
        mod_pp_mm_ns: median_u128(&mut mod_pp_mm),
        residual_ns: median_u128(&mut residual),
    }
}

fn summarize_cmt(samples: &[ParkCmtMeasurement]) -> CmtSummary {
    fn median_field(
        samples: &[ParkCmtMeasurement],
        field: impl Fn(&ParkCmtMeasurement) -> u128,
    ) -> u128 {
        let mut values: Vec<_> = samples.iter().map(field).collect();
        median_u128(&mut values)
    }

    CmtSummary {
        total_ns: median_field(samples, |m| m.total_ns),
        input_shift_ns: median_field(samples, |m| m.input_shift_ns),
        first_tweak_ns: median_field(samples, |m| m.first_tweak_ns),
        normalization_ns: median_field(samples, |m| m.normalization_ns),
        automorphism_ns: median_field(samples, |m| m.automorphism_ns),
        key_switch_ns: median_field(samples, |m| m.key_switch_ns),
        second_tweak_ns: median_field(samples, |m| m.second_tweak_ns),
        correction_ns: median_field(samples, |m| m.correction_ns),
        residual_ns: median_field(samples, |m| m.residual_ns()),
    }
}

fn characterize_cmt(
    degree: usize,
    ciphertexts: &[RnsRlweCiphertext],
    galois_keys: &[RnsGaloisKey],
) -> CmtSummary {
    for _ in 0..WARMUPS {
        let (output, measurement) =
            rns_transpose_measured(black_box(ciphertexts), black_box(galois_keys));

        assert_eq!(measurement.degree, degree);
        assert_eq!(measurement.rns_limbs, MODULI.len());
        assert_eq!(measurement.automorphism_calls, degree - 1);
        assert_eq!(measurement.key_switch_calls, degree - 1);

        black_box(output);
    }

    let mut samples = Vec::with_capacity(REPEATS);

    for _ in 0..REPEATS {
        let (output, measurement) =
            rns_transpose_measured(black_box(ciphertexts), black_box(galois_keys));

        black_box(output);
        samples.push(measurement);
    }

    summarize_cmt(&samples)
}

fn backend_name(backend: ParkModPpMmBackend) -> &'static str {
    match backend {
        ParkModPpMmBackend::Reference => "Reference",
        ParkModPpMmBackend::TransposedRhs => "TransposedRhs",
    }
}

fn characterize(
    degree: usize,
    backend: ParkModPpMmBackend,
    lhs: &[RnsRlweCiphertext],
    rhs: &[RnsRlweCiphertext],
    galois_keys: &[RnsGaloisKey],
) -> Summary {
    for _ in 0..WARMUPS {
        let (output, measurement) = rns_ccmm_quadratic_with_backend_measured(
            black_box(lhs),
            black_box(rhs),
            black_box(galois_keys),
            backend,
        );

        assert_eq!(measurement.degree, degree);
        assert_eq!(measurement.rns_limbs, MODULI.len());
        assert_eq!(measurement.backend, backend);
        assert_eq!(measurement.cmt_calls, 3);
        assert_eq!(measurement.logical_mod_pp_mm_calls, 4);
        assert_eq!(measurement.physical_mod_pp_mm_calls, 4 * MODULI.len());

        black_box(output);
    }

    let mut samples = Vec::with_capacity(REPEATS);

    for _ in 0..REPEATS {
        let (output, measurement) = rns_ccmm_quadratic_with_backend_measured(
            black_box(lhs),
            black_box(rhs),
            black_box(galois_keys),
            backend,
        );

        black_box(output);
        samples.push(measurement);
    }

    summarize(&samples)
}

fn pct(part: u128, total: u128) -> f64 {
    if total == 0 {
        0.0
    } else {
        100.0 * part as f64 / total as f64
    }
}

fn us(ns: u128) -> f64 {
    ns as f64 / 1_000.0
}

fn main() {
    println!("PARK_CCMM_CHARACTERIZATION_VERSION=1");
    println!("MATRIX_ORDER_EQUALS_RING_DEGREE=YES");
    println!("RNS_LIMBS={}", MODULI.len());
    println!("MODULI={},{},{}", MODULI[0], MODULI[1], MODULI[2]);
    println!("WARMUPS={WARMUPS}");
    println!("REPEATS={REPEATS}");
    println!("TIMING_STATISTIC=median");
    println!("TIMED_REGION=park-rns-quadratic-algorithm8");
    println!("SETUP_INCLUDED_IN_TIMING=NO");

    println!(
        "CSV_HEADER=degree,rns_limbs,backend,total_us,cmt_us,\
mod_pp_mm_us,residual_us,cmt_pct,mod_pp_mm_pct,residual_pct,\
logical_cmt,logical_mod_pp_mm,physical_mod_pp_mm,\
speedup_vs_reference"
    );

    println!(
        "CMT_CSV_HEADER=degree,rns_limbs,total_us,input_shift_us,\
first_tweak_us,normalization_us,automorphism_us,key_switch_us,\
second_tweak_us,correction_us,residual_us,key_switch_pct,\
automorphism_pct,tweak_pct"
    );

    for &degree in DEGREES {
        let moduli = moduli();
        let basis = ModulusBasis::new(moduli);

        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        let secret = secret_coefficients(degree);

        let lhs_plain = deterministic_matrix(degree, 0x5100_0000 ^ degree as u64);

        let rhs_plain = deterministic_matrix(degree, 0x5200_0000 ^ degree as u64);

        let lhs = encrypt_columns(
            &lhs_plain,
            &basis,
            &secret,
            &plan,
            0x5300_0000 ^ degree as u64,
        );

        let rhs = encrypt_columns(
            &rhs_plain,
            &basis,
            &secret,
            &plan,
            0x5400_0000 ^ degree as u64,
        );

        let keys = galois_keys(degree, &basis, &secret, 0x5500_0000 ^ degree as u64);

        let cmt_input = encrypt_rows(
            &lhs_plain,
            &basis,
            &secret,
            &plan,
            0x5600_0000 ^ degree as u64,
        );

        let cmt = characterize_cmt(degree, &cmt_input, &keys);

        let tweak_ns = cmt.first_tweak_ns + cmt.second_tweak_ns;

        println!(
            "CMT_RESULT,{},{},{:.3},{:.3},{:.3},{:.3},{:.3},\
{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            degree,
            MODULI.len(),
            us(cmt.total_ns),
            us(cmt.input_shift_ns),
            us(cmt.first_tweak_ns),
            us(cmt.normalization_ns),
            us(cmt.automorphism_ns),
            us(cmt.key_switch_ns),
            us(cmt.second_tweak_ns),
            us(cmt.correction_ns),
            us(cmt.residual_ns),
            pct(cmt.key_switch_ns, cmt.total_ns),
            pct(cmt.automorphism_ns, cmt.total_ns),
            pct(tweak_ns, cmt.total_ns),
        );

        let reference = characterize(degree, ParkModPpMmBackend::Reference, &lhs, &rhs, &keys);

        let transposed = characterize(degree, ParkModPpMmBackend::TransposedRhs, &lhs, &rhs, &keys);

        for (backend, summary) in [
            (ParkModPpMmBackend::Reference, reference),
            (ParkModPpMmBackend::TransposedRhs, transposed),
        ] {
            let speedup = reference.total_ns as f64 / summary.total_ns as f64;

            println!(
                "RESULT,{},{},{},{:.3},{:.3},{:.3},{:.3},\
{:.3},{:.3},{:.3},{},{},{},{:.6}",
                degree,
                MODULI.len(),
                backend_name(backend),
                us(summary.total_ns),
                us(summary.cmt_ns),
                us(summary.mod_pp_mm_ns),
                us(summary.residual_ns),
                pct(summary.cmt_ns, summary.total_ns),
                pct(summary.mod_pp_mm_ns, summary.total_ns),
                pct(summary.residual_ns, summary.total_ns),
                3,
                4,
                4 * MODULI.len(),
                speedup,
            );
        }
    }

    println!("PARK_CCMM_CHARACTERIZATION_STATUS=PASS");
}
