use std::time::{Duration, Instant};

use ccmm_rs::ccmm::{ccmm, decrypt_ckks_matrix, encrypt_ckks_matrix_with_rng};
use ccmm_rs::ckks::{project_secret_key_to_low, CkksParameters};
use ccmm_rs::eval::MultiplicationKey;
use ccmm_rs::matrix::BatchMatrix;
use ccmm_rs::rlwe::SecretKey;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const CAMPAIGN_SEEDS: u64 = 32;

#[derive(Debug)]
struct Trial {
    keygen: Duration,
    eval_keygen: Duration,
    encrypt_lhs: Duration,
    encrypt_rhs: Duration,
    ccmm: Duration,
    decrypt: Duration,
    max_abs_error: f64,
    mean_abs_error: f64,
}

fn matrix(values_column_major: [f64; 4]) -> BatchMatrix<f64> {
    BatchMatrix::from_vec_column_major(2, 2, 1, values_column_major.to_vec())
}

fn reference_matmul(lhs: &BatchMatrix<f64>, rhs: &BatchMatrix<f64>) -> BatchMatrix<f64> {
    let mut out = BatchMatrix::<f64>::new(lhs.rows(), rhs.cols(), 1);

    for col in 0..rhs.cols() {
        for row in 0..lhs.rows() {
            let mut sum = 0.0;

            for k in 0..lhs.cols() {
                sum += *lhs.get(0, row, k) * *rhs.get(0, k, col);
            }

            out.set(0, row, col, sum);
        }
    }

    out
}

fn run_trial(
    params: CkksParameters,
    seed: u64,
    lhs: &BatchMatrix<f64>,
    rhs: &BatchMatrix<f64>,
    expected: &BatchMatrix<f64>,
) -> Trial {
    let mut key_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x1000);

    let start = Instant::now();
    let high_secret = SecretKey::generate_with_rng(params.high_rlwe(), &mut key_rng);
    let keygen = start.elapsed();

    let low_secret = project_secret_key_to_low(params, &high_secret);

    let mut eval_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x2000);

    let start = Instant::now();
    let multiplication_key = MultiplicationKey::generate_with_rng(
        params.high_rlwe(),
        &high_secret,
        65_536,
        &mut eval_rng,
    );
    let eval_keygen = start.elapsed();

    let mut lhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x3000);

    let start = Instant::now();
    let lhs_ct = encrypt_ckks_matrix_with_rng(params, &high_secret, lhs, &mut lhs_rng);
    let encrypt_lhs = start.elapsed();

    let mut rhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x4000);

    let start = Instant::now();
    let rhs_ct = encrypt_ckks_matrix_with_rng(params, &high_secret, rhs, &mut rhs_rng);
    let encrypt_rhs = start.elapsed();

    let start = Instant::now();
    let result = ccmm(params, &lhs_ct, &rhs_ct, &multiplication_key);
    let ccmm_time = start.elapsed();

    let start = Instant::now();
    let actual = decrypt_ckks_matrix(params, &low_secret, &result);
    let decrypt = start.elapsed();

    let mut max_abs_error = 0.0_f64;
    let mut sum_abs_error = 0.0_f64;
    let mut count = 0_usize;

    for col in 0..actual.cols() {
        for row in 0..actual.rows() {
            let error = (*actual.get(0, row, col) - *expected.get(0, row, col)).abs();

            max_abs_error = max_abs_error.max(error);
            sum_abs_error += error;
            count += 1;
        }
    }

    Trial {
        keygen,
        eval_keygen,
        encrypt_lhs,
        encrypt_rhs,
        ccmm: ccmm_time,
        decrypt,
        max_abs_error,
        mean_abs_error: sum_abs_error / count as f64,
    }
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn min(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn main() {
    let params = CkksParameters::new(8, 2_147_483_647, 65_537, 65_537.0, 1);

    let lhs = matrix([1.0, 3.0, 2.0, 4.0]);

    let rhs = matrix([5.0, 7.0, 6.0, 8.0]);

    let expected = reference_matmul(&lhs, &rhs);

    let trials: Vec<_> = (0..CAMPAIGN_SEEDS)
        .map(|seed| run_trial(params, seed, &lhs, &rhs, &expected))
        .collect();

    let keygen_us: Vec<_> = trials.iter().map(|t| micros(t.keygen)).collect();

    let eval_keygen_us: Vec<_> = trials.iter().map(|t| micros(t.eval_keygen)).collect();

    let encrypt_lhs_us: Vec<_> = trials.iter().map(|t| micros(t.encrypt_lhs)).collect();

    let encrypt_rhs_us: Vec<_> = trials.iter().map(|t| micros(t.encrypt_rhs)).collect();

    let ccmm_us: Vec<_> = trials.iter().map(|t| micros(t.ccmm)).collect();

    let decrypt_us: Vec<_> = trials.iter().map(|t| micros(t.decrypt)).collect();

    let max_errors: Vec<_> = trials.iter().map(|t| t.max_abs_error).collect();

    let mean_errors: Vec<_> = trials.iter().map(|t| t.mean_abs_error).collect();

    println!("CCMM_BENCH_VERSION=1");
    println!("CAMPAIGN_SEEDS={CAMPAIGN_SEEDS}");
    println!("RING_DEGREE={}", params.degree());
    println!("HIGH_MODULUS={}", params.high_modulus().value());
    println!("LOW_MODULUS={}", params.low_modulus().value());
    println!("RESCALE_PRIME={}", params.rescale_prime());
    println!("INITIAL_SCALE={:.17e}", params.initial_scale());
    println!("MATRIX_ROWS=2");
    println!("MATRIX_COLS=2");

    println!("KEYGEN_MEAN_US={:.6}", mean(&keygen_us));
    println!("EVAL_KEYGEN_MEAN_US={:.6}", mean(&eval_keygen_us));
    println!("ENCRYPT_LHS_MEAN_US={:.6}", mean(&encrypt_lhs_us));
    println!("ENCRYPT_RHS_MEAN_US={:.6}", mean(&encrypt_rhs_us));

    println!("CCMM_MIN_US={:.6}", min(&ccmm_us));
    println!("CCMM_MEAN_US={:.6}", mean(&ccmm_us));
    println!("CCMM_MAX_US={:.6}", max(&ccmm_us));

    println!("DECRYPT_MEAN_US={:.6}", mean(&decrypt_us));

    println!("MAX_ABS_ERROR_MAX={:.17e}", max(&max_errors));

    println!("MEAN_ABS_ERROR_MEAN={:.17e}", mean(&mean_errors));

    let tolerance = 1.0e-2;

    println!("BENCH_ERROR_TOLERANCE={tolerance:.17e}");

    if max(&max_errors) <= tolerance {
        println!("CCMM_BENCH_STATUS=PASS");
    } else {
        println!("CCMM_BENCH_STATUS=FAIL");
        std::process::exit(1);
    }
}
