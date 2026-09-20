use std::hint::black_box;
use std::time::{Duration, Instant};

use ccmm_rs::matrix::BatchMatrix;

const VERSION: u32 = 1;
const WARMUP: usize = 1;
const REPEATS: usize = 5;

#[derive(Debug, Clone, Copy)]
struct TensorWorkload {
    name: &'static str,
    batches: usize,
    m: usize,
    k: usize,
    n: usize,
    interpretation: &'static str,
}

#[derive(Debug)]
struct Measurement {
    elapsed: Duration,
    checksum: f64,
}

fn workloads() -> Vec<TensorWorkload> {
    vec![
        TensorWorkload {
            name: "batched-8x64x64-by-8x64x64",
            batches: 8,
            m: 64,
            k: 64,
            n: 64,
            interpretation: "batched-gemm",
        },
        TensorWorkload {
            name: "batched-16x128x64-by-16x64x128",
            batches: 16,
            m: 128,
            k: 64,
            n: 128,
            interpretation: "batched-gemm",
        },
        TensorWorkload {
            name: "batched-32x32x128-by-32x128x64",
            batches: 32,
            m: 32,
            k: 128,
            n: 64,
            interpretation: "batched-gemm",
        },
        TensorWorkload {
            name: "conv1x1-32x32-c64-f64",
            batches: 1,
            m: 1024,
            k: 64,
            n: 64,
            interpretation: "flattened-1x1-convolution",
        },
        TensorWorkload {
            name: "conv1x1-16x16-c128-f128",
            batches: 1,
            m: 256,
            k: 128,
            n: 128,
            interpretation: "flattened-1x1-convolution",
        },
        TensorWorkload {
            name: "conv1x1-8x8-c256-f256",
            batches: 1,
            m: 64,
            k: 256,
            n: 256,
            interpretation: "flattened-1x1-convolution",
        },
        TensorWorkload {
            name: "im2col-32x32-c3-k3-f64",
            batches: 1,
            m: 1024,
            k: 27,
            n: 64,
            interpretation: "same-shape-im2col-3x3-convolution",
        },
        TensorWorkload {
            name: "im2col-16x16-c64-k3-f64",
            batches: 1,
            m: 256,
            k: 576,
            n: 64,
            interpretation: "same-shape-im2col-3x3-convolution",
        },
        TensorWorkload {
            name: "im2col-8x8-c64-k3-f128",
            batches: 1,
            m: 64,
            k: 576,
            n: 128,
            interpretation: "same-shape-im2col-3x3-convolution",
        },
    ]
}

fn deterministic_value(index: usize, salt: u64) -> f64 {
    let mut x = index as u64 ^ salt;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    let mixed = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
    let signed = (mixed % 2001) as i64 - 1000;
    signed as f64 / 4096.0
}

fn make_matrix(rows: usize, cols: usize, batches: usize, salt: u64) -> BatchMatrix<f64> {
    let elements = rows
        .checked_mul(cols)
        .and_then(|v| v.checked_mul(batches))
        .expect("matrix element count overflow");

    let mut data = Vec::with_capacity(elements);

    for batch in 0..batches {
        for col in 0..cols {
            for row in 0..rows {
                let index = batch
                    .checked_mul(rows * cols)
                    .and_then(|v| v.checked_add(col * rows + row))
                    .expect("matrix index overflow");
                data.push(deterministic_value(index, salt));
            }
        }
    }

    BatchMatrix::from_vec_column_major(rows, cols, batches, data)
}

fn batched_matmul_reference(lhs: &BatchMatrix<f64>, rhs: &BatchMatrix<f64>) -> BatchMatrix<f64> {
    assert_eq!(lhs.batches(), rhs.batches());
    assert_eq!(lhs.cols(), rhs.rows());

    let mut out = BatchMatrix::<f64>::new(lhs.rows(), rhs.cols(), lhs.batches());

    for batch in 0..lhs.batches() {
        for col in 0..rhs.cols() {
            for row in 0..lhs.rows() {
                let mut sum = 0.0_f64;
                for inner in 0..lhs.cols() {
                    sum += *lhs.get(batch, row, inner) * *rhs.get(batch, inner, col);
                }
                out.set(batch, row, col, sum);
            }
        }
    }

    out
}

fn checksum(matrix: &BatchMatrix<f64>) -> f64 {
    matrix
        .raw()
        .iter()
        .enumerate()
        .map(|(index, value)| *value * ((index % 97) + 1) as f64)
        .sum()
}

fn run_once(lhs: &BatchMatrix<f64>, rhs: &BatchMatrix<f64>) -> Measurement {
    let start = Instant::now();
    let result = black_box(batched_matmul_reference(black_box(lhs), black_box(rhs)));
    let elapsed = start.elapsed();

    Measurement {
        elapsed,
        checksum: checksum(&result),
    }
}

fn median_us(values: &mut [f64]) -> f64 {
    values.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    values[values.len() / 2]
}

fn main() {
    println!("R3_4B_TENSOR_CLEARTEXT_VERSION={VERSION}");
    println!("IMPLEMENTATION=dependency-free-reference-batched-gemm");
    println!("LAYOUT=batch-major,column-major-within-batch");
    println!("WARMUP={WARMUP}");
    println!("REPEATS={REPEATS}");
    println!("TIMING_STATISTIC=median");
    println!("BOOTSTRAPPING=not-applicable");

    for workload in workloads() {
        let lhs = make_matrix(workload.m, workload.k, workload.batches, 0x34B4_1001);
        let rhs = make_matrix(workload.k, workload.n, workload.batches, 0x34B4_2002);

        for _ in 0..WARMUP {
            black_box(run_once(&lhs, &rhs));
        }

        let mut times_us = Vec::with_capacity(REPEATS);
        let mut observed_checksum = None;

        for _ in 0..REPEATS {
            let measurement = run_once(&lhs, &rhs);
            times_us.push(measurement.elapsed.as_secs_f64() * 1.0e6);

            if let Some(reference) = observed_checksum {
                let delta: f64 = measurement.checksum - reference;
                assert!(
                    delta.abs() <= 1.0e-12,
                    "non-deterministic checksum for {}",
                    workload.name
                );
            } else {
                observed_checksum = Some(measurement.checksum);
            }
        }

        let median = median_us(&mut times_us);

        let scalar_multiplies = (workload.batches as u128)
            .checked_mul(workload.m as u128)
            .and_then(|v| v.checked_mul(workload.k as u128))
            .and_then(|v| v.checked_mul(workload.n as u128))
            .expect("multiply count overflow");

        let scalar_additions = (workload.batches as u128)
            .checked_mul(workload.m as u128)
            .and_then(|v| v.checked_mul(workload.n as u128))
            .and_then(|v| v.checked_mul(workload.k.saturating_sub(1) as u128))
            .expect("addition count overflow");

        let flops = scalar_multiplies
            .checked_mul(2)
            .expect("FLOP count overflow");

        let gflops = flops as f64 / (median * 1.0e-6) / 1.0e9;

        println!();
        println!("WORKLOAD={}", workload.name);
        println!("INTERPRETATION={}", workload.interpretation);
        println!("BATCHES={}", workload.batches);
        println!("M={}", workload.m);
        println!("K={}", workload.k);
        println!("N={}", workload.n);
        println!(
            "LHS_ELEMENTS={}",
            workload.batches * workload.m * workload.k
        );
        println!(
            "RHS_ELEMENTS={}",
            workload.batches * workload.k * workload.n
        );
        println!(
            "OUTPUT_ELEMENTS={}",
            workload.batches * workload.m * workload.n
        );
        println!("SCALAR_MULTIPLIES={scalar_multiplies}");
        println!("SCALAR_ADDITIONS={scalar_additions}");
        println!("REFERENCE_FLOPS={flops}");
        println!("MEDIAN_US={median:.3}");
        println!("REFERENCE_GFLOP_S={gflops:.6}");
        println!(
            "CHECKSUM={:.17e}",
            observed_checksum.expect("missing checksum")
        );
    }

    println!();
    println!("R3_4B_TENSOR_CLEARTEXT_STATUS=PASS");
}
