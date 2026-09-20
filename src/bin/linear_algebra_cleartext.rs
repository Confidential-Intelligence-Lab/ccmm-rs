use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

use ccmm_rs::matrix::BatchMatrix;

const VERSION: u32 = 1;
const WARMUP: usize = 1;
const REPEATS: usize = 5;

#[derive(Debug, Clone, Copy)]
struct Workload {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
}

#[derive(Debug)]
struct Measurement {
    elapsed: Duration,
    checksum: f64,
}

fn workload_sets(profile: &str) -> Vec<Workload> {
    match profile {
        "smoke" => vec![
            Workload {
                name: "square-2",
                m: 2,
                k: 2,
                n: 2,
            },
            Workload {
                name: "square-8",
                m: 8,
                k: 8,
                n: 8,
            },
            Workload {
                name: "gemv-1x16-by-16x1",
                m: 1,
                k: 16,
                n: 1,
            },
            Workload {
                name: "rect-4x16-by-16x8",
                m: 4,
                k: 16,
                n: 8,
            },
        ],
        "standard" => vec![
            Workload {
                name: "square-16",
                m: 16,
                k: 16,
                n: 16,
            },
            Workload {
                name: "square-32",
                m: 32,
                k: 32,
                n: 32,
            },
            Workload {
                name: "square-64",
                m: 64,
                k: 64,
                n: 64,
            },
            Workload {
                name: "square-128",
                m: 128,
                k: 128,
                n: 128,
            },
            Workload {
                name: "gemv-1x256-by-256x1",
                m: 1,
                k: 256,
                n: 1,
            },
            Workload {
                name: "mlp-64x256-by-256x128",
                m: 64,
                k: 256,
                n: 128,
            },
            Workload {
                name: "mlp-64x128-by-128x64",
                m: 64,
                k: 128,
                n: 64,
            },
            Workload {
                name: "conv1x1-1024x64-by-64x64",
                m: 1024,
                k: 64,
                n: 64,
            },
        ],
        "frontier" => vec![
            Workload {
                name: "square-2048",
                m: 2048,
                k: 2048,
                n: 2048,
            },
            Workload {
                name: "square-4096",
                m: 4096,
                k: 4096,
                n: 4096,
            },
        ],
        "extended" => vec![
            Workload {
                name: "square-256",
                m: 256,
                k: 256,
                n: 256,
            },
            Workload {
                name: "square-512",
                m: 512,
                k: 512,
                n: 512,
            },
            Workload {
                name: "square-1024",
                m: 1024,
                k: 1024,
                n: 1024,
            },
            Workload {
                name: "conv1x1-4096x64-by-64x64",
                m: 4096,
                k: 64,
                n: 64,
            },
            Workload {
                name: "fc-1024x1024-by-1024x10",
                m: 1024,
                k: 1024,
                n: 10,
            },
        ],
        other => panic!("unknown profile {other}; expected smoke, standard, extended, or frontier"),
    }
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

fn make_matrix(rows: usize, cols: usize, salt: u64) -> BatchMatrix<f64> {
    let mut data = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            let logical_index = row
                .checked_add(col.checked_mul(rows).expect("matrix index overflow"))
                .expect("matrix index overflow");
            data.push(deterministic_value(logical_index, salt));
        }
    }
    BatchMatrix::from_vec_column_major(rows, cols, 1, data)
}

fn matmul_reference(lhs: &BatchMatrix<f64>, rhs: &BatchMatrix<f64>) -> BatchMatrix<f64> {
    assert_eq!(lhs.batches(), 1);
    assert_eq!(rhs.batches(), 1);
    assert_eq!(lhs.cols(), rhs.rows());

    let mut out = BatchMatrix::<f64>::new(lhs.rows(), rhs.cols(), 1);

    for col in 0..rhs.cols() {
        for row in 0..lhs.rows() {
            let mut sum = 0.0_f64;
            for inner in 0..lhs.cols() {
                sum += *lhs.get(0, row, inner) * *rhs.get(0, inner, col);
            }
            out.set(0, row, col, sum);
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
    let result = black_box(matmul_reference(black_box(lhs), black_box(rhs)));
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
    let args: Vec<String> = env::args().collect();
    let profile = args
        .windows(2)
        .find(|pair| pair[0] == "--profile")
        .map(|pair| pair[1].as_str())
        .unwrap_or("smoke");

    let workloads = workload_sets(profile);

    println!("R3_4B_CLEARTEXT_LINEAR_ALGEBRA_VERSION={VERSION}");
    println!("PROFILE={profile}");
    println!("LAYOUT=batch-major,column-major-within-batch");
    println!("BATCHES=1");
    println!("WARMUP={WARMUP}");
    println!("REPEATS={REPEATS}");
    println!("TIMING_STATISTIC=median");
    println!("IMPLEMENTATION=dependency-free-reference-triple-loop");
    println!("BOOTSTRAPPING=not-applicable");

    for workload in workloads {
        let lhs = make_matrix(workload.m, workload.k, 0x34B0_1001);
        let rhs = make_matrix(workload.k, workload.n, 0x34B0_2002);

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
                    "non-deterministic checksum for {}: reference={reference}, observed={}",
                    workload.name,
                    measurement.checksum
                );
            } else {
                observed_checksum = Some(measurement.checksum);
            }
        }

        let median = median_us(&mut times_us);
        let scalar_multiplies = (workload.m as u128)
            .checked_mul(workload.k as u128)
            .and_then(|value| value.checked_mul(workload.n as u128))
            .expect("multiply count overflow");
        let scalar_additions = (workload.m as u128)
            .checked_mul(workload.n as u128)
            .and_then(|value| value.checked_mul(workload.k.saturating_sub(1) as u128))
            .expect("addition count overflow");
        let flops = scalar_multiplies
            .checked_mul(2)
            .expect("FLOP count overflow");
        let seconds = median * 1.0e-6;
        let gflops = flops as f64 / seconds / 1.0e9;

        println!();
        println!("WORKLOAD={}", workload.name);
        println!("M={}", workload.m);
        println!("K={}", workload.k);
        println!("N={}", workload.n);
        println!("LHS_ELEMENTS={}", workload.m * workload.k);
        println!("RHS_ELEMENTS={}", workload.k * workload.n);
        println!("OUTPUT_ELEMENTS={}", workload.m * workload.n);
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
    println!("R3_4B_CLEARTEXT_LINEAR_ALGEBRA_STATUS=PASS");
}
