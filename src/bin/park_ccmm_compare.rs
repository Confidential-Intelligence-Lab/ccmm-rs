//! End-to-end Park CKKS CC-MM correctness and latency comparison.
//!
//! The timed region contains only:
//!
//!     Park Algorithm 8
//!       + relinearization
//!       + one CKKS rescale
//!
//! Matrix generation, cleartext reference computation, encryption,
//! evaluation-key generation, decryption, reconstruction, validation,
//! and printing are excluded from the timed region.

use std::env;
use std::hint::black_box;
use std::time::Instant;

use ccmm_rs::ccmm::park::{rns_ckks_ccmm_relinearize_rescale_with_backend, ParkModPpMmBackend};
use ccmm_rs::ckks::{CkksChainState, RnsCkksCiphertext, RnsGaloisKey};
use ccmm_rs::grafting::{
    decrypt_rns_raw, encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout,
    RnsMultiplicationKey,
};
use ccmm_rs::ring::{Modulus, ModulusBasis, ModulusChain, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const MODULI: [u64; 3] = [12_289, 40_961, 65_537];
const RELATIVE_ERROR_TOLERANCE: f64 = 1.0e-3;

fn secret_coefficients(degree: usize) -> Vec<i8> {
    const PATTERN: [i8; 8] = [-1, 0, 1, 1, 0, -1, 1, 0];

    (0..degree)
        .map(|index| PATTERN[index % PATTERN.len()])
        .collect()
}

fn lhs_matrix(degree: usize) -> Vec<Vec<f64>> {
    (0..degree)
        .map(|row| {
            (0..degree)
                .map(|col| {
                    0.015 + 0.002 * row as f64 - 0.001 * col as f64 + 0.0002 * (row * col) as f64
                })
                .collect()
        })
        .collect()
}

fn rhs_matrix(degree: usize) -> Vec<Vec<f64>> {
    (0..degree)
        .map(|row| {
            (0..degree)
                .map(|col| {
                    -0.010 + 0.0015 * row as f64 + 0.001 * col as f64 - 0.0001 * (row * col) as f64
                })
                .collect()
        })
        .collect()
}

fn clear_matrix_product(lhs: &[Vec<f64>], rhs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let degree = lhs.len();

    let mut output = vec![vec![0.0_f64; degree]; degree];

    for row in 0..degree {
        for column in 0..degree {
            output[row][column] = (0..degree).map(|k| lhs[row][k] * rhs[k][column]).sum();
        }
    }

    output
}

fn encode_encrypt_columns(
    matrix: &[Vec<f64>],
    scale: f64,
    chain: &ModulusChain,
    secret: &[i8],
    plan: &RnsNttPlan,
    seed: u64,
) -> Vec<RnsCkksCiphertext> {
    let degree = matrix.len();
    let basis = chain.top();

    assert!(matrix.iter().all(|row| row.len() == degree));

    let scaled: Vec<Vec<i128>> = matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| (value * scale).round() as i128)
                .collect()
        })
        .collect();

    (0..degree)
        .map(|column| {
            let residues = basis
                .moduli()
                .iter()
                .copied()
                .map(|modulus| {
                    let q = i128::from(modulus.value());

                    Polynomial::new(
                        modulus,
                        (0..degree)
                            .map(|row| {
                                let value = scaled[row][column];
                                (((value % q) + q) % q) as u64
                            })
                            .collect(),
                    )
                })
                .collect();

            let plaintext = RnsPolynomial::from_residues(residues);

            let mut rng = ChaCha20Rng::seed_from_u64(seed ^ ((column as u64) << 16));

            let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                secret,
                plan,
                &mut rng,
            );

            RnsCkksCiphertext::new(rlwe, CkksChainState::top(chain, scale), chain)
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

fn decrypt_matrix(ciphertexts: &[RnsCkksCiphertext], secret: &[i8]) -> Vec<Vec<f64>> {
    let degree = ciphertexts.len();
    let output_scale = ciphertexts[0].scale();

    let mut matrix = vec![vec![0.0_f64; degree]; degree];

    for (column, ciphertext) in ciphertexts.iter().enumerate() {
        let decrypted = decrypt_rns_raw(ciphertext.rlwe(), secret);
        let modulus = decrypted.composite_modulus();

        for (row, value) in decrypted.reconstruct_coefficients().into_iter().enumerate() {
            let centered = if value > modulus / 2 {
                value as i128 - modulus as i128
            } else {
                value as i128
            };

            matrix[row][column] = centered as f64 / output_scale;
        }
    }

    matrix
}

fn error_metrics(expected: &[Vec<f64>], actual: &[Vec<f64>]) -> (f64, (usize, usize), f64, f64) {
    let degree = expected.len();

    let mut max_abs_error = 0.0_f64;
    let mut max_error_at = (0_usize, 0_usize);
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;

    for row in 0..degree {
        for column in 0..degree {
            let error = actual[row][column] - expected[row][column];
            let abs_error = error.abs();

            if abs_error > max_abs_error {
                max_abs_error = abs_error;
                max_error_at = (row, column);
            }

            squared_error += error * error;
            squared_reference += expected[row][column] * expected[row][column];
        }
    }

    let relative_l2_error = if squared_reference > 0.0 {
        (squared_error / squared_reference).sqrt()
    } else {
        squared_error.sqrt()
    };

    let max_reference = expected
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    assert!(
        max_reference > 0.0,
        "authors-style relative error requires a positive reference maximum"
    );

    let relative_error = max_abs_error / max_reference;

    (
        max_abs_error,
        max_error_at,
        relative_l2_error,
        relative_error,
    )
}

fn print_matrix_prefix(name: &str, matrix: &[Vec<f64>]) {
    let rows = matrix.len().min(4);
    let cols = matrix[0].len().min(4);

    println!("{name}");

    for row in matrix.iter().take(rows) {
        for value in row.iter().take(cols) {
            print!("{value:>13.7}");
        }

        if matrix[0].len() > cols {
            print!(" ...");
        }

        println!();
    }

    if matrix.len() > rows {
        println!("...");
    }
}

fn print_expected_vs_computed(expected: &[Vec<f64>], actual: &[Vec<f64>]) {
    let rows = expected.len().min(4);
    let cols = expected[0].len().min(4);

    println!("= CCMM");
    println!("Expected                              Computed");

    for row in 0..rows {
        for column in 0..cols {
            println!(
                "[{row:>3},{column:>3}] {:>16.9e}    {:>16.9e}",
                expected[row][column], actual[row][column],
            );
        }
    }

    if expected.len() > rows || expected[0].len() > cols {
        println!("...");
    }
}

fn main() {
    let degree: usize = env::args()
        .nth(1)
        .expect("usage: park_ccmm_compare <8|64|128>")
        .parse()
        .expect("dimension must be an integer");

    assert!(
        matches!(degree, 8 | 64 | 128),
        "supported dimensions are 8 (smoke), 64, and 128"
    );

    let moduli: Vec<Modulus> = MODULI.iter().copied().map(Modulus::new).collect();

    let basis = ModulusBasis::new(moduli);
    let chain = ModulusChain::from_top_basis(basis.clone());
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

    let secret = secret_coefficients(degree);

    // Match the tested CKKS path: the input scale equals the top-level
    // modulus that will be dropped by the single rescale.
    let input_scale = basis
        .moduli()
        .last()
        .expect("basis must be nonempty")
        .value() as f64;

    let lhs_clear = lhs_matrix(degree);
    let rhs_clear = rhs_matrix(degree);

    println!("FHE_RS_PARK_CCMM_COMPARE_VERSION=1");
    println!("METHOD=ParkAlgorithm8");
    println!("DIMENSION={degree}x{degree}");
    println!("RING_DEGREE={degree}");
    println!("RNS_LIMBS={}", basis.len());
    println!(
        "MODULI={}",
        basis
            .moduli()
            .iter()
            .map(|modulus| modulus.value().to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("BACKEND=ParallelWide");
    println!("RAYON_THREADS={}", rayon::current_num_threads());
    println!("INPUT_SCALE={input_scale:.12e}");
    println!("SETUP_INCLUDED_IN_TIMING=NO");
    println!("TIMED_REGION=park-ccmm+relinearization+rescale");
    println!("ITERATIONS=1");

    println!();
    print_matrix_prefix("Input A (prefix)", &lhs_clear);
    println!();
    print_matrix_prefix("Input B (prefix)", &rhs_clear);

    println!();
    println!("Computing cleartext MM...");
    let expected = clear_matrix_product(&lhs_clear, &rhs_clear);
    println!("Computed cleartext MM");

    println!();
    println!("Encrypting matrices...");

    let lhs = encode_encrypt_columns(
        &lhs_clear,
        input_scale,
        &chain,
        &secret,
        &plan,
        0xC5A0_0000 ^ degree as u64,
    );

    let rhs = encode_encrypt_columns(
        &rhs_clear,
        input_scale,
        &chain,
        &secret,
        &plan,
        0xC5B0_0000 ^ degree as u64,
    );

    println!("Encrypted matrices");

    println!("Generating evaluation keys...");

    let keys = galois_keys(degree, &basis, &secret, 0xC5C0_0000 ^ degree as u64);

    let layout = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);
    let mut multiplication_key_rng = ChaCha20Rng::seed_from_u64(0xC5D0_0000 ^ degree as u64);

    let multiplication_key = RnsMultiplicationKey::generate_with_rng(
        degree,
        2,
        0,
        &secret,
        layout,
        &mut multiplication_key_rng,
    );

    println!("Generated evaluation keys");

    println!();
    println!("Computing CCMM...");

    let start = Instant::now();

    let output = rns_ckks_ccmm_relinearize_rescale_with_backend(
        black_box(&lhs),
        black_box(&rhs),
        black_box(&keys),
        black_box(&multiplication_key),
        black_box(&chain),
        ParkModPpMmBackend::ParallelWide,
    );

    let elapsed = start.elapsed();

    black_box(&output);

    println!("CCMM computed");

    assert_eq!(
        output.len(),
        degree,
        "Park CKKS CC-MM must return one ciphertext per output column"
    );

    assert!(
        output.iter().all(|ciphertext| ciphertext.level() == 1),
        "Park CKKS CC-MM comparison requires exactly one rescale"
    );

    let output_scale = output[0].scale();

    println!("Decrypting result...");
    let actual = decrypt_matrix(&output, &secret);
    println!("Decrypted result");

    let (max_abs_error, max_error_at, relative_l2_error, relative_error) =
        error_metrics(&expected, &actual);

    let log2_relative_l2_error = if relative_l2_error > 0.0 {
        relative_l2_error.log2()
    } else {
        f64::NEG_INFINITY
    };

    let log2_relative_error = if relative_error > 0.0 {
        relative_error.log2()
    } else {
        f64::NEG_INFINITY
    };

    println!();
    print_expected_vs_computed(&expected, &actual);

    println!();
    println!("Computation time: {:.3} ms", elapsed.as_secs_f64() * 1.0e3);
    println!("Output scale: {output_scale:.12e}");
    println!("Max absolute error: {max_abs_error:.12e}");
    println!("Max error at: {},{}", max_error_at.0, max_error_at.1);
    println!("Relative error: {relative_error:.12e}");
    println!("Relative error log2: {log2_relative_error:.6}");
    println!("Relative L2 error: {relative_l2_error:.12e}");
    println!("Relative L2 error log2: {log2_relative_l2_error:.6}");

    println!();
    println!("LATENCY_MS={:.6}", elapsed.as_secs_f64() * 1.0e3);
    println!("OUTPUT_SCALE={output_scale:.12e}");
    println!("MAX_ABS_ERROR={max_abs_error:.12e}");
    println!("MAX_ERROR_AT={},{}", max_error_at.0, max_error_at.1);
    println!("RELATIVE_ERROR={relative_error:.12e}");
    println!("LOG2_RELATIVE_ERROR={log2_relative_error:.6}");
    println!("RELATIVE_L2_ERROR={relative_l2_error:.12e}");
    println!("LOG2_RELATIVE_L2_ERROR={log2_relative_l2_error:.6}");
    println!(
        "RESULT,{degree},ParallelWide,{:.6},{max_abs_error:.12e},{relative_error:.12e},{relative_l2_error:.12e}",
        elapsed.as_secs_f64() * 1.0e3
    );

    if relative_error <= RELATIVE_ERROR_TOLERANCE {
        println!("STATUS=PASS");
    } else {
        println!("STATUS=FAIL");
        panic!(
            "Park CKKS CC-MM relative error {relative_error:.12e} \
             exceeds tolerance {:.12e}",
            RELATIVE_ERROR_TOLERANCE
        );
    }
}
