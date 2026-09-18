use std::{env, time::Instant};

use ccmm_rs::ckks::{
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsCkksLevelKeys,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout,
    RnsKeygenConfig, RnsMultiplicationKey,
};
use ccmm_rs::matrix::RnsCkksCiphertextMatrix;
use ccmm_rs::ring::{Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn encode_rns(
    slots: &[Complex64],
    embedding: &CkksCanonicalEmbedding,
    basis: &ccmm_rs::ring::ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    let raw = embedding.slots_to_coefficients(slots);

    let signed: Vec<i128> = raw
        .iter()
        .map(|&coefficient| {
            let scaled = coefficient * scale;
            assert!(scaled.is_finite(), "scaled CKKS coefficient must be finite");
            scaled.round() as i128
        })
        .collect();

    let residues = basis
        .moduli()
        .iter()
        .copied()
        .map(|modulus| {
            let q = i128::from(modulus.value());

            Polynomial::new(
                modulus,
                signed
                    .iter()
                    .map(|&value| {
                        let residue = ((value % q) + q) % q;
                        residue as u64
                    })
                    .collect(),
            )
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

fn centered(value: u128, modulus: u128) -> i128 {
    if value > modulus / 2 {
        value as i128 - modulus as i128
    } else {
        value as i128
    }
}

fn main() {
    let dimension: usize = env::args()
        .nth(1)
        .map(|value| value.parse().expect("matrix dimension must be an integer"))
        .unwrap_or(2);

    assert!(
        dimension == 2 || dimension == 4,
        "benchmark currently supports matrix dimensions 2 or 4"
    );

    let profile = research_profile_4096();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let slot_count = profile.slot_count();
    let scale = profile.initial_scale();
    let top_basis = chain.top().clone();
    let top_plan = RnsNttPlan::new(top_basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);

    assert_eq!(degree, 4096);
    assert_eq!(slot_count, 2048);
    assert_eq!(chain.len(), 3);
    assert!(!profile.security_bearing());

    let secret: Vec<i8> = (0..degree)
        .map(|index| match index % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 1,
        })
        .collect();

    /*
     * Every scalar matrix entry is replicated across all CKKS slots.
     *
     * Column-major:
     *
     * A = [  0.25    0.50  ]
     *     [ -0.25    0.125 ]
     *
     * B = [ 0.50   -0.25 ]
     *     [ 0.25    0.50 ]
     */
    let lhs_values: Vec<f64> = (0..dimension * dimension)
        .map(|index| {
            let row = index % dimension;
            let col = index / dimension;
            ((row + 1) as f64 * 0.0625) - ((col + 1) as f64 * 0.03125)
        })
        .collect();

    let rhs_values: Vec<f64> = (0..dimension * dimension)
        .map(|index| {
            let row = index % dimension;
            let col = index / dimension;
            ((col + 1) as f64 * 0.046875) - ((row + 1) as f64 * 0.015625)
        })
        .collect();

    let mut expected = vec![vec![0.0_f64; dimension]; dimension];

    for (row, expected_row) in expected.iter_mut().enumerate() {
        for (col, expected_value) in expected_row.iter_mut().enumerate() {
            let mut sum = 0.0;

            for inner in 0..dimension {
                let lhs = lhs_values[row + inner * dimension];
                let rhs = rhs_values[inner + col * dimension];
                sum += lhs * rhs;
            }

            *expected_value = sum;
        }
    }

    let start = Instant::now();

    let lhs_plaintexts: Vec<_> = lhs_values
        .iter()
        .map(|&value| {
            let slots = vec![Complex64::new(value, 0.0); slot_count];
            encode_rns(&slots, &embedding, &top_basis, scale)
        })
        .collect();

    let rhs_plaintexts: Vec<_> = rhs_values
        .iter()
        .map(|&value| {
            let slots = vec![Complex64::new(value, 0.0); slot_count];
            encode_rns(&slots, &embedding, &top_basis, scale)
        })
        .collect();

    let encode_time = start.elapsed();

    let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };

    let start = Instant::now();

    let lhs_ciphertexts: Vec<_> = lhs_plaintexts
        .iter()
        .enumerate()
        .map(|(index, plaintext)| {
            let mut rng = ChaCha20Rng::seed_from_u64(0xCC10_0000_u64.wrapping_add(index as u64));

            let inner = encrypt_rns_raw_with_distribution_ntt_rng(
                plaintext,
                2,
                distribution,
                &secret,
                &top_plan,
                &mut rng,
            );

            RnsCkksCiphertext::new(inner, CkksChainState::top(&chain, scale), &chain)
        })
        .collect();

    let encrypt_lhs_time = start.elapsed();

    let start = Instant::now();

    let rhs_ciphertexts: Vec<_> = rhs_plaintexts
        .iter()
        .enumerate()
        .map(|(index, plaintext)| {
            let mut rng = ChaCha20Rng::seed_from_u64(0xCC20_0000_u64.wrapping_add(index as u64));

            let inner = encrypt_rns_raw_with_distribution_ntt_rng(
                plaintext,
                2,
                distribution,
                &secret,
                &top_plan,
                &mut rng,
            );

            RnsCkksCiphertext::new(inner, CkksChainState::top(&chain, scale), &chain)
        })
        .collect();

    let encrypt_rhs_time = start.elapsed();

    /*
     * R2.9 limitation:
     *
     * Ciphertext encryption uses Gaussian sigma=3.19.
     * The multiplication key deliberately remains zero-noise because the
     * current gadget/relinearization architecture has not yet been validated
     * for security-bearing noisy evaluation keys.
     */
    let mut key_rng = ChaCha20Rng::seed_from_u64(0xCC30_0000);

    let start = Instant::now();

    let multiplication_key = RnsMultiplicationKey::generate_with_ntt_rng(
        RnsKeygenConfig {
            degree,
            plaintext_modulus: 2,
            noise_bound: 0,
            layout: RnsGadgetLayout::new(top_basis.clone(), vec![1, 2]),
            plan: &top_plan,
        },
        &secret,
        &mut key_rng,
    );

    let eval_keygen_time = start.elapsed();

    let mut level_keys = RnsCkksLevelKeys::new(0, top_basis.clone());
    level_keys.set_multiplication_key(multiplication_key);

    let mut evaluation_keys = RnsCkksEvaluationKeys::new();
    evaluation_keys.insert_level(level_keys);

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let lhs = RnsCkksCiphertextMatrix::from_vec_column_major(dimension, dimension, lhs_ciphertexts);

    let rhs = RnsCkksCiphertextMatrix::from_vec_column_major(dimension, dimension, rhs_ciphertexts);

    let start = Instant::now();
    let result = lhs.matmul_with_ntt(&rhs, &evaluator, &top_plan);
    let ccmm_time = start.elapsed();

    assert_eq!(result.rows(), dimension);
    assert_eq!(result.cols(), dimension);
    assert_eq!(result.level(), 1);

    let output_scale = result.scale();
    let result_plan = RnsNttPlan::new(result.get(0, 0).basis().moduli().to_vec(), degree);

    let start = Instant::now();

    let mut max_slot_error = 0.0_f64;
    let mut sum_slot_error = 0.0_f64;
    let mut sum_squared_slot_error = 0.0_f64;
    let mut error_count = 0_usize;

    for col in 0..dimension {
        for (row, expected_row) in expected.iter().enumerate() {
            let ciphertext = result.get(row, col);

            let decrypted = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), &secret, &result_plan);

            let modulus = decrypted.composite_modulus();

            let coefficients: Vec<f64> = decrypted
                .reconstruct_coefficients()
                .into_iter()
                .map(|value| centered(value, modulus) as f64 / ciphertext.scale())
                .collect();

            let observed = embedding.coefficients_to_slots(&coefficients);
            let reference = Complex64::new(expected_row[col], 0.0);

            for actual in observed {
                let error = (actual - reference).norm();

                max_slot_error = max_slot_error.max(error);
                sum_slot_error += error;
                sum_squared_slot_error += error * error;
                error_count += 1;
            }
        }
    }

    let decrypt_decode_time = start.elapsed();

    let mean_slot_error = sum_slot_error / error_count as f64;
    let rms_slot_error = (sum_squared_slot_error / error_count as f64).sqrt();

    let tolerance = 1.0e-3;
    let status = if max_slot_error < tolerance {
        "PASS"
    } else {
        "FAIL"
    };

    println!("RNS_CCMM_BENCH_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("PARAMETER_CLASS={:?}", profile.parameter_class());
    println!("SECURITY_BEARING={}", profile.security_bearing());
    println!("RING_DEGREE={degree}");
    println!("SLOT_COUNT={slot_count}");
    println!("CHAIN_LEVELS={}", chain.len());
    println!("TOTAL_MODULUS_BITS={}", profile.total_modulus_bits());
    println!("MATRIX_ROWS={dimension}");
    println!("MATRIX_INNER={dimension}");
    println!("MATRIX_COLS={dimension}");
    println!("MULTIPLICATIVE_DEPTH=1");
    println!("SCALAR_MULTIPLIES={}", dimension * dimension * dimension);
    println!(
        "CIPHERTEXT_ADDITIONS={}",
        dimension * dimension * (dimension - 1)
    );
    println!("CIPHERTEXT_ERROR_DISTRIBUTION=discrete-gaussian");
    println!("CIPHERTEXT_ERROR_SIGMA=3.19");
    println!("EVALUATION_KEY_NOISE_BOUND=0");
    println!("INPUT_SCALE={scale:.17e}");
    println!("OUTPUT_SCALE={output_scale:.17e}");
    println!("ENCODE_US={:.6}", encode_time.as_secs_f64() * 1.0e6);
    println!(
        "ENCRYPT_LHS_US={:.6}",
        encrypt_lhs_time.as_secs_f64() * 1.0e6
    );
    println!(
        "ENCRYPT_RHS_US={:.6}",
        encrypt_rhs_time.as_secs_f64() * 1.0e6
    );
    println!(
        "EVAL_KEYGEN_US={:.6}",
        eval_keygen_time.as_secs_f64() * 1.0e6
    );
    println!("CCMM_US={:.6}", ccmm_time.as_secs_f64() * 1.0e6);
    println!(
        "DECRYPT_DECODE_US={:.6}",
        decrypt_decode_time.as_secs_f64() * 1.0e6
    );
    println!("MAX_SLOT_ERROR={max_slot_error:.12e}");
    println!("MEAN_SLOT_ERROR={mean_slot_error:.12e}");
    println!("RMS_SLOT_ERROR={rms_slot_error:.12e}");
    println!("ERROR_TOLERANCE={tolerance:.12e}");
    println!("RNS_CCMM_BENCH_STATUS={status}");

    if status != "PASS" {
        std::process::exit(1);
    }
}
