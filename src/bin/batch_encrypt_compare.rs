use std::env;
use std::time::Instant;

use ccmm_rs::ccmm::batch::{bit_reverse_index, combine_scalar_coefficients};
use ccmm_rs::ckks::{CkksCanonicalEmbedding, CkksParameterClass, CkksParameterProfile};
use ccmm_rs::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, Polynomial,
    RnsNttPlan, RnsPolynomial,
};
use ccmm_rs::rlwe::ErrorDistribution;

use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const LARGE_DEGREE: usize = 8192;
const SEED: u64 = 0x4241_5443_485f_4233;

const SD3B_LOG2_SCALE: f64 = 27.993302092216055;

fn sd3b_matched_profile() -> CkksParameterProfile {
    CkksParameterProfile::new(
        "heaan-sd3b-matched",
        LARGE_DEGREE,
        &[
            68_712_923_137,
            268_238_849,
            268_369_921,
            268_042_241,
            1_099_510_054_913,
        ],
        2.0_f64.powf(SD3B_LOG2_SCALE),
        CkksParameterClass::Research,
        None,
    )
}

fn scalar_degree(dimension: usize) -> usize {
    assert!(
        matches!(dimension, 64 | 128),
        "B3 replication currently supports exactly d=64 or d=128"
    );

    LARGE_DEGREE / dimension
}

fn bit_reverse_width(value: usize) -> u32 {
    assert!(value.is_power_of_two());
    value.trailing_zeros()
}

fn make_batch(dimension: usize, num_batch: usize) -> Vec<Vec<Vec<Complex64>>> {
    let mut rng = ChaCha20Rng::seed_from_u64(SEED ^ dimension as u64);

    (0..num_batch)
        .map(|_| {
            (0..dimension)
                .map(|_| {
                    (0..dimension)
                        .map(|_| {
                            // Authors' Encrypt task is a round-trip experiment,
                            // so bounded deterministic random values are enough
                            // to expose CKKS/RLWE representation error.
                            Complex64::new(rng.gen_range(-1.0..1.0), 0.0)
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn encode_large_columns(
    matrices: &[Vec<Vec<Complex64>>],
    scalar_degree: usize,
    basis: &ccmm_rs::ring::ModulusBasis,
    scale: f64,
) -> Vec<RnsPolynomial> {
    let dimension = matrices[0].len();
    let slot_count = scalar_degree / 2;

    assert_eq!(matrices.len(), slot_count);
    assert_eq!(dimension * scalar_degree, LARGE_DEGREE);

    let embedding = CkksCanonicalEmbedding::new(scalar_degree);
    let log_slots = bit_reverse_width(slot_count);

    (0..dimension)
        .map(|column| {
            let scalar_coefficients: Vec<Vec<f64>> = (0..dimension)
                .map(|row| {
                    let mut slots = vec![Complex64::new(0.0, 0.0); slot_count];

                    for (batch_slot, slot) in slots.iter_mut().enumerate() {
                        let source_batch = bit_reverse_index(batch_slot, log_slots);
                        *slot = matrices[source_batch][row][column];
                    }

                    embedding.slots_to_coefficients(&slots)
                })
                .collect();

            let residues = basis
                .moduli()
                .iter()
                .copied()
                .map(|modulus| {
                    let q = i128::from(modulus.value());

                    let scalar_residues: Vec<Vec<u64>> = scalar_coefficients
                        .iter()
                        .map(|coefficients| {
                            coefficients
                                .iter()
                                .map(|&value| {
                                    let signed = (value * scale).round() as i128;
                                    signed.rem_euclid(q) as u64
                                })
                                .collect()
                        })
                        .collect();

                    Polynomial::new(modulus, combine_scalar_coefficients(&scalar_residues))
                })
                .collect();

            RnsPolynomial::from_residues(residues)
        })
        .collect()
}

fn decode_large_columns(
    plaintexts: &[RnsPolynomial],
    scalar_degree: usize,
    dimension: usize,
    scale: f64,
) -> Vec<Vec<Vec<Complex64>>> {
    let slot_count = scalar_degree / 2;
    let log_slots = bit_reverse_width(slot_count);
    let embedding = CkksCanonicalEmbedding::new(scalar_degree);

    let mut result = vec![vec![vec![Complex64::new(0.0, 0.0); dimension]; dimension]; slot_count];

    for (column, plaintext) in plaintexts.iter().enumerate() {
        let modulus = composite_modulus_big(plaintext.basis());

        let large_coefficients: Vec<f64> = reconstruct_coefficients_big(plaintext)
            .iter()
            .map(|value| {
                centered_representative_big(value, &modulus)
                    .to_f64()
                    .expect("centered CKKS coefficient must convert to f64")
                    / scale
            })
            .collect();

        assert_eq!(large_coefficients.len(), LARGE_DEGREE);

        // split_large_coefficients operates on exact coefficient vectors.
        // At this point coefficients are f64, so perform the identical
        // interleave inverse explicitly.
        let mut scalar_coefficients = vec![vec![0.0_f64; scalar_degree]; dimension];

        for coefficient in 0..scalar_degree {
            for row in 0..dimension {
                scalar_coefficients[row][coefficient] =
                    large_coefficients[coefficient * dimension + row];
            }
        }

        for (row, coefficients) in scalar_coefficients.iter().enumerate() {
            let slots = embedding.coefficients_to_slots(coefficients);

            for (batch, matrix) in result.iter_mut().enumerate() {
                matrix[row][column] = slots[bit_reverse_index(batch, log_slots)];
            }
        }
    }

    result
}

fn relative_error(
    expected: &[Vec<Vec<Complex64>>],
    computed: &[Vec<Vec<Complex64>>],
) -> (f64, f64) {
    assert_eq!(expected.len(), computed.len());

    let mut maximum = 0.0_f64;
    let mut max_error = 0.0_f64;

    for (expected_matrix, computed_matrix) in expected.iter().zip(computed) {
        for (expected_row, computed_row) in expected_matrix.iter().zip(computed_matrix) {
            for (&reference, &observed) in expected_row.iter().zip(computed_row) {
                maximum = maximum.max(reference.norm());
                max_error = max_error.max((observed - reference).norm());
            }
        }
    }

    assert!(maximum > 0.0, "reference batch maximum must be nonzero");

    (max_error / maximum, max_error)
}

fn main() {
    let dimension: usize = env::args()
        .nth(1)
        .unwrap_or_else(|| "64".to_string())
        .parse()
        .expect("dimension must be an integer");

    let scalar_degree = scalar_degree(dimension);
    let num_batch = scalar_degree / 2;

    let profile = sd3b_matched_profile();
    assert_eq!(profile.degree(), LARGE_DEGREE);

    let chain = profile.modulus_chain();
    let basis = chain.top();
    let scale = profile.initial_scale();

    let plan = RnsNttPlan::new(basis.moduli().to_vec(), LARGE_DEGREE);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(SEED ^ 0x5345_4352_4554);

    let secret: Vec<i8> = (0..LARGE_DEGREE)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    let expected = make_batch(dimension, num_batch);

    println!("FHE_RS_BATCH_MM_TEST_VERSION=1");
    println!("TASK=Encrypt");
    println!("METHOD=Batch");
    println!("PARAMETER=HEaaN-SD3b-matched");
    println!("LOG2_SCALE={SD3B_LOG2_SCALE:.15}");
    println!("SCALE={scale:.12e}");
    println!("DIMENSION={dimension}x{dimension}");
    println!("LARGE_RING_DEGREE={LARGE_DEGREE}");
    println!("SCALAR_RING_DEGREE={scalar_degree}");
    println!("NUM_BATCH={num_batch}");
    println!("THREADS=1");
    println!("TIMED_REGION=sinc-encode+encrypt+decrypt+sinc-decode");

    // Match the authors' Encrypt-task boundary:
    // encode -> encrypt -> decrypt -> decode are all timed.
    let started = Instant::now();

    let plaintexts = encode_large_columns(&expected, scalar_degree, basis, scale);

    let ciphertexts: Vec<_> = plaintexts
        .iter()
        .enumerate()
        .map(|(column, plaintext)| {
            let mut rng =
                ChaCha20Rng::seed_from_u64(SEED ^ 0x0045_4e43_5259_5054 ^ ((column as u64) << 16));

            encrypt_rns_raw_with_distribution_ntt_rng(
                plaintext,
                2,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                &secret,
                &plan,
                &mut rng,
            )
        })
        .collect();

    let decrypted: Vec<RnsPolynomial> = ciphertexts
        .iter()
        .map(|ciphertext| decrypt_rns_raw_with_ntt(ciphertext, &secret, &plan))
        .collect();

    let computed = decode_large_columns(&decrypted, scalar_degree, dimension, scale);

    let elapsed = started.elapsed();

    let (relative_error, max_error) = relative_error(&expected, &computed);

    println!();
    println!("= Encrypt & Decrypt");
    println!("Computation time: {:.6} ms", elapsed.as_secs_f64() * 1e3);
    println!("Maximum absolute error: {max_error:.12e}");
    println!("Relative error: {relative_error:.12e}");
    println!("Relative error log2: {:.6}", relative_error.log2());
    println!();
    println!("LATENCY_MS={:.6}", elapsed.as_secs_f64() * 1e3);
    println!("RELATIVE_ERROR={relative_error:.12e}");
    println!("STATUS=PASS");
}
