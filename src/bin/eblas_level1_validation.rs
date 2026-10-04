use ccmm_rs::ckks::{
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::fft::{
    execute_fft1_plan, execute_fft1_plan_cp, fft1_butterfly_cp, Fft1Plan, Fft1Shape, FftDirection,
};
use ccmm_rs::eblas::{add_cc, axpy_cp, scale_complex_cp, scale_cp};
use ccmm_rs::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng};
use ccmm_rs::matrix::RnsCkksCiphertextMatrix;
use ccmm_rs::ring::{ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const SIGMA: f64 = 3.19;
const TOLERANCE: f64 = 3.0e-3;

fn encode_rns(
    value: f64,
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    let slots = vec![Complex64::new(value, 0.0); embedding.slot_count()];
    let coefficients = embedding.slots_to_coefficients(&slots);
    let residues = basis
        .moduli()
        .iter()
        .copied()
        .map(|modulus| {
            let q = i128::from(modulus.value());
            Polynomial::new(
                modulus,
                coefficients
                    .iter()
                    .map(|&x| ((x * scale).round() as i128).rem_euclid(q) as u64)
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

#[allow(clippy::too_many_arguments)]
fn encrypt_scalar(
    value: f64,
    seed: u64,
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
    secret: &[i8],
    plan: &RnsNttPlan,
    chain: &ccmm_rs::ring::ModulusChain,
    distribution: ErrorDistribution,
) -> RnsCkksCiphertext {
    let plaintext = encode_rns(value, embedding, basis, scale);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        distribution,
        secret,
        plan,
        &mut rng,
    );
    RnsCkksCiphertext::new(rlwe, CkksChainState::top(chain, scale), chain)
}

#[allow(clippy::too_many_arguments)]
fn encrypt_vector(
    values: &[f64],
    seed_base: u64,
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
    secret: &[i8],
    plan: &RnsNttPlan,
    chain: &ccmm_rs::ring::ModulusChain,
    distribution: ErrorDistribution,
) -> RnsCkksCiphertextMatrix {
    let data = values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            encrypt_scalar(
                value,
                seed_base ^ index as u64,
                embedding,
                basis,
                scale,
                secret,
                plan,
                chain,
                distribution,
            )
        })
        .collect();
    RnsCkksCiphertextMatrix::from_vec_column_major(values.len(), 1, data)
}

fn decode_complex_scalar(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Complex64 {
    let plan = RnsNttPlan::new(
        ciphertext.basis().moduli().to_vec(),
        ciphertext.rlwe().degree(),
    );
    let plaintext = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);
    let modulus = plaintext.composite_modulus();
    let coefficients: Vec<f64> = plaintext
        .reconstruct_coefficients()
        .into_iter()
        .map(|x| centered(x, modulus) as f64 / ciphertext.scale())
        .collect();

    let slots = embedding.coefficients_to_slots(&coefficients);

    slots.iter().copied().sum::<Complex64>() / slots.len() as f64
}

fn decode_scalar(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> f64 {
    decode_complex_scalar(ciphertext, secret, embedding).re
}

fn decode_vector(
    matrix: &RnsCkksCiphertextMatrix,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<f64> {
    (0..matrix.rows())
        .map(|row| decode_scalar(matrix.get(row, 0), secret, embedding))
        .collect()
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn main() {
    let profile = research_profile_4096();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let initial_scale = profile.initial_scale();
    let basis = chain.top().clone();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x35D1_0000);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let x = [1.25, -2.0, 0.375, 4.5];
    let y = [-0.5, 1.25, 2.0, -3.0];
    let alpha = -0.625_f64;
    let complex_alpha = Complex64::new(0.6, -0.8);

    let x_ct = encrypt_vector(
        &x,
        0x35D1_1000,
        &embedding,
        &basis,
        initial_scale,
        &secret,
        &plan,
        &chain,
        distribution,
    );
    let y_ct = encrypt_vector(
        &y,
        0x35D1_2000,
        &embedding,
        &basis,
        initial_scale,
        &secret,
        &plan,
        &chain,
        distribution,
    );

    let keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &keys);

    let added = add_cc(&evaluator, &x_ct, &y_ct);
    let scaled = scale_cp(&x_ct, alpha, &embedding, &chain, &plan);
    let complex_scaled = scale_complex_cp(&x_ct, complex_alpha, &embedding, &chain, &plan);
    let axpy = axpy_cp(&evaluator, alpha, &x_ct, &y_ct, &embedding, &chain, &plan);

    let add_expected: Vec<f64> = x.iter().zip(y).map(|(&a, b)| a + b).collect();
    let scale_expected: Vec<f64> = x.iter().map(|&value| alpha * value).collect();
    let axpy_expected: Vec<f64> = x
        .iter()
        .zip(y)
        .map(|(&x_value, y_value)| alpha * x_value + y_value)
        .collect();

    let add_actual = decode_vector(&added, &secret, &embedding);
    let scale_actual = decode_vector(&scaled, &secret, &embedding);
    let complex_scale_actual: Vec<Complex64> = (0..complex_scaled.rows())
        .map(|row| decode_complex_scalar(complex_scaled.get(row, 0), &secret, &embedding))
        .collect();
    let axpy_actual = decode_vector(&axpy, &secret, &embedding);

    let add_error = max_error(&add_actual, &add_expected);
    let scale_error = max_error(&scale_actual, &scale_expected);
    let complex_scale_error = complex_scale_actual
        .iter()
        .zip(x.iter())
        .map(|(actual, &value)| (*actual - complex_alpha * value).norm())
        .fold(0.0_f64, f64::max);
    let axpy_error = max_error(&axpy_actual, &axpy_expected);

    let input_level = x_ct.get(0, 0).level();
    let add_level = added.get(0, 0).level();
    let scale_level = scaled.get(0, 0).level();
    let complex_scale_level = complex_scaled.get(0, 0).level();
    let axpy_level = axpy.get(0, 0).level();

    let input_scale = x_ct.get(0, 0).scale();
    let add_scale = added.get(0, 0).scale();
    let scale_scale = scaled.get(0, 0).scale();
    let complex_scale_scale = complex_scaled.get(0, 0).scale();
    let axpy_scale = axpy.get(0, 0).scale();

    let dropped = chain
        .dropped_modulus(input_level)
        .expect("top level must have a dropped modulus")
        .value() as f64;
    let expected_scale_after_scale = input_scale * dropped / dropped;
    let scale_relative_error =
        ((scale_scale - expected_scale_after_scale) / expected_scale_after_scale).abs();
    let complex_scale_relative_error =
        ((complex_scale_scale - expected_scale_after_scale) / expected_scale_after_scale).abs();
    let axpy_scale_relative_error = ((axpy_scale - input_scale) / input_scale).abs();

    let add_pass = add_error <= TOLERANCE && add_level == input_level && add_scale == input_scale;
    let scale_pass = scale_error <= TOLERANCE
        && scale_level == input_level + 1
        && scale_relative_error <= 1.0e-12;
    let complex_scale_pass = complex_scale_error <= TOLERANCE
        && complex_scale_level == input_level + 1
        && complex_scale_relative_error <= 1.0e-12;
    let axpy_pass = axpy_error <= TOLERANCE
        && axpy_level == input_level + 1
        && axpy_scale_relative_error <= 1.0e-12;

    println!("R3_5D_EBLAS_LEVEL1_VERSION=2");
    println!("PROFILE=research-4096");
    println!("VECTOR_LENGTH={}", x.len());
    println!("ALPHA={alpha:.12}");
    println!("INPUT_LEVEL={input_level}");
    println!("INPUT_SCALE={input_scale:.12e}");
    println!("ADD_LEVEL={add_level}");
    println!("ADD_SCALE={add_scale:.12e}");
    println!("ADD_MAX_ERROR={add_error:.12e}");
    println!("ADD_STATUS={}", if add_pass { "PASS" } else { "FAIL" });
    println!("SCALE_LEVEL={scale_level}");
    println!("SCALE_SCALE={scale_scale:.12e}");
    println!("SCALE_RELATIVE_SCALE_ERROR={scale_relative_error:.12e}");
    println!("SCALE_MAX_ERROR={scale_error:.12e}");
    println!("SCALE_STATUS={}", if scale_pass { "PASS" } else { "FAIL" });
    println!(
        "COMPLEX_ALPHA={:.12}{:+.12}i",
        complex_alpha.re, complex_alpha.im,
    );
    println!("COMPLEX_SCALE_LEVEL={complex_scale_level}");
    println!("COMPLEX_SCALE_SCALE={complex_scale_scale:.12e}");
    println!("COMPLEX_SCALE_RELATIVE_SCALE_ERROR={complex_scale_relative_error:.12e}");
    println!("COMPLEX_SCALE_MAX_ERROR={complex_scale_error:.12e}");
    println!(
        "COMPLEX_SCALE_STATUS={}",
        if complex_scale_pass { "PASS" } else { "FAIL" }
    );
    println!("AXPY_LEVEL={axpy_level}");
    println!("AXPY_SCALE={axpy_scale:.12e}");
    println!("AXPY_RELATIVE_SCALE_ERROR={axpy_scale_relative_error:.12e}");
    println!("AXPY_MAX_ERROR={axpy_error:.12e}");
    println!("AXPY_STATUS={}", if axpy_pass { "PASS" } else { "FAIL" });
    println!(
        "R3_5D_EBLAS_LEVEL1_STATUS={}",
        if add_pass && scale_pass && complex_scale_pass && axpy_pass {
            "PASS"
        } else {
            "FAIL"
        }
    );

    assert!(add_pass, "eBLAS ADD validation failed");
    assert!(scale_pass, "eBLAS SCALE validation failed");
    assert!(complex_scale_pass, "eBLAS complex SCALE validation failed");
    assert!(axpy_pass, "eBLAS AXPY validation failed");

    let fft_twiddles = [
        ("ONE", Complex64::new(1.0, 0.0)),
        ("NEG_ONE", Complex64::new(-1.0, 0.0)),
        ("I", Complex64::new(0.0, 1.0)),
        ("NEG_I", Complex64::new(0.0, -1.0)),
        (
            "EXP_NEG_I_PI_4",
            Complex64::new(
                std::f64::consts::FRAC_1_SQRT_2,
                -std::f64::consts::FRAC_1_SQRT_2,
            ),
        ),
    ];

    for (name, twiddle) in fft_twiddles {
        let (upper, lower) =
            fft1_butterfly_cp(&evaluator, &x_ct, &y_ct, twiddle, &embedding, &chain, &plan);

        assert_eq!(upper.level(), x_ct.level() + 1);
        assert_eq!(lower.level(), x_ct.level() + 1);
        assert_eq!(upper.scale(), x_ct.scale());
        assert_eq!(lower.scale(), x_ct.scale());
        assert_eq!(upper.get(0, 0).basis(), chain.level(x_ct.level() + 1));
        assert_eq!(lower.get(0, 0).basis(), chain.level(x_ct.level() + 1));

        let mut max_upper_error = 0.0_f64;
        let mut max_lower_error = 0.0_f64;

        for row in 0..x.len() {
            let expected_product = twiddle * Complex64::new(y[row], 0.0);
            let expected_upper = Complex64::new(x[row], 0.0) + expected_product;
            let expected_lower = Complex64::new(x[row], 0.0) - expected_product;

            let actual_upper = decode_complex_scalar(upper.get(row, 0), &secret, &embedding);
            let actual_lower = decode_complex_scalar(lower.get(row, 0), &secret, &embedding);

            max_upper_error = max_upper_error.max((actual_upper - expected_upper).norm());
            max_lower_error = max_lower_error.max((actual_lower - expected_lower).norm());
        }

        println!(
            "FFT1_BUTTERFLY_CP_CASE={} TWIDDLE=({:.12e},{:.12e}) LEVEL={} SCALE={:.12e} U_MAX_ABS={:.12e} V_MAX_ABS={:.12e}",
            name,
            twiddle.re,
            twiddle.im,
            upper.level(),
            upper.scale(),
            max_upper_error,
            max_lower_error,
        );

        assert!(
            max_upper_error < TOLERANCE,
            "FFT1 CP butterfly upper-output error {max_upper_error:e}"
        );
        assert!(
            max_lower_error < TOLERANCE,
            "FFT1 CP butterfly lower-output error {max_lower_error:e}"
        );
    }

    for &(fft_length, salt) in &[(2usize, 0x1200_u64), (4usize, 0x1400_u64)] {
        let fft_shape = Fft1Shape::new(fft_length);
        let fft_plan = Fft1Plan::new(fft_shape);

        let input_values: Vec<f64> = (0..fft_length)
            .map(|index| {
                let raw = ((index * 7 + fft_length * 5) % 19) as f64;
                (raw - 9.0) / 5.0
            })
            .collect();

        let encrypted_values: Vec<RnsCkksCiphertextMatrix> = input_values
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                encrypt_vector(
                    &[value],
                    salt ^ index as u64,
                    &embedding,
                    &basis,
                    initial_scale,
                    &secret,
                    &plan,
                    &chain,
                    distribution,
                )
            })
            .collect();

        let expected_input: Vec<Complex64> = input_values
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect();

        let expected_plan = execute_fft1_plan(&fft_plan, FftDirection::Forward, &expected_input);

        let actual_encrypted = execute_fft1_plan_cp(
            &fft_plan,
            FftDirection::Forward,
            &encrypted_values,
            &evaluator,
            &embedding,
            &chain,
        );

        let actual: Vec<Complex64> = actual_encrypted
            .iter()
            .map(|value| decode_complex_scalar(value.get(0, 0), &secret, &embedding))
            .collect();

        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for (actual_value, expected_value) in actual.iter().zip(&expected_plan) {
            let error = *actual_value - *expected_value;

            squared_error += error.norm_sqr();
            squared_reference += expected_value.norm_sqr();
            max_abs = max_abs.max(error.norm());
        }

        let rel_l2 = if squared_reference > 0.0 {
            (squared_error / squared_reference).sqrt()
        } else {
            squared_error.sqrt()
        };

        let output_level = actual_encrypted[0].level();
        let expected_output_level =
            encrypted_values[0].level() + fft_plan.stage_count().saturating_sub(1);

        println!(
            "FFT1_CP_FULL_CASE=N{} STAGES={} BUTTERFLIES={} INPUT_LEVEL={} OUTPUT_LEVEL={} LEVELS_CONSUMED={} REL_L2={:.12e} MAX_ABS={:.12e}",
            fft_length,
            fft_plan.stage_count(),
            fft_plan.butterfly_count(),
            encrypted_values[0].level(),
            output_level,
            output_level - encrypted_values[0].level(),
            rel_l2,
            max_abs,
        );

        assert_eq!(
            output_level, expected_output_level,
            "encrypted FFT1 output level must match stage depth"
        );
        assert!(
            rel_l2 < TOLERANCE,
            "encrypted FFT1 relative L2 error {rel_l2:e}"
        );
        assert!(
            max_abs < TOLERANCE,
            "encrypted FFT1 maximum absolute error {max_abs:e}"
        );
    }
}
