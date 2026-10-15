//! R16.5.18c.2: paired Gaussian auxiliary-P characterization.
//! Small-degree correctness experiment; not a security validation.

use ccmm_rs::grafting::{encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout};
use ccmm_rs::ring::{
    auxiliary_mod_down_reference, composite_modulus_big, reconstruct_coefficients_big,
    rns_from_big_coefficients, Modulus, ModulusBasis, RnsNttPlan, RnsPolynomial,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{Signed, ToPrimitive};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::time::Instant;

const N: usize = 16;

fn canonical(value: &BigInt, modulus: &BigUint) -> BigUint {
    let m = BigInt::from(modulus.clone());
    let mut result = value % &m;
    if result.sign() == Sign::Minus {
        result += m;
    }
    result.to_biguint().unwrap()
}

fn signed_poly(moduli: &[Modulus], values: &[BigInt]) -> RnsPolynomial {
    let basis = ModulusBasis::new(moduli.to_vec());
    let modulus = composite_modulus_big(&basis);
    let coefficients = values
        .iter()
        .map(|value| canonical(value, &modulus))
        .collect::<Vec<_>>();

    rns_from_big_coefficients(moduli.to_vec(), &coefficients)
}

fn centered_error(actual: &RnsPolynomial, expected: &RnsPolynomial) -> Vec<BigInt> {
    assert_eq!(actual.basis(), expected.basis());

    let modulus = composite_modulus_big(actual.basis());
    let m = BigInt::from(modulus.clone());
    let half = &m / 2;

    reconstruct_coefficients_big(actual)
        .iter()
        .zip(reconstruct_coefficients_big(expected).iter())
        .map(|(a, e)| {
            let mut difference = BigInt::from(a.clone()) - BigInt::from(e.clone());
            difference %= &m;

            if difference > half {
                difference -= &m;
            }
            if difference < -&half {
                difference += &m;
            }
            difference
        })
        .collect()
}

fn max_error(errors: &[BigInt]) -> BigInt {
    errors.iter().map(BigInt::abs).max().unwrap()
}

fn rms_error(errors: &[BigInt]) -> f64 {
    let sum: f64 = errors
        .iter()
        .map(|x| {
            let value = x.to_f64().expect("diagnostic error fits f64");
            value * value
        })
        .sum();
    (sum / errors.len() as f64).sqrt()
}

fn ciphertext_parts(key: &ccmm_rs::grafting::RnsRlweCiphertext) -> (RnsPolynomial, RnsPolynomial) {
    let b = RnsPolynomial::from_residues(key.limbs().iter().map(|limb| limb.b().clone()).collect());
    let a = RnsPolynomial::from_residues(key.limbs().iter().map(|limb| limb.a().clone()).collect());
    (b, a)
}

fn run_configuration(seed: u64, base_log: u32, auxiliary_count: usize) {
    let q_basis = ModulusBasis::new(vec![
        Modulus::new(12_289),
        Modulus::new(40_961),
        Modulus::new(65_537),
    ]);

    let auxiliary_primes = [Modulus::new(114_689), Modulus::new(786_433)];
    assert!((1..=2).contains(&auxiliary_count));

    let p_basis = ModulusBasis::new(auxiliary_primes[..auxiliary_count].to_vec());

    let mut qp_moduli = q_basis.moduli().to_vec();
    qp_moduli.extend_from_slice(p_basis.moduli());

    let p_unsigned = composite_modulus_big(&p_basis);
    let p = BigInt::from(p_unsigned.clone());

    let q_plan = RnsNttPlan::new(q_basis.moduli().to_vec(), N);
    let qp_plan = RnsNttPlan::new(qp_moduli.clone(), N);
    let layout = BoundedGadgetLayout::new(q_basis.clone(), base_log);

    let original_secret: [i8; N] = [1, 0, -1, 1, 0, -1, 1, 0, -1, 0, 1, 0, 1, -1, 0, 1];
    let transformed_secret: [i8; N] = [-1, 1, 0, 1, -1, 0, 0, 1, 1, -1, 0, 0, 1, 0, -1, 1];

    let original_signed = original_secret
        .iter()
        .map(|&x| BigInt::from(x))
        .collect::<Vec<_>>();
    let transformed_signed = transformed_secret
        .iter()
        .map(|&x| BigInt::from(x))
        .collect::<Vec<_>>();

    let b_values = [
        17, -11, 29, 3, -19, 7, 13, -5, 23, -31, 37, -41, 43, -47, 53, -59,
    ]
    .into_iter()
    .map(BigInt::from)
    .collect::<Vec<_>>();

    let a_values = [
        127, -88, 301, -194, 63, 91, -117, 205, 307, -401, 503, -607, 709, -811, 907, -1009,
    ]
    .into_iter()
    .map(BigInt::from)
    .collect::<Vec<_>>();

    let b_q = signed_poly(q_basis.moduli(), &b_values);
    let a_q = signed_poly(q_basis.moduli(), &a_values);

    let original_s_q = signed_poly(q_basis.moduli(), &original_signed);
    let original_s_qp = signed_poly(&qp_moduli, &original_signed);
    let transformed_s_q = signed_poly(q_basis.moduli(), &transformed_signed);

    let expected = b_q.add(&q_plan.negacyclic_mul(&a_q, &transformed_s_q));

    let decomposition = layout.decompose(&a_q);
    assert_eq!(
        decomposition.reconstruct_coefficients_big(),
        reconstruct_coefficients_big(&a_q),
    );

    let initial_b = signed_poly(
        &qp_moduli,
        &b_values.iter().map(|x| x * &p).collect::<Vec<_>>(),
    );

    let mut zero_b = initial_b.clone();
    let mut gaussian_b = initial_b;
    let mut shared_a = RnsPolynomial::zero(qp_moduli.clone(), N);
    let mut predicted_noise = RnsPolynomial::zero(qp_moduli.clone(), N);

    let mut mask_rng = ChaCha20Rng::seed_from_u64(seed);
    let mut noise_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x18c2_0000);

    let mut gadget_factor = BigInt::from(1_u8);

    let keygen_start = Instant::now();

    for index in 0..layout.digit_count() {
        let digit = decomposition
            .digit(index)
            .iter()
            .map(|&x| BigInt::from(x))
            .collect::<Vec<_>>();

        let digit_qp = signed_poly(&qp_moduli, &digit);

        let target_values = transformed_signed
            .iter()
            .map(|x| x * &p * &gadget_factor)
            .collect::<Vec<_>>();

        let target_qp = signed_poly(&qp_moduli, &target_values);

        // The shared zero-noise key supplies the actual uniform mask.
        let zero_key = encrypt_rns_raw_with_distribution_ntt_rng(
            &target_qp,
            2,
            ErrorDistribution::BoundedUniform { bound: 0 },
            &original_secret,
            &qp_plan,
            &mut mask_rng,
        );

        // Generate an independent Gaussian key only to extract its
        // actual sampled error polynomial. Its mask is discarded.
        let noisy_key = encrypt_rns_raw_with_distribution_ntt_rng(
            &target_qp,
            2,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            &original_secret,
            &qp_plan,
            &mut noise_rng,
        );

        let (key_b, key_a) = ciphertext_parts(&zero_key);
        let (noisy_b, noisy_a) = ciphertext_parts(&noisy_key);

        let noisy_decrypted = noisy_b.add(&qp_plan.negacyclic_mul(&noisy_a, &original_s_qp));

        let sampled_error = noisy_decrypted.sub(&target_qp);

        // This injection changes b only. Both switching paths
        // therefore use exactly the same evaluation-key masks.
        let injected_b = key_b.add(&sampled_error);

        zero_b = zero_b.add(&qp_plan.negacyclic_mul(&digit_qp, &key_b));
        gaussian_b = gaussian_b.add(&qp_plan.negacyclic_mul(&digit_qp, &injected_b));
        shared_a = shared_a.add(&qp_plan.negacyclic_mul(&digit_qp, &key_a));
        predicted_noise = predicted_noise.add(&qp_plan.negacyclic_mul(&digit_qp, &sampled_error));

        gadget_factor *= BigInt::from(1_u128 << base_log);
    }

    let keygen_ms = keygen_start.elapsed().as_secs_f64() * 1000.0;

    let switch_start = Instant::now();

    let zero_decrypted = zero_b.add(&qp_plan.negacyclic_mul(&shared_a, &original_s_qp));
    let gaussian_decrypted = gaussian_b.add(&qp_plan.negacyclic_mul(&shared_a, &original_s_qp));

    // Exact, uncontaminated key-noise attribution.
    let observed_noise = gaussian_decrypted.sub(&zero_decrypted);
    assert_eq!(
        centered_error(&observed_noise, &predicted_noise),
        vec![BigInt::from(0); N],
        "observed QP noise differs from gadget-weighted key noise",
    );

    let expected_before = signed_poly(
        &qp_moduli,
        &reconstruct_coefficients_big(&expected)
            .iter()
            .map(|x| BigInt::from(x.clone()) * &p)
            .collect::<Vec<_>>(),
    );

    assert_eq!(
        max_error(&centered_error(&zero_decrypted, &expected_before,)),
        BigInt::from(0),
        "zero-noise switching must be exact before modulus-down",
    );

    let zero_out_b = auxiliary_mod_down_reference(&zero_b, &q_basis, &p_basis);
    let gaussian_out_b = auxiliary_mod_down_reference(&gaussian_b, &q_basis, &p_basis);
    let out_a = auxiliary_mod_down_reference(&shared_a, &q_basis, &p_basis);

    let secret_product = q_plan.negacyclic_mul(&out_a, &original_s_q);

    let zero_output = zero_out_b.add(&secret_product);
    let gaussian_output = gaussian_out_b.add(&secret_product);

    let rounding_error = centered_error(&zero_output, &expected);
    let total_error = centered_error(&gaussian_output, &expected);
    let gaussian_delta = centered_error(&gaussian_output, &zero_output);
    let key_noise = centered_error(&observed_noise, &RnsPolynomial::zero(qp_moduli.clone(), N));

    let rounding_max = max_error(&rounding_error);
    let total_max = max_error(&total_error);
    let gaussian_delta_max = max_error(&gaussian_delta);
    let key_noise_max = max_error(&key_noise);

    // Each rounded component contributes at most one integer unit;
    // decryption convolves the a-component rounding with a ternary
    // secret. This is a conservative deterministic rounding bound.
    let secret_l1: i64 = original_secret.iter().map(|&x| i64::from(x).abs()).sum();
    let rounding_bound = BigInt::from(1 + secret_l1);

    assert!(
        rounding_max <= rounding_bound,
        "zero-noise rounding exceeded deterministic bound"
    );

    // Difference between two rounded b-components is bounded by
    // the unrounded noise divided by P, plus one unit of rounding.
    // The shared a-component cancels exactly.
    let gaussian_bound = (&key_noise_max + &p - BigInt::from(1)) / &p + BigInt::from(1);

    assert!(
        gaussian_delta_max <= gaussian_bound,
        "Gaussian output delta exceeded derived modulus-down bound"
    );

    let switching_ms = switch_start.elapsed().as_secs_f64() * 1000.0;

    let key_polynomials = layout.digit_count() * 2;
    let key_bytes = key_polynomials * qp_moduli.len() * N * std::mem::size_of::<u64>();

    println!("R16_5_18C2_CASE=seed:{seed},base:{base_log},p_count:{auxiliary_count}");
    println!("R16_5_18C2_DEGREE={N}");
    println!("R16_5_18C2_P_BITS={}", p_unsigned.bits());
    println!("R16_5_18C2_DIGITS={}", layout.digit_count());
    println!("R16_5_18C2_KEY_NOISE_MAX={key_noise_max}");
    println!("R16_5_18C2_ROUNDING_MAX={rounding_max}");
    println!("R16_5_18C2_GAUSSIAN_DELTA_MAX={gaussian_delta_max}");
    println!("R16_5_18C2_TOTAL_MAX={total_max}");
    println!("R16_5_18C2_TOTAL_RMS={:.6}", rms_error(&total_error));
    println!("R16_5_18C2_GAUSSIAN_BOUND={gaussian_bound}");
    println!("R16_5_18C2_KEYGEN_MS={keygen_ms:.3}");
    println!("R16_5_18C2_SWITCHING_MS={switching_ms:.3}");
    println!("R16_5_18C2_KEY_BYTES={key_bytes}");
}

#[test]
fn r16_5_18c2_qp_noise_characterization() {
    for auxiliary_count in [1_usize, 2] {
        for base_log in [4_u32, 8, 12] {
            for seed in [11_u64, 29, 47] {
                run_configuration(seed, base_log, auxiliary_count);
            }
        }
    }

    println!("R16_5_18C2_STATUS=PASS");
    println!("R16_5_18C2_SECURITY_STATUS=TOY_PARAMETERS_ONLY");
}
