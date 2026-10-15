//! R16.5.18b.2: randomized auxiliary-P key-switching diagnostics.
//! Small-degree correctness experiment; not a security validation.

use ccmm_rs::grafting::{encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout};
use ccmm_rs::ring::{
    auxiliary_mod_down_reference, composite_modulus_big, reconstruct_coefficients_big,
    rns_from_big_coefficients, Modulus, ModulusBasis, RnsNttPlan, RnsPolynomial,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Signed;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const N: usize = 8;
const BASE_LOG: u32 = 4;

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

fn run_case(seed: u64, gaussian: bool) -> (BigInt, BigInt) {
    let q_basis = ModulusBasis::new(vec![Modulus::new(97), Modulus::new(193)]);
    let p_basis = ModulusBasis::new(vec![Modulus::new(257)]);

    let mut qp_moduli = q_basis.moduli().to_vec();
    qp_moduli.extend_from_slice(p_basis.moduli());

    let qp_basis = ModulusBasis::new(qp_moduli.clone());
    let qp_modulus = composite_modulus_big(&qp_basis);
    let p = BigInt::from(composite_modulus_big(&p_basis));

    let q_plan = RnsNttPlan::new(q_basis.moduli().to_vec(), N);
    let qp_plan = RnsNttPlan::new(qp_moduli.clone(), N);
    let layout = BoundedGadgetLayout::new(q_basis.clone(), BASE_LOG);

    let original_secret: [i8; N] = [1, 0, -1, 1, 0, -1, 1, 0];
    let transformed_secret: [i8; N] = [-1, 1, 0, 1, -1, 0, 0, 1];

    let original_signed = original_secret
        .iter()
        .map(|&x| BigInt::from(x))
        .collect::<Vec<_>>();

    let transformed_signed = transformed_secret
        .iter()
        .map(|&x| BigInt::from(x))
        .collect::<Vec<_>>();

    let b_values = [17, -11, 29, 3, -19, 7, 13, -5]
        .into_iter()
        .map(BigInt::from)
        .collect::<Vec<_>>();

    let a_values = [127, -88, 301, -194, 63, 91, -117, 205]
        .into_iter()
        .map(BigInt::from)
        .collect::<Vec<_>>();

    let b_q = signed_poly(q_basis.moduli(), &b_values);
    let a_q = signed_poly(q_basis.moduli(), &a_values);

    let original_s_q = signed_poly(q_basis.moduli(), &original_signed);
    let transformed_s_q = signed_poly(q_basis.moduli(), &transformed_signed);

    let expected = b_q.add(&q_plan.negacyclic_mul(&a_q, &transformed_s_q));

    let decomposition = layout.decompose(&a_q);

    assert_eq!(
        decomposition.reconstruct_coefficients_big(),
        reconstruct_coefficients_big(&a_q)
    );

    let mut accumulator_b = signed_poly(
        &qp_moduli,
        &b_values.iter().map(|x| x * &p).collect::<Vec<_>>(),
    );
    let mut accumulator_a = RnsPolynomial::zero(qp_moduli.clone(), N);

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let distribution = if gaussian {
        ErrorDistribution::DiscreteGaussian { sigma: 3.19 }
    } else {
        ErrorDistribution::BoundedUniform { bound: 0 }
    };

    let mut gadget_factor = BigInt::from(1_u8);

    for digit_index in 0..layout.digit_count() {
        let digit = decomposition
            .digit(digit_index)
            .iter()
            .map(|&x| BigInt::from(x))
            .collect::<Vec<_>>();

        let digit_qp = signed_poly(&qp_moduli, &digit);

        let target = transformed_signed
            .iter()
            .map(|x| x * &p * &gadget_factor)
            .collect::<Vec<_>>();

        let target_qp = signed_poly(&qp_moduli, &target);

        let key = encrypt_rns_raw_with_distribution_ntt_rng(
            &target_qp,
            2,
            distribution,
            &original_secret,
            &qp_plan,
            &mut rng,
        );

        let key_b =
            RnsPolynomial::from_residues(key.limbs().iter().map(|limb| limb.b().clone()).collect());
        let key_a =
            RnsPolynomial::from_residues(key.limbs().iter().map(|limb| limb.a().clone()).collect());

        accumulator_b = accumulator_b.add(&qp_plan.negacyclic_mul(&digit_qp, &key_b));
        accumulator_a = accumulator_a.add(&qp_plan.negacyclic_mul(&digit_qp, &key_a));

        gadget_factor *= BigInt::from(1_u128 << BASE_LOG);
    }

    let original_s_qp = signed_poly(&qp_moduli, &original_signed);

    let decrypted_before =
        accumulator_b.add(&qp_plan.negacyclic_mul(&accumulator_a, &original_s_qp));

    let expected_before = signed_poly(
        &qp_moduli,
        &reconstruct_coefficients_big(&expected)
            .iter()
            .map(|x| BigInt::from(x.clone()) * &p)
            .collect::<Vec<_>>(),
    );

    let before_error = centered_error(&decrypted_before, &expected_before);
    let before_max = max_error(&before_error);

    if !gaussian {
        assert_eq!(
            before_max,
            BigInt::from(0),
            "zero-noise QP identity must be exact before modulus-down"
        );
    }

    // Reference component-wise modulus-down, including rounding.
    let output_b = auxiliary_mod_down_reference(&accumulator_b, &q_basis, &p_basis);
    let output_a = auxiliary_mod_down_reference(&accumulator_a, &q_basis, &p_basis);

    let decrypted_after = output_b.add(&q_plan.negacyclic_mul(&output_a, &original_s_q));

    let after_error = centered_error(&decrypted_after, &expected);
    let after_max = max_error(&after_error);

    let label = if gaussian { "GAUSSIAN" } else { "ZERO_NOISE" };

    println!("R16_5_18B2_MODE={label}");
    println!("R16_5_18B2_SEED={seed}");
    println!("R16_5_18B2_BEFORE_MAX_ERROR={before_max}");
    println!("R16_5_18B2_AFTER_MAX_ERROR={after_max}");

    assert!(
        after_max < BigInt::from(1000),
        "unexpectedly large QP switching error"
    );

    // Confirm that the chosen toy modulus is not being used
    // to make a cryptographic security claim.
    assert!(qp_modulus > composite_modulus_big(&q_basis));

    (before_max, after_max)
}

#[test]
fn r16_5_18b2_randomized_qp_switch() {
    for seed in [11_u64, 29, 47] {
        let (before, after) = run_case(seed, false);
        assert_eq!(before, BigInt::from(0));
        println!("R16_5_18B2_ZERO_NOISE_AFTER={after}");
    }

    for seed in [11_u64, 29, 47] {
        let (before, after) = run_case(seed, true);
        println!("R16_5_18B2_GAUSSIAN_BEFORE={before}");
        println!("R16_5_18B2_GAUSSIAN_AFTER={after}");
    }

    println!("R16_5_18B2_STATUS=PASS");
    println!("R16_5_18B2_SECURITY_STATUS=TOY_PARAMETERS_ONLY");
}
