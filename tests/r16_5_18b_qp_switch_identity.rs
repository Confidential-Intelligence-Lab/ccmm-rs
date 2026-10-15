//! R16.5.18b.1: deterministic QP key-switching identity.
//!
//! Synthetic evaluation keys only. No cryptographic security claim.

use ccmm_rs::grafting::BoundedGadgetLayout;
use ccmm_rs::ring::{
    auxiliary_mod_down_reference, composite_modulus_big, reconstruct_coefficients_big,
    rns_from_big_coefficients, Modulus, ModulusBasis, RnsNttPlan, RnsPolynomial,
};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Zero;

fn canonical(value: &BigInt, modulus: &BigUint) -> BigUint {
    let m = BigInt::from_biguint(Sign::Plus, modulus.clone());
    let mut r = value % &m;
    if r.sign() == Sign::Minus {
        r += &m;
    }
    r.to_biguint().unwrap()
}

fn polynomial_from_signed(moduli: &[Modulus], coefficients: &[BigInt]) -> RnsPolynomial {
    let basis = ModulusBasis::new(moduli.to_vec());
    let modulus = composite_modulus_big(&basis);
    let residues: Vec<_> = coefficients
        .iter()
        .map(|value| canonical(value, &modulus))
        .collect();

    rns_from_big_coefficients(moduli.to_vec(), &residues)
}

#[test]
fn r16_5_18b_qp_synthetic_switch_identity() {
    const N: usize = 8;
    const BASE_LOG: u32 = 4;

    let q_basis = ModulusBasis::new(vec![Modulus::new(97), Modulus::new(193)]);
    let p_basis = ModulusBasis::new(vec![Modulus::new(257)]);

    let mut qp_moduli = q_basis.moduli().to_vec();
    qp_moduli.extend_from_slice(p_basis.moduli());

    let q = composite_modulus_big(&q_basis);
    let p = composite_modulus_big(&p_basis);
    let p_signed = BigInt::from_biguint(Sign::Plus, p);

    let qp_plan = RnsNttPlan::new(qp_moduli.clone(), N);
    let q_plan = RnsNttPlan::new(q_basis.moduli().to_vec(), N);

    let layout = BoundedGadgetLayout::new(q_basis.clone(), BASE_LOG);

    // Synthetic transformed ciphertext components.
    let b_values: Vec<BigInt> = [17, -11, 29, 3, -19, 7, 13, -5]
        .into_iter()
        .map(BigInt::from)
        .collect();

    let a_values: Vec<BigInt> = [127, -88, 301, -194, 63, 91, -117, 205]
        .into_iter()
        .map(BigInt::from)
        .collect();

    // A nontrivial transformed ternary secret.
    let transformed_secret: Vec<BigInt> = [1, 0, -1, 1, -1, 0, 1, 0]
        .into_iter()
        .map(BigInt::from)
        .collect();

    let b_q = polynomial_from_signed(q_basis.moduli(), &b_values);
    let a_q = polynomial_from_signed(q_basis.moduli(), &a_values);
    let s_q = polynomial_from_signed(q_basis.moduli(), &transformed_secret);

    let expected_product = q_plan.negacyclic_mul(&a_q, &s_q);
    let expected_b = reconstruct_coefficients_big(&b_q);
    let expected_as = reconstruct_coefficients_big(&expected_product);

    let expected: Vec<BigUint> = expected_b
        .iter()
        .zip(expected_as.iter())
        .map(|(b, as_value)| {
            canonical(
                &(BigInt::from(b.clone()) + BigInt::from(as_value.clone())),
                &q,
            )
        })
        .collect();

    // Verify that the bounded decomposition reconstructs a exactly.
    let decomposition = layout.decompose(&a_q);
    let reconstructed = decomposition.reconstruct_coefficients_big();
    assert_eq!(reconstructed, reconstruct_coefficients_big(&a_q));

    // Synthetic key: B_i = P * BASE^i * s', A_i = 0.
    // Accumulate in QP.
    let mut accumulator = polynomial_from_signed(
        &qp_moduli,
        &b_values.iter().map(|v| v * &p_signed).collect::<Vec<_>>(),
    );

    let mut gadget_factor = BigInt::from(1_u8);
    let gadget_base = BigInt::from(1_u128 << BASE_LOG);

    for digit_index in 0..layout.digit_count() {
        let digit_values: Vec<BigInt> = decomposition
            .digit(digit_index)
            .iter()
            .copied()
            .map(BigInt::from)
            .collect();

        let digit_qp = polynomial_from_signed(&qp_moduli, &digit_values);

        let key_values: Vec<BigInt> = transformed_secret
            .iter()
            .map(|s| s * &p_signed * &gadget_factor)
            .collect();

        let key_qp = polynomial_from_signed(&qp_moduli, &key_values);
        let product = qp_plan.negacyclic_mul(&digit_qp, &key_qp);

        let accumulated: Vec<BigUint> = reconstruct_coefficients_big(&accumulator)
            .iter()
            .zip(reconstruct_coefficients_big(&product).iter())
            .map(|(lhs, rhs)| {
                canonical(
                    &(BigInt::from(lhs.clone()) + BigInt::from(rhs.clone())),
                    &composite_modulus_big(&ModulusBasis::new(qp_moduli.clone())),
                )
            })
            .collect();

        accumulator = rns_from_big_coefficients(qp_moduli.clone(), &accumulated);

        gadget_factor *= &gadget_base;
    }

    let switched = auxiliary_mod_down_reference(&accumulator, &q_basis, &p_basis);

    let actual = reconstruct_coefficients_big(&switched);

    assert_eq!(
        actual, expected,
        "synthetic QP switching must reproduce the transformed-secret identity"
    );

    assert!(!actual.iter().all(BigUint::is_zero));

    println!("R16_5_18B1_STATUS=PASS");
    println!("R16_5_18B1_DEGREE={N}");
    println!("R16_5_18B1_GADGET_BASE_LOG={BASE_LOG}");
    println!("R16_5_18B1_GADGET_DIGITS={}", layout.digit_count());
    println!("R16_5_18B1_SECURITY_STATUS=INSECURE_ALGEBRAIC_TEST_ONLY");
}
