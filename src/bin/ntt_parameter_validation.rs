use std::time::Instant;

use ccmm_rs::ring::{
    find_ntt_prime_below_bits, Limb128, Limb32, Limb64, LimbNttPlan, LimbPolynomial,
    PhysicalLimbArithmetic,
};

const DEGREE: usize = 4096;

fn sparse_coefficients_u128(modulus: u128) -> (Vec<u128>, Vec<u128>) {
    let mut lhs = vec![0_u128; DEGREE];
    let mut rhs = vec![0_u128; DEGREE];

    for (index, value) in [
        (0_usize, 3_u128),
        (1, 5),
        (17, 11),
        (255, 19),
        (1024, 23),
        (2047, 29),
        (4095, 31),
    ] {
        lhs[index] = value % modulus;
    }

    for (index, value) in [
        (0_usize, 7_u128),
        (2, 13),
        (31, 17),
        (511, 37),
        (1536, 41),
        (3001, 43),
        (4095, 47),
    ] {
        rhs[index] = value % modulus;
    }

    (lhs, rhs)
}

fn sparse_negacyclic_oracle(lhs: &[u128], rhs: &[u128], modulus: u128) -> Vec<u128> {
    use num_bigint::BigUint;
    use num_traits::ToPrimitive;

    let q = BigUint::from(modulus);
    let mut output = vec![BigUint::from(0_u8); lhs.len()];

    let lhs_nonzero: Vec<(usize, u128)> = lhs
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value != 0)
        .collect();

    let rhs_nonzero: Vec<(usize, u128)> = rhs
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value != 0)
        .collect();

    for &(lhs_index, lhs_value) in &lhs_nonzero {
        for &(rhs_index, rhs_value) in &rhs_nonzero {
            let product = (BigUint::from(lhs_value) * BigUint::from(rhs_value)) % &q;
            let raw_index = lhs_index + rhs_index;

            if raw_index < lhs.len() {
                output[raw_index] = (&output[raw_index] + &product) % &q;
            } else {
                let index = raw_index - lhs.len();

                output[index] = if output[index] >= product {
                    &output[index] - &product
                } else {
                    &q - (&product - &output[index])
                };
            }
        }
    }

    output
        .into_iter()
        .map(|value| {
            value
                .to_u128()
                .expect("coefficient reduced modulo u128 must fit")
        })
        .collect()
}

fn validate32(modulus: u32) {
    let arithmetic = Limb32::new(modulus);
    let start = Instant::now();
    let plan = LimbNttPlan::new(arithmetic, DEGREE);
    let plan_us = start.elapsed().as_secs_f64() * 1.0e6;

    let psi = plan.psi();

    assert_eq!(plan.arithmetic().pow_mod(psi, DEGREE as u128), modulus - 1);
    assert_eq!(plan.arithmetic().pow_mod(psi, (2 * DEGREE) as u128), 1);

    let (lhs128, rhs128) = sparse_coefficients_u128(u128::from(modulus));

    let lhs = LimbPolynomial::new(
        arithmetic,
        lhs128.iter().map(|&value| value as u32).collect(),
    );
    let rhs = LimbPolynomial::new(
        arithmetic,
        rhs128.iter().map(|&value| value as u32).collect(),
    );

    assert_eq!(plan.inverse(&plan.forward(&lhs)), lhs);

    let start = Instant::now();
    let product = plan.negacyclic_mul(&lhs, &rhs);
    let mul_us = start.elapsed().as_secs_f64() * 1.0e6;

    let actual: Vec<u128> = product
        .coefficients()
        .iter()
        .map(|&value| u128::from(value))
        .collect();

    assert_eq!(
        actual,
        sparse_negacyclic_oracle(&lhs128, &rhs128, u128::from(modulus))
    );

    println!("WIDTH=32");
    println!("MODULUS={modulus}");
    println!("MODULUS_BITS={}", 32 - modulus.leading_zeros());
    println!("DEGREE={DEGREE}");
    println!("PSI={psi}");
    println!("ROOT_ORDER_CHECK=PASS");
    println!("ROUNDTRIP=PASS");
    println!("SPARSE_NEGACYCLIC_PRODUCT=PASS");
    println!("PLAN_US={plan_us:.3}");
    println!("NTT_MULTIPLY_US={mul_us:.3}");
}

fn validate64(modulus: u64) {
    let arithmetic = Limb64::new(modulus);
    let start = Instant::now();
    let plan = LimbNttPlan::new(arithmetic, DEGREE);
    let plan_us = start.elapsed().as_secs_f64() * 1.0e6;

    let psi = plan.psi();

    assert_eq!(plan.arithmetic().pow_mod(psi, DEGREE as u128), modulus - 1);
    assert_eq!(plan.arithmetic().pow_mod(psi, (2 * DEGREE) as u128), 1);

    let (lhs128, rhs128) = sparse_coefficients_u128(u128::from(modulus));

    let lhs = LimbPolynomial::new(
        arithmetic,
        lhs128.iter().map(|&value| value as u64).collect(),
    );
    let rhs = LimbPolynomial::new(
        arithmetic,
        rhs128.iter().map(|&value| value as u64).collect(),
    );

    assert_eq!(plan.inverse(&plan.forward(&lhs)), lhs);

    let start = Instant::now();
    let product = plan.negacyclic_mul(&lhs, &rhs);
    let mul_us = start.elapsed().as_secs_f64() * 1.0e6;

    let actual: Vec<u128> = product
        .coefficients()
        .iter()
        .map(|&value| u128::from(value))
        .collect();

    assert_eq!(
        actual,
        sparse_negacyclic_oracle(&lhs128, &rhs128, u128::from(modulus))
    );

    println!();
    println!("WIDTH=64");
    println!("MODULUS={modulus}");
    println!("MODULUS_BITS={}", 64 - modulus.leading_zeros());
    println!("DEGREE={DEGREE}");
    println!("PSI={psi}");
    println!("ROOT_ORDER_CHECK=PASS");
    println!("ROUNDTRIP=PASS");
    println!("SPARSE_NEGACYCLIC_PRODUCT=PASS");
    println!("PLAN_US={plan_us:.3}");
    println!("NTT_MULTIPLY_US={mul_us:.3}");
}

fn validate128(modulus: u128) {
    let arithmetic = Limb128::new(modulus);
    let start = Instant::now();
    let plan = LimbNttPlan::new(arithmetic, DEGREE);
    let plan_us = start.elapsed().as_secs_f64() * 1.0e6;

    let psi = plan.psi();

    assert_eq!(plan.arithmetic().pow_mod(psi, DEGREE as u128), modulus - 1);
    assert_eq!(plan.arithmetic().pow_mod(psi, (2 * DEGREE) as u128), 1);

    let (lhs_values, rhs_values) = sparse_coefficients_u128(modulus);

    let lhs = LimbPolynomial::new(arithmetic, lhs_values.clone());
    let rhs = LimbPolynomial::new(arithmetic, rhs_values.clone());

    assert_eq!(plan.inverse(&plan.forward(&lhs)), lhs);

    let start = Instant::now();
    let product = plan.negacyclic_mul(&lhs, &rhs);
    let mul_us = start.elapsed().as_secs_f64() * 1.0e6;

    assert_eq!(
        product.coefficients(),
        sparse_negacyclic_oracle(&lhs_values, &rhs_values, modulus)
    );

    println!();
    println!("WIDTH=128");
    println!("MODULUS={modulus}");
    println!("MODULUS_BITS={}", 128 - modulus.leading_zeros());
    println!("DEGREE={DEGREE}");
    println!("PSI={psi}");
    println!("PRIMALITY_STATUS=reproducible-probable-prime");
    println!("ROOT_ORDER_CHECK=PASS");
    println!("ROUNDTRIP=PASS");
    println!("SPARSE_NEGACYCLIC_PRODUCT=PASS");
    println!("PLAN_US={plan_us:.3}");
    println!("NTT_MULTIPLY_US={mul_us:.3}");
}

fn main() {
    let q32 = find_ntt_prime_below_bits(32, DEGREE).expect("32-bit NTT prime discovery failed");
    let q64 = find_ntt_prime_below_bits(64, DEGREE).expect("64-bit NTT prime discovery failed");
    let q128 = find_ntt_prime_below_bits(120, DEGREE)
        .expect("wide u128 NTT probable-prime discovery failed");

    assert!(q32 <= u32::MAX as u128);
    assert!(q64 <= u64::MAX as u128);
    assert!(q128 > u64::MAX as u128);

    println!("R3_3D_0B_NTT_PARAMETER_VALIDATION_VERSION=1");
    validate32(q32 as u32);
    validate64(q64 as u64);
    validate128(q128);
    println!();
    println!("R3_3D_0B_STATUS=PASS");
}
