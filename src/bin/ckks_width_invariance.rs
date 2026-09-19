use ccmm_rs::{
    ckks::CkksCanonicalEmbedding,
    ring::{Limb128, Limb32, Limb64, LimbRnsPolynomial, CANONICAL_NTT_DEGREE, NTT128_MODULUS},
};
use num_bigint::{BigInt, BigUint, Sign};
use num_complex::Complex64;
use num_traits::{ToPrimitive, Zero};

const SCALE_BITS: u32 = 35;
const SCALE: f64 = (1_u64 << SCALE_BITS) as f64;
const BASIS32: [u32; 4] = [1_073_692_673, 1_073_668_097, 1_073_651_713, 1_073_643_521];
const BASIS64: [u64; 2] = [1_152_921_504_606_830_593, 1_152_921_504_606_748_673];

fn centered_big(value: &BigUint, modulus: &BigUint) -> BigInt {
    let half = modulus >> 1_usize;
    if value > &half {
        BigInt::from_biguint(Sign::Plus, value.clone())
            - BigInt::from_biguint(Sign::Plus, modulus.clone())
    } else {
        BigInt::from_biguint(Sign::Plus, value.clone())
    }
}

fn encode_scaled(coefficients: &[f64], modulus: &BigUint) -> Vec<BigUint> {
    coefficients
        .iter()
        .map(|&coefficient| {
            let scaled = (coefficient * SCALE).round();
            assert!(scaled.is_finite());
            assert!(scaled >= i64::MIN as f64 && scaled <= i64::MAX as f64);
            let signed = scaled as i64;
            if signed >= 0 {
                BigUint::from(signed as u64) % modulus
            } else {
                let magnitude = BigUint::from(signed.unsigned_abs()) % modulus;
                if magnitude.is_zero() {
                    BigUint::zero()
                } else {
                    modulus - magnitude
                }
            }
        })
        .collect()
}

fn decode_scaled(coefficients: &[BigUint], modulus: &BigUint) -> Vec<f64> {
    coefficients
        .iter()
        .map(|value| {
            centered_big(value, modulus)
                .to_f64()
                .expect("centered CKKS coefficient must fit f64")
                / SCALE
        })
        .collect()
}

fn max_slot_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(lhs, rhs)| (*lhs - *rhs).norm())
        .fold(0.0_f64, f64::max)
}

fn test_slots(slot_count: usize) -> Vec<Complex64> {
    (0..slot_count)
        .map(|index| {
            if index < 16 {
                let i = index as f64;
                Complex64::new(
                    ((i * 0.125).sin() + 0.25 * (i * 0.03125).cos()) * 0.5,
                    ((i * 0.09375).cos() - 0.125 * (i * 0.0625).sin()) * 0.25,
                )
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect()
}

fn main() {
    let degree = CANONICAL_NTT_DEGREE;
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = test_slots(embedding.slot_count());
    let coefficients = embedding.slots_to_coefficients(&slots);

    let arithmetic32 = BASIS32.into_iter().map(Limb32::new).collect::<Vec<_>>();
    let arithmetic64 = BASIS64.into_iter().map(Limb64::new).collect::<Vec<_>>();
    let arithmetic128 = vec![Limb128::new(NTT128_MODULUS)];

    let q32 = arithmetic32.iter().fold(BigUint::from(1_u8), |q, limb| {
        q * BigUint::from(ccmm_rs::ring::PhysicalLimbArithmetic::modulus(limb))
    });
    let q64 = arithmetic64.iter().fold(BigUint::from(1_u8), |q, limb| {
        q * BigUint::from(ccmm_rs::ring::PhysicalLimbArithmetic::modulus(limb))
    });
    let q128 = BigUint::from(NTT128_MODULUS);

    let encoded32 = encode_scaled(&coefficients, &q32);
    let encoded64 = encode_scaled(&coefficients, &q64);
    let encoded128 = encode_scaled(&coefficients, &q128);

    let rns32 = LimbRnsPolynomial::from_big_coefficients(arithmetic32, &encoded32);
    let rns64 = LimbRnsPolynomial::from_big_coefficients(arithmetic64, &encoded64);
    let rns128 = LimbRnsPolynomial::from_big_coefficients(arithmetic128, &encoded128);

    let reconstructed32 = rns32.reconstruct_coefficients_big();
    let reconstructed64 = rns64.reconstruct_coefficients_big();
    let reconstructed128 = rns128.reconstruct_coefficients_big();

    assert_eq!(reconstructed32, encoded32);
    assert_eq!(reconstructed64, encoded64);
    assert_eq!(reconstructed128, encoded128);

    let observed32 = embedding.coefficients_to_slots(&decode_scaled(&reconstructed32, &q32));
    let observed64 = embedding.coefficients_to_slots(&decode_scaled(&reconstructed64, &q64));
    let observed128 = embedding.coefficients_to_slots(&decode_scaled(&reconstructed128, &q128));

    let error32 = max_slot_error(&observed32, &slots);
    let error64 = max_slot_error(&observed64, &slots);
    let error128 = max_slot_error(&observed128, &slots);
    let cross_32_64 = max_slot_error(&observed32, &observed64);
    let cross_64_128 = max_slot_error(&observed64, &observed128);
    let cross_32_128 = max_slot_error(&observed32, &observed128);

    let tolerance = 1.0e-6;
    assert!(error32 <= tolerance);
    assert!(error64 <= tolerance);
    assert!(error128 <= tolerance);
    assert!(cross_32_64 <= tolerance);
    assert!(cross_64_128 <= tolerance);
    assert!(cross_32_128 <= tolerance);

    println!("R3_3D_2_CKKS_WIDTH_INVARIANCE_VERSION=1");
    println!("DEGREE={degree}");
    println!("SLOT_COUNT={}", embedding.slot_count());
    println!("ACTIVE_TEST_SLOTS=16");
    println!("SCALE_BITS={SCALE_BITS}");
    println!("SCALE={SCALE:.1}");
    println!("W32_PHYSICAL_LIMBS={}", rns32.limb_count());
    println!("W64_PHYSICAL_LIMBS={}", rns64.limb_count());
    println!("W128_PHYSICAL_LIMBS={}", rns128.limb_count());
    println!("W32_LOGICAL_MODULUS_BITS={}", q32.bits());
    println!("W64_LOGICAL_MODULUS_BITS={}", q64.bits());
    println!("W128_LOGICAL_MODULUS_BITS={}", q128.bits());
    println!("W32_MAX_SLOT_ERROR={error32:.12e}");
    println!("W64_MAX_SLOT_ERROR={error64:.12e}");
    println!("W128_MAX_SLOT_ERROR={error128:.12e}");
    println!("CROSS_WIDTH_32_64_MAX_SLOT_ERROR={cross_32_64:.12e}");
    println!("CROSS_WIDTH_64_128_MAX_SLOT_ERROR={cross_64_128:.12e}");
    println!("CROSS_WIDTH_32_128_MAX_SLOT_ERROR={cross_32_128:.12e}");
    println!("TOLERANCE={tolerance:.12e}");
    println!("RNS_ROUNDTRIP_32=PASS");
    println!("RNS_ROUNDTRIP_64=PASS");
    println!("RNS_ROUNDTRIP_128=PASS");
    println!("CKKS_WIDTH_INVARIANCE=PASS");
    println!("R3_3D_2_STATUS=PASS");
}
