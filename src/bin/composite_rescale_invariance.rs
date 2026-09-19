use ccmm_rs::{
    ckks::CkksCanonicalEmbedding,
    ring::{
        Limb128, Limb32, Limb64, LimbRnsPolynomial, CANONICAL_NTT_DEGREE, NTT128_MODULUS,
        NTT32_MODULUS,
    },
};
use num_bigint::{BigInt, BigUint, Sign};
use num_complex::Complex64;
use num_traits::Zero;

const DEGREE: usize = CANONICAL_NTT_DEGREE;
const SCALE_BITS: u32 = 35;
const SCALE: f64 = (1_u64 << SCALE_BITS) as f64;
const TOLERANCE: f64 = 1.0e-6;

const DROP32: [u32; 4] = [1_073_692_673, 1_073_668_097, 1_073_651_713, 1_073_643_521];

const DROP64: [u64; 2] = [1_152_921_504_606_830_593, 1_152_921_504_606_748_673];

fn product_u32(values: &[u32]) -> BigUint {
    values
        .iter()
        .fold(BigUint::from(1_u8), |acc, &q| acc * BigUint::from(q))
}

fn product_u64(values: &[u64]) -> BigUint {
    values
        .iter()
        .fold(BigUint::from(1_u8), |acc, &q| acc * BigUint::from(q))
}

fn centered(value: &BigUint, modulus: &BigUint) -> BigInt {
    let half = modulus >> 1_usize;
    if value > &half {
        BigInt::from_biguint(Sign::Plus, value.clone())
            - BigInt::from_biguint(Sign::Plus, modulus.clone())
    } else {
        BigInt::from_biguint(Sign::Plus, value.clone())
    }
}

fn encode_post_rescale(coefficients: &[f64], modulus: &BigUint) -> Vec<BigUint> {
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

fn decode_post_rescale(coefficients: &[BigUint], modulus: &BigUint) -> Vec<f64> {
    coefficients
        .iter()
        .map(|value| {
            let integer = centered(value, modulus);
            let signed = integer
                .to_string()
                .parse::<f64>()
                .expect("centered CKKS coefficient must convert to f64");
            signed / SCALE
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
                let x = index as f64;
                Complex64::new(
                    0.25 * (0.17 * x).sin() + 0.125,
                    0.125 * (0.11 * x).cos() - 0.0625,
                )
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect()
}

fn main() {
    let embedding = CkksCanonicalEmbedding::new(DEGREE);
    let slots = test_slots(embedding.slot_count());
    let floating_coefficients = embedding.slots_to_coefficients(&slots);

    // The surviving logical level is exactly the same prime in all three
    // realizations; only its physical word backend changes.
    let persistent = BigUint::from(NTT32_MODULUS);
    let post = encode_post_rescale(&floating_coefficients, &persistent);

    let divisor32 = product_u32(&DROP32);
    let divisor64 = product_u64(&DROP64);
    let divisor128 = BigUint::from(NTT128_MODULUS);

    assert_eq!(divisor32.bits(), 120);
    assert_eq!(divisor64.bits(), 120);
    assert_eq!(divisor128.bits(), 120);

    // Exact-multiple inputs isolate logical-level semantics:
    // floor((post * D) / D) == post for each physical realization.
    let pre32 = post
        .iter()
        .map(|value| value * &divisor32)
        .collect::<Vec<_>>();
    let pre64 = post
        .iter()
        .map(|value| value * &divisor64)
        .collect::<Vec<_>>();
    let pre128 = post
        .iter()
        .map(|value| value * &divisor128)
        .collect::<Vec<_>>();

    let mut basis32 = vec![Limb32::new(NTT32_MODULUS)];
    basis32.extend(DROP32.into_iter().map(Limb32::new));

    let mut basis64 = vec![Limb64::new(u64::from(NTT32_MODULUS))];
    basis64.extend(DROP64.into_iter().map(Limb64::new));

    let basis128 = vec![
        Limb128::new(u128::from(NTT32_MODULUS)),
        Limb128::new(NTT128_MODULUS),
    ];

    let top32 = LimbRnsPolynomial::from_big_coefficients(basis32, &pre32);
    let top64 = LimbRnsPolynomial::from_big_coefficients(basis64, &pre64);
    let top128 = LimbRnsPolynomial::from_big_coefficients(basis128, &pre128);

    assert_eq!(top32.limb_count(), 5);
    assert_eq!(top64.limb_count(), 3);
    assert_eq!(top128.limb_count(), 2);

    let next32 = top32.rescale_drop_trailing_reference(4);
    let next64 = top64.rescale_drop_trailing_reference(2);
    let next128 = top128.rescale_drop_trailing_reference(1);

    assert_eq!(next32.limb_count(), 1);
    assert_eq!(next64.limb_count(), 1);
    assert_eq!(next128.limb_count(), 1);

    let reconstructed32 = next32.reconstruct_coefficients_big();
    let reconstructed64 = next64.reconstruct_coefficients_big();
    let reconstructed128 = next128.reconstruct_coefficients_big();

    assert_eq!(reconstructed32, post);
    assert_eq!(reconstructed64, post);
    assert_eq!(reconstructed128, post);
    assert_eq!(reconstructed32, reconstructed64);
    assert_eq!(reconstructed64, reconstructed128);

    let decoded_coefficients = decode_post_rescale(&reconstructed32, &persistent);
    let observed = embedding.coefficients_to_slots(&decoded_coefficients);
    let error = max_slot_error(&observed, &slots);

    assert!(error <= TOLERANCE);

    println!("R3_3D_3_COMPOSITE_RESCALE_INVARIANCE_VERSION=1");
    println!("DEGREE={DEGREE}");
    println!("SLOT_COUNT={}", embedding.slot_count());
    println!("ACTIVE_TEST_SLOTS=16");
    println!("POST_RESCALE_SCALE_BITS={SCALE_BITS}");
    println!("POST_RESCALE_SCALE={SCALE:.1}");
    println!("PERSISTENT_LOGICAL_MODULUS={NTT32_MODULUS}");
    println!("PERSISTENT_LOGICAL_MODULUS_BITS=32");
    println!("W32_TOP_PHYSICAL_LIMBS=5");
    println!("W64_TOP_PHYSICAL_LIMBS=3");
    println!("W128_TOP_PHYSICAL_LIMBS=2");
    println!("W32_DROPPED_PHYSICAL_LIMBS=4");
    println!("W64_DROPPED_PHYSICAL_LIMBS=2");
    println!("W128_DROPPED_PHYSICAL_LIMBS=1");
    println!("W32_LOGICAL_DIVISOR_BITS={}", divisor32.bits());
    println!("W64_LOGICAL_DIVISOR_BITS={}", divisor64.bits());
    println!("W128_LOGICAL_DIVISOR_BITS={}", divisor128.bits());
    println!("W32_POST_PHYSICAL_LIMBS={}", next32.limb_count());
    println!("W64_POST_PHYSICAL_LIMBS={}", next64.limb_count());
    println!("W128_POST_PHYSICAL_LIMBS={}", next128.limb_count());
    println!("POST_RESCALE_INTEGER_STATE_EXACT=PASS");
    println!("CROSS_WIDTH_POST_RESCALE_STATE_EXACT=PASS");
    println!("MAX_SLOT_ERROR={error:.12e}");
    println!("TOLERANCE={TOLERANCE:.12e}");
    println!("CKKS_POST_RESCALE_SEMANTICS=PASS");
    println!("COMPOSITE_LOGICAL_LEVEL_INVARIANCE=PASS");
    println!("R3_3D_3_STATUS=PASS");
}
