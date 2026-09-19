use std::hint::black_box;
use std::mem::size_of;
use std::time::Instant;

use ccmm_rs::ring::{
    Limb128, Limb32, Limb64, LimbNttPlan, LimbRnsNttPlan, LimbRnsPolynomial, NttLimbArithmetic,
    CANONICAL_NTT_DEGREE,
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;

const DEGREE: usize = CANONICAL_NTT_DEGREE;
const WARMUP: usize = 2;
const REPEATS: usize = 9;

const BASIS32: [u32; 4] = [1_073_692_673, 1_073_668_097, 1_073_651_713, 1_073_643_521];

const BASIS64: [u64; 2] = [1_152_921_504_606_830_593, 1_152_921_504_606_748_673];

const BASIS128: [u128; 1] = [1_329_227_995_784_915_872_903_807_060_279_713_793];

fn input_coefficients(offset: u64) -> Vec<BigUint> {
    let mut values = vec![BigUint::from(0_u8); DEGREE];

    // Keep support in the low 256 coefficients. The exact schoolbook product
    // then has degree < 512 and never wraps modulo x^N+1, while the NTT still
    // executes over the full N=4096 transform.
    for (index, slot) in values.iter_mut().take(256).enumerate() {
        let i = index as u64;
        let value = (offset + 17 * i + 5 * i * i + 3 * i * i * i) & ((1_u64 << 20) - 1);
        *slot = BigUint::from(value);
    }

    values
}

fn median_us<F>(mut operation: F) -> f64
where
    F: FnMut(),
{
    for _ in 0..WARMUP {
        operation();
    }

    let mut samples = Vec::with_capacity(REPEATS);

    for _ in 0..REPEATS {
        let start = Instant::now();
        operation();
        samples.push(start.elapsed().as_secs_f64() * 1.0e6);
    }

    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn product_bits<A>(arithmetic: &[A]) -> u64
where
    A: NttLimbArithmetic,
{
    arithmetic
        .iter()
        .fold(BigUint::from(1_u8), |product, limb| {
            product * BigUint::from(A::word_to_u128(limb.modulus()))
        })
        .bits()
}

fn benchmark_case<A>(
    name: &str,
    physical_word_bits: u32,
    arithmetic: Vec<A>,
    lhs_values: &[BigUint],
    rhs_values: &[BigUint],
) -> Vec<BigUint>
where
    A: NttLimbArithmetic + Clone + PartialEq + Eq,
{
    let limb_count = arithmetic.len();
    let modulus_bits = product_bits(&arithmetic);

    let lhs = LimbRnsPolynomial::from_big_coefficients(arithmetic.clone(), lhs_values);
    let rhs = LimbRnsPolynomial::from_big_coefficients(arithmetic.clone(), rhs_values);

    let plan_construction_us = median_us(|| {
        black_box(LimbRnsNttPlan::new(arithmetic.clone(), DEGREE));
    });

    let plans: Vec<LimbNttPlan<A>> = arithmetic
        .iter()
        .cloned()
        .map(|limb| LimbNttPlan::new(limb, DEGREE))
        .collect();

    let forward_us = median_us(|| {
        for (index, plan) in plans.iter().enumerate() {
            black_box(plan.forward(lhs.residue(index)));
        }
    });

    let lhs_ntt: Vec<Vec<A::Word>> = plans
        .iter()
        .enumerate()
        .map(|(index, plan)| plan.forward(lhs.residue(index)))
        .collect();

    let rhs_ntt: Vec<Vec<A::Word>> = plans
        .iter()
        .enumerate()
        .map(|(index, plan)| plan.forward(rhs.residue(index)))
        .collect();

    let inverse_us = median_us(|| {
        for (index, plan) in plans.iter().enumerate() {
            black_box(plan.inverse(&lhs_ntt[index]));
        }
    });

    let pointwise_us = median_us(|| {
        for (limb_index, plan) in plans.iter().enumerate() {
            let product: Vec<A::Word> = lhs_ntt[limb_index]
                .iter()
                .copied()
                .zip(rhs_ntt[limb_index].iter().copied())
                .map(|(lhs, rhs)| plan.arithmetic().mul_mod(lhs, rhs))
                .collect();
            black_box(product);
        }
    });

    let rns_plan = LimbRnsNttPlan::new(arithmetic.clone(), DEGREE);

    let full_rns_multiply_us = median_us(|| {
        black_box(rns_plan.negacyclic_mul(&lhs, &rhs));
    });

    let product = rns_plan.negacyclic_mul(&lhs, &rhs);

    let crt_reconstruct_us = median_us(|| {
        black_box(product.reconstruct_coefficients_big());
    });

    let end_to_end_us = median_us(|| {
        let product = rns_plan.negacyclic_mul(&lhs, &rhs);
        black_box(product.reconstruct_coefficients_big());
    });

    let reconstructed = product.reconstruct_coefficients_big();

    let bytes_per_word = size_of::<A::Word>();
    let coefficient_storage_bytes = DEGREE * limb_count * bytes_per_word;
    let ntt_domain_storage_bytes = coefficient_storage_bytes;
    let ntt_count_per_multiply = 3 * limb_count;

    println!("BASIS={name}");
    println!("PHYSICAL_WORD_BITS={physical_word_bits}");
    println!("PHYSICAL_LIMBS={limb_count}");
    println!("COMPOSITE_MODULUS_BITS={modulus_bits}");
    println!("NTT_COUNT_PER_MULTIPLY={ntt_count_per_multiply}");
    println!("WORD_BYTES={bytes_per_word}");
    println!("COEFFICIENT_STORAGE_BYTES_PER_POLYNOMIAL={coefficient_storage_bytes}");
    println!("NTT_DOMAIN_STORAGE_BYTES_PER_POLYNOMIAL={ntt_domain_storage_bytes}");
    println!("PLAN_CONSTRUCTION_MEDIAN_US={plan_construction_us:.3}");
    println!("FORWARD_ALL_LIMBS_MEDIAN_US={forward_us:.3}");
    println!("INVERSE_ALL_LIMBS_MEDIAN_US={inverse_us:.3}");
    println!("POINTWISE_ALL_LIMBS_MEDIAN_US={pointwise_us:.3}");
    println!("FULL_RNS_MULTIPLY_MEDIAN_US={full_rns_multiply_us:.3}");
    println!("CRT_RECONSTRUCT_MEDIAN_US={crt_reconstruct_us:.3}");
    println!("END_TO_END_MUL_CRT_MEDIAN_US={end_to_end_us:.3}");
    println!();

    reconstructed
}

fn exact_schoolbook_product(lhs: &[BigUint], rhs: &[BigUint]) -> Vec<BigUint> {
    let mut output = vec![BigUint::from(0_u8); DEGREE];

    for (lhs_index, lhs_value) in lhs.iter().take(256).enumerate() {
        for (rhs_index, rhs_value) in rhs.iter().take(256).enumerate() {
            output[lhs_index + rhs_index] += lhs_value * rhs_value;
        }
    }

    output
}

fn main() {
    let lhs_values = input_coefficients(7);
    let rhs_values = input_coefficients(29);

    println!("R3_3D_1C_WIDTH_CHARACTERIZATION_VERSION=1");
    println!("DEGREE={DEGREE}");
    println!("TARGET_LOGICAL_MODULUS_BITS=120");
    println!("WARMUP={WARMUP}");
    println!("REPEATS={REPEATS}");
    println!("TIMING_STATISTIC=median");
    println!();

    let result32 = benchmark_case(
        "W32_X4",
        32,
        BASIS32.into_iter().map(Limb32::new).collect(),
        &lhs_values,
        &rhs_values,
    );

    let result64 = benchmark_case(
        "W64_X2",
        64,
        BASIS64.into_iter().map(Limb64::new).collect(),
        &lhs_values,
        &rhs_values,
    );

    let result128 = benchmark_case(
        "W128_X1",
        128,
        BASIS128.into_iter().map(Limb128::new).collect(),
        &lhs_values,
        &rhs_values,
    );

    let oracle = exact_schoolbook_product(&lhs_values, &rhs_values);

    assert_eq!(result32, oracle);
    assert_eq!(result64, oracle);
    assert_eq!(result128, oracle);
    assert_eq!(result32, result64);
    assert_eq!(result64, result128);

    // Small checksum makes the artifact convenient to inspect without dumping
    // all 4096 reconstructed coefficients.
    let checksum: BigUint = oracle.iter().cloned().sum();
    println!("CROSS_WIDTH_EXACT_RESULT=PASS");
    println!("REFERENCE_SCHOOLBOOK_RESULT=PASS");
    println!(
        "RESULT_CHECKSUM={}",
        checksum
            .to_u128()
            .expect("benchmark checksum must fit u128")
    );
    println!("R3_3D_1C_STATUS=PASS");
}
