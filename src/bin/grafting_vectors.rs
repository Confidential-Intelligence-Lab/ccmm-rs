use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use ccmm_rs::ckks::{multiply_relinearize_rescale_rns_ckks, CkksChainState, RnsCkksCiphertext};
use ccmm_rs::grafting::{
    decrypt_rns_raw, RnsGadgetLayout, RnsMultiplicationKey, RnsRlweCiphertext,
};
use ccmm_rs::ring::{Modulus, ModulusBasis, ModulusChain, Polynomial};
use ccmm_rs::rlwe::RlweCiphertext;

fn chain() -> ModulusChain {
    ModulusChain::from_top_basis(ModulusBasis::new(vec![
        Modulus::new(12_289),
        Modulus::new(40_961),
        Modulus::new(65_537),
    ]))
}

fn ternary_secret(degree: usize) -> Vec<i8> {
    (0..degree)
        .map(|index| match index % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 1,
        })
        .collect()
}

fn encode_coefficients(values: &[f64], scale: f64, modulus: u128) -> Vec<u128> {
    let modulus_i = i128::try_from(modulus).expect("modulus exceeds i128");

    values
        .iter()
        .map(|&value| {
            let scaled = (value * scale).round() as i128;

            let canonical = (scaled % modulus_i + modulus_i) % modulus_i;

            canonical as u128
        })
        .collect()
}

fn encrypt_values(
    chain: &ModulusChain,
    level: usize,
    values: &[f64],
    scale: f64,
    secret: &[i8],
    seed: u64,
) -> RnsCkksCiphertext {
    let basis = chain.level(level);

    let degree = values.len();

    let encoded = encode_coefficients(values, scale, basis.composite_modulus());

    let limbs = basis
        .moduli()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, modulus)| {
            let message = Polynomial::new(
                modulus,
                encoded
                    .iter()
                    .map(|&value| (value % u128::from(modulus.value())) as u64)
                    .collect(),
            );

            let secret_poly = Polynomial::new(
                modulus,
                secret
                    .iter()
                    .map(|&value| match value {
                        -1 => modulus.value() - 1,
                        0 => 0,
                        1 => 1,
                        _ => panic!("secret must be ternary"),
                    })
                    .collect(),
            );

            /*
             * Deterministic zero-noise RLWE ciphertext:
             *
             *     b + a*s = m
             *
             * so choose deterministic a and set
             *
             *     b = m - a*s.
             *
             * This is for differential validation only, not
             * production encryption.
             */
            let a_coefficients = (0..degree)
                .map(|coefficient_index| {
                    let x = seed
                        .wrapping_add((index as u64) << 32)
                        .wrapping_add(coefficient_index as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15);

                    x % modulus.value()
                })
                .collect();

            let a = Polynomial::new(modulus, a_coefficients);

            let a_times_s = a.negacyclic_mul(&secret_poly);

            let b = message.sub(&a_times_s);

            RlweCiphertext::new(b, a)
        })
        .collect();

    RnsCkksCiphertext::new(
        RnsRlweCiphertext::from_limbs(limbs),
        CkksChainState::new(chain, level, scale),
        chain,
    )
}

fn centered(value: u128, modulus: u128) -> i128 {
    let value = i128::try_from(value).expect("coefficient exceeds i128");

    let modulus = i128::try_from(modulus).expect("modulus exceeds i128");

    if value > modulus / 2 {
        value - modulus
    } else {
        value
    }
}

fn decrypt_decode(ciphertext: &RnsCkksCiphertext, secret: &[i8]) -> Vec<f64> {
    let polynomial = decrypt_rns_raw(ciphertext.rlwe(), secret);

    let modulus = polynomial.composite_modulus();

    polynomial
        .reconstruct_coefficients()
        .into_iter()
        .map(|value| centered(value, modulus) as f64 / ciphertext.scale())
        .collect()
}

fn print_vector(name: &str, values: &[f64]) {
    let values = values
        .iter()
        .map(|value| format!("{value:.17}"))
        .collect::<Vec<_>>()
        .join(",");

    println!("{name}={values}");
}

fn main() {
    let chain = chain();

    let degree = 8;

    let secret = ternary_secret(degree);

    let lhs = [0.125, -0.0625, 0.03125, 0.0, 0.0, 0.0, 0.0, 0.0];

    let rhs = [0.0625, 0.03125, -0.0625, 0.0, 0.0, 0.0, 0.0, 0.0];

    let third = [0.125, -0.0625, 0.03125, 0.0, 0.0, 0.0, 0.0, 0.0];

    let level0_scale = 65_537.0;

    let lhs_ct = encrypt_values(&chain, 0, &lhs, level0_scale, &secret, 0xD700);

    let rhs_ct = encrypt_values(&chain, 0, &rhs, level0_scale, &secret, 0xD701);

    let level0_layout = RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]);

    let mut level0_rng = ChaCha20Rng::seed_from_u64(0xD702);

    let level0_key = RnsMultiplicationKey::generate_with_rng(
        degree,
        2,
        0,
        &secret,
        level0_layout,
        &mut level0_rng,
    );

    let level1 = multiply_relinearize_rescale_rns_ckks(&lhs_ct, &rhs_ct, &level0_key, &chain);

    let level1_decoded = decrypt_decode(&level1, &secret);

    let level1_operand_scale = 40_961.0;

    let third_ct = encrypt_values(&chain, 1, &third, level1_operand_scale, &secret, 0xD703);

    let level1_layout = RnsGadgetLayout::new(chain.level(1).clone(), vec![1, 1]);

    let mut level1_rng = ChaCha20Rng::seed_from_u64(0xD704);

    let level1_key = RnsMultiplicationKey::generate_with_rng(
        degree,
        2,
        0,
        &secret,
        level1_layout,
        &mut level1_rng,
    );

    let level2 = multiply_relinearize_rescale_rns_ckks(&level1, &third_ct, &level1_key, &chain);

    let level2_decoded = decrypt_decode(&level2, &secret);

    println!("GRAFTING_VECTOR_VERSION=1");
    println!("DEGREE={degree}");

    println!("LEVEL0_MODULUS={}", chain.composite_modulus(0));

    println!("LEVEL1_MODULUS={}", chain.composite_modulus(1));

    println!("LEVEL2_MODULUS={}", chain.composite_modulus(2));

    println!("LEVEL0_SCALE={level0_scale:.17}");

    println!("LEVEL1_SCALE={:.17}", level1.scale());

    println!("LEVEL1_OPERAND_SCALE={level1_operand_scale:.17}");

    println!("LEVEL2_SCALE={:.17}", level2.scale());

    print_vector("INPUT_LHS", &lhs);

    print_vector("INPUT_RHS", &rhs);

    print_vector("INPUT_THIRD", &third);

    print_vector("LEVEL1_DECODED", &level1_decoded);

    print_vector("LEVEL2_DECODED", &level2_decoded);

    println!("GRAFTING_VECTOR_STATUS=PASS");
}
