//! R16.5.18g.1: exact differential validation of P-side CRT modulus-down.

use ccmm_rs::ring::{
    auxiliary_mod_down_reference, auxiliary_mod_down_rns_candidate, composite_modulus_big,
    rns_from_big_coefficients, Modulus, ModulusBasis,
};
use num_bigint::{BigInt, BigUint, Sign};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn compare(q: ModulusBasis, p: ModulusBasis, coefficients: &[BigUint]) {
    let mut moduli = q.moduli().to_vec();
    moduli.extend_from_slice(p.moduli());

    let polynomial = rns_from_big_coefficients(moduli, coefficients);

    let reference = auxiliary_mod_down_reference(&polynomial, &q, &p);
    let candidate = auxiliary_mod_down_rns_candidate(&polynomial, &q, &p);

    assert_eq!(candidate.moduli(), reference.moduli());
    assert_eq!(candidate.degree(), reference.degree());

    for index in 0..q.len() {
        assert_eq!(
            candidate.residue(index).coefficients(),
            reference.residue(index).coefficients(),
            "mismatch at Q limb {index}"
        );
    }
}

#[test]
fn r16_5_18g1_signed_boundaries() {
    let q = ModulusBasis::new(vec![Modulus::new(97), Modulus::new(193)]);
    let p = ModulusBasis::new(vec![Modulus::new(257)]);

    let qp = composite_modulus_big(&ModulusBasis::new(
        q.moduli().iter().chain(p.moduli()).copied().collect(),
    ));
    let qp_signed = BigInt::from(qp.clone());

    let signed = [
        -1000_i64, -386, -257, -129, -128, -1, 0, 1, 128, 129, 257, 386, 1000,
    ];

    let coefficients: Vec<BigUint> = signed
        .iter()
        .map(|&value| {
            let mut residue = BigInt::from(value) % &qp_signed;
            if residue.sign() == Sign::Minus {
                residue += &qp_signed;
            }
            residue.to_biguint().unwrap()
        })
        .collect();

    compare(q, p, &coefficients);
    println!("R16_5_18G1_SIGNED_BOUNDARIES=PASS");
}

#[test]
fn r16_5_18g1_randomized_toy() {
    let q = ModulusBasis::new(vec![
        Modulus::new(12_289),
        Modulus::new(40_961),
        Modulus::new(65_537),
    ]);
    let p = ModulusBasis::new(vec![Modulus::new(114_689), Modulus::new(786_433)]);

    let qp = composite_modulus_big(&ModulusBasis::new(
        q.moduli().iter().chain(p.moduli()).copied().collect(),
    ));

    let mut rng = ChaCha20Rng::seed_from_u64(0x18_0000_0001);

    for _ in 0..16 {
        let coefficients: Vec<BigUint> = (0..64)
            .map(|_| {
                let random = BigUint::from(rng.next_u64()) | (BigUint::from(rng.next_u64()) << 64);
                random % &qp
            })
            .collect();

        compare(q.clone(), p.clone(), &coefficients);
    }

    println!("R16_5_18G1_RANDOMIZED_TOY=PASS");
}

#[test]
fn r16_5_18g1_uhd_auxiliary_basis() {
    let q = ModulusBasis::new(vec![
        Modulus::new(12_289),
        Modulus::new(40_961),
        Modulus::new(65_537),
    ]);

    let p = ModulusBasis::new(vec![
        Modulus::new(1_125_899_908_022_273),
        Modulus::new(1_125_899_908_612_097),
    ]);

    let qp = composite_modulus_big(&ModulusBasis::new(
        q.moduli().iter().chain(p.moduli()).copied().collect(),
    ));

    let mut rng = ChaCha20Rng::seed_from_u64(0x18_0000_0002);

    let coefficients: Vec<BigUint> = (0..64)
        .map(|_| {
            let random = (BigUint::from(rng.next_u64()) << 128)
                | (BigUint::from(rng.next_u64()) << 64)
                | BigUint::from(rng.next_u64());
            random % &qp
        })
        .collect();

    compare(q, p, &coefficients);
    println!("R16_5_18G1_UHD_AUXILIARY_BASIS=PASS");
}
