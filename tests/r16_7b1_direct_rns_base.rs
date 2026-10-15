//! R16.7b.1: direct-RNS hybrid-QP base-term equivalence.
//!
//! Compare P*b modulo QP against the canonical wide-CRT reference.
//! This regression does not modify production rotation arithmetic.

use ccmm_rs::ring::{
    composite_modulus_big, reconstruct_coefficients_big, rns_from_big_coefficients, Modulus,
    ModulusBasis, Polynomial, RnsPolynomial,
};

fn direct_base(input: &RnsPolynomial, auxiliary: &ModulusBasis) -> RnsPolynomial {
    let degree = input.degree();
    let mut residues = Vec::new();

    for (index, &modulus) in input.moduli().iter().enumerate() {
        let p_factor = auxiliary.moduli().iter().fold(1_u64, |product, p| {
            modulus.mul(product, p.value() % modulus.value())
        });

        let coefficients = input
            .residue(index)
            .coefficients()
            .iter()
            .map(|&coefficient| modulus.mul(coefficient, p_factor))
            .collect();

        residues.push(Polynomial::new(modulus, coefficients));
    }

    for &modulus in auxiliary.moduli() {
        residues.push(Polynomial::zero(modulus, degree));
    }

    RnsPolynomial::from_residues(residues)
}

fn check_case(q: ModulusBasis, p: ModulusBasis, coefficients: &[u128]) {
    let input = RnsPolynomial::from_coefficients(q.moduli().to_vec(), coefficients);

    let canonical = reconstruct_coefficients_big(&input);
    let product_p = composite_modulus_big(&p);

    let mut qp_moduli = q.moduli().to_vec();
    qp_moduli.extend_from_slice(p.moduli());

    let reference = rns_from_big_coefficients(
        qp_moduli,
        &canonical
            .iter()
            .map(|coefficient| coefficient * &product_p)
            .collect::<Vec<_>>(),
    );

    let direct = direct_base(&input, &p);

    assert_eq!(reference, direct, "R16.7b.1 direct-RNS base-term mismatch");
}

#[test]
fn r16_7b1_direct_rns_base() {
    let q = ModulusBasis::new(vec![
        Modulus::new(12_289),
        Modulus::new(40_961),
        Modulus::new(65_537),
    ]);

    let p = ModulusBasis::new(vec![Modulus::new(114_689), Modulus::new(147_457)]);

    let cases: Vec<Vec<u128>> = vec![
        vec![0; 16],
        vec![1; 16],
        (0..16).map(|i| i as u128).collect(),
        (0..16).map(|i| (i * i * 37) as u128).collect(),
        (0..16)
            .map(|i| if i % 2 == 0 { u128::MAX } else { 0 })
            .collect(),
    ];

    for coefficients in &cases {
        check_case(q.clone(), p.clone(), coefficients);
    }

    println!("R16_7B1_CASES={}", cases.len());
    println!("R16_7B1_Q_LIMBS={}", q.len());
    println!("R16_7B1_P_LIMBS={}", p.len());
    println!("R16_7B1_EXACT_EQUALITY=PASS");
}
