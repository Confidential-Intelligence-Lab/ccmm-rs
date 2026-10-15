//! R16.5.18d.1: actual Galois automorphism and hybrid key switch.

use ccmm_rs::ckks::{
    apply_hybrid_rns_galois_automorphism_with_prepared_ntt, apply_rns_automorphism,
    PreparedHybridRnsGaloisKey,
};
use ccmm_rs::grafting::{BoundedGadgetLayout, RnsRlweCiphertext};
use ccmm_rs::ring::{
    composite_modulus_big, reconstruct_coefficients_big, Modulus, ModulusBasis, RnsNttPlan,
    RnsPolynomial,
};
use ccmm_rs::rlwe::{ErrorDistribution, RlweCiphertext};
use num_bigint::BigInt;
use num_traits::Signed;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const N: usize = 16;

fn centered_error(a: &RnsPolynomial, b: &RnsPolynomial) -> BigInt {
    let modulus = BigInt::from(composite_modulus_big(a.basis()));
    let half = &modulus / 2;

    reconstruct_coefficients_big(a)
        .iter()
        .zip(reconstruct_coefficients_big(b).iter())
        .map(|(x, y)| {
            let mut difference = (BigInt::from(x.clone()) - BigInt::from(y.clone())) % &modulus;
            if difference > half {
                difference -= &modulus;
            }
            if difference < -&half {
                difference += &modulus;
            }
            difference.abs()
        })
        .max()
        .unwrap()
}

fn signed_poly(moduli: &[Modulus], values: &[i64]) -> RnsPolynomial {
    RnsPolynomial::from_residues(
        moduli
            .iter()
            .map(|&modulus| {
                ccmm_rs::ring::Polynomial::new(
                    modulus,
                    values
                        .iter()
                        .map(|&x| i128::from(x).rem_euclid(i128::from(modulus.value())) as u64)
                        .collect(),
                )
            })
            .collect(),
    )
}

fn parts(ct: &RnsRlweCiphertext) -> (RnsPolynomial, RnsPolynomial) {
    (
        RnsPolynomial::from_residues(ct.limbs().iter().map(|x| x.b().clone()).collect()),
        RnsPolynomial::from_residues(ct.limbs().iter().map(|x| x.a().clone()).collect()),
    )
}

#[test]
fn r16_5_18d1_hybrid_galois() {
    let q = ModulusBasis::new(vec![
        Modulus::new(12_289),
        Modulus::new(40_961),
        Modulus::new(65_537),
    ]);
    let p = ModulusBasis::new(vec![Modulus::new(114_689), Modulus::new(786_433)]);

    let mut qp_moduli = q.moduli().to_vec();
    qp_moduli.extend_from_slice(p.moduli());

    let q_plan = RnsNttPlan::new(q.moduli().to_vec(), N);
    let qp_plan = RnsNttPlan::new(qp_moduli, N);

    let secret: [i8; N] = [1, 0, -1, 1, 0, -1, 1, 0, -1, 0, 1, 0, 1, -1, 0, 1];
    let secret_values = secret.iter().map(|&x| i64::from(x)).collect::<Vec<_>>();
    let s = signed_poly(q.moduli(), &secret_values);

    let message_values = [
        17, -11, 29, 3, -19, 7, 13, -5, 23, -31, 37, -41, 43, -47, 53, -59,
    ];
    let a_values = [
        127, -88, 301, -194, 63, 91, -117, 205, 307, -401, 503, -607, 709, -811, 907, -1009,
    ];

    let message = signed_poly(q.moduli(), &message_values);
    let a = signed_poly(q.moduli(), &a_values);
    let b = message.sub(&q_plan.negacyclic_mul(&a, &s));

    let input = RnsRlweCiphertext::from_limbs(
        q.moduli()
            .iter()
            .enumerate()
            .map(|(index, _)| {
                RlweCiphertext::new(b.residue(index).clone(), a.residue(index).clone())
            })
            .collect(),
    );

    // Exercise two nonidentity automorphisms and both key distributions.
    for exponent in [3_usize, 5] {
        let expected_ct = apply_rns_automorphism(&input, exponent);
        let (expected_b, expected_a) = parts(&expected_ct);

        // Validate against the actual automorphism of the plaintext:
        // decrypting under the transformed secret is not needed,
        // because the original plaintext is available as an oracle.
        let expected_message_ct = RnsRlweCiphertext::from_limbs(
            q.moduli()
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    RlweCiphertext::new(
                        message.residue(index).clone(),
                        ccmm_rs::ring::Polynomial::zero(q.modulus(index), N),
                    )
                })
                .collect(),
        );
        let transformed_message = apply_rns_automorphism(&expected_message_ct, exponent);
        let (oracle, _) = parts(&transformed_message);

        assert_eq!(expected_b.degree(), N);
        assert_eq!(expected_a.degree(), N);

        for (mode, distribution) in [
            ("ZERO_NOISE", ErrorDistribution::BoundedUniform { bound: 0 }),
            (
                "GAUSSIAN",
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            ),
        ] {
            let mut rng = ChaCha20Rng::seed_from_u64(
                0x18d1_0000 + exponent as u64 + if mode == "GAUSSIAN" { 100 } else { 0 },
            );

            let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(q.clone(), 8),
                p.clone(),
                &secret,
                exponent,
                distribution,
                2,
                &qp_plan,
                &mut rng,
            );

            let output =
                apply_hybrid_rns_galois_automorphism_with_prepared_ntt(&input, &key, &qp_plan);

            let (output_b, output_a) = parts(&output);
            let decrypted = output_b.add(&q_plan.negacyclic_mul(&output_a, &s));

            let max_error = centered_error(&decrypted, &oracle);

            println!("R16_5_18D1_EXPONENT={exponent}");
            println!("R16_5_18D1_MODE={mode}");
            println!("R16_5_18D1_DIGITS={}", key.digit_count());
            println!("R16_5_18D1_MAX_ERROR={max_error}");

            assert!(
                max_error <= BigInt::from(16),
                "hybrid Galois switching exceeded toy precision bound"
            );
        }
    }

    println!("R16_5_18D1_STATUS=PASS");
    println!("R16_5_18D1_SECURITY_STATUS=TOY_PARAMETERS_ONLY");
}
