//! R16.5.18f.2a: CKKS hybrid rotation wrapper regression.
//! Toy parameters only; validates API semantics, not security.

use ccmm_rs::ckks::{
    apply_hybrid_rns_galois_automorphism_with_prepared_ntt,
    rotate_hybrid_rns_ckks_with_prepared_ntt, rotate_left_hybrid_rns_ckks_with_prepared_ntt,
    rotate_right_hybrid_rns_ckks_with_prepared_ntt, CkksChainState, PreparedHybridRnsGaloisKey,
    RnsCkksCiphertext,
};
use ccmm_rs::grafting::{BoundedGadgetLayout, RnsRlweCiphertext};
use ccmm_rs::ring::{Modulus, ModulusBasis, ModulusChain, Polynomial, RnsNttPlan};
use ccmm_rs::rlwe::{ErrorDistribution, RlweCiphertext};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const N: usize = 16;

fn fixture() -> (
    RnsCkksCiphertext,
    ModulusChain,
    RnsNttPlan,
    PreparedHybridRnsGaloisKey,
) {
    let q = ModulusBasis::new(vec![
        Modulus::new(12_289),
        Modulus::new(40_961),
        Modulus::new(65_537),
    ]);
    let p = ModulusBasis::new(vec![Modulus::new(114_689), Modulus::new(786_433)]);

    let mut qp_moduli = q.moduli().to_vec();
    qp_moduli.extend_from_slice(p.moduli());
    let qp_plan = RnsNttPlan::new(qp_moduli, N);

    let chain = ModulusChain::from_top_basis(q.clone());
    let state = CkksChainState::top(&chain, 1024.0);

    let limbs = q
        .moduli()
        .iter()
        .copied()
        .map(|modulus| {
            let b = Polynomial::new(
                modulus,
                (0..N)
                    .map(|i| (17 * i as u64 + 3) % modulus.value())
                    .collect(),
            );
            let a = Polynomial::new(
                modulus,
                (0..N)
                    .map(|i| (11 * i as u64 + 5) % modulus.value())
                    .collect(),
            );
            RlweCiphertext::new(b, a)
        })
        .collect();

    let ciphertext = RnsCkksCiphertext::new(RnsRlweCiphertext::from_limbs(limbs), state, &chain);

    let secret: [i8; N] = [1, 0, -1, 1, 0, -1, 1, 0, -1, 0, 1, 0, 1, -1, 0, 1];

    let mut rng = ChaCha20Rng::seed_from_u64(0x18f2_0001);

    // For N=16, a one-slot left rotation uses exponent 5.
    let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
        BoundedGadgetLayout::new(q, 8),
        p,
        &secret,
        5,
        ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        2,
        &qp_plan,
        &mut rng,
    );

    (ciphertext, chain, qp_plan, key)
}

fn assert_same_rlwe(a: &RnsRlweCiphertext, b: &RnsRlweCiphertext) {
    assert_eq!(a.basis(), b.basis());
    assert_eq!(a.degree(), b.degree());

    for (left, right) in a.limbs().iter().zip(b.limbs()) {
        assert_eq!(left.a(), right.a());
        assert_eq!(left.b(), right.b());
    }
}

#[test]
fn r16_5_18f2a_wrapper_preserves_ciphertext_and_state() {
    let (input, chain, qp_plan, key) = fixture();

    let expected =
        apply_hybrid_rns_galois_automorphism_with_prepared_ntt(input.rlwe(), &key, &qp_plan);

    let output = rotate_left_hybrid_rns_ckks_with_prepared_ntt(&input, 1, &key, &chain, &qp_plan);

    assert_same_rlwe(output.rlwe(), &expected);
    assert_eq!(output.level(), input.level());
    assert_eq!(output.scale(), input.scale());
    assert_eq!(output.basis(), input.basis());

    let general = rotate_hybrid_rns_ckks_with_prepared_ntt(&input, 5, &key, &chain, &qp_plan);
    assert_same_rlwe(general.rlwe(), &expected);

    // For 8 slots, right rotation by 7 equals left rotation by 1.
    let right = rotate_right_hybrid_rns_ckks_with_prepared_ntt(&input, 7, &key, &chain, &qp_plan);
    assert_same_rlwe(right.rlwe(), &expected);

    println!("R16_5_18F2A_EQUIVALENCE=PASS");
    println!("R16_5_18F2A_STATE=PASS");
}

#[test]
#[should_panic(expected = "exponent does not match evaluation key")]
fn r16_5_18f2a_rejects_wrong_exponent() {
    let (input, chain, qp_plan, key) = fixture();

    let _ = rotate_left_hybrid_rns_ckks_with_prepared_ntt(&input, 2, &key, &chain, &qp_plan);
}

#[test]
#[should_panic(expected = "QP plan basis mismatch")]
fn r16_5_18f2a_rejects_wrong_qp_plan() {
    let (input, chain, _qp_plan, key) = fixture();

    let wrong_plan = RnsNttPlan::new(
        vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
            Modulus::new(786_433),
            Modulus::new(114_689),
        ],
        N,
    );

    let _ = rotate_left_hybrid_rns_ckks_with_prepared_ntt(&input, 1, &key, &chain, &wrong_plan);
}
