//! R16.5.18g.2: full-UHD auxiliary modulus-down benchmark.
//!
//! Uses the exact research Q profile from R16.5.18e.2.
//! Research parameters only; no security claim.

use std::time::Instant;

use ccmm_rs::ckks::research_profile_32768_for_levels;
use ccmm_rs::ring::{
    auxiliary_mod_down_reference, auxiliary_mod_down_rns_candidate, Modulus, ModulusBasis,
    Polynomial, RnsPolynomial,
};

const DEGREE: usize = 32768;
const REPETITIONS: usize = 3;

const P_PRIMES: [u64; 2] = [1125899908022273, 1125899908612097];

fn deterministic_polynomial(q_basis: &ModulusBasis, p_basis: &ModulusBasis) -> RnsPolynomial {
    let moduli: Vec<Modulus> = q_basis
        .moduli()
        .iter()
        .chain(p_basis.moduli().iter())
        .copied()
        .collect();

    let residues: Vec<Polynomial> = moduli
        .iter()
        .enumerate()
        .map(|(limb, &modulus)| {
            let coefficients = (0..DEGREE)
                .map(|index| {
                    let mut value = (index as u64)
                        .wrapping_mul(0x9e3779b97f4a7c15)
                        .wrapping_add((limb as u64).wrapping_mul(0xbf58476d1ce4e5b9));

                    value ^= value >> 30;
                    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
                    value ^= value >> 27;
                    value = value.wrapping_mul(0x94d049bb133111eb);
                    value ^= value >> 31;

                    value % modulus.value()
                })
                .collect();

            Polynomial::new(modulus, coefficients)
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

fn median(mut values: Vec<f64>) -> f64 {
    assert!(!values.is_empty());
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn assert_exact_equal(
    reference: &RnsPolynomial,
    candidate: &RnsPolynomial,
    q_basis: &ModulusBasis,
) -> usize {
    assert_eq!(reference.moduli(), candidate.moduli());
    assert_eq!(reference.moduli(), q_basis.moduli());

    let mut compared = 0usize;

    for limb in 0..q_basis.len() {
        for index in 0..DEGREE {
            assert_eq!(
                reference.residue(limb).coefficient(index),
                candidate.residue(limb).coefficient(index),
                "mismatch at Q limb {limb}, coefficient {index}"
            );
            compared += 1;
        }
    }

    compared
}

#[test]
#[ignore = "explicit full-UHD BigInt benchmark"]
fn r16_5_18g2_uhd_mod_down_benchmark() {
    assert_eq!(
        std::env::var("R16_18G2_AUTHORIZE").ok().as_deref(),
        Some("YES"),
        "set R16_18G2_AUTHORIZE=YES"
    );

    let profile = research_profile_32768_for_levels(13, 3);
    let q_basis = profile.modulus_chain().top().clone();

    assert_eq!(profile.degree(), DEGREE);
    assert_eq!(q_basis.len(), 16);

    let p_basis = ModulusBasis::new(P_PRIMES.into_iter().map(Modulus::new).collect());

    assert_eq!(p_basis.len(), 2);

    for &p in &P_PRIMES {
        assert_eq!(p % (2 * DEGREE as u64), 1);
        assert!(!q_basis.moduli().iter().any(|q| q.value() == p));
    }

    let polynomial = deterministic_polynomial(&q_basis, &p_basis);

    println!("R16_5_18G2_DEGREE={DEGREE}");
    println!("R16_5_18G2_Q_LIMBS={}", q_basis.len());
    println!("R16_5_18G2_P_LIMBS={}", p_basis.len());
    println!("R16_5_18G2_REPETITIONS={REPETITIONS}");

    let mut reference_times = Vec::with_capacity(REPETITIONS);
    let mut candidate_times = Vec::with_capacity(REPETITIONS);
    let mut compared = 0usize;

    for repetition in 0..REPETITIONS {
        let start = Instant::now();
        let reference = auxiliary_mod_down_reference(&polynomial, &q_basis, &p_basis);
        let reference_seconds = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let candidate = auxiliary_mod_down_rns_candidate(&polynomial, &q_basis, &p_basis);
        let candidate_seconds = start.elapsed().as_secs_f64();

        compared = assert_exact_equal(&reference, &candidate, &q_basis);

        reference_times.push(reference_seconds);
        candidate_times.push(candidate_seconds);

        println!(
            "R16_5_18G2_RUN_{}_REFERENCE_SECONDS={:.6}",
            repetition + 1,
            reference_seconds
        );
        println!(
            "R16_5_18G2_RUN_{}_CANDIDATE_SECONDS={:.6}",
            repetition + 1,
            candidate_seconds
        );
    }

    let reference_median = median(reference_times);
    let candidate_median = median(candidate_times);

    println!("R16_5_18G2_REFERENCE_MEDIAN_SECONDS={reference_median:.6}");
    println!("R16_5_18G2_CANDIDATE_MEDIAN_SECONDS={candidate_median:.6}");
    println!(
        "R16_5_18G2_SPEEDUP={:.4}",
        reference_median / candidate_median
    );
    println!("R16_5_18G2_COMPARED_RESIDUES={compared}");
    println!("R16_5_18G2_EXACT_EQUALITY=PASS");
    println!("R16_5_18G2_STATUS=PASS");
}
