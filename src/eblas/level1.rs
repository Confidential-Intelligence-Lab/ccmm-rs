use crate::ckks::{
    multiply_plain_rns_ckks_with_ntt, rescale_rns_ckks_to_next, CkksCanonicalEmbedding,
    RnsCkksEvaluator,
};
use crate::matrix::RnsCkksCiphertextMatrix;
use crate::ring::{ModulusBasis, ModulusChain, Polynomial, RnsNttPlan, RnsPolynomial};
use num_complex::Complex64;

fn encode_replicated_scalar(
    value: f64,
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    assert!(value.is_finite(), "eBLAS SCALE scalar must be finite");
    assert!(
        scale.is_finite() && scale > 0.0,
        "eBLAS SCALE plaintext scale must be finite and positive"
    );

    let slots = vec![Complex64::new(value, 0.0); embedding.slot_count()];
    let coefficients = embedding.slots_to_coefficients(&slots);

    let residues = basis
        .moduli()
        .iter()
        .copied()
        .map(|modulus| {
            let q = i128::from(modulus.value());
            Polynomial::new(
                modulus,
                coefficients
                    .iter()
                    .map(|&coefficient| {
                        let scaled = coefficient * scale;
                        assert!(
                            scaled.is_finite(),
                            "eBLAS SCALE encoded coefficient must be finite"
                        );
                        (scaled.round() as i128).rem_euclid(q) as u64
                    })
                    .collect(),
            )
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

/// Element-wise encrypted addition.
///
/// Both operands must have identical shapes and CKKS states. Addition consumes
/// no CKKS level and delegates scalar ciphertext addition to the existing
/// [`RnsCkksEvaluator`].
pub fn add_cc(
    evaluator: &RnsCkksEvaluator<'_>,
    lhs: &RnsCkksCiphertextMatrix,
    rhs: &RnsCkksCiphertextMatrix,
) -> RnsCkksCiphertextMatrix {
    assert_eq!(
        lhs.rows(),
        rhs.rows(),
        "eBLAS ADD operand row counts must match"
    );
    assert_eq!(
        lhs.cols(),
        rhs.cols(),
        "eBLAS ADD operand column counts must match"
    );

    let mut data = Vec::with_capacity(lhs.rows() * lhs.cols());
    for col in 0..lhs.cols() {
        for row in 0..lhs.rows() {
            data.push(evaluator.add(lhs.get(row, col), rhs.get(row, col)));
        }
    }

    RnsCkksCiphertextMatrix::from_vec_column_major(lhs.rows(), lhs.cols(), data)
}

/// Scales an encrypted matrix by a public real scalar.
///
/// The scalar is encoded at the active level using that level's dropped
/// modulus as its plaintext scale. Ciphertext-plaintext multiplication raises
/// the scale by that factor and one CKKS rescale returns the result to the next
/// chain level with approximately the incoming ciphertext scale.
///
/// This operation therefore consumes exactly one CKKS level.
pub fn scale_cp(
    input: &RnsCkksCiphertextMatrix,
    alpha: f64,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertextMatrix {
    assert!(alpha.is_finite(), "eBLAS SCALE scalar must be finite");

    let first = input.get(0, 0);
    first.assert_matches_chain(chain);

    assert_eq!(
        embedding.degree(),
        first.rlwe().degree(),
        "eBLAS SCALE embedding degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.degree(),
        first.rlwe().degree(),
        "eBLAS SCALE NTT plan degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.moduli(),
        first.basis().moduli(),
        "eBLAS SCALE NTT plan basis must match ciphertext basis"
    );

    let dropped = chain
        .dropped_modulus(first.level())
        .expect("eBLAS SCALE cannot consume the final CKKS chain level");
    let plaintext_scale = dropped.value() as f64;
    let scalar = encode_replicated_scalar(alpha, embedding, first.basis(), plaintext_scale);

    let mut data = Vec::with_capacity(input.rows() * input.cols());
    for col in 0..input.cols() {
        for row in 0..input.rows() {
            let ciphertext = input.get(row, col);

            assert_eq!(
                ciphertext.level(),
                first.level(),
                "eBLAS SCALE matrix entries must have matching levels"
            );
            assert_eq!(
                ciphertext.basis(),
                first.basis(),
                "eBLAS SCALE matrix entries must have matching bases"
            );
            assert_eq!(
                ciphertext.scale(),
                first.scale(),
                "eBLAS SCALE matrix entries must have matching scales"
            );

            let product =
                multiply_plain_rns_ckks_with_ntt(ciphertext, &scalar, plaintext_scale, chain, plan);
            data.push(rescale_rns_ckks_to_next(&product, chain));
        }
    }

    RnsCkksCiphertextMatrix::from_vec_column_major(input.rows(), input.cols(), data)
}
