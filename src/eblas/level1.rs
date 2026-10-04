use crate::ckks::{
    mod_switch_rns_ckks_to_next, multiply_plain_rns_ckks_with_ntt, rescale_rns_ckks_to_next,
    CkksCanonicalEmbedding, RnsCkksEvaluator,
};
use crate::matrix::RnsCkksCiphertextMatrix;
use crate::ring::{ModulusBasis, ModulusChain, Polynomial, RnsNttPlan, RnsPolynomial};
use num_complex::Complex64;

fn encode_complex_slots(
    slots: &[Complex64],
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    assert_eq!(
        slots.len(),
        embedding.slot_count(),
        "eBLAS packed plaintext slot count must match CKKS embedding slot count"
    );
    assert!(
        slots
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite()),
        "eBLAS packed plaintext slots must be finite"
    );
    assert!(
        scale.is_finite() && scale > 0.0,
        "eBLAS packed plaintext scale must be finite and positive"
    );

    let coefficients = embedding.slots_to_coefficients(slots);

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

    scale_complex_cp(input, Complex64::new(alpha, 0.0), embedding, chain, plan)
}

/// Scales an encrypted matrix by a public complex scalar.
///
/// The scalar is replicated across CKKS slots and encoded at the active
/// level using that level's dropped modulus as its plaintext scale.
/// Ciphertext-plaintext multiplication followed by one CKKS rescale consumes
/// exactly one chain level while returning approximately to the incoming
/// ciphertext scale.
///
/// This primitive is the public-twiddle multiplication required by structured
/// encrypted FFT execution.
pub fn scale_complex_cp(
    input: &RnsCkksCiphertextMatrix,
    alpha: Complex64,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertextMatrix {
    assert!(
        alpha.re.is_finite() && alpha.im.is_finite(),
        "eBLAS complex SCALE scalar must be finite"
    );

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
    let slots = vec![alpha; embedding.slot_count()];
    let scalar = encode_complex_slots(&slots, embedding, first.basis(), plaintext_scale);

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

/// Multiplies one encrypted CKKS value by an arbitrary public complex slot vector.
///
/// `public_slots[k]` multiplies logical CKKS slot `k`. The public vector must
/// contain exactly `embedding.slot_count()` entries.
///
/// The public vector is encoded at the active level using that level's dropped
/// modulus as its plaintext scale. Ciphertext-plaintext multiplication followed
/// by one CKKS rescale consumes exactly one chain level and returns
/// approximately to the incoming ciphertext scale.
///
/// This is the packed public-mask/twiddle primitive used by structured eBLAS
/// transforms. It performs no ciphertext-ciphertext multiplication and requires
/// no relinearization.
pub fn multiply_complex_slots_cp(
    input: &crate::ckks::RnsCkksCiphertext,
    public_slots: &[Complex64],
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> crate::ckks::RnsCkksCiphertext {
    input.assert_matches_chain(chain);

    assert_eq!(
        embedding.degree(),
        input.rlwe().degree(),
        "eBLAS packed CP embedding degree must match ciphertext ring degree"
    );
    assert_eq!(
        public_slots.len(),
        embedding.slot_count(),
        "eBLAS packed CP public slot count must match CKKS embedding slot count"
    );
    assert_eq!(
        plan.degree(),
        input.rlwe().degree(),
        "eBLAS packed CP NTT plan degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.moduli(),
        input.basis().moduli(),
        "eBLAS packed CP NTT plan basis must match ciphertext basis"
    );

    let dropped = chain
        .dropped_modulus(input.level())
        .expect("eBLAS packed CP multiplication cannot consume the final CKKS chain level");
    let plaintext_scale = dropped.value() as f64;

    let plaintext = encode_complex_slots(public_slots, embedding, input.basis(), plaintext_scale);

    let product = multiply_plain_rns_ckks_with_ntt(input, &plaintext, plaintext_scale, chain, plan);

    rescale_rns_ckks_to_next(&product, chain)
}

/// Computes `alpha * x + y` for encrypted matrices and a public real scalar.
///
/// Both encrypted operands must have identical shapes and CKKS states on input.
/// `SCALE(alpha, x)` consumes one CKKS level while restoring the incoming scale.
/// `y` is modulus-switched to the same next level without changing its scale,
/// after which the aligned ciphertexts are added.
///
/// This operation consumes exactly one CKKS level.
pub fn axpy_cp(
    evaluator: &RnsCkksEvaluator<'_>,
    alpha: f64,
    x: &RnsCkksCiphertextMatrix,
    y: &RnsCkksCiphertextMatrix,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertextMatrix {
    assert_eq!(
        x.rows(),
        y.rows(),
        "eBLAS AXPY operand row counts must match"
    );
    assert_eq!(
        x.cols(),
        y.cols(),
        "eBLAS AXPY operand column counts must match"
    );
    assert_eq!(
        x.level(),
        y.level(),
        "eBLAS AXPY operands must have matching levels"
    );
    assert_eq!(
        x.scale(),
        y.scale(),
        "eBLAS AXPY operands must have matching scales"
    );
    assert_eq!(
        x.get(0, 0).basis(),
        y.get(0, 0).basis(),
        "eBLAS AXPY operands must have matching bases"
    );

    let scaled_x = scale_cp(x, alpha, embedding, chain, plan);

    let mut y_data = Vec::with_capacity(y.rows() * y.cols());
    for col in 0..y.cols() {
        for row in 0..y.rows() {
            y_data.push(mod_switch_rns_ckks_to_next(y.get(row, col), chain));
        }
    }
    let aligned_y = RnsCkksCiphertextMatrix::from_vec_column_major(y.rows(), y.cols(), y_data);

    add_cc(evaluator, &scaled_x, &aligned_y)
}
