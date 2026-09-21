//! CKKS application-support helpers.
//!
//! These helpers centralize mechanical encoding, seeded encryption, and
//! decryption/decoding used by application examples. Application policy,
//! parameter-profile selection, key generation, workload semantics, and eBLAS
//! dispatch remain explicit in the applications.

use crate::ckks::{CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext};
use crate::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng};
use crate::matrix::RnsCkksCiphertextMatrix;
use crate::ring::{ModulusBasis, ModulusChain, Polynomial, RnsNttPlan, RnsPolynomial};
use crate::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// References required for reproducible scalar CKKS encryption in applications.
pub struct EncryptionContext<'a> {
    pub embedding: &'a CkksCanonicalEmbedding,
    pub basis: &'a ModulusBasis,
    pub scale: f64,
    pub secret: &'a [i8],
    pub plan: &'a RnsNttPlan,
    pub chain: &'a ModulusChain,
    pub distribution: ErrorDistribution,
}

/// Encodes CKKS slots into an RNS polynomial at the supplied scale.
pub fn encode_rns(
    slots: &[Complex64],
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
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
                    .map(|&value| {
                        let signed = (value * scale).round() as i128;
                        signed.rem_euclid(q) as u64
                    })
                    .collect(),
            )
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

fn centered(value: u128, modulus: u128) -> i128 {
    if value > modulus / 2 {
        value as i128 - modulus as i128
    } else {
        value as i128
    }
}

/// Encrypts one real scalar replicated across all CKKS slots.
///
/// `seed` makes application experiments reproducible; it is not intended as a
/// production randomness interface.
pub fn encrypt_scalar(value: f64, seed: u64, context: &EncryptionContext<'_>) -> RnsCkksCiphertext {
    let slots = vec![Complex64::new(value, 0.0); context.secret.len() / 2];
    let plaintext = encode_rns(&slots, context.embedding, context.basis, context.scale);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        context.distribution,
        context.secret,
        context.plan,
        &mut rng,
    );

    RnsCkksCiphertext::new(
        rlwe,
        CkksChainState::top(context.chain, context.scale),
        context.chain,
    )
}

/// Encodes one public real scalar replicated across all CKKS slots.
pub fn encode_plain_scalar(
    value: f64,
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    let slots = vec![Complex64::new(value, 0.0); embedding.slot_count()];
    encode_rns(&slots, embedding, basis, scale)
}

/// Decrypts a replicated-scalar CKKS ciphertext.
///
/// Returns `(mean_real_slot_value, maximum_imaginary_residual)`.
pub fn decode_scalar(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> (f64, f64) {
    let plan = RnsNttPlan::new(
        ciphertext.basis().moduli().to_vec(),
        ciphertext.rlwe().degree(),
    );
    let plaintext = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);
    let modulus = plaintext.composite_modulus();
    let coefficients: Vec<f64> = plaintext
        .reconstruct_coefficients()
        .into_iter()
        .map(|value| centered(value, modulus) as f64 / ciphertext.scale())
        .collect();
    let slots = embedding.coefficients_to_slots(&coefficients);
    let mean = slots.iter().map(|slot| slot.re).sum::<f64>() / slots.len() as f64;
    let max_imag = slots
        .iter()
        .map(|slot| slot.im.abs())
        .fold(0.0_f64, f64::max);

    (mean, max_imag)
}

/// Decodes a column-major ciphertext matrix of replicated CKKS scalars.
///
/// The returned vector preserves the matrix's column-major storage order.
/// The second return value is the maximum imaginary residual over all entries.
pub fn decode_matrix(
    matrix: &RnsCkksCiphertextMatrix,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> (Vec<f64>, f64) {
    let mut values = Vec::with_capacity(matrix.rows() * matrix.cols());
    let mut max_imag = 0.0_f64;

    for col in 0..matrix.cols() {
        for row in 0..matrix.rows() {
            let (value, imag) = decode_scalar(matrix.get(row, col), secret, embedding);
            values.push(value);
            max_imag = max_imag.max(imag);
        }
    }

    (values, max_imag)
}
