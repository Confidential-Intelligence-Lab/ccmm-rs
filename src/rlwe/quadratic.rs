use crate::ring::{NttPlan, Polynomial};

use super::{RlweCiphertext, RlweParameters, SecretKey};

/// Degree-2 RLWE ciphertext produced by multiplying two rank-1 ciphertexts.
///
/// It represents:
///
/// `c0 + c1*s + c2*s^2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlweQuadraticCiphertext {
    c0: Polynomial,
    c1: Polynomial,
    c2: Polynomial,
}

impl RlweQuadraticCiphertext {
    pub fn new(c0: Polynomial, c1: Polynomial, c2: Polynomial) -> Self {
        for other in [&c1, &c2] {
            assert_eq!(
                c0.modulus(),
                other.modulus(),
                "quadratic ciphertext polynomial moduli must match"
            );

            assert_eq!(
                c0.degree(),
                other.degree(),
                "quadratic ciphertext polynomial degrees must match"
            );
        }

        Self { c0, c1, c2 }
    }

    pub fn c0(&self) -> &Polynomial {
        &self.c0
    }

    pub fn c1(&self) -> &Polynomial {
        &self.c1
    }

    pub fn c2(&self) -> &Polynomial {
        &self.c2
    }

    pub fn into_components(self) -> (Polynomial, Polynomial, Polynomial) {
        (self.c0, self.c1, self.c2)
    }
}

/// Multiplies two rank-1 RLWE ciphertexts without relinearization.
///
/// For ciphertexts:
///
/// `lhs = b0 + a0*s`
///
/// and
///
/// `rhs = b1 + a1*s`,
///
/// this returns:
///
/// `c0 = b0*b1`
///
/// `c1 = b0*a1 + a0*b1`
///
/// `c2 = a0*a1`.
pub fn tensor(lhs: &RlweCiphertext, rhs: &RlweCiphertext) -> RlweQuadraticCiphertext {
    assert_eq!(
        lhs.b().modulus(),
        rhs.b().modulus(),
        "ciphertext moduli must match"
    );

    assert_eq!(
        lhs.b().degree(),
        rhs.b().degree(),
        "ciphertext degrees must match"
    );

    let c0 = lhs.b().negacyclic_mul(rhs.b());

    let c1 = lhs
        .b()
        .negacyclic_mul(rhs.a())
        .add(&lhs.a().negacyclic_mul(rhs.b()));

    let c2 = lhs.a().negacyclic_mul(rhs.a());

    RlweQuadraticCiphertext::new(c0, c1, c2)
}

/// NTT-backed multiplication of two rank-1 RLWE ciphertexts.
///
/// This is semantically identical to `tensor`, but all polynomial products
/// use the supplied negacyclic NTT plan.
pub fn tensor_with_ntt(
    lhs: &RlweCiphertext,
    rhs: &RlweCiphertext,
    plan: &NttPlan,
) -> RlweQuadraticCiphertext {
    assert_eq!(
        lhs.b().modulus(),
        rhs.b().modulus(),
        "ciphertext moduli must match"
    );
    assert_eq!(
        lhs.b().degree(),
        rhs.b().degree(),
        "ciphertext degrees must match"
    );
    assert_eq!(
        plan.modulus(),
        lhs.b().modulus(),
        "NTT plan modulus must match ciphertext modulus"
    );
    assert_eq!(
        plan.degree(),
        lhs.b().degree(),
        "NTT plan degree must match ciphertext degree"
    );

    let c0 = plan.negacyclic_mul(lhs.b(), rhs.b());

    let c1 = plan
        .negacyclic_mul(lhs.b(), rhs.a())
        .add(&plan.negacyclic_mul(lhs.a(), rhs.b()));

    let c2 = plan.negacyclic_mul(lhs.a(), rhs.a());

    RlweQuadraticCiphertext::new(c0, c1, c2)
}

/// Decrypts a degree-2 RLWE ciphertext without decoding.
///
/// Computes:
///
/// `c0 + c1*s + c2*s^2`.
pub fn decrypt_quadratic_raw(
    params: RlweParameters,
    secret_key: &SecretKey,
    ciphertext: &RlweQuadraticCiphertext,
) -> Polynomial {
    assert_eq!(
        ciphertext.c0().modulus(),
        params.modulus(),
        "quadratic ciphertext modulus must match RLWE parameters"
    );

    assert_eq!(
        ciphertext.c0().degree(),
        params.degree(),
        "quadratic ciphertext degree must match RLWE parameters"
    );

    let s = secret_key.polynomial();
    let s_squared = s.negacyclic_mul(s);

    let c1s = ciphertext.c1().negacyclic_mul(s);
    let c2s2 = ciphertext.c2().negacyclic_mul(&s_squared);

    ciphertext.c0().add(&c1s).add(&c2s2)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ring::Modulus;

    use super::*;
    use crate::rlwe::{decrypt_raw, encrypt_with_rng, RlwePlaintext};

    fn params() -> RlweParameters {
        RlweParameters::new(8, Modulus::new(12_289), 16, 1)
    }

    #[test]
    fn ntt_tensor_matches_reference_exactly() {
        use crate::ring::make_ntt_plan;

        let params = params();
        let plan = make_ntt_plan(params.modulus(), params.degree());

        for seed in 0_u64..32 {
            let mut key_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x9300);
            let key = SecretKey::generate_with_rng(params, &mut key_rng);

            let lhs_plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);
            let rhs_plaintext = RlwePlaintext::new(params, vec![8, 7, 6, 5, 4, 3, 2, 1]);

            let mut lhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x9310);
            let mut rhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x9320);

            let lhs = encrypt_with_rng(params, &key, &lhs_plaintext, &mut lhs_rng);
            let rhs = encrypt_with_rng(params, &key, &rhs_plaintext, &mut rhs_rng);

            assert_eq!(
                tensor_with_ntt(&lhs, &rhs, &plan),
                tensor(&lhs, &rhs),
                "NTT tensor diverged for seed {seed}"
            );
        }
    }

    #[test]
    fn tensor_expands_ciphertext_product_correctly() {
        let q = Modulus::new(97);

        let lhs = RlweCiphertext::new(
            Polynomial::new(q, vec![1, 2, 3, 4]),
            Polynomial::new(q, vec![5, 6, 7, 8]),
        );

        let rhs = RlweCiphertext::new(
            Polynomial::new(q, vec![9, 10, 11, 12]),
            Polynomial::new(q, vec![13, 14, 15, 16]),
        );

        let product = tensor(&lhs, &rhs);

        let expected_c0 = lhs.b().negacyclic_mul(rhs.b());

        let expected_c1 = lhs
            .b()
            .negacyclic_mul(rhs.a())
            .add(&lhs.a().negacyclic_mul(rhs.b()));

        let expected_c2 = lhs.a().negacyclic_mul(rhs.a());

        assert_eq!(product.c0(), &expected_c0);
        assert_eq!(product.c1(), &expected_c1);
        assert_eq!(product.c2(), &expected_c2);
    }

    #[test]
    fn quadratic_decryption_matches_product_of_raw_decryptions() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(1);
        let key = SecretKey::generate_with_rng(params, &mut key_rng);

        let lhs_plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let rhs_plaintext = RlwePlaintext::new(params, vec![8, 7, 6, 5, 4, 3, 2, 1]);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(2);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(3);

        let lhs = encrypt_with_rng(params, &key, &lhs_plaintext, &mut lhs_rng);

        let rhs = encrypt_with_rng(params, &key, &rhs_plaintext, &mut rhs_rng);

        let lhs_raw = decrypt_raw(params, &key, &lhs);
        let rhs_raw = decrypt_raw(params, &key, &rhs);

        let expected = lhs_raw.negacyclic_mul(&rhs_raw);

        let product = tensor(&lhs, &rhs);

        let actual = decrypt_quadratic_raw(params, &key, &product);

        assert_eq!(actual, expected);
    }

    #[test]
    fn quadratic_decryption_identity_holds_for_many_seeds() {
        let params = params();

        for seed in 0_u64..32 {
            let mut key_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x1111);

            let key = SecretKey::generate_with_rng(params, &mut key_rng);

            let lhs_plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);

            let rhs_plaintext = RlwePlaintext::new(params, vec![8, 7, 6, 5, 4, 3, 2, 1]);

            let mut lhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x2222);

            let mut rhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x3333);

            let lhs = encrypt_with_rng(params, &key, &lhs_plaintext, &mut lhs_rng);

            let rhs = encrypt_with_rng(params, &key, &rhs_plaintext, &mut rhs_rng);

            let expected =
                decrypt_raw(params, &key, &lhs).negacyclic_mul(&decrypt_raw(params, &key, &rhs));

            let actual = decrypt_quadratic_raw(params, &key, &tensor(&lhs, &rhs));

            assert_eq!(
                actual, expected,
                "quadratic decryption mismatch for seed {seed}"
            );
        }
    }

    #[test]
    fn tensor_is_symmetric_for_commutative_ring() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(50);
        let key = SecretKey::generate_with_rng(params, &mut key_rng);

        let plaintext = RlwePlaintext::new(params, vec![1; 8]);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(51);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(52);

        let lhs = encrypt_with_rng(params, &key, &plaintext, &mut lhs_rng);

        let rhs = encrypt_with_rng(params, &key, &plaintext, &mut rhs_rng);

        assert_eq!(tensor(&lhs, &rhs), tensor(&rhs, &lhs));
    }
}
