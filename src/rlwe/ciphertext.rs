use crate::ring::Polynomial;

/// Rank-1 RLWE ciphertext `(b, a)`.
///
/// Decryption computes:
///
/// `b + a*s`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlweCiphertext {
    b: Polynomial,
    a: Polynomial,
}

impl RlweCiphertext {
    pub fn new(b: Polynomial, a: Polynomial) -> Self {
        assert_eq!(
            b.modulus(),
            a.modulus(),
            "ciphertext polynomial moduli must match"
        );

        assert_eq!(
            b.degree(),
            a.degree(),
            "ciphertext polynomial degrees must match"
        );

        Self { b, a }
    }

    pub fn b(&self) -> &Polynomial {
        &self.b
    }

    pub fn a(&self) -> &Polynomial {
        &self.a
    }

    pub fn into_components(self) -> (Polynomial, Polynomial) {
        (self.b, self.a)
    }
}
