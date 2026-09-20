use crate::ckks::{multiply_plain_rns_ckks_with_ntt, rescale_rns_ckks_to_next, RnsCkksCiphertext};
use crate::grafting::RnsRlweCiphertext;
use crate::ring::{ModulusBasis, ModulusChain, RnsNttPlan, RnsPolynomial};
use crate::rlwe::RlweCiphertext;

use super::RnsCkksCiphertextMatrix;

/// Dense column-major matrix of encoded RNS CKKS plaintext polynomials.
#[derive(Debug, Clone, PartialEq)]
pub struct RnsCkksPlaintextMatrix {
    rows: usize,
    cols: usize,
    scale: f64,
    data: Vec<RnsPolynomial>,
}

impl RnsCkksPlaintextMatrix {
    pub fn from_vec_column_major(
        rows: usize,
        cols: usize,
        scale: f64,
        data: Vec<RnsPolynomial>,
    ) -> Self {
        assert!(
            rows > 0,
            "RNS CKKS plaintext matrix row count must be positive"
        );
        assert!(
            cols > 0,
            "RNS CKKS plaintext matrix column count must be positive"
        );
        assert!(
            scale.is_finite() && scale > 0.0,
            "RNS CKKS plaintext matrix scale must be finite and positive"
        );

        let expected = rows
            .checked_mul(cols)
            .expect("RNS CKKS plaintext matrix dimensions overflow");

        assert_eq!(
            data.len(),
            expected,
            "RNS CKKS plaintext matrix data length must equal rows * cols"
        );

        let first = &data[0];

        for plaintext in &data[1..] {
            assert_eq!(
                plaintext.basis(),
                first.basis(),
                "RNS CKKS plaintext matrix entries must have matching bases"
            );
            assert_eq!(
                plaintext.degree(),
                first.degree(),
                "RNS CKKS plaintext matrix entries must have matching ring degrees"
            );
        }

        Self {
            rows,
            cols,
            scale,
            data,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn basis(&self) -> &ModulusBasis {
        self.data[0].basis()
    }

    pub fn ring_degree(&self) -> usize {
        self.data[0].degree()
    }

    /// Returns the mathematical transpose while preserving column-major storage.
    pub fn transpose(&self) -> Self {
        let mut data = Vec::with_capacity(self.data.len());
        for col in 0..self.rows {
            for row in 0..self.cols {
                data.push(self.get(col, row).clone());
            }
        }
        Self::from_vec_column_major(self.cols, self.rows, self.scale, data)
    }

    pub fn get(&self, row: usize, col: usize) -> &RnsPolynomial {
        assert!(
            row < self.rows,
            "RNS CKKS plaintext matrix row index out of bounds"
        );
        assert!(
            col < self.cols,
            "RNS CKKS plaintext matrix column index out of bounds"
        );

        &self.data[row + col * self.rows]
    }
}

fn add_ciphertexts(
    lhs: &RnsCkksCiphertext,
    rhs: &RnsCkksCiphertext,
    chain: &ModulusChain,
) -> RnsCkksCiphertext {
    lhs.assert_matches_chain(chain);
    rhs.assert_matches_chain(chain);

    assert_eq!(
        lhs.level(),
        rhs.level(),
        "RNS CKKS addition requires matching levels"
    );
    assert_eq!(
        lhs.basis(),
        rhs.basis(),
        "RNS CKKS addition requires matching bases"
    );
    assert_eq!(
        lhs.scale(),
        rhs.scale(),
        "RNS CKKS addition requires matching scales"
    );

    let limbs = lhs
        .rlwe()
        .limbs()
        .iter()
        .zip(rhs.rlwe().limbs())
        .map(|(lhs_limb, rhs_limb)| {
            RlweCiphertext::new(
                lhs_limb.b().add(rhs_limb.b()),
                lhs_limb.a().add(rhs_limb.a()),
            )
        })
        .collect();

    RnsCkksCiphertext::new(
        RnsRlweCiphertext::from_limbs(limbs),
        lhs.state().clone(),
        chain,
    )
}

impl RnsCkksCiphertextMatrix {
    /// Ciphertext-plaintext matrix multiplication for leveled RNS CKKS.
    ///
    /// Each dot product is accumulated at product scale, then rescaled once.
    /// No evaluation key or relinearization is required.
    pub fn matmul_plain_with_ntt(
        &self,
        rhs: &RnsCkksPlaintextMatrix,
        chain: &ModulusChain,
        plan: &RnsNttPlan,
    ) -> Self {
        assert_eq!(
            self.cols(),
            rhs.rows(),
            "RNS CKKS ciphertext/plaintext matrix dimensions must be compatible"
        );
        assert_eq!(
            self.get(0, 0).basis(),
            rhs.basis(),
            "RNS CKKS ciphertext/plaintext matrix bases must match"
        );
        assert_eq!(
            self.ring_degree(),
            rhs.ring_degree(),
            "RNS CKKS ciphertext/plaintext matrix ring degrees must match"
        );
        assert_eq!(
            plan.degree(),
            self.ring_degree(),
            "RNS NTT plan degree must match matrix ring degree"
        );
        assert_eq!(
            plan.moduli(),
            self.get(0, 0).basis().moduli(),
            "RNS NTT plan basis must match matrix basis"
        );

        let mut data = Vec::with_capacity(self.rows() * rhs.cols());

        for col in 0..rhs.cols() {
            for row in 0..self.rows() {
                let mut accumulator = multiply_plain_rns_ckks_with_ntt(
                    self.get(row, 0),
                    rhs.get(0, col),
                    rhs.scale(),
                    chain,
                    plan,
                );

                for inner in 1..self.cols() {
                    let product = multiply_plain_rns_ckks_with_ntt(
                        self.get(row, inner),
                        rhs.get(inner, col),
                        rhs.scale(),
                        chain,
                        plan,
                    );

                    accumulator = add_ciphertexts(&accumulator, &product, chain);
                }

                data.push(rescale_rns_ckks_to_next(&accumulator, chain));
            }
        }

        Self::from_vec_column_major(self.rows(), rhs.cols(), data)
    }
}
