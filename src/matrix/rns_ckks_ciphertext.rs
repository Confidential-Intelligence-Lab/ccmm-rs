use crate::ckks::{RnsCkksCiphertext, RnsCkksEvaluator};
use crate::ring::RnsNttPlan;

/// Dense column-major matrix of leveled RNS CKKS ciphertexts.
///
/// Storage follows the same convention as the existing matrix layer:
///
/// ```text
/// index = row + column * rows
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RnsCkksCiphertextMatrix {
    rows: usize,
    cols: usize,
    data: Vec<RnsCkksCiphertext>,
}

impl RnsCkksCiphertextMatrix {
    pub fn from_vec_column_major(rows: usize, cols: usize, data: Vec<RnsCkksCiphertext>) -> Self {
        assert!(rows > 0, "RNS CKKS matrix row count must be positive");
        assert!(cols > 0, "RNS CKKS matrix column count must be positive");

        let expected = rows
            .checked_mul(cols)
            .expect("RNS CKKS matrix dimensions overflow");

        assert_eq!(
            data.len(),
            expected,
            "RNS CKKS matrix data length must equal rows * cols"
        );

        let first = &data[0];

        for ciphertext in &data[1..] {
            assert_eq!(
                ciphertext.level(),
                first.level(),
                "RNS CKKS matrix entries must have matching levels"
            );
            assert_eq!(
                ciphertext.basis(),
                first.basis(),
                "RNS CKKS matrix entries must have matching bases"
            );
            assert_eq!(
                ciphertext.scale(),
                first.scale(),
                "RNS CKKS matrix entries must have matching scales"
            );
            assert_eq!(
                ciphertext.rlwe().degree(),
                first.rlwe().degree(),
                "RNS CKKS matrix entries must have matching ring degrees"
            );
        }

        Self { rows, cols, data }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, row: usize, col: usize) -> &RnsCkksCiphertext {
        assert!(row < self.rows, "RNS CKKS matrix row index out of bounds");
        assert!(
            col < self.cols,
            "RNS CKKS matrix column index out of bounds"
        );

        &self.data[self.index(row, col)]
    }

    pub fn level(&self) -> usize {
        self.data[0].level()
    }

    pub fn scale(&self) -> f64 {
        self.data[0].scale()
    }

    pub fn ring_degree(&self) -> usize {
        self.data[0].rlwe().degree()
    }

    pub fn into_vec_column_major(self) -> Vec<RnsCkksCiphertext> {
        self.data
    }

    fn index(&self, row: usize, col: usize) -> usize {
        row + col * self.rows
    }
}

impl RnsCkksCiphertextMatrix {
    /// Multiplies two encrypted matrices using the NTT-backed leveled
    /// RNS CKKS evaluator.
    ///
    /// For
    ///
    /// ```text
    /// lhs: m x k
    /// rhs: k x n
    /// ```
    ///
    /// the result is `m x n`. Each scalar product is independently
    /// multiplied, relinearized, and rescaled before the products of
    /// each dot product are accumulated.
    pub fn matmul_with_ntt(
        &self,
        rhs: &Self,
        evaluator: &RnsCkksEvaluator<'_>,
        plan: &RnsNttPlan,
    ) -> Self {
        assert_eq!(
            self.cols, rhs.rows,
            "RNS CKKS matrix dimensions must be compatible"
        );
        assert_eq!(
            self.level(),
            rhs.level(),
            "RNS CKKS matrix operands must have matching levels"
        );
        assert_eq!(
            self.data[0].basis(),
            rhs.data[0].basis(),
            "RNS CKKS matrix operands must have matching bases"
        );
        assert_eq!(
            self.scale(),
            rhs.scale(),
            "RNS CKKS matrix operands must have matching scales"
        );
        assert_eq!(
            self.ring_degree(),
            rhs.ring_degree(),
            "RNS CKKS matrix operands must have matching ring degrees"
        );

        assert_eq!(
            plan.degree(),
            self.ring_degree(),
            "RNS NTT plan degree must match matrix ciphertext degree"
        );
        assert_eq!(
            plan.moduli(),
            self.data[0].basis().moduli(),
            "RNS NTT plan basis must match matrix ciphertext basis"
        );

        let mut data = Vec::with_capacity(self.rows * rhs.cols);

        // Column-major output order.
        for col in 0..rhs.cols {
            for row in 0..self.rows {
                let mut accumulator =
                    evaluator.multiply_with_ntt(self.get(row, 0), rhs.get(0, col), plan);

                for inner in 1..self.cols {
                    let product = evaluator.multiply_with_ntt(
                        self.get(row, inner),
                        rhs.get(inner, col),
                        plan,
                    );

                    accumulator = evaluator.add(&accumulator, &product);
                }

                data.push(accumulator);
            }
        }

        Self::from_vec_column_major(self.rows, rhs.cols, data)
    }
    /// Multiplies two encrypted CKKS matrices using the bounded-base
    /// Gaussian-capable R3.1 multiplication path.
    pub fn matmul_bounded_with_ntt(
        &self,
        rhs: &Self,
        multiplication_key: &crate::grafting::BoundedRnsMultiplicationKey,
        chain: &crate::ring::ModulusChain,
        plan: &RnsNttPlan,
    ) -> Self {
        assert_eq!(
            self.cols, rhs.rows,
            "RNS CKKS matrix dimensions must be compatible"
        );
        assert_eq!(
            self.level(),
            rhs.level(),
            "RNS CKKS matrix operands must have matching levels"
        );
        assert_eq!(
            self.data[0].basis(),
            rhs.data[0].basis(),
            "RNS CKKS matrix operands must have matching bases"
        );
        assert_eq!(
            self.scale(),
            rhs.scale(),
            "RNS CKKS matrix operands must have matching scales"
        );
        assert_eq!(
            self.ring_degree(),
            rhs.ring_degree(),
            "RNS CKKS matrix operands must have matching ring degrees"
        );
        assert_eq!(
            multiplication_key.layout().full_basis(),
            self.data[0].basis(),
            "bounded multiplication-key basis must match matrix ciphertext basis"
        );
        assert_eq!(
            plan.degree(),
            self.ring_degree(),
            "RNS NTT plan degree must match matrix ciphertext degree"
        );
        assert_eq!(
            plan.moduli(),
            self.data[0].basis().moduli(),
            "RNS NTT plan basis must match matrix ciphertext basis"
        );

        fn add_ciphertexts(
            lhs: &RnsCkksCiphertext,
            rhs: &RnsCkksCiphertext,
            chain: &crate::ring::ModulusChain,
        ) -> RnsCkksCiphertext {
            use crate::grafting::RnsRlweCiphertext;
            use crate::rlwe::RlweCiphertext;

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

        let mut data = Vec::with_capacity(self.rows * rhs.cols);

        for col in 0..rhs.cols {
            for row in 0..self.rows {
                let mut accumulator =
                    crate::ckks::multiply_relinearize_rescale_rns_ckks_bounded_with_ntt(
                        self.get(row, 0),
                        rhs.get(0, col),
                        multiplication_key,
                        chain,
                        plan,
                    );

                for inner in 1..self.cols {
                    let product =
                        crate::ckks::multiply_relinearize_rescale_rns_ckks_bounded_with_ntt(
                            self.get(row, inner),
                            rhs.get(inner, col),
                            multiplication_key,
                            chain,
                            plan,
                        );

                    accumulator = add_ciphertexts(&accumulator, &product, chain);
                }

                data.push(accumulator);
            }
        }

        Self::from_vec_column_major(self.rows, rhs.cols, data)
    }
}

#[cfg(test)]
mod tests {
    use crate::ckks::CkksChainState;
    use crate::grafting::RnsRlweCiphertext;
    use crate::ring::{Modulus, ModulusBasis, ModulusChain, Polynomial};
    use crate::rlwe::RlweCiphertext;

    use super::*;

    fn chain() -> ModulusChain {
        ModulusChain::from_top_basis(ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]))
    }

    fn zero_ciphertext(
        chain: &ModulusChain,
        level: usize,
        degree: usize,
        scale: f64,
    ) -> RnsCkksCiphertext {
        let limbs = chain
            .level(level)
            .moduli()
            .iter()
            .copied()
            .map(|modulus| {
                RlweCiphertext::new(
                    Polynomial::zero(modulus, degree),
                    Polynomial::zero(modulus, degree),
                )
            })
            .collect();

        RnsCkksCiphertext::new(
            RnsRlweCiphertext::from_limbs(limbs),
            CkksChainState::new(chain, level, scale),
            chain,
        )
    }

    #[test]
    fn matrix_preserves_column_major_layout() {
        let chain = chain();

        let entries = (0..4)
            .map(|_| zero_ciphertext(&chain, 0, 8, 256.0))
            .collect();

        let matrix = RnsCkksCiphertextMatrix::from_vec_column_major(2, 2, entries);

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.len(), 4);
        assert_eq!(matrix.level(), 0);
        assert_eq!(matrix.scale(), 256.0);
        assert_eq!(matrix.ring_degree(), 8);

        assert_eq!(matrix.get(0, 0).level(), 0);
        assert_eq!(matrix.get(1, 0).level(), 0);
        assert_eq!(matrix.get(0, 1).level(), 0);
        assert_eq!(matrix.get(1, 1).level(), 0);
    }

    #[test]
    #[should_panic(expected = "matching levels")]
    fn matrix_rejects_mixed_levels() {
        let chain = chain();

        let data = vec![
            zero_ciphertext(&chain, 0, 8, 256.0),
            zero_ciphertext(&chain, 1, 8, 256.0),
        ];

        let _ = RnsCkksCiphertextMatrix::from_vec_column_major(2, 1, data);
    }

    #[test]
    #[should_panic(expected = "matching scales")]
    fn matrix_rejects_mixed_scales() {
        let chain = chain();

        let data = vec![
            zero_ciphertext(&chain, 0, 8, 256.0),
            zero_ciphertext(&chain, 0, 8, 512.0),
        ];

        let _ = RnsCkksCiphertextMatrix::from_vec_column_major(2, 1, data);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn matrix_rejects_wrong_data_length() {
        let chain = chain();

        let data = vec![zero_ciphertext(&chain, 0, 8, 256.0)];

        let _ = RnsCkksCiphertextMatrix::from_vec_column_major(2, 2, data);
    }

    #[test]
    fn ntt_ccmm_2x2_matches_canonical_slot_reference() {
        use num_complex::Complex64;
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        use crate::ckks::{
            CkksCanonicalEmbedding, CkksChainState, CkksSlotEncoder, RnsCkksEvaluationKeys,
            RnsCkksLevelKeys,
        };
        use crate::grafting::{
            decrypt_rns_raw, encrypt_rns_raw_with_ntt_rng, RnsGadgetLayout, RnsKeygenConfig,
            RnsMultiplicationKey,
        };
        use crate::ring::{Modulus, ModulusBasis, ModulusChain, RnsNttPlan, RnsPolynomial};

        let degree = 8;
        let scale = 65_537.0;
        let secret = [-1, 0, 1, 1, 0, -1, 1, 0];

        let chain = ModulusChain::from_top_basis(ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]));
        let plan = RnsNttPlan::new(chain.level(0).moduli().to_vec(), degree);

        let layout = RnsGadgetLayout::new(chain.level(0).clone(), vec![1, 2]);
        let mut key_rng = ChaCha20Rng::seed_from_u64(0xCC10);
        let multiplication_key = RnsMultiplicationKey::generate_with_ntt_rng(
            RnsKeygenConfig {
                degree,
                plaintext_modulus: 2,
                noise_bound: 0,
                layout,
                plan: &plan,
            },
            &secret,
            &mut key_rng,
        );

        let mut level_keys = RnsCkksLevelKeys::new(0, chain.level(0).clone());
        level_keys.set_multiplication_key(multiplication_key);

        let mut evaluation_keys = RnsCkksEvaluationKeys::new();
        evaluation_keys.insert_level(level_keys);

        let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

        let encode_encrypt = |value: f64, seed: u64| {
            let slots = vec![Complex64::new(value, 0.0); degree / 2];

            /*
             * Canonical CKKS encoding produces one signed polynomial.
             * Project the same logical polynomial into the active RNS basis.
             */
            let temporary_modulus = Modulus::new(2_147_483_647);
            let encoder = CkksSlotEncoder::new(degree, temporary_modulus, scale);
            let encoded = encoder.encode_slots(&slots);

            let signed: Vec<i128> = encoded
                .coefficients()
                .iter()
                .map(|&coefficient| {
                    let value = i128::from(coefficient);
                    let modulus = i128::from(temporary_modulus.value());
                    if value > modulus / 2 {
                        value - modulus
                    } else {
                        value
                    }
                })
                .collect();

            let composite_modulus = chain.level(0).composite_modulus();

            let coefficients: Vec<u128> = signed
                .iter()
                .map(|&value| {
                    let modulus = composite_modulus as i128;
                    ((value % modulus + modulus) % modulus) as u128
                })
                .collect();

            let message =
                RnsPolynomial::from_coefficients(chain.level(0).moduli().to_vec(), &coefficients);

            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let inner = encrypt_rns_raw_with_ntt_rng(&message, 2, 0, &secret, &plan, &mut rng);

            RnsCkksCiphertext::new(inner, CkksChainState::new(&chain, 0, scale), &chain)
        };

        /*
         * Column-major storage:
         *
         * A = [  0.25    0.50  ]
         *     [ -0.25    0.125 ]
         *
         * B = [ 0.50   -0.25 ]
         *     [ 0.25    0.50 ]
         */
        let lhs = RnsCkksCiphertextMatrix::from_vec_column_major(
            2,
            2,
            vec![
                encode_encrypt(0.25, 0xCC20),
                encode_encrypt(-0.25, 0xCC21),
                encode_encrypt(0.50, 0xCC22),
                encode_encrypt(0.125, 0xCC23),
            ],
        );

        let rhs = RnsCkksCiphertextMatrix::from_vec_column_major(
            2,
            2,
            vec![
                encode_encrypt(0.50, 0xCC30),
                encode_encrypt(0.25, 0xCC31),
                encode_encrypt(-0.25, 0xCC32),
                encode_encrypt(0.50, 0xCC33),
            ],
        );

        let result = lhs.matmul_with_ntt(&rhs, &evaluator, &plan);

        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 2);
        assert_eq!(result.level(), 1);
        assert_eq!(result.data[0].basis(), chain.level(1));

        let expected = [[0.25, 0.1875], [-0.09375, 0.125]];

        let tolerance = 0.002;

        for (row, expected_row) in expected.iter().enumerate() {
            for (col, &expected_value) in expected_row.iter().enumerate() {
                let ciphertext = result.get(row, col);
                let decrypted = decrypt_rns_raw(ciphertext.rlwe(), &secret);
                let modulus = decrypted.composite_modulus();

                let coefficients: Vec<f64> = decrypted
                    .reconstruct_coefficients()
                    .into_iter()
                    .map(|value| {
                        let value = value as i128;
                        let modulus = modulus as i128;
                        let centered = if value > modulus / 2 {
                            value - modulus
                        } else {
                            value
                        };
                        centered as f64 / ciphertext.scale()
                    })
                    .collect();

                let slots =
                    CkksCanonicalEmbedding::new(degree).coefficients_to_slots(&coefficients);

                for (slot, actual) in slots.iter().enumerate() {
                    let error = (actual.re - expected_value).abs();

                    assert!(
                        error <= tolerance,
                        "C[{row},{col}] slot {slot}: \
                         actual={actual:?}, \
                         expected={}, \
                         error={error}, \
                         tolerance={tolerance}",
                        expected[row][col]
                    );

                    assert!(
                        actual.im.abs() <= tolerance,
                        "C[{row},{col}] slot {slot}: unexpected imaginary component {actual:?}"
                    );
                }
            }
        }
    }
}
