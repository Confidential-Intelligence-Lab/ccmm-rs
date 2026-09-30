//! Batch matrix representation compatible with the CRYPTO CCMM artifact.
//!
//! The reference HEaaN implementation uses `RingSwitchHelper::combine`
//! and `split` to map `d` scalar-ring polynomials of degree `N_s` to one
//! large-ring polynomial of degree `N_l = d * N_s`.
//!
//! Black-box validation against the bundled HEaaN implementation establishes
//! the exact coefficient mapping:
//!
//!     large[j * d + r] = scalar[r][j]
//!
//! where:
//!   * `r` is the scalar-ring part / matrix-row index,
//!   * `j` is the scalar-ring coefficient index.
//!
//! `split` is the exact inverse.

/// Interleave equal-length scalar-ring coefficient vectors into one
/// large-ring coefficient vector using the HEaaN RingSwitchHelper layout.
///
/// If there are `d` parts, each of length `n`, the result has length `d*n`
/// and satisfies:
///
/// `result[j * d + r] == parts[r][j]`.
pub fn combine_scalar_coefficients(parts: &[Vec<u64>]) -> Vec<u64> {
    assert!(
        !parts.is_empty(),
        "batch ring combine requires at least one scalar part"
    );

    let scalar_degree = parts[0].len();

    assert!(
        scalar_degree > 0,
        "batch ring combine requires nonempty scalar polynomials"
    );

    assert!(
        parts.iter().all(|part| part.len() == scalar_degree),
        "all scalar parts must have the same degree"
    );

    let dimension = parts.len();
    let mut combined = vec![0_u64; dimension * scalar_degree];

    for coefficient in 0..scalar_degree {
        for row in 0..dimension {
            combined[coefficient * dimension + row] = parts[row][coefficient];
        }
    }

    combined
}

/// Split one large-ring coefficient vector into `dimension` scalar-ring
/// coefficient vectors using the inverse HEaaN RingSwitchHelper layout.
///
/// The input degree must be divisible by `dimension`.
pub fn split_large_coefficients(combined: &[u64], dimension: usize) -> Vec<Vec<u64>> {
    assert!(
        dimension > 0,
        "batch ring split requires a nonzero dimension"
    );

    assert!(
        !combined.is_empty(),
        "batch ring split requires a nonempty large polynomial"
    );

    assert_eq!(
        combined.len() % dimension,
        0,
        "large-ring degree must be divisible by batch dimension"
    );

    let scalar_degree = combined.len() / dimension;
    let mut parts = vec![vec![0_u64; scalar_degree]; dimension];

    for coefficient in 0..scalar_degree {
        for row in 0..dimension {
            parts[row][coefficient] = combined[coefficient * dimension + row];
        }
    }

    parts
}

use crate::ckks::CkksCanonicalEmbedding;
use crate::grafting::RnsRlweCiphertext;
use crate::ring::{Polynomial, RnsPolynomial};
use crate::rlwe::RlweCiphertext;
use num_complex::Complex64;

/// Combine scalar-ring RNS polynomials into one large-ring RNS polynomial
/// using the exact HEaaN RingSwitchHelper coefficient layout.
///
/// For `d` parts of scalar degree `N_s`, the output degree is `d * N_s` and
/// every residue limb satisfies
///
///     large[j * d + r] = parts[r][j].
///
/// The operation is performed independently in every RNS residue. No CRT
/// reconstruction is involved.
pub fn combine_rns_polynomials(parts: &[&RnsPolynomial]) -> RnsPolynomial {
    assert!(
        !parts.is_empty(),
        "batch RNS ring combine requires at least one scalar part"
    );

    let scalar_degree = parts[0].degree();
    let basis = parts[0].basis();

    assert!(
        scalar_degree > 0,
        "batch RNS ring combine requires nonempty scalar polynomials"
    );

    for part in parts {
        assert_eq!(
            part.degree(),
            scalar_degree,
            "all scalar RNS parts must have the same degree"
        );
        assert_eq!(
            part.basis(),
            basis,
            "all scalar RNS parts must have the same modulus basis"
        );
    }

    let dimension = parts.len();

    let residues = (0..basis.len())
        .map(|limb_index| {
            let modulus = parts[0].residue(limb_index).modulus();

            let scalar_coefficients: Vec<Vec<u64>> = parts
                .iter()
                .map(|part| part.residue(limb_index).coefficients().to_vec())
                .collect();

            Polynomial::new(modulus, combine_scalar_coefficients(&scalar_coefficients))
        })
        .collect();

    let combined = RnsPolynomial::from_residues(residues);

    assert_eq!(combined.degree(), dimension * scalar_degree);

    combined
}

/// Split one large-ring RNS polynomial into scalar-ring RNS polynomials using
/// the exact inverse HEaaN RingSwitchHelper coefficient layout.
pub fn split_rns_polynomial(combined: &RnsPolynomial, dimension: usize) -> Vec<RnsPolynomial> {
    assert!(
        dimension > 0,
        "batch RNS ring split requires a nonzero dimension"
    );

    assert_eq!(
        combined.degree() % dimension,
        0,
        "large-ring RNS degree must be divisible by batch dimension"
    );

    let scalar_degree = combined.degree() / dimension;

    let mut part_residues: Vec<Vec<Polynomial>> = (0..dimension).map(|_| Vec::new()).collect();

    for residue in combined.residues() {
        let split = split_large_coefficients(residue.coefficients(), dimension);

        for row in 0..dimension {
            part_residues[row].push(Polynomial::new(residue.modulus(), split[row].clone()));
        }
    }

    let parts: Vec<RnsPolynomial> = part_residues
        .into_iter()
        .map(RnsPolynomial::from_residues)
        .collect();

    assert!(
        parts.iter().all(|part| part.degree() == scalar_degree),
        "split RNS parts must all have the scalar degree"
    );

    parts
}

/// Combine scalar-ring RNS RLWE ciphertexts into one large-ring ciphertext.
///
/// Both RLWE components are combined independently using the exact HEaaN
/// RingSwitchHelper coefficient layout, residue by residue.
pub fn combine_rns_rlwe_ciphertexts(parts: &[&RnsRlweCiphertext]) -> RnsRlweCiphertext {
    assert!(
        !parts.is_empty(),
        "batch RNS RLWE combine requires at least one ciphertext"
    );

    let scalar_degree = parts[0].degree();
    let basis = parts[0].basis();
    let limb_count = parts[0].limbs().len();

    for part in parts {
        assert_eq!(
            part.degree(),
            scalar_degree,
            "all scalar RNS RLWE ciphertexts must have the same degree"
        );
        assert_eq!(
            part.basis(),
            basis,
            "all scalar RNS RLWE ciphertexts must have the same modulus basis"
        );
        assert_eq!(
            part.limbs().len(),
            limb_count,
            "all scalar RNS RLWE ciphertexts must have the same limb count"
        );
    }

    let limbs = (0..limb_count)
        .map(|limb_index| {
            let limb_parts: Vec<&RlweCiphertext> =
                parts.iter().map(|part| part.limb(limb_index)).collect();

            let b_parts: Vec<Vec<u64>> = limb_parts
                .iter()
                .map(|limb| limb.b().coefficients().to_vec())
                .collect();

            let a_parts: Vec<Vec<u64>> = limb_parts
                .iter()
                .map(|limb| limb.a().coefficients().to_vec())
                .collect();

            let modulus = limb_parts[0].b().modulus();

            RlweCiphertext::new(
                Polynomial::new(modulus, combine_scalar_coefficients(&b_parts)),
                Polynomial::new(modulus, combine_scalar_coefficients(&a_parts)),
            )
        })
        .collect();

    RnsRlweCiphertext::from_limbs(limbs)
}

/// Split one large-ring RNS RLWE ciphertext into scalar-ring ciphertexts using
/// the exact inverse HEaaN RingSwitchHelper coefficient layout.
pub fn split_rns_rlwe_ciphertext(
    combined: &RnsRlweCiphertext,
    dimension: usize,
) -> Vec<RnsRlweCiphertext> {
    assert!(
        dimension > 0,
        "batch RNS RLWE split requires a nonzero dimension"
    );

    assert_eq!(
        combined.degree() % dimension,
        0,
        "large-ring RNS RLWE degree must be divisible by batch dimension"
    );

    let limb_count = combined.limbs().len();

    let mut part_limbs: Vec<Vec<RlweCiphertext>> = (0..dimension).map(|_| Vec::new()).collect();

    for limb_index in 0..limb_count {
        let limb = combined.limb(limb_index);

        let split_b = split_large_coefficients(limb.b().coefficients(), dimension);
        let split_a = split_large_coefficients(limb.a().coefficients(), dimension);

        for row in 0..dimension {
            let modulus = limb.b().modulus();

            part_limbs[row].push(RlweCiphertext::new(
                Polynomial::new(modulus, split_b[row].clone()),
                Polynomial::new(modulus, split_a[row].clone()),
            ));
        }
    }

    part_limbs
        .into_iter()
        .map(RnsRlweCiphertext::from_limbs)
        .collect()
}

/// Structural SinC plaintext representation used by the CRYPTO Batch method.
///
/// Each matrix column is represented by one large-ring coefficient vector.
/// If the scalar ring has degree `N_s` and the logical matrix dimension is
/// `d`, each large-ring column has degree
///
///     N_l = d * N_s.
///
/// The row-polynomial mapping matches HEaaN `RingSwitchHelper::combine`:
///
///     large[j * d + r] = scalar_row[r][j].
#[derive(Debug, Clone)]
pub struct SinCBatchPlaintext {
    scalar_degree: usize,
    dimension: usize,
    columns: Vec<Vec<f64>>,
}

impl SinCBatchPlaintext {
    pub fn scalar_degree(&self) -> usize {
        self.scalar_degree
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn large_degree(&self) -> usize {
        self.scalar_degree * self.dimension
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn columns(&self) -> &[Vec<f64>] {
        &self.columns
    }
}

fn log2_exact(value: usize) -> u32 {
    assert!(
        value.is_power_of_two(),
        "SinC slot count must be a power of two"
    );
    value.trailing_zeros()
}

/// Reverse exactly `bits` low-order bits.
///
/// This reproduces the slot permutation used by the authors'
/// `SinCEnDecoder`.
pub fn bit_reverse_index(index: usize, bits: u32) -> usize {
    assert!(bits < usize::BITS, "bit-reversal width exceeds usize width");

    if bits == 0 {
        return 0;
    }

    index.reverse_bits() >> (usize::BITS - bits)
}

fn combine_scalar_f64(parts: &[Vec<f64>]) -> Vec<f64> {
    assert!(
        !parts.is_empty(),
        "SinC combine requires at least one scalar polynomial"
    );

    let scalar_degree = parts[0].len();

    assert!(
        parts.iter().all(|part| part.len() == scalar_degree),
        "all SinC scalar polynomials must have equal degree"
    );

    let dimension = parts.len();
    let mut combined = vec![0.0_f64; scalar_degree * dimension];

    for coefficient in 0..scalar_degree {
        for row in 0..dimension {
            combined[coefficient * dimension + row] = parts[row][coefficient];
        }
    }

    combined
}

fn split_large_f64(combined: &[f64], dimension: usize) -> Vec<Vec<f64>> {
    assert!(dimension > 0, "SinC split dimension must be nonzero");
    assert_eq!(
        combined.len() % dimension,
        0,
        "large-ring degree must be divisible by SinC dimension"
    );

    let scalar_degree = combined.len() / dimension;
    let mut parts = vec![vec![0.0_f64; scalar_degree]; dimension];

    for coefficient in 0..scalar_degree {
        for row in 0..dimension {
            parts[row][coefficient] = combined[coefficient * dimension + row];
        }
    }

    parts
}

/// Encode a full Batch-method matrix batch using the structural semantics of
/// the authors' `SinCEnDecoder::encode`.
///
/// `matrices[batch][row][column]` is packed as follows:
///
/// 1. for each `(row, column)`, collect that matrix entry across batches;
/// 2. place batches in the authors' bit-reversed CKKS slot order;
/// 3. CKKS canonical-encode those scalar-ring slots;
/// 4. combine all row polynomials into one large-ring polynomial per column
///    using the experimentally recovered HEaaN interleave.
///
/// This function intentionally stops before RNS quantization/encryption.
/// It validates the SinC representation independently of cryptographic noise.
pub fn sinc_encode_batch(
    matrices: &[Vec<Vec<Complex64>>],
    scalar_degree: usize,
) -> SinCBatchPlaintext {
    assert!(!matrices.is_empty(), "SinC batch must not be empty");
    assert!(
        scalar_degree >= 2 && scalar_degree.is_power_of_two(),
        "SinC scalar degree must be a power of two >= 2"
    );

    let slot_count = scalar_degree / 2;

    assert_eq!(
        matrices.len(),
        slot_count,
        "authors-compatible SinC encoding requires a full scalar CKKS slot batch"
    );

    let dimension = matrices[0].len();

    assert!(dimension > 0, "SinC matrices must have nonzero dimension");

    let num_columns = matrices[0][0].len();

    assert!(num_columns > 0, "SinC matrices must have nonzero columns");

    assert!(
        matrices.iter().all(|matrix| {
            matrix.len() == dimension && matrix.iter().all(|row| row.len() == num_columns)
        }),
        "all SinC matrices must have identical dimensions"
    );

    let embedding = CkksCanonicalEmbedding::new(scalar_degree);
    assert_eq!(embedding.slot_count(), slot_count);

    let log_slots = log2_exact(slot_count);

    let mut columns = Vec::with_capacity(num_columns);

    for column in 0..num_columns {
        let mut scalar_rows = Vec::with_capacity(dimension);

        for row in 0..dimension {
            let mut slots = vec![Complex64::new(0.0, 0.0); slot_count];

            for (batch_slot, slot) in slots.iter_mut().enumerate() {
                let source_batch = bit_reverse_index(batch_slot, log_slots);

                *slot = matrices[source_batch][row][column];
            }

            let coefficients = embedding.slots_to_coefficients(&slots);

            assert_eq!(
                coefficients.len(),
                scalar_degree,
                "canonical embedding returned unexpected scalar degree"
            );

            scalar_rows.push(coefficients);
        }

        columns.push(combine_scalar_f64(&scalar_rows));
    }

    SinCBatchPlaintext {
        scalar_degree,
        dimension,
        columns,
    }
}

/// Decode a structural SinC plaintext using the inverse of
/// `sinc_encode_batch`.
pub fn sinc_decode_batch(plaintext: &SinCBatchPlaintext) -> Vec<Vec<Vec<Complex64>>> {
    let scalar_degree = plaintext.scalar_degree;
    let dimension = plaintext.dimension;
    let slot_count = scalar_degree / 2;
    let log_slots = log2_exact(slot_count);
    let num_columns = plaintext.columns.len();

    let embedding = CkksCanonicalEmbedding::new(scalar_degree);

    let mut result = vec![vec![vec![Complex64::new(0.0, 0.0); num_columns]; dimension]; slot_count];

    for column in 0..num_columns {
        let scalar_rows = split_large_f64(&plaintext.columns[column], dimension);

        assert_eq!(scalar_rows.len(), dimension);

        for (row, scalar_row) in scalar_rows.iter().enumerate() {
            let slots = embedding.coefficients_to_slots(scalar_row);

            assert_eq!(
                slots.len(),
                slot_count,
                "canonical decoder returned unexpected slot count"
            );

            for (batch, matrix) in result.iter_mut().enumerate() {
                matrix[row][column] = slots[bit_reverse_index(batch, log_slots)];
            }
        }
    }

    result
}

/// Authors' CPMM half-row packing.
///
/// For a real d x d matrix M with even d, returns the complex
/// (d/2) x d matrix H defined by
///
///     H[r][c] = M[r][c] + i M[r + d/2][c].
///
pub fn as_half_row(matrix: &[Vec<f64>]) -> Vec<Vec<Complex64>> {
    let dimension = matrix.len();

    assert!(dimension > 0, "matrix must be non-empty");
    assert_eq!(
        dimension % 2,
        0,
        "half-row packing requires an even row dimension"
    );

    for row in matrix {
        assert_eq!(
            row.len(),
            dimension,
            "half-row packing requires a square matrix"
        );
    }

    let half = dimension / 2;
    let mut packed = vec![vec![Complex64::new(0.0, 0.0); dimension]; half];

    for row in 0..half {
        for column in 0..dimension {
            packed[row][column] = Complex64::new(matrix[row][column], matrix[row + half][column]);
        }
    }

    packed
}

/// Inverse of [`as_half_row`].
///
/// For a complex (d/2) x d matrix H, reconstructs the real d x d
/// matrix whose upper half is Re(H) and lower half is Im(H).
pub fn as_double_row(matrix: &[Vec<Complex64>]) -> Vec<Vec<f64>> {
    let half = matrix.len();

    assert!(half > 0, "matrix must be non-empty");

    let dimension = matrix[0].len();

    assert_eq!(
        dimension,
        2 * half,
        "double-row unpacking expects a (d/2) x d matrix"
    );

    for row in matrix {
        assert_eq!(
            row.len(),
            dimension,
            "double-row unpacking requires uniform rows"
        );
    }

    let mut unpacked = vec![vec![0.0; dimension]; dimension];

    for row in 0..half {
        for column in 0..dimension {
            unpacked[row][column] = matrix[row][column].re;
            unpacked[row + half][column] = matrix[row][column].im;
        }
    }

    unpacked
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    #[test]
    fn authors_half_double_row_exact_layout() {
        let matrix = vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![10.0, 11.0, 12.0, 13.0],
            vec![20.0, 21.0, 22.0, 23.0],
            vec![30.0, 31.0, 32.0, 33.0],
        ];

        let packed = super::as_half_row(&matrix);

        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0].len(), 4);

        assert_eq!(packed[0][0], Complex64::new(0.0, 20.0));
        assert_eq!(packed[0][3], Complex64::new(3.0, 23.0));
        assert_eq!(packed[1][0], Complex64::new(10.0, 30.0));
        assert_eq!(packed[1][3], Complex64::new(13.0, 33.0));

        let recovered = super::as_double_row(&packed);
        assert_eq!(recovered, matrix);
    }

    #[test]
    fn authors_half_double_row_roundtrip_d64() {
        let dimension = 64;
        let matrix: Vec<Vec<f64>> = (0..dimension)
            .map(|row| {
                (0..dimension)
                    .map(|column| (row * dimension + column) as f64 / 4096.0)
                    .collect()
            })
            .collect();

        let packed = super::as_half_row(&matrix);

        assert_eq!(packed.len(), 32);
        assert_eq!(packed[0].len(), 64);

        let recovered = super::as_double_row(&packed);
        assert_eq!(recovered, matrix);
    }

    #[test]
    fn authors_half_double_row_roundtrip_d128() {
        let dimension = 128;
        let matrix: Vec<Vec<f64>> = (0..dimension)
            .map(|row| {
                (0..dimension)
                    .map(|column| (row * dimension + column) as f64 / 16384.0)
                    .collect()
            })
            .collect();

        let packed = super::as_half_row(&matrix);

        assert_eq!(packed.len(), 64);
        assert_eq!(packed[0].len(), 128);

        let recovered = super::as_double_row(&packed);
        assert_eq!(recovered, matrix);
    }

    use super::{combine_scalar_coefficients, split_large_coefficients};

    #[test]
    fn heaan_ring_switch_layout_matches_oracle() {
        let scalar_degree = 64_usize;
        let dimension = 4_usize;

        let parts: Vec<Vec<u64>> = (0..dimension)
            .map(|row| {
                (0..scalar_degree)
                    .map(|coefficient| ((row + 1) * 1000 + coefficient) as u64)
                    .collect()
            })
            .collect();

        let combined = combine_scalar_coefficients(&parts);

        assert_eq!(combined.len(), 256);

        // Exact prefix observed from HEaaN RingSwitchHelper::combine.
        assert_eq!(
            &combined[..16],
            &[
                1000, 2000, 3000, 4000, 1001, 2001, 3001, 4001, 1002, 2002, 3002, 4002, 1003, 2003,
                3003, 4003,
            ]
        );

        for coefficient in 0..scalar_degree {
            for row in 0..dimension {
                assert_eq!(
                    combined[coefficient * dimension + row],
                    parts[row][coefficient],
                    "HEaaN layout mismatch at coefficient={coefficient}, row={row}"
                );
            }
        }

        let recovered = split_large_coefficients(&combined, dimension);

        assert_eq!(recovered, parts);
    }

    #[test]
    fn combine_split_roundtrip_multiple_dimensions() {
        for dimension in [1_usize, 2, 4, 8, 16] {
            for scalar_degree in [1_usize, 8, 64, 128] {
                let parts: Vec<Vec<u64>> = (0..dimension)
                    .map(|row| {
                        (0..scalar_degree)
                            .map(|coefficient| 10_000 * row as u64 + coefficient as u64)
                            .collect()
                    })
                    .collect();

                let combined = combine_scalar_coefficients(&parts);

                let recovered = split_large_coefficients(&combined, dimension);

                assert_eq!(
                    recovered, parts,
                    "roundtrip mismatch for d={dimension}, Ns={scalar_degree}"
                );
            }
        }
    }
}

#[cfg(test)]
mod sinc_tests {
    use super::{bit_reverse_index, sinc_decode_batch, sinc_encode_batch};
    use num_complex::Complex64;

    fn deterministic_batch(
        batch_count: usize,
        rows: usize,
        cols: usize,
    ) -> Vec<Vec<Vec<Complex64>>> {
        (0..batch_count)
            .map(|batch| {
                (0..rows)
                    .map(|row| {
                        (0..cols)
                            .map(|col| {
                                let real =
                                    batch as f64 * 0.01 + row as f64 * 0.001 + col as f64 * 0.0001;

                                let imag = batch as f64 * -0.0003 + row as f64 * 0.00002
                                    - col as f64 * 0.00001;

                                Complex64::new(real, imag)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn max_batch_error(expected: &[Vec<Vec<Complex64>>], actual: &[Vec<Vec<Complex64>>]) -> f64 {
        let mut max_error = 0.0_f64;

        for batch in 0..expected.len() {
            for row in 0..expected[batch].len() {
                for col in 0..expected[batch][row].len() {
                    max_error =
                        max_error.max((expected[batch][row][col] - actual[batch][row][col]).norm());
                }
            }
        }

        max_error
    }

    use super::{
        combine_rns_polynomials, combine_rns_rlwe_ciphertexts, split_rns_polynomial,
        split_rns_rlwe_ciphertext,
    };
    use crate::grafting::RnsRlweCiphertext;
    use crate::ring::{Polynomial, RnsPolynomial};
    use crate::rlwe::RlweCiphertext;

    fn oracle_rns_parts() -> Vec<RnsPolynomial> {
        let moduli = vec![
            crate::ring::Modulus::new(12_289),
            crate::ring::Modulus::new(40_961),
        ];

        (0..4)
            .map(|row| {
                let coefficients: Vec<u128> =
                    (0..64).map(|j| ((row + 1) * 1000 + j) as u128).collect();

                RnsPolynomial::from_coefficients(moduli.clone(), &coefficients)
            })
            .collect()
    }

    #[test]
    fn rns_ring_switch_ntt_roundtrip_is_exact() {
        let parts = oracle_rns_parts();
        let refs: Vec<&RnsPolynomial> = parts.iter().collect();

        let combined = combine_rns_polynomials(&refs);

        let split = split_rns_polynomial(&combined, 4);

        let recovered_parts: Vec<RnsPolynomial> = split
            .iter()
            .map(|part| {
                let plan = crate::ring::RnsNttPlan::new(part.moduli().to_vec(), part.degree());

                let transformed = plan.forward(part);
                plan.inverse(&transformed)
            })
            .collect();

        assert_eq!(recovered_parts, parts);

        let recovered_refs: Vec<&RnsPolynomial> = recovered_parts.iter().collect();

        let recombined = combine_rns_polynomials(&recovered_refs);

        assert_eq!(recombined, combined);

        println!("RNS_RING_SWITCH_NTT_ROUNDTRIP=PASS");
    }

    #[test]
    fn rns_ntt_pointwise_product_matches_coefficient_oracle() {
        let parts = oracle_rns_parts();

        let lhs = &parts[0];
        let rhs = &parts[1];

        let plan = crate::ring::RnsNttPlan::new(lhs.moduli().to_vec(), lhs.degree());

        let lhs_ntt = plan.forward(lhs);
        let rhs_ntt = plan.forward(rhs);

        let actual = plan.inverse(&lhs_ntt.pointwise_mul(&rhs_ntt));

        // Independent coefficient-domain oracle, residue by residue.
        let expected = RnsPolynomial::from_residues(
            lhs.residues()
                .iter()
                .zip(rhs.residues())
                .map(|(lhs, rhs)| lhs.negacyclic_mul(rhs))
                .collect(),
        );

        assert_eq!(actual, expected);

        println!("RNS_NTT_POINTWISE_PRODUCT=PASS");
    }

    #[test]
    fn rns_ring_switch_matches_heaan_oracle_layout() {
        let parts = oracle_rns_parts();
        let refs: Vec<&RnsPolynomial> = parts.iter().collect();

        let combined = combine_rns_polynomials(&refs);

        assert_eq!(combined.degree(), 256);

        for residue in combined.residues() {
            let q = residue.modulus().value();

            for j in 0..64 {
                for row in 0..4 {
                    let expected = (((row + 1) * 1000 + j) as u64) % q;

                    assert_eq!(
                        residue.coefficient(j * 4 + row),
                        expected,
                        "HEaaN layout mismatch at coefficient {j}, row {row}, modulus {q}"
                    );
                }
            }
        }

        println!("RNS_HEAAN_LAYOUT=PASS");
    }

    #[test]
    fn rns_ring_switch_split_roundtrip_is_exact() {
        let parts = oracle_rns_parts();
        let refs: Vec<&RnsPolynomial> = parts.iter().collect();

        let combined = combine_rns_polynomials(&refs);
        let recovered = split_rns_polynomial(&combined, 4);

        assert_eq!(recovered, parts);

        println!("RNS_SPLIT_ROUNDTRIP=PASS");
    }

    #[test]
    fn rns_rlwe_ring_switch_split_roundtrip_is_exact() {
        let moduli = [
            crate::ring::Modulus::new(12_289),
            crate::ring::Modulus::new(40_961),
        ];

        let parts: Vec<RnsRlweCiphertext> = (0..4)
            .map(|row| {
                let limbs = moduli
                    .iter()
                    .copied()
                    .map(|modulus| {
                        let b = Polynomial::new(
                            modulus,
                            (0..64).map(|j| ((row + 1) * 1000 + j) as u64).collect(),
                        );

                        let a = Polynomial::new(
                            modulus,
                            (0..64).map(|j| ((row + 1) * 5000 + j) as u64).collect(),
                        );

                        RlweCiphertext::new(b, a)
                    })
                    .collect();

                RnsRlweCiphertext::from_limbs(limbs)
            })
            .collect();

        let refs: Vec<&RnsRlweCiphertext> = parts.iter().collect();

        let combined = combine_rns_rlwe_ciphertexts(&refs);

        assert_eq!(combined.degree(), 256);

        let recovered = split_rns_rlwe_ciphertext(&combined, 4);

        assert_eq!(recovered, parts);

        for limb_index in 0..combined.limbs().len() {
            let limb = combined.limb(limb_index);
            let q = limb.b().modulus().value();

            for j in 0..64 {
                for row in 0..4 {
                    assert_eq!(
                        limb.b().coefficient(j * 4 + row),
                        (((row + 1) * 1000 + j) as u64) % q
                    );

                    assert_eq!(
                        limb.a().coefficient(j * 4 + row),
                        (((row + 1) * 5000 + j) as u64) % q
                    );
                }
            }
        }

        println!("RNS_RLWE_B_COMPONENT=PASS");
        println!("RNS_RLWE_A_COMPONENT=PASS");
        println!("RNS_RLWE_SPLIT_ROUNDTRIP=PASS");
    }

    #[test]
    fn sinc_bit_reversal_matches_expected_order() {
        // 8 slots => three-bit reversal:
        //
        // 000 -> 000 = 0
        // 001 -> 100 = 4
        // 010 -> 010 = 2
        // 011 -> 110 = 6
        // 100 -> 001 = 1
        // 101 -> 101 = 5
        // 110 -> 011 = 3
        // 111 -> 111 = 7
        let observed: Vec<_> = (0..8).map(|i| bit_reverse_index(i, 3)).collect();

        assert_eq!(observed, vec![0, 4, 2, 6, 1, 5, 3, 7]);
    }

    #[test]
    fn sinc_structural_roundtrip_tiny_valid_case() {
        // HEaaN itself requires scalar ring degree >= 64.
        //
        // Ns = 64
        // slots = batch count = 32
        // d = 4
        // Nl = 256
        let scalar_degree = 64;
        let dimension = 4;
        let batch_count = scalar_degree / 2;

        let input = deterministic_batch(batch_count, dimension, dimension);

        let encoded = sinc_encode_batch(&input, scalar_degree);

        assert_eq!(encoded.scalar_degree(), 64);
        assert_eq!(encoded.dimension(), 4);
        assert_eq!(encoded.large_degree(), 256);
        assert_eq!(encoded.num_columns(), 4);

        let decoded = sinc_decode_batch(&encoded);

        let max_error = max_batch_error(&input, &decoded);

        println!("SINC_TINY_MAX_ERROR={max_error:.12e}");

        assert!(
            max_error < 1.0e-10,
            "tiny SinC structural roundtrip error {max_error:e}"
        );
    }

    #[test]
    fn sinc_structural_roundtrip_authors_d64_layout() {
        // Authors' Batch d=64 configuration:
        //
        // large degree  = 8192
        // scalar degree = 128
        // CKKS slots    = 64
        // batch count   = 64
        let scalar_degree = 128;
        let dimension = 64;
        let batch_count = 64;

        let input = deterministic_batch(batch_count, dimension, dimension);

        let encoded = sinc_encode_batch(&input, scalar_degree);

        assert_eq!(encoded.large_degree(), 8192);
        assert_eq!(encoded.num_columns(), 64);

        let decoded = sinc_decode_batch(&encoded);

        let max_error = max_batch_error(&input, &decoded);

        println!("SINC_D64_MAX_ERROR={max_error:.12e}");

        assert!(
            max_error < 1.0e-9,
            "authors d=64 SinC roundtrip error {max_error:e}"
        );
    }

    fn deterministic_real_batch(batch_count: usize, dimension: usize) -> Vec<Vec<Vec<f64>>> {
        (0..batch_count)
            .map(|batch| {
                (0..dimension)
                    .map(|row| {
                        (0..dimension)
                            .map(|col| {
                                batch as f64 * 0.01 + row as f64 * 0.001 + col as f64 * 0.0001
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn max_real_batch_error(expected: &[Vec<Vec<f64>>], actual: &[Vec<Vec<f64>>]) -> f64 {
        let mut max_error = 0.0_f64;

        assert_eq!(expected.len(), actual.len());

        for batch in 0..expected.len() {
            assert_eq!(expected[batch].len(), actual[batch].len());

            for row in 0..expected[batch].len() {
                assert_eq!(expected[batch][row].len(), actual[batch][row].len());

                for col in 0..expected[batch][row].len() {
                    max_error =
                        max_error.max((expected[batch][row][col] - actual[batch][row][col]).abs());
                }
            }
        }

        max_error
    }

    fn cpmm_sinc_roundtrip(dimension: usize, scalar_degree: usize) -> f64 {
        let batch_count = scalar_degree / 2;

        let input = deterministic_real_batch(batch_count, dimension);

        let half_rows: Vec<Vec<Vec<Complex64>>> = input
            .iter()
            .map(|matrix| super::as_half_row(matrix))
            .collect();

        assert_eq!(half_rows.len(), batch_count);
        assert_eq!(half_rows[0].len(), dimension / 2);
        assert_eq!(half_rows[0][0].len(), dimension);

        let encoded = sinc_encode_batch(&half_rows, scalar_degree);

        assert_eq!(encoded.scalar_degree(), scalar_degree);
        assert_eq!(encoded.dimension(), dimension / 2);
        assert_eq!(encoded.num_columns(), dimension);

        // Authors' CPMM doubles the scalar-ring degree while half-row
        // packing halves the logical row count. Their product remains
        // the full degree-8192 large ring.
        assert_eq!(encoded.large_degree(), 8192);

        let decoded_half = sinc_decode_batch(&encoded);

        assert_eq!(decoded_half.len(), batch_count);

        let recovered: Vec<Vec<Vec<f64>>> = decoded_half
            .iter()
            .map(|matrix| super::as_double_row(matrix))
            .collect();

        max_real_batch_error(&input, &recovered)
    }

    #[test]
    fn cpmm_sinc_structural_roundtrip_authors_d64_layout() {
        // Authors' Batch CPMM d=64:
        //
        // large degree       = 8192
        // real rows          = 64
        // half-row rows      = 32
        // scalar degree      = 256
        // scalar CKKS slots  = 128
        // real batch count   = 128
        //
        // 32 * 256 = 8192.
        let max_error = cpmm_sinc_roundtrip(64, 256);

        println!("CPMM_SINC_D64_MAX_ERROR={max_error:.12e}");

        assert!(
            max_error < 1.0e-9,
            "authors CPMM d=64 SinC roundtrip error {max_error:e}"
        );
    }

    #[test]
    fn cpmm_sinc_structural_roundtrip_authors_d128_layout() {
        // Authors' Batch CPMM d=128:
        //
        // large degree       = 8192
        // real rows          = 128
        // half-row rows      = 64
        // scalar degree      = 128
        // scalar CKKS slots  = 64
        // real batch count   = 64
        //
        // 64 * 128 = 8192.
        let max_error = cpmm_sinc_roundtrip(128, 128);

        println!("CPMM_SINC_D128_MAX_ERROR={max_error:.12e}");

        assert!(
            max_error < 1.0e-9,
            "authors CPMM d=128 SinC roundtrip error {max_error:e}"
        );
    }

    fn sd3b_moduli() -> Vec<crate::ring::Modulus> {
        // Authors' BatchMMTest uses INIT_LEVEL = 1.
        //
        // HEaaN indexes the prime chain by level, so the active
        // ciphertext/plaintext modulus for Batch CPMM is:
        //
        //     Q_1 = q_0 * q_1
        //
        // and the subsequent rescale removes q_1.
        vec![
            crate::ring::Modulus::new(68_712_923_137),
            crate::ring::Modulus::new(268_238_849),
        ]
    }

    fn sd3b_scale() -> f64 {
        2.0_f64.powf(27.993302092216055)
    }

    fn deterministic_cpmm_input(
        batch_count: usize,
        dimension: usize,
        salt: usize,
    ) -> Vec<Vec<Vec<f64>>> {
        (0..batch_count)
            .map(|batch| {
                (0..dimension)
                    .map(|row| {
                        (0..dimension)
                            .map(|col| {
                                let x = (batch * 17 + row * 13 + col * 7 + salt * 19) % 41;

                                (x as f64 - 20.0) / 20.0
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn clear_batch_mm(lhs: &[Vec<Vec<f64>>], rhs: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
        assert_eq!(lhs.len(), rhs.len());

        lhs.iter()
            .zip(rhs)
            .map(|(lhs, rhs)| {
                let dimension = lhs.len();

                (0..dimension)
                    .map(|row| {
                        (0..dimension)
                            .map(|col| {
                                (0..dimension)
                                    .map(|inner| lhs[row][inner] * rhs[inner][col])
                                    .sum()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn quantize_coefficients(
        coefficients: &[f64],
        basis: &crate::ring::ModulusBasis,
        scale: f64,
    ) -> crate::ring::RnsPolynomial {
        let residues = basis
            .moduli()
            .iter()
            .copied()
            .map(|modulus| {
                let q = i128::from(modulus.value());

                crate::ring::Polynomial::new(
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

        crate::ring::RnsPolynomial::from_residues(residues)
    }

    fn encode_packed_real_matrix(
        matrices: &[Vec<Vec<f64>>],
        scalar_degree: usize,
        basis: &crate::ring::ModulusBasis,
        scale: f64,
    ) -> Vec<Vec<crate::ring::RnsPolynomial>> {
        let batch_count = matrices.len();
        let dimension = matrices[0].len();

        assert_eq!(batch_count, scalar_degree / 2);

        let embedding = crate::ckks::CkksCanonicalEmbedding::new(scalar_degree);
        let log_slots = batch_count.trailing_zeros();

        (0..dimension)
            .map(|row| {
                (0..dimension)
                    .map(|col| {
                        let mut slots = vec![num_complex::Complex64::new(0.0, 0.0); batch_count];

                        for (slot, value) in slots.iter_mut().enumerate() {
                            let source_batch = super::bit_reverse_index(slot, log_slots);

                            *value =
                                num_complex::Complex64::new(matrices[source_batch][row][col], 0.0);
                        }

                        let coefficients = embedding.slots_to_coefficients(&slots);

                        quantize_coefficients(&coefficients, basis, scale)
                    })
                    .collect()
            })
            .collect()
    }

    fn raw_cp_product(
        ciphertext: &crate::grafting::RnsRlweCiphertext,
        plaintext: &crate::ring::RnsPolynomial,
        plan: &crate::ring::RnsNttPlan,
    ) -> crate::grafting::RnsRlweCiphertext {
        let limbs = ciphertext
            .limbs()
            .iter()
            .enumerate()
            .map(|(index, limb)| {
                let limb_plan = plan.plan(index);
                let plain = plaintext.residue(index);

                crate::rlwe::RlweCiphertext::new(
                    limb_plan.negacyclic_mul(limb.b(), plain),
                    limb_plan.negacyclic_mul(limb.a(), plain),
                )
            })
            .collect();

        crate::grafting::RnsRlweCiphertext::from_limbs(limbs)
    }

    fn raw_rlwe_add(
        lhs: &crate::grafting::RnsRlweCiphertext,
        rhs: &crate::grafting::RnsRlweCiphertext,
    ) -> crate::grafting::RnsRlweCiphertext {
        assert_eq!(lhs.basis(), rhs.basis());
        assert_eq!(lhs.degree(), rhs.degree());

        let limbs = lhs
            .limbs()
            .iter()
            .zip(rhs.limbs())
            .map(|(lhs, rhs)| {
                crate::rlwe::RlweCiphertext::new(lhs.b().add(rhs.b()), lhs.a().add(rhs.a()))
            })
            .collect();

        crate::grafting::RnsRlweCiphertext::from_limbs(limbs)
    }

    fn batch_rlwe_sub(
        lhs: &crate::grafting::RnsRlweCiphertext,
        rhs: &crate::grafting::RnsRlweCiphertext,
    ) -> crate::grafting::RnsRlweCiphertext {
        assert_eq!(lhs.basis(), rhs.basis());
        assert_eq!(lhs.degree(), rhs.degree());

        crate::grafting::RnsRlweCiphertext::from_limbs(
            lhs.limbs()
                .iter()
                .zip(rhs.limbs())
                .map(|(lhs, rhs)| {
                    crate::rlwe::RlweCiphertext::new(lhs.b().sub(rhs.b()), lhs.a().sub(rhs.a()))
                })
                .collect(),
        )
    }

    fn batch_rlwe_scalar_mul(
        ciphertext: &crate::grafting::RnsRlweCiphertext,
        scalars: &[u64],
    ) -> crate::grafting::RnsRlweCiphertext {
        assert_eq!(
            ciphertext.limbs().len(),
            scalars.len(),
            "Batch RNS scalar multiplication requires one scalar per limb"
        );

        crate::grafting::RnsRlweCiphertext::from_limbs(
            ciphertext
                .limbs()
                .iter()
                .zip(scalars)
                .map(|(limb, &scalar)| {
                    crate::rlwe::RlweCiphertext::new(
                        limb.b().scalar_mul(scalar),
                        limb.a().scalar_mul(scalar),
                    )
                })
                .collect(),
        )
    }

    fn batch_polynomial_mul_monomial_signed(
        polynomial: &crate::ring::Polynomial,
        exponent: i64,
    ) -> crate::ring::Polynomial {
        let degree = polynomial.degree();
        let period = 2_i128 * degree as i128;
        let normalized = (i128::from(exponent)).rem_euclid(period) as usize;
        let shift = normalized % degree;
        let negative = normalized >= degree;

        let modulus = polynomial.modulus();
        let mut monomial = vec![0_u64; degree];
        monomial[shift] = if negative { modulus.value() - 1 } else { 1 };

        polynomial.negacyclic_mul(&crate::ring::Polynomial::new(modulus, monomial))
    }

    fn batch_rlwe_mul_monomial_signed(
        ciphertext: &crate::grafting::RnsRlweCiphertext,
        exponent: i64,
    ) -> crate::grafting::RnsRlweCiphertext {
        crate::grafting::RnsRlweCiphertext::from_limbs(
            ciphertext
                .limbs()
                .iter()
                .map(|limb| {
                    crate::rlwe::RlweCiphertext::new(
                        batch_polynomial_mul_monomial_signed(limb.b(), exponent),
                        batch_polynomial_mul_monomial_signed(limb.a(), exponent),
                    )
                })
                .collect(),
        )
    }

    fn batch_large_dft(
        input: &[crate::grafting::RnsRlweCiphertext],
        multiplier: i64,
    ) -> Vec<crate::grafting::RnsRlweCiphertext> {
        let dimension = input.len();

        assert!(
            dimension.is_power_of_two(),
            "Batch large DFT dimension must be a power of two"
        );
        assert!(!input.is_empty(), "Batch large DFT requires input");

        if dimension == 1 {
            return vec![input[0].clone()];
        }

        let half = dimension / 2;

        let even: Vec<_> = (0..half).map(|i| input[2 * i].clone()).collect();
        let odd: Vec<_> = (0..half).map(|i| input[2 * i + 1].clone()).collect();

        let even = batch_large_dft(&even, multiplier * 2);
        let odd = batch_large_dft(&odd, multiplier * 2);

        let mut output = Vec::with_capacity(dimension);
        output.resize_with(dimension, || input[0].clone());

        for k in 0..half {
            let twiddled = batch_rlwe_mul_monomial_signed(&odd[k], multiplier * k as i64);

            output[k] = raw_rlwe_add(&even[k], &twiddled);
            output[k + half] = batch_rlwe_sub(&even[k], &twiddled);
        }

        output
    }

    fn batch_large_crt(
        ciphertexts: &[crate::grafting::RnsRlweCiphertext],
        scalar_degree: usize,
    ) -> Vec<crate::grafting::RnsRlweCiphertext> {
        let dimension = ciphertexts.len();

        assert!(!ciphertexts.is_empty());
        assert!(dimension.is_power_of_two());

        let large_degree = ciphertexts[0].degree();

        assert_eq!(
            scalar_degree * dimension,
            large_degree,
            "Batch CRT geometry must satisfy Ns * d = N"
        );

        let inverse_dimension: Vec<u64> = ciphertexts[0]
            .basis()
            .moduli()
            .iter()
            .map(|modulus| modulus.inverse_prime(dimension as u64))
            .collect();

        let normalized: Vec<_> = ciphertexts
            .iter()
            .enumerate()
            .map(|(column, ciphertext)| {
                let scaled = batch_rlwe_scalar_mul(ciphertext, &inverse_dimension);
                batch_rlwe_mul_monomial_signed(&scaled, column as i64)
            })
            .collect();

        batch_large_dft(&normalized, (2 * scalar_degree) as i64)
    }

    fn batch_large_inv_crt(
        ciphertexts: &[crate::grafting::RnsRlweCiphertext],
        scalar_degree: usize,
    ) -> Vec<crate::grafting::RnsRlweCiphertext> {
        let dimension = ciphertexts.len();

        assert!(!ciphertexts.is_empty());
        assert!(dimension.is_power_of_two());

        let large_degree = ciphertexts[0].degree();

        assert_eq!(
            scalar_degree * dimension,
            large_degree,
            "Batch inverse CRT geometry must satisfy Ns * d = N"
        );

        batch_large_dft(ciphertexts, -((2 * scalar_degree) as i64))
            .into_iter()
            .enumerate()
            .map(|(column, ciphertext)| {
                batch_rlwe_mul_monomial_signed(&ciphertext, -(column as i64))
            })
            .collect()
    }

    fn batch_mod_inverse_odd(value: usize, modulus: usize) -> usize {
        assert!(value % 2 == 1);
        assert!(modulus.is_power_of_two());

        let mut t: i128 = 0;
        let mut new_t: i128 = 1;
        let mut r = modulus as i128;
        let mut new_r = value as i128;

        while new_r != 0 {
            let quotient = r / new_r;

            let next_t = t - quotient * new_t;
            t = new_t;
            new_t = next_t;

            let next_r = r - quotient * new_r;
            r = new_r;
            new_r = next_r;
        }

        assert_eq!(r, 1, "Batch Galois exponent must be invertible modulo 2N");

        t.rem_euclid(modulus as i128) as usize
    }

    fn batch_galois_key_for_exponent(
        keys: &[crate::ckks::RnsGaloisKey],
        exponent: usize,
    ) -> &crate::ckks::RnsGaloisKey {
        keys.iter()
            .find(|key| key.exponent() == exponent)
            .unwrap_or_else(|| panic!("missing Batch C-MT RNS Galois key for exponent {exponent}"))
    }

    fn batch_scrambled_auto(
        ciphertexts: &[crate::grafting::RnsRlweCiphertext],
        scalar_degree: usize,
        galois_keys: &[crate::ckks::RnsGaloisKey],
    ) -> Vec<crate::grafting::RnsRlweCiphertext> {
        let dimension = ciphertexts.len();

        assert!(!ciphertexts.is_empty());

        let large_degree = ciphertexts[0].degree();
        let two_n = 2 * large_degree;

        assert_eq!(
            scalar_degree * dimension,
            large_degree,
            "Batch scrambled automorphism geometry must satisfy Ns * d = N"
        );

        (0..dimension)
            .map(|index| {
                let exponent = 2 * scalar_degree * index + 1;
                let inverse = batch_mod_inverse_odd(exponent, two_n);

                assert_eq!(
                    (inverse - 1) % (2 * scalar_degree),
                    0,
                    "inverse Batch automorphism exponent must remain in subgroup"
                );

                let inverse_index = (inverse - 1) / (2 * scalar_degree);

                if exponent == 1 {
                    ciphertexts[inverse_index].clone()
                } else {
                    let key = batch_galois_key_for_exponent(galois_keys, exponent);
                    crate::ckks::apply_rns_galois_automorphism(&ciphertexts[inverse_index], key)
                }
            })
            .collect()
    }

    fn batch_ciphertext_transpose(
        ciphertexts: &[crate::grafting::RnsRlweCiphertext],
        scalar_degree: usize,
        galois_keys: &[crate::ckks::RnsGaloisKey],
    ) -> Vec<crate::grafting::RnsRlweCiphertext> {
        let crt = batch_large_crt(ciphertexts, scalar_degree);
        let transformed = batch_scrambled_auto(&crt, scalar_degree, galois_keys);
        batch_large_inv_crt(&transformed, scalar_degree)
    }

    fn raw_cc_product(
        lhs: &crate::grafting::RnsRlweCiphertext,
        rhs: &crate::grafting::RnsRlweCiphertext,
        plan: &crate::ring::RnsNttPlan,
    ) -> crate::grafting::RnsQuadraticCiphertext {
        crate::grafting::rns_tensor_with_ntt(lhs, rhs, plan)
    }

    fn batch_ccmm_rns_component(
        ciphertext: &crate::grafting::RnsRlweCiphertext,
        b_component: bool,
    ) -> crate::ring::RnsPolynomial {
        crate::ring::RnsPolynomial::from_residues(
            ciphertext
                .limbs()
                .iter()
                .map(|limb| {
                    if b_component {
                        limb.b().clone()
                    } else {
                        limb.a().clone()
                    }
                })
                .collect(),
        )
    }

    /// Exact Rust analogue of MatrixCutter::slice for rank-1 RLWE.
    ///
    /// The authors use rankedRow(num_row, rank, row):
    ///
    ///     rows [0, d)   = b-component scalar-ring slices
    ///     rows [d, 2d)  = a-component scalar-ring slices
    ///
    /// yielding a 2d x num_columns polynomial matrix.
    fn batch_ccmm_slice_ranked(
        ciphertexts: &[crate::grafting::RnsRlweCiphertext],
        dimension: usize,
    ) -> Vec<Vec<crate::ring::RnsPolynomial>> {
        assert!(!ciphertexts.is_empty());
        assert!(dimension > 0);

        let sliced: Vec<_> = ciphertexts
            .iter()
            .map(|ciphertext| {
                assert_eq!(
                    ciphertext.degree() % dimension,
                    0,
                    "Batch CCMM large degree must be divisible by d"
                );

                let b = batch_ccmm_rns_component(ciphertext, true);
                let a = batch_ccmm_rns_component(ciphertext, false);

                (
                    super::split_rns_polynomial(&b, dimension),
                    super::split_rns_polynomial(&a, dimension),
                )
            })
            .collect();

        (0..2 * dimension)
            .map(|ranked_row| {
                sliced
                    .iter()
                    .map(|(b_rows, a_rows)| {
                        if ranked_row < dimension {
                            b_rows[ranked_row].clone()
                        } else {
                            a_rows[ranked_row - dimension].clone()
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn batch_ccmm_transpose_matrix<T: Clone>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
        assert!(!matrix.is_empty());
        assert!(!matrix[0].is_empty());

        let columns = matrix[0].len();

        assert!(
            matrix.iter().all(|row| row.len() == columns),
            "Batch CCMM matrix must be rectangular"
        );

        (0..columns)
            .map(|column| matrix.iter().map(|row| row[column].clone()).collect())
            .collect()
    }

    fn batch_ccmm_rns_product(
        lhs: &crate::ring::RnsPolynomial,
        rhs: &crate::ring::RnsPolynomial,
        plan: &crate::ring::RnsNttPlan,
    ) -> crate::ring::RnsPolynomial {
        assert_eq!(lhs.basis(), rhs.basis());
        assert_eq!(lhs.degree(), rhs.degree());
        assert_eq!(plan.degree(), lhs.degree());
        assert_eq!(plan.moduli(), lhs.basis().moduli());

        let lhs_ntt = plan.forward(lhs);
        let rhs_ntt = plan.forward(rhs);

        plan.inverse(&lhs_ntt.pointwise_mul(&rhs_ntt))
    }

    fn batch_ccmm_rns_add(
        lhs: &crate::ring::RnsPolynomial,
        rhs: &crate::ring::RnsPolynomial,
    ) -> crate::ring::RnsPolynomial {
        assert_eq!(lhs.basis(), rhs.basis());
        assert_eq!(lhs.degree(), rhs.degree());

        crate::ring::RnsPolynomial::from_residues(
            lhs.residues()
                .iter()
                .zip(rhs.residues())
                .map(|(lhs, rhs)| lhs.add(rhs))
                .collect(),
        )
    }

    /// Authors' modular matrix multiplication:
    ///
    ///     (2d x d) * (d x 2d) -> (2d x 2d)
    ///
    /// Entries are scalar-ring RNS polynomials, not ciphertexts.
    fn batch_ccmm_modular_product(
        lhs: &[Vec<crate::ring::RnsPolynomial>],
        rhs_transposed: &[Vec<crate::ring::RnsPolynomial>],
        plan: &crate::ring::RnsNttPlan,
    ) -> Vec<Vec<crate::ring::RnsPolynomial>> {
        assert!(!lhs.is_empty(), "Batch CCMM lhs must not be empty");
        assert!(
            !rhs_transposed.is_empty(),
            "Batch CCMM rhs must not be empty"
        );

        let inner = lhs[0].len();

        assert!(inner > 0);
        assert!(lhs.iter().all(|row| row.len() == inner));

        assert_eq!(
            rhs_transposed.len(),
            inner,
            "Batch CCMM modular inner dimensions must match"
        );

        let output_columns = rhs_transposed[0].len();

        assert!(output_columns > 0);
        assert!(rhs_transposed.iter().all(|row| row.len() == output_columns));

        (0..lhs.len())
            .map(|row| {
                (0..output_columns)
                    .map(|column| {
                        let mut accumulator =
                            batch_ccmm_rns_product(&lhs[row][0], &rhs_transposed[0][column], plan);

                        for (lhs_entry, rhs_row) in
                            lhs[row].iter().zip(rhs_transposed.iter()).skip(1)
                        {
                            let product = batch_ccmm_rns_product(lhs_entry, &rhs_row[column], plan);

                            accumulator = batch_ccmm_rns_add(&accumulator, &product);
                        }

                        accumulator
                    })
                    .collect()
            })
            .collect()
    }

    fn batch_ccmm_split_former_latter<T: Clone>(
        product: &[Vec<T>],
        dimension: usize,
    ) -> (Vec<Vec<T>>, Vec<Vec<T>>) {
        assert_eq!(
            product.len(),
            2 * dimension,
            "authors Batch CCMM product must have 2d rows"
        );

        assert!(
            product.iter().all(|row| row.len() == 2 * dimension),
            "authors Batch CCMM product must have 2d columns"
        );

        (product[..dimension].to_vec(), product[dimension..].to_vec())
    }

    /// Exact inverse of batch_ccmm_slice_ranked for one 2d x d layer matrix.
    ///
    /// rows [0,d)  -> b polynomial
    /// rows [d,2d) -> a polynomial
    fn batch_ccmm_stack_ranked(
        layers: &[Vec<crate::ring::RnsPolynomial>],
        dimension: usize,
    ) -> Vec<crate::grafting::RnsRlweCiphertext> {
        assert_eq!(layers.len(), 2 * dimension);
        assert!(
            layers.iter().all(|row| row.len() == dimension),
            "Batch CCMM stacked layer matrix must be 2d x d"
        );

        (0..dimension)
            .map(|column| {
                let b_refs: Vec<&crate::ring::RnsPolynomial> =
                    (0..dimension).map(|row| &layers[row][column]).collect();

                let a_refs: Vec<&crate::ring::RnsPolynomial> = (0..dimension)
                    .map(|row| &layers[row + dimension][column])
                    .collect();

                let b = super::combine_rns_polynomials(&b_refs);
                let a = super::combine_rns_polynomials(&a_refs);

                assert_eq!(b.basis(), a.basis());
                assert_eq!(b.degree(), a.degree());

                crate::grafting::RnsRlweCiphertext::from_limbs(
                    b.residues()
                        .iter()
                        .zip(a.residues())
                        .map(|(b, a)| crate::rlwe::RlweCiphertext::new(b.clone(), a.clone()))
                        .collect(),
                )
            })
            .collect()
    }

    /// HEaaN Manipulator::addCtSkCt semantic mapping:
    ///
    ///     former + s * latter
    ///
    /// If
    ///
    ///     former = b_f + a_f s
    ///     latter = b_l + a_l s
    ///
    /// then
    ///
    ///     c0 = b_f
    ///     c1 = a_f + b_l
    ///     c2 = a_l
    fn batch_ccmm_add_ct_sk_ct(
        former: &crate::grafting::RnsRlweCiphertext,
        latter: &crate::grafting::RnsRlweCiphertext,
    ) -> crate::grafting::RnsQuadraticCiphertext {
        assert_eq!(former.basis(), latter.basis());
        assert_eq!(former.degree(), latter.degree());

        let former_b = batch_ccmm_rns_component(former, true);
        let former_a = batch_ccmm_rns_component(former, false);
        let latter_b = batch_ccmm_rns_component(latter, true);
        let latter_a = batch_ccmm_rns_component(latter, false);

        crate::grafting::RnsQuadraticCiphertext::from_rns_polynomials(
            former_b,
            batch_ccmm_rns_add(&former_a, &latter_b),
            latter_a,
        )
    }

    fn raw_quadratic_add(
        lhs: &crate::grafting::RnsQuadraticCiphertext,
        rhs: &crate::grafting::RnsQuadraticCiphertext,
    ) -> crate::grafting::RnsQuadraticCiphertext {
        assert_eq!(lhs.c0().basis(), rhs.c0().basis());
        assert_eq!(lhs.c0().degree(), rhs.c0().degree());

        fn add_rns(
            lhs: &crate::ring::RnsPolynomial,
            rhs: &crate::ring::RnsPolynomial,
        ) -> crate::ring::RnsPolynomial {
            assert_eq!(lhs.basis(), rhs.basis());
            assert_eq!(lhs.degree(), rhs.degree());

            crate::ring::RnsPolynomial::from_residues(
                lhs.residues()
                    .iter()
                    .zip(rhs.residues())
                    .map(|(lhs, rhs)| lhs.add(rhs))
                    .collect(),
            )
        }

        crate::grafting::RnsQuadraticCiphertext::from_rns_polynomials(
            add_rns(lhs.c0(), rhs.c0()),
            add_rns(lhs.c1(), rhs.c1()),
            add_rns(lhs.c2(), rhs.c2()),
        )
    }

    fn decode_large_columns(
        ciphertexts: &[crate::ckks::RnsCkksCiphertext],
        secret: &[i8],
        scalar_degree: usize,
        half_rows: usize,
    ) -> super::SinCBatchPlaintext {
        use num_traits::ToPrimitive;

        let mut columns = Vec::with_capacity(ciphertexts.len());

        for ciphertext in ciphertexts {
            let plan = crate::ring::RnsNttPlan::new(
                ciphertext.basis().moduli().to_vec(),
                ciphertext.rlwe().degree(),
            );

            let plaintext =
                crate::grafting::decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);

            let modulus = crate::ring::composite_modulus_big(plaintext.basis());

            let coefficients = crate::ring::reconstruct_coefficients_big(&plaintext)
                .iter()
                .map(|value| {
                    crate::ring::centered_representative_big(value, &modulus)
                        .to_f64()
                        .expect("decoded CKKS coefficient must fit f64")
                        / ciphertext.scale()
                })
                .collect();

            columns.push(coefficients);
        }

        super::SinCBatchPlaintext {
            scalar_degree,
            dimension: half_rows,
            columns,
        }
    }

    fn cpmm_batch_capability(dimension: usize, scalar_degree: usize) -> f64 {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        const LARGE_DEGREE: usize = 8192;

        let batch_count = scalar_degree / 2;
        let half_rows = dimension / 2;
        let scale = sd3b_scale();

        assert_eq!(half_rows * scalar_degree, LARGE_DEGREE);

        let lhs_clear = deterministic_cpmm_input(batch_count, dimension, 1);
        let rhs_clear = deterministic_cpmm_input(batch_count, dimension, 2);

        let expected = clear_batch_mm(&lhs_clear, &rhs_clear);

        let lhs_half: Vec<Vec<Vec<num_complex::Complex64>>> = lhs_clear
            .iter()
            .map(|matrix| super::as_half_row(matrix))
            .collect();

        let lhs_sinc = super::sinc_encode_batch(&lhs_half, scalar_degree);

        assert_eq!(lhs_sinc.large_degree(), LARGE_DEGREE);
        assert_eq!(lhs_sinc.num_columns(), dimension);

        let basis = crate::ring::ModulusBasis::new(sd3b_moduli());

        let chain = crate::ring::ModulusChain::from_top_basis(basis.clone());

        let large_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), LARGE_DEGREE);

        let scalar_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), scalar_degree);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4241_5443_485F_4350 ^ dimension as u64);

        let mut secret: Vec<i8> = (0..LARGE_DEGREE)
            .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
            .collect();

        if secret.iter().all(|&x| x == 0) {
            secret[0] = 1;
        }

        let lhs_ciphertexts: Vec<crate::ckks::RnsCkksCiphertext> = lhs_sinc
            .columns()
            .iter()
            .enumerate()
            .map(|(column, coefficients)| {
                let plaintext = quantize_coefficients(coefficients, &basis, scale);

                let mut rng = ChaCha20Rng::seed_from_u64(0x4350_4D4D_0000_0000 ^ column as u64);

                let rlwe = crate::grafting::encrypt_rns_raw_with_distribution_ntt_rng(
                    &plaintext,
                    2,
                    crate::rlwe::ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                    &secret,
                    &large_plan,
                    &mut rng,
                );

                crate::ckks::RnsCkksCiphertext::new(
                    rlwe,
                    crate::ckks::CkksChainState::top(&chain, scale),
                    &chain,
                )
            })
            .collect();

        let rhs_plain = encode_packed_real_matrix(&rhs_clear, scalar_degree, &basis, scale);

        // Split each encrypted large-ring column into the authors'
        // scalar-ring row representation.
        let lhs_split: Vec<Vec<crate::grafting::RnsRlweCiphertext>> = lhs_ciphertexts
            .iter()
            .map(|ciphertext| super::split_rns_rlwe_ciphertext(ciphertext.rlwe(), half_rows))
            .collect();

        // Compute the scalar-ring h x d ciphertext/plaintext
        // matrix product.
        let mut output_rows: Vec<Vec<crate::grafting::RnsRlweCiphertext>> =
            Vec::with_capacity(half_rows);

        for row in 0..half_rows {
            let mut output_row = Vec::with_capacity(dimension);

            for col in 0..dimension {
                let mut accumulator =
                    raw_cp_product(&lhs_split[0][row], &rhs_plain[0][col], &scalar_plan);

                for inner in 1..dimension {
                    let product = raw_cp_product(
                        &lhs_split[inner][row],
                        &rhs_plain[inner][col],
                        &scalar_plan,
                    );

                    accumulator = raw_rlwe_add(&accumulator, &product);
                }

                output_row.push(accumulator);
            }

            output_rows.push(output_row);
        }

        // Recombine each output column into one degree-8192
        // ciphertext, then apply the single CKKS rescale.
        let mut output_ciphertexts = Vec::with_capacity(dimension);

        for col in 0..dimension {
            let refs: Vec<&crate::grafting::RnsRlweCiphertext> =
                (0..half_rows).map(|row| &output_rows[row][col]).collect();

            let combined = super::combine_rns_rlwe_ciphertexts(&refs);

            assert_eq!(combined.degree(), LARGE_DEGREE);

            let product = crate::ckks::RnsCkksCiphertext::new(
                combined,
                crate::ckks::CkksChainState::top(&chain, scale * scale),
                &chain,
            );

            output_ciphertexts.push(crate::ckks::rescale_rns_ckks_to_next(&product, &chain));
        }

        assert!(output_ciphertexts
            .iter()
            .all(|ciphertext| ciphertext.level() == 1));

        let decoded = decode_large_columns(&output_ciphertexts, &secret, scalar_degree, half_rows);

        let half_result = super::sinc_decode_batch(&decoded);

        let actual: Vec<Vec<Vec<f64>>> = half_result
            .iter()
            .map(|matrix| super::as_double_row(matrix))
            .collect();

        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;

        for batch in 0..batch_count {
            for row in 0..dimension {
                for col in 0..dimension {
                    let error = actual[batch][row][col] - expected[batch][row][col];

                    squared_error += error * error;
                    squared_reference += expected[batch][row][col] * expected[batch][row][col];
                }
            }
        }

        (squared_error / squared_reference).sqrt()
    }

    #[test]
    fn batch_cmt_small_encrypted_transpose_is_exact() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        const LARGE_DEGREE: usize = 32;
        const DIMENSION: usize = 4;
        const SCALAR_DEGREE: usize = 8;

        assert_eq!(DIMENSION * SCALAR_DEGREE, LARGE_DEGREE);

        let moduli = vec![
            crate::ring::Modulus::new(193),
            crate::ring::Modulus::new(257),
        ];

        let basis = crate::ring::ModulusBasis::new(moduli.clone());
        let plan = crate::ring::RnsNttPlan::new(moduli, LARGE_DEGREE);

        let secret = [
            1_i8, 0, -1, 1, 0, 1, -1, 0, 1, -1, 0, 0, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, 1, -1, 0, 0,
            1, -1, 0, 1, 0, -1,
        ];

        // Clear matrix:
        //
        //   11 12 13 14
        //   21 22 23 24
        //   31 32 33 34
        //   41 42 43 44
        //
        // In the combined large polynomial, scalar row r occupies
        // coefficients r, r+d, r+2d, ...
        //
        // For this structural test we place the row/column value in
        // coefficient zero of each scalar component, i.e. large
        // coefficient index r.
        let clear: Vec<Vec<u128>> = (0..DIMENSION)
            .map(|row| {
                (0..DIMENSION)
                    .map(|column| ((row + 1) * 10 + (column + 1)) as u128)
                    .collect()
            })
            .collect();

        let plaintext_columns: Vec<crate::ring::RnsPolynomial> = (0..DIMENSION)
            .map(|column| {
                let mut coefficients = vec![0_u128; LARGE_DEGREE];

                for row in 0..DIMENSION {
                    coefficients[row] = clear[row][column];
                }

                crate::ring::RnsPolynomial::from_coefficients(
                    basis.moduli().to_vec(),
                    &coefficients,
                )
            })
            .collect();

        let ciphertexts: Vec<crate::grafting::RnsRlweCiphertext> = plaintext_columns
            .iter()
            .enumerate()
            .map(|(column, plaintext)| {
                let mut rng = ChaCha20Rng::seed_from_u64(0xB520_0000 ^ column as u64);

                crate::grafting::encrypt_rns_raw_with_distribution_ntt_rng(
                    plaintext,
                    2,
                    crate::rlwe::ErrorDistribution::BoundedUniform { bound: 0 },
                    &secret,
                    &plan,
                    &mut rng,
                )
            })
            .collect();

        // Authors' direct scrambledAuto exponents:
        //
        //     h_i = 2 * Ns * i + 1
        //
        // h_0 = 1 is identity and needs no Galois key.
        let layout = crate::grafting::RnsGadgetLayout::new(basis.clone(), vec![1, 1]);

        let galois_keys: Vec<crate::ckks::RnsGaloisKey> = (1..DIMENSION)
            .map(|index| 2 * SCALAR_DEGREE * index + 1)
            .map(|exponent| {
                let mut rng = ChaCha20Rng::seed_from_u64(0xB521_0000 ^ exponent as u64);

                crate::ckks::RnsGaloisKey::generate_with_rng(
                    LARGE_DEGREE,
                    2,
                    0,
                    &secret,
                    exponent,
                    layout.clone(),
                    &mut rng,
                )
            })
            .collect();

        let transposed = batch_ciphertext_transpose(&ciphertexts, SCALAR_DEGREE, &galois_keys);

        assert_eq!(transposed.len(), DIMENSION);

        // Decrypt and split each resulting large-ring column back
        // into the four scalar-ring rows.
        for output_column in 0..DIMENSION {
            let decrypted = crate::grafting::decrypt_rns_raw_with_ntt(
                &transposed[output_column],
                &secret,
                &plan,
            );

            let rows = super::split_rns_polynomial(&decrypted, DIMENSION);

            assert_eq!(rows.len(), DIMENSION);

            for (output_row, row) in rows.iter().enumerate() {
                let observed = row.residue(0).coefficient(0) as u128;

                let expected = clear[output_column][output_row];

                assert_eq!(
                    observed, expected,
                    "Batch C-MT mismatch at ({output_row},{output_column})"
                );
            }
        }

        println!("BATCH_CMT_SMALL_EXACT_TRANSPOSE=PASS");
    }

    #[test]
    fn batch_large_crt_inverse_roundtrip_is_exact() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let large_degree = 32;
        let dimension = 4;
        let scalar_degree = large_degree / dimension;

        let moduli = vec![
            crate::ring::Modulus::new(193),
            crate::ring::Modulus::new(257),
        ];
        let basis = crate::ring::ModulusBasis::new(moduli.clone());
        let plan = crate::ring::RnsNttPlan::new(moduli, large_degree);
        let secret = [
            1_i8, 0, -1, 1, 0, 1, -1, 0, 1, -1, 0, 0, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, 1, -1, 0, 0,
            1, -1, 0, 1, 0, -1,
        ];

        let ciphertexts: Vec<_> = (0..dimension)
            .map(|column| {
                let values: Vec<u128> = (0..large_degree)
                    .map(|coefficient| ((column * 17 + coefficient * 7 + 3) % 23) as u128)
                    .collect();

                let plaintext =
                    crate::ring::RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &values);

                let mut rng = ChaCha20Rng::seed_from_u64(0xB520_0000 ^ column as u64);

                crate::grafting::encrypt_rns_raw_with_distribution_ntt_rng(
                    &plaintext,
                    2,
                    crate::rlwe::ErrorDistribution::BoundedUniform { bound: 0 },
                    &secret,
                    &plan,
                    &mut rng,
                )
            })
            .collect();

        let transformed = batch_large_crt(&ciphertexts, scalar_degree);
        let recovered = batch_large_inv_crt(&transformed, scalar_degree);

        assert_eq!(
            recovered, ciphertexts,
            "Batch largeCRT/largeInvCRT must be an exact ciphertext-domain roundtrip"
        );

        println!("BATCH_LARGE_CRT_INVERSE_ROUNDTRIP=PASS");
    }

    #[test]
    fn batch_ccmm_source_faithful_small_end_to_end() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        const DIMENSION: usize = 2;
        const SCALAR_DEGREE: usize = 8;
        const LARGE_DEGREE: usize = DIMENSION * SCALAR_DEGREE;

        let moduli = vec![
            crate::ring::Modulus::new(12_289),
            crate::ring::Modulus::new(40_961),
        ];

        let basis = crate::ring::ModulusBasis::new(moduli.clone());

        let large_plan = crate::ring::RnsNttPlan::new(moduli.clone(), LARGE_DEGREE);

        let scalar_plan = crate::ring::RnsNttPlan::new(moduli, SCALAR_DEGREE);

        let secret: Vec<i8> = (0..LARGE_DEGREE)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 1,
            })
            .collect();

        let lhs_clear = [[1_u128, 2], [3, 4]];
        let rhs_clear = [[5_u128, 6], [7, 8]];

        let expected = [[19_u128, 22], [43, 50]];

        fn encode_columns(
            clear: &[[u128; DIMENSION]; DIMENSION],
            basis: &crate::ring::ModulusBasis,
        ) -> Vec<crate::ring::RnsPolynomial> {
            (0..DIMENSION)
                .map(|column| {
                    let mut coefficients = vec![0_u128; LARGE_DEGREE];

                    // Constant term of scalar row r lives at
                    // large coefficient r in RingSwitchHelper layout.
                    for row in 0..DIMENSION {
                        coefficients[row] = clear[row][column];
                    }

                    crate::ring::RnsPolynomial::from_coefficients(
                        basis.moduli().to_vec(),
                        &coefficients,
                    )
                })
                .collect()
        }

        fn encrypt_columns(
            plaintexts: &[crate::ring::RnsPolynomial],
            secret: &[i8],
            plan: &crate::ring::RnsNttPlan,
            seed: u64,
        ) -> Vec<crate::grafting::RnsRlweCiphertext> {
            plaintexts
                .iter()
                .enumerate()
                .map(|(column, plaintext)| {
                    let mut rng = ChaCha20Rng::seed_from_u64(seed ^ column as u64);

                    crate::grafting::encrypt_rns_raw_with_distribution_ntt_rng(
                        plaintext,
                        2,
                        crate::rlwe::ErrorDistribution::BoundedUniform { bound: 0 },
                        secret,
                        plan,
                        &mut rng,
                    )
                })
                .collect()
        }

        let lhs_plain = encode_columns(&lhs_clear, &basis);
        let rhs_plain = encode_columns(&rhs_clear, &basis);

        let lhs = encrypt_columns(&lhs_plain, &secret, &large_plan, 0xB540_0000);

        let rhs = encrypt_columns(&rhs_plain, &secret, &large_plan, 0xB541_0000);

        // C-MT evaluation keys:
        // h_i = 2 * Ns * i + 1.
        let galois_layout = crate::grafting::RnsGadgetLayout::new(basis.clone(), vec![1, 1]);

        let galois_keys: Vec<crate::ckks::RnsGaloisKey> = (1..DIMENSION)
            .map(|index| 2 * SCALAR_DEGREE * index + 1)
            .map(|exponent| {
                let mut rng = ChaCha20Rng::seed_from_u64(0xB542_0000 ^ exponent as u64);

                crate::ckks::RnsGaloisKey::generate_with_rng(
                    LARGE_DEGREE,
                    2,
                    0,
                    &secret,
                    exponent,
                    galois_layout.clone(),
                    &mut rng,
                )
            })
            .collect();

        // ------------------------------------------------------------------
        // Authors MatrixEvaluator::matrixMult pipeline.
        // ------------------------------------------------------------------

        // C-MT rhs.
        let rhs_t = batch_ciphertext_transpose(&rhs, SCALAR_DEGREE, &galois_keys);

        // MatrixCutter::slice:
        //
        // lhs       -> 2d x d
        // rhs_t     -> 2d x d
        let lhs_mod = batch_ccmm_slice_ranked(&lhs, DIMENSION);

        let rhs_t_mod = batch_ccmm_slice_ranked(&rhs_t, DIMENSION);

        assert_eq!(lhs_mod.len(), 2 * DIMENSION);
        assert_eq!(rhs_t_mod.len(), 2 * DIMENSION);

        // Match authors' d x 2d rhs shape.
        let rhs_t_mod_t = batch_ccmm_transpose_matrix(&rhs_t_mod);

        assert_eq!(rhs_t_mod_t.len(), DIMENSION);
        assert!(rhs_t_mod_t.iter().all(|row| row.len() == 2 * DIMENSION));

        // Modular polynomial MM:
        //
        //     (2d x d) * (d x 2d)
        //       -> 2d x 2d
        let product_mod = batch_ccmm_modular_product(&lhs_mod, &rhs_t_mod_t, &scalar_plan);

        assert_eq!(product_mod.len(), 2 * DIMENSION);
        assert!(product_mod.iter().all(|row| row.len() == 2 * DIMENSION));

        // Divide into former/latter d x 2d blocks.
        let (former_mod, latter_mod) = batch_ccmm_split_former_latter(&product_mod, DIMENSION);

        // Match shape back to 2d x d.
        let former_mod_t = batch_ccmm_transpose_matrix(&former_mod);

        let latter_mod_t = batch_ccmm_transpose_matrix(&latter_mod);

        assert_eq!(former_mod_t.len(), 2 * DIMENSION);
        assert_eq!(latter_mod_t.len(), 2 * DIMENSION);

        // MatrixCutter::stack:
        //
        // first d rows  -> b
        // second d rows -> a
        let former_t = batch_ccmm_stack_ranked(&former_mod_t, DIMENSION);

        let latter_t = batch_ccmm_stack_ranked(&latter_mod_t, DIMENSION);

        assert_eq!(former_t.len(), DIMENSION);
        assert_eq!(latter_t.len(), DIMENSION);

        // Authors' second pair of C-MTs.
        let former = batch_ciphertext_transpose(&former_t, SCALAR_DEGREE, &galois_keys);

        let latter = batch_ciphertext_transpose(&latter_t, SCALAR_DEGREE, &galois_keys);

        // addCtSkCt produces the degree-2 ciphertext.
        let quadratic: Vec<_> = former
            .iter()
            .zip(&latter)
            .map(|(former, latter)| batch_ccmm_add_ct_sk_ct(former, latter))
            .collect();

        // Zero-noise multiplication key gives an exact relinearization
        // oracle for this structural integration test.
        let multiplication_layout =
            crate::grafting::RnsGadgetLayout::new(basis.clone(), vec![1, 1]);

        let mut multiplication_key_rng = ChaCha20Rng::seed_from_u64(0xB543_0000);

        let multiplication_key = crate::grafting::RnsMultiplicationKey::generate_with_rng(
            LARGE_DEGREE,
            2,
            0,
            &secret,
            multiplication_layout,
            &mut multiplication_key_rng,
        );

        let relinearized: Vec<_> = quadratic
            .iter()
            .map(|product| {
                crate::grafting::rns_relinearize_with_ntt(product, &multiplication_key, &large_plan)
            })
            .collect();

        // Compare the final encrypted columns against clear A*B.
        for (column, ciphertext) in relinearized.iter().enumerate().take(DIMENSION) {
            let decrypted =
                crate::grafting::decrypt_rns_raw_with_ntt(ciphertext, &secret, &large_plan);

            let rows = super::split_rns_polynomial(&decrypted, DIMENSION);

            assert_eq!(rows.len(), DIMENSION);

            for row in 0..DIMENSION {
                let observed = rows[row].residue(0).coefficient(0) as u128;

                assert_eq!(
                    observed, expected[row][column],
                    "source-faithful Batch CCMM mismatch \
                     at ({row},{column})"
                );
            }
        }

        println!("BATCH_CCMM_SOURCE_FAITHFUL_LAYOUT=PASS");
        println!("BATCH_CCMM_ADD_CTSKCT=PASS");
        println!("BATCH_CCMM_RELINEARIZATION=PASS");
        println!("BATCH_CCMM_SMALL_END_TO_END=PASS");
    }

    #[test]
    fn batch_ccmm_scalar_quadratic_accumulation_matches_direct_product() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let degree = 8;

        let moduli = vec![
            crate::ring::Modulus::new(12_289),
            crate::ring::Modulus::new(40_961),
        ];

        let basis = crate::ring::ModulusBasis::new(moduli.clone());
        let plan = crate::ring::RnsNttPlan::new(moduli, degree);

        let secret = [-1_i8, 0, 1, 1, 0, -1, 1, 0];

        fn encrypt(
            basis: &crate::ring::ModulusBasis,
            plan: &crate::ring::RnsNttPlan,
            secret: &[i8],
            values: &[u128],
            seed: u64,
        ) -> crate::grafting::RnsRlweCiphertext {
            let plaintext =
                crate::ring::RnsPolynomial::from_coefficients(basis.moduli().to_vec(), values);

            let mut rng = ChaCha20Rng::seed_from_u64(seed);

            crate::grafting::encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                crate::rlwe::ErrorDistribution::BoundedUniform { bound: 0 },
                secret,
                plan,
                &mut rng,
            )
        }

        let a0 = encrypt(&basis, &plan, &secret, &[1, 2, 3, 4, 5, 6, 7, 8], 0xCC00);

        let a1 = encrypt(&basis, &plan, &secret, &[2, 1, 0, 1, 2, 1, 0, 1], 0xCC01);

        let b0 = encrypt(&basis, &plan, &secret, &[1, 0, 1, 0, 1, 0, 1, 0], 0xCC10);

        let b1 = encrypt(&basis, &plan, &secret, &[0, 1, 0, 1, 0, 1, 0, 1], 0xCC11);

        let p0 = raw_cc_product(&a0, &b0, &plan);
        let p1 = raw_cc_product(&a1, &b1, &plan);

        let accumulated = raw_quadratic_add(&p0, &p1);

        for limb in 0..basis.len() {
            let expected_c0 = p0.c0().residue(limb).add(p1.c0().residue(limb));

            let expected_c1 = p0.c1().residue(limb).add(p1.c1().residue(limb));

            let expected_c2 = p0.c2().residue(limb).add(p1.c2().residue(limb));

            assert_eq!(accumulated.c0().residue(limb), &expected_c0);

            assert_eq!(accumulated.c1().residue(limb), &expected_c1);

            assert_eq!(accumulated.c2().residue(limb), &expected_c2);
        }

        println!("BATCH_CCMM_SCALAR_QUADRATIC_ACCUMULATION=PASS");
    }

    fn ccmm_batch_capability(dimension: usize, scalar_degree: usize) -> f64 {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        const LARGE_DEGREE: usize = 8192;

        let batch_count = scalar_degree / 2;
        let scale = sd3b_scale();

        assert_eq!(
            dimension * scalar_degree,
            LARGE_DEGREE,
            "Batch CCMM requires d * Ns = N"
        );

        // Authors' full-row Batch CCMM geometry.
        assert_eq!(batch_count, scalar_degree / 2);

        let lhs_clear = deterministic_cpmm_input(batch_count, dimension, 11);

        let rhs_clear = deterministic_cpmm_input(batch_count, dimension, 17);

        let expected = clear_batch_mm(&lhs_clear, &rhs_clear);

        // Unlike CPMM, CCMM uses the full d x d matrix representation.
        let lhs_complex: Vec<Vec<Vec<num_complex::Complex64>>> = lhs_clear
            .iter()
            .map(|matrix| {
                matrix
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|&value| num_complex::Complex64::new(value, 0.0))
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let rhs_complex: Vec<Vec<Vec<num_complex::Complex64>>> = rhs_clear
            .iter()
            .map(|matrix| {
                matrix
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|&value| num_complex::Complex64::new(value, 0.0))
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let lhs_sinc = super::sinc_encode_batch(&lhs_complex, scalar_degree);

        let rhs_sinc = super::sinc_encode_batch(&rhs_complex, scalar_degree);

        assert_eq!(lhs_sinc.large_degree(), LARGE_DEGREE);
        assert_eq!(rhs_sinc.large_degree(), LARGE_DEGREE);

        assert_eq!(lhs_sinc.dimension(), dimension);
        assert_eq!(rhs_sinc.dimension(), dimension);

        assert_eq!(lhs_sinc.num_columns(), dimension);
        assert_eq!(rhs_sinc.num_columns(), dimension);

        let basis = crate::ring::ModulusBasis::new(sd3b_moduli());

        let chain = crate::ring::ModulusChain::from_top_basis(basis.clone());

        let large_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), LARGE_DEGREE);

        let scalar_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), scalar_degree);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4241_5443_485f_4343 ^ dimension as u64);

        let mut secret: Vec<i8> = (0..LARGE_DEGREE)
            .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
            .collect();

        if secret.iter().all(|&value| value == 0) {
            secret[0] = 1;
        }

        let encrypt_columns =
            |columns: &[Vec<f64>], seed: u64| -> Vec<crate::grafting::RnsRlweCiphertext> {
                columns
                    .iter()
                    .enumerate()
                    .map(|(column, coefficients)| {
                        let plaintext = quantize_coefficients(coefficients, &basis, scale);

                        let mut rng = ChaCha20Rng::seed_from_u64(seed ^ column as u64);

                        crate::grafting::encrypt_rns_raw_with_distribution_ntt_rng(
                            &plaintext,
                            2,
                            crate::rlwe::ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                            &secret,
                            &large_plan,
                            &mut rng,
                        )
                    })
                    .collect()
            };

        let lhs = encrypt_columns(lhs_sinc.columns(), 0x4343_4d4d_4c48_5300);

        let rhs = encrypt_columns(rhs_sinc.columns(), 0x4343_4d4d_5248_5300);

        assert_eq!(lhs.len(), dimension);
        assert_eq!(rhs.len(), dimension);

        // ----------------------------------------------------------
        // Evaluation keys for authors' C-MT.
        //
        // h_i = 2 * Ns * i + 1, i = 1..d-1.
        // ----------------------------------------------------------

        let galois_layout = crate::grafting::RnsGadgetLayout::new(basis.clone(), vec![1, 1]);

        let galois_keys: Vec<crate::ckks::RnsGaloisKey> = (1..dimension)
            .map(|index| 2 * scalar_degree * index + 1)
            .map(|exponent| {
                let mut rng = ChaCha20Rng::seed_from_u64(0x434d_5400_0000_0000 ^ exponent as u64);

                crate::ckks::RnsGaloisKey::generate_with_rng(
                    LARGE_DEGREE,
                    2,
                    0,
                    &secret,
                    exponent,
                    galois_layout.clone(),
                    &mut rng,
                )
            })
            .collect();

        assert_eq!(galois_keys.len(), dimension - 1,);

        // ----------------------------------------------------------
        // Authors' CCMM pipeline.
        // ----------------------------------------------------------

        // 1. C-MT(rhs).
        let rhs_t = batch_ciphertext_transpose(&rhs, scalar_degree, &galois_keys);

        // 2. MatrixCutter::slice:
        //
        //        lhs   -> 2d x d
        //        rhs_t -> 2d x d
        let lhs_mod = batch_ccmm_slice_ranked(&lhs, dimension);

        let rhs_t_mod = batch_ccmm_slice_ranked(&rhs_t, dimension);

        assert_eq!(lhs_mod.len(), 2 * dimension,);

        assert_eq!(rhs_t_mod.len(), 2 * dimension,);

        // 3. transpose rhs modular representation:
        //
        //        2d x d -> d x 2d
        let rhs_t_mod_t = batch_ccmm_transpose_matrix(&rhs_t_mod);

        assert_eq!(rhs_t_mod_t.len(), dimension,);

        assert!(rhs_t_mod_t.iter().all(|row| { row.len() == 2 * dimension }));

        // 4. Modular polynomial MM:
        //
        //      (2d x d)(d x 2d) -> 2d x 2d
        let product_mod = batch_ccmm_modular_product(&lhs_mod, &rhs_t_mod_t, &scalar_plan);

        assert_eq!(product_mod.len(), 2 * dimension,);

        assert!(product_mod.iter().all(|row| { row.len() == 2 * dimension }));

        // 5. Former/latter d x 2d blocks.
        let (former_mod, latter_mod) = batch_ccmm_split_former_latter(&product_mod, dimension);

        // 6. Transpose both:
        //
        //        d x 2d -> 2d x d
        let former_mod_t = batch_ccmm_transpose_matrix(&former_mod);

        let latter_mod_t = batch_ccmm_transpose_matrix(&latter_mod);

        // 7. MatrixCutter::stack.
        let former_t = batch_ccmm_stack_ranked(&former_mod_t, dimension);

        let latter_t = batch_ccmm_stack_ranked(&latter_mod_t, dimension);

        assert_eq!(former_t.len(), dimension);
        assert_eq!(latter_t.len(), dimension);

        // 8. Authors' second pair of C-MTs.
        let former = batch_ciphertext_transpose(&former_t, scalar_degree, &galois_keys);

        let latter = batch_ciphertext_transpose(&latter_t, scalar_degree, &galois_keys);

        // 9. addCtSkCt -> degree-2 ciphertext.
        let quadratic: Vec<_> = former
            .iter()
            .zip(&latter)
            .map(|(former, latter)| batch_ccmm_add_ct_sk_ct(former, latter))
            .collect();

        assert_eq!(quadratic.len(), dimension,);

        // ----------------------------------------------------------
        // Relinearization.
        // ----------------------------------------------------------

        let multiplication_layout =
            crate::grafting::RnsGadgetLayout::new(basis.clone(), vec![1, 1]);

        let mut multiplication_key_rng = ChaCha20Rng::seed_from_u64(0x4343_4d4d_524c_4b00);

        let multiplication_key = crate::grafting::RnsMultiplicationKey::generate_with_rng(
            LARGE_DEGREE,
            2,
            0,
            &secret,
            multiplication_layout,
            &mut multiplication_key_rng,
        );

        let relinearized: Vec<_> = quadratic
            .iter()
            .map(|product| {
                crate::grafting::rns_relinearize_with_ntt(product, &multiplication_key, &large_plan)
            })
            .collect();

        assert_eq!(relinearized.len(), dimension,);

        // ----------------------------------------------------------
        // CKKS product semantics:
        //
        // input scale = Delta
        // product scale = Delta^2
        // one rescale removes q1.
        // ----------------------------------------------------------

        let output_ciphertexts: Vec<crate::ckks::RnsCkksCiphertext> = relinearized
            .into_iter()
            .map(|rlwe| {
                let product = crate::ckks::RnsCkksCiphertext::new(
                    rlwe,
                    crate::ckks::CkksChainState::top(&chain, scale * scale),
                    &chain,
                );

                crate::ckks::rescale_rns_ckks_to_next(&product, &chain)
            })
            .collect();

        assert_eq!(output_ciphertexts.len(), dimension,);

        assert!(output_ciphertexts
            .iter()
            .all(|ciphertext| { ciphertext.level() == 1 }));

        // Full-row CCMM decode.  Unlike CPMM, dimension is d,
        // not d/2.
        let decoded = decode_large_columns(&output_ciphertexts, &secret, scalar_degree, dimension);

        let actual = super::sinc_decode_batch(&decoded);

        assert_eq!(actual.len(), batch_count);
        assert_eq!(actual[0].len(), dimension);
        assert_eq!(actual[0][0].len(), dimension);

        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        let mut maximum_absolute_error = 0.0_f64;
        let mut maximum_imaginary = 0.0_f64;

        for (actual_matrix, expected_matrix) in actual.iter().zip(&expected) {
            for (actual_row, expected_row) in actual_matrix.iter().zip(expected_matrix) {
                for (&actual_value, &expected_value) in actual_row.iter().zip(expected_row) {
                    let difference =
                        actual_value - num_complex::Complex64::new(expected_value, 0.0);

                    squared_error += difference.norm_sqr();
                    squared_reference += expected_value * expected_value;

                    maximum_absolute_error = maximum_absolute_error.max(difference.norm());

                    maximum_imaginary = maximum_imaginary.max(actual_value.im.abs());
                }
            }
        }

        let relative_error = (squared_error / squared_reference).sqrt();

        println!("BATCH_CCMM_DIMENSION={dimension}");
        println!("BATCH_CCMM_SCALAR_DEGREE={scalar_degree}");
        println!("BATCH_CCMM_BATCH_COUNT={batch_count}");
        println!("BATCH_CCMM_SCALE={scale:.12e}");
        println!(
            "BATCH_CCMM_MAX_ABSOLUTE_ERROR=\
             {maximum_absolute_error:.12e}"
        );
        println!(
            "BATCH_CCMM_MAX_IMAGINARY=\
             {maximum_imaginary:.12e}"
        );
        println!(
            "BATCH_CCMM_RELATIVE_L2_ERROR=\
             {relative_error:.12e}"
        );

        relative_error
    }

    #[test]
    fn batch_ccmm_end_to_end_authors_d64() {
        let relative_error = ccmm_batch_capability(64, 128);

        println!(
            "BATCH_CCMM_D64_RELATIVE_L2_ERROR=\
             {relative_error:.12e}"
        );

        assert!(
            relative_error < 1.0e-3,
            "Batch CCMM d=64 relative error \
             {relative_error:e}"
        );

        println!("BATCH_CCMM_D64_STATUS=PASS");
    }

    #[test]
    fn batch_ccmm_end_to_end_authors_d128() {
        let relative_error = ccmm_batch_capability(128, 64);

        println!(
            "BATCH_CCMM_D128_RELATIVE_L2_ERROR=\
             {relative_error:.12e}"
        );

        assert!(
            relative_error < 1.0e-3,
            "Batch CCMM d=128 relative error \
             {relative_error:e}"
        );

        println!("BATCH_CCMM_D128_STATUS=PASS");
    }

    #[test]
    fn batch_cpmm_end_to_end_authors_d64() {
        let relative_error = cpmm_batch_capability(64, 256);

        println!("BATCH_CPMM_D64_RELATIVE_L2_ERROR={relative_error:.12e}");

        assert!(
            relative_error < 1.0e-3,
            "Batch CPMM d=64 relative error {relative_error:e}"
        );

        println!("BATCH_CPMM_D64_STATUS=PASS");
    }

    #[test]
    fn batch_cpmm_end_to_end_authors_d128() {
        let relative_error = cpmm_batch_capability(128, 128);

        println!("BATCH_CPMM_D128_RELATIVE_L2_ERROR={relative_error:.12e}");

        assert!(
            relative_error < 1.0e-3,
            "Batch CPMM d=128 relative error {relative_error:e}"
        );

        println!("BATCH_CPMM_D128_STATUS=PASS");
    }

    #[test]
    fn sinc_structural_roundtrip_authors_d128_layout() {
        // Authors' Batch d=128 configuration:
        //
        // large degree  = 8192
        // scalar degree = 64
        // CKKS slots    = 32
        // batch count   = 32
        let scalar_degree = 64;
        let dimension = 128;
        let batch_count = 32;

        let input = deterministic_batch(batch_count, dimension, dimension);

        let encoded = sinc_encode_batch(&input, scalar_degree);

        assert_eq!(encoded.large_degree(), 8192);
        assert_eq!(encoded.num_columns(), 128);

        let decoded = sinc_decode_batch(&encoded);

        let max_error = max_batch_error(&input, &decoded);

        println!("SINC_D128_MAX_ERROR={max_error:.12e}");

        assert!(
            max_error < 1.0e-9,
            "authors d=128 SinC roundtrip error {max_error:e}"
        );
    }
}
