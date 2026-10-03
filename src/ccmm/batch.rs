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

/// Extract one RNS polynomial component from an RNS RLWE ciphertext.
///
/// `b_component == true` selects the `b` component; `false` selects `a`.
fn batch_rns_rlwe_component(
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

pub fn batch_cpmm_modular_product(
    lhs_split: &[Vec<crate::grafting::RnsRlweCiphertext>],
    rhs_plain: &[Vec<crate::ring::RnsPolynomial>],
    plan: &crate::ring::PreparedRnsNttPlan,
) -> Vec<Vec<crate::grafting::RnsRlweCiphertext>> {
    const BLOCK: usize = 16;

    assert!(!lhs_split.is_empty(), "Batch CPMM lhs must not be empty");
    assert!(!rhs_plain.is_empty(), "Batch CPMM rhs must not be empty");

    let inner = lhs_split.len();
    let output_rows = lhs_split[0].len();

    assert!(output_rows > 0);
    assert!(
        lhs_split.iter().all(|column| column.len() == output_rows),
        "Batch CPMM split ciphertext matrix must be rectangular"
    );

    assert_eq!(
        rhs_plain.len(),
        inner,
        "Batch CPMM inner dimensions must match"
    );

    let output_columns = rhs_plain[0].len();

    assert!(output_columns > 0);
    assert!(
        rhs_plain.iter().all(|row| row.len() == output_columns),
        "Batch CPMM plaintext matrix must be rectangular"
    );

    let degree = plan.degree();
    let limb_count = plan.moduli().len();

    let kernel_start = std::time::Instant::now();

    // --------------------------------------------------------------
    // Transform every input exactly once.
    // --------------------------------------------------------------

    let forward_start = std::time::Instant::now();

    let lhs_b_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = lhs_split
        .iter()
        .map(|column| {
            column
                .iter()
                .map(|ciphertext| {
                    let b = batch_rns_rlwe_component(ciphertext, true);
                    plan.forward(&b)
                })
                .collect()
        })
        .collect();

    let lhs_a_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = lhs_split
        .iter()
        .map(|column| {
            column
                .iter()
                .map(|ciphertext| {
                    let a = batch_rns_rlwe_component(ciphertext, false);
                    plan.forward(&a)
                })
                .collect()
        })
        .collect();

    let rhs_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = rhs_plain
        .iter()
        .map(|row| row.iter().map(|entry| plan.forward(entry)).collect())
        .collect();

    let forward_elapsed = forward_start.elapsed();

    // Output NTT values:
    //
    // [limb][coordinate][row * output_columns + column]
    let layer_size = output_rows * output_columns;

    let mut out_b = vec![vec![vec![0_u64; layer_size]; degree]; limb_count];

    let mut out_a = vec![vec![vec![0_u64; layer_size]; degree]; limb_count];

    // --------------------------------------------------------------
    // Dense modular GEMM per (limb, NTT coordinate).
    // --------------------------------------------------------------

    let mut materialize_elapsed = std::time::Duration::ZERO;
    let mut blocked_gemm_elapsed = std::time::Duration::ZERO;

    for limb_index in 0..limb_count {
        let modulus = plan.moduli()[limb_index];

        for coordinate in 0..degree {
            let materialize_start = std::time::Instant::now();

            // Flatten this scalar layer into contiguous dense matrices.
            //
            // lhs_*: output_rows x inner
            // rhs:   inner x output_columns
            let mut lhs_b_layer = vec![0_u64; output_rows * inner];
            let mut lhs_a_layer = vec![0_u64; output_rows * inner];
            let mut rhs_layer = vec![0_u64; inner * output_columns];

            for row in 0..output_rows {
                for k in 0..inner {
                    lhs_b_layer[row * inner + k] =
                        lhs_b_ntt[k][row].residue(limb_index).values()[coordinate];

                    lhs_a_layer[row * inner + k] =
                        lhs_a_ntt[k][row].residue(limb_index).values()[coordinate];
                }
            }

            for k in 0..inner {
                for column in 0..output_columns {
                    rhs_layer[k * output_columns + column] =
                        rhs_ntt[k][column].residue(limb_index).values()[coordinate];
                }
            }

            materialize_elapsed += materialize_start.elapsed();

            let c_b = &mut out_b[limb_index][coordinate];
            let c_a = &mut out_a[limb_index][coordinate];

            let gemm_start = std::time::Instant::now();

            // O5B: delayed modular reduction.
            //
            // For canonical residues:
            //
            //     product < q^2
            //     dot product < inner * q^2
            //
            // Verify that the complete dot product fits in u128, then
            // accumulate with ordinary wide integer arithmetic and reduce
            // exactly once per output entry.
            let q = u128::from(modulus.value());
            let max_product = (q - 1) * (q - 1);

            max_product
                .checked_mul(inner as u128)
                .expect("CPMM delayed-reduction dot product must fit in u128");

            let mut wide_b = vec![0_u128; layer_size];
            let mut wide_a = vec![0_u128; layer_size];

            // Blocked i-k-j traversal.
            //
            // No modular reduction occurs in the cubic loop.
            for ii in (0..output_rows).step_by(BLOCK) {
                let i_end = (ii + BLOCK).min(output_rows);

                for kk in (0..inner).step_by(BLOCK) {
                    let k_end = (kk + BLOCK).min(inner);

                    for jj in (0..output_columns).step_by(BLOCK) {
                        let j_end = (jj + BLOCK).min(output_columns);

                        for row in ii..i_end {
                            for k in kk..k_end {
                                let lhs_b_value = u128::from(lhs_b_layer[row * inner + k]);

                                let lhs_a_value = u128::from(lhs_a_layer[row * inner + k]);

                                let rhs_base = k * output_columns;
                                let out_base = row * output_columns;

                                for column in jj..j_end {
                                    let rhs_value = u128::from(rhs_layer[rhs_base + column]);

                                    wide_b[out_base + column] += lhs_b_value * rhs_value;

                                    wide_a[out_base + column] += lhs_a_value * rhs_value;
                                }
                            }
                        }
                    }
                }
            }

            // One exact modular reduction per output entry.
            for index in 0..layer_size {
                c_b[index] = (wide_b[index] % q) as u64;
                c_a[index] = (wide_a[index] % q) as u64;
            }

            blocked_gemm_elapsed += gemm_start.elapsed();
        }
    }

    // --------------------------------------------------------------
    // Reconstruct one RNS NTT polynomial per output component.
    // --------------------------------------------------------------

    let reconstruct_start = std::time::Instant::now();

    let output: Vec<Vec<crate::grafting::RnsRlweCiphertext>> = (0..output_rows)
        .map(|row| {
            (0..output_columns)
                .map(|column| {
                    let mut b_residues = Vec::with_capacity(limb_count);
                    let mut a_residues = Vec::with_capacity(limb_count);

                    for limb_index in 0..limb_count {
                        let limb_plan = plan.plan(limb_index);

                        let b_values: Vec<u64> = (0..degree)
                            .map(|coordinate| {
                                out_b[limb_index][coordinate][row * output_columns + column]
                            })
                            .collect();

                        let a_values: Vec<u64> = (0..degree)
                            .map(|coordinate| {
                                out_a[limb_index][coordinate][row * output_columns + column]
                            })
                            .collect();

                        b_residues.push(crate::ring::NttPolynomial::from_prepared_values(
                            limb_plan, b_values,
                        ));

                        a_residues.push(crate::ring::NttPolynomial::from_prepared_values(
                            limb_plan, a_values,
                        ));
                    }

                    let b_ntt = crate::ring::RnsNttPolynomial::from_residues(b_residues);

                    let a_ntt = crate::ring::RnsNttPolynomial::from_residues(a_residues);

                    let b = plan.inverse(&b_ntt);
                    let a = plan.inverse(&a_ntt);

                    crate::grafting::RnsRlweCiphertext::from_limbs(
                        b.residues()
                            .iter()
                            .zip(a.residues())
                            .map(|(b_limb, a_limb)| {
                                crate::rlwe::RlweCiphertext::new(b_limb.clone(), a_limb.clone())
                            })
                            .collect(),
                    )
                })
                .collect()
        })
        .collect();

    let reconstruct_inverse_elapsed = reconstruct_start.elapsed();
    let kernel_elapsed = kernel_start.elapsed();

    println!(
        "BATCH_CPMM_O5_PHASE_FORWARD_NTT_MS={:.3}",
        forward_elapsed.as_secs_f64() * 1.0e3
    );
    println!(
        "BATCH_CPMM_O5_PHASE_LAYER_MATERIALIZE_MS={:.3}",
        materialize_elapsed.as_secs_f64() * 1.0e3
    );
    println!(
        "BATCH_CPMM_O5_PHASE_BLOCKED_GEMM_MS={:.3}",
        blocked_gemm_elapsed.as_secs_f64() * 1.0e3
    );
    println!(
        "BATCH_CPMM_O5_PHASE_RECONSTRUCT_INVERSE_MS={:.3}",
        reconstruct_inverse_elapsed.as_secs_f64() * 1.0e3
    );
    println!(
        "BATCH_CPMM_O5_PHASE_KERNEL_TOTAL_MS={:.3}",
        kernel_elapsed.as_secs_f64() * 1.0e3
    );

    output
}

/// Execute one packed Batch ciphertext-plaintext matrix multiplication.
///
/// The left operand consists of encrypted large-ring matrix columns. The right
/// operand is the packed scalar-ring plaintext representation used by the
/// Batch CPMM mechanism. Encoding, encryption, key generation, decryption, and
/// decoding are intentionally outside this execution boundary.
///
/// The Batch CPMM representation decomposes each large-ring ciphertext into
/// `dimension / 2` scalar-ring rows. Consequently,
///
/// `(dimension / 2) * scalar_plan.degree() == large_ring_degree`.
pub fn batch_cpmm_execute(
    lhs_ciphertexts: &[crate::ckks::RnsCkksCiphertext],
    rhs_plain: &[Vec<crate::ring::RnsPolynomial>],
    dimension: usize,
    scalar_plan: &crate::ring::RnsNttPlan,
    chain: &crate::ring::ModulusChain,
    scale: f64,
) -> Vec<crate::ckks::RnsCkksCiphertext> {
    assert!(dimension > 0, "Batch CPMM dimension must be nonzero");
    assert!(dimension % 2 == 0, "Batch CPMM dimension must be even");
    assert_eq!(
        lhs_ciphertexts.len(),
        dimension,
        "Batch CPMM encrypted left operand must contain one large-ring ciphertext per column"
    );
    assert_eq!(
        rhs_plain.len(),
        dimension,
        "Batch CPMM plaintext right operand must contain one row per matrix row"
    );

    let half_rows = dimension / 2;
    let scalar_degree = scalar_plan.degree();

    assert!(!lhs_ciphertexts.is_empty());

    let large_degree = lhs_ciphertexts[0].rlwe().degree();

    assert_eq!(
        half_rows
            .checked_mul(scalar_degree)
            .expect("Batch CPMM decomposition degree overflow"),
        large_degree,
        "Batch CPMM decomposition must match the large-ring degree"
    );

    for ciphertext in lhs_ciphertexts {
        ciphertext.assert_matches_chain(chain);
        assert_eq!(
            ciphertext.rlwe().degree(),
            large_degree,
            "Batch CPMM ciphertexts must have the same large-ring degree"
        );
        assert_eq!(
            ciphertext.level(),
            0,
            "Batch CPMM inputs must be at the top CKKS level"
        );
    }

    for row in rhs_plain {
        assert_eq!(
            row.len(),
            dimension,
            "Batch CPMM plaintext right operand must be square"
        );

        for polynomial in row {
            assert_eq!(
                polynomial.degree(),
                scalar_degree,
                "Batch CPMM plaintext polynomial degree must match scalar NTT plan"
            );
        }
    }

    let lhs_split: Vec<Vec<crate::grafting::RnsRlweCiphertext>> = lhs_ciphertexts
        .iter()
        .map(|ciphertext| split_rns_rlwe_ciphertext(ciphertext.rlwe(), half_rows))
        .collect();

    let prepared_scalar_plan = crate::ring::PreparedRnsNttPlan::new(scalar_plan);

    let output_rows = batch_cpmm_modular_product(&lhs_split, rhs_plain, &prepared_scalar_plan);

    let mut output_ciphertexts = Vec::with_capacity(dimension);

    for col in 0..dimension {
        let refs: Vec<&crate::grafting::RnsRlweCiphertext> =
            (0..half_rows).map(|row| &output_rows[row][col]).collect();

        let combined = combine_rns_rlwe_ciphertexts(&refs);

        assert_eq!(
            combined.degree(),
            large_degree,
            "Batch CPMM reconstructed ciphertext degree changed"
        );

        let product = crate::ckks::RnsCkksCiphertext::new(
            combined,
            crate::ckks::CkksChainState::top(chain, scale * scale),
            chain,
        );

        output_ciphertexts.push(crate::ckks::rescale_rns_ckks_to_next(&product, chain));
    }

    assert!(
        output_ciphertexts
            .iter()
            .all(|ciphertext| ciphertext.level() == 1),
        "Batch CPMM outputs must be at CKKS level one after rescale"
    );

    output_ciphertexts
}

// === E2A PRODUCTION BATCH CCMM DEFAULT CLOSURE ===
//
// Frozen coefficient-domain Batch CCMM implementation used by the
// production/eBLAS path. Experimental O18 resident selectors remain
// test-private and are intentionally not part of this dependency closure.

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
    let normalized = i128::from(exponent).rem_euclid(period) as usize;

    let shift = normalized % degree;
    let sign = if normalized >= degree { -1_i8 } else { 1_i8 };

    polynomial.mul_monomial_signed(shift, sign)
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

fn batch_polynomial_butterfly_monomial(
    even: &crate::ring::Polynomial,
    odd: &crate::ring::Polynomial,
    exponent: i64,
) -> (crate::ring::Polynomial, crate::ring::Polynomial) {
    assert_eq!(
        even.modulus(),
        odd.modulus(),
        "Batch polynomial butterfly moduli must match"
    );

    assert_eq!(
        even.degree(),
        odd.degree(),
        "Batch polynomial butterfly degrees must match"
    );

    let modulus = even.modulus();
    let degree = even.degree();

    let period = 2_i128 * degree as i128;

    let normalized = i128::from(exponent).rem_euclid(period) as usize;

    let shift = normalized % degree;

    let global_negative = normalized >= degree;

    let mut sum = Vec::with_capacity(degree);

    let mut diff = Vec::with_capacity(degree);

    for output_index in 0..degree {
        let (source_index, wrapped) = if output_index >= shift {
            (output_index - shift, false)
        } else {
            (output_index + degree - shift, true)
        };

        let mut twiddled = odd.coefficients()[source_index];

        if wrapped ^ global_negative {
            twiddled = modulus.neg(twiddled);
        }

        let even_value = even.coefficients()[output_index];

        sum.push(modulus.add_canonical(even_value, twiddled));

        diff.push(modulus.sub_canonical(even_value, twiddled));
    }

    (
        crate::ring::Polynomial::new(modulus, sum),
        crate::ring::Polynomial::new(modulus, diff),
    )
}

fn batch_rlwe_butterfly_monomial(
    even: &crate::grafting::RnsRlweCiphertext,
    odd: &crate::grafting::RnsRlweCiphertext,
    exponent: i64,
) -> (
    crate::grafting::RnsRlweCiphertext,
    crate::grafting::RnsRlweCiphertext,
) {
    assert_eq!(
        even.basis(),
        odd.basis(),
        "Batch RLWE butterfly bases must match"
    );

    assert_eq!(
        even.degree(),
        odd.degree(),
        "Batch RLWE butterfly degrees must match"
    );

    let mut upper_limbs = Vec::with_capacity(even.limbs().len());

    let mut lower_limbs = Vec::with_capacity(even.limbs().len());

    for (even_limb, odd_limb) in even.limbs().iter().zip(odd.limbs()) {
        let (upper_b, lower_b) =
            batch_polynomial_butterfly_monomial(even_limb.b(), odd_limb.b(), exponent);

        let (upper_a, lower_a) =
            batch_polynomial_butterfly_monomial(even_limb.a(), odd_limb.a(), exponent);

        upper_limbs.push(crate::rlwe::RlweCiphertext::new(upper_b, upper_a));

        lower_limbs.push(crate::rlwe::RlweCiphertext::new(lower_b, lower_a));
    }

    (
        crate::grafting::RnsRlweCiphertext::from_limbs(upper_limbs),
        crate::grafting::RnsRlweCiphertext::from_limbs(lower_limbs),
    )
}

fn batch_bit_reverse_permute<T>(values: &mut [T]) {
    let n = values.len();

    assert!(
        n.is_power_of_two(),
        "Batch bit reversal requires power-of-two length"
    );

    let mut j = 0_usize;

    for i in 1..n {
        let mut bit = n >> 1;

        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if i < j {
            values.swap(i, j);
        }
    }
}

fn batch_large_dft_iterative(
    input: &[crate::grafting::RnsRlweCiphertext],
    multiplier: i64,
) -> Vec<crate::grafting::RnsRlweCiphertext> {
    let dimension = input.len();

    assert!(
        dimension.is_power_of_two(),
        "Batch iterative large DFT dimension must be a power of two"
    );

    assert!(
        !input.is_empty(),
        "Batch iterative large DFT requires input"
    );

    let mut output = input.to_vec();

    batch_bit_reverse_permute(&mut output);

    let mut len = 2_usize;

    while len <= dimension {
        let half = len / 2;

        // Recursive implementation doubles the multiplier at each
        // descent. In iterative order, stage len therefore uses:
        //
        //     multiplier * (dimension / len)
        //
        // as the stage base exponent.
        let stage_multiplier = multiplier * (dimension / len) as i64;

        for start in (0..dimension).step_by(len) {
            for k in 0..half {
                let upper_index = start + k;
                let lower_index = start + k + half;

                let (upper, lower) = batch_rlwe_butterfly_monomial(
                    &output[upper_index],
                    &output[lower_index],
                    stage_multiplier * k as i64,
                );

                output[upper_index] = upper;
                output[lower_index] = lower;
            }
        }

        len *= 2;
    }

    output
}

fn batch_large_crt_profiled(
    ciphertexts: &[crate::grafting::RnsRlweCiphertext],
    scalar_degree: usize,
) -> (
    Vec<crate::grafting::RnsRlweCiphertext>,
    std::time::Duration,
    std::time::Duration,
) {
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

    let normalize_start = std::time::Instant::now();

    let normalized: Vec<_> = ciphertexts
        .iter()
        .enumerate()
        .map(|(column, ciphertext)| {
            let scaled = batch_rlwe_scalar_mul(ciphertext, &inverse_dimension);

            batch_rlwe_mul_monomial_signed(&scaled, column as i64)
        })
        .collect();

    let normalize_elapsed = normalize_start.elapsed();

    let dft_start = std::time::Instant::now();

    let output = batch_large_dft_iterative(&normalized, (2 * scalar_degree) as i64);

    let dft_elapsed = dft_start.elapsed();

    (output, normalize_elapsed, dft_elapsed)
}

fn batch_large_inv_crt_profiled(
    ciphertexts: &[crate::grafting::RnsRlweCiphertext],
    scalar_degree: usize,
) -> (
    Vec<crate::grafting::RnsRlweCiphertext>,
    std::time::Duration,
    std::time::Duration,
) {
    let dimension = ciphertexts.len();

    assert!(!ciphertexts.is_empty());
    assert!(dimension.is_power_of_two());

    let large_degree = ciphertexts[0].degree();

    assert_eq!(
        scalar_degree * dimension,
        large_degree,
        "Batch inverse CRT geometry must satisfy Ns * d = N"
    );

    let dft_start = std::time::Instant::now();

    let transformed = batch_large_dft_iterative(ciphertexts, -((2 * scalar_degree) as i64));

    let dft_elapsed = dft_start.elapsed();

    let post_start = std::time::Instant::now();

    let output = transformed
        .into_iter()
        .enumerate()
        .map(|(column, ciphertext)| batch_rlwe_mul_monomial_signed(&ciphertext, -(column as i64)))
        .collect();

    let post_elapsed = post_start.elapsed();

    (output, dft_elapsed, post_elapsed)
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

fn batch_prepared_galois_key_for_exponent(
    keys: &[crate::ckks::PreparedRnsGaloisKey],
    exponent: usize,
) -> &crate::ckks::PreparedRnsGaloisKey {
    keys.iter()
        .find(|key| key.exponent() == exponent)
        .unwrap_or_else(|| {
            panic!("missing prepared Batch C-MT RNS Galois key for exponent {exponent}")
        })
}

fn batch_scrambled_auto_prepared_dynamic(
    ciphertexts: &[crate::grafting::RnsRlweCiphertext],
    scalar_degree: usize,
    galois_keys: &[crate::ckks::PreparedRnsGaloisKey],
    plan: &crate::ring::PreparedRnsNttPlan,
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

    let mut automorphism_seconds = 0.0_f64;

    let mut decompose_seconds = 0.0_f64;
    let mut base_forward_seconds = 0.0_f64;
    let mut digit_prepare_seconds = 0.0_f64;
    let mut digit_forward_seconds = 0.0_f64;
    let mut mac_seconds = 0.0_f64;
    let mut inverse_seconds = 0.0_f64;

    let mut base_forward_count = 0_usize;
    let mut digit_forward_count = 0_usize;
    let mut inverse_count = 0_usize;

    let mut output = Vec::with_capacity(dimension);

    for index in 0..dimension {
        let exponent = 2 * scalar_degree * index + 1;

        let inverse = batch_mod_inverse_odd(exponent, two_n);

        assert_eq!(
            (inverse - 1) % (2 * scalar_degree),
            0,
            "inverse Batch automorphism exponent must remain in subgroup"
        );

        let inverse_index = (inverse - 1) / (2 * scalar_degree);

        if exponent == 1 {
            output.push(ciphertexts[inverse_index].clone());
        } else {
            let key = batch_prepared_galois_key_for_exponent(galois_keys, exponent);

            let (transformed, automorphism, profile) =
                crate::ckks::apply_rns_galois_automorphism_with_prepared_dynamic_ntt_profiled(
                    &ciphertexts[inverse_index],
                    key,
                    plan,
                );

            automorphism_seconds += automorphism;

            decompose_seconds += profile.decompose_seconds;

            base_forward_seconds += profile.base_forward_seconds;

            digit_prepare_seconds += profile.digit_prepare_seconds;

            digit_forward_seconds += profile.digit_forward_seconds;

            mac_seconds += profile.mac_seconds;

            inverse_seconds += profile.inverse_seconds;

            base_forward_count += profile.base_forward_count;

            digit_forward_count += profile.digit_forward_count;

            inverse_count += profile.inverse_count;

            output.push(transformed);
        }
    }

    let accounted_key_switch_seconds = decompose_seconds
        + base_forward_seconds
        + digit_prepare_seconds
        + digit_forward_seconds
        + mac_seconds
        + inverse_seconds;

    println!(
        "BATCH_GALOIS_PHASE_AUTOMORPHISM_MS={:.3}",
        automorphism_seconds * 1.0e3
    );

    println!(
        "BATCH_KEY_SWITCH_PHASE_DECOMPOSE_MS={:.3}",
        decompose_seconds * 1.0e3
    );

    println!(
        "BATCH_KEY_SWITCH_PHASE_BASE_FORWARD_MS={:.3}",
        base_forward_seconds * 1.0e3
    );

    println!(
        "BATCH_KEY_SWITCH_PHASE_DIGIT_PREPARE_MS={:.3}",
        digit_prepare_seconds * 1.0e3
    );

    println!(
        "BATCH_KEY_SWITCH_PHASE_DIGIT_FORWARD_MS={:.3}",
        digit_forward_seconds * 1.0e3
    );

    println!("BATCH_KEY_SWITCH_PHASE_MAC_MS={:.3}", mac_seconds * 1.0e3);

    println!(
        "BATCH_KEY_SWITCH_PHASE_INVERSE_MS={:.3}",
        inverse_seconds * 1.0e3
    );

    println!("BATCH_KEY_SWITCH_BASE_FORWARD_COUNT={}", base_forward_count);

    println!(
        "BATCH_KEY_SWITCH_DIGIT_FORWARD_COUNT={}",
        digit_forward_count
    );

    println!("BATCH_KEY_SWITCH_INVERSE_COUNT={}", inverse_count);

    println!(
        "BATCH_KEY_SWITCH_BASE_FORWARD_US_PER_TRANSFORM={:.3}",
        if base_forward_count == 0 {
            0.0
        } else {
            base_forward_seconds * 1.0e6 / base_forward_count as f64
        }
    );

    println!(
        "BATCH_KEY_SWITCH_DIGIT_FORWARD_US_PER_TRANSFORM={:.3}",
        if digit_forward_count == 0 {
            0.0
        } else {
            digit_forward_seconds * 1.0e6 / digit_forward_count as f64
        }
    );

    println!(
        "BATCH_KEY_SWITCH_INVERSE_US_PER_TRANSFORM={:.3}",
        if inverse_count == 0 {
            0.0
        } else {
            inverse_seconds * 1.0e6 / inverse_count as f64
        }
    );

    println!(
        "BATCH_GALOIS_PHASE_KEY_SWITCH_ACCOUNTED_MS={:.3}",
        accounted_key_switch_seconds * 1.0e3
    );

    output
}

fn batch_ciphertext_transpose_prepared_dynamic(
    ciphertexts: &[crate::grafting::RnsRlweCiphertext],
    scalar_degree: usize,
    galois_keys: &[crate::ckks::PreparedRnsGaloisKey],
    plan: &crate::ring::PreparedRnsNttPlan,
) -> Vec<crate::grafting::RnsRlweCiphertext> {
    let total_start = std::time::Instant::now();

    let phase_start = std::time::Instant::now();

    let (crt, crt_normalize_elapsed, crt_dft_elapsed) =
        batch_large_crt_profiled(ciphertexts, scalar_degree);

    let crt_elapsed = phase_start.elapsed();

    let phase_start = std::time::Instant::now();

    let transformed = batch_scrambled_auto_prepared_dynamic(&crt, scalar_degree, galois_keys, plan);

    let scrambled_auto_elapsed = phase_start.elapsed();

    let phase_start = std::time::Instant::now();

    let (output, inv_crt_dft_elapsed, inv_crt_post_elapsed) =
        batch_large_inv_crt_profiled(&transformed, scalar_degree);

    let inv_crt_elapsed = phase_start.elapsed();

    let total_elapsed = total_start.elapsed();

    println!(
        "BATCH_CMT_CRT_NORMALIZE_MS={:.3}",
        crt_normalize_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CMT_CRT_DFT_MS={:.3}",
        crt_dft_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CMT_LARGE_CRT_MS={:.3}",
        crt_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CMT_SCRAMBLED_AUTO_MS={:.3}",
        scrambled_auto_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CMT_INV_CRT_DFT_MS={:.3}",
        inv_crt_dft_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CMT_INV_CRT_POST_MS={:.3}",
        inv_crt_post_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CMT_LARGE_INV_CRT_MS={:.3}",
        inv_crt_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CMT_TOTAL_MS={:.3}",
        total_elapsed.as_secs_f64() * 1.0e3
    );

    output
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
                split_rns_polynomial(&b, dimension),
                split_rns_polynomial(&a, dimension),
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

fn batch_ccmm_modular_product_delayed_ntt(
    lhs: &[Vec<crate::ring::RnsPolynomial>],
    rhs_transposed: &[Vec<crate::ring::RnsPolynomial>],
    plan: &crate::ring::PreparedRnsNttPlan,
) -> Vec<Vec<crate::ring::RnsPolynomial>> {
    let block = std::env::var("CCMM_GEMM_BLOCK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(128);

    assert!(block > 0, "CCMM GEMM block size must be nonzero");

    assert!(!lhs.is_empty(), "Batch CCMM lhs must not be empty");
    assert!(
        !rhs_transposed.is_empty(),
        "Batch CCMM rhs must not be empty"
    );

    let output_rows = lhs.len();
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

    let degree = plan.degree();
    let limb_count = plan.moduli().len();

    // Transform each polynomial matrix entry exactly once.
    let forward_start = std::time::Instant::now();

    let lhs_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = lhs
        .iter()
        .map(|row| row.iter().map(|entry| plan.forward(entry)).collect())
        .collect();

    let rhs_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = rhs_transposed
        .iter()
        .map(|row| row.iter().map(|entry| plan.forward(entry)).collect())
        .collect();

    let forward_elapsed = forward_start.elapsed();

    let output_layer_size = output_rows * output_columns;

    // [limb][coordinate][row * output_columns + column]
    let mut output_layers = vec![vec![vec![0_u64; output_layer_size]; degree]; limb_count];

    let mut materialize_elapsed = std::time::Duration::ZERO;
    let mut gemm_elapsed = std::time::Duration::ZERO;

    let mut limb_materialize_elapsed = vec![std::time::Duration::ZERO; limb_count];

    let mut limb_gemm_elapsed = vec![std::time::Duration::ZERO; limb_count];

    let mut accumulate_elapsed = std::time::Duration::ZERO;

    let mut reduction_elapsed = std::time::Duration::ZERO;

    let mut limb_accumulate_elapsed = vec![std::time::Duration::ZERO; limb_count];

    let mut limb_reduction_elapsed = vec![std::time::Duration::ZERO; limb_count];

    for (limb_index, output_limb_layers) in output_layers.iter_mut().enumerate().take(limb_count) {
        let modulus = plan.moduli()[limb_index];
        let q = u128::from(modulus.value());

        let max_product = (q - 1) * (q - 1);

        max_product
            .checked_mul(inner as u128)
            .expect("CCMM delayed-reduction dot product must fit in u128");

        for (coordinate, output_layer) in output_limb_layers.iter_mut().enumerate().take(degree) {
            let materialize_start = std::time::Instant::now();

            let mut lhs_layer = vec![0_u64; output_rows * inner];

            let mut rhs_layer = vec![0_u64; inner * output_columns];

            for row in 0..output_rows {
                for k in 0..inner {
                    lhs_layer[row * inner + k] =
                        lhs_ntt[row][k].residue(limb_index).values()[coordinate];
                }
            }

            for k in 0..inner {
                for column in 0..output_columns {
                    rhs_layer[k * output_columns + column] =
                        rhs_ntt[k][column].residue(limb_index).values()[coordinate];
                }
            }

            let materialize_coordinate_elapsed = materialize_start.elapsed();

            materialize_elapsed += materialize_coordinate_elapsed;

            limb_materialize_elapsed[limb_index] += materialize_coordinate_elapsed;

            let mut wide = vec![0_u128; output_layer_size];

            let gemm_start = std::time::Instant::now();

            let accumulate_start = std::time::Instant::now();

            // Dense blocked i-k-j multiplication with no modular
            // reduction in the cubic loop.
            for ii in (0..output_rows).step_by(block) {
                let i_end = (ii + block).min(output_rows);

                for kk in (0..inner).step_by(block) {
                    let k_end = (kk + block).min(inner);

                    for jj in (0..output_columns).step_by(block) {
                        let j_end = (jj + block).min(output_columns);

                        for row in ii..i_end {
                            let lhs_base = row * inner;

                            let out_base = row * output_columns;

                            for k in kk..k_end {
                                let lhs_value = u128::from(lhs_layer[lhs_base + k]);

                                let rhs_base = k * output_columns;

                                for column in jj..j_end {
                                    let rhs_value = u128::from(rhs_layer[rhs_base + column]);

                                    wide[out_base + column] += lhs_value * rhs_value;
                                }
                            }
                        }
                    }
                }
            }

            let accumulate_coordinate_elapsed = accumulate_start.elapsed();

            accumulate_elapsed += accumulate_coordinate_elapsed;

            limb_accumulate_elapsed[limb_index] += accumulate_coordinate_elapsed;

            let reduction_start = std::time::Instant::now();

            // One exact reduction per output matrix entry.
            for index in 0..output_layer_size {
                output_layer[index] = (wide[index] % q) as u64;
            }

            let reduction_coordinate_elapsed = reduction_start.elapsed();

            reduction_elapsed += reduction_coordinate_elapsed;

            limb_reduction_elapsed[limb_index] += reduction_coordinate_elapsed;

            let gemm_coordinate_elapsed = gemm_start.elapsed();

            gemm_elapsed += gemm_coordinate_elapsed;

            limb_gemm_elapsed[limb_index] += gemm_coordinate_elapsed;
        }
    }

    for limb_index in 0..limb_count {
        println!(
            "BATCH_CCMM_DELAYED_LIMB_{}_MATERIALIZE_MS={:.3}",
            limb_index,
            limb_materialize_elapsed[limb_index].as_secs_f64() * 1.0e3
        );

        println!(
            "BATCH_CCMM_DELAYED_LIMB_{}_GEMM_MS={:.3}",
            limb_index,
            limb_gemm_elapsed[limb_index].as_secs_f64() * 1.0e3
        );

        println!(
            "BATCH_CCMM_DELAYED_LIMB_{}_ACCUMULATE_MS={:.3}",
            limb_index,
            limb_accumulate_elapsed[limb_index].as_secs_f64() * 1.0e3
        );

        println!(
            "BATCH_CCMM_DELAYED_LIMB_{}_REDUCTION_MS={:.3}",
            limb_index,
            limb_reduction_elapsed[limb_index].as_secs_f64() * 1.0e3
        );
    }

    let assemble_start = std::time::Instant::now();

    let output_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = (0..output_rows)
        .map(|row| {
            (0..output_columns)
                .map(|column| {
                    let residues = (0..limb_count)
                        .map(|limb_index| {
                            let values: Vec<u64> = (0..degree)
                                .map(|coordinate| {
                                    output_layers[limb_index][coordinate]
                                        [row * output_columns + column]
                                })
                                .collect();

                            crate::ring::NttPolynomial::from_prepared_values(
                                plan.plan(limb_index),
                                values,
                            )
                        })
                        .collect();

                    crate::ring::RnsNttPolynomial::from_residues(residues)
                })
                .collect()
        })
        .collect();

    let assemble_elapsed = assemble_start.elapsed();

    let inverse_start = std::time::Instant::now();

    let output: Vec<Vec<crate::ring::RnsPolynomial>> = output_ntt
        .iter()
        .map(|row| row.iter().map(|ntt| plan.inverse(ntt)).collect())
        .collect();

    let inverse_elapsed = inverse_start.elapsed();

    let reconstruct_elapsed = assemble_elapsed + inverse_elapsed;

    println!(
        "BATCH_CCMM_DELAYED_PHASE_FORWARD_NTT_MS={:.3}",
        forward_elapsed.as_secs_f64() * 1.0e3
    );
    println!(
        "BATCH_CCMM_DELAYED_PHASE_LAYER_MATERIALIZE_MS={:.3}",
        materialize_elapsed.as_secs_f64() * 1.0e3
    );
    println!(
        "BATCH_CCMM_DELAYED_PHASE_ACCUMULATE_MS={:.3}",
        accumulate_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CCMM_DELAYED_PHASE_REDUCTION_MS={:.3}",
        reduction_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CCMM_DELAYED_PHASE_GEMM_MS={:.3}",
        gemm_elapsed.as_secs_f64() * 1.0e3
    );
    println!(
        "BATCH_CCMM_DELAYED_PHASE_ASSEMBLE_NTT_MS={:.3}",
        assemble_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CCMM_DELAYED_PHASE_INVERSE_NTT_MS={:.3}",
        inverse_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CCMM_DELAYED_PHASE_RECONSTRUCT_INVERSE_MS={:.3}",
        reconstruct_elapsed.as_secs_f64() * 1.0e3
    );

    output
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

            let b = combine_rns_polynomials(&b_refs);
            let a = combine_rns_polynomials(&a_refs);

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

/// Execute one packed Batch ciphertext-ciphertext matrix multiplication.
///
/// This is the frozen coefficient-domain Batch CCMM path used by eBLAS.
/// Encoding, encryption, evaluation-key generation/preparation, decryption,
/// and decoding are intentionally outside this execution boundary.
///
/// The execution sequence is:
///
/// 1. ranked slicing of the encrypted left operand,
/// 2. ciphertext matrix transpose (C-MT) of the right operand,
/// 3. ranked slicing and transpose of the transformed right operand,
/// 4. delayed-reduction scalar-ring modular matrix multiplication,
/// 5. former/latter reconstruction,
/// 6. a second pair of ciphertext matrix transposes,
/// 7. `addCtSkCt` quadratic reconstruction,
/// 8. prepared relinearization,
/// 9. one CKKS rescale.
pub struct BatchCcmmExecutionContext<'a> {
    pub scalar_plan: &'a crate::ring::RnsNttPlan,
    pub prepared_scalar_plan: &'a crate::ring::PreparedRnsNttPlan,
    pub prepared_large_plan: &'a crate::ring::PreparedRnsNttPlan,
    pub prepared_galois_keys: &'a [crate::ckks::PreparedRnsGaloisKey],
    pub prepared_multiplication_key: &'a crate::grafting::PreparedRnsMultiplicationKey,
    pub chain: &'a crate::ring::ModulusChain,
    pub scale: f64,
}

/// Execute one packed Batch ciphertext-ciphertext matrix multiplication.
pub fn batch_ccmm_execute(
    lhs: &[crate::grafting::RnsRlweCiphertext],
    rhs: &[crate::grafting::RnsRlweCiphertext],
    dimension: usize,
    context: &BatchCcmmExecutionContext<'_>,
) -> Vec<crate::ckks::RnsCkksCiphertext> {
    let scalar_plan = context.scalar_plan;
    let prepared_scalar_plan = context.prepared_scalar_plan;
    let prepared_large_plan = context.prepared_large_plan;
    let prepared_galois_keys = context.prepared_galois_keys;
    let prepared_multiplication_key = context.prepared_multiplication_key;
    let chain = context.chain;
    let scale = context.scale;
    assert!(dimension > 0, "Batch CCMM dimension must be positive");

    assert_eq!(
        lhs.len(),
        dimension,
        "Batch CCMM left operand must contain d ciphertext columns"
    );
    assert_eq!(
        rhs.len(),
        dimension,
        "Batch CCMM right operand must contain d ciphertext columns"
    );

    let scalar_degree = scalar_plan.degree();

    assert_eq!(
        prepared_scalar_plan.degree(),
        scalar_degree,
        "Batch CCMM prepared scalar plan degree must match scalar plan"
    );
    assert_eq!(
        prepared_scalar_plan.moduli(),
        scalar_plan.moduli(),
        "Batch CCMM prepared scalar plan basis must match scalar plan"
    );

    assert_eq!(
        prepared_large_plan.degree(),
        dimension
            .checked_mul(scalar_degree)
            .expect("Batch CCMM geometry overflow"),
        "Batch CCMM geometry must satisfy d * Ns = N"
    );

    assert_eq!(
        prepared_galois_keys.len(),
        dimension - 1,
        "Batch CCMM requires one prepared Galois key per non-identity C-MT exponent"
    );

    for ciphertext in lhs.iter().chain(rhs.iter()) {
        assert_eq!(
            ciphertext.degree(),
            prepared_large_plan.degree(),
            "Batch CCMM ciphertext degree must match large-ring plan"
        );

        assert_eq!(
            ciphertext.basis(),
            chain.top(),
            "Batch CCMM ciphertext basis must match the top CKKS chain basis"
        );
    }

    // ----------------------------------------------------------
    // 1. Ranked slicing of lhs.
    // ----------------------------------------------------------

    let lhs_mod = batch_ccmm_slice_ranked(lhs, dimension);

    assert_eq!(lhs_mod.len(), 2 * dimension);

    // ----------------------------------------------------------
    // 2. Default coefficient-domain RHS C-MT.
    // ----------------------------------------------------------

    let rhs_t = batch_ciphertext_transpose_prepared_dynamic(
        rhs,
        scalar_degree,
        prepared_galois_keys,
        prepared_large_plan,
    );

    // ----------------------------------------------------------
    // 3. Ranked slice and transpose RHS.
    // ----------------------------------------------------------

    let rhs_t_mod = batch_ccmm_slice_ranked(&rhs_t, dimension);

    assert_eq!(rhs_t_mod.len(), 2 * dimension);

    let rhs_t_mod_t = batch_ccmm_transpose_matrix(&rhs_t_mod);

    assert_eq!(rhs_t_mod_t.len(), dimension);

    assert!(rhs_t_mod_t.iter().all(|row| row.len() == 2 * dimension));

    // ----------------------------------------------------------
    // 4. Delayed-reduction modular matrix multiplication.
    // ----------------------------------------------------------

    let product_mod =
        batch_ccmm_modular_product_delayed_ntt(&lhs_mod, &rhs_t_mod_t, prepared_scalar_plan);

    assert_eq!(product_mod.len(), 2 * dimension);

    assert!(product_mod.iter().all(|row| row.len() == 2 * dimension));

    // ----------------------------------------------------------
    // 5-7. Former/latter reconstruction.
    // ----------------------------------------------------------

    let (former_mod, latter_mod) = batch_ccmm_split_former_latter(&product_mod, dimension);

    let former_mod_t = batch_ccmm_transpose_matrix(&former_mod);

    let latter_mod_t = batch_ccmm_transpose_matrix(&latter_mod);

    let former_t = batch_ccmm_stack_ranked(&former_mod_t, dimension);

    let latter_t = batch_ccmm_stack_ranked(&latter_mod_t, dimension);

    assert_eq!(former_t.len(), dimension);
    assert_eq!(latter_t.len(), dimension);

    // ----------------------------------------------------------
    // 8. Second pair of default coefficient-domain C-MTs.
    // ----------------------------------------------------------

    let former = batch_ciphertext_transpose_prepared_dynamic(
        &former_t,
        scalar_degree,
        prepared_galois_keys,
        prepared_large_plan,
    );

    let latter = batch_ciphertext_transpose_prepared_dynamic(
        &latter_t,
        scalar_degree,
        prepared_galois_keys,
        prepared_large_plan,
    );

    // ----------------------------------------------------------
    // 9. addCtSkCt -> quadratic ciphertext.
    // ----------------------------------------------------------

    let quadratic: Vec<_> = former
        .iter()
        .zip(&latter)
        .map(|(former, latter)| batch_ccmm_add_ct_sk_ct(former, latter))
        .collect();

    assert_eq!(quadratic.len(), dimension);

    // ----------------------------------------------------------
    // 10. Prepared relinearization.
    // ----------------------------------------------------------

    let relinearized: Vec<_> = quadratic
        .iter()
        .map(|product| {
            crate::grafting::rns_relinearize_with_prepared_dynamic_ntt(
                product,
                prepared_multiplication_key,
                prepared_large_plan,
            )
        })
        .collect();

    assert_eq!(relinearized.len(), dimension);

    // ----------------------------------------------------------
    // 11. CKKS product semantics: Delta^2 -> Delta.
    // ----------------------------------------------------------

    let output_ciphertexts: Vec<crate::ckks::RnsCkksCiphertext> = relinearized
        .into_iter()
        .map(|rlwe| {
            let product = crate::ckks::RnsCkksCiphertext::new(
                rlwe,
                crate::ckks::CkksChainState::top(chain, scale * scale),
                chain,
            );

            crate::ckks::rescale_rns_ckks_to_next(&product, chain)
        })
        .collect();

    assert_eq!(output_ciphertexts.len(), dimension);

    assert!(output_ciphertexts
        .iter()
        .all(|ciphertext| ciphertext.level() == 1));

    output_ciphertexts
}

/// Quantizes one real coefficient vector into the supplied RNS basis.
///
/// This is the representation boundary between floating-point CKKS
/// coefficients and their integer RNS plaintext representation.
fn quantize_real_coefficients(
    coefficients: &[f64],
    basis: &crate::ring::ModulusBasis,
    scale: f64,
) -> crate::ring::RnsPolynomial {
    assert!(
        scale.is_finite() && scale > 0.0,
        "CKKS scale must be positive"
    );

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

/// Quantizes the large-ring columns of a structural SinC plaintext.
///
/// SinC packing itself remains independent of the modulus basis and CKKS
/// scale. This function supplies exactly that later CKKS/RNS representation
/// step without performing encryption.
pub fn quantize_sinc_batch_plaintext(
    plaintext: &SinCBatchPlaintext,
    basis: &crate::ring::ModulusBasis,
    scale: f64,
) -> Vec<crate::ring::RnsPolynomial> {
    plaintext
        .columns()
        .iter()
        .map(|coefficients| quantize_real_coefficients(coefficients, basis, scale))
        .collect()
}

/// Encodes one physical batch of real square matrices as the scalar-ring
/// plaintext representation required by Batch CPMM.
///
/// `matrices` is batch-major. Its batch count must equal `scalar_degree / 2`.
/// Values occupying the same matrix coordinate across physical lanes are
/// placed into canonical CKKS slots using the authors' bit-reversed lane
/// ordering, transformed to coefficients, and quantized into the supplied RNS
/// basis.
///
/// This function performs no encryption and no Batch CPMM execution.
pub fn encode_cpmm_real_batch(
    matrices: &[Vec<Vec<f64>>],
    scalar_degree: usize,
    basis: &crate::ring::ModulusBasis,
    scale: f64,
) -> Vec<Vec<crate::ring::RnsPolynomial>> {
    assert!(
        !matrices.is_empty(),
        "Batch CPMM real batch must not be empty"
    );
    assert!(
        scalar_degree > 0 && scalar_degree % 2 == 0,
        "Batch CPMM scalar degree must be positive and even"
    );

    let batch_count = matrices.len();
    let dimension = matrices[0].len();

    assert!(
        dimension > 0,
        "Batch CPMM matrix dimension must be positive"
    );
    assert_eq!(
        batch_count,
        scalar_degree / 2,
        "Batch CPMM real batch count must equal the CKKS slot count"
    );
    assert!(
        batch_count.is_power_of_two(),
        "Batch CPMM real batch count must be a power of two"
    );
    assert!(
        matrices.iter().all(|matrix| {
            matrix.len() == dimension && matrix.iter().all(|row| row.len() == dimension)
        }),
        "Batch CPMM real matrices must have one common square dimension"
    );

    let embedding = crate::ckks::CkksCanonicalEmbedding::new(scalar_degree);
    let log_slots = batch_count.trailing_zeros();

    (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|col| {
                    let mut slots = vec![num_complex::Complex64::new(0.0, 0.0); batch_count];

                    for (slot, value) in slots.iter_mut().enumerate() {
                        let source_batch = bit_reverse_index(slot, log_slots);
                        *value = num_complex::Complex64::new(matrices[source_batch][row][col], 0.0);
                    }

                    let coefficients = embedding.slots_to_coefficients(&slots);
                    quantize_real_coefficients(&coefficients, basis, scale)
                })
                .collect()
        })
        .collect()
}

/// Decrypts Batch CPMM large-ring output columns back into structural SinC
/// coefficient form.
///
/// This is the inverse cryptographic representation boundary used after Batch
/// CPMM execution. SinC slot decoding remains a separate structural step.
pub fn decode_cpmm_large_columns(
    ciphertexts: &[crate::ckks::RnsCkksCiphertext],
    secret: &[i8],
    scalar_degree: usize,
    half_rows: usize,
) -> SinCBatchPlaintext {
    use num_traits::ToPrimitive;

    assert!(
        scalar_degree > 0,
        "Batch CPMM scalar degree must be positive"
    );
    assert!(half_rows > 0, "Batch CPMM half-row count must be positive");
    assert!(
        !ciphertexts.is_empty(),
        "Batch CPMM output must contain at least one ciphertext column"
    );

    let large_degree = scalar_degree
        .checked_mul(half_rows)
        .expect("Batch CPMM decoded large-ring degree overflow");

    assert_eq!(
        secret.len(),
        large_degree,
        "Batch CPMM decoding secret degree must match the large ring"
    );

    let mut columns = Vec::with_capacity(ciphertexts.len());

    for ciphertext in ciphertexts {
        assert_eq!(
            ciphertext.rlwe().degree(),
            large_degree,
            "Batch CPMM output ciphertext degree must match the SinC geometry"
        );

        let plan = crate::ring::RnsNttPlan::new(
            ciphertext.basis().moduli().to_vec(),
            ciphertext.rlwe().degree(),
        );

        let plaintext = crate::grafting::decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);

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

    SinCBatchPlaintext {
        scalar_degree,
        dimension: half_rows,
        columns,
    }
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
        super::quantize_real_coefficients(coefficients, basis, scale)
    }

    fn encode_packed_real_matrix(
        matrices: &[Vec<Vec<f64>>],
        scalar_degree: usize,
        basis: &crate::ring::ModulusBasis,
        scale: f64,
    ) -> Vec<Vec<crate::ring::RnsPolynomial>> {
        super::encode_cpmm_real_batch(matrices, scalar_degree, basis, scale)
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

    /// NTT-resident ciphertext/plaintext matrix multiplication used by
    /// authors-style Batch CPMM.
    ///
    /// Input geometry:
    ///
    ///     lhs_split  : d x h ciphertext matrix
    ///     rhs_plain  : d x d plaintext matrix
    ///
    /// Output geometry:
    ///
    ///     h x d ciphertext matrix
    ///
    /// Every ciphertext component and plaintext entry is transformed exactly
    /// once. The cubic matrix-product loop then consists only of pointwise
    /// NTT-domain products and additions. Each output component is inverse
    /// transformed exactly once.
    fn batch_cpmm_modular_product_prepared_ntt(
        lhs_split: &[Vec<crate::grafting::RnsRlweCiphertext>],
        rhs_plain: &[Vec<crate::ring::RnsPolynomial>],
        plan: &crate::ring::PreparedRnsNttPlan,
    ) -> Vec<Vec<crate::grafting::RnsRlweCiphertext>> {
        assert!(!lhs_split.is_empty(), "Batch CPMM lhs must not be empty");
        assert!(!rhs_plain.is_empty(), "Batch CPMM rhs must not be empty");

        let inner = lhs_split.len();
        let output_rows = lhs_split[0].len();

        assert!(output_rows > 0);
        assert!(
            lhs_split.iter().all(|column| column.len() == output_rows),
            "Batch CPMM split ciphertext matrix must be rectangular"
        );

        assert_eq!(
            rhs_plain.len(),
            inner,
            "Batch CPMM inner dimensions must match"
        );

        let output_columns = rhs_plain[0].len();

        assert!(output_columns > 0);
        assert!(
            rhs_plain.iter().all(|row| row.len() == output_columns),
            "Batch CPMM plaintext matrix must be rectangular"
        );

        let kernel_start = std::time::Instant::now();

        // Transform the two RLWE components of every ciphertext entry once.
        let forward_start = std::time::Instant::now();

        let lhs_b_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = lhs_split
            .iter()
            .map(|column| {
                column
                    .iter()
                    .map(|ciphertext| {
                        let b = batch_ccmm_rns_component(ciphertext, true);
                        plan.forward(&b)
                    })
                    .collect()
            })
            .collect();

        let lhs_a_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = lhs_split
            .iter()
            .map(|column| {
                column
                    .iter()
                    .map(|ciphertext| {
                        let a = batch_ccmm_rns_component(ciphertext, false);
                        plan.forward(&a)
                    })
                    .collect()
            })
            .collect();

        // Transform every plaintext matrix entry once.
        let rhs_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = rhs_plain
            .iter()
            .map(|row| row.iter().map(|entry| plan.forward(entry)).collect())
            .collect();

        let forward_elapsed = forward_start.elapsed();
        let mut inverse_rebuild_elapsed = std::time::Duration::ZERO;

        let output: Vec<Vec<crate::grafting::RnsRlweCiphertext>> = (0..output_rows)
            .map(|row| {
                (0..output_columns)
                    .map(|column| {
                        let mut b_acc = lhs_b_ntt[0][row].pointwise_mul(&rhs_ntt[0][column]);

                        let mut a_acc = lhs_a_ntt[0][row].pointwise_mul(&rhs_ntt[0][column]);

                        for inner_index in 1..inner {
                            b_acc.pointwise_mul_add_assign(
                                &lhs_b_ntt[inner_index][row],
                                &rhs_ntt[inner_index][column],
                            );

                            a_acc.pointwise_mul_add_assign(
                                &lhs_a_ntt[inner_index][row],
                                &rhs_ntt[inner_index][column],
                            );
                        }

                        let inverse_start = std::time::Instant::now();

                        let b = plan.inverse(&b_acc);
                        let a = plan.inverse(&a_acc);

                        assert_eq!(b.basis(), a.basis());
                        assert_eq!(b.degree(), a.degree());

                        let ciphertext = crate::grafting::RnsRlweCiphertext::from_limbs(
                            b.residues()
                                .iter()
                                .zip(a.residues())
                                .map(|(b_limb, a_limb)| {
                                    crate::rlwe::RlweCiphertext::new(b_limb.clone(), a_limb.clone())
                                })
                                .collect(),
                        );

                        inverse_rebuild_elapsed += inverse_start.elapsed();

                        ciphertext
                    })
                    .collect()
            })
            .collect();

        let kernel_elapsed = kernel_start.elapsed();

        let ntt_mac_elapsed = kernel_elapsed
            .saturating_sub(forward_elapsed)
            .saturating_sub(inverse_rebuild_elapsed);

        println!(
            "BATCH_CPMM_PHASE_FORWARD_NTT_MS={:.3}",
            forward_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CPMM_PHASE_NTT_MAC_MS={:.3}",
            ntt_mac_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CPMM_PHASE_INVERSE_REBUILD_MS={:.3}",
            inverse_rebuild_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CPMM_PHASE_KERNEL_TOTAL_MS={:.3}",
            kernel_elapsed.as_secs_f64() * 1.0e3
        );

        output
    }

    /// Dense blocked layer-major CPMM.
    ///
    /// For each fixed (RNS limb, NTT coordinate), materialize ordinary
    /// modular matrices and execute a cache-oriented i-k-j blocked GEMM.
    ///
    /// Arithmetic is intentionally unchanged from the O4 path: canonical
    /// modular multiplication and addition only. This isolates the effect of
    /// dense storage and GEMM organization.
    fn batch_cpmm_modular_product_blocked_ntt(
        lhs_split: &[Vec<crate::grafting::RnsRlweCiphertext>],
        rhs_plain: &[Vec<crate::ring::RnsPolynomial>],
        plan: &crate::ring::PreparedRnsNttPlan,
    ) -> Vec<Vec<crate::grafting::RnsRlweCiphertext>> {
        const BLOCK: usize = 16;

        assert!(!lhs_split.is_empty(), "Batch CPMM lhs must not be empty");
        assert!(!rhs_plain.is_empty(), "Batch CPMM rhs must not be empty");

        let inner = lhs_split.len();
        let output_rows = lhs_split[0].len();

        assert!(output_rows > 0);
        assert!(
            lhs_split.iter().all(|column| column.len() == output_rows),
            "Batch CPMM split ciphertext matrix must be rectangular"
        );

        assert_eq!(
            rhs_plain.len(),
            inner,
            "Batch CPMM inner dimensions must match"
        );

        let output_columns = rhs_plain[0].len();

        assert!(output_columns > 0);
        assert!(
            rhs_plain.iter().all(|row| row.len() == output_columns),
            "Batch CPMM plaintext matrix must be rectangular"
        );

        let degree = plan.degree();
        let limb_count = plan.moduli().len();

        let kernel_start = std::time::Instant::now();

        // --------------------------------------------------------------
        // Transform every input exactly once.
        // --------------------------------------------------------------

        let forward_start = std::time::Instant::now();

        let lhs_b_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = lhs_split
            .iter()
            .map(|column| {
                column
                    .iter()
                    .map(|ciphertext| {
                        let b = batch_ccmm_rns_component(ciphertext, true);
                        plan.forward(&b)
                    })
                    .collect()
            })
            .collect();

        let lhs_a_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = lhs_split
            .iter()
            .map(|column| {
                column
                    .iter()
                    .map(|ciphertext| {
                        let a = batch_ccmm_rns_component(ciphertext, false);
                        plan.forward(&a)
                    })
                    .collect()
            })
            .collect();

        let rhs_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = rhs_plain
            .iter()
            .map(|row| row.iter().map(|entry| plan.forward(entry)).collect())
            .collect();

        let forward_elapsed = forward_start.elapsed();

        // Output NTT values:
        //
        // [limb][coordinate][row * output_columns + column]
        let layer_size = output_rows * output_columns;

        let mut out_b = vec![vec![vec![0_u64; layer_size]; degree]; limb_count];

        let mut out_a = vec![vec![vec![0_u64; layer_size]; degree]; limb_count];

        // --------------------------------------------------------------
        // Dense modular GEMM per (limb, NTT coordinate).
        // --------------------------------------------------------------

        let mut materialize_elapsed = std::time::Duration::ZERO;
        let mut blocked_gemm_elapsed = std::time::Duration::ZERO;

        for limb_index in 0..limb_count {
            let modulus = plan.moduli()[limb_index];

            for coordinate in 0..degree {
                let materialize_start = std::time::Instant::now();

                // Flatten this scalar layer into contiguous dense matrices.
                //
                // lhs_*: output_rows x inner
                // rhs:   inner x output_columns
                let mut lhs_b_layer = vec![0_u64; output_rows * inner];
                let mut lhs_a_layer = vec![0_u64; output_rows * inner];
                let mut rhs_layer = vec![0_u64; inner * output_columns];

                for row in 0..output_rows {
                    for k in 0..inner {
                        lhs_b_layer[row * inner + k] =
                            lhs_b_ntt[k][row].residue(limb_index).values()[coordinate];

                        lhs_a_layer[row * inner + k] =
                            lhs_a_ntt[k][row].residue(limb_index).values()[coordinate];
                    }
                }

                for k in 0..inner {
                    for column in 0..output_columns {
                        rhs_layer[k * output_columns + column] =
                            rhs_ntt[k][column].residue(limb_index).values()[coordinate];
                    }
                }

                materialize_elapsed += materialize_start.elapsed();

                let c_b = &mut out_b[limb_index][coordinate];
                let c_a = &mut out_a[limb_index][coordinate];

                let gemm_start = std::time::Instant::now();

                // O5B: delayed modular reduction.
                //
                // For canonical residues:
                //
                //     product < q^2
                //     dot product < inner * q^2
                //
                // Verify that the complete dot product fits in u128, then
                // accumulate with ordinary wide integer arithmetic and reduce
                // exactly once per output entry.
                let q = u128::from(modulus.value());
                let max_product = (q - 1) * (q - 1);

                max_product
                    .checked_mul(inner as u128)
                    .expect("CPMM delayed-reduction dot product must fit in u128");

                let mut wide_b = vec![0_u128; layer_size];
                let mut wide_a = vec![0_u128; layer_size];

                // Blocked i-k-j traversal.
                //
                // No modular reduction occurs in the cubic loop.
                for ii in (0..output_rows).step_by(BLOCK) {
                    let i_end = (ii + BLOCK).min(output_rows);

                    for kk in (0..inner).step_by(BLOCK) {
                        let k_end = (kk + BLOCK).min(inner);

                        for jj in (0..output_columns).step_by(BLOCK) {
                            let j_end = (jj + BLOCK).min(output_columns);

                            for row in ii..i_end {
                                for k in kk..k_end {
                                    let lhs_b_value = u128::from(lhs_b_layer[row * inner + k]);

                                    let lhs_a_value = u128::from(lhs_a_layer[row * inner + k]);

                                    let rhs_base = k * output_columns;
                                    let out_base = row * output_columns;

                                    for column in jj..j_end {
                                        let rhs_value = u128::from(rhs_layer[rhs_base + column]);

                                        wide_b[out_base + column] += lhs_b_value * rhs_value;

                                        wide_a[out_base + column] += lhs_a_value * rhs_value;
                                    }
                                }
                            }
                        }
                    }
                }

                // One exact modular reduction per output entry.
                for index in 0..layer_size {
                    c_b[index] = (wide_b[index] % q) as u64;
                    c_a[index] = (wide_a[index] % q) as u64;
                }

                blocked_gemm_elapsed += gemm_start.elapsed();
            }
        }

        // --------------------------------------------------------------
        // Reconstruct one RNS NTT polynomial per output component.
        // --------------------------------------------------------------

        let reconstruct_start = std::time::Instant::now();

        let output: Vec<Vec<crate::grafting::RnsRlweCiphertext>> = (0..output_rows)
            .map(|row| {
                (0..output_columns)
                    .map(|column| {
                        let mut b_residues = Vec::with_capacity(limb_count);
                        let mut a_residues = Vec::with_capacity(limb_count);

                        for limb_index in 0..limb_count {
                            let limb_plan = plan.plan(limb_index);

                            let b_values: Vec<u64> = (0..degree)
                                .map(|coordinate| {
                                    out_b[limb_index][coordinate][row * output_columns + column]
                                })
                                .collect();

                            let a_values: Vec<u64> = (0..degree)
                                .map(|coordinate| {
                                    out_a[limb_index][coordinate][row * output_columns + column]
                                })
                                .collect();

                            b_residues.push(crate::ring::NttPolynomial::from_prepared_values(
                                limb_plan, b_values,
                            ));

                            a_residues.push(crate::ring::NttPolynomial::from_prepared_values(
                                limb_plan, a_values,
                            ));
                        }

                        let b_ntt = crate::ring::RnsNttPolynomial::from_residues(b_residues);

                        let a_ntt = crate::ring::RnsNttPolynomial::from_residues(a_residues);

                        let b = plan.inverse(&b_ntt);
                        let a = plan.inverse(&a_ntt);

                        crate::grafting::RnsRlweCiphertext::from_limbs(
                            b.residues()
                                .iter()
                                .zip(a.residues())
                                .map(|(b_limb, a_limb)| {
                                    crate::rlwe::RlweCiphertext::new(b_limb.clone(), a_limb.clone())
                                })
                                .collect(),
                        )
                    })
                    .collect()
            })
            .collect();

        let reconstruct_inverse_elapsed = reconstruct_start.elapsed();
        let kernel_elapsed = kernel_start.elapsed();

        println!(
            "BATCH_CPMM_O5_PHASE_FORWARD_NTT_MS={:.3}",
            forward_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CPMM_O5_PHASE_LAYER_MATERIALIZE_MS={:.3}",
            materialize_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CPMM_O5_PHASE_BLOCKED_GEMM_MS={:.3}",
            blocked_gemm_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CPMM_O5_PHASE_RECONSTRUCT_INVERSE_MS={:.3}",
            reconstruct_inverse_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CPMM_O5_PHASE_KERNEL_TOTAL_MS={:.3}",
            kernel_elapsed.as_secs_f64() * 1.0e3
        );

        output
    }

    /// HEaaN-style layer-major CPMM.
    ///
    /// This is mathematically identical to
    /// `batch_cpmm_modular_product_prepared_ntt`, but reorganizes the
    /// NTT-domain matrix product as:
    ///
    ///     RNS limb -> NTT coordinate -> dense modular GEMM
    ///
    /// rather than:
    ///
    ///     matrix row -> matrix column -> polynomial pointwise operation.
    ///
    /// The scalar modular arithmetic remains the portable `Modulus` API.
    fn batch_cpmm_modular_product_layer_major(
        lhs_split: &[Vec<crate::grafting::RnsRlweCiphertext>],
        rhs_plain: &[Vec<crate::ring::RnsPolynomial>],
        plan: &crate::ring::PreparedRnsNttPlan,
    ) -> Vec<Vec<crate::grafting::RnsRlweCiphertext>> {
        assert!(!lhs_split.is_empty(), "Batch CPMM lhs must not be empty");
        assert!(!rhs_plain.is_empty(), "Batch CPMM rhs must not be empty");

        let inner = lhs_split.len();
        let output_rows = lhs_split[0].len();

        assert!(output_rows > 0);
        assert!(
            lhs_split.iter().all(|column| column.len() == output_rows),
            "Batch CPMM split ciphertext matrix must be rectangular"
        );

        assert_eq!(
            rhs_plain.len(),
            inner,
            "Batch CPMM inner dimensions must match"
        );

        let output_columns = rhs_plain[0].len();

        assert!(output_columns > 0);
        assert!(
            rhs_plain.iter().all(|row| row.len() == output_columns),
            "Batch CPMM plaintext matrix must be rectangular"
        );

        let degree = plan.degree();
        let limb_count = plan.moduli().len();

        // ------------------------------------------------------------------
        // Transform every input exactly once.
        // ------------------------------------------------------------------

        let lhs_b_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = lhs_split
            .iter()
            .map(|column| {
                column
                    .iter()
                    .map(|ciphertext| {
                        let b = batch_ccmm_rns_component(ciphertext, true);
                        plan.forward(&b)
                    })
                    .collect()
            })
            .collect();

        let lhs_a_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = lhs_split
            .iter()
            .map(|column| {
                column
                    .iter()
                    .map(|ciphertext| {
                        let a = batch_ccmm_rns_component(ciphertext, false);
                        plan.forward(&a)
                    })
                    .collect()
            })
            .collect();

        let rhs_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = rhs_plain
            .iter()
            .map(|row| row.iter().map(|entry| plan.forward(entry)).collect())
            .collect();

        // Output storage:
        //
        // [row][column][limb][degree]
        let mut out_b =
            vec![vec![vec![vec![0_u64; degree]; limb_count]; output_columns]; output_rows];

        let mut out_a =
            vec![vec![vec![vec![0_u64; degree]; limb_count]; output_columns]; output_rows];

        // ------------------------------------------------------------------
        // HEaaN-style execution order:
        //
        //   limb
        //     -> NTT coordinate
        //         -> dense modular matrix multiply
        //
        // For each fixed (limb, coordinate), the encrypted polynomial
        // problem is now an ordinary modular matrix product.
        // ------------------------------------------------------------------

        for limb_index in 0..limb_count {
            let modulus = plan.moduli()[limb_index];

            for coordinate in 0..degree {
                for row in 0..output_rows {
                    for column in 0..output_columns {
                        let mut b_acc = 0_u64;
                        let mut a_acc = 0_u64;

                        for inner_index in 0..inner {
                            let rhs_value = rhs_ntt[inner_index][column]
                                .residue(limb_index)
                                .values()[coordinate];

                            let lhs_b_value = lhs_b_ntt[inner_index][row]
                                .residue(limb_index)
                                .values()[coordinate];

                            let lhs_a_value = lhs_a_ntt[inner_index][row]
                                .residue(limb_index)
                                .values()[coordinate];

                            b_acc =
                                modulus.add_canonical(b_acc, modulus.mul(lhs_b_value, rhs_value));

                            a_acc =
                                modulus.add_canonical(a_acc, modulus.mul(lhs_a_value, rhs_value));
                        }

                        out_b[row][column][limb_index][coordinate] = b_acc;
                        out_a[row][column][limb_index][coordinate] = a_acc;
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Rebuild RNS NTT polynomials, inverse transform once/output, then
        // reconstruct RLWE ciphertexts.
        // ------------------------------------------------------------------

        (0..output_rows)
            .map(|row| {
                (0..output_columns)
                    .map(|column| {
                        let b_ntt = crate::ring::RnsNttPolynomial::from_residues(
                            (0..limb_count)
                                .map(|limb_index| {
                                    crate::ring::NttPolynomial::from_prepared_values(
                                        plan.plan(limb_index),
                                        out_b[row][column][limb_index].clone(),
                                    )
                                })
                                .collect(),
                        );

                        let a_ntt = crate::ring::RnsNttPolynomial::from_residues(
                            (0..limb_count)
                                .map(|limb_index| {
                                    crate::ring::NttPolynomial::from_prepared_values(
                                        plan.plan(limb_index),
                                        out_a[row][column][limb_index].clone(),
                                    )
                                })
                                .collect(),
                        );

                        let b = plan.inverse(&b_ntt);
                        let a = plan.inverse(&a_ntt);

                        crate::grafting::RnsRlweCiphertext::from_limbs(
                            b.residues()
                                .iter()
                                .zip(a.residues())
                                .map(|(b_limb, a_limb)| {
                                    crate::rlwe::RlweCiphertext::new(b_limb.clone(), a_limb.clone())
                                })
                                .collect(),
                        )
                    })
                    .collect()
            })
            .collect()
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
        let normalized = i128::from(exponent).rem_euclid(period) as usize;

        let shift = normalized % degree;
        let sign = if normalized >= degree { -1_i8 } else { 1_i8 };

        polynomial.mul_monomial_signed(shift, sign)
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

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct BatchRnsNttRlweCiphertext {
        b: crate::ring::RnsNttPolynomial,
        a: crate::ring::RnsNttPolynomial,
    }

    impl BatchRnsNttRlweCiphertext {
        fn from_coefficient(
            ciphertext: &crate::grafting::RnsRlweCiphertext,
            plan: &crate::ring::RnsNttPlan,
        ) -> Self {
            let b = batch_ccmm_rns_component(ciphertext, true);
            let a = batch_ccmm_rns_component(ciphertext, false);

            Self {
                b: plan.forward(&b),
                a: plan.forward(&a),
            }
        }

        fn to_coefficient(
            &self,
            plan: &crate::ring::RnsNttPlan,
        ) -> crate::grafting::RnsRlweCiphertext {
            let b = plan.inverse(&self.b);
            let a = plan.inverse(&self.a);

            assert_eq!(b.basis(), a.basis());
            assert_eq!(b.degree(), a.degree());

            crate::grafting::RnsRlweCiphertext::from_limbs(
                b.residues()
                    .iter()
                    .zip(a.residues())
                    .map(|(b_limb, a_limb)| {
                        crate::rlwe::RlweCiphertext::new(b_limb.clone(), a_limb.clone())
                    })
                    .collect(),
            )
        }

        fn add(&self, rhs: &Self) -> Self {
            Self {
                b: self.b.add(&rhs.b),
                a: self.a.add(&rhs.a),
            }
        }

        fn sub(&self, rhs: &Self) -> Self {
            Self {
                b: self.b.sub(&rhs.b),
                a: self.a.sub(&rhs.a),
            }
        }

        fn mul_monomial_signed(&self, plan: &crate::ring::RnsNttPlan, exponent: i64) -> Self {
            Self {
                b: self.b.mul_monomial_signed(plan, exponent),
                a: self.a.mul_monomial_signed(plan, exponent),
            }
        }
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

    /// Computes both branches of a negacyclic monomial butterfly in one
    /// coefficient traversal:
    ///
    ///     sum  = even + X^exponent * odd
    ///     diff = even - X^exponent * odd
    ///
    /// This avoids materializing the intermediate monomial product and
    /// avoids separate add/sub traversals.
    fn batch_polynomial_butterfly_monomial(
        even: &crate::ring::Polynomial,
        odd: &crate::ring::Polynomial,
        exponent: i64,
    ) -> (crate::ring::Polynomial, crate::ring::Polynomial) {
        assert_eq!(
            even.modulus(),
            odd.modulus(),
            "Batch polynomial butterfly moduli must match"
        );

        assert_eq!(
            even.degree(),
            odd.degree(),
            "Batch polynomial butterfly degrees must match"
        );

        let modulus = even.modulus();
        let degree = even.degree();

        let period = 2_i128 * degree as i128;

        let normalized = i128::from(exponent).rem_euclid(period) as usize;

        let shift = normalized % degree;

        let global_negative = normalized >= degree;

        let mut sum = Vec::with_capacity(degree);

        let mut diff = Vec::with_capacity(degree);

        for output_index in 0..degree {
            let (source_index, wrapped) = if output_index >= shift {
                (output_index - shift, false)
            } else {
                (output_index + degree - shift, true)
            };

            let mut twiddled = odd.coefficients()[source_index];

            if wrapped ^ global_negative {
                twiddled = modulus.neg(twiddled);
            }

            let even_value = even.coefficients()[output_index];

            sum.push(modulus.add_canonical(even_value, twiddled));

            diff.push(modulus.sub_canonical(even_value, twiddled));
        }

        (
            crate::ring::Polynomial::new(modulus, sum),
            crate::ring::Polynomial::new(modulus, diff),
        )
    }

    fn batch_rlwe_butterfly_monomial(
        even: &crate::grafting::RnsRlweCiphertext,
        odd: &crate::grafting::RnsRlweCiphertext,
        exponent: i64,
    ) -> (
        crate::grafting::RnsRlweCiphertext,
        crate::grafting::RnsRlweCiphertext,
    ) {
        assert_eq!(
            even.basis(),
            odd.basis(),
            "Batch RLWE butterfly bases must match"
        );

        assert_eq!(
            even.degree(),
            odd.degree(),
            "Batch RLWE butterfly degrees must match"
        );

        let mut upper_limbs = Vec::with_capacity(even.limbs().len());

        let mut lower_limbs = Vec::with_capacity(even.limbs().len());

        for (even_limb, odd_limb) in even.limbs().iter().zip(odd.limbs()) {
            let (upper_b, lower_b) =
                batch_polynomial_butterfly_monomial(even_limb.b(), odd_limb.b(), exponent);

            let (upper_a, lower_a) =
                batch_polynomial_butterfly_monomial(even_limb.a(), odd_limb.a(), exponent);

            upper_limbs.push(crate::rlwe::RlweCiphertext::new(upper_b, upper_a));

            lower_limbs.push(crate::rlwe::RlweCiphertext::new(lower_b, lower_a));
        }

        (
            crate::grafting::RnsRlweCiphertext::from_limbs(upper_limbs),
            crate::grafting::RnsRlweCiphertext::from_limbs(lower_limbs),
        )
    }

    fn batch_bit_reverse_permute<T>(values: &mut [T]) {
        let n = values.len();

        assert!(
            n.is_power_of_two(),
            "Batch bit reversal requires power-of-two length"
        );

        let mut j = 0_usize;

        for i in 1..n {
            let mut bit = n >> 1;

            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }

            j ^= bit;

            if i < j {
                values.swap(i, j);
            }
        }
    }

    fn batch_large_dft_iterative(
        input: &[crate::grafting::RnsRlweCiphertext],
        multiplier: i64,
    ) -> Vec<crate::grafting::RnsRlweCiphertext> {
        let dimension = input.len();

        assert!(
            dimension.is_power_of_two(),
            "Batch iterative large DFT dimension must be a power of two"
        );

        assert!(
            !input.is_empty(),
            "Batch iterative large DFT requires input"
        );

        let mut output = input.to_vec();

        batch_bit_reverse_permute(&mut output);

        let mut len = 2_usize;

        while len <= dimension {
            let half = len / 2;

            // Recursive implementation doubles the multiplier at each
            // descent. In iterative order, stage len therefore uses:
            //
            //     multiplier * (dimension / len)
            //
            // as the stage base exponent.
            let stage_multiplier = multiplier * (dimension / len) as i64;

            for start in (0..dimension).step_by(len) {
                for k in 0..half {
                    let upper_index = start + k;
                    let lower_index = start + k + half;

                    let (upper, lower) = batch_rlwe_butterfly_monomial(
                        &output[upper_index],
                        &output[lower_index],
                        stage_multiplier * k as i64,
                    );

                    output[upper_index] = upper;
                    output[lower_index] = lower;
                }
            }

            len *= 2;
        }

        output
    }

    fn batch_large_dft_ntt(
        input: &[BatchRnsNttRlweCiphertext],
        multiplier: i64,
        plan: &crate::ring::RnsNttPlan,
    ) -> Vec<BatchRnsNttRlweCiphertext> {
        let dimension = input.len();

        assert!(
            dimension.is_power_of_two(),
            "Batch NTT-domain large DFT dimension must be a power of two"
        );

        assert!(
            !input.is_empty(),
            "Batch NTT-domain large DFT requires input"
        );

        if dimension == 1 {
            return vec![input[0].clone()];
        }

        let half = dimension / 2;

        let even: Vec<_> = (0..half).map(|i| input[2 * i].clone()).collect();

        let odd: Vec<_> = (0..half).map(|i| input[2 * i + 1].clone()).collect();

        let even = batch_large_dft_ntt(&even, multiplier * 2, plan);

        let odd = batch_large_dft_ntt(&odd, multiplier * 2, plan);

        let mut output = Vec::with_capacity(dimension);

        output.resize_with(dimension, || input[0].clone());

        for k in 0..half {
            let twiddled = odd[k].mul_monomial_signed(plan, multiplier * k as i64);

            output[k] = even[k].add(&twiddled);

            output[k + half] = even[k].sub(&twiddled);
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

    fn batch_large_inv_crt_ntt(
        ciphertexts: &[BatchRnsNttRlweCiphertext],
        scalar_degree: usize,
        plan: &crate::ring::RnsNttPlan,
    ) -> Vec<BatchRnsNttRlweCiphertext> {
        let dimension = ciphertexts.len();

        assert!(!ciphertexts.is_empty());
        assert!(dimension.is_power_of_two());

        let large_degree = plan.degree();

        assert_eq!(
            scalar_degree * dimension,
            large_degree,
            "Batch NTT-domain inverse CRT geometry must satisfy Ns * d = N"
        );

        batch_large_dft_ntt(ciphertexts, -((2 * scalar_degree) as i64), plan)
            .into_iter()
            .enumerate()
            .map(|(column, ciphertext)| ciphertext.mul_monomial_signed(plan, -(column as i64)))
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

    fn batch_prepared_galois_key_for_exponent(
        keys: &[crate::ckks::PreparedRnsGaloisKey],
        exponent: usize,
    ) -> &crate::ckks::PreparedRnsGaloisKey {
        keys.iter()
            .find(|key| key.exponent() == exponent)
            .unwrap_or_else(|| {
                panic!("missing prepared Batch C-MT RNS Galois key for exponent {exponent}")
            })
    }

    fn batch_scrambled_auto(
        ciphertexts: &[crate::grafting::RnsRlweCiphertext],
        scalar_degree: usize,
        galois_keys: &[crate::ckks::PreparedRnsGaloisKey],
        plan: &crate::ring::RnsNttPlan,
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

        let mut automorphism_seconds = 0.0_f64;

        let mut decompose_seconds = 0.0_f64;
        let mut base_forward_seconds = 0.0_f64;
        let mut digit_prepare_seconds = 0.0_f64;
        let mut digit_forward_seconds = 0.0_f64;
        let mut mac_seconds = 0.0_f64;
        let mut inverse_seconds = 0.0_f64;

        let mut output = Vec::with_capacity(dimension);

        for index in 0..dimension {
            let exponent = 2 * scalar_degree * index + 1;
            let inverse = batch_mod_inverse_odd(exponent, two_n);

            assert_eq!(
                (inverse - 1) % (2 * scalar_degree),
                0,
                "inverse Batch automorphism exponent must remain in subgroup"
            );

            let inverse_index = (inverse - 1) / (2 * scalar_degree);

            if exponent == 1 {
                output.push(ciphertexts[inverse_index].clone());
            } else {
                let key = batch_prepared_galois_key_for_exponent(galois_keys, exponent);

                let (transformed, automorphism, profile) =
                    crate::ckks::apply_rns_galois_automorphism_with_prepared_ntt_profiled(
                        &ciphertexts[inverse_index],
                        key,
                        plan,
                    );

                automorphism_seconds += automorphism;

                decompose_seconds += profile.decompose_seconds;
                base_forward_seconds += profile.base_forward_seconds;
                digit_prepare_seconds += profile.digit_prepare_seconds;
                digit_forward_seconds += profile.digit_forward_seconds;
                mac_seconds += profile.mac_seconds;
                inverse_seconds += profile.inverse_seconds;

                output.push(transformed);
            }
        }

        let accounted_key_switch_seconds = decompose_seconds
            + base_forward_seconds
            + digit_prepare_seconds
            + digit_forward_seconds
            + mac_seconds
            + inverse_seconds;

        println!(
            "BATCH_GALOIS_PHASE_AUTOMORPHISM_MS={:.3}",
            automorphism_seconds * 1.0e3
        );

        println!(
            "BATCH_KEY_SWITCH_PHASE_DECOMPOSE_MS={:.3}",
            decompose_seconds * 1.0e3
        );

        println!(
            "BATCH_KEY_SWITCH_PHASE_BASE_FORWARD_MS={:.3}",
            base_forward_seconds * 1.0e3
        );

        println!(
            "BATCH_KEY_SWITCH_PHASE_DIGIT_PREPARE_MS={:.3}",
            digit_prepare_seconds * 1.0e3
        );

        println!(
            "BATCH_KEY_SWITCH_PHASE_DIGIT_FORWARD_MS={:.3}",
            digit_forward_seconds * 1.0e3
        );

        println!("BATCH_KEY_SWITCH_PHASE_MAC_MS={:.3}", mac_seconds * 1.0e3);

        println!(
            "BATCH_KEY_SWITCH_PHASE_INVERSE_MS={:.3}",
            inverse_seconds * 1.0e3
        );

        println!(
            "BATCH_GALOIS_PHASE_KEY_SWITCH_ACCOUNTED_MS={:.3}",
            accounted_key_switch_seconds * 1.0e3
        );

        output
    }

    fn batch_ciphertext_transpose(
        ciphertexts: &[crate::grafting::RnsRlweCiphertext],
        scalar_degree: usize,
        galois_keys: &[crate::ckks::PreparedRnsGaloisKey],
        plan: &crate::ring::RnsNttPlan,
    ) -> Vec<crate::grafting::RnsRlweCiphertext> {
        let total_start = std::time::Instant::now();

        let phase_start = std::time::Instant::now();
        let crt = batch_large_crt(ciphertexts, scalar_degree);
        let crt_elapsed = phase_start.elapsed();

        let phase_start = std::time::Instant::now();
        let transformed = batch_scrambled_auto(&crt, scalar_degree, galois_keys, plan);
        let scrambled_auto_elapsed = phase_start.elapsed();

        let phase_start = std::time::Instant::now();
        let output = batch_large_inv_crt(&transformed, scalar_degree);
        let inv_crt_elapsed = phase_start.elapsed();

        let total_elapsed = total_start.elapsed();

        println!(
            "BATCH_CMT_LARGE_CRT_MS={:.3}",
            crt_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CMT_SCRAMBLED_AUTO_MS={:.3}",
            scrambled_auto_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CMT_LARGE_INV_CRT_MS={:.3}",
            inv_crt_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CMT_TOTAL_MS={:.3}",
            total_elapsed.as_secs_f64() * 1.0e3
        );

        output
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

    fn batch_ccmm_slice_ranked_resident_ntt(
        ciphertexts: &[crate::grafting::RnsNttRlweCiphertext],
        dimension: usize,
        large_plan: &crate::ring::RnsNttPlan,
        scalar_plan: &crate::ring::RnsNttPlan,
    ) -> Vec<Vec<crate::ring::RnsNttPolynomial>> {
        assert!(!ciphertexts.is_empty());
        assert!(dimension > 0);

        let sliced: Vec<_> = ciphertexts
            .iter()
            .map(|ciphertext| {
                assert_eq!(
                    ciphertext.degree(),
                    large_plan.degree(),
                    "resident Batch CCMM ciphertext degree must match large NTT plan"
                );

                assert_eq!(
                    ciphertext.moduli(),
                    large_plan.moduli(),
                    "resident Batch CCMM ciphertext basis must match large NTT plan"
                );

                let b_rows = batch_large_ntt_to_scalar_ntt_radix2(
                    ciphertext.b(),
                    dimension,
                    large_plan,
                    scalar_plan,
                );

                let a_rows = batch_large_ntt_to_scalar_ntt_radix2(
                    ciphertext.a(),
                    dimension,
                    large_plan,
                    scalar_plan,
                );

                assert_eq!(b_rows.len(), dimension);
                assert_eq!(a_rows.len(), dimension);

                (b_rows, a_rows)
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
    fn batch_ccmm_modular_product_ntt(
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

        // Transform each matrix entry exactly once.
        let lhs_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = lhs
            .iter()
            .map(|row| row.iter().map(|entry| plan.forward(entry)).collect())
            .collect();

        let rhs_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = rhs_transposed
            .iter()
            .map(|row| row.iter().map(|entry| plan.forward(entry)).collect())
            .collect();

        (0..lhs_ntt.len())
            .map(|row| {
                (0..output_columns)
                    .map(|column| {
                        let mut accumulator = lhs_ntt[row][0].pointwise_mul(&rhs_ntt[0][column]);

                        for (lhs_entry, rhs_row) in lhs_ntt[row].iter().zip(rhs_ntt.iter()).skip(1)
                        {
                            let product = lhs_entry.pointwise_mul(&rhs_row[column]);

                            accumulator = accumulator.add(&product);
                        }

                        // One inverse NTT per output polynomial.
                        plan.inverse(&accumulator)
                    })
                    .collect()
            })
            .collect()
    }

    /// Dense layer-major CCMM modular product with delayed reduction.
    ///
    /// For each fixed (RNS limb, NTT coordinate), the polynomial matrix
    /// multiplication becomes an ordinary dense modular matrix product.
    /// Complete dot products are accumulated in u128 and reduced exactly
    /// once per output entry when the conservative worst-case bound fits.
    ///
    /// This is the CCMM analogue of the CPMM O5B kernel.
    fn batch_ccmm_modular_product_delayed_ntt(
        lhs: &[Vec<crate::ring::RnsPolynomial>],
        rhs_transposed: &[Vec<crate::ring::RnsPolynomial>],
        plan: &crate::ring::PreparedRnsNttPlan,
    ) -> Vec<Vec<crate::ring::RnsPolynomial>> {
        let block = std::env::var("CCMM_GEMM_BLOCK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(128);

        assert!(block > 0, "CCMM GEMM block size must be nonzero");

        assert!(!lhs.is_empty(), "Batch CCMM lhs must not be empty");
        assert!(
            !rhs_transposed.is_empty(),
            "Batch CCMM rhs must not be empty"
        );

        let output_rows = lhs.len();
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

        let degree = plan.degree();
        let limb_count = plan.moduli().len();

        // Transform each polynomial matrix entry exactly once.
        let forward_start = std::time::Instant::now();

        let lhs_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = lhs
            .iter()
            .map(|row| row.iter().map(|entry| plan.forward(entry)).collect())
            .collect();

        let rhs_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = rhs_transposed
            .iter()
            .map(|row| row.iter().map(|entry| plan.forward(entry)).collect())
            .collect();

        let forward_elapsed = forward_start.elapsed();

        let output_layer_size = output_rows * output_columns;

        // [limb][coordinate][row * output_columns + column]
        let mut output_layers = vec![vec![vec![0_u64; output_layer_size]; degree]; limb_count];

        let mut materialize_elapsed = std::time::Duration::ZERO;
        let mut gemm_elapsed = std::time::Duration::ZERO;

        let mut limb_materialize_elapsed = vec![std::time::Duration::ZERO; limb_count];

        let mut limb_gemm_elapsed = vec![std::time::Duration::ZERO; limb_count];

        let mut accumulate_elapsed = std::time::Duration::ZERO;

        let mut reduction_elapsed = std::time::Duration::ZERO;

        let mut limb_accumulate_elapsed = vec![std::time::Duration::ZERO; limb_count];

        let mut limb_reduction_elapsed = vec![std::time::Duration::ZERO; limb_count];

        for (limb_index, output_limb_layers) in
            output_layers.iter_mut().enumerate().take(limb_count)
        {
            let modulus = plan.moduli()[limb_index];
            let q = u128::from(modulus.value());

            let max_product = (q - 1) * (q - 1);

            max_product
                .checked_mul(inner as u128)
                .expect("CCMM delayed-reduction dot product must fit in u128");

            for (coordinate, output_layer) in output_limb_layers.iter_mut().enumerate().take(degree)
            {
                let materialize_start = std::time::Instant::now();

                let mut lhs_layer = vec![0_u64; output_rows * inner];

                let mut rhs_layer = vec![0_u64; inner * output_columns];

                for row in 0..output_rows {
                    for k in 0..inner {
                        lhs_layer[row * inner + k] =
                            lhs_ntt[row][k].residue(limb_index).values()[coordinate];
                    }
                }

                for k in 0..inner {
                    for column in 0..output_columns {
                        rhs_layer[k * output_columns + column] =
                            rhs_ntt[k][column].residue(limb_index).values()[coordinate];
                    }
                }

                let materialize_coordinate_elapsed = materialize_start.elapsed();

                materialize_elapsed += materialize_coordinate_elapsed;

                limb_materialize_elapsed[limb_index] += materialize_coordinate_elapsed;

                let mut wide = vec![0_u128; output_layer_size];

                let gemm_start = std::time::Instant::now();

                let accumulate_start = std::time::Instant::now();

                // Dense blocked i-k-j multiplication with no modular
                // reduction in the cubic loop.
                for ii in (0..output_rows).step_by(block) {
                    let i_end = (ii + block).min(output_rows);

                    for kk in (0..inner).step_by(block) {
                        let k_end = (kk + block).min(inner);

                        for jj in (0..output_columns).step_by(block) {
                            let j_end = (jj + block).min(output_columns);

                            for row in ii..i_end {
                                let lhs_base = row * inner;

                                let out_base = row * output_columns;

                                for k in kk..k_end {
                                    let lhs_value = u128::from(lhs_layer[lhs_base + k]);

                                    let rhs_base = k * output_columns;

                                    for column in jj..j_end {
                                        let rhs_value = u128::from(rhs_layer[rhs_base + column]);

                                        wide[out_base + column] += lhs_value * rhs_value;
                                    }
                                }
                            }
                        }
                    }
                }

                let accumulate_coordinate_elapsed = accumulate_start.elapsed();

                accumulate_elapsed += accumulate_coordinate_elapsed;

                limb_accumulate_elapsed[limb_index] += accumulate_coordinate_elapsed;

                let reduction_start = std::time::Instant::now();

                // One exact reduction per output matrix entry.
                for index in 0..output_layer_size {
                    output_layer[index] = (wide[index] % q) as u64;
                }

                let reduction_coordinate_elapsed = reduction_start.elapsed();

                reduction_elapsed += reduction_coordinate_elapsed;

                limb_reduction_elapsed[limb_index] += reduction_coordinate_elapsed;

                let gemm_coordinate_elapsed = gemm_start.elapsed();

                gemm_elapsed += gemm_coordinate_elapsed;

                limb_gemm_elapsed[limb_index] += gemm_coordinate_elapsed;
            }
        }

        for limb_index in 0..limb_count {
            println!(
                "BATCH_CCMM_DELAYED_LIMB_{}_MATERIALIZE_MS={:.3}",
                limb_index,
                limb_materialize_elapsed[limb_index].as_secs_f64() * 1.0e3
            );

            println!(
                "BATCH_CCMM_DELAYED_LIMB_{}_GEMM_MS={:.3}",
                limb_index,
                limb_gemm_elapsed[limb_index].as_secs_f64() * 1.0e3
            );

            println!(
                "BATCH_CCMM_DELAYED_LIMB_{}_ACCUMULATE_MS={:.3}",
                limb_index,
                limb_accumulate_elapsed[limb_index].as_secs_f64() * 1.0e3
            );

            println!(
                "BATCH_CCMM_DELAYED_LIMB_{}_REDUCTION_MS={:.3}",
                limb_index,
                limb_reduction_elapsed[limb_index].as_secs_f64() * 1.0e3
            );
        }

        let assemble_start = std::time::Instant::now();

        let output_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = (0..output_rows)
            .map(|row| {
                (0..output_columns)
                    .map(|column| {
                        let residues = (0..limb_count)
                            .map(|limb_index| {
                                let values: Vec<u64> = (0..degree)
                                    .map(|coordinate| {
                                        output_layers[limb_index][coordinate]
                                            [row * output_columns + column]
                                    })
                                    .collect();

                                crate::ring::NttPolynomial::from_prepared_values(
                                    plan.plan(limb_index),
                                    values,
                                )
                            })
                            .collect();

                        crate::ring::RnsNttPolynomial::from_residues(residues)
                    })
                    .collect()
            })
            .collect();

        let assemble_elapsed = assemble_start.elapsed();

        let inverse_start = std::time::Instant::now();

        let output: Vec<Vec<crate::ring::RnsPolynomial>> = output_ntt
            .iter()
            .map(|row| row.iter().map(|ntt| plan.inverse(ntt)).collect())
            .collect();

        let inverse_elapsed = inverse_start.elapsed();

        let reconstruct_elapsed = assemble_elapsed + inverse_elapsed;

        println!(
            "BATCH_CCMM_DELAYED_PHASE_FORWARD_NTT_MS={:.3}",
            forward_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_DELAYED_PHASE_LAYER_MATERIALIZE_MS={:.3}",
            materialize_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_DELAYED_PHASE_ACCUMULATE_MS={:.3}",
            accumulate_elapsed.as_secs_f64() * 1.0e3
        );

        println!(
            "BATCH_CCMM_DELAYED_PHASE_REDUCTION_MS={:.3}",
            reduction_elapsed.as_secs_f64() * 1.0e3
        );

        println!(
            "BATCH_CCMM_DELAYED_PHASE_GEMM_MS={:.3}",
            gemm_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_DELAYED_PHASE_ASSEMBLE_NTT_MS={:.3}",
            assemble_elapsed.as_secs_f64() * 1.0e3
        );

        println!(
            "BATCH_CCMM_DELAYED_PHASE_INVERSE_NTT_MS={:.3}",
            inverse_elapsed.as_secs_f64() * 1.0e3
        );

        println!(
            "BATCH_CCMM_DELAYED_PHASE_RECONSTRUCT_INVERSE_MS={:.3}",
            reconstruct_elapsed.as_secs_f64() * 1.0e3
        );

        output
    }

    fn batch_ccmm_modular_product_delayed_ntt_rhs_resident(
        lhs: &[Vec<crate::ring::RnsPolynomial>],
        rhs_transposed_ntt: &[Vec<crate::ring::RnsNttPolynomial>],
        plan: &crate::ring::PreparedRnsNttPlan,
    ) -> Vec<Vec<crate::ring::RnsPolynomial>> {
        let block = std::env::var("CCMM_GEMM_BLOCK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(128);

        assert!(block > 0, "CCMM GEMM block size must be nonzero");

        assert!(!lhs.is_empty(), "Batch CCMM lhs must not be empty");
        assert!(
            !rhs_transposed_ntt.is_empty(),
            "Batch CCMM rhs must not be empty"
        );

        let output_rows = lhs.len();
        let inner = lhs[0].len();

        assert!(inner > 0);
        assert!(lhs.iter().all(|row| row.len() == inner));

        assert_eq!(
            rhs_transposed_ntt.len(),
            inner,
            "Batch CCMM modular inner dimensions must match"
        );

        let output_columns = rhs_transposed_ntt[0].len();

        assert!(output_columns > 0);
        assert!(rhs_transposed_ntt
            .iter()
            .all(|row| row.len() == output_columns));

        let degree = plan.degree();
        let limb_count = plan.moduli().len();

        // Transform each polynomial matrix entry exactly once.
        let forward_start = std::time::Instant::now();

        let lhs_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = lhs
            .iter()
            .map(|row| row.iter().map(|entry| plan.forward(entry)).collect())
            .collect();

        let rhs_ntt = rhs_transposed_ntt;

        let forward_elapsed = forward_start.elapsed();

        let output_layer_size = output_rows * output_columns;

        // [limb][coordinate][row * output_columns + column]
        let mut output_layers = vec![vec![vec![0_u64; output_layer_size]; degree]; limb_count];

        let mut materialize_elapsed = std::time::Duration::ZERO;
        let mut gemm_elapsed = std::time::Duration::ZERO;

        let mut limb_materialize_elapsed = vec![std::time::Duration::ZERO; limb_count];

        let mut limb_gemm_elapsed = vec![std::time::Duration::ZERO; limb_count];

        let mut accumulate_elapsed = std::time::Duration::ZERO;

        let mut reduction_elapsed = std::time::Duration::ZERO;

        let mut limb_accumulate_elapsed = vec![std::time::Duration::ZERO; limb_count];

        let mut limb_reduction_elapsed = vec![std::time::Duration::ZERO; limb_count];

        for (limb_index, output_limb_layers) in
            output_layers.iter_mut().enumerate().take(limb_count)
        {
            let modulus = plan.moduli()[limb_index];
            let q = u128::from(modulus.value());

            let max_product = (q - 1) * (q - 1);

            max_product
                .checked_mul(inner as u128)
                .expect("CCMM delayed-reduction dot product must fit in u128");

            for (coordinate, output_layer) in output_limb_layers.iter_mut().enumerate().take(degree)
            {
                let materialize_start = std::time::Instant::now();

                let mut lhs_layer = vec![0_u64; output_rows * inner];

                let mut rhs_layer = vec![0_u64; inner * output_columns];

                for row in 0..output_rows {
                    for k in 0..inner {
                        lhs_layer[row * inner + k] =
                            lhs_ntt[row][k].residue(limb_index).values()[coordinate];
                    }
                }

                for k in 0..inner {
                    for column in 0..output_columns {
                        rhs_layer[k * output_columns + column] =
                            rhs_ntt[k][column].residue(limb_index).values()[coordinate];
                    }
                }

                let materialize_coordinate_elapsed = materialize_start.elapsed();

                materialize_elapsed += materialize_coordinate_elapsed;

                limb_materialize_elapsed[limb_index] += materialize_coordinate_elapsed;

                let mut wide = vec![0_u128; output_layer_size];

                let gemm_start = std::time::Instant::now();

                let accumulate_start = std::time::Instant::now();

                // Dense blocked i-k-j multiplication with no modular
                // reduction in the cubic loop.
                for ii in (0..output_rows).step_by(block) {
                    let i_end = (ii + block).min(output_rows);

                    for kk in (0..inner).step_by(block) {
                        let k_end = (kk + block).min(inner);

                        for jj in (0..output_columns).step_by(block) {
                            let j_end = (jj + block).min(output_columns);

                            for row in ii..i_end {
                                let lhs_base = row * inner;

                                let out_base = row * output_columns;

                                for k in kk..k_end {
                                    let lhs_value = u128::from(lhs_layer[lhs_base + k]);

                                    let rhs_base = k * output_columns;

                                    for column in jj..j_end {
                                        let rhs_value = u128::from(rhs_layer[rhs_base + column]);

                                        wide[out_base + column] += lhs_value * rhs_value;
                                    }
                                }
                            }
                        }
                    }
                }

                let accumulate_coordinate_elapsed = accumulate_start.elapsed();

                accumulate_elapsed += accumulate_coordinate_elapsed;

                limb_accumulate_elapsed[limb_index] += accumulate_coordinate_elapsed;

                let reduction_start = std::time::Instant::now();

                // One exact reduction per output matrix entry.
                for index in 0..output_layer_size {
                    output_layer[index] = (wide[index] % q) as u64;
                }

                let reduction_coordinate_elapsed = reduction_start.elapsed();

                reduction_elapsed += reduction_coordinate_elapsed;

                limb_reduction_elapsed[limb_index] += reduction_coordinate_elapsed;

                let gemm_coordinate_elapsed = gemm_start.elapsed();

                gemm_elapsed += gemm_coordinate_elapsed;

                limb_gemm_elapsed[limb_index] += gemm_coordinate_elapsed;
            }
        }

        for limb_index in 0..limb_count {
            println!(
                "BATCH_CCMM_DELAYED_LIMB_{}_MATERIALIZE_MS={:.3}",
                limb_index,
                limb_materialize_elapsed[limb_index].as_secs_f64() * 1.0e3
            );

            println!(
                "BATCH_CCMM_DELAYED_LIMB_{}_GEMM_MS={:.3}",
                limb_index,
                limb_gemm_elapsed[limb_index].as_secs_f64() * 1.0e3
            );

            println!(
                "BATCH_CCMM_DELAYED_LIMB_{}_ACCUMULATE_MS={:.3}",
                limb_index,
                limb_accumulate_elapsed[limb_index].as_secs_f64() * 1.0e3
            );

            println!(
                "BATCH_CCMM_DELAYED_LIMB_{}_REDUCTION_MS={:.3}",
                limb_index,
                limb_reduction_elapsed[limb_index].as_secs_f64() * 1.0e3
            );
        }

        let assemble_start = std::time::Instant::now();

        let output_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = (0..output_rows)
            .map(|row| {
                (0..output_columns)
                    .map(|column| {
                        let residues = (0..limb_count)
                            .map(|limb_index| {
                                let values: Vec<u64> = (0..degree)
                                    .map(|coordinate| {
                                        output_layers[limb_index][coordinate]
                                            [row * output_columns + column]
                                    })
                                    .collect();

                                crate::ring::NttPolynomial::from_prepared_values(
                                    plan.plan(limb_index),
                                    values,
                                )
                            })
                            .collect();

                        crate::ring::RnsNttPolynomial::from_residues(residues)
                    })
                    .collect()
            })
            .collect();

        let assemble_elapsed = assemble_start.elapsed();

        let inverse_start = std::time::Instant::now();

        let output: Vec<Vec<crate::ring::RnsPolynomial>> = output_ntt
            .iter()
            .map(|row| row.iter().map(|ntt| plan.inverse(ntt)).collect())
            .collect();

        let inverse_elapsed = inverse_start.elapsed();

        let reconstruct_elapsed = assemble_elapsed + inverse_elapsed;

        println!(
            "BATCH_CCMM_DELAYED_PHASE_FORWARD_NTT_RHS_RESIDENT_MS={:.3}",
            forward_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_DELAYED_PHASE_LAYER_MATERIALIZE_MS={:.3}",
            materialize_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_DELAYED_PHASE_ACCUMULATE_MS={:.3}",
            accumulate_elapsed.as_secs_f64() * 1.0e3
        );

        println!(
            "BATCH_CCMM_DELAYED_PHASE_REDUCTION_MS={:.3}",
            reduction_elapsed.as_secs_f64() * 1.0e3
        );

        println!(
            "BATCH_CCMM_DELAYED_PHASE_GEMM_MS={:.3}",
            gemm_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_DELAYED_PHASE_ASSEMBLE_NTT_MS={:.3}",
            assemble_elapsed.as_secs_f64() * 1.0e3
        );

        println!(
            "BATCH_CCMM_DELAYED_PHASE_INVERSE_NTT_MS={:.3}",
            inverse_elapsed.as_secs_f64() * 1.0e3
        );

        println!(
            "BATCH_CCMM_DELAYED_PHASE_RECONSTRUCT_INVERSE_MS={:.3}",
            reconstruct_elapsed.as_secs_f64() * 1.0e3
        );

        output
    }

    fn batch_ccmm_modular_product_prepared_ntt(
        lhs: &[Vec<crate::ring::RnsPolynomial>],
        rhs_transposed: &[Vec<crate::ring::RnsPolynomial>],
        plan: &crate::ring::PreparedRnsNttPlan,
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

        // Transform each matrix entry exactly once.
        let lhs_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = lhs
            .iter()
            .map(|row| row.iter().map(|entry| plan.forward(entry)).collect())
            .collect();

        let rhs_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = rhs_transposed
            .iter()
            .map(|row| row.iter().map(|entry| plan.forward(entry)).collect())
            .collect();

        (0..lhs_ntt.len())
            .map(|row| {
                (0..output_columns)
                    .map(|column| {
                        let mut accumulator = lhs_ntt[row][0].pointwise_mul(&rhs_ntt[0][column]);

                        for (lhs_entry, rhs_row) in lhs_ntt[row].iter().zip(rhs_ntt.iter()).skip(1)
                        {
                            let product = lhs_entry.pointwise_mul(&rhs_row[column]);

                            accumulator = accumulator.add(&product);
                        }

                        // One inverse NTT per output polynomial.
                        plan.inverse(&accumulator)
                    })
                    .collect()
            })
            .collect()
    }

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
        super::decode_cpmm_large_columns(ciphertexts, secret, scalar_degree, half_rows)
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

        // Authors-equivalent MatrixEvaluator::matrixMult timing boundary.
        // Encoding, encryption, and packed-plaintext preparation are complete.
        let geometry = crate::eblas::BatchGemmGeometry::new(
            crate::eblas::BatchGemmMechanism::Cpmm,
            dimension,
            scalar_degree,
            LARGE_DEGREE,
        );

        let matrix_mult_start = std::time::Instant::now();

        let output_ciphertexts = crate::eblas::batch_gemm_cpmm(
            geometry,
            &lhs_ciphertexts,
            &rhs_plain,
            &scalar_plan,
            &chain,
            scale,
        );

        let matrix_mult_elapsed = matrix_mult_start.elapsed();

        println!(
            "BATCH_CPMM_MATRIX_MULT_MS={:.3}",
            matrix_mult_elapsed.as_secs_f64() * 1.0e3
        );
        println!("BATCH_CPMM_DIMENSION={dimension}");
        println!("BATCH_CPMM_SCALAR_DEGREE={scalar_degree}");
        println!("BATCH_CPMM_BATCH_COUNT={batch_count}");

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

    fn cpmm_batch_gemv_capability(dimension: usize, scalar_degree: usize) -> (f64, f64) {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        const LARGE_DEGREE: usize = 8192;

        let batch_count = scalar_degree / 2;
        let half_rows = dimension / 2;
        let scale = sd3b_scale();

        assert_eq!(half_rows * scalar_degree, LARGE_DEGREE);

        let lhs_clear = deterministic_cpmm_input(batch_count, dimension, 1);

        // Reuse the deterministic CPMM family, but retain only one logical
        // public vector per packed batch.
        let vector_source = deterministic_cpmm_input(batch_count, dimension, 3);

        let vectors: Vec<Vec<f64>> = vector_source
            .iter()
            .map(|matrix| (0..dimension).map(|row| matrix[row][0]).collect())
            .collect();

        let expected: Vec<Vec<f64>> = lhs_clear
            .iter()
            .zip(&vectors)
            .map(|(matrix, vector)| {
                (0..dimension)
                    .map(|row| {
                        (0..dimension)
                            .map(|inner| matrix[row][inner] * vector[inner])
                            .sum()
                    })
                    .collect()
            })
            .collect();

        // Embed each logical d x 1 public vector as the first column of a
        // d x d plaintext matrix. All remaining columns are exactly zero.
        let rhs_clear: Vec<Vec<Vec<f64>>> = vectors
            .iter()
            .map(|vector| {
                (0..dimension)
                    .map(|row| {
                        let mut embedded_row = vec![0.0_f64; dimension];
                        embedded_row[0] = vector[row];
                        embedded_row
                    })
                    .collect()
            })
            .collect();

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

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4241_5443_485F_4756 ^ dimension as u64);

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

                let mut rng = ChaCha20Rng::seed_from_u64(0x4756_4D4D_0000_0000 ^ column as u64);

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

        let geometry = crate::eblas::BatchGemmGeometry::new(
            crate::eblas::BatchGemmMechanism::Cpmm,
            dimension,
            scalar_degree,
            LARGE_DEGREE,
        );

        let gemv_start = std::time::Instant::now();

        let output = crate::eblas::batch_gemv_cpmm(
            geometry,
            &lhs_ciphertexts,
            &rhs_plain,
            &scalar_plan,
            &chain,
            scale,
        );

        let gemv_elapsed = gemv_start.elapsed();

        assert_eq!(
            output.rlwe().degree(),
            LARGE_DEGREE,
            "Batch CPMM GEMV must preserve the large-ring degree"
        );

        assert_eq!(
            output.basis().moduli().len() + 1,
            basis.moduli().len(),
            "Batch CPMM GEMV must consume exactly one modulus level"
        );

        assert!(
            output.scale().is_finite() && output.scale() > 0.0,
            "Batch CPMM GEMV output scale must be finite and positive"
        );

        let decoded = decode_large_columns(
            std::slice::from_ref(&output),
            &secret,
            scalar_degree,
            half_rows,
        );

        let half_result = super::sinc_decode_batch(&decoded);

        assert_eq!(half_result.len(), batch_count);

        let actual: Vec<Vec<f64>> = half_result
            .iter()
            .map(|matrix| {
                assert_eq!(matrix.len(), half_rows);
                assert!(
                    matrix.iter().all(|row| row.len() == 1),
                    "Batch CPMM GEMV must decode exactly one logical column"
                );

                let mut vector = Vec::with_capacity(dimension);

                vector.extend(matrix.iter().map(|row| row[0].re));
                vector.extend(matrix.iter().map(|row| row[0].im));

                vector
            })
            .collect();

        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        let mut max_abs_error = 0.0_f64;

        for batch in 0..batch_count {
            for row in 0..dimension {
                let error = actual[batch][row] - expected[batch][row];

                assert!(
                    actual[batch][row].is_finite(),
                    "Batch CPMM GEMV decoded a non-finite value"
                );

                squared_error += error * error;
                squared_reference += expected[batch][row] * expected[batch][row];

                max_abs_error = max_abs_error.max(error.abs());
            }
        }

        assert!(
            squared_reference > 0.0,
            "Batch CPMM GEMV reference norm must be nonzero"
        );

        let relative_l2 = (squared_error / squared_reference).sqrt();

        println!("BATCH_CPMM_GEMV_DIMENSION={dimension}");
        println!("BATCH_CPMM_GEMV_SCALAR_DEGREE={scalar_degree}");
        println!("BATCH_CPMM_GEMV_BATCH_COUNT={batch_count}");
        println!("BATCH_CPMM_GEMV_OUTPUT_VALUES={}", batch_count * dimension);
        println!(
            "BATCH_CPMM_GEMV_MATRIX_MULT_MS={:.3}",
            gemv_elapsed.as_secs_f64() * 1.0e3
        );
        println!("BATCH_CPMM_GEMV_REL_L2={relative_l2:.12e}");
        println!("BATCH_CPMM_GEMV_MAX_ABS_ERROR={max_abs_error:.12e}");

        (relative_l2, max_abs_error)
    }

    #[test]
    fn batch_cpmm_gemv_d64_matches_clear_reference() {
        let (relative_l2, max_abs_error) = cpmm_batch_gemv_capability(64, 256);

        println!("BATCH_CPMM_GEMV_D64_REL_L2={relative_l2:.12e}");
        println!("BATCH_CPMM_GEMV_D64_MAX_ABS_ERROR={max_abs_error:.12e}");

        assert!(
            relative_l2 < 1.0e-3,
            "authors-scale Batch CPMM GEMV relative L2 error \
             {relative_l2:e}"
        );
    }

    fn cpmm_batch_dot_capability(dimension: usize, scalar_degree: usize) -> (f64, f64, f64) {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        const LARGE_DEGREE: usize = 8192;

        let batch_count = scalar_degree / 2;
        let half_rows = dimension / 2;
        let scale = sd3b_scale();

        assert_eq!(half_rows * scalar_degree, LARGE_DEGREE);

        let lhs_source = deterministic_cpmm_input(batch_count, dimension, 4);
        let rhs_source = deterministic_cpmm_input(batch_count, dimension, 5);

        let lhs_vectors: Vec<Vec<f64>> = lhs_source
            .iter()
            .map(|matrix| (0..dimension).map(|column| matrix[0][column]).collect())
            .collect();

        let rhs_vectors: Vec<Vec<f64>> = rhs_source
            .iter()
            .map(|matrix| (0..dimension).map(|row| matrix[row][0]).collect())
            .collect();

        let expected: Vec<f64> = lhs_vectors
            .iter()
            .zip(&rhs_vectors)
            .map(|(lhs, rhs)| (0..dimension).map(|index| lhs[index] * rhs[index]).sum())
            .collect();

        // Encrypted DOT lhs:
        //
        // [ x^T ]
        // [  0  ]
        // [  .  ]
        // [  0  ]
        let lhs_clear: Vec<Vec<Vec<f64>>> = lhs_vectors
            .iter()
            .map(|vector| {
                (0..dimension)
                    .map(|row| {
                        if row == 0 {
                            vector.clone()
                        } else {
                            vec![0.0_f64; dimension]
                        }
                    })
                    .collect()
            })
            .collect();

        // Public DOT rhs:
        //
        // [ y | 0 ... 0 ]
        let rhs_clear: Vec<Vec<Vec<f64>>> = rhs_vectors
            .iter()
            .map(|vector| {
                (0..dimension)
                    .map(|row| {
                        let mut embedded_row = vec![0.0_f64; dimension];
                        embedded_row[0] = vector[row];
                        embedded_row
                    })
                    .collect()
            })
            .collect();

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

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4241_5443_485F_444F ^ dimension as u64);

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

                let mut rng = ChaCha20Rng::seed_from_u64(0x444F_544D_0000_0000 ^ column as u64);

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

        let geometry = crate::eblas::BatchGemmGeometry::new(
            crate::eblas::BatchGemmMechanism::Cpmm,
            dimension,
            scalar_degree,
            LARGE_DEGREE,
        );

        let dot_start = std::time::Instant::now();

        let output = crate::eblas::batch_dot_cpmm(
            geometry,
            &lhs_ciphertexts,
            &rhs_plain,
            &scalar_plan,
            &chain,
            scale,
        );

        let dot_elapsed = dot_start.elapsed();

        assert_eq!(
            output.rlwe().degree(),
            LARGE_DEGREE,
            "Batch CPMM DOT must preserve the large-ring degree"
        );

        assert_eq!(
            output.basis().moduli().len() + 1,
            basis.moduli().len(),
            "Batch CPMM DOT must consume exactly one modulus level"
        );

        assert!(
            output.scale().is_finite() && output.scale() > 0.0,
            "Batch CPMM DOT output scale must be finite and positive"
        );

        let decoded = decode_large_columns(
            std::slice::from_ref(&output),
            &secret,
            scalar_degree,
            half_rows,
        );

        let half_result = super::sinc_decode_batch(&decoded);

        assert_eq!(half_result.len(), batch_count);

        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        let mut max_abs_error = 0.0_f64;
        let mut max_structural_zero = 0.0_f64;

        for batch in 0..batch_count {
            let matrix = &half_result[batch];

            assert_eq!(matrix.len(), half_rows);
            assert!(
                matrix.iter().all(|row| row.len() == 1),
                "Batch CPMM DOT must decode one logical column"
            );

            let mut logical = Vec::with_capacity(dimension);

            logical.extend(matrix.iter().map(|row| row[0].re));
            logical.extend(matrix.iter().map(|row| row[0].im));

            assert!(
                logical.iter().all(|value| value.is_finite()),
                "Batch CPMM DOT decoded a non-finite value"
            );

            let error = logical[0] - expected[batch];

            squared_error += error * error;
            squared_reference += expected[batch] * expected[batch];

            max_abs_error = max_abs_error.max(error.abs());

            for &value in &logical[1..] {
                max_structural_zero = max_structural_zero.max(value.abs());
            }
        }

        assert!(
            squared_reference > 0.0,
            "Batch CPMM DOT reference norm must be nonzero"
        );

        let relative_l2 = (squared_error / squared_reference).sqrt();

        println!("BATCH_CPMM_DOT_DIMENSION={dimension}");
        println!("BATCH_CPMM_DOT_SCALAR_DEGREE={scalar_degree}");
        println!("BATCH_CPMM_DOT_BATCH_COUNT={batch_count}");
        println!("BATCH_CPMM_DOT_OUTPUT_VALUES={batch_count}");
        println!(
            "BATCH_CPMM_DOT_MATRIX_MULT_MS={:.3}",
            dot_elapsed.as_secs_f64() * 1.0e3
        );
        println!("BATCH_CPMM_DOT_REL_L2={relative_l2:.12e}");
        println!("BATCH_CPMM_DOT_MAX_ABS_ERROR={max_abs_error:.12e}");
        println!("BATCH_CPMM_DOT_MAX_STRUCTURAL_ZERO={max_structural_zero:.12e}");

        (relative_l2, max_abs_error, max_structural_zero)
    }

    #[test]
    fn batch_cpmm_gemv_d128_matches_clear_reference() {
        let (relative_l2, max_abs_error) = cpmm_batch_gemv_capability(128, 128);

        println!("BATCH_CPMM_GEMV_D128_REL_L2={relative_l2:.12e}");
        println!("BATCH_CPMM_GEMV_D128_MAX_ABS_ERROR={max_abs_error:.12e}");

        assert!(
            relative_l2 < 1.0e-3,
            "authors-scale Batch CPMM GEMV d=128 relative L2 error \
             {relative_l2:e}"
        );
    }

    #[test]
    fn batch_cpmm_dot_d64_matches_clear_reference() {
        let (relative_l2, max_abs_error, max_structural_zero) = cpmm_batch_dot_capability(64, 256);

        println!("BATCH_CPMM_DOT_D64_REL_L2={relative_l2:.12e}");
        println!("BATCH_CPMM_DOT_D64_MAX_ABS_ERROR={max_abs_error:.12e}");
        println!("BATCH_CPMM_DOT_D64_MAX_STRUCTURAL_ZERO={max_structural_zero:.12e}");

        assert!(
            relative_l2 < 1.0e-3,
            "authors-scale Batch CPMM DOT relative L2 error \
             {relative_l2:e}"
        );

        assert!(
            max_structural_zero < 1.0e-3,
            "authors-scale Batch CPMM DOT structural-zero error \
             {max_structural_zero:e}"
        );
    }

    #[test]
    fn batch_cpmm_dot_d128_matches_clear_reference() {
        let (relative_l2, max_abs_error, max_structural_zero) = cpmm_batch_dot_capability(128, 128);

        println!("BATCH_CPMM_DOT_D128_REL_L2={relative_l2:.12e}");
        println!("BATCH_CPMM_DOT_D128_MAX_ABS_ERROR={max_abs_error:.12e}");
        println!("BATCH_CPMM_DOT_D128_MAX_STRUCTURAL_ZERO={max_structural_zero:.12e}");

        assert!(
            relative_l2 < 1.0e-3,
            "authors-scale Batch CPMM DOT d=128 relative L2 error \
             {relative_l2:e}"
        );

        assert!(
            max_structural_zero < 1.0e-3,
            "authors-scale Batch CPMM DOT d=128 structural-zero error \
             {max_structural_zero:e}"
        );
    }

    #[test]
    fn batch_galois_ntt_matches_reference_exactly() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        const DEGREE: usize = 16;

        let moduli = vec![
            crate::ring::Modulus::new(193),
            crate::ring::Modulus::new(257),
        ];

        let basis = crate::ring::ModulusBasis::new(moduli.clone());
        let plan = crate::ring::RnsNttPlan::new(moduli, DEGREE);

        let secret: Vec<i8> = (0..DEGREE)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 1,
            })
            .collect();

        let values: Vec<u128> = (0..DEGREE)
            .map(|index| ((index * 13 + 7) % 31) as u128)
            .collect();

        let plaintext =
            crate::ring::RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &values);

        let mut encrypt_rng = ChaCha20Rng::seed_from_u64(0xB5A0_0001);

        let ciphertext = crate::grafting::encrypt_rns_raw_with_distribution_ntt_rng(
            &plaintext,
            2,
            crate::rlwe::ErrorDistribution::BoundedUniform { bound: 0 },
            &secret,
            &plan,
            &mut encrypt_rng,
        );

        let layout = crate::grafting::RnsGadgetLayout::new(basis, vec![1, 1]);

        let exponent = 5;

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xB5A0_0002);

        let key = crate::ckks::RnsGaloisKey::generate_with_rng(
            DEGREE,
            2,
            0,
            &secret,
            exponent,
            layout,
            &mut key_rng,
        );

        let reference = crate::ckks::apply_rns_galois_automorphism(&ciphertext, &key);

        let optimized =
            crate::ckks::apply_rns_galois_automorphism_with_ntt(&ciphertext, &key, &plan);

        let prepared_key = crate::ckks::PreparedRnsGaloisKey::new(&key, &plan);

        let prepared = crate::ckks::apply_rns_galois_automorphism_with_prepared_ntt(
            &ciphertext,
            &prepared_key,
            &plan,
        );

        assert_eq!(optimized, reference);
        assert_eq!(prepared, reference);
        assert_eq!(prepared, optimized);

        println!("BATCH_GALOIS_NTT_EQUIVALENCE=PASS");
        println!("BATCH_GALOIS_PREPARED_NTT_EQUIVALENCE=PASS");
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

        let prepared_galois_keys: Vec<crate::ckks::PreparedRnsGaloisKey> = galois_keys
            .iter()
            .map(|key| crate::ckks::PreparedRnsGaloisKey::new(key, &plan))
            .collect();

        let transposed =
            batch_ciphertext_transpose(&ciphertexts, SCALAR_DEGREE, &prepared_galois_keys, &plan);

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
    fn batch_cpmm_layer_major_matches_prepared_ntt_exactly() {
        const DIMENSION: usize = 4;
        const HALF_ROWS: usize = 2;
        const DEGREE: usize = 8;

        let moduli = vec![
            crate::ring::Modulus::new(193),
            crate::ring::Modulus::new(257),
        ];

        let plan = crate::ring::RnsNttPlan::new(moduli.clone(), DEGREE);
        let prepared_plan = crate::ring::PreparedRnsNttPlan::new(&plan);

        fn make_ciphertext(
            moduli: &[crate::ring::Modulus],
            seed: usize,
        ) -> crate::grafting::RnsRlweCiphertext {
            crate::grafting::RnsRlweCiphertext::from_limbs(
                moduli
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(limb_index, modulus)| {
                        let b = crate::ring::Polynomial::new(
                            modulus,
                            (0..DEGREE)
                                .map(|index| {
                                    (((seed + 1) * 17 + (limb_index + 1) * 13 + index * 7) as u64)
                                        % modulus.value()
                                })
                                .collect(),
                        );

                        let a = crate::ring::Polynomial::new(
                            modulus,
                            (0..DEGREE)
                                .map(|index| {
                                    (((seed + 3) * 19 + (limb_index + 1) * 11 + index * 5) as u64)
                                        % modulus.value()
                                })
                                .collect(),
                        );

                        crate::rlwe::RlweCiphertext::new(b, a)
                    })
                    .collect(),
            )
        }

        fn make_plaintext(
            moduli: &[crate::ring::Modulus],
            seed: usize,
        ) -> crate::ring::RnsPolynomial {
            let coefficients: Vec<u128> = (0..DEGREE)
                .map(|index| ((seed + 1) * 23 + index * 9 + 3) as u128)
                .collect();

            crate::ring::RnsPolynomial::from_coefficients(moduli.to_vec(), &coefficients)
        }

        let lhs_split: Vec<Vec<_>> = (0..DIMENSION)
            .map(|inner| {
                (0..HALF_ROWS)
                    .map(|row| make_ciphertext(&moduli, 100 * inner + row))
                    .collect()
            })
            .collect();

        let rhs_plain: Vec<Vec<_>> = (0..DIMENSION)
            .map(|inner| {
                (0..DIMENSION)
                    .map(|column| make_plaintext(&moduli, 1000 + inner * DIMENSION + column))
                    .collect()
            })
            .collect();

        let o1 = batch_cpmm_modular_product_prepared_ntt(&lhs_split, &rhs_plain, &prepared_plan);

        let o2 = batch_cpmm_modular_product_layer_major(&lhs_split, &rhs_plain, &prepared_plan);

        let o5 = batch_cpmm_modular_product_blocked_ntt(&lhs_split, &rhs_plain, &prepared_plan);

        assert_eq!(
            o2, o1,
            "layer-major CPMM must exactly match prepared-NTT CPMM"
        );

        assert_eq!(
            o5, o1,
            "blocked dense CPMM must exactly match prepared-NTT CPMM"
        );

        println!("BATCH_CPMM_LAYER_MAJOR_EQUIVALENCE=PASS");
        println!("BATCH_CPMM_BLOCKED_NTT_EQUIVALENCE=PASS");
    }

    #[test]
    fn batch_cpmm_prepared_ntt_modular_product_matches_reference_exactly() {
        const DIMENSION: usize = 4;
        const HALF_ROWS: usize = 2;
        const DEGREE: usize = 8;

        let moduli = vec![
            crate::ring::Modulus::new(193),
            crate::ring::Modulus::new(257),
        ];

        let plan = crate::ring::RnsNttPlan::new(moduli.clone(), DEGREE);
        let prepared_plan = crate::ring::PreparedRnsNttPlan::new(&plan);

        fn make_ciphertext(
            moduli: &[crate::ring::Modulus],
            seed: usize,
        ) -> crate::grafting::RnsRlweCiphertext {
            let limbs = moduli
                .iter()
                .copied()
                .enumerate()
                .map(|(limb_index, modulus)| {
                    let b = crate::ring::Polynomial::new(
                        modulus,
                        (0..DEGREE)
                            .map(|index| {
                                (((seed + 1) * 17 + (limb_index + 1) * 13 + index * 7) as u64)
                                    % modulus.value()
                            })
                            .collect(),
                    );

                    let a = crate::ring::Polynomial::new(
                        modulus,
                        (0..DEGREE)
                            .map(|index| {
                                (((seed + 3) * 19 + (limb_index + 1) * 11 + index * 5) as u64)
                                    % modulus.value()
                            })
                            .collect(),
                    );

                    crate::rlwe::RlweCiphertext::new(b, a)
                })
                .collect();

            crate::grafting::RnsRlweCiphertext::from_limbs(limbs)
        }

        fn make_plaintext(
            moduli: &[crate::ring::Modulus],
            seed: usize,
        ) -> crate::ring::RnsPolynomial {
            let coefficients: Vec<u128> = (0..DEGREE)
                .map(|index| ((seed + 1) * 23 + index * 9 + 3) as u128)
                .collect();

            crate::ring::RnsPolynomial::from_coefficients(moduli.to_vec(), &coefficients)
        }

        // d x h ciphertext representation.
        let lhs_split: Vec<Vec<_>> = (0..DIMENSION)
            .map(|inner| {
                (0..HALF_ROWS)
                    .map(|row| make_ciphertext(&moduli, 100 * inner + row))
                    .collect()
            })
            .collect();

        // d x d plaintext matrix.
        let rhs_plain: Vec<Vec<_>> = (0..DIMENSION)
            .map(|inner| {
                (0..DIMENSION)
                    .map(|column| make_plaintext(&moduli, 1000 + inner * DIMENSION + column))
                    .collect()
            })
            .collect();

        // Current source-faithful CPMM reference.
        let reference: Vec<Vec<_>> = (0..HALF_ROWS)
            .map(|row| {
                (0..DIMENSION)
                    .map(|column| {
                        let mut accumulator =
                            raw_cp_product(&lhs_split[0][row], &rhs_plain[0][column], &plan);

                        for inner in 1..DIMENSION {
                            let product = raw_cp_product(
                                &lhs_split[inner][row],
                                &rhs_plain[inner][column],
                                &plan,
                            );

                            accumulator = raw_rlwe_add(&accumulator, &product);
                        }

                        accumulator
                    })
                    .collect()
            })
            .collect();

        let optimized =
            batch_cpmm_modular_product_prepared_ntt(&lhs_split, &rhs_plain, &prepared_plan);

        assert_eq!(
            optimized, reference,
            "prepared NTT CPMM modular product must exactly match the current reference path"
        );

        println!("BATCH_CPMM_PREPARED_NTT_MODULAR_PRODUCT_EQUIVALENCE=PASS");
    }

    fn batch_inverse_cyclic_ntt_radix2(
        values: &mut [u64],
        modulus: crate::ring::Modulus,
        root: u64,
    ) {
        let dimension = values.len();

        assert!(
            dimension > 0 && dimension.is_power_of_two(),
            "cross-frequency inverse NTT dimension must be a positive power of two"
        );

        // Bit-reverse permutation.
        let bits = dimension.trailing_zeros();

        for index in 0..dimension {
            let reversed = index.reverse_bits() >> (usize::BITS - bits);

            if reversed > index {
                values.swap(index, reversed);
            }
        }

        let root_inverse = modulus.inverse_prime(root);

        let mut len = 2_usize;

        while len <= dimension {
            let half = len / 2;

            let root_step = modulus.pow(root_inverse, (dimension / len) as u64);

            let mut twiddles = Vec::with_capacity(half);

            let mut twiddle = 1_u64;

            for _ in 0..half {
                twiddles.push(twiddle);

                twiddle = modulus.mul(twiddle, root_step);
            }

            for start in (0..dimension).step_by(len) {
                for (offset, &twiddle) in twiddles.iter().enumerate() {
                    let even = values[start + offset];

                    let odd = modulus.mul(values[start + offset + half], twiddle);

                    values[start + offset] = modulus.add_canonical(even, odd);

                    values[start + offset + half] = modulus.sub_canonical(even, odd);
                }
            }

            len *= 2;
        }

        let dimension_inverse = modulus.inverse_prime(dimension as u64);

        for value in values {
            *value = modulus.mul(*value, dimension_inverse);
        }
    }

    fn batch_large_ntt_to_scalar_ntt_radix2(
        large_ntt: &crate::ring::RnsNttPolynomial,
        dimension: usize,
        large_plan: &crate::ring::RnsNttPlan,
        scalar_plan: &crate::ring::RnsNttPlan,
    ) -> Vec<crate::ring::RnsNttPolynomial> {
        assert!(dimension > 0);
        assert!(dimension.is_power_of_two());

        let large_degree = large_ntt.degree();

        assert_eq!(
            large_degree,
            large_plan.degree(),
            "large NTT polynomial degree must match large plan"
        );

        assert_eq!(
            large_degree % dimension,
            0,
            "large degree must be divisible by Batch dimension"
        );

        let scalar_degree = large_degree / dimension;

        assert_eq!(
            scalar_degree,
            scalar_plan.degree(),
            "scalar plan degree must match N / d"
        );

        assert_eq!(
            large_plan.moduli(),
            scalar_plan.moduli(),
            "large and scalar plans must use the same RNS basis"
        );

        let mut output_values =
            vec![vec![vec![0_u64; scalar_degree]; large_plan.moduli().len()]; dimension];

        for limb_index in 0..large_plan.moduli().len() {
            let modulus = large_plan.moduli()[limb_index];

            let large_limb_plan = large_plan.plan(limb_index);

            let scalar_limb_plan = scalar_plan.plan(limb_index);

            let psi_large = large_limb_plan.psi();

            let psi_scalar = scalar_limb_plan.psi();

            let derived_scalar = modulus.pow(psi_large, dimension as u64);

            let two_ns = 2 * scalar_degree;

            let root_exponent = (0..two_ns)
                .step_by(2)
                .map(|offset| offset + 1)
                .find(|&candidate| modulus.pow(derived_scalar, candidate as u64) == psi_scalar)
                .expect("scalar NTT root must belong to derived large-root subgroup");

            let rho = modulus.pow(psi_large, (2 * scalar_degree) as u64);

            let large_values = large_ntt.residue(limb_index).values();

            for scalar_frequency in 0..scalar_degree {
                let odd = (root_exponent * (2 * scalar_frequency + 1)) % (2 * scalar_degree);

                assert_eq!(
                    odd % 2,
                    1,
                    "mapped negacyclic evaluation exponent must remain odd"
                );

                let k0 = (odd - 1) / 2;

                let z0 = modulus.pow(psi_large, (2 * k0 + 1) as u64);

                let mut cross_frequency = Vec::with_capacity(dimension);

                for m in 0..dimension {
                    cross_frequency.push(large_values[k0 + m * scalar_degree]);
                }

                batch_inverse_cyclic_ntt_radix2(&mut cross_frequency, modulus, rho);

                let z0_inverse = modulus.inverse_prime(z0);

                let mut untwiddle = 1_u64;

                for (row, row_output) in output_values.iter_mut().enumerate() {
                    if row > 0 {
                        untwiddle = modulus.mul(untwiddle, z0_inverse);
                    }

                    row_output[limb_index][scalar_frequency] =
                        modulus.mul(cross_frequency[row], untwiddle);
                }
            }
        }

        output_values
            .into_iter()
            .map(|row_limbs| {
                let residues = row_limbs
                    .into_iter()
                    .enumerate()
                    .map(|(limb_index, values)| {
                        crate::ring::NttPolynomial::from_canonical_values(
                            scalar_plan.plan(limb_index),
                            values,
                        )
                    })
                    .collect();

                crate::ring::RnsNttPolynomial::from_residues(residues)
            })
            .collect()
    }

    fn batch_large_ntt_to_scalar_ntt_oracle(
        large_ntt: &crate::ring::RnsNttPolynomial,
        dimension: usize,
        large_plan: &crate::ring::RnsNttPlan,
        scalar_plan: &crate::ring::RnsNttPlan,
    ) -> Vec<crate::ring::RnsNttPolynomial> {
        assert!(dimension > 0);
        assert!(dimension.is_power_of_two());

        let large_degree = large_ntt.degree();

        assert_eq!(
            large_degree,
            large_plan.degree(),
            "large NTT polynomial degree must match large plan"
        );

        assert_eq!(
            large_degree % dimension,
            0,
            "large degree must be divisible by Batch dimension"
        );

        let scalar_degree = large_degree / dimension;

        assert_eq!(
            scalar_degree,
            scalar_plan.degree(),
            "scalar plan degree must match N / d"
        );

        assert_eq!(
            large_plan.moduli(),
            scalar_plan.moduli(),
            "large and scalar plans must use the same RNS basis"
        );

        assert_eq!(
            large_ntt.moduli(),
            large_plan.moduli(),
            "large NTT polynomial basis must match large plan"
        );

        // [row][limb][scalar_frequency]
        let mut output_values =
            vec![vec![vec![0_u64; scalar_degree]; large_plan.moduli().len()]; dimension];

        for limb_index in 0..large_plan.moduli().len() {
            let modulus = large_plan.moduli()[limb_index];

            let large_limb_plan = large_plan.plan(limb_index);

            let scalar_limb_plan = scalar_plan.plan(limb_index);

            let psi_large = large_limb_plan.psi();

            let psi_scalar = scalar_limb_plan.psi();

            let derived_scalar = modulus.pow(psi_large, dimension as u64);

            // Recover odd u such that
            //
            // psi_scalar = (psi_large^d)^u.
            let two_ns = 2 * scalar_degree;

            let root_exponent = (0..two_ns)
                .step_by(2)
                .map(|offset| offset + 1)
                .find(|&candidate| modulus.pow(derived_scalar, candidate as u64) == psi_scalar)
                .expect("scalar NTT root must belong to derived large-root subgroup");

            // Moving by Ns in the large NTT frequency index
            // multiplies the large evaluation point by rho.
            //
            // rho has exact order d.
            let rho = modulus.pow(psi_large, (2 * scalar_degree) as u64);

            let rho_inverse = modulus.inverse_prime(rho);

            let dimension_inverse = modulus.inverse_prime(dimension as u64);

            let large_values = large_ntt.residue(limb_index).values();

            for scalar_frequency in 0..scalar_degree {
                // Scalar evaluation point:
                //
                // y_t = psi_scalar^(2t+1)
                //
                // Find k0 in [0, Ns) satisfying:
                //
                // (psi_large^(2k0+1))^d = y_t.
                let odd = (root_exponent * (2 * scalar_frequency + 1)) % (2 * scalar_degree);

                assert_eq!(
                    odd % 2,
                    1,
                    "mapped negacyclic evaluation exponent must remain odd"
                );

                let k0 = (odd - 1) / 2;

                let z0 = modulus.pow(psi_large, (2 * k0 + 1) as u64);

                // For m = 0..d:
                //
                // V_m = A(z0 * rho^m)
                //
                //     = sum_r
                //         z0^r rho^(m r) P_r(y_t).
                //
                // Therefore inverse cyclic DFT across m gives
                //
                // S_r = z0^r P_r(y_t).
                //
                // Finally multiply by z0^(-r).
                for (row, row_output) in output_values.iter_mut().enumerate() {
                    let mut accumulator = 0_u64;

                    let mut inverse_twiddle = 1_u64;

                    let inverse_step = modulus.pow(rho_inverse, row as u64);

                    for m in 0..dimension {
                        let large_frequency = k0 + m * scalar_degree;

                        let value = large_values[large_frequency];

                        accumulator =
                            modulus.add_canonical(accumulator, modulus.mul(value, inverse_twiddle));

                        inverse_twiddle = modulus.mul(inverse_twiddle, inverse_step);
                    }

                    let scaled = modulus.mul(accumulator, dimension_inverse);

                    let z0_inverse = modulus.inverse_prime(z0);

                    let untwiddle = modulus.pow(z0_inverse, row as u64);

                    row_output[limb_index][scalar_frequency] = modulus.mul(scaled, untwiddle);
                }
            }
        }

        output_values
            .into_iter()
            .map(|row_limbs| {
                let residues = row_limbs
                    .into_iter()
                    .enumerate()
                    .map(|(limb_index, values)| {
                        crate::ring::NttPolynomial::from_canonical_values(
                            scalar_plan.plan(limb_index),
                            values,
                        )
                    })
                    .collect();

                crate::ring::RnsNttPolynomial::from_residues(residues)
            })
            .collect()
    }

    #[test]
    fn batch_delayed_mm_rhs_resident_matches_coefficient_rhs_exactly() {
        let basis = crate::ring::ModulusBasis::new(sd3b_moduli());

        let degree = 8_usize;
        let rows = 4_usize;
        let inner = 4_usize;
        let columns = 8_usize;

        let canonical_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), degree);

        let prepared_plan = crate::ring::PreparedRnsNttPlan::new(&canonical_plan);

        let make_polynomial = |seed: u64| {
            crate::ring::RnsPolynomial::from_residues(
                basis
                    .moduli()
                    .iter()
                    .enumerate()
                    .map(|(limb_index, &modulus)| {
                        let q = modulus.value();

                        let coefficients = (0..degree)
                            .map(|index| {
                                (seed + 17_u64 * index as u64 + 11_u64 * limb_index as u64) % q
                            })
                            .collect();

                        crate::ring::Polynomial::new(modulus, coefficients)
                    })
                    .collect(),
            )
        };

        let lhs: Vec<Vec<_>> = (0..rows)
            .map(|row| {
                (0..inner)
                    .map(|k| make_polynomial(19 + 101 * row as u64 + 37 * k as u64))
                    .collect()
            })
            .collect();

        let rhs: Vec<Vec<_>> = (0..inner)
            .map(|k| {
                (0..columns)
                    .map(|column| make_polynomial(53 + 73 * k as u64 + 29 * column as u64))
                    .collect()
            })
            .collect();

        let rhs_ntt: Vec<Vec<_>> = rhs
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| prepared_plan.forward(entry))
                    .collect()
            })
            .collect();

        let expected = batch_ccmm_modular_product_delayed_ntt(&lhs, &rhs, &prepared_plan);

        let actual =
            batch_ccmm_modular_product_delayed_ntt_rhs_resident(&lhs, &rhs_ntt, &prepared_plan);

        assert_eq!(
            actual, expected,
            "RHS-resident delayed modular MM must exactly match coefficient-RHS kernel"
        );

        println!("O18D_DELAYED_MM_RHS_RESIDENT_EXACT=PASS");
    }

    #[test]
    fn batch_resident_rhs_ntt_slice_matches_coefficient_path_exactly() {
        let basis = crate::ring::ModulusBasis::new(sd3b_moduli());

        let dimension = 8_usize;

        let scalar_degree = 8_usize;

        let large_degree = dimension * scalar_degree;

        let large_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), large_degree);

        let prepared_large_plan = crate::ring::PreparedRnsNttPlan::new(&large_plan);

        let scalar_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), scalar_degree);

        let make_component = |seed: u64| {
            crate::ring::RnsPolynomial::from_residues(
                basis
                    .moduli()
                    .iter()
                    .enumerate()
                    .map(|(limb_index, &modulus)| {
                        let q = modulus.value();

                        let coefficients = (0..large_degree)
                            .map(|index| {
                                (seed + 37_u64 * index as u64 + 13_u64 * limb_index as u64) % q
                            })
                            .collect();

                        crate::ring::Polynomial::new(modulus, coefficients)
                    })
                    .collect(),
            )
        };

        let coefficient_ciphertexts: Vec<_> = (0..dimension)
            .map(|column| {
                let b = make_component(19_u64 + 101_u64 * column as u64);

                let a = make_component(53_u64 + 131_u64 * column as u64);

                crate::grafting::RnsRlweCiphertext::from_limbs(
                    b.residues()
                        .iter()
                        .zip(a.residues())
                        .map(|(b_limb, a_limb)| {
                            crate::rlwe::RlweCiphertext::new(b_limb.clone(), a_limb.clone())
                        })
                        .collect(),
                )
            })
            .collect();

        let resident_ciphertexts: Vec<_> = coefficient_ciphertexts
            .iter()
            .map(|ciphertext| {
                crate::grafting::RnsNttRlweCiphertext::from_coefficient(
                    ciphertext,
                    &prepared_large_plan,
                )
            })
            .collect();

        // Reference:
        //
        // coefficient ciphertexts
        //   -> ranked scalar-ring slices
        //   -> transpose
        //   -> scalar-ring forward NTT.
        let reference_coeff = batch_ccmm_slice_ranked(&coefficient_ciphertexts, dimension);

        let reference_coeff_t = batch_ccmm_transpose_matrix(&reference_coeff);

        let reference_ntt: Vec<Vec<_>> = reference_coeff_t
            .iter()
            .map(|row| row.iter().map(|entry| scalar_plan.forward(entry)).collect())
            .collect();

        // Direct resident path:
        //
        // large-ring NTT ciphertexts
        //   -> direct O18d bridge
        //   -> ranked scalar-ring NTT slices
        //   -> transpose.
        let direct_ranked = batch_ccmm_slice_ranked_resident_ntt(
            &resident_ciphertexts,
            dimension,
            &large_plan,
            &scalar_plan,
        );

        let direct_ntt = batch_ccmm_transpose_matrix(&direct_ranked);

        assert_eq!(
            direct_ntt, reference_ntt,
            "resident RHS NTT slicing must exactly match coefficient slice/transpose/forward"
        );

        assert_eq!(direct_ntt.len(), dimension,);

        assert!(direct_ntt.iter().all(|row| { row.len() == 2 * dimension }),);

        println!("O18D_RESIDENT_RHS_NTT_SLICE_EXACT=PASS");
    }

    #[test]
    fn batch_large_ntt_to_scalar_ntt_d64_characterization() {
        let basis = crate::ring::ModulusBasis::new(sd3b_moduli());

        let dimension = 64_usize;

        let scalar_degree = 128_usize;

        let large_degree = dimension * scalar_degree;

        assert_eq!(large_degree, 8192, "authors d64 geometry must use N = 8192");

        let large_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), large_degree);

        let scalar_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), scalar_degree);

        let residues = basis
            .moduli()
            .iter()
            .enumerate()
            .map(|(limb_index, &modulus)| {
                let q = modulus.value();

                let coefficients = (0..large_degree)
                    .map(|index| (41_u64 + 131_u64 * index as u64 + 17_u64 * limb_index as u64) % q)
                    .collect();

                crate::ring::Polynomial::new(modulus, coefficients)
            })
            .collect();

        let large = crate::ring::RnsPolynomial::from_residues(residues);

        let large_ntt = large_plan.forward(&large);

        let old_start = std::time::Instant::now();

        let recovered_large = large_plan.inverse(&large_ntt);

        let split = super::split_rns_polynomial(&recovered_large, dimension);

        let old_output: Vec<_> = split.iter().map(|part| scalar_plan.forward(part)).collect();

        let old_elapsed = old_start.elapsed();

        let new_start = std::time::Instant::now();

        let new_output =
            batch_large_ntt_to_scalar_ntt_radix2(&large_ntt, dimension, &large_plan, &scalar_plan);

        let new_elapsed = new_start.elapsed();

        assert_eq!(
            new_output, old_output,
            "O18d.4 direct d64 bridge must exactly match inverse/split/forward boundary"
        );

        println!(
            "O18D_D64_OLD_BOUNDARY_MS={:.3}",
            old_elapsed.as_secs_f64() * 1.0e3
        );

        println!(
            "O18D_D64_DIRECT_BRIDGE_MS={:.3}",
            new_elapsed.as_secs_f64() * 1.0e3
        );

        println!(
            "O18D_D64_SPEEDUP={:.3}",
            old_elapsed.as_secs_f64() / new_elapsed.as_secs_f64()
        );

        println!("O18D_D64_BRIDGE_EXACT=PASS");
    }

    #[test]
    fn batch_large_ntt_to_scalar_ntt_radix2_matches_oracle_exactly() {
        let basis = crate::ring::ModulusBasis::new(sd3b_moduli());

        for &(dimension, scalar_degree) in &[(4_usize, 8_usize), (8_usize, 8_usize)] {
            let large_degree = dimension * scalar_degree;

            let large_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), large_degree);

            let scalar_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), scalar_degree);

            let residues = basis
                .moduli()
                .iter()
                .enumerate()
                .map(|(limb_index, &modulus)| {
                    let q = modulus.value();

                    let coefficients = (0..large_degree)
                        .map(|index| {
                            (23_u64 + 31_u64 * index as u64 + 7_u64 * limb_index as u64) % q
                        })
                        .collect();

                    crate::ring::Polynomial::new(modulus, coefficients)
                })
                .collect();

            let large = crate::ring::RnsPolynomial::from_residues(residues);

            let large_ntt = large_plan.forward(&large);

            let oracle = batch_large_ntt_to_scalar_ntt_oracle(
                &large_ntt,
                dimension,
                &large_plan,
                &scalar_plan,
            );

            let actual = batch_large_ntt_to_scalar_ntt_radix2(
                &large_ntt,
                dimension,
                &large_plan,
                &scalar_plan,
            );

            assert_eq!(
                actual, oracle,
                "radix-2 direct bridge must exactly match the O18d.2 oracle"
            );
        }

        println!("O18D_RADIX2_BRIDGE_EXACT=PASS");
    }

    #[test]
    fn batch_large_ntt_to_scalar_ntt_matches_split_then_forward_exactly() {
        let basis = crate::ring::ModulusBasis::new(sd3b_moduli());

        let dimension = 8_usize;

        let scalar_degree = 8_usize;

        let large_degree = dimension * scalar_degree;

        let large_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), large_degree);

        let scalar_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), scalar_degree);

        let residues = basis
            .moduli()
            .iter()
            .enumerate()
            .map(|(limb_index, &modulus)| {
                let q = modulus.value();

                let coefficients = (0..large_degree)
                    .map(|index| (17_u64 + 29_u64 * index as u64 + 11_u64 * limb_index as u64) % q)
                    .collect();

                crate::ring::Polynomial::new(modulus, coefficients)
            })
            .collect();

        let large = crate::ring::RnsPolynomial::from_residues(residues);

        let large_ntt = large_plan.forward(&large);

        let expected: Vec<_> = super::split_rns_polynomial(&large, dimension)
            .iter()
            .map(|part| scalar_plan.forward(part))
            .collect();

        let actual =
            batch_large_ntt_to_scalar_ntt_oracle(&large_ntt, dimension, &large_plan, &scalar_plan);

        assert_eq!(
            actual,
            expected,
            "direct large-NTT to scalar-NTT bridge must exactly match coefficient split followed by scalar forward NTT"
        );

        println!("O18D_LARGE_NTT_TO_SCALAR_NTT_EXACT=PASS");
    }

    #[test]
    fn batch_large_to_scalar_ntt_root_relation() {
        let basis = crate::ring::ModulusBasis::new(sd3b_moduli());

        for &(large_degree, dimension) in &[(64_usize, 8_usize), (128_usize, 8_usize)] {
            let scalar_degree = large_degree / dimension;

            let large_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), large_degree);

            let scalar_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), scalar_degree);

            for limb_index in 0..basis.len() {
                let modulus = basis.modulus(limb_index);

                let psi_large = large_plan.plan(limb_index).psi();

                let psi_scalar = scalar_plan.plan(limb_index).psi();

                let derived_scalar = modulus.pow(psi_large, dimension as u64);

                // Both are primitive 2*Ns roots, but they need not
                // be the same generator. Recover the odd exponent u:
                //
                // psi_scalar = derived_scalar^u.
                let two_ns = 2 * scalar_degree;

                let exponent = (0..two_ns)
                    .step_by(2)
                    .map(|offset| offset + 1)
                    .find(|&candidate| modulus.pow(derived_scalar, candidate as u64) == psi_scalar)
                    .expect("scalar NTT root must belong to the derived large-root subgroup");

                println!(
                    "O18D_ROOT_RELATION_N{}_D{}_LIMB{}_U={}",
                    large_degree, dimension, limb_index, exponent
                );

                assert_eq!(modulus.pow(derived_scalar, exponent as u64,), psi_scalar);
            }
        }

        println!("O18D_ROOT_RELATION=PASS");
    }

    #[test]
    fn batch_large_inv_crt_ntt_matches_coefficient_domain_exactly() {
        let basis = crate::ring::ModulusBasis::new(sd3b_moduli());

        let large_degree = 64;
        let dimension = 8;
        let scalar_degree = large_degree / dimension;

        let plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), large_degree);

        let make_ciphertext = |seed: u128| {
            let b = crate::ring::RnsPolynomial::from_coefficients(
                basis.moduli().to_vec(),
                &(0..large_degree)
                    .map(|i| {
                        let x = i as u128;
                        seed + 7 * x + 3 * x * x
                    })
                    .collect::<Vec<_>>(),
            );

            let a = crate::ring::RnsPolynomial::from_coefficients(
                basis.moduli().to_vec(),
                &(0..large_degree)
                    .map(|i| {
                        let x = i as u128;
                        seed + 23 + 11 * x + 5 * x * x
                    })
                    .collect::<Vec<_>>(),
            );

            crate::grafting::RnsRlweCiphertext::from_limbs(
                b.residues()
                    .iter()
                    .zip(a.residues())
                    .map(|(b_limb, a_limb)| {
                        crate::rlwe::RlweCiphertext::new(b_limb.clone(), a_limb.clone())
                    })
                    .collect(),
            )
        };

        let input: Vec<_> = (0..dimension)
            .map(|index| make_ciphertext(41 + 19 * index as u128))
            .collect();

        let reference = batch_large_inv_crt(&input, scalar_degree);

        let input_ntt: Vec<_> = input
            .iter()
            .map(|ciphertext| BatchRnsNttRlweCiphertext::from_coefficient(ciphertext, &plan))
            .collect();

        let actual_ntt = batch_large_inv_crt_ntt(&input_ntt, scalar_degree, &plan);

        let actual: Vec<_> = actual_ntt
            .iter()
            .map(|ciphertext| ciphertext.to_coefficient(&plan))
            .collect();

        assert_eq!(
            actual, reference,
            "NTT-domain Batch inverse CRT must exactly match coefficient-domain path"
        );

        println!("BATCH_LARGE_INV_CRT_NTT_EQUIVALENCE=PASS");
    }

    #[test]
    fn batch_large_dft_iterative_matches_recursive_exactly() {
        let basis = crate::ring::ModulusBasis::new(sd3b_moduli());

        let degree = 64;

        let make_ciphertext = |seed: u128| {
            let b = crate::ring::RnsPolynomial::from_coefficients(
                basis.moduli().to_vec(),
                &(0..degree)
                    .map(|i| {
                        let x = i as u128;
                        seed + 7 * x + 3 * x * x
                    })
                    .collect::<Vec<_>>(),
            );

            let a = crate::ring::RnsPolynomial::from_coefficients(
                basis.moduli().to_vec(),
                &(0..degree)
                    .map(|i| {
                        let x = i as u128;
                        seed + 23 + 11 * x + 5 * x * x
                    })
                    .collect::<Vec<_>>(),
            );

            crate::grafting::RnsRlweCiphertext::from_limbs(
                b.residues()
                    .iter()
                    .zip(a.residues())
                    .map(|(b_limb, a_limb)| {
                        crate::rlwe::RlweCiphertext::new(b_limb.clone(), a_limb.clone())
                    })
                    .collect(),
            )
        };

        for dimension in [1_usize, 2, 4, 8, 16] {
            let input: Vec<_> = (0..dimension)
                .map(|index| make_ciphertext(31 + 17 * index as u128))
                .collect();

            for multiplier in [-32_i64, -16, -8, -2, 2, 8, 16, 32] {
                let reference = batch_large_dft(&input, multiplier);

                let actual = batch_large_dft_iterative(&input, multiplier);

                assert_eq!(
                    actual,
                    reference,
                    "iterative Batch large DFT mismatch: dimension={dimension}, multiplier={multiplier}"
                );
            }
        }

        println!("BATCH_LARGE_DFT_ITERATIVE_EQUIVALENCE=PASS");
    }

    #[test]
    fn batch_large_dft_ntt_matches_coefficient_domain_exactly() {
        let basis = crate::ring::ModulusBasis::new(sd3b_moduli());

        let degree = 64;

        let plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), degree);

        const DIMENSION: usize = 8;

        let make_ciphertext = |seed: u128| {
            let b = crate::ring::RnsPolynomial::from_coefficients(
                basis.moduli().to_vec(),
                &(0..degree)
                    .map(|i| {
                        let x = i as u128;
                        seed + 7 * x + 3 * x * x
                    })
                    .collect::<Vec<_>>(),
            );

            let a = crate::ring::RnsPolynomial::from_coefficients(
                basis.moduli().to_vec(),
                &(0..degree)
                    .map(|i| {
                        let x = i as u128;
                        seed + 19 + 11 * x + 5 * x * x
                    })
                    .collect::<Vec<_>>(),
            );

            crate::grafting::RnsRlweCiphertext::from_limbs(
                b.residues()
                    .iter()
                    .zip(a.residues())
                    .map(|(b_limb, a_limb)| {
                        crate::rlwe::RlweCiphertext::new(b_limb.clone(), a_limb.clone())
                    })
                    .collect(),
            )
        };

        let input: Vec<_> = (0..DIMENSION)
            .map(|index| make_ciphertext(31 + 17 * index as u128))
            .collect();

        let ntt_input: Vec<_> = input
            .iter()
            .map(|ciphertext| BatchRnsNttRlweCiphertext::from_coefficient(ciphertext, &plan))
            .collect();

        for multiplier in [-32_i64, -16, -8, -2, 2, 8, 16, 32] {
            let reference = batch_large_dft(&input, multiplier);

            let actual_ntt = batch_large_dft_ntt(&ntt_input, multiplier, &plan);

            let actual: Vec<_> = actual_ntt
                .iter()
                .map(|ciphertext| ciphertext.to_coefficient(&plan))
                .collect();

            assert_eq!(
                actual, reference,
                "NTT-domain large DFT mismatch for multiplier={multiplier}"
            );
        }

        println!("BATCH_LARGE_DFT_NTT_EQUIVALENCE=PASS");
    }

    #[test]
    fn batch_ntt_rlwe_roundtrip_matches_coefficient_exactly() {
        let basis = crate::ring::ModulusBasis::new(sd3b_moduli());
        let degree = 64;

        let plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), degree);

        let make_poly = |offset: u128| {
            crate::ring::RnsPolynomial::from_coefficients(
                basis.moduli().to_vec(),
                &(0..degree)
                    .map(|i| {
                        let x = i as u128;
                        offset + 7 * x + 3 * x * x
                    })
                    .collect::<Vec<_>>(),
            )
        };

        let b = make_poly(11);
        let a = make_poly(29);

        let ciphertext = crate::grafting::RnsRlweCiphertext::from_limbs(
            b.residues()
                .iter()
                .zip(a.residues())
                .map(|(b_limb, a_limb)| {
                    crate::rlwe::RlweCiphertext::new(b_limb.clone(), a_limb.clone())
                })
                .collect(),
        );

        let ntt = BatchRnsNttRlweCiphertext::from_coefficient(&ciphertext, &plan);

        let recovered = ntt.to_coefficient(&plan);

        assert_eq!(recovered, ciphertext);

        println!("BATCH_NTT_RLWE_ROUNDTRIP=PASS");
    }

    #[test]
    fn batch_ccmm_ntt_modular_product_matches_reference() {
        const DIMENSION: usize = 4;
        const INNER: usize = 4;
        const DEGREE: usize = 8;

        let moduli = vec![
            crate::ring::Modulus::new(193),
            crate::ring::Modulus::new(257),
        ];

        let plan = crate::ring::RnsNttPlan::new(moduli.clone(), DEGREE);

        fn make_poly(moduli: &[crate::ring::Modulus], seed: usize) -> crate::ring::RnsPolynomial {
            let coefficients: Vec<u128> = (0..DEGREE)
                .map(|index| ((seed + 1) * 17 + index * 11) as u128)
                .collect();

            crate::ring::RnsPolynomial::from_coefficients(moduli.to_vec(), &coefficients)
        }

        let lhs: Vec<Vec<_>> = (0..2 * DIMENSION)
            .map(|row| {
                (0..INNER)
                    .map(|column| make_poly(&moduli, row * INNER + column))
                    .collect()
            })
            .collect();

        let rhs: Vec<Vec<_>> = (0..INNER)
            .map(|row| {
                (0..2 * DIMENSION)
                    .map(|column| make_poly(&moduli, 100 + row * 2 * DIMENSION + column))
                    .collect()
            })
            .collect();

        let reference = batch_ccmm_modular_product(&lhs, &rhs, &plan);

        let canonical = batch_ccmm_modular_product_ntt(&lhs, &rhs, &plan);

        let prepared_plan = crate::ring::PreparedRnsNttPlan::new(&plan);

        let prepared = batch_ccmm_modular_product_prepared_ntt(&lhs, &rhs, &prepared_plan);

        let delayed = batch_ccmm_modular_product_delayed_ntt(&lhs, &rhs, &prepared_plan);

        assert_eq!(canonical, reference);
        assert_eq!(prepared, reference);
        assert_eq!(prepared, canonical);
        assert_eq!(delayed, reference);
        assert_eq!(delayed, prepared);

        println!("BATCH_CCMM_NTT_MODULAR_PRODUCT_EQUIVALENCE=PASS");
        println!("BATCH_CCMM_PREPARED_NTT_MODULAR_PRODUCT_EQUIVALENCE=PASS");
        println!("BATCH_CCMM_DELAYED_NTT_MODULAR_PRODUCT_EQUIVALENCE=PASS");
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

        let prepared_galois_keys: Vec<crate::ckks::PreparedRnsGaloisKey> = galois_keys
            .iter()
            .map(|key| crate::ckks::PreparedRnsGaloisKey::new(key, &large_plan))
            .collect();

        // ------------------------------------------------------------------
        // Authors MatrixEvaluator::matrixMult pipeline.
        // ------------------------------------------------------------------

        // C-MT rhs.
        let rhs_t =
            batch_ciphertext_transpose(&rhs, SCALAR_DEGREE, &prepared_galois_keys, &large_plan);

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
        let former = batch_ciphertext_transpose(
            &former_t,
            SCALAR_DEGREE,
            &prepared_galois_keys,
            &large_plan,
        );

        let latter = batch_ciphertext_transpose(
            &latter_t,
            SCALAR_DEGREE,
            &prepared_galois_keys,
            &large_plan,
        );

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
        let total_start = std::time::Instant::now();
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

        let phase_start = std::time::Instant::now();
        let lhs_sinc = super::sinc_encode_batch(&lhs_complex, scalar_degree);

        let rhs_sinc = super::sinc_encode_batch(&rhs_complex, scalar_degree);

        let encode_elapsed = phase_start.elapsed();
        assert_eq!(lhs_sinc.large_degree(), LARGE_DEGREE);
        assert_eq!(rhs_sinc.large_degree(), LARGE_DEGREE);

        assert_eq!(lhs_sinc.dimension(), dimension);
        assert_eq!(rhs_sinc.dimension(), dimension);

        assert_eq!(lhs_sinc.num_columns(), dimension);
        assert_eq!(rhs_sinc.num_columns(), dimension);

        let basis = crate::ring::ModulusBasis::new(sd3b_moduli());

        let chain = crate::ring::ModulusChain::from_top_basis(basis.clone());

        let large_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), LARGE_DEGREE);

        let prepared_large_plan = crate::ring::PreparedRnsNttPlan::new(&large_plan);

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

        let phase_start = std::time::Instant::now();
        let lhs = encrypt_columns(lhs_sinc.columns(), 0x4343_4d4d_4c48_5300);

        let rhs = encrypt_columns(rhs_sinc.columns(), 0x4343_4d4d_5248_5300);

        let encrypt_elapsed = phase_start.elapsed();
        assert_eq!(lhs.len(), dimension);
        assert_eq!(rhs.len(), dimension);

        // ----------------------------------------------------------
        // Evaluation keys for authors' C-MT.
        //
        // h_i = 2 * Ns * i + 1, i = 1..d-1.
        // ----------------------------------------------------------

        let phase_start = std::time::Instant::now();
        let galois_layout = crate::grafting::RnsGadgetLayout::new(basis.clone(), vec![1, 1]);

        let galois_keys: Vec<crate::ckks::RnsGaloisKey> = (1..dimension)
            .map(|index| 2 * scalar_degree * index + 1)
            .map(|exponent| {
                let mut rng = ChaCha20Rng::seed_from_u64(0x434d_5400_0000_0000 ^ exponent as u64);

                crate::ckks::RnsGaloisKey::generate_with_ntt_rng(
                    crate::grafting::RnsKeygenConfig {
                        degree: LARGE_DEGREE,
                        plaintext_modulus: 2,
                        noise_bound: 0,
                        layout: galois_layout.clone(),
                        plan: &large_plan,
                    },
                    &secret,
                    exponent,
                    &mut rng,
                )
            })
            .collect();

        assert_eq!(galois_keys.len(), dimension - 1,);
        let galois_keygen_elapsed = phase_start.elapsed();

        let phase_start = std::time::Instant::now();

        let prepared_galois_keys: Vec<crate::ckks::PreparedRnsGaloisKey> = galois_keys
            .iter()
            .map(|key| crate::ckks::PreparedRnsGaloisKey::new(key, &large_plan))
            .collect();

        assert_eq!(prepared_galois_keys.len(), dimension - 1,);
        let galois_prepare_elapsed = phase_start.elapsed();

        // ----------------------------------------------------------
        // Authors' CCMM pipeline.
        //
        // Scalar-plan preparation and evaluation-key preparation remain
        // outside the authors-equivalent MatrixEvaluator::matrixMult
        // timing boundary.
        // ----------------------------------------------------------

        let phase_start = std::time::Instant::now();

        let prepared_scalar_plan = crate::ring::PreparedRnsNttPlan::new(&scalar_plan);

        let scalar_ntt_prepare_elapsed = phase_start.elapsed();

        // ----------------------------------------------------------
        // Multiplication evaluation key preparation.
        // ----------------------------------------------------------

        let phase_start = std::time::Instant::now();

        let multiplication_layout =
            crate::grafting::RnsGadgetLayout::new(basis.clone(), vec![1, 1]);

        let mut multiplication_key_rng = ChaCha20Rng::seed_from_u64(0x4343_4d4d_524c_4b00);

        let multiplication_key = crate::grafting::RnsMultiplicationKey::generate_with_ntt_rng(
            crate::grafting::RnsKeygenConfig {
                degree: LARGE_DEGREE,
                plaintext_modulus: 2,
                noise_bound: 0,
                layout: multiplication_layout,
                plan: &large_plan,
            },
            &secret,
            &mut multiplication_key_rng,
        );

        let multiplication_keygen_elapsed = phase_start.elapsed();

        let phase_start = std::time::Instant::now();

        let prepared_multiplication_key =
            crate::grafting::PreparedRnsMultiplicationKey::new(&multiplication_key, &large_plan);

        let multiplication_key_prepare_elapsed = phase_start.elapsed();

        let context = super::BatchCcmmExecutionContext {
            scalar_plan: &scalar_plan,
            prepared_scalar_plan: &prepared_scalar_plan,
            prepared_large_plan: &prepared_large_plan,
            prepared_galois_keys: &prepared_galois_keys,
            prepared_multiplication_key: &prepared_multiplication_key,
            chain: &chain,
            scale,
        };

        let geometry = crate::eblas::BatchGemmGeometry::new(
            crate::eblas::BatchGemmMechanism::Ccmm,
            dimension,
            scalar_degree,
            LARGE_DEGREE,
        );

        let matrix_mult_start = std::time::Instant::now();

        let output_ciphertexts = crate::eblas::batch_gemm_ccmm(geometry, &lhs, &rhs, &context);

        let matrix_mult_elapsed = matrix_mult_start.elapsed();

        assert!(output_ciphertexts
            .iter()
            .all(|ciphertext| { ciphertext.level() == 1 }));

        // Full-row CCMM decode.  Unlike CPMM, dimension is d,
        // not d/2.
        let phase_start = std::time::Instant::now();
        let decoded = decode_large_columns(&output_ciphertexts, &secret, scalar_degree, dimension);

        let actual = super::sinc_decode_batch(&decoded);

        assert_eq!(actual.len(), batch_count);
        assert_eq!(actual[0].len(), dimension);
        assert_eq!(actual[0][0].len(), dimension);

        let decrypt_decode_elapsed = phase_start.elapsed();
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

        let total_elapsed = total_start.elapsed();

        println!(
            "BATCH_CCMM_PHASE_ENCODE_MS={:.3}",
            encode_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_PHASE_ENCRYPT_MS={:.3}",
            encrypt_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_PHASE_GALOIS_KEYGEN_MS={:.3}",
            galois_keygen_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_PHASE_GALOIS_PREPARE_MS={:.3}",
            galois_prepare_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_PHASE_SCALAR_NTT_PREPARE_MS={:.3}",
            scalar_ntt_prepare_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_PHASE_MULT_KEYGEN_MS={:.3}",
            multiplication_keygen_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_PHASE_MULT_KEY_PREPARE_MS={:.3}",
            multiplication_key_prepare_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_PHASE_DECRYPT_DECODE_MS={:.3}",
            decrypt_decode_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_MATRIX_MULT_MS={:.3}",
            matrix_mult_elapsed.as_secs_f64() * 1.0e3
        );
        println!(
            "BATCH_CCMM_TOTAL_MS={:.3}",
            total_elapsed.as_secs_f64() * 1.0e3
        );

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

    fn ccmm_batch_gemv_dot_capability(
        dimension: usize,
        scalar_degree: usize,
        dot: bool,
    ) -> (f64, f64, f64) {
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

        let lhs_source =
            deterministic_cpmm_input(batch_count, dimension, if dot { 23 } else { 19 });
        let rhs_source =
            deterministic_cpmm_input(batch_count, dimension, if dot { 29 } else { 21 });

        let (lhs_clear, rhs_clear, expected_vectors, expected_scalars) = if dot {
            let lhs_vectors: Vec<Vec<f64>> = lhs_source
                .iter()
                .map(|matrix| (0..dimension).map(|column| matrix[0][column]).collect())
                .collect();

            let rhs_vectors: Vec<Vec<f64>> = rhs_source
                .iter()
                .map(|matrix| (0..dimension).map(|row| matrix[row][0]).collect())
                .collect();

            let expected_scalars: Vec<f64> = lhs_vectors
                .iter()
                .zip(&rhs_vectors)
                .map(|(lhs, rhs)| (0..dimension).map(|index| lhs[index] * rhs[index]).sum())
                .collect();

            let lhs_clear: Vec<Vec<Vec<f64>>> = lhs_vectors
                .iter()
                .map(|vector| {
                    (0..dimension)
                        .map(|row| {
                            if row == 0 {
                                vector.clone()
                            } else {
                                vec![0.0_f64; dimension]
                            }
                        })
                        .collect()
                })
                .collect();

            let rhs_clear: Vec<Vec<Vec<f64>>> = rhs_vectors
                .iter()
                .map(|vector| {
                    (0..dimension)
                        .map(|row| {
                            let mut embedded_row = vec![0.0_f64; dimension];
                            embedded_row[0] = vector[row];
                            embedded_row
                        })
                        .collect()
                })
                .collect();

            (lhs_clear, rhs_clear, Vec::new(), expected_scalars)
        } else {
            let vectors: Vec<Vec<f64>> = rhs_source
                .iter()
                .map(|matrix| (0..dimension).map(|row| matrix[row][0]).collect())
                .collect();

            let expected_vectors: Vec<Vec<f64>> = lhs_source
                .iter()
                .zip(&vectors)
                .map(|(matrix, vector)| {
                    (0..dimension)
                        .map(|row| {
                            (0..dimension)
                                .map(|inner| matrix[row][inner] * vector[inner])
                                .sum()
                        })
                        .collect()
                })
                .collect();

            let rhs_clear: Vec<Vec<Vec<f64>>> = vectors
                .iter()
                .map(|vector| {
                    (0..dimension)
                        .map(|row| {
                            let mut embedded_row = vec![0.0_f64; dimension];
                            embedded_row[0] = vector[row];
                            embedded_row
                        })
                        .collect()
                })
                .collect();

            (lhs_source, rhs_clear, expected_vectors, Vec::new())
        };

        let to_complex = |matrices: &[Vec<Vec<f64>>]| {
            matrices
                .iter()
                .map(|matrix| {
                    matrix
                        .iter()
                        .map(|row| {
                            row.iter()
                                .map(|&value| num_complex::Complex64::new(value, 0.0))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };

        let lhs_complex = to_complex(&lhs_clear);
        let rhs_complex = to_complex(&rhs_clear);

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
        let prepared_large_plan = crate::ring::PreparedRnsNttPlan::new(&large_plan);

        let scalar_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), scalar_degree);
        let prepared_scalar_plan = crate::ring::PreparedRnsNttPlan::new(&scalar_plan);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(
            0x4343_4d4d_5644_0000 ^ dimension as u64 ^ if dot { 0x444f_5400 } else { 0x4745_4d56 },
        );

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

        let lhs = encrypt_columns(
            lhs_sinc.columns(),
            0x4343_5644_4c48_5300 ^ if dot { 1 } else { 0 },
        );
        let rhs = encrypt_columns(
            rhs_sinc.columns(),
            0x4343_5644_5248_5300 ^ if dot { 1 } else { 0 },
        );

        let galois_layout = crate::grafting::RnsGadgetLayout::new(basis.clone(), vec![1, 1]);

        let galois_keys: Vec<crate::ckks::RnsGaloisKey> = (1..dimension)
            .map(|index| 2 * scalar_degree * index + 1)
            .map(|exponent| {
                let mut rng = ChaCha20Rng::seed_from_u64(0x434d_5400_5644_0000 ^ exponent as u64);

                crate::ckks::RnsGaloisKey::generate_with_ntt_rng(
                    crate::grafting::RnsKeygenConfig {
                        degree: LARGE_DEGREE,
                        plaintext_modulus: 2,
                        noise_bound: 0,
                        layout: galois_layout.clone(),
                        plan: &large_plan,
                    },
                    &secret,
                    exponent,
                    &mut rng,
                )
            })
            .collect();

        let prepared_galois_keys: Vec<crate::ckks::PreparedRnsGaloisKey> = galois_keys
            .iter()
            .map(|key| crate::ckks::PreparedRnsGaloisKey::new(key, &large_plan))
            .collect();

        let multiplication_layout =
            crate::grafting::RnsGadgetLayout::new(basis.clone(), vec![1, 1]);

        let mut multiplication_key_rng = ChaCha20Rng::seed_from_u64(0x4343_5644_524c_4b00);

        let multiplication_key = crate::grafting::RnsMultiplicationKey::generate_with_ntt_rng(
            crate::grafting::RnsKeygenConfig {
                degree: LARGE_DEGREE,
                plaintext_modulus: 2,
                noise_bound: 0,
                layout: multiplication_layout,
                plan: &large_plan,
            },
            &secret,
            &mut multiplication_key_rng,
        );

        let prepared_multiplication_key =
            crate::grafting::PreparedRnsMultiplicationKey::new(&multiplication_key, &large_plan);

        let context = super::BatchCcmmExecutionContext {
            scalar_plan: &scalar_plan,
            prepared_scalar_plan: &prepared_scalar_plan,
            prepared_large_plan: &prepared_large_plan,
            prepared_galois_keys: &prepared_galois_keys,
            prepared_multiplication_key: &prepared_multiplication_key,
            chain: &chain,
            scale,
        };

        let geometry = crate::eblas::BatchGemmGeometry::new(
            crate::eblas::BatchGemmMechanism::Ccmm,
            dimension,
            scalar_degree,
            LARGE_DEGREE,
        );

        let operation_start = std::time::Instant::now();

        let output = if dot {
            crate::eblas::batch_dot_ccmm(geometry, &lhs, &rhs, &context)
        } else {
            crate::eblas::batch_gemv_ccmm(geometry, &lhs, &rhs, &context)
        };

        let operation_elapsed = operation_start.elapsed();

        assert_eq!(
            output.level(),
            1,
            "Batch CCMM GEMV/DOT must consume exactly one modulus level"
        );
        assert_eq!(
            output.rlwe().degree(),
            LARGE_DEGREE,
            "Batch CCMM GEMV/DOT must preserve the large-ring degree"
        );
        assert!(
            output.scale().is_finite() && output.scale() > 0.0,
            "Batch CCMM GEMV/DOT output scale must be finite and positive"
        );

        let decoded = decode_large_columns(
            std::slice::from_ref(&output),
            &secret,
            scalar_degree,
            dimension,
        );

        let actual = super::sinc_decode_batch(&decoded);

        assert_eq!(actual.len(), batch_count);

        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        let mut max_abs_error = 0.0_f64;
        let mut max_structural_zero = 0.0_f64;
        let mut max_imaginary = 0.0_f64;

        for batch in 0..batch_count {
            let matrix = &actual[batch];

            assert_eq!(matrix.len(), dimension);
            assert!(
                matrix.iter().all(|row| row.len() == 1),
                "Batch CCMM GEMV/DOT must decode exactly one logical column"
            );

            for (row, matrix_row) in matrix.iter().enumerate() {
                let value = matrix_row[0];

                assert!(
                    value.re.is_finite() && value.im.is_finite(),
                    "Batch CCMM GEMV/DOT decoded a non-finite value"
                );

                max_imaginary = max_imaginary.max(value.im.abs());

                let expected = if dot {
                    if row == 0 {
                        expected_scalars[batch]
                    } else {
                        0.0
                    }
                } else {
                    expected_vectors[batch][row]
                };

                let error = value.re - expected;

                if !dot || row == 0 {
                    squared_error += error * error;
                    squared_reference += expected * expected;
                    max_abs_error = max_abs_error.max(error.abs());
                } else {
                    max_structural_zero = max_structural_zero.max(value.norm());
                }
            }
        }

        assert!(
            squared_reference > 0.0,
            "Batch CCMM GEMV/DOT reference norm must be nonzero"
        );

        let relative_l2 = (squared_error / squared_reference).sqrt();
        let prefix = if dot {
            "BATCH_CCMM_DOT"
        } else {
            "BATCH_CCMM_GEMV"
        };

        println!("{prefix}_DIMENSION={dimension}");
        println!("{prefix}_SCALAR_DEGREE={scalar_degree}");
        println!("{prefix}_BATCH_COUNT={batch_count}");
        println!(
            "{prefix}_OUTPUT_VALUES={}",
            if dot {
                batch_count
            } else {
                batch_count * dimension
            }
        );
        println!(
            "{prefix}_MATRIX_MULT_MS={:.3}",
            operation_elapsed.as_secs_f64() * 1.0e3
        );
        println!("{prefix}_REL_L2={relative_l2:.12e}");
        println!("{prefix}_MAX_ABS_ERROR={max_abs_error:.12e}");
        println!("{prefix}_MAX_STRUCTURAL_ZERO={max_structural_zero:.12e}");
        println!("{prefix}_MAX_IMAGINARY={max_imaginary:.12e}");

        (
            relative_l2,
            max_abs_error,
            max_structural_zero.max(max_imaginary),
        )
    }

    #[test]
    fn batch_ccmm_gemv_d64_matches_clear_reference() {
        let (relative_l2, max_abs_error, residual) = ccmm_batch_gemv_dot_capability(64, 128, false);

        println!("BATCH_CCMM_GEMV_D64_REL_L2={relative_l2:.12e}");
        println!("BATCH_CCMM_GEMV_D64_MAX_ABS_ERROR={max_abs_error:.12e}");
        println!("BATCH_CCMM_GEMV_D64_MAX_RESIDUAL={residual:.12e}");

        assert!(
            relative_l2 < 1.0e-3,
            "authors-scale Batch CCMM GEMV d=64 relative L2 error {relative_l2:e}"
        );
        assert!(
            residual < 1.0e-3,
            "authors-scale Batch CCMM GEMV d=64 residual {residual:e}"
        );

        println!("BATCH_CCMM_GEMV_D64_STATUS=PASS");
    }

    #[test]
    fn batch_ccmm_gemv_d128_matches_clear_reference() {
        let (relative_l2, max_abs_error, residual) = ccmm_batch_gemv_dot_capability(128, 64, false);

        println!("BATCH_CCMM_GEMV_D128_REL_L2={relative_l2:.12e}");
        println!("BATCH_CCMM_GEMV_D128_MAX_ABS_ERROR={max_abs_error:.12e}");
        println!("BATCH_CCMM_GEMV_D128_MAX_RESIDUAL={residual:.12e}");

        assert!(
            relative_l2 < 1.0e-3,
            "authors-scale Batch CCMM GEMV d=128 relative L2 error {relative_l2:e}"
        );
        assert!(
            residual < 1.0e-3,
            "authors-scale Batch CCMM GEMV d=128 residual {residual:e}"
        );

        println!("BATCH_CCMM_GEMV_D128_STATUS=PASS");
    }

    #[test]
    fn batch_ccmm_dot_d64_matches_clear_reference() {
        let (relative_l2, max_abs_error, residual) = ccmm_batch_gemv_dot_capability(64, 128, true);

        println!("BATCH_CCMM_DOT_D64_REL_L2={relative_l2:.12e}");
        println!("BATCH_CCMM_DOT_D64_MAX_ABS_ERROR={max_abs_error:.12e}");
        println!("BATCH_CCMM_DOT_D64_MAX_RESIDUAL={residual:.12e}");

        assert!(
            relative_l2 < 1.0e-3,
            "authors-scale Batch CCMM DOT d=64 relative L2 error {relative_l2:e}"
        );
        assert!(
            residual < 1.0e-3,
            "authors-scale Batch CCMM DOT d=64 residual {residual:e}"
        );

        println!("BATCH_CCMM_DOT_D64_STATUS=PASS");
    }

    #[test]
    fn batch_ccmm_dot_d128_matches_clear_reference() {
        let (relative_l2, max_abs_error, residual) = ccmm_batch_gemv_dot_capability(128, 64, true);

        println!("BATCH_CCMM_DOT_D128_REL_L2={relative_l2:.12e}");
        println!("BATCH_CCMM_DOT_D128_MAX_ABS_ERROR={max_abs_error:.12e}");
        println!("BATCH_CCMM_DOT_D128_MAX_RESIDUAL={residual:.12e}");

        assert!(
            relative_l2 < 1.0e-3,
            "authors-scale Batch CCMM DOT d=128 relative L2 error {relative_l2:e}"
        );
        assert!(
            residual < 1.0e-3,
            "authors-scale Batch CCMM DOT d=128 residual {residual:e}"
        );

        println!("BATCH_CCMM_DOT_D128_STATUS=PASS");
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
