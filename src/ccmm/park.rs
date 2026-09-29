//! Park-style encrypted matrix multiplication primitives.
//!
//! This module implements the reduction machinery underlying the
//! ciphertext-matrix algorithms of Park et al.
//!
//! The first implemented primitive is `Tweak`, Algorithm 3 in the
//! ciphertext-matrix-transpose construction.
//!
//! This module is intentionally separate from the existing reference/native
//! CPMM and CCMM implementations.

use crate::ring::Polynomial;
use crate::rlwe::RlweCiphertext;

/// Adds two RLWE ciphertexts componentwise.
fn ciphertext_add(lhs: &RlweCiphertext, rhs: &RlweCiphertext) -> RlweCiphertext {
    RlweCiphertext::new(lhs.b().add(rhs.b()), lhs.a().add(rhs.a()))
}

/// Subtracts two RLWE ciphertexts componentwise.
fn ciphertext_sub(lhs: &RlweCiphertext, rhs: &RlweCiphertext) -> RlweCiphertext {
    RlweCiphertext::new(lhs.b().sub(rhs.b()), lhs.a().sub(rhs.a()))
}

/// Returns `X^exponent * polynomial` in `Z_q[X]/(X^N + 1)`.
///
/// The exponent is interpreted modulo `2N`, using `X^N = -1`.
fn polynomial_mul_monomial(polynomial: &Polynomial, exponent: usize) -> Polynomial {
    let degree = polynomial.degree();
    let modulus = polynomial.modulus();

    let exponent = exponent % (2 * degree);
    let negative = exponent >= degree;
    let shift = exponent % degree;

    let mut monomial = vec![0_u64; degree];

    monomial[shift] = if negative { modulus.value() - 1 } else { 1 };

    polynomial.negacyclic_mul(&Polynomial::new(modulus, monomial))
}

/// Returns `X^exponent * ciphertext`.
fn ciphertext_mul_monomial(ciphertext: &RlweCiphertext, exponent: usize) -> RlweCiphertext {
    RlweCiphertext::new(
        polynomial_mul_monomial(ciphertext.b(), exponent),
        polynomial_mul_monomial(ciphertext.a(), exponent),
    )
}

/// Multiplicative inverse modulo `modulus`.
///
/// This is used by Park C-MT both for:
///
/// - `N^{-1} mod q`; and
/// - `(2j+1)^{-1} mod 2N`.
fn modular_inverse(value: u64, modulus: u64) -> u64 {
    assert!(modulus > 1, "modulus must exceed one");

    let mut t: i128 = 0;
    let mut new_t: i128 = 1;
    let mut r: i128 = modulus as i128;
    let mut new_r: i128 = (value % modulus) as i128;

    while new_r != 0 {
        let quotient = r / new_r;

        let next_t = t - quotient * new_t;
        t = new_t;
        new_t = next_t;

        let next_r = r - quotient * new_r;
        r = new_r;
        new_r = next_r;
    }

    assert_eq!(r, 1, "value must be invertible modulo modulus");

    let modulus_i128 = modulus as i128;
    let inverse = ((t % modulus_i128) + modulus_i128) % modulus_i128;

    inverse as u64
}

/// Multiplies both RLWE components by a scalar modulo q.
fn ciphertext_scalar_mul(ciphertext: &RlweCiphertext, scalar: u64) -> RlweCiphertext {
    RlweCiphertext::new(
        ciphertext.b().scalar_mul(scalar),
        ciphertext.a().scalar_mul(scalar),
    )
}

/// Returns the Galois key for an odd exponent.
///
/// Exponent one is the identity automorphism and therefore needs no key.
fn galois_key_for_exponent(
    keys: &[crate::ckks::GaloisKey],
    exponent: usize,
) -> &crate::ckks::GaloisKey {
    keys.iter()
        .find(|key| key.exponent() == exponent)
        .unwrap_or_else(|| panic!("missing Park C-MT Galois key for exponent {exponent}"))
}

/// Park Algorithm 3: Tweak.
///
/// For `n` ciphertexts over ring degree `N`, where both `n` and `N` are
/// powers of two and `n` divides `N`, returns:
///
/// ```text
/// out[j] = sum_i X^(2*i*j*N/n) * input[i].
/// ```
///
/// This is the structured divide-and-conquer transform used twice by the
/// Park ciphertext-matrix-transpose algorithm.
pub fn tweak(ciphertexts: &[RlweCiphertext]) -> Vec<RlweCiphertext> {
    assert!(
        !ciphertexts.is_empty(),
        "Tweak requires at least one ciphertext"
    );

    let n = ciphertexts.len();

    assert!(
        n.is_power_of_two(),
        "Tweak ciphertext count must be a power of two"
    );

    let degree = ciphertexts[0].b().degree();

    assert!(
        degree.is_power_of_two(),
        "Tweak ring degree must be a power of two"
    );

    assert!(
        n <= degree && degree % n == 0,
        "Tweak ciphertext count must divide ring degree"
    );

    let modulus = ciphertexts[0].b().modulus();

    for ciphertext in ciphertexts {
        assert_eq!(
            ciphertext.b().degree(),
            degree,
            "Tweak ciphertext degrees must match"
        );
        assert_eq!(
            ciphertext.b().modulus(),
            modulus,
            "Tweak ciphertext moduli must match"
        );
        assert_eq!(ciphertext.a().degree(), degree);
        assert_eq!(ciphertext.a().modulus(), modulus);
    }

    if n == 1 {
        return vec![ciphertexts[0].clone()];
    }

    // Algorithm 3 progressively doubles the solved subproblem.
    let mut output = vec![ciphertexts[0].clone(); n];
    output[0] = ciphertexts[0].clone();

    let log_n = n.trailing_zeros() as usize;

    for ell in 0..log_n {
        let width = 1usize << ell;
        let stride = n >> (ell + 1);

        let odd_inputs: Vec<RlweCiphertext> = (0..width)
            .map(|j| ciphertexts[(2 * j + 1) * stride].clone())
            .collect();

        let aux = tweak(&odd_inputs);

        // Preserve the previous stage because both branches depend on it.
        let previous: Vec<RlweCiphertext> = output[..width].to_vec();

        for j in 0..width {
            let exponent = j * degree / width;
            let rotated = ciphertext_mul_monomial(&aux[j], exponent);

            output[j] = ciphertext_add(&previous[j], &rotated);
            output[j + width] = ciphertext_sub(&previous[j], &rotated);
        }
    }

    output
}

/// RNS form of Park Algorithm 3: Tweak.
///
/// The Park Tweak transform contains only ciphertext addition/subtraction and
/// multiplication by public monomials. These operations are defined
/// independently in every CRT/RNS residue, so the RNS transform is exactly the
/// collection of scalar Park Tweak transforms over the active modulus basis.
///
/// This function intentionally uses the scalar implementation as the
/// specification path. RNS/NTT specialization is a separate optimization.
pub fn rns_tweak(
    ciphertexts: &[crate::grafting::RnsRlweCiphertext],
) -> Vec<crate::grafting::RnsRlweCiphertext> {
    assert!(
        !ciphertexts.is_empty(),
        "RNS Tweak requires at least one ciphertext"
    );

    let n = ciphertexts.len();
    let degree = ciphertexts[0].degree();
    let basis = ciphertexts[0].basis().clone();

    assert!(
        ciphertexts
            .iter()
            .all(|ciphertext| ciphertext.degree() == degree),
        "RNS Tweak ciphertext degrees must match"
    );

    assert!(
        ciphertexts
            .iter()
            .all(|ciphertext| ciphertext.basis() == &basis),
        "RNS Tweak ciphertext bases must match"
    );

    let limb_count = basis.len();

    let transformed_by_limb: Vec<Vec<RlweCiphertext>> = (0..limb_count)
        .map(|limb_index| {
            let limb_inputs: Vec<RlweCiphertext> = ciphertexts
                .iter()
                .map(|ciphertext| ciphertext.limb(limb_index).clone())
                .collect();

            tweak(&limb_inputs)
        })
        .collect();

    (0..n)
        .map(|ciphertext_index| {
            let limbs = transformed_by_limb
                .iter()
                .map(|limb_outputs| limb_outputs[ciphertext_index].clone())
                .collect();

            crate::grafting::RnsRlweCiphertext::from_limbs(limbs)
        })
        .collect()
}

fn rns_ciphertext_mul_monomial(
    ciphertext: &crate::grafting::RnsRlweCiphertext,
    exponent: usize,
) -> crate::grafting::RnsRlweCiphertext {
    let limbs = ciphertext
        .limbs()
        .iter()
        .map(|limb| ciphertext_mul_monomial(limb, exponent))
        .collect();

    crate::grafting::RnsRlweCiphertext::from_limbs(limbs)
}

fn rns_ciphertext_scalar_mul(
    ciphertext: &crate::grafting::RnsRlweCiphertext,
    scalars: &[u64],
) -> crate::grafting::RnsRlweCiphertext {
    assert_eq!(
        ciphertext.limbs().len(),
        scalars.len(),
        "RNS scalar multiplication requires one scalar per limb"
    );

    let limbs = ciphertext
        .limbs()
        .iter()
        .zip(scalars)
        .map(|(limb, &scalar)| ciphertext_scalar_mul(limb, scalar))
        .collect();

    crate::grafting::RnsRlweCiphertext::from_limbs(limbs)
}

fn rns_galois_key_for_exponent(
    keys: &[crate::ckks::RnsGaloisKey],
    exponent: usize,
) -> &crate::ckks::RnsGaloisKey {
    keys.iter()
        .find(|key| key.exponent() == exponent)
        .unwrap_or_else(|| panic!("missing RNS Galois key for exponent {exponent}"))
}

/// RNS form of Park Algorithm 4: ciphertext matrix transpose (C-MT).
///
/// The public polynomial arithmetic is performed independently in each active
/// CRT limb. The automorphism/key-switch step remains RNS-native through
/// `RnsGaloisKey` and `apply_rns_galois_automorphism`.
///
/// Input and output follow the same row/column packing convention as
/// [`transpose`].
pub fn rns_transpose(
    ciphertexts: &[crate::grafting::RnsRlweCiphertext],
    galois_keys: &[crate::ckks::RnsGaloisKey],
) -> Vec<crate::grafting::RnsRlweCiphertext> {
    assert!(!ciphertexts.is_empty(), "RNS C-MT requires ciphertexts");

    let degree = ciphertexts[0].degree();
    let basis = ciphertexts[0].basis().clone();

    assert_eq!(
        ciphertexts.len(),
        degree,
        "Park Algorithm 4 requires N ciphertexts for ring degree N"
    );

    assert!(
        degree.is_power_of_two(),
        "Park RNS C-MT requires power-of-two ring degree"
    );

    assert!(
        ciphertexts
            .iter()
            .all(|ciphertext| ciphertext.degree() == degree),
        "RNS C-MT ciphertext degrees must match"
    );

    assert!(
        ciphertexts
            .iter()
            .all(|ciphertext| ciphertext.basis() == &basis),
        "RNS C-MT ciphertext bases must match"
    );

    assert!(
        basis
            .moduli()
            .iter()
            .all(|modulus| modulus.value() % 2 == 1),
        "Park RNS C-MT requires N invertible modulo every active modulus"
    );

    let two_n = 2 * degree;

    // Algorithm 4, Step 1:
    //
    // aux <- Tweak((X^i * ct_i)_i)
    let shifted_inputs: Vec<crate::grafting::RnsRlweCiphertext> = ciphertexts
        .iter()
        .enumerate()
        .map(|(i, ciphertext)| rns_ciphertext_mul_monomial(ciphertext, i))
        .collect();

    let aux = rns_tweak(&shifted_inputs);

    // Algorithm 4, Steps 2--5.
    //
    // N^{-1} is represented independently in every CRT limb.
    let n_inverses: Vec<u64> = basis
        .moduli()
        .iter()
        .map(|modulus| modular_inverse(degree as u64, modulus.value()))
        .collect();

    let mut transformed = Vec::with_capacity(degree);

    for j in 0..degree {
        let exponent = 2 * j + 1;

        let inverse_exponent = modular_inverse(exponent as u64, two_n as u64) as usize;

        assert_eq!(
            inverse_exponent % 2,
            1,
            "inverse of odd RNS C-MT exponent must remain odd"
        );

        let source_index = (inverse_exponent - 1) / 2;
        let normalized = rns_ciphertext_scalar_mul(&aux[source_index], &n_inverses);

        let automorphed = if exponent == 1 {
            normalized
        } else {
            let key = rns_galois_key_for_exponent(galois_keys, exponent);

            crate::ckks::apply_rns_galois_automorphism(&normalized, key)
        };

        transformed.push(automorphed);
    }

    // Algorithm 4, Step 6.
    let second_tweak = rns_tweak(&transformed);

    // Algorithm 4, Steps 7--9.
    //
    // Preserve the scalar implementation's experimentally validated boundary
    // treatment: output 0 is copied directly, while the remaining outputs
    // receive Park's negacyclic monomial/sign correction.
    let mut output = Vec::with_capacity(degree);
    output.push(second_tweak[0].clone());

    for output_index in 1..degree {
        let source_index = degree - output_index;
        let exponent = degree - output_index;

        let corrected = rns_ciphertext_mul_monomial(&second_tweak[source_index], exponent);

        let minus_one: Vec<u64> = basis
            .moduli()
            .iter()
            .map(|modulus| modulus.value() - 1)
            .collect();

        output.push(rns_ciphertext_scalar_mul(&corrected, &minus_one));
    }

    output
}

/// Park Algorithm 4: ciphertext matrix transpose (C-MT).
///
/// Input consists of `N` RLWE ciphertexts over a ring of degree `N`.
/// Ciphertext `i` encrypts one row:
///
/// ```text
/// m_i(X) = sum_j M[i,j] X^j.
/// ```
///
/// The output consists of `N` ciphertexts encrypting the columns:
///
/// ```text
/// m'_j(X) = sum_i M[i,j] X^i.
/// ```
///
/// This implementation follows Algorithm 4 directly:
///
/// 1. `Tweak(X^i * ct_i)`;
/// 2. reindex by `(2j+1)^{-1} mod 2N` and multiply by `N^{-1} mod q`;
/// 3. apply `sigma_(2j+1)` and switch back to the original secret;
/// 4. apply `Tweak` again;
/// 5. apply the final Park monomial/sign correction.
///
/// The identity automorphism (`j = 0`, exponent `1`) does not require
/// a switching key.
pub fn transpose(
    params: crate::rlwe::RlweParameters,
    ciphertexts: &[RlweCiphertext],
    galois_keys: &[crate::ckks::GaloisKey],
) -> Vec<RlweCiphertext> {
    assert!(!ciphertexts.is_empty(), "C-MT requires ciphertexts");

    let degree = params.degree();
    let modulus = params.modulus();

    assert_eq!(
        ciphertexts.len(),
        degree,
        "Park Algorithm 4 requires N ciphertexts for ring degree N"
    );

    assert!(
        degree.is_power_of_two(),
        "Park C-MT requires power-of-two ring degree"
    );

    assert_eq!(
        modulus.value() % 2,
        1,
        "Park C-MT requires N invertible modulo q; CKKS q must be odd"
    );

    for ciphertext in ciphertexts {
        assert_eq!(
            ciphertext.b().degree(),
            degree,
            "C-MT ciphertext degree must match RLWE degree"
        );
        assert_eq!(
            ciphertext.b().modulus(),
            modulus,
            "C-MT ciphertext modulus must match RLWE modulus"
        );
        assert_eq!(ciphertext.a().degree(), degree);
        assert_eq!(ciphertext.a().modulus(), modulus);
    }

    let two_n = 2 * degree;

    // Algorithm 4, Step 1:
    //
    // aux <- Tweak((X^i * ct_i)_i)
    let shifted_inputs: Vec<RlweCiphertext> = ciphertexts
        .iter()
        .enumerate()
        .map(|(i, ciphertext)| ciphertext_mul_monomial(ciphertext, i))
        .collect();

    let aux = tweak(&shifted_inputs);

    // Algorithm 4, Steps 2--5.
    let n_inverse = modular_inverse(degree as u64, modulus.value());

    let mut transformed = Vec::with_capacity(degree);

    for j in 0..degree {
        let exponent = 2 * j + 1;

        // index =
        //   (-1 + (2j+1)^(-1) mod 2N) / 2
        let inverse_exponent = modular_inverse(exponent as u64, two_n as u64) as usize;

        assert_eq!(
            inverse_exponent % 2,
            1,
            "inverse of odd C-MT exponent must remain odd"
        );

        let source_index = (inverse_exponent - 1) / 2;

        let normalized = ciphertext_scalar_mul(&aux[source_index], n_inverse);

        let automorphed = if exponent == 1 {
            normalized
        } else {
            let key = galois_key_for_exponent(galois_keys, exponent);

            crate::ckks::apply_galois_automorphism(params, &normalized, key)
        };

        transformed.push(automorphed);
    }

    // Algorithm 4, Step 6.
    let second_tweak = tweak(&transformed);

    // Algorithm 4, Steps 7--9:
    //
    // ct'_(j mod N) =
    //   -X^(N-j) * ct''_((N-j) mod N)
    //
    // for j = 1..N.
    let zero = RlweCiphertext::new(
        Polynomial::zero(modulus, degree),
        Polynomial::zero(modulus, degree),
    );

    let mut output = vec![zero; degree];

    // Handle the constant-coefficient column explicitly.
    output[0] = second_tweak[0].clone();

    // Apply the negacyclic correction to the remaining columns.
    for (output_index, output_slot) in output.iter_mut().enumerate().take(degree).skip(1) {
        let source_index = degree - output_index;
        let exponent = degree - output_index;

        let corrected = ciphertext_mul_monomial(&second_tweak[source_index], exponent);

        *output_slot = ciphertext_scalar_mul(&corrected, modulus.value() - 1);
    }

    output
}

/// Ordinary dense matrix over Z_q used by the Park reduction.
type ParkMatrix = Vec<Vec<u64>>;

fn park_matrix_transpose(matrix: &[Vec<u64>]) -> ParkMatrix {
    assert!(!matrix.is_empty());

    let rows = matrix.len();
    let cols = matrix[0].len();

    assert!(matrix.iter().all(|row| row.len() == cols));

    let mut result = vec![vec![0_u64; rows]; cols];

    for (row, source_row) in matrix.iter().enumerate() {
        for (col, &value) in source_row.iter().enumerate() {
            result[col][row] = value;
        }
    }

    result
}

fn park_matrix_add(lhs: &[Vec<u64>], rhs: &[Vec<u64>], modulus: u64) -> ParkMatrix {
    assert_eq!(lhs.len(), rhs.len());
    assert!(!lhs.is_empty());
    assert_eq!(lhs[0].len(), rhs[0].len());

    lhs.iter()
        .zip(rhs)
        .map(|(lhs_row, rhs_row)| {
            lhs_row
                .iter()
                .zip(rhs_row)
                .map(|(&lhs_value, &rhs_value)| {
                    ((lhs_value as u128 + rhs_value as u128) % modulus as u128) as u64
                })
                .collect()
        })
        .collect()
}

/// Park Mod-PP-MM: ordinary modular matrix multiplication.
fn park_matrix_mul(lhs: &[Vec<u64>], rhs: &[Vec<u64>], modulus: u64) -> ParkMatrix {
    assert!(!lhs.is_empty());
    assert!(!rhs.is_empty());

    let rows = lhs.len();
    let inner = lhs[0].len();
    let cols = rhs[0].len();

    assert!(lhs.iter().all(|row| row.len() == inner));
    assert_eq!(rhs.len(), inner);
    assert!(rhs.iter().all(|row| row.len() == cols));

    let mut result = vec![vec![0_u64; cols]; rows];
    let modulus_u128 = modulus as u128;

    for (row, result_row) in result.iter_mut().enumerate() {
        for (col, result_value) in result_row.iter_mut().enumerate() {
            let mut accumulator = 0_u128;

            for (k, rhs_row) in rhs.iter().enumerate().take(inner) {
                accumulator += (lhs[row][k] as u128) * (rhs_row[col] as u128);
                accumulator %= modulus_u128;
            }

            *result_value = accumulator as u64;
        }
    }

    result
}

/// Interprets ciphertext polynomial coefficients as matrix columns.
fn park_components_by_columns(ciphertexts: &[RlweCiphertext], select_a: bool) -> ParkMatrix {
    assert!(!ciphertexts.is_empty());

    let degree = ciphertexts.len();

    assert!(ciphertexts.iter().all(|ciphertext| {
        ciphertext.a().degree() == degree && ciphertext.b().degree() == degree
    }));

    let mut matrix = vec![vec![0_u64; degree]; degree];

    for (column, ciphertext) in ciphertexts.iter().enumerate() {
        let polynomial = if select_a {
            ciphertext.a()
        } else {
            ciphertext.b()
        };

        for (row, &coefficient) in polynomial.coefficients().iter().enumerate() {
            matrix[row][column] = coefficient;
        }
    }

    matrix
}

/// Constructs a synthetic Park `(A,0)` bundle with `A` packed by rows.
///
/// FHE-rs stores ciphertexts as `(b,a)`, so each generated ciphertext is
/// `(b=0, a=row)`.
fn park_synthetic_a_bundle_from_rows(
    matrix: &[Vec<u64>],
    modulus: crate::ring::Modulus,
) -> Vec<RlweCiphertext> {
    assert!(!matrix.is_empty());

    let degree = matrix.len();

    assert!(
        matrix.iter().all(|row| row.len() == degree),
        "Park synthetic row bundle must be square"
    );

    matrix
        .iter()
        .map(|row| {
            let a = Polynomial::new(modulus, row.clone());
            let b = Polynomial::zero(modulus, degree);

            RlweCiphertext::new(b, a)
        })
        .collect()
}

fn park_rns_components_by_columns(
    ciphertexts: &[crate::grafting::RnsRlweCiphertext],
    limb_index: usize,
    select_a: bool,
) -> ParkMatrix {
    assert!(!ciphertexts.is_empty());

    let degree = ciphertexts.len();

    assert!(
        ciphertexts
            .iter()
            .all(|ciphertext| ciphertext.degree() == degree),
        "Park RNS component bundle must contain N degree-N ciphertexts"
    );

    park_components_by_columns(
        &ciphertexts
            .iter()
            .map(|ciphertext| ciphertext.limb(limb_index).clone())
            .collect::<Vec<_>>(),
        select_a,
    )
}

fn park_rns_synthetic_a_bundle_from_rows(
    matrices_by_limb: &[ParkMatrix],
    moduli: &[crate::ring::Modulus],
) -> Vec<crate::grafting::RnsRlweCiphertext> {
    assert!(!matrices_by_limb.is_empty());
    assert_eq!(
        matrices_by_limb.len(),
        moduli.len(),
        "Park RNS synthetic bundle requires one matrix per modulus"
    );

    let degree = matrices_by_limb[0].len();

    assert!(
        matrices_by_limb.iter().all(|matrix| {
            matrix.len() == degree && matrix.iter().all(|row| row.len() == degree)
        }),
        "Park RNS synthetic matrices must be square and dimension-compatible"
    );

    (0..degree)
        .map(|row| {
            let limbs = matrices_by_limb
                .iter()
                .zip(moduli.iter().copied())
                .map(|(matrix, modulus)| {
                    let a = Polynomial::new(modulus, matrix[row].clone());
                    let b = Polynomial::zero(modulus, degree);
                    RlweCiphertext::new(b, a)
                })
                .collect();

            crate::grafting::RnsRlweCiphertext::from_limbs(limbs)
        })
        .collect()
}

/// RNS form of Park Algorithm 8 up to, but excluding, relinearization
/// and CKKS rescaling.
///
/// Both operands are column-wise N x N RNS RLWE ciphertext bundles.
/// Park's three C-MT operations use the RNS-native transform and RNS
/// Galois/key-switch machinery. The four Mod-PP-MM operations are evaluated
/// independently modulo every active RNS prime using the scalar reference
/// matrix kernel.
///
/// The returned bundle contains one RNS degree-two ciphertext per output
/// matrix column, with FHE-rs convention
///
/// ```text
/// c0 + c1*s + c2*s^2.
/// ```
/// Mod-PP-MM acceleration is deliberately separate from this correctness path.
pub fn rns_ccmm_quadratic(
    lhs: &[crate::grafting::RnsRlweCiphertext],
    rhs: &[crate::grafting::RnsRlweCiphertext],
    galois_keys: &[crate::ckks::RnsGaloisKey],
) -> Vec<crate::grafting::RnsQuadraticCiphertext> {
    assert!(!lhs.is_empty(), "Park RNS CC-MM lhs must not be empty");

    let degree = lhs[0].degree();
    let basis = lhs[0].basis().clone();
    let moduli = basis.moduli();

    assert_eq!(
        lhs.len(),
        degree,
        "Park RNS CC-MM lhs must contain N column ciphertexts"
    );
    assert_eq!(
        rhs.len(),
        degree,
        "Park RNS CC-MM rhs must contain N column ciphertexts"
    );

    assert!(
        lhs.iter()
            .chain(rhs.iter())
            .all(|ciphertext| ciphertext.degree() == degree),
        "Park RNS CC-MM ciphertext degrees must match"
    );

    assert!(
        lhs.iter()
            .chain(rhs.iter())
            .all(|ciphertext| ciphertext.basis() == &basis),
        "Park RNS CC-MM ciphertext bases must match"
    );

    // Algorithm 8, Step 1.
    let rhs_transposed = rns_transpose(rhs, galois_keys);

    // Algorithm 8, Step 2: four Mod-PP-MM operations independently
    // in every CRT residue.
    let mut c00_by_limb = Vec::with_capacity(moduli.len());
    let mut c01_by_limb = Vec::with_capacity(moduli.len());
    let mut c10_by_limb = Vec::with_capacity(moduli.len());
    let mut c11_by_limb = Vec::with_capacity(moduli.len());

    for (limb_index, modulus) in moduli.iter().copied().enumerate() {
        let q = modulus.value();

        let a = park_rns_components_by_columns(lhs, limb_index, true);
        let b = park_rns_components_by_columns(lhs, limb_index, false);

        let a_tilde = park_matrix_transpose(&park_rns_components_by_columns(
            &rhs_transposed,
            limb_index,
            true,
        ));
        let b_tilde = park_matrix_transpose(&park_rns_components_by_columns(
            &rhs_transposed,
            limb_index,
            false,
        ));

        c00_by_limb.push(park_matrix_mul(&a, &a_tilde, q));
        c01_by_limb.push(park_matrix_mul(&a, &b_tilde, q));
        c10_by_limb.push(park_matrix_mul(&b, &a_tilde, q));
        c11_by_limb.push(park_matrix_mul(&b, &b_tilde, q));
    }

    // Algorithm 8, Steps 3 and 4. C00 and C10 enter C-MT packed by rows.
    let c00_bundle = park_rns_synthetic_a_bundle_from_rows(&c00_by_limb, moduli);
    let c10_bundle = park_rns_synthetic_a_bundle_from_rows(&c10_by_limb, moduli);

    let d01 = rns_transpose(&c00_bundle, galois_keys);
    let d23 = rns_transpose(&c10_bundle, galois_keys);

    // Reconstruct each output column independently in every RNS limb:
    //
    //     c2 = D0
    //     c1 = D1 + D2 + C01
    //     c0 = D3 + C11.
    (0..degree)
        .map(|column| {
            let mut c0_residues = Vec::with_capacity(moduli.len());
            let mut c1_residues = Vec::with_capacity(moduli.len());
            let mut c2_residues = Vec::with_capacity(moduli.len());

            for (limb_index, modulus) in moduli.iter().copied().enumerate() {
                let q = modulus.value();

                let d0 = park_rns_components_by_columns(&d01, limb_index, true);
                let d1 = park_rns_components_by_columns(&d01, limb_index, false);
                let d2 = park_rns_components_by_columns(&d23, limb_index, true);
                let d3 = park_rns_components_by_columns(&d23, limb_index, false);

                let c1_matrix =
                    park_matrix_add(&park_matrix_add(&d1, &d2, q), &c01_by_limb[limb_index], q);
                let c0_matrix = park_matrix_add(&d3, &c11_by_limb[limb_index], q);

                c0_residues.push(Polynomial::new(
                    modulus,
                    (0..degree).map(|row| c0_matrix[row][column]).collect(),
                ));

                c1_residues.push(Polynomial::new(
                    modulus,
                    (0..degree).map(|row| c1_matrix[row][column]).collect(),
                ));

                c2_residues.push(Polynomial::new(
                    modulus,
                    (0..degree).map(|row| d0[row][column]).collect(),
                ));
            }

            crate::grafting::RnsQuadraticCiphertext::from_rns_polynomials(
                crate::ring::RnsPolynomial::from_residues(c0_residues),
                crate::ring::RnsPolynomial::from_residues(c1_residues),
                crate::ring::RnsPolynomial::from_residues(c2_residues),
            )
        })
        .collect()
}

/// RNS Park Algorithm 8 followed by RNS relinearization.
///
/// This converts each degree-two Park output
///
/// ```text
/// c0 + c1*s + c2*s^2
/// ```
///
/// into an RNS RLWE ciphertext under `s`. CKKS rescaling is deliberately
/// excluded from this layer.
pub fn rns_ccmm_relinearized(
    lhs: &[crate::grafting::RnsRlweCiphertext],
    rhs: &[crate::grafting::RnsRlweCiphertext],
    galois_keys: &[crate::ckks::RnsGaloisKey],
    multiplication_key: &crate::grafting::RnsMultiplicationKey,
) -> Vec<crate::grafting::RnsRlweCiphertext> {
    rns_ccmm_quadratic(lhs, rhs, galois_keys)
        .iter()
        .map(|product| crate::grafting::rns_relinearize(product, multiplication_key))
        .collect()
}

/// Executes Park CC-MM over same-level RNS CKKS ciphertext bundles,
/// relinearizes each output column, and rescales once.
///
/// Both operands use Park's column-wise matrix packing. All columns in both
/// operands must share one CKKS level and active RNS basis.
///
/// The CKKS state transition matches ordinary ciphertext multiplication:
///
/// ```text
/// level L, scales Delta_l and Delta_r
///     -> Park CC-MM + relinearization at level L
///     -> scale Delta_l * Delta_r
///     -> rescale by chain.dropped_modulus(L)
///     -> level L + 1
/// ```
pub fn rns_ckks_ccmm_relinearize_rescale(
    lhs: &[crate::ckks::RnsCkksCiphertext],
    rhs: &[crate::ckks::RnsCkksCiphertext],
    galois_keys: &[crate::ckks::RnsGaloisKey],
    multiplication_key: &crate::grafting::RnsMultiplicationKey,
    chain: &crate::ring::ModulusChain,
) -> Vec<crate::ckks::RnsCkksCiphertext> {
    assert!(!lhs.is_empty(), "Park CKKS CC-MM lhs must not be empty");

    assert_eq!(
        lhs.len(),
        rhs.len(),
        "Park CKKS CC-MM operands must contain the same number of columns"
    );

    let lhs_state = lhs[0].state();
    let rhs_state = rhs[0].state();

    lhs_state.assert_matches_chain(chain);
    rhs_state.assert_matches_chain(chain);

    assert_eq!(
        lhs_state.level(),
        rhs_state.level(),
        "Park CKKS CC-MM requires matching operand levels"
    );

    assert_eq!(
        lhs_state.basis(),
        rhs_state.basis(),
        "Park CKKS CC-MM requires matching operand bases"
    );

    assert!(
        chain.has_next_level(lhs_state.level()),
        "Park CKKS CC-MM requires a next chain level for rescaling"
    );

    for ciphertext in lhs {
        ciphertext.assert_matches_chain(chain);

        assert_eq!(
            ciphertext.level(),
            lhs_state.level(),
            "all Park CKKS lhs columns must share one level"
        );

        assert_eq!(
            ciphertext.basis(),
            lhs_state.basis(),
            "all Park CKKS lhs columns must share one basis"
        );

        assert_eq!(
            ciphertext.scale(),
            lhs_state.scale(),
            "all Park CKKS lhs columns must share one scale"
        );
    }

    for ciphertext in rhs {
        ciphertext.assert_matches_chain(chain);

        assert_eq!(
            ciphertext.level(),
            rhs_state.level(),
            "all Park CKKS rhs columns must share one level"
        );

        assert_eq!(
            ciphertext.basis(),
            rhs_state.basis(),
            "all Park CKKS rhs columns must share one basis"
        );

        assert_eq!(
            ciphertext.scale(),
            rhs_state.scale(),
            "all Park CKKS rhs columns must share one scale"
        );
    }

    assert_eq!(
        multiplication_key.layout().full_basis(),
        lhs_state.basis(),
        "Park RNS multiplication-key basis must match active CKKS level"
    );

    let lhs_rlwe: Vec<_> = lhs
        .iter()
        .map(|ciphertext| ciphertext.rlwe().clone())
        .collect();

    let rhs_rlwe: Vec<_> = rhs
        .iter()
        .map(|ciphertext| ciphertext.rlwe().clone())
        .collect();

    let relinearized = rns_ccmm_relinearized(&lhs_rlwe, &rhs_rlwe, galois_keys, multiplication_key);

    let product_state = lhs_state.after_multiply(rhs_state, chain);

    relinearized
        .into_iter()
        .map(|ciphertext| {
            let product =
                crate::ckks::RnsCkksCiphertext::new(ciphertext, product_state.clone(), chain);

            crate::ckks::rescale_rns_ckks_to_next(&product, chain)
        })
        .collect()
}

/// Park Algorithm 8 up to, but excluding, relinearization and rescaling.
///
/// Both input operands are column-wise N x N RLWE ciphertext bundles,
/// matching Park Algorithm 8:
///
/// ```text
/// ciphertext j encrypts matrix column j.
/// ```
///
/// The returned vector is likewise column-wise and contains one degree-two
/// RLWE ciphertext per output matrix column.
///
/// This reference implementation deliberately uses ordinary O(N^3)
/// modular matrix multiplication.  Acceleration is a separate concern.
pub fn ccmm_quadratic(
    params: crate::rlwe::RlweParameters,
    lhs: &[RlweCiphertext],
    rhs: &[RlweCiphertext],
    galois_keys: &[crate::ckks::GaloisKey],
) -> Vec<crate::rlwe::RlweQuadraticCiphertext> {
    let degree = params.degree();
    let modulus = params.modulus();
    let q = modulus.value();

    assert_eq!(
        lhs.len(),
        degree,
        "Park CC-MM lhs must contain N column ciphertexts"
    );

    assert_eq!(
        rhs.len(),
        degree,
        "Park CC-MM rhs must contain N column ciphertexts"
    );

    // ------------------------------------------------------------------
    // Algorithm 8, Step 1.
    //
    // RHS:
    //
    //     ColumnBundle(A',B')
    //           |
    //          C-MT
    //           |
    //       RowBundle(A~,B~)
    // ------------------------------------------------------------------

    let rhs_transposed = transpose(params, rhs, galois_keys);

    // Original lhs satisfies:
    //
    //     M = S* A + B,
    //
    // where ciphertext j contributes column j of A and B.
    let a = park_components_by_columns(lhs, true);
    let b = park_components_by_columns(lhs, false);

    // C-MT(rhs) gives raw component matrices A_out, B_out with
    //
    //     M'^T = S* A_out + B_out.
    //
    // Therefore Park's row-side representation is
    //
    //     M' = A_out^T S*^T + B_out^T.
    let a_tilde = park_matrix_transpose(&park_components_by_columns(&rhs_transposed, true));
    let b_tilde = park_matrix_transpose(&park_components_by_columns(&rhs_transposed, false));

    // ------------------------------------------------------------------
    // Algorithm 8, Step 2: four Mod-PP-MM operations.
    // ------------------------------------------------------------------

    let c00 = park_matrix_mul(&a, &a_tilde, q);
    let c01 = park_matrix_mul(&a, &b_tilde, q);
    let c10 = park_matrix_mul(&b, &a_tilde, q);
    let c11 = park_matrix_mul(&b, &b_tilde, q);

    // ------------------------------------------------------------------
    // Algorithm 8, Steps 3 and 4.
    //
    // C00 and C10 must enter C-MT packed BY ROWS.
    // ------------------------------------------------------------------

    let c00_bundle = park_synthetic_a_bundle_from_rows(&c00, modulus);
    let c10_bundle = park_synthetic_a_bundle_from_rows(&c10, modulus);

    let d01 = transpose(params, &c00_bundle, galois_keys);
    let d23 = transpose(params, &c10_bundle, galois_keys);

    let d0 = park_components_by_columns(&d01, true);
    let d1 = park_components_by_columns(&d01, false);
    let d2 = park_components_by_columns(&d23, true);
    let d3 = park_components_by_columns(&d23, false);

    // ------------------------------------------------------------------
    // Algorithm 8 pre-relinearization reconstruction:
    //
    //     c2 = D0
    //     c1 = D1 + D2 + C01
    //     c0 = D3 + C11
    //
    // FHE-rs quadratic convention:
    //
    //     c0 + c1*s + c2*s^2.
    // ------------------------------------------------------------------

    let c1_matrix = park_matrix_add(&park_matrix_add(&d1, &d2, q), &c01, q);

    let c0_matrix = park_matrix_add(&d3, &c11, q);

    (0..degree)
        .map(|column| {
            let c0 = Polynomial::new(
                modulus,
                (0..degree).map(|row| c0_matrix[row][column]).collect(),
            );

            let c1 = Polynomial::new(
                modulus,
                (0..degree).map(|row| c1_matrix[row][column]).collect(),
            );

            let c2 = Polynomial::new(modulus, (0..degree).map(|row| d0[row][column]).collect());

            crate::rlwe::RlweQuadraticCiphertext::new(c0, c1, c2)
        })
        .collect()
}

/// Multiplies two column-wise Park matrix ciphertext bundles and
/// relinearizes each output column back to a rank-1 RLWE ciphertext.
///
/// This completes the relinearization portion of Park Algorithm 8,
/// Step 5. Rescaling is deliberately left to the CKKS/RNS integration
/// layer.
pub fn ccmm_relinearized(
    params: crate::rlwe::RlweParameters,
    lhs: &[RlweCiphertext],
    rhs: &[RlweCiphertext],
    galois_keys: &[crate::ckks::GaloisKey],
    multiplication_key: &crate::eval::MultiplicationKey,
) -> Vec<RlweCiphertext> {
    ccmm_quadratic(params, lhs, rhs, galois_keys)
        .iter()
        .map(|product| crate::eval::relinearize(params, product, multiplication_key))
        .collect()
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::grafting::RnsRlweCiphertext;
    use crate::ring::{Modulus, Polynomial};
    use crate::rlwe::{decrypt_raw, encrypt_raw_with_rng, RlweParameters, SecretKey};

    use super::*;

    fn direct_tweak(ciphertexts: &[RlweCiphertext]) -> Vec<RlweCiphertext> {
        let n = ciphertexts.len();
        let degree = ciphertexts[0].b().degree();
        let modulus = ciphertexts[0].b().modulus();

        (0..n)
            .map(|j| {
                let mut b = Polynomial::zero(modulus, degree);
                let mut a = Polynomial::zero(modulus, degree);

                for (i, ciphertext) in ciphertexts.iter().enumerate() {
                    let exponent = 2 * i * j * degree / n;

                    b = b.add(&polynomial_mul_monomial(ciphertext.b(), exponent));
                    a = a.add(&polynomial_mul_monomial(ciphertext.a(), exponent));
                }

                RlweCiphertext::new(b, a)
            })
            .collect()
    }

    fn deterministic_ciphertext(modulus: Modulus, degree: usize, seed: u64) -> RlweCiphertext {
        let b = Polynomial::new(
            modulus,
            (0..degree)
                .map(|i| (seed + 3 * i as u64) % modulus.value())
                .collect(),
        );

        let a = Polynomial::new(
            modulus,
            (0..degree)
                .map(|i| (2 * seed + 5 * i as u64) % modulus.value())
                .collect(),
        );

        RlweCiphertext::new(b, a)
    }

    fn park_rns_test_moduli() -> [Modulus; 3] {
        [
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]
    }

    fn park_rns_test_secret_coefficients() -> [i8; 8] {
        [-1, 0, 1, 1, 0, -1, 1, 0]
    }

    fn park_rns_test_secret(modulus: Modulus) -> SecretKey {
        let coefficients = park_rns_test_secret_coefficients();

        SecretKey::from_polynomial(Polynomial::new(
            modulus,
            coefficients
                .iter()
                .map(|&value| match value {
                    -1 => modulus.value() - 1,
                    0 => 0,
                    1 => 1,
                    _ => unreachable!(),
                })
                .collect(),
        ))
    }

    fn encrypt_rns_coefficient_rows(matrix: &[Vec<u64>], seed: u64) -> Vec<RnsRlweCiphertext> {
        let degree = matrix.len();
        let moduli = park_rns_test_moduli();

        assert_eq!(degree, 8);
        assert!(matrix.iter().all(|row| row.len() == degree));

        matrix
            .iter()
            .enumerate()
            .map(|(row_index, row)| {
                let limbs = moduli
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(limb_index, modulus)| {
                        let params = RlweParameters::new(degree, modulus, 2, 0);
                        let secret = park_rns_test_secret(modulus);

                        let plaintext = Polynomial::new(
                            modulus,
                            row.iter().map(|&value| value % modulus.value()).collect(),
                        );

                        let mut rng = ChaCha20Rng::seed_from_u64(
                            seed ^ ((row_index as u64) << 8) ^ limb_index as u64,
                        );

                        encrypt_raw_with_rng(params, &secret, &plaintext, &mut rng)
                    })
                    .collect();

                RnsRlweCiphertext::from_limbs(limbs)
            })
            .collect()
    }

    fn park_rns_galois_keys(seed: u64) -> Vec<crate::ckks::RnsGaloisKey> {
        let degree = 8;
        let moduli = park_rns_test_moduli();
        let basis = crate::ring::ModulusBasis::new(moduli.to_vec());
        let layout = crate::grafting::RnsGadgetLayout::new(basis, vec![1, 2]);
        let secret_coefficients = park_rns_test_secret_coefficients();

        (1..degree)
            .map(|j| 2 * j + 1)
            .map(|exponent| {
                let mut rng = ChaCha20Rng::seed_from_u64(seed ^ exponent as u64);

                crate::ckks::RnsGaloisKey::generate_with_rng(
                    degree,
                    2,
                    0,
                    &secret_coefficients,
                    exponent,
                    layout.clone(),
                    &mut rng,
                )
            })
            .collect()
    }

    fn decrypt_rns_coefficient_rows(ciphertexts: &[RnsRlweCiphertext]) -> Vec<Vec<Vec<u64>>> {
        let degree = ciphertexts.len();
        let moduli = park_rns_test_moduli();

        moduli
            .iter()
            .copied()
            .enumerate()
            .map(|(limb_index, modulus)| {
                let params = RlweParameters::new(degree, modulus, 2, 0);
                let secret = park_rns_test_secret(modulus);

                ciphertexts
                    .iter()
                    .map(|ciphertext| {
                        decrypt_raw(params, &secret, ciphertext.limb(limb_index))
                            .coefficients()
                            .to_vec()
                    })
                    .collect()
            })
            .collect()
    }

    fn encrypt_rns_coefficient_columns(matrix: &[Vec<u64>], seed: u64) -> Vec<RnsRlweCiphertext> {
        let degree = matrix.len();
        let moduli = park_rns_test_moduli();

        assert_eq!(degree, 8);
        assert!(matrix.iter().all(|row| row.len() == degree));

        (0..degree)
            .map(|column| {
                let limbs = moduli
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(limb_index, modulus)| {
                        let params = RlweParameters::new(degree, modulus, 2, 0);
                        let secret = park_rns_test_secret(modulus);

                        let plaintext = Polynomial::new(
                            modulus,
                            (0..degree)
                                .map(|row| matrix[row][column] % modulus.value())
                                .collect(),
                        );

                        let mut rng = ChaCha20Rng::seed_from_u64(
                            seed ^ ((column as u64) << 8) ^ limb_index as u64,
                        );

                        encrypt_raw_with_rng(params, &secret, &plaintext, &mut rng)
                    })
                    .collect();

                RnsRlweCiphertext::from_limbs(limbs)
            })
            .collect()
    }

    fn decrypt_rns_quadratic_columns(
        ciphertexts: &[crate::grafting::RnsQuadraticCiphertext],
    ) -> Vec<Vec<Vec<u64>>> {
        let degree = ciphertexts.len();
        let moduli = park_rns_test_moduli();

        moduli
            .iter()
            .copied()
            .enumerate()
            .map(|(limb_index, modulus)| {
                let secret = park_rns_test_secret(modulus);
                let s = secret.polynomial();
                let s_squared = s.negacyclic_mul(s);

                let mut matrix = vec![vec![0_u64; degree]; degree];

                for (column, ciphertext) in ciphertexts.iter().enumerate() {
                    let plaintext = ciphertext
                        .c0()
                        .residue(limb_index)
                        .add(&ciphertext.c1().residue(limb_index).negacyclic_mul(s))
                        .add(
                            &ciphertext
                                .c2()
                                .residue(limb_index)
                                .negacyclic_mul(&s_squared),
                        );

                    for (row, matrix_row) in matrix.iter_mut().enumerate() {
                        matrix_row[column] = plaintext.coefficients()[row];
                    }
                }

                matrix
            })
            .collect()
    }

    #[test]
    fn rns_park_ckks_ccmm_decodes_approximate_matrix_product() {
        let degree = 8;
        let moduli = park_rns_test_moduli();

        let chain = crate::ring::ModulusChain::from_top_basis(crate::ring::ModulusBasis::new(
            moduli.to_vec(),
        ));

        // Match the top-level rescale divisor so multiplication followed
        // by one rescale returns to approximately the original scale.
        let scale = moduli[moduli.len() - 1].value() as f64;

        let lhs_clear: Vec<Vec<f64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        0.015 + 0.002 * row as f64 - 0.001 * col as f64
                            + 0.0002 * (row * col) as f64
                    })
                    .collect()
            })
            .collect();

        let rhs_clear: Vec<Vec<f64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        -0.010 + 0.0015 * row as f64 + 0.001 * col as f64
                            - 0.0001 * (row * col) as f64
                    })
                    .collect()
            })
            .collect();

        let encode_matrix = |matrix: &[Vec<f64>], seed: u64| {
            let scaled: Vec<Vec<i128>> = matrix
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&value| (value * scale).round() as i128)
                        .collect()
                })
                .collect();

            let residue_matrix = |modulus: crate::ring::Modulus| {
                let q = i128::from(modulus.value());

                scaled
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|&value| (((value % q) + q) % q) as u64)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            };

            let matrices_by_limb: Vec<Vec<Vec<u64>>> =
                moduli.iter().copied().map(residue_matrix).collect();
            let mut rng = ChaCha20Rng::seed_from_u64(seed);

            (0..degree)
                .map(|column| {
                    let limbs = moduli
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(limb_index, modulus)| {
                            let params = crate::rlwe::RlweParameters::new(degree, modulus, 2, 0);

                            let message = crate::ring::Polynomial::new(
                                modulus,
                                (0..degree)
                                    .map(|row| matrices_by_limb[limb_index][row][column])
                                    .collect(),
                            );

                            crate::rlwe::encrypt_raw_with_rng(
                                params,
                                &park_rns_test_secret(modulus),
                                &message,
                                &mut rng,
                            )
                        })
                        .collect();

                    let rlwe = crate::grafting::RnsRlweCiphertext::from_limbs(limbs);

                    crate::ckks::RnsCkksCiphertext::new(
                        rlwe,
                        crate::ckks::CkksChainState::top(&chain, scale),
                        &chain,
                    )
                })
                .collect::<Vec<_>>()
        };

        let lhs = encode_matrix(&lhs_clear, 0xC4D4_0000);
        let rhs = encode_matrix(&rhs_clear, 0xC4D5_0000);

        let galois_keys = park_rns_galois_keys(0xC4D6_0000);

        let layout = crate::grafting::RnsGadgetLayout::new(chain.top().clone(), vec![1, 2]);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xC4D7_0000);

        let multiplication_key = crate::grafting::RnsMultiplicationKey::generate_with_rng(
            degree,
            2,
            0,
            &park_rns_test_secret_coefficients(),
            layout,
            &mut key_rng,
        );

        let output = rns_ckks_ccmm_relinearize_rescale(
            &lhs,
            &rhs,
            &galois_keys,
            &multiplication_key,
            &chain,
        );

        let output_scale = output[0].scale();

        let secret = park_rns_test_secret_coefficients();

        let mut actual = vec![vec![0.0_f64; degree]; degree];

        for (column, ciphertext) in output.iter().enumerate() {
            let decrypted = crate::grafting::decrypt_rns_raw(ciphertext.rlwe(), &secret);

            let modulus = decrypted.composite_modulus();

            for (row, value) in decrypted.reconstruct_coefficients().into_iter().enumerate() {
                let centered = if value > modulus / 2 {
                    value as i128 - modulus as i128
                } else {
                    value as i128
                };

                actual[row][column] = centered as f64 / output_scale;
            }
        }

        let mut expected = vec![vec![0.0_f64; degree]; degree];

        for row in 0..degree {
            for column in 0..degree {
                expected[row][column] = (0..degree)
                    .map(|k| lhs_clear[row][k] * rhs_clear[k][column])
                    .sum();
            }
        }

        let mut max_error = 0.0_f64;
        let mut max_error_at = (0_usize, 0_usize);

        for row in 0..degree {
            for column in 0..degree {
                let error = (actual[row][column] - expected[row][column]).abs();

                if error > max_error {
                    max_error = error;
                    max_error_at = (row, column);
                }
            }
        }

        println!("PARK_CKKS_CCMM_OUTPUT_SCALE={output_scale:.12e}");
        println!("PARK_CKKS_CCMM_MAX_ERROR={max_error:.12e}");
        println!(
            "PARK_CKKS_CCMM_MAX_ERROR_AT={},{}",
            max_error_at.0, max_error_at.1
        );

        assert!(
            max_error < 5.0e-4,
            "Park CKKS CC-MM numerical error {max_error:.12e} \
             at ({}, {}) exceeds 5e-4",
            max_error_at.0,
            max_error_at.1
        );
    }

    #[test]
    fn rns_park_ckks_ccmm_advances_level_and_scale() {
        let degree = 8;
        let moduli = park_rns_test_moduli();

        let chain = crate::ring::ModulusChain::from_top_basis(crate::ring::ModulusBasis::new(
            moduli.to_vec(),
        ));

        // Choosing the trailing top-level modulus as both operand scales
        // makes the expected post-rescale scale equal to that modulus.
        let input_scale = moduli[moduli.len() - 1].value() as f64;

        let lhs_matrix: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| 3 + 5 * row as u64 + 7 * col as u64 + row as u64 * col as u64)
                    .collect()
            })
            .collect();

        let rhs_matrix: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| 11 + 13 * row as u64 + 17 * col as u64 + 2 * row as u64 * col as u64)
                    .collect()
            })
            .collect();

        let lhs_rlwe = encrypt_rns_coefficient_columns(&lhs_matrix, 0xC4D0_0000);
        let rhs_rlwe = encrypt_rns_coefficient_columns(&rhs_matrix, 0xC4D1_0000);

        let lhs: Vec<_> = lhs_rlwe
            .into_iter()
            .map(|ciphertext| {
                crate::ckks::RnsCkksCiphertext::new(
                    ciphertext,
                    crate::ckks::CkksChainState::top(&chain, input_scale),
                    &chain,
                )
            })
            .collect();

        let rhs: Vec<_> = rhs_rlwe
            .into_iter()
            .map(|ciphertext| {
                crate::ckks::RnsCkksCiphertext::new(
                    ciphertext,
                    crate::ckks::CkksChainState::top(&chain, input_scale),
                    &chain,
                )
            })
            .collect();

        let galois_keys = park_rns_galois_keys(0xC4D2_0000);

        let layout = crate::grafting::RnsGadgetLayout::new(chain.top().clone(), vec![1, 2]);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xC4D3_0000);

        let multiplication_key = crate::grafting::RnsMultiplicationKey::generate_with_rng(
            degree,
            2,
            0,
            &park_rns_test_secret_coefficients(),
            layout,
            &mut key_rng,
        );

        let output = rns_ckks_ccmm_relinearize_rescale(
            &lhs,
            &rhs,
            &galois_keys,
            &multiplication_key,
            &chain,
        );

        assert_eq!(
            output.len(),
            degree,
            "Park CKKS CC-MM must return N column ciphertexts"
        );

        let dropped = chain
            .dropped_modulus(0)
            .expect("top level must have a rescale divisor");

        let expected_scale = input_scale * input_scale / dropped.value() as f64;

        for ciphertext in &output {
            assert_eq!(
                ciphertext.level(),
                1,
                "Park CKKS CC-MM must rescale exactly once"
            );

            assert_eq!(
                ciphertext.basis(),
                chain.level(1),
                "Park CKKS CC-MM output basis must match level 1"
            );

            assert_eq!(
                ciphertext.basis().len(),
                moduli.len() - 1,
                "Park CKKS CC-MM must drop exactly one RNS limb"
            );

            assert_eq!(
                ciphertext.scale(),
                expected_scale,
                "Park CKKS CC-MM output scale must follow multiply-rescale"
            );

            ciphertext.assert_matches_chain(&chain);
        }
    }

    #[test]
    fn rns_park_ccmm_relinearized_matches_asymmetric_clear_product() {
        let degree = 8;
        let moduli = park_rns_test_moduli();

        let lhs: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| 5 + 7 * row as u64 + 11 * col as u64 + 3 * row as u64 * col as u64)
                    .collect()
            })
            .collect();

        let rhs: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| 13 + 17 * row as u64 + 19 * col as u64 + 5 * row as u64 * col as u64)
                    .collect()
            })
            .collect();

        let lhs_ciphertexts = encrypt_rns_coefficient_columns(&lhs, 0xC4C4_0000);
        let rhs_ciphertexts = encrypt_rns_coefficient_columns(&rhs, 0xC4C5_0000);

        let galois_keys = park_rns_galois_keys(0xC4C6_0000);

        let basis = crate::ring::ModulusBasis::new(moduli.to_vec());

        let layout = crate::grafting::RnsGadgetLayout::new(basis, vec![1, 2]);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xC4C7_0000);

        let multiplication_key = crate::grafting::RnsMultiplicationKey::generate_with_rng(
            degree,
            2,
            0,
            &park_rns_test_secret_coefficients(),
            layout,
            &mut key_rng,
        );

        let output = rns_ccmm_relinearized(
            &lhs_ciphertexts,
            &rhs_ciphertexts,
            &galois_keys,
            &multiplication_key,
        );

        let output_by_limb = decrypt_rns_coefficient_rows(&output);

        for (limb_index, modulus) in moduli.iter().copied().enumerate() {
            let actual = transpose_cleartext(&output_by_limb[limb_index]);

            let expected = park_matrix_mul(&lhs, &rhs, modulus.value());

            assert_eq!(
                actual,
                expected,
                "RNS Park relinearized CC-MM mismatch at limb \
                 {limb_index}, modulus {}",
                modulus.value()
            );
        }
    }

    #[test]
    fn rns_park_ccmm_relinearization_preserves_quadratic_decryption() {
        let degree = 8;
        let basis = crate::ring::ModulusBasis::new(park_rns_test_moduli().to_vec());

        let lhs: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| 7 + 11 * row as u64 + 13 * col as u64 + 2 * row as u64 * col as u64)
                    .collect()
            })
            .collect();

        let rhs: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        3 + 17 * row as u64
                            + 19 * col as u64
                            + 5 * row as u64 * col as u64
                            + col as u64 * col as u64
                    })
                    .collect()
            })
            .collect();

        let lhs_ciphertexts = encrypt_rns_coefficient_columns(&lhs, 0xC4C0_0000);
        let rhs_ciphertexts = encrypt_rns_coefficient_columns(&rhs, 0xC4C1_0000);

        let galois_keys = park_rns_galois_keys(0xC4C2_0000);

        let quadratic = rns_ccmm_quadratic(&lhs_ciphertexts, &rhs_ciphertexts, &galois_keys);

        let layout = crate::grafting::RnsGadgetLayout::new(basis, vec![1, 2]);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xC4C3_0000);

        let multiplication_key = crate::grafting::RnsMultiplicationKey::generate_with_rng(
            degree,
            2,
            0,
            &park_rns_test_secret_coefficients(),
            layout,
            &mut key_rng,
        );

        let relinearized: Vec<RnsRlweCiphertext> = quadratic
            .iter()
            .map(|product| crate::grafting::rns_relinearize(product, &multiplication_key))
            .collect();

        let quadratic_plaintexts = decrypt_rns_quadratic_columns(&quadratic);

        let relinearized_plaintexts = decrypt_rns_coefficient_rows(&relinearized);

        for (limb_index, modulus) in park_rns_test_moduli().iter().copied().enumerate() {
            let relinearized_matrix = transpose_cleartext(&relinearized_plaintexts[limb_index]);

            assert_eq!(
                relinearized_matrix,
                quadratic_plaintexts[limb_index],
                "Park RNS relinearization changed decryption at limb \
                 {limb_index}, modulus {}",
                modulus.value()
            );
        }
    }

    #[test]
    fn rns_park_ccmm_quadratic_identity_products() {
        let degree = 8;
        let moduli = park_rns_test_moduli();

        let matrix: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| 19 + 23 * row as u64 + 29 * col as u64 + 3 * row as u64 * col as u64)
                    .collect()
            })
            .collect();

        let mut identity = vec![vec![0_u64; degree]; degree];
        for (index, row) in identity.iter_mut().enumerate() {
            row[index] = 1;
        }

        let matrix_ciphertexts = encrypt_rns_coefficient_columns(&matrix, 0xC4B3_0000);
        let identity_ciphertexts = encrypt_rns_coefficient_columns(&identity, 0xC4B4_0000);
        let keys = park_rns_galois_keys(0xC4B5_0000);

        let right = rns_ccmm_quadratic(&matrix_ciphertexts, &identity_ciphertexts, &keys);

        let left = rns_ccmm_quadratic(&identity_ciphertexts, &matrix_ciphertexts, &keys);

        let right_by_limb = decrypt_rns_quadratic_columns(&right);
        let left_by_limb = decrypt_rns_quadratic_columns(&left);

        for (limb_index, modulus) in moduli.iter().copied().enumerate() {
            let expected: Vec<Vec<u64>> = matrix
                .iter()
                .map(|row| row.iter().map(|&value| value % modulus.value()).collect())
                .collect();

            assert_eq!(
                right_by_limb[limb_index],
                expected,
                "RNS Park quadratic CC-MM failed M x I at limb \
                 {limb_index}, modulus {}",
                modulus.value()
            );

            assert_eq!(
                left_by_limb[limb_index],
                expected,
                "RNS Park quadratic CC-MM failed I x M at limb \
                 {limb_index}, modulus {}",
                modulus.value()
            );
        }
    }

    #[test]
    fn rns_park_ccmm_quadratic_basis_products() {
        let degree = 8;
        let moduli = park_rns_test_moduli();
        let keys = park_rns_galois_keys(0xC4B6_0000);

        for i in 0..degree {
            for j in 0..degree {
                let mut lhs = vec![vec![0_u64; degree]; degree];
                lhs[i][j] = 1;

                let lhs_ciphertexts =
                    encrypt_rns_coefficient_columns(&lhs, 0xC4B7_0000 ^ (degree * i + j) as u64);

                for k in 0..degree {
                    for l in 0..degree {
                        let mut rhs = vec![vec![0_u64; degree]; degree];
                        rhs[k][l] = 1;

                        let rhs_ciphertexts = encrypt_rns_coefficient_columns(
                            &rhs,
                            0xC4B8_0000 ^ (degree * k + l) as u64,
                        );

                        let product = rns_ccmm_quadratic(&lhs_ciphertexts, &rhs_ciphertexts, &keys);

                        let actual_by_limb = decrypt_rns_quadratic_columns(&product);

                        let mut expected = vec![vec![0_u64; degree]; degree];

                        if j == k {
                            expected[i][l] = 1;
                        }

                        for (limb_index, modulus) in moduli.iter().copied().enumerate() {
                            assert_eq!(
                                actual_by_limb[limb_index],
                                expected,
                                "RNS Park quadratic CC-MM failed \
                                 E_{{{i},{j}}} E_{{{k},{l}}} at limb \
                                 {limb_index}, modulus {}",
                                modulus.value()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rns_park_ccmm_quadratic_matches_asymmetric_clear_product() {
        let degree = 8;
        let moduli = park_rns_test_moduli();

        let lhs: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| 3 + 11 * row as u64 + 7 * col as u64 + 2 * row as u64 * col as u64)
                    .collect()
            })
            .collect();

        let rhs: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        5 + 13 * row as u64
                            + 17 * col as u64
                            + 3 * row as u64 * col as u64
                            + row as u64 * row as u64
                    })
                    .collect()
            })
            .collect();

        let lhs_ciphertexts = encrypt_rns_coefficient_columns(&lhs, 0xC4B0_0000);
        let rhs_ciphertexts = encrypt_rns_coefficient_columns(&rhs, 0xC4B1_0000);

        let keys = park_rns_galois_keys(0xC4B2_0000);

        let product = rns_ccmm_quadratic(&lhs_ciphertexts, &rhs_ciphertexts, &keys);

        assert_eq!(product.len(), degree);

        let actual_by_limb = decrypt_rns_quadratic_columns(&product);

        for (limb_index, modulus) in moduli.iter().copied().enumerate() {
            let expected = park_matrix_mul(&lhs, &rhs, modulus.value());

            assert_eq!(
                actual_by_limb[limb_index],
                expected,
                "RNS Park quadratic CC-MM mismatch at limb {limb_index}, \
                 modulus {}",
                modulus.value()
            );
        }
    }

    #[test]
    fn rns_cmt_maps_basis_matrix_eij_to_eji() {
        let degree = 8;
        let moduli = park_rns_test_moduli();
        let keys = park_rns_galois_keys(0xC4A2_0000);

        for source_row in 0..degree {
            for source_col in 0..degree {
                let mut matrix = vec![vec![0_u64; degree]; degree];
                matrix[source_row][source_col] = 1;

                let ciphertexts = encrypt_rns_coefficient_rows(
                    &matrix,
                    0xC4A3_0000 ^ (degree * source_row + source_col) as u64,
                );

                let transposed = rns_transpose(&ciphertexts, &keys);
                let actual_by_limb = decrypt_rns_coefficient_rows(&transposed);

                let expected = transpose_cleartext(&matrix);

                for (limb_index, modulus) in moduli.iter().copied().enumerate() {
                    assert_eq!(
                        actual_by_limb[limb_index],
                        expected,
                        "RNS Park C-MT failed basis E_{{{source_row},{source_col}}} \
                         at limb {limb_index}, modulus {}",
                        modulus.value()
                    );
                }
            }
        }
    }

    #[test]
    fn rns_cmt_is_involution_on_encrypted_matrix() {
        let degree = 8;
        let moduli = park_rns_test_moduli();
        let keys = park_rns_galois_keys(0xC4A4_0000);

        let matrix: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| 37 + 19 * row as u64 + 23 * col as u64 + 3 * row as u64 * col as u64)
                    .collect()
            })
            .collect();

        let ciphertexts = encrypt_rns_coefficient_rows(&matrix, 0xC4A5_0000);

        let once = rns_transpose(&ciphertexts, &keys);
        let twice = rns_transpose(&once, &keys);

        let actual_by_limb = decrypt_rns_coefficient_rows(&twice);

        for (limb_index, modulus) in moduli.iter().copied().enumerate() {
            let expected: Vec<Vec<u64>> = matrix
                .iter()
                .map(|row| row.iter().map(|&value| value % modulus.value()).collect())
                .collect();

            assert_eq!(
                actual_by_limb[limb_index],
                expected,
                "RNS Park C-MT involution failed at limb {limb_index}, \
                 modulus {}",
                modulus.value()
            );
        }
    }

    #[test]
    fn rns_cmt_transposes_asymmetric_matrix_exactly() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let degree = 8;
        let moduli = [
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ];

        let basis = crate::ring::ModulusBasis::new(moduli.to_vec());
        let secret_coefficients = [-1_i8, 0, 1, 1, 0, -1, 1, 0];

        let matrix: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| 1 + (17 * row + 5 * col + row * col) as u64)
                    .collect()
            })
            .collect();

        let ciphertexts: Vec<RnsRlweCiphertext> = matrix
            .iter()
            .enumerate()
            .map(|(row_index, row)| {
                let limbs = moduli
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(limb_index, modulus)| {
                        let params = RlweParameters::new(degree, modulus, 2, 0);

                        let secret = SecretKey::from_polynomial(Polynomial::new(
                            modulus,
                            secret_coefficients
                                .iter()
                                .map(|&value| match value {
                                    -1 => modulus.value() - 1,
                                    0 => 0,
                                    1 => 1,
                                    _ => unreachable!(),
                                })
                                .collect(),
                        ));

                        let plaintext = Polynomial::new(
                            modulus,
                            row.iter().map(|&value| value % modulus.value()).collect(),
                        );

                        let mut rng = ChaCha20Rng::seed_from_u64(
                            0xC4A0_0000 ^ ((row_index as u64) << 8) ^ limb_index as u64,
                        );

                        encrypt_raw_with_rng(params, &secret, &plaintext, &mut rng)
                    })
                    .collect();

                RnsRlweCiphertext::from_limbs(limbs)
            })
            .collect();

        let layout = crate::grafting::RnsGadgetLayout::new(basis.clone(), vec![1, 2]);

        let galois_keys: Vec<crate::ckks::RnsGaloisKey> = (1..degree)
            .map(|j| 2 * j + 1)
            .map(|exponent| {
                let mut rng = ChaCha20Rng::seed_from_u64(0xC4A1_0000 ^ exponent as u64);

                crate::ckks::RnsGaloisKey::generate_with_rng(
                    degree,
                    2,
                    0,
                    &secret_coefficients,
                    exponent,
                    layout.clone(),
                    &mut rng,
                )
            })
            .collect();

        let transposed = rns_transpose(&ciphertexts, &galois_keys);

        assert_eq!(transposed.len(), degree);

        for (output_col, ciphertext) in transposed.iter().enumerate() {
            for (limb_index, modulus) in moduli.iter().copied().enumerate() {
                let secret = SecretKey::from_polynomial(Polynomial::new(
                    modulus,
                    secret_coefficients
                        .iter()
                        .map(|&value| match value {
                            -1 => modulus.value() - 1,
                            0 => 0,
                            1 => 1,
                            _ => unreachable!(),
                        })
                        .collect(),
                ));

                let params = RlweParameters::new(degree, modulus, 2, 0);
                let observed = decrypt_raw(params, &secret, ciphertext.limb(limb_index));

                let expected: Vec<u64> = (0..degree)
                    .map(|row| matrix[row][output_col] % modulus.value())
                    .collect();

                assert_eq!(
                    observed.coefficients(),
                    expected.as_slice(),
                    "RNS C-MT mismatch at output column {output_col}, \
                     limb {limb_index}, modulus {}",
                    modulus.value()
                );
            }
        }
    }

    #[test]
    fn rns_tweak_matches_independent_scalar_limbs() {
        let moduli = [
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ];
        let degree = 8;

        for n in [1_usize, 2, 4, 8] {
            let input: Vec<RnsRlweCiphertext> = (0..n)
                .map(|ciphertext_index| {
                    let limbs = moduli
                        .iter()
                        .enumerate()
                        .map(|(limb_index, &modulus)| {
                            deterministic_ciphertext(
                                modulus,
                                degree,
                                1000 + 100 * ciphertext_index as u64 + 17 * limb_index as u64,
                            )
                        })
                        .collect();

                    RnsRlweCiphertext::from_limbs(limbs)
                })
                .collect();

            let actual = rns_tweak(&input);

            assert_eq!(
                actual.len(),
                n,
                "RNS Park Tweak returned wrong bundle length for n={n}"
            );

            for limb_index in 0..moduli.len() {
                let scalar_input: Vec<RlweCiphertext> = input
                    .iter()
                    .map(|ciphertext| ciphertext.limb(limb_index).clone())
                    .collect();

                let expected = tweak(&scalar_input);

                for output_index in 0..n {
                    assert_eq!(
                        actual[output_index].limb(limb_index),
                        &expected[output_index],
                        "RNS Park Tweak diverged from independent scalar limb \
                         for n={n}, limb={limb_index}, output={output_index}"
                    );
                }
            }
        }
    }

    #[test]
    fn monomial_multiplication_respects_negacyclic_wrap() {
        let q = Modulus::new(97);
        let p = Polynomial::new(q, vec![1, 2, 3, 4]);

        assert_eq!(
            polynomial_mul_monomial(&p, 1).coefficients(),
            &[93, 1, 2, 3]
        );

        assert_eq!(polynomial_mul_monomial(&p, 4), p.neg());

        assert_eq!(polynomial_mul_monomial(&p, 8), p);
    }

    #[test]
    fn tweak_matches_direct_definition_for_n1_n2_n4_n8() {
        let q = Modulus::new(12_289);
        let degree = 8;

        for n in [1_usize, 2, 4, 8] {
            let input: Vec<RlweCiphertext> = (0..n)
                .map(|i| deterministic_ciphertext(q, degree, 17 + i as u64))
                .collect();

            let expected = direct_tweak(&input);
            let actual = tweak(&input);

            assert_eq!(
                actual, expected,
                "Park Tweak diverged from direct definition for n={n}"
            );
        }
    }

    #[test]
    fn tweak_preserves_rlwe_decryption_linearity() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 1);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x5041_524b);
        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x0054_5745_414b);

        let ciphertexts: Vec<RlweCiphertext> = (0..8)
            .map(|row| {
                let message = Polynomial::new(
                    params.modulus(),
                    (0..params.degree())
                        .map(|col| {
                            (13 + 7 * row as u64 + 11 * col as u64) % params.modulus().value()
                        })
                        .collect(),
                );

                encrypt_raw_with_rng(params, &secret, &message, &mut encryption_rng)
            })
            .collect();

        let transformed = tweak(&ciphertexts);

        let decrypted_inputs: Vec<Polynomial> = ciphertexts
            .iter()
            .map(|ct| decrypt_raw(params, &secret, ct))
            .collect();

        for (j, ciphertext) in transformed.iter().enumerate() {
            let mut expected = Polynomial::zero(params.modulus(), params.degree());

            for (i, plaintext) in decrypted_inputs.iter().enumerate() {
                let exponent = 2 * i * j * params.degree() / ciphertexts.len();

                expected = expected.add(&polynomial_mul_monomial(plaintext, exponent));
            }

            let actual = decrypt_raw(params, &secret, ciphertext);

            assert_eq!(
                actual, expected,
                "decrypted Park Tweak output diverged at index {j}"
            );
        }
    }

    fn park_galois_keys(
        params: RlweParameters,
        secret: &SecretKey,
        seed: u64,
    ) -> Vec<crate::ckks::GaloisKey> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        (1..params.degree())
            .map(|j| {
                let exponent = 2 * j + 1;

                crate::ckks::GaloisKey::generate_with_rng(params, secret, exponent, 16, &mut rng)
            })
            .collect()
    }

    fn encrypt_coefficient_columns(
        params: RlweParameters,
        secret: &SecretKey,
        matrix: &[Vec<u64>],
        seed: u64,
    ) -> Vec<RlweCiphertext> {
        let degree = params.degree();

        assert_eq!(matrix.len(), degree);
        assert!(matrix.iter().all(|row| row.len() == degree));

        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        (0..degree)
            .map(|column| {
                let message = Polynomial::new(
                    params.modulus(),
                    (0..degree).map(|row| matrix[row][column]).collect(),
                );

                encrypt_raw_with_rng(params, secret, &message, &mut rng)
            })
            .collect()
    }

    fn encrypt_coefficient_rows(
        params: RlweParameters,
        secret: &SecretKey,
        rows: &[Vec<u64>],
        seed: u64,
    ) -> Vec<RlweCiphertext> {
        assert_eq!(rows.len(), params.degree());

        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        rows.iter()
            .map(|row| {
                assert_eq!(row.len(), params.degree());

                let message = Polynomial::new(params.modulus(), row.clone());

                encrypt_raw_with_rng(params, secret, &message, &mut rng)
            })
            .collect()
    }

    fn decrypt_coefficient_rows(
        params: RlweParameters,
        secret: &SecretKey,
        ciphertexts: &[RlweCiphertext],
    ) -> Vec<Vec<u64>> {
        ciphertexts
            .iter()
            .map(|ciphertext| {
                decrypt_raw(params, secret, ciphertext)
                    .coefficients()
                    .to_vec()
            })
            .collect()
    }

    fn transpose_cleartext(matrix: &[Vec<u64>]) -> Vec<Vec<u64>> {
        let n = matrix.len();

        assert!(matrix.iter().all(|row| row.len() == n));

        (0..n)
            .map(|col| (0..n).map(|row| matrix[row][col]).collect())
            .collect()
    }

    #[test]
    fn modular_inverse_matches_expected_values() {
        assert_eq!(modular_inverse(8, 12_289), 10_753);

        for exponent in [1_u64, 3, 5, 7, 9, 11, 13, 15] {
            let inverse = modular_inverse(exponent, 16);

            assert_eq!((exponent * inverse) % 16, 1);
        }
    }

    /// Builds the ordinary N x N coefficient matrix for multiplication by
    /// `polynomial` in Z_q[X]/(X^N + 1).
    ///
    /// Column j is the coefficient vector of polynomial * X^j.
    fn negacyclic_multiplication_matrix(polynomial: &Polynomial) -> Vec<Vec<u64>> {
        let degree = polynomial.degree();
        let modulus = polynomial.modulus();

        let mut matrix = vec![vec![0_u64; degree]; degree];

        for column in 0..degree {
            let mut monomial_coefficients = vec![0_u64; degree];
            monomial_coefficients[column] = 1;

            let monomial = Polynomial::new(modulus, monomial_coefficients);
            let product = polynomial.negacyclic_mul(&monomial);

            for (row, &coefficient) in product.coefficients().iter().enumerate() {
                matrix[row][column] = coefficient;
            }
        }

        matrix
    }

    fn modular_matrix_mul(lhs: &[Vec<u64>], rhs: &[Vec<u64>], modulus: u64) -> Vec<Vec<u64>> {
        let rows = lhs.len();
        let inner = lhs[0].len();
        let cols = rhs[0].len();

        assert_eq!(rhs.len(), inner);

        let mut result = vec![vec![0_u64; cols]; rows];

        for row in 0..rows {
            for col in 0..cols {
                let mut accumulator = 0_u128;

                for (k, rhs_row) in rhs.iter().enumerate().take(inner) {
                    accumulator += (lhs[row][k] as u128) * (rhs_row[col] as u128);
                    accumulator %= modulus as u128;
                }

                result[row][col] = accumulator as u64;
            }
        }

        result
    }

    fn modular_matrix_add(lhs: &[Vec<u64>], rhs: &[Vec<u64>], modulus: u64) -> Vec<Vec<u64>> {
        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(lhs[0].len(), rhs[0].len());

        lhs.iter()
            .zip(rhs)
            .map(|(lhs_row, rhs_row)| {
                lhs_row
                    .iter()
                    .zip(rhs_row)
                    .map(|(&lhs_value, &rhs_value)| {
                        ((lhs_value as u128 + rhs_value as u128) % modulus as u128) as u64
                    })
                    .collect()
            })
            .collect()
    }

    fn modular_matrix_transpose(matrix: &[Vec<u64>]) -> Vec<Vec<u64>> {
        let rows = matrix.len();
        let cols = matrix[0].len();

        let mut result = vec![vec![0_u64; rows]; cols];

        for (row, matrix_row) in matrix.iter().enumerate() {
            for (col, &value) in matrix_row.iter().enumerate() {
                result[col][row] = value;
            }
        }

        result
    }

    fn ciphertext_component_matrix(
        ciphertexts: &[RlweCiphertext],
        select_a: bool,
    ) -> Vec<Vec<u64>> {
        let degree = ciphertexts.len();
        let mut matrix = vec![vec![0_u64; degree]; degree];

        // Ciphertext j is interpreted as coefficient column j.
        for (column, ciphertext) in ciphertexts.iter().enumerate() {
            let polynomial = if select_a {
                ciphertext.a()
            } else {
                ciphertext.b()
            };

            for (row, &coefficient) in polynomial.coefficients().iter().enumerate() {
                matrix[row][column] = coefficient;
            }
        }

        matrix
    }

    fn matrix_from_ciphertext_columns(
        ciphertexts: &[RlweCiphertext],
        select_a: bool,
    ) -> Vec<Vec<u64>> {
        ciphertext_component_matrix(ciphertexts, select_a)
    }

    fn matrix_from_ciphertext_rows(
        ciphertexts: &[RlweCiphertext],
        select_a: bool,
    ) -> Vec<Vec<u64>> {
        modular_matrix_transpose(&ciphertext_component_matrix(ciphertexts, select_a))
    }

    fn synthetic_a_bundle_from_rows(matrix: &[Vec<u64>], modulus: Modulus) -> Vec<RlweCiphertext> {
        let degree = matrix.len();
        assert_eq!(matrix[0].len(), degree);

        (0..degree)
            .map(|row| {
                let a = Polynomial::new(modulus, matrix[row].clone());
                let b = Polynomial::zero(modulus, degree);
                RlweCiphertext::new(b, a)
            })
            .collect()
    }

    #[test]
    fn park_cmt_component_identity_for_synthetic_bundle() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x434d_545f_434f_4d50);
        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let galois_keys = park_galois_keys(params, &secret, 0x434d_545f_4b45_5953);

        let degree = params.degree();
        let modulus = params.modulus();
        let q = modulus.value();

        // Arbitrary, asymmetric ordinary matrix C.
        let c: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        (7 + 11 * row as u64 + 17 * col as u64 + 5 * row as u64 * col as u64) % q
                    })
                    .collect()
            })
            .collect();

        // Construct Park's synthetic pair (C, 0).
        //
        // Park notation is (A,B), whereas FHE-rs stores (b,a).
        // Therefore each ciphertext column is (b=0, a=C[:,j]).
        let synthetic: Vec<RlweCiphertext> = (0..degree)
            .map(|column| {
                let a = Polynomial::new(modulus, (0..degree).map(|row| c[row][column]).collect());

                let b = Polynomial::zero(modulus, degree);

                RlweCiphertext::new(b, a)
            })
            .collect();

        let transformed = transpose(params, &synthetic, &galois_keys);

        // FHE-rs component matrices after C-MT.
        let d0 = ciphertext_component_matrix(&transformed, true);
        let d1 = ciphertext_component_matrix(&transformed, false);

        // S* is defined operationally: the matrix representing
        // multiplication by s(X) in the negacyclic ring.
        let s_star = negacyclic_multiplication_matrix(secret.polynomial());
        let s_star_transpose = modular_matrix_transpose(&s_star);

        // Park Algorithm 8 requires:
        //
        // Because C is packed by columns, C-MT gives:
        //
        //     C^T S*^T = S* D0 + D1.
        //
        let lhs = modular_matrix_mul(&modular_matrix_transpose(&c), &s_star_transpose, q);

        let rhs = modular_matrix_add(&modular_matrix_mul(&s_star, &d0, q), &d1, q);

        assert_eq!(
            lhs, rhs,
            "C-MT component identity C^T*S*^T = S*D0 + D1 failed"
        );
    }

    #[test]
    fn park_algorithm8_representation_flow_matches_matrix_algebra() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5041_524b_5033_4101);
        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let galois_keys = park_galois_keys(params, &secret, 0x5041_524b_5033_4102);

        let degree = params.degree();
        let modulus = params.modulus();
        let q = modulus.value();

        let lhs_plain: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        (3 + 7 * row as u64 + 11 * col as u64 + 2 * row as u64 * col as u64) % q
                    })
                    .collect()
            })
            .collect();

        let rhs_plain: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        (5 + 13 * row as u64 + 17 * col as u64 + 3 * row as u64 * col as u64) % q
                    })
                    .collect()
            })
            .collect();

        let lhs = encrypt_coefficient_rows(params, &secret, &lhs_plain, 0x5041_524b_5033_4103);

        let rhs = encrypt_coefficient_rows(params, &secret, &rhs_plain, 0x5041_524b_5033_4104);

        // Existing C-MT contract:
        // row-packed RHS -> column-packed RHS^T.
        let rhs_cmt = transpose(params, &rhs, &galois_keys);

        // LHS input is row-packed, so its raw component matrices are read
        // by rows. The C-MT output is column-packed, so its components are
        // read by columns.
        let a = matrix_from_ciphertext_rows(&lhs, true);
        let b = matrix_from_ciphertext_rows(&lhs, false);

        let a_tilde = matrix_from_ciphertext_columns(&rhs_cmt, true);
        let b_tilde = matrix_from_ciphertext_columns(&rhs_cmt, false);

        let c00 = modular_matrix_mul(&a, &a_tilde, q);
        let c01 = modular_matrix_mul(&a, &b_tilde, q);
        let c10 = modular_matrix_mul(&b, &a_tilde, q);
        let c11 = modular_matrix_mul(&b, &b_tilde, q);

        // Algorithm 8 Steps 3-4 require C00 and C10 to enter C-MT
        // by rows, not by columns.
        let c00_bundle = synthetic_a_bundle_from_rows(&c00, modulus);
        let c10_bundle = synthetic_a_bundle_from_rows(&c10, modulus);

        let d01 = transpose(params, &c00_bundle, &galois_keys);
        let d23 = transpose(params, &c10_bundle, &galois_keys);

        let d0 = matrix_from_ciphertext_columns(&d01, true);
        let d1 = matrix_from_ciphertext_columns(&d01, false);
        let d2 = matrix_from_ciphertext_columns(&d23, true);
        let d3 = matrix_from_ciphertext_columns(&d23, false);

        let s_star = negacyclic_multiplication_matrix(secret.polynomial());
        let s_star_t = modular_matrix_transpose(&s_star);

        // These are precisely the two component identities required by
        // Algorithm 8 after its second and third C-MTs.
        assert_eq!(
            modular_matrix_mul(&c00, &s_star_t, q),
            modular_matrix_add(&modular_matrix_mul(&s_star, &d0, q), &d1, q,),
            "C00 row-packed C-MT identity failed",
        );

        assert_eq!(
            modular_matrix_mul(&c10, &s_star_t, q),
            modular_matrix_add(&modular_matrix_mul(&s_star, &d2, q), &d3, q,),
            "C10 row-packed C-MT identity failed",
        );

        // Keep the four PP-MM outputs live and dimension-checked.
        assert_eq!(c01.len(), degree);
        assert_eq!(c11.len(), degree);
        assert!(c01.iter().all(|row| row.len() == degree));
        assert!(c11.iter().all(|row| row.len() == degree));
    }
    #[test]
    fn park_cmt_transposes_asymmetric_matrix_exactly() {
        // Noise-free parameters isolate Park Algorithm 4 algebra from
        // approximation/noise behavior.
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x5041_524b_434d_5401);

        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let keys = park_galois_keys(params, &secret, 0x5041_524b_434d_5402);

        let matrix: Vec<Vec<u64>> = (0..8)
            .map(|row| (0..8).map(|col| 1 + 10 * row as u64 + col as u64).collect())
            .collect();

        let ciphertexts = encrypt_coefficient_rows(params, &secret, &matrix, 0x5041_524b_434d_5403);

        let transposed = transpose(params, &ciphertexts, &keys);

        let actual = decrypt_coefficient_rows(params, &secret, &transposed);

        let expected = transpose_cleartext(&matrix);

        assert_eq!(
            actual, expected,
            "Park C-MT failed asymmetric 8x8 transpose"
        );
    }

    #[test]
    fn park_cmt_maps_basis_matrix_eij_to_eji() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x5041_524b_434d_5411);

        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let keys = park_galois_keys(params, &secret, 0x5041_524b_434d_5412);

        for source_row in 0..8 {
            for source_col in 0..8 {
                let mut matrix = vec![vec![0_u64; 8]; 8];
                matrix[source_row][source_col] = 1;

                let ciphertexts = encrypt_coefficient_rows(
                    params,
                    &secret,
                    &matrix,
                    0x5041_524b_434d_5500 + (8 * source_row + source_col) as u64,
                );

                let transposed = transpose(params, &ciphertexts, &keys);

                let actual = decrypt_coefficient_rows(params, &secret, &transposed);

                let expected = transpose_cleartext(&matrix);

                assert_eq!(
                    actual, expected,
                    "Park C-MT failed basis E_{{{source_row},{source_col}}}"
                );
            }
        }
    }

    #[test]
    fn park_cmt_is_involution_on_encrypted_matrix() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x5041_524b_434d_5421);

        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let keys = park_galois_keys(params, &secret, 0x5041_524b_434d_5422);

        let matrix: Vec<Vec<u64>> = (0..8)
            .map(|row| {
                (0..8)
                    .map(|col| {
                        (37 + 19 * row as u64 + 23 * col as u64 + 3 * row as u64 * col as u64)
                            % params.modulus().value()
                    })
                    .collect()
            })
            .collect();

        let ciphertexts = encrypt_coefficient_rows(params, &secret, &matrix, 0x5041_524b_434d_5423);

        let once = transpose(params, &ciphertexts, &keys);

        let twice = transpose(params, &once, &keys);

        let actual = decrypt_coefficient_rows(params, &secret, &twice);

        assert_eq!(actual, matrix, "Park C-MT involution failed");
    }

    fn decrypt_park_quadratic_columns(
        secret: &SecretKey,
        ciphertexts: &[crate::rlwe::RlweQuadraticCiphertext],
    ) -> Vec<Vec<u64>> {
        let degree = secret.polynomial().degree();
        let mut matrix = vec![vec![0_u64; degree]; degree];

        let s = secret.polynomial();
        let s_squared = s.negacyclic_mul(s);

        for (column, ciphertext) in ciphertexts.iter().enumerate() {
            let plaintext = ciphertext
                .c0()
                .add(&ciphertext.c1().negacyclic_mul(s))
                .add(&ciphertext.c2().negacyclic_mul(&s_squared));

            for (row, matrix_row) in matrix.iter_mut().enumerate() {
                matrix_row[column] = plaintext.coefficients()[row];
            }
        }

        matrix
    }

    fn clear_matrix_product(lhs: &[Vec<u64>], rhs: &[Vec<u64>], modulus: u64) -> Vec<Vec<u64>> {
        modular_matrix_mul(lhs, rhs, modulus)
    }

    #[test]
    fn park_ccmm_quadratic_right_identity() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5041_524b_5033_4201);
        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let galois_keys = park_galois_keys(params, &secret, 0x5041_524b_5033_4202);

        let degree = params.degree();

        let matrix: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| 19 + 23 * row as u64 + 29 * col as u64 + 3 * row as u64 * col as u64)
                    .collect()
            })
            .collect();

        let mut identity = vec![vec![0_u64; degree]; degree];
        for (index, row) in identity.iter_mut().enumerate() {
            row[index] = 1;
        }

        let lhs = encrypt_coefficient_columns(params, &secret, &matrix, 0x5041_524b_5033_4203);

        let rhs = encrypt_coefficient_columns(params, &secret, &identity, 0x5041_524b_5033_4204);

        let product = ccmm_quadratic(params, &lhs, &rhs, &galois_keys);

        assert_eq!(
            decrypt_park_quadratic_columns(&secret, &product),
            matrix,
            "Park quadratic CC-MM failed M x I"
        );
    }

    #[test]
    fn park_ccmm_quadratic_left_identity() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5041_524b_5033_4211);
        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let galois_keys = park_galois_keys(params, &secret, 0x5041_524b_5033_4212);

        let degree = params.degree();

        let matrix: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| 31 + 17 * row as u64 + 13 * col as u64 + 5 * row as u64 * col as u64)
                    .collect()
            })
            .collect();

        let mut identity = vec![vec![0_u64; degree]; degree];
        for (index, row) in identity.iter_mut().enumerate() {
            row[index] = 1;
        }

        let lhs = encrypt_coefficient_columns(params, &secret, &identity, 0x5041_524b_5033_4213);

        let rhs = encrypt_coefficient_columns(params, &secret, &matrix, 0x5041_524b_5033_4214);

        let product = ccmm_quadratic(params, &lhs, &rhs, &galois_keys);

        assert_eq!(
            decrypt_park_quadratic_columns(&secret, &product),
            matrix,
            "Park quadratic CC-MM failed I x M"
        );
    }

    #[test]
    fn park_ccmm_quadratic_matches_asymmetric_clear_product() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5041_524b_5033_4221);
        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let galois_keys = park_galois_keys(params, &secret, 0x5041_524b_5033_4222);

        let degree = params.degree();
        let q = params.modulus().value();

        let lhs_plain: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        (3 + 5 * row as u64 + 7 * col as u64 + 2 * row as u64 * col as u64) % q
                    })
                    .collect()
            })
            .collect();

        let rhs_plain: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        (11 + 13 * row as u64 + 17 * col as u64 + 3 * row as u64 * col as u64) % q
                    })
                    .collect()
            })
            .collect();

        let lhs = encrypt_coefficient_columns(params, &secret, &lhs_plain, 0x5041_524b_5033_4223);

        let rhs = encrypt_coefficient_columns(params, &secret, &rhs_plain, 0x5041_524b_5033_4224);

        let product = ccmm_quadratic(params, &lhs, &rhs, &galois_keys);

        let actual = decrypt_park_quadratic_columns(&secret, &product);

        let expected = clear_matrix_product(&lhs_plain, &rhs_plain, q);

        assert_eq!(
            actual, expected,
            "Park quadratic CC-MM failed asymmetric product"
        );
    }

    #[test]
    fn park_ccmm_quadratic_basis_products() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5041_524b_5033_4231);
        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let galois_keys = park_galois_keys(params, &secret, 0x5041_524b_5033_4232);

        let degree = params.degree();

        for i in 0..degree {
            for j in 0..degree {
                for k in 0..degree {
                    let mut lhs_plain = vec![vec![0_u64; degree]; degree];
                    lhs_plain[i][j] = 1;

                    let mut rhs_plain = vec![vec![0_u64; degree]; degree];
                    rhs_plain[j][k] = 1;

                    let lhs = encrypt_coefficient_columns(
                        params,
                        &secret,
                        &lhs_plain,
                        0x5041_524b_6000_0000 + (i * 64 + j * 8 + k) as u64,
                    );

                    let rhs = encrypt_coefficient_columns(
                        params,
                        &secret,
                        &rhs_plain,
                        0x5041_524b_7000_0000 + (i * 64 + j * 8 + k) as u64,
                    );

                    let product = ccmm_quadratic(params, &lhs, &rhs, &galois_keys);

                    let actual = decrypt_park_quadratic_columns(&secret, &product);

                    let mut expected = vec![vec![0_u64; degree]; degree];
                    expected[i][k] = 1;

                    assert_eq!(
                        actual, expected,
                        "Park basis product E_{{{i},{j}}} E_{{{j},{k}}} failed"
                    );
                }
            }
        }
    }

    #[test]
    fn park_ccmm_relinearization_preserves_quadratic_decryption() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);
        let degree = params.degree();
        let q = params.modulus().value();

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5033_425f_5345_4352);
        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let galois_keys = park_galois_keys(params, &secret, 0x5033_425f_4741_4c4f);

        let mut multiplication_key_rng = ChaCha20Rng::seed_from_u64(0x5033_425f_5245_4c49);
        let multiplication_key = crate::eval::MultiplicationKey::generate_with_rng(
            params,
            &secret,
            16,
            &mut multiplication_key_rng,
        );

        let lhs_plain: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        (3 + 7 * row as u64 + 11 * col as u64 + 5 * row as u64 * col as u64) % q
                    })
                    .collect()
            })
            .collect();

        let rhs_plain: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        (13 + 17 * row as u64 + 19 * col as u64 + 3 * row as u64 * col as u64) % q
                    })
                    .collect()
            })
            .collect();

        let lhs = encrypt_coefficient_columns(params, &secret, &lhs_plain, 0x5033_425f_4c48_5301);
        let rhs = encrypt_coefficient_columns(params, &secret, &rhs_plain, 0x5033_425f_5248_5301);

        let quadratic = ccmm_quadratic(params, &lhs, &rhs, &galois_keys);
        let relinearized = ccmm_relinearized(params, &lhs, &rhs, &galois_keys, &multiplication_key);

        assert_eq!(quadratic.len(), degree);
        assert_eq!(relinearized.len(), degree);

        for column in 0..degree {
            let before = crate::rlwe::decrypt_quadratic_raw(params, &secret, &quadratic[column]);

            let after = decrypt_raw(params, &secret, &relinearized[column]);

            assert_eq!(
                after, before,
                "Park relinearization changed decrypted output column {column}"
            );
        }
    }

    #[test]
    fn park_ccmm_relinearized_matches_asymmetric_clear_product() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);
        let degree = params.degree();
        let q = params.modulus().value();

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5033_425f_4153_594d);
        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let galois_keys = park_galois_keys(params, &secret, 0x5033_425f_474b_4153);

        let mut multiplication_key_rng = ChaCha20Rng::seed_from_u64(0x5033_425f_4d4b_4153);
        let multiplication_key = crate::eval::MultiplicationKey::generate_with_rng(
            params,
            &secret,
            16,
            &mut multiplication_key_rng,
        );

        let lhs_plain: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        (5 + 7 * row as u64 + 13 * col as u64 + 3 * row as u64 * col as u64) % q
                    })
                    .collect()
            })
            .collect();

        let rhs_plain: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        (11 + 17 * row as u64 + 19 * col as u64 + 5 * row as u64 * col as u64) % q
                    })
                    .collect()
            })
            .collect();

        let expected = modular_matrix_mul(&lhs_plain, &rhs_plain, q);

        let lhs = encrypt_coefficient_columns(params, &secret, &lhs_plain, 0x5033_425f_4c48_5302);
        let rhs = encrypt_coefficient_columns(params, &secret, &rhs_plain, 0x5033_425f_5248_5302);

        let output = ccmm_relinearized(params, &lhs, &rhs, &galois_keys, &multiplication_key);

        let actual_rows = decrypt_coefficient_rows(params, &secret, &output);
        let actual = transpose_cleartext(&actual_rows);

        assert_eq!(
            actual, expected,
            "relinearized Park CC-MM diverged from asymmetric clear product"
        );
    }

    #[test]
    fn park_ccmm_relinearized_basis_products() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);
        let degree = params.degree();

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5033_425f_4241_5349);
        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let galois_keys = park_galois_keys(params, &secret, 0x5033_425f_474b_4241);

        let mut multiplication_key_rng = ChaCha20Rng::seed_from_u64(0x5033_425f_4d4b_4241);
        let multiplication_key = crate::eval::MultiplicationKey::generate_with_rng(
            params,
            &secret,
            16,
            &mut multiplication_key_rng,
        );

        for i in 0..degree {
            for j in 0..degree {
                for k in 0..degree {
                    let mut lhs_plain = vec![vec![0_u64; degree]; degree];
                    let mut rhs_plain = vec![vec![0_u64; degree]; degree];
                    let mut expected = vec![vec![0_u64; degree]; degree];

                    lhs_plain[i][j] = 1;
                    rhs_plain[j][k] = 1;
                    expected[i][k] = 1;

                    let case = ((i * degree + j) * degree + k) as u64;

                    let lhs = encrypt_coefficient_columns(
                        params,
                        &secret,
                        &lhs_plain,
                        0x5033_425f_4c00_0000 ^ case,
                    );

                    let rhs = encrypt_coefficient_columns(
                        params,
                        &secret,
                        &rhs_plain,
                        0x5033_425f_5200_0000 ^ case,
                    );

                    let output =
                        ccmm_relinearized(params, &lhs, &rhs, &galois_keys, &multiplication_key);

                    let actual_rows = decrypt_coefficient_rows(params, &secret, &output);
                    let actual = transpose_cleartext(&actual_rows);

                    assert_eq!(
                        actual, expected,
                        "relinearized Park basis product failed \
                         E_{{{i},{j}}} * E_{{{j},{k}}} = E_{{{i},{k}}}"
                    );
                }
            }
        }
    }
}
