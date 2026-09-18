use crate::ring::{ModulusBasis, Polynomial, RnsPolynomial};

use super::Pow2Polynomial;

/// Hybrid RNS representation with ordinary odd-prime limbs and one
/// power-of-two sprout limb.
///
/// The represented modulus is
/// `Q = product(q_i) * 2^sprout_bits`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pow2RnsPolynomial {
    ordinary: RnsPolynomial,
    sprout: Pow2Polynomial,
}

impl Pow2RnsPolynomial {
    pub fn from_coefficients(
        ordinary_basis: &ModulusBasis,
        sprout_bits: u32,
        coefficients: &[u128],
    ) -> Self {
        assert!(
            sprout_bits > 0 && sprout_bits <= 63,
            "power-of-two sprout bits must be in 1..=63"
        );

        let ordinary =
            RnsPolynomial::from_coefficients(ordinary_basis.moduli().to_vec(), coefficients);

        let sprout_modulus = 1_u128 << sprout_bits;

        let sprout = Pow2Polynomial::new(
            sprout_bits,
            coefficients
                .iter()
                .map(|&value| (value % sprout_modulus) as u64)
                .collect(),
        );

        Self { ordinary, sprout }
    }

    pub fn from_parts(ordinary: RnsPolynomial, sprout: Pow2Polynomial) -> Self {
        assert_eq!(
            ordinary.degree(),
            sprout.degree(),
            "ordinary and sprout polynomial degrees must match"
        );

        Self { ordinary, sprout }
    }

    pub fn ordinary(&self) -> &RnsPolynomial {
        &self.ordinary
    }

    pub fn sprout(&self) -> &Pow2Polynomial {
        &self.sprout
    }

    pub fn ordinary_basis(&self) -> &ModulusBasis {
        self.ordinary.basis()
    }

    pub fn sprout_bits(&self) -> u32 {
        self.sprout.bits()
    }

    pub fn degree(&self) -> usize {
        self.ordinary.degree()
    }

    pub fn composite_modulus(&self) -> u128 {
        self.ordinary
            .basis()
            .composite_modulus()
            .checked_mul(1_u128 << self.sprout_bits())
            .expect("hybrid RNS modulus exceeds u128")
    }

    /// Exact CRT reconstruction of the hybrid odd-RNS / power-of-two
    /// representation.
    pub fn reconstruct_coefficients(&self) -> Vec<u128> {
        let ordinary_values = self.ordinary.reconstruct_coefficients();

        let ordinary_modulus = self.ordinary_basis().composite_modulus();

        let pow2_modulus = 1_u128 << self.sprout_bits();

        let ordinary_mod_pow2 = ordinary_modulus % pow2_modulus;

        let inverse = inverse_odd_mod_power_of_two(ordinary_mod_pow2 as u64, self.sprout_bits());

        ordinary_values
            .iter()
            .zip(self.sprout.coefficients())
            .map(|(&ordinary_value, &sprout_value)| {
                let ordinary_low = ordinary_value % pow2_modulus;

                let difference =
                    (pow2_modulus + u128::from(sprout_value) - ordinary_low) % pow2_modulus;

                let t = (difference * u128::from(inverse)) % pow2_modulus;

                ordinary_value + ordinary_modulus * t
            })
            .collect()
    }
}

/// Inverse rescale by `2^e`.
///
/// This maps a basis ending in `2^delta` to one ending in
/// `2^(delta+e)` while representing the integer `2^e * a`.
pub fn inv_rs_power_of_two(polynomial: &Pow2RnsPolynomial, e: u32) -> Pow2RnsPolynomial {
    assert!(e > 0, "Inv-RS exponent must be positive");

    let target_bits = polynomial
        .sprout_bits()
        .checked_add(e)
        .expect("Inv-RS sprout size overflow");

    assert!(target_bits <= 63, "Inv-RS target sprout exceeds 63 bits");

    let factor = 1_u64 << e;

    let ordinary_residues: Vec<Polynomial> = polynomial
        .ordinary()
        .residues()
        .iter()
        .map(|residue| {
            let modulus = residue.modulus();

            Polynomial::new(
                modulus,
                residue
                    .coefficients()
                    .iter()
                    .map(|&value| modulus.mul(value, factor % modulus.value()))
                    .collect(),
            )
        })
        .collect();

    let ordinary = RnsPolynomial::from_residues(ordinary_residues);

    let sprout = Pow2Polynomial::new(
        target_bits,
        polynomial
            .sprout()
            .coefficients()
            .iter()
            .map(|&value| value << e)
            .collect(),
    );

    Pow2RnsPolynomial::from_parts(ordinary, sprout)
}

/// Rescale by `2^e`.
///
/// Input has a `2^(delta+e)` sprout. The operation subtracts the
/// residue modulo `2^e`, divides exactly by `2^e`, and returns a
/// representation with a `2^delta` sprout.
pub fn rs_power_of_two(polynomial: &Pow2RnsPolynomial, e: u32) -> Pow2RnsPolynomial {
    assert!(e > 0, "RS exponent must be positive");

    assert!(
        polynomial.sprout_bits() > e,
        "RS exponent must be smaller than sprout bit size"
    );

    let target_bits = polynomial.sprout_bits() - e;

    let factor = 1_u64 << e;

    let low_mask = factor - 1;

    let ordinary_residues: Vec<Polynomial> = polynomial
        .ordinary()
        .residues()
        .iter()
        .map(|residue| {
            let modulus = residue.modulus();

            let factor_mod_q = factor % modulus.value();

            let inverse = modulus.inverse_prime(factor_mod_q);

            let coefficients = residue
                .coefficients()
                .iter()
                .zip(polynomial.sprout().coefficients())
                .map(|(&ordinary_value, &sprout_value)| {
                    let low = sprout_value & low_mask;

                    let difference = modulus.sub(ordinary_value, low % modulus.value());

                    modulus.mul(difference, inverse)
                })
                .collect();

            Polynomial::new(modulus, coefficients)
        })
        .collect();

    let ordinary = RnsPolynomial::from_residues(ordinary_residues);

    let sprout = Pow2Polynomial::new(
        target_bits,
        polynomial
            .sprout()
            .coefficients()
            .iter()
            .map(|&value| {
                let low = value & low_mask;

                (value - low) >> e
            })
            .collect(),
    );

    Pow2RnsPolynomial::from_parts(ordinary, sprout)
}

/// Multiplicative inverse of an odd integer modulo `2^bits`.
fn inverse_odd_mod_power_of_two(value: u64, bits: u32) -> u64 {
    assert!(value & 1 == 1, "value must be odd modulo a power of two");

    assert!(
        bits > 0 && bits <= 63,
        "power-of-two inverse bits must be in 1..=63"
    );

    // Newton iteration doubles the number of correct bits each round.
    let mut inverse = 1_u64;

    for _ in 0..6 {
        inverse = inverse.wrapping_mul(2_u64.wrapping_sub(value.wrapping_mul(inverse)));
    }

    let mask = (1_u64 << bits) - 1;

    inverse & mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::{Modulus, ModulusBasis};

    fn basis() -> ModulusBasis {
        ModulusBasis::new(vec![Modulus::new(12_289), Modulus::new(40_961)])
    }

    #[test]
    fn hybrid_representation_roundtrips_through_crt() {
        let basis = basis();
        let bits = 12;

        let modulus = basis.composite_modulus() * (1_u128 << bits);

        let coefficients = vec![0, 1, 17, 42, 4095, 65_536, modulus / 2, modulus - 1];

        let polynomial = Pow2RnsPolynomial::from_coefficients(&basis, bits, &coefficients);

        assert_eq!(polynomial.reconstruct_coefficients(), coefficients);
    }

    #[test]
    fn inv_rs_represents_exact_multiplication_by_power_of_two() {
        let basis = basis();
        let bits = 8;
        let e = 5;

        let source_modulus = basis.composite_modulus() * (1_u128 << bits);

        let coefficients = vec![0, 1, 17, 42, 1_001, source_modulus / 7];

        let source = Pow2RnsPolynomial::from_coefficients(&basis, bits, &coefficients);

        let lifted = inv_rs_power_of_two(&source, e);

        let target_modulus = lifted.composite_modulus();

        let expected: Vec<u128> = coefficients
            .iter()
            .map(|value| (value * (1_u128 << e)) % target_modulus)
            .collect();

        assert_eq!(lifted.reconstruct_coefficients(), expected);

        assert_eq!(lifted.sprout_bits(), bits + e);
    }

    #[test]
    fn rs_after_inv_rs_is_exact_identity() {
        let basis = basis();

        for bits in [4_u32, 8, 12, 16] {
            for e in [1_u32, 2, 3] {
                let source_modulus = basis.composite_modulus() * (1_u128 << bits);

                let coefficients: Vec<u128> = (0..32_u128)
                    .map(|index| (17 + 31 * index + 7 * index * index) % source_modulus)
                    .collect();

                let source = Pow2RnsPolynomial::from_coefficients(&basis, bits, &coefficients);

                let lifted = inv_rs_power_of_two(&source, e);

                let recovered = rs_power_of_two(&lifted, e);

                assert_eq!(recovered, source, "RS/Inv-RS mismatch: bits={bits}, e={e}");
            }
        }
    }

    #[test]
    fn rs_matches_integer_definition() {
        let basis = basis();
        let delta = 8;
        let e = 4;

        let target_bits = delta + e;

        let modulus = basis.composite_modulus() * (1_u128 << target_bits);

        let coefficients: Vec<u128> = (0..32_u128)
            .map(|index| (101 + 37 * index + 11 * index * index) % modulus)
            .collect();

        let source = Pow2RnsPolynomial::from_coefficients(&basis, target_bits, &coefficients);

        let rescaled = rs_power_of_two(&source, e);

        let low_mask = (1_u128 << e) - 1;

        let expected: Vec<u128> = coefficients
            .iter()
            .map(|&value| {
                let low = value & low_mask;

                (value - low) >> e
            })
            .collect();

        assert_eq!(rescaled.reconstruct_coefficients(), expected);
    }

    #[test]
    fn differential_campaign_matches_integer_rs_definition() {
        let basis = basis();

        for delta in [4_u32, 8, 12, 16] {
            for e in [1_u32, 2, 3] {
                let input_bits = delta + e;

                let modulus = basis.composite_modulus() * (1_u128 << input_bits);

                for seed in 0_u128..32 {
                    let coefficients: Vec<u128> = (0..16_u128)
                        .map(|index| (97 * seed + 29 * index + 5 * index * index + 13) % modulus)
                        .collect();

                    let source =
                        Pow2RnsPolynomial::from_coefficients(&basis, input_bits, &coefficients);

                    let actual = rs_power_of_two(&source, e).reconstruct_coefficients();

                    let mask = (1_u128 << e) - 1;

                    let expected: Vec<u128> = coefficients
                        .iter()
                        .map(|&value| (value - (value & mask)) >> e)
                        .collect();

                    assert_eq!(
                        actual, expected,
                        "RS mismatch: delta={delta}, e={e}, seed={seed}"
                    );
                }
            }
        }
    }
}
