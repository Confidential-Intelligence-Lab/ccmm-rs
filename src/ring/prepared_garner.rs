//! Prepared arbitrary-precision Garner CRT reconstruction.
//!
//! Precomputes modulus-dependent constants for repeated reconstruction
//! over the same RNS basis.

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};

use super::{ModulusBasis, RnsPolynomial};

pub struct PreparedGarner {
    moduli: Vec<u64>,
    prefix_products: Vec<BigUint>,
    inverses: Vec<u64>,
}

fn inverse_mod(value: u64, modulus: u64) -> u64 {
    let (mut old_r, mut r) = (i128::from(value), i128::from(modulus));
    let (mut old_s, mut s) = (1_i128, 0_i128);

    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
    }

    assert_eq!(old_r, 1);
    old_s.rem_euclid(i128::from(modulus)) as u64
}

impl PreparedGarner {
    pub fn new(basis: &ModulusBasis) -> Self {
        let moduli: Vec<u64> = basis.moduli().iter().map(|m| m.value()).collect();

        let mut prefix = BigUint::one();
        let mut prefix_products = Vec::with_capacity(moduli.len());
        let mut inverses = Vec::with_capacity(moduli.len());

        for &modulus in &moduli {
            let reduced = (&prefix % modulus)
                .to_u64()
                .expect("reduced prefix fits u64");

            prefix_products.push(prefix.clone());
            inverses.push(inverse_mod(reduced, modulus));

            prefix *= modulus;
        }

        Self {
            moduli,
            prefix_products,
            inverses,
        }
    }

    pub fn reconstruct(&self, polynomial: &RnsPolynomial) -> Vec<BigUint> {
        assert_eq!(polynomial.residues().len(), self.moduli.len());

        for (residue, &modulus) in polynomial.residues().iter().zip(&self.moduli) {
            assert_eq!(residue.modulus().value(), modulus);
        }

        (0..polynomial.degree())
            .map(|coefficient_index| {
                let mut value = BigUint::zero();

                for (limb_index, residue) in polynomial.residues().iter().enumerate() {
                    let modulus = self.moduli[limb_index];
                    let residue_value = residue.coefficients()[coefficient_index];

                    let value_mod = (&value % modulus).to_u64().expect("reduced value fits u64");

                    let delta = if residue_value >= value_mod {
                        residue_value - value_mod
                    } else {
                        modulus - (value_mod - residue_value)
                    };

                    let correction = (u128::from(delta) * u128::from(self.inverses[limb_index]))
                        % u128::from(modulus);

                    value += &self.prefix_products[limb_index] * correction;
                }

                value
            })
            .collect()
    }
}
