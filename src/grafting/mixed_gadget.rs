use super::Pow2RnsPolynomial;
use crate::grafting::RnsGadgetLayout;
use crate::ring::ModulusBasis;

/// Gadget decomposition over an odd-prime RNS basis plus one
/// power-of-two sprout block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedGadgetLayout {
    ordinary: RnsGadgetLayout,
    sprout_bits: u32,
}

impl MixedGadgetLayout {
    pub fn new(
        ordinary_basis: ModulusBasis,
        ordinary_block_sizes: Vec<usize>,
        sprout_bits: u32,
    ) -> Self {
        assert!(
            sprout_bits > 0 && sprout_bits <= 63,
            "mixed gadget sprout bits must be in 1..=63"
        );

        Self {
            ordinary: RnsGadgetLayout::new(ordinary_basis, ordinary_block_sizes),
            sprout_bits,
        }
    }

    pub fn ordinary_layout(&self) -> &RnsGadgetLayout {
        &self.ordinary
    }

    pub fn ordinary_basis(&self) -> &ModulusBasis {
        self.ordinary.full_basis()
    }

    pub fn sprout_bits(&self) -> u32 {
        self.sprout_bits
    }

    pub fn sprout_modulus(&self) -> u128 {
        1_u128 << self.sprout_bits
    }

    pub fn composite_modulus(&self) -> u128 {
        self.ordinary_basis()
            .composite_modulus()
            .checked_mul(self.sprout_modulus())
            .expect("mixed gadget modulus exceeds u128")
    }

    pub fn block_count(&self) -> usize {
        self.ordinary.block_count() + 1
    }

    /// Full-modulus CRT idempotent for an ordinary gadget block.
    pub fn ordinary_idempotent(&self, block_index: usize) -> u128 {
        let q = self.composite_modulus();

        let qi = self.ordinary.blocks()[block_index].composite_modulus();

        let q_hat = q / qi;

        let inverse = inverse_mod_u128(q_hat % qi, qi);

        q_hat
            .checked_mul(inverse)
            .expect("mixed ordinary CRT idempotent exceeds u128")
            % q
    }

    /// Full-modulus CRT idempotent for the power-of-two sprout block.
    pub fn sprout_idempotent(&self) -> u128 {
        let q = self.composite_modulus();

        let sprout_q = self.sprout_modulus();

        let q_hat = q / sprout_q;

        let inverse = inverse_odd_mod_power_of_two(q_hat as u64, self.sprout_bits);

        q_hat
            .checked_mul(u128::from(inverse))
            .expect("mixed sprout CRT idempotent exceeds u128")
            % q
    }

    pub fn decompose(&self, polynomial: &Pow2RnsPolynomial) -> MixedGadgetDecomposition {
        assert_eq!(
            polynomial.ordinary_basis(),
            self.ordinary_basis(),
            "hybrid polynomial ordinary basis must match mixed gadget layout"
        );

        assert_eq!(
            polynomial.sprout_bits(),
            self.sprout_bits,
            "hybrid polynomial sprout size must match mixed gadget layout"
        );

        let canonical = polynomial.reconstruct_coefficients();

        let ordinary_digits = self
            .ordinary
            .blocks()
            .iter()
            .map(|block| {
                let qi = block.composite_modulus();

                canonical.iter().map(|value| value % qi).collect::<Vec<_>>()
            })
            .collect();

        let sprout_mask = self.sprout_modulus() - 1;

        let sprout_digit = canonical
            .iter()
            .map(|value| (value & sprout_mask) as u64)
            .collect();

        MixedGadgetDecomposition {
            layout: self.clone(),
            ordinary_digits,
            sprout_digit,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedGadgetDecomposition {
    layout: MixedGadgetLayout,
    ordinary_digits: Vec<Vec<u128>>,
    sprout_digit: Vec<u64>,
}

impl MixedGadgetDecomposition {
    pub fn layout(&self) -> &MixedGadgetLayout {
        &self.layout
    }

    pub fn ordinary_digits(&self) -> &[Vec<u128>] {
        &self.ordinary_digits
    }

    pub fn ordinary_digit(&self, index: usize) -> &[u128] {
        &self.ordinary_digits[index]
    }

    pub fn sprout_digit(&self) -> &[u64] {
        &self.sprout_digit
    }

    pub fn reconstruct_coefficients(&self) -> Vec<u128> {
        let q = self.layout.composite_modulus();

        let degree = self.sprout_digit.len();

        let mut output = vec![0_u128; degree];

        for block_index in 0..self.ordinary_digits.len() {
            let idempotent = self.layout.ordinary_idempotent(block_index);

            for (coefficient_index, &digit) in self.ordinary_digits[block_index].iter().enumerate()
            {
                let term = digit
                    .checked_mul(idempotent)
                    .expect("mixed CRT ordinary term exceeds u128")
                    % q;

                output[coefficient_index] = (output[coefficient_index] + term) % q;
            }
        }

        let sprout_idempotent = self.layout.sprout_idempotent();

        for (coefficient_index, &digit) in self.sprout_digit.iter().enumerate() {
            let term = u128::from(digit)
                .checked_mul(sprout_idempotent)
                .expect("mixed CRT sprout term exceeds u128")
                % q;

            output[coefficient_index] = (output[coefficient_index] + term) % q;
        }

        output
    }
}

fn inverse_mod_u128(value: u128, modulus: u128) -> u128 {
    let mut t: i128 = 0;
    let mut new_t: i128 = 1;

    let mut r = i128::try_from(modulus).expect("modulus exceeds i128");

    let mut new_r = i128::try_from(value).expect("value exceeds i128");

    while new_r != 0 {
        let quotient = r / new_r;

        (t, new_t) = (new_t, t - quotient * new_t);

        (r, new_r) = (new_r, r - quotient * new_r);
    }

    assert_eq!(r, 1, "mixed CRT factor is not invertible");

    if t < 0 {
        t += i128::try_from(modulus).expect("modulus exceeds i128");
    }

    t as u128
}

fn inverse_odd_mod_power_of_two(value: u64, bits: u32) -> u64 {
    assert!(
        value & 1 == 1,
        "power-of-two CRT inverse requires odd input"
    );

    let mut inverse = 1_u64;

    for _ in 0..6 {
        inverse = inverse.wrapping_mul(2_u64.wrapping_sub(value.wrapping_mul(inverse)));
    }

    inverse & ((1_u64 << bits) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grafting::Pow2GraftedBasis;
    use crate::ring::{Modulus, ModulusBasis};

    fn basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ])
    }

    #[test]
    fn mixed_layout_has_terminal_sprout_block() {
        let layout = MixedGadgetLayout::new(basis(), vec![1, 2], 12);

        assert_eq!(layout.block_count(), 3);

        assert_eq!(layout.ordinary_layout().block_count(), 2);
    }

    #[test]
    fn ordinary_idempotents_are_correct_over_full_hybrid_modulus() {
        let layout = MixedGadgetLayout::new(basis(), vec![1, 2], 12);

        let sprout_q = layout.sprout_modulus();

        for i in 0..layout.ordinary_layout().block_count() {
            let e = layout.ordinary_idempotent(i);

            let own_q = layout.ordinary_layout().blocks()[i].composite_modulus();

            assert_eq!(e % own_q, 1);

            assert_eq!(e % sprout_q, 0);

            for j in 0..layout.ordinary_layout().block_count() {
                if i != j {
                    let other_q = layout.ordinary_layout().blocks()[j].composite_modulus();

                    assert_eq!(e % other_q, 0);
                }
            }
        }
    }

    #[test]
    fn sprout_idempotent_is_one_on_pow2_and_zero_on_odd_basis() {
        let layout = MixedGadgetLayout::new(basis(), vec![1, 2], 12);

        let e = layout.sprout_idempotent();

        assert_eq!(e % layout.sprout_modulus(), 1);

        for &modulus in layout.ordinary_basis().moduli() {
            assert_eq!(e % u128::from(modulus.value(),), 0);
        }
    }

    #[test]
    fn decomposition_reconstructs_exact_hybrid_coefficients() {
        let layout = MixedGadgetLayout::new(basis(), vec![1, 2], 12);

        let grafted = Pow2GraftedBasis::new(basis(), 12);

        let q = grafted.composite_modulus();

        let coefficients = vec![0, 1, 17, 42, 4_095, 65_536, q / 2, q - 1];

        let polynomial = grafted.from_coefficients(&coefficients);

        let decomposition = layout.decompose(&polynomial);

        assert_eq!(decomposition.reconstruct_coefficients(), coefficients);
    }

    #[test]
    fn decomposition_campaign_is_exact() {
        for bits in [4_u32, 8, 12, 16] {
            for block_sizes in [vec![3], vec![1, 2], vec![2, 1], vec![1, 1, 1]] {
                let layout = MixedGadgetLayout::new(basis(), block_sizes, bits);

                let grafted = Pow2GraftedBasis::new(basis(), bits);

                let q = grafted.composite_modulus();

                for seed in 0_u128..32 {
                    let coefficients: Vec<u128> = (0..32_u128)
                        .map(|index| (97 * seed + 31 * index + 7 * index * index + 11) % q)
                        .collect();

                    let polynomial = grafted.from_coefficients(&coefficients);

                    let actual = layout.decompose(&polynomial).reconstruct_coefficients();

                    assert_eq!(
                        actual, coefficients,
                        "mixed gadget mismatch: bits={bits}, seed={seed}"
                    );
                }
            }
        }
    }
}
