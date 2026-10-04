use crate::ring::{ModulusBasis, RnsPolynomial};

/// One contiguous RNS gadget block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsGadgetBlock {
    start: usize,
    end: usize,
    basis: ModulusBasis,
}

impl RnsGadgetBlock {
    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn basis(&self) -> &ModulusBasis {
        &self.basis
    }

    pub fn composite_modulus(&self) -> u128 {
        self.basis.composite_modulus()
    }
}

/// Partition of a full RNS basis into contiguous gadget blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsGadgetLayout {
    full_basis: ModulusBasis,
    blocks: Vec<RnsGadgetBlock>,
}

impl RnsGadgetLayout {
    pub fn new(full_basis: ModulusBasis, block_sizes: Vec<usize>) -> Self {
        assert!(
            !block_sizes.is_empty(),
            "RNS gadget layout must contain at least one block"
        );

        assert!(
            block_sizes.iter().all(|&size| size > 0),
            "RNS gadget blocks must be nonempty"
        );

        assert_eq!(
            block_sizes.iter().sum::<usize>(),
            full_basis.len(),
            "RNS gadget blocks must cover the full basis exactly"
        );

        let mut blocks = Vec::with_capacity(block_sizes.len());
        let mut start = 0;

        for size in block_sizes {
            let end = start + size;

            let basis = ModulusBasis::new(full_basis.moduli()[start..end].to_vec());

            blocks.push(RnsGadgetBlock { start, end, basis });

            start = end;
        }

        Self { full_basis, blocks }
    }

    pub fn full_basis(&self) -> &ModulusBasis {
        &self.full_basis
    }

    pub fn blocks(&self) -> &[RnsGadgetBlock] {
        &self.blocks
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the CRT idempotent for one gadget block reduced modulo one
    /// target RNS limb.
    ///
    /// For a singleton gadget block containing exactly one modulus q_i,
    /// the CRT idempotent has the one-hot RNS representation
    ///
    /// E_i mod q_j = 1 when i == j, and 0 otherwise.
    ///
    /// This representation avoids materializing the full composite modulus Q
    /// and is therefore suitable for wide RNS bases.
    pub fn crt_idempotent_residue(&self, block_index: usize, limb_index: usize) -> u64 {
        assert!(
            block_index < self.blocks.len(),
            "gadget block index out of range"
        );
        assert!(
            limb_index < self.full_basis.len(),
            "RNS limb index out of range"
        );

        let block = &self.blocks[block_index];

        assert_eq!(
            block.end() - block.start(),
            1,
            "RNS-native CRT idempotent residues currently require singleton gadget blocks"
        );

        if limb_index == block.start() {
            1
        } else {
            0
        }
    }

    /// CRT idempotent for gadget block `i`.
    ///
    /// If block modulus is Qi and full modulus is Q:
    ///
    /// E_i = (Q/Qi) * inverse(Q/Qi mod Qi) mod Q.
    pub fn crt_idempotent(&self, index: usize) -> u128 {
        let q = self.full_basis.composite_modulus();

        let qi = self.blocks[index].composite_modulus();

        let q_hat = q / qi;

        let inverse = inverse_mod_u128(q_hat % qi, qi);

        q_hat
            .checked_mul(inverse)
            .expect("CRT idempotent exceeds u128")
            % q
    }

    /// Decomposes one RNS polynomial into canonical block residues.
    pub fn decompose(&self, polynomial: &RnsPolynomial) -> RnsGadgetDecomposition {
        assert_eq!(
            polynomial.basis(),
            &self.full_basis,
            "polynomial basis must match RNS gadget layout"
        );

        let needs_reconstruction = self
            .blocks
            .iter()
            .any(|block| block.end() - block.start() != 1);

        let canonical = needs_reconstruction.then(|| polynomial.reconstruct_coefficients());

        let digits = self
            .blocks
            .iter()
            .map(|block| {
                if block.end() - block.start() == 1 {
                    polynomial
                        .residue(block.start())
                        .coefficients()
                        .iter()
                        .map(|&value| u128::from(value))
                        .collect::<Vec<_>>()
                } else {
                    let qi = block.composite_modulus();

                    canonical
                        .as_ref()
                        .expect("multi-limb gadget block requires CRT reconstruction")
                        .iter()
                        .map(|value| value % qi)
                        .collect::<Vec<_>>()
                }
            })
            .collect();

        RnsGadgetDecomposition {
            layout: self.clone(),
            digits,
        }
    }
}

/// Canonical coefficient digits for an RNS gadget decomposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsGadgetDecomposition {
    layout: RnsGadgetLayout,
    digits: Vec<Vec<u128>>,
}

impl RnsGadgetDecomposition {
    pub fn layout(&self) -> &RnsGadgetLayout {
        &self.layout
    }

    pub fn digits(&self) -> &[Vec<u128>] {
        &self.digits
    }

    pub fn digit(&self, index: usize) -> &[u128] {
        &self.digits[index]
    }

    /// Reconstructs the full canonical coefficients modulo Q.
    pub fn reconstruct_coefficients(&self) -> Vec<u128> {
        let q = self.layout.full_basis().composite_modulus();

        let degree = self.digits[0].len();

        let mut output = vec![0_u128; degree];

        for block_index in 0..self.digits.len() {
            let idempotent = self.layout.crt_idempotent(block_index);

            for (coefficient_index, &digit) in self.digits[block_index].iter().enumerate() {
                let term = digit
                    .checked_mul(idempotent)
                    .expect("CRT reconstruction term exceeds u128")
                    % q;

                output[coefficient_index] = (output[coefficient_index] + term) % q;
            }
        }

        output
    }

    /// Lifts a block digit into the full RNS basis as the canonical
    /// coefficient representative `x mod Q_i`.
    ///
    /// Multiplication by the corresponding CRT idempotent is deliberately
    /// kept separate; that scalar becomes the evaluation-key gadget factor.
    pub fn lift_digit(&self, index: usize) -> RnsPolynomial {
        RnsPolynomial::from_coefficients(
            self.layout.full_basis().moduli().to_vec(),
            &self.digits[index],
        )
    }
}

fn inverse_mod_u128(value: u128, modulus: u128) -> u128 {
    assert!(modulus > 1, "inverse modulus must exceed one");

    let mut t: i128 = 0;
    let mut new_t: i128 = 1;

    let mut r = i128::try_from(modulus).expect("modulus exceeds i128");

    let mut new_r = i128::try_from(value).expect("value exceeds i128");

    while new_r != 0 {
        let quotient = r / new_r;

        (t, new_t) = (new_t, t - quotient * new_t);

        (r, new_r) = (new_r, r - quotient * new_r);
    }

    assert_eq!(r, 1, "CRT factor is not invertible");

    if t < 0 {
        t += i128::try_from(modulus).expect("modulus exceeds i128");
    }

    t as u128
}

#[cfg(test)]
mod tests {

    #[test]
    fn singleton_crt_idempotent_residues_are_one_hot() {
        let basis = ModulusBasis::new(vec![
            crate::ring::Modulus::new(12_289),
            crate::ring::Modulus::new(40_961),
            crate::ring::Modulus::new(65_537),
        ]);

        let layout = RnsGadgetLayout::new(basis, vec![1, 1, 1]);

        for block_index in 0..3 {
            for limb_index in 0..3 {
                let expected = if block_index == limb_index { 1 } else { 0 };

                assert_eq!(
                    layout.crt_idempotent_residue(block_index, limb_index),
                    expected
                );
            }
        }
    }

    #[test]
    fn singleton_crt_idempotent_residues_match_legacy_idempotents() {
        let basis = ModulusBasis::new(vec![
            crate::ring::Modulus::new(12_289),
            crate::ring::Modulus::new(40_961),
            crate::ring::Modulus::new(65_537),
        ]);

        let layout = RnsGadgetLayout::new(basis, vec![1, 1, 1]);

        for block_index in 0..layout.block_count() {
            let legacy = layout.crt_idempotent(block_index);

            for limb_index in 0..layout.full_basis().len() {
                let modulus = layout.full_basis().modulus(limb_index);

                assert_eq!(
                    layout.crt_idempotent_residue(block_index, limb_index),
                    (legacy % u128::from(modulus.value())) as u64
                );
            }
        }
    }

    use super::*;
    use crate::ring::Modulus;

    fn basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
            Modulus::new(114_689),
        ])
    }

    #[test]
    fn blocks_partition_full_basis_exactly() {
        let layout = RnsGadgetLayout::new(basis(), vec![1, 2, 1]);

        assert_eq!(layout.block_count(), 3);

        assert_eq!(layout.blocks()[0].basis().moduli(), &[Modulus::new(12_289)]);

        assert_eq!(
            layout.blocks()[1].basis().moduli(),
            &[Modulus::new(40_961), Modulus::new(65_537),]
        );

        assert_eq!(
            layout.blocks()[2].basis().moduli(),
            &[Modulus::new(114_689)]
        );
    }

    #[test]
    fn crt_idempotents_are_one_on_own_block_and_zero_on_others() {
        let layout = RnsGadgetLayout::new(basis(), vec![1, 2, 1]);

        for i in 0..layout.block_count() {
            let e = layout.crt_idempotent(i);

            for j in 0..layout.block_count() {
                let qj = layout.blocks()[j].composite_modulus();

                if i == j {
                    assert_eq!(e % qj, 1);
                } else {
                    assert_eq!(e % qj, 0);
                }
            }
        }
    }

    #[test]
    fn decomposition_reconstructs_exact_coefficients() {
        let basis = basis();

        let q = basis.composite_modulus();

        let coefficients = vec![0, 1, 17, 42, 1_001, 65_536, q / 2, q - 1];

        let polynomial = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &coefficients);

        for block_sizes in [
            vec![4],
            vec![1, 3],
            vec![2, 2],
            vec![1, 2, 1],
            vec![1, 1, 1, 1],
        ] {
            let layout = RnsGadgetLayout::new(basis.clone(), block_sizes);

            let decomposition = layout.decompose(&polynomial);

            assert_eq!(decomposition.reconstruct_coefficients(), coefficients);
        }
    }

    #[test]
    fn lifted_digit_has_full_basis() {
        let basis = basis();

        let polynomial = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &[1, 2, 3, 4]);

        let layout = RnsGadgetLayout::new(basis.clone(), vec![2, 2]);

        let decomposition = layout.decompose(&polynomial);

        for index in 0..2 {
            assert_eq!(decomposition.lift_digit(index).basis(), &basis);
        }
    }

    #[test]
    fn differential_campaign_reconstructs_for_multiple_layouts() {
        let basis = basis();

        let q = basis.composite_modulus();

        for seed in 0_u128..32 {
            let coefficients: Vec<u128> = (0..32_u128)
                .map(|index| (97 * seed + 31 * index + 7 * index * index + 11) % q)
                .collect();

            let polynomial =
                RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &coefficients);

            for block_sizes in [
                vec![4],
                vec![1, 3],
                vec![2, 2],
                vec![1, 2, 1],
                vec![1, 1, 1, 1],
            ] {
                let decomposition =
                    RnsGadgetLayout::new(basis.clone(), block_sizes).decompose(&polynomial);

                assert_eq!(
                    decomposition.reconstruct_coefficients(),
                    coefficients,
                    "RNS gadget reconstruction mismatch for seed {seed}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "cover the full basis")]
    fn rejects_incomplete_layout() {
        let _ = RnsGadgetLayout::new(basis(), vec![1, 2]);
    }

    #[test]
    #[should_panic(expected = "nonempty")]
    fn rejects_empty_block() {
        let _ = RnsGadgetLayout::new(basis(), vec![1, 0, 3]);
    }
}
