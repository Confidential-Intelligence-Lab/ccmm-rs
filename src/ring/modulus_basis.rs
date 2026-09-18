use super::Modulus;

/// Ordered, validated residue-number-system modulus basis.
///
/// A basis represents:
///
/// `Q = q_0 * q_1 * ... * q_{L-1}`
///
/// where all `q_i` are pairwise coprime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModulusBasis {
    moduli: Vec<Modulus>,
}

impl ModulusBasis {
    pub fn new(moduli: Vec<Modulus>) -> Self {
        assert!(
            !moduli.is_empty(),
            "modulus basis must contain at least one modulus"
        );

        for (index, lhs) in moduli.iter().enumerate() {
            for rhs in &moduli[index + 1..] {
                assert_eq!(
                    gcd(lhs.value(), rhs.value()),
                    1,
                    "modulus basis elements must be pairwise coprime"
                );
            }
        }

        Self { moduli }
    }

    pub fn len(&self) -> usize {
        self.moduli.len()
    }

    pub fn is_empty(&self) -> bool {
        self.moduli.is_empty()
    }

    pub fn moduli(&self) -> &[Modulus] {
        &self.moduli
    }

    pub fn modulus(&self, index: usize) -> Modulus {
        self.moduli[index]
    }

    pub fn composite_modulus(&self) -> u128 {
        self.moduli.iter().fold(1_u128, |product, modulus| {
            product
                .checked_mul(u128::from(modulus.value()))
                .expect("composite modulus exceeds u128")
        })
    }

    pub fn contains(&self, modulus: Modulus) -> bool {
        self.moduli.contains(&modulus)
    }

    pub fn prefix(&self, length: usize) -> Self {
        assert!(
            length > 0 && length <= self.len(),
            "basis prefix length must be in 1..=basis length"
        );

        Self::new(self.moduli[..length].to_vec())
    }

    pub fn without_last(&self) -> Self {
        assert!(
            self.len() > 1,
            "cannot remove the last modulus from a one-limb basis"
        );

        self.prefix(self.len() - 1)
    }
}

fn gcd(mut lhs: u64, mut rhs: u64) -> u64 {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }

    lhs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ])
    }

    #[test]
    fn preserves_ordered_moduli() {
        let basis = basis();

        assert_eq!(
            basis.moduli(),
            &[
                Modulus::new(12_289),
                Modulus::new(40_961),
                Modulus::new(65_537),
            ]
        );
    }

    #[test]
    fn composite_modulus_is_product() {
        let basis = basis();

        assert_eq!(
            basis.composite_modulus(),
            12_289_u128 * 40_961_u128 * 65_537_u128
        );
    }

    #[test]
    fn prefix_preserves_order() {
        let basis = basis();

        assert_eq!(
            basis.prefix(2).moduli(),
            &[Modulus::new(12_289), Modulus::new(40_961),]
        );
    }

    #[test]
    fn without_last_drops_exactly_one_modulus() {
        let basis = basis();

        assert_eq!(basis.without_last(), basis.prefix(2));
    }

    #[test]
    fn contains_reports_membership() {
        let basis = basis();

        assert!(basis.contains(Modulus::new(40_961)));

        assert!(!basis.contains(Modulus::new(97)));
    }

    #[test]
    #[should_panic(expected = "at least one")]
    fn rejects_empty_basis() {
        let _ = ModulusBasis::new(vec![]);
    }

    #[test]
    #[should_panic(expected = "pairwise coprime")]
    fn rejects_non_coprime_basis() {
        let _ = ModulusBasis::new(vec![Modulus::new(15), Modulus::new(21)]);
    }

    #[test]
    #[should_panic(expected = "cannot remove")]
    fn rejects_dropping_only_modulus() {
        let basis = ModulusBasis::new(vec![Modulus::new(12_289)]);

        let _ = basis.without_last();
    }
}
