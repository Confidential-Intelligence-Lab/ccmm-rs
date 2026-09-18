use crate::ring::Modulus;

/// Representation selected for one universal sprout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UniversalSprout {
    /// Exact power-of-two modulus `2^bits`.
    ///
    /// This representation is used for small bit sizes where no suitable
    /// NTT-friendly prime can exist for the configured ring degree.
    PowerOfTwo { bits: u32 },

    /// Prime modulus satisfying the negacyclic NTT requirement
    ///
    /// ```text
    /// q = 1 mod 2N
    /// ```
    NttPrime { bits: u32, modulus: Modulus },
}

impl UniversalSprout {
    pub fn bits(&self) -> u32 {
        match self {
            Self::PowerOfTwo { bits } | Self::NttPrime { bits, .. } => *bits,
        }
    }

    /// Logical sprout modulus as an integer.
    pub fn modulus(&self) -> u128 {
        match self {
            Self::PowerOfTwo { bits } => {
                assert!(*bits < 128, "power-of-two sprout exceeds u128");

                1_u128 << bits
            }

            Self::NttPrime { modulus, .. } => u128::from(modulus.value()),
        }
    }

    pub fn is_power_of_two(&self) -> bool {
        matches!(self, Self::PowerOfTwo { .. })
    }

    pub fn is_ntt_prime(&self) -> bool {
        matches!(self, Self::NttPrime { .. })
    }
}

/// Policy for constructing universal sprouts for one ring degree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniversalSproutPolicy {
    degree: usize,
    machine_word_bits: u32,
}

impl UniversalSproutPolicy {
    pub fn new(degree: usize, machine_word_bits: u32) -> Self {
        assert!(
            degree > 0 && degree.is_power_of_two(),
            "ring degree must be a positive power of two"
        );

        assert!(
            machine_word_bits > 0 && machine_word_bits <= 63,
            "machine-word bit size must be in 1..=63"
        );

        Self {
            degree,
            machine_word_bits,
        }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn machine_word_bits(&self) -> u32 {
        self.machine_word_bits
    }

    /// Minimum bit size that could possibly hold an NTT-friendly modulus.
    ///
    /// Since a negacyclic NTT modulus must satisfy
    ///
    /// ```text
    /// q = 1 mod 2N
    /// ```
    ///
    /// it must be strictly larger than `2N`.
    pub fn minimum_ntt_bits(&self) -> u32 {
        let minimum_value = (2_u128 * self.degree as u128) + 1;

        u128::BITS - minimum_value.leading_zeros()
    }

    /// Constructs the canonical power-of-two representation for a small
    /// universal sprout.
    pub fn power_of_two(&self, bits: u32) -> UniversalSprout {
        assert!(bits > 0, "sprout bit size must be positive");

        assert!(
            bits <= self.machine_word_bits,
            "sprout exceeds configured machine-word size"
        );

        UniversalSprout::PowerOfTwo { bits }
    }

    /// Validates and constructs an NTT-friendly prime sprout.
    pub fn ntt_prime(&self, bits: u32, modulus: Modulus) -> UniversalSprout {
        assert!(
            bits > 0 && bits <= self.machine_word_bits,
            "sprout bit size must fit configured machine word"
        );

        let value = modulus.value();

        assert_eq!(
            (value - 1) % (2 * self.degree) as u64,
            0,
            "sprout prime must satisfy q = 1 mod 2N"
        );

        assert_eq!(
            bit_length_u64(value),
            bits,
            "sprout prime bit length does not match declaration"
        );

        UniversalSprout::NttPrime { bits, modulus }
    }
}

fn bit_length_u64(value: u64) -> u32 {
    u64::BITS - value.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_reports_ntt_threshold_for_degree_32768() {
        let policy = UniversalSproutPolicy::new(1 << 15, 63);

        // 2N + 1 = 65537, requiring 17 bits.
        assert_eq!(policy.minimum_ntt_bits(), 17);
    }

    #[test]
    fn power_of_two_sprout_has_exact_requested_modulus() {
        let policy = UniversalSproutPolicy::new(1 << 15, 63);

        for bits in 1_u32..=20 {
            let sprout = policy.power_of_two(bits);

            assert_eq!(sprout.bits(), bits);

            assert_eq!(sprout.modulus(), 1_u128 << bits);

            assert!(sprout.is_power_of_two());
        }
    }

    #[test]
    fn ntt_prime_sprout_validates_ring_constraint() {
        let policy = UniversalSproutPolicy::new(8, 63);

        let sprout = policy.ntt_prime(7, Modulus::new(97));

        assert!(sprout.is_ntt_prime());

        assert_eq!(sprout.bits(), 7);

        assert_eq!(sprout.modulus(), 97);
    }

    #[test]
    #[should_panic(expected = "q = 1 mod 2N")]
    fn rejects_prime_incompatible_with_ring_degree() {
        let policy = UniversalSproutPolicy::new(8, 63);

        let _ = policy.ntt_prime(7, Modulus::new(101));
    }

    #[test]
    #[should_panic(expected = "bit length")]
    fn rejects_wrong_declared_prime_bit_size() {
        let policy = UniversalSproutPolicy::new(8, 63);

        let _ = policy.ntt_prime(6, Modulus::new(97));
    }

    #[test]
    #[should_panic(expected = "machine-word")]
    fn rejects_power_of_two_larger_than_word() {
        let policy = UniversalSproutPolicy::new(1 << 15, 63);

        let _ = policy.power_of_two(64);
    }

    #[test]
    fn all_sub_threshold_sizes_have_power_of_two_representation() {
        let policy = UniversalSproutPolicy::new(1 << 15, 63);

        for bits in 1..policy.minimum_ntt_bits() {
            let sprout = policy.power_of_two(bits);

            assert_eq!(sprout.modulus(), 1_u128 << bits);
        }
    }
}
