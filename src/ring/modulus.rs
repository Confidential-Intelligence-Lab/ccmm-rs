/// Unsigned modular arithmetic over `Z_q`.
///
/// Values supplied to arithmetic operations may be unreduced. Results are
/// always returned in the canonical interval `[0, q)`.
///
/// Multiplication uses `u128` intermediates so products of two `u64`
/// operands cannot overflow before modular reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Modulus {
    value: u64,
}

impl Modulus {
    /// Constructs a modulus `q`.
    ///
    /// # Panics
    ///
    /// Panics if `q < 2`.
    pub fn new(value: u64) -> Self {
        assert!(value >= 2, "modulus must be at least 2");
        Self { value }
    }

    /// Returns the modulus `q`.
    pub fn value(self) -> u64 {
        self.value
    }

    /// Reduces `a` into `[0, q)`.
    pub fn reduce(self, a: u64) -> u64 {
        a % self.value
    }

    /// Computes `(a + b) mod q`.
    pub fn add(self, a: u64, b: u64) -> u64 {
        let q = u128::from(self.value);
        ((u128::from(a) % q + u128::from(b) % q) % q) as u64
    }

    /// Computes `(a - b) mod q`.
    pub fn sub(self, a: u64, b: u64) -> u64 {
        let q = u128::from(self.value);
        let a = u128::from(a) % q;
        let b = u128::from(b) % q;

        ((a + q - b) % q) as u64
    }

    /// Computes `(-a) mod q`.
    pub fn neg(self, a: u64) -> u64 {
        let reduced = self.reduce(a);

        if reduced == 0 {
            0
        } else {
            self.value - reduced
        }
    }

    /// Computes `(a * b) mod q`.
    pub fn mul(self, a: u64, b: u64) -> u64 {
        let q = u128::from(self.value);

        ((u128::from(a) * u128::from(b)) % q) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_preserves_modulus() {
        let modulus = Modulus::new(17);

        assert_eq!(modulus.value(), 17);
    }

    #[test]
    #[should_panic(expected = "modulus must be at least 2")]
    fn rejects_zero() {
        let _ = Modulus::new(0);
    }

    #[test]
    #[should_panic(expected = "modulus must be at least 2")]
    fn rejects_one() {
        let _ = Modulus::new(1);
    }

    #[test]
    fn reduction_is_canonical() {
        let modulus = Modulus::new(17);

        assert_eq!(modulus.reduce(0), 0);
        assert_eq!(modulus.reduce(16), 16);
        assert_eq!(modulus.reduce(17), 0);
        assert_eq!(modulus.reduce(35), 1);
    }

    #[test]
    fn addition_reduces_result() {
        let modulus = Modulus::new(17);

        assert_eq!(modulus.add(5, 7), 12);
        assert_eq!(modulus.add(16, 16), 15);
        assert_eq!(modulus.add(17, 18), 1);
    }

    #[test]
    fn subtraction_wraps_modulus() {
        let modulus = Modulus::new(17);

        assert_eq!(modulus.sub(7, 5), 2);
        assert_eq!(modulus.sub(0, 1), 16);
        assert_eq!(modulus.sub(17, 18), 16);
    }

    #[test]
    fn negation_is_canonical() {
        let modulus = Modulus::new(17);

        assert_eq!(modulus.neg(0), 0);
        assert_eq!(modulus.neg(1), 16);
        assert_eq!(modulus.neg(16), 1);
        assert_eq!(modulus.neg(17), 0);
        assert_eq!(modulus.neg(18), 16);
    }

    #[test]
    fn multiplication_reduces_result() {
        let modulus = Modulus::new(17);

        assert_eq!(modulus.mul(5, 7), 1);
        assert_eq!(modulus.mul(16, 16), 1);
        assert_eq!(modulus.mul(17, 23), 0);
    }

    #[test]
    fn multiplication_handles_large_u64_operands() {
        let modulus = Modulus::new(1_000_000_007);

        let a = u64::MAX;
        let b = u64::MAX - 1;

        let expected = ((u128::from(a) * u128::from(b)) % u128::from(modulus.value())) as u64;

        assert_eq!(modulus.mul(a, b), expected);
    }

    #[test]
    fn addition_handles_large_u64_operands() {
        let modulus = Modulus::new(u64::MAX - 58);

        let a = u64::MAX;
        let b = u64::MAX;

        let q = u128::from(modulus.value());
        let expected = ((u128::from(a) % q + u128::from(b) % q) % q) as u64;

        assert_eq!(modulus.add(a, b), expected);
    }

    #[test]
    fn ring_identities_hold() {
        let modulus = Modulus::new(97);

        for a in 0..97 {
            assert_eq!(modulus.add(a, 0), a);
            assert_eq!(modulus.sub(a, a), 0);
            assert_eq!(modulus.add(a, modulus.neg(a)), 0);
            assert_eq!(modulus.mul(a, 0), 0);
            assert_eq!(modulus.mul(a, 1), a);
        }
    }
}
