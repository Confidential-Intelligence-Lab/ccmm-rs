/// Unsigned modular arithmetic over `Z_q`.
///
/// Values supplied to arithmetic operations may be unreduced. Results are
/// always returned in the canonical interval `[0, q)`.
///
/// Multiplication uses `u128` intermediates so products of two `u64`
/// operands cannot overflow before modular reduction.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedModulus {
    modulus: Modulus,
    reciprocal: u128,
}

impl PreparedModulus {
    /// Prepares a fixed modulus for Barrett-style reduction.
    ///
    /// The reciprocal is floor(2^128 / q). It is computed once and reused
    /// across hot modular multiplications.
    pub fn new(modulus: Modulus) -> Self {
        let q = u128::from(modulus.value());

        // floor(2^128 / q) without constructing 2^128 directly:
        //
        // floor((2^128 - 1) / q) is either floor(2^128 / q) or one less.
        // Correct it using the remainder.
        let max = u128::MAX;
        let mut reciprocal = max / q;
        let remainder = max % q;

        if remainder == q - 1 {
            reciprocal += 1;
        }

        Self {
            modulus,
            reciprocal,
        }
    }

    pub fn modulus(self) -> Modulus {
        self.modulus
    }

    pub fn reciprocal(self) -> u128 {
        self.reciprocal
    }

    /// Exact portable multiplication of canonical residues.
    ///
    /// This first implementation keeps the product in u128 and uses the
    /// prepared reciprocal to estimate the quotient. Correction restores the
    /// exact canonical residue.
    pub fn mul_canonical(self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.modulus.value());
        debug_assert!(b < self.modulus.value());

        let q = u128::from(self.modulus.value());
        let x = u128::from(a) * u128::from(b);

        // For the SD3b large modulus x is at most 72 bits.
        //
        // Compute floor(x * reciprocal / 2^128). Since x is small enough,
        // split the reciprocal product into high and low halves manually.
        let x_hi = x >> 64;
        let x_lo = x as u64 as u128;

        let r_hi = self.reciprocal >> 64;
        let r_lo = self.reciprocal as u64 as u128;

        let p0 = x_lo * r_lo;
        let p1 = x_lo * r_hi;
        let p2 = x_hi * r_lo;
        let p3 = x_hi * r_hi;

        let carry = ((p0 >> 64) + (p1 & ((1_u128 << 64) - 1)) + (p2 & ((1_u128 << 64) - 1))) >> 64;

        let quotient = p3 + (p1 >> 64) + (p2 >> 64) + carry;

        let mut reduced = x - quotient * q;

        while reduced >= q {
            reduced -= q;
        }

        reduced as u64
    }
}

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

    /// Adds two already-canonical residues modulo q.
    ///
    /// Both operands must lie in `[0, q)`. This avoids the general-purpose
    /// reductions performed by `add`.
    pub fn add_canonical(self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.value, "lhs residue must be canonical");
        debug_assert!(b < self.value, "rhs residue must be canonical");

        let (sum, overflow) = a.overflowing_add(b);

        if overflow {
            sum.wrapping_sub(self.value)
        } else if sum >= self.value {
            sum - self.value
        } else {
            sum
        }
    }

    /// Subtracts two already-canonical residues modulo q.
    ///
    /// Both operands must lie in `[0, q)`. This avoids the general-purpose
    /// reductions performed by `sub`.
    pub fn sub_canonical(self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.value, "lhs residue must be canonical");
        debug_assert!(b < self.value, "rhs residue must be canonical");

        if a >= b {
            a - b
        } else {
            self.value - (b - a)
        }
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

    /// Multiplies two already-canonical residues modulo q.
    ///
    /// When the square of the modulus fits in `u64`, the product of two
    /// canonical residues also fits in `u64`, avoiding the general `u128`
    /// multiplication/reduction path. Larger moduli retain the exact
    /// portable `u128` implementation.
    pub fn mul_canonical(self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.value, "lhs residue must be canonical");
        debug_assert!(b < self.value, "rhs residue must be canonical");

        if u128::from(self.value) * u128::from(self.value) <= u128::from(u64::MAX) {
            (a * b) % self.value
        } else {
            ((u128::from(a) * u128::from(b)) % u128::from(self.value)) as u64
        }
    }

    /// Computes `base^exponent mod q` by repeated squaring.
    pub fn pow(self, base: u64, mut exponent: u64) -> u64 {
        let mut base = self.reduce(base);
        let mut result = 1_u64;

        while exponent > 0 {
            if exponent & 1 == 1 {
                result = self.mul(result, base);
            }

            base = self.mul(base, base);
            exponent >>= 1;
        }

        result
    }

    /// Computes the multiplicative inverse modulo a prime modulus.
    ///
    /// This uses Fermat's little theorem:
    ///
    /// `a^{-1} = a^{q-2} mod q`.
    ///
    /// The caller is responsible for ensuring that `q` is prime.
    ///
    /// # Panics
    ///
    /// Panics if `a == 0 mod q`.
    pub fn inverse_prime(self, a: u64) -> u64 {
        let reduced = self.reduce(a);

        assert!(reduced != 0, "zero has no multiplicative inverse");

        self.pow(reduced, self.value - 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_multiplication_matches_general_multiplication() {
        let moduli = [Modulus::new(268_238_849), Modulus::new(68_712_923_137)];

        for modulus in moduli {
            let q = modulus.value();

            let values = [0, 1, 2, q / 2, q - 2, q - 1];

            for &lhs in &values {
                for &rhs in &values {
                    assert_eq!(
                        modulus.mul_canonical(lhs, rhs),
                        modulus.mul(lhs, rhs),
                        "canonical multiplication mismatch for q={q}, lhs={lhs}, rhs={rhs}"
                    );
                }
            }
        }

        println!("MODULUS_CANONICAL_MUL_EQUIVALENCE=PASS");
    }

    #[test]
    fn prepared_modulus_multiplication_matches_general_multiplication() {
        let moduli = [Modulus::new(268_238_849), Modulus::new(68_712_923_137)];

        for modulus in moduli {
            let prepared = PreparedModulus::new(modulus);
            let q = modulus.value();

            let edge_values = [0, 1, 2, 3, q / 3, q / 2, q - 3, q - 2, q - 1];

            for &lhs in &edge_values {
                for &rhs in &edge_values {
                    assert_eq!(
                        prepared.mul_canonical(lhs, rhs),
                        modulus.mul(lhs, rhs),
                        "prepared multiplication mismatch for q={q}, lhs={lhs}, rhs={rhs}"
                    );
                }
            }

            let mut state = 0x1234_5678_9abc_def0_u64;

            for _ in 0..10_000 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;

                let lhs = state % q;

                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;

                let rhs = state % q;

                assert_eq!(
                    prepared.mul_canonical(lhs, rhs),
                    modulus.mul(lhs, rhs),
                    "prepared multiplication mismatch for q={q}, lhs={lhs}, rhs={rhs}"
                );
            }
        }

        println!("PREPARED_MODULUS_MUL_EQUIVALENCE=PASS");
    }

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
    fn canonical_add_sub_match_general_arithmetic() {
        let moduli = [
            2_u64,
            3,
            17,
            97,
            12_289,
            268_238_849,
            68_712_923_137,
            u64::MAX - 58,
        ];

        for &q in &moduli {
            let modulus = Modulus::new(q);

            let values = [0, 1 % q, 2 % q, q / 2, q.saturating_sub(2), q - 1];

            for &a in &values {
                for &b in &values {
                    assert!(a < q);
                    assert!(b < q);

                    assert_eq!(
                        modulus.add_canonical(a, b),
                        modulus.add(a, b),
                        "canonical add mismatch for q={q}, a={a}, b={b}"
                    );

                    assert_eq!(
                        modulus.sub_canonical(a, b),
                        modulus.sub(a, b),
                        "canonical sub mismatch for q={q}, a={a}, b={b}"
                    );
                }
            }
        }

        println!("MODULUS_CANONICAL_ADD_SUB_EQUIVALENCE=PASS");
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
    fn modular_exponentiation_works() {
        let modulus = Modulus::new(17);

        assert_eq!(modulus.pow(3, 0), 1);
        assert_eq!(modulus.pow(3, 1), 3);
        assert_eq!(modulus.pow(3, 4), 13);
        assert_eq!(modulus.pow(20, 4), 13);
    }

    #[test]
    fn inverse_prime_works() {
        let modulus = Modulus::new(17);

        for value in 1..17 {
            let inverse = modulus.inverse_prime(value);
            assert_eq!(modulus.mul(value, inverse), 1);
        }
    }

    #[test]
    #[should_panic(expected = "zero has no multiplicative inverse")]
    fn inverse_prime_rejects_zero() {
        let _ = Modulus::new(17).inverse_prime(0);
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
