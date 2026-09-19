use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};

/// Arithmetic contract for one physical RNS limb.
///
/// This abstraction is deliberately below `ModulusBasis`, CKKS logical
/// levels, and Grafting. It models only arithmetic modulo one physical
/// modulus and therefore does not impose any CKKS level semantics.
pub trait PhysicalLimbArithmetic {
    type Word: Copy + Eq + core::fmt::Debug;

    const WORD_BITS: u32;

    fn modulus(&self) -> Self::Word;

    fn canonicalize(&self, value: Self::Word) -> Self::Word;

    fn add_mod(&self, lhs: Self::Word, rhs: Self::Word) -> Self::Word;

    fn sub_mod(&self, lhs: Self::Word, rhs: Self::Word) -> Self::Word;

    fn neg_mod(&self, value: Self::Word) -> Self::Word;

    fn mul_mod(&self, lhs: Self::Word, rhs: Self::Word) -> Self::Word;

    fn pow_mod(&self, base: Self::Word, exponent: u128) -> Self::Word;

    fn inv_mod(&self, value: Self::Word) -> Option<Self::Word>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limb32 {
    modulus: u32,
}

impl Limb32 {
    pub fn new(modulus: u32) -> Self {
        assert!(modulus > 1, "physical modulus must exceed one");
        Self { modulus }
    }
}

impl PhysicalLimbArithmetic for Limb32 {
    type Word = u32;

    const WORD_BITS: u32 = 32;

    fn modulus(&self) -> Self::Word {
        self.modulus
    }

    fn canonicalize(&self, value: Self::Word) -> Self::Word {
        value % self.modulus
    }

    fn add_mod(&self, lhs: Self::Word, rhs: Self::Word) -> Self::Word {
        let q = u64::from(self.modulus);
        ((u64::from(lhs % self.modulus) + u64::from(rhs % self.modulus)) % q) as u32
    }

    fn sub_mod(&self, lhs: Self::Word, rhs: Self::Word) -> Self::Word {
        let lhs = lhs % self.modulus;
        let rhs = rhs % self.modulus;

        if lhs >= rhs {
            lhs - rhs
        } else {
            self.modulus - (rhs - lhs)
        }
    }

    fn neg_mod(&self, value: Self::Word) -> Self::Word {
        let value = value % self.modulus;
        if value == 0 {
            0
        } else {
            self.modulus - value
        }
    }

    fn mul_mod(&self, lhs: Self::Word, rhs: Self::Word) -> Self::Word {
        let q = u64::from(self.modulus);
        ((u64::from(lhs % self.modulus) * u64::from(rhs % self.modulus)) % q) as u32
    }

    fn pow_mod(&self, base: Self::Word, mut exponent: u128) -> Self::Word {
        let mut base = self.canonicalize(base);
        let mut result = 1 % self.modulus;

        while exponent != 0 {
            if exponent & 1 == 1 {
                result = self.mul_mod(result, base);
            }

            exponent >>= 1;

            if exponent != 0 {
                base = self.mul_mod(base, base);
            }
        }

        result
    }

    fn inv_mod(&self, value: Self::Word) -> Option<Self::Word> {
        inverse_u128(u128::from(value % self.modulus), u128::from(self.modulus))
            .map(|value| value as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limb64 {
    modulus: u64,
}

impl Limb64 {
    pub fn new(modulus: u64) -> Self {
        assert!(modulus > 1, "physical modulus must exceed one");
        Self { modulus }
    }
}

impl PhysicalLimbArithmetic for Limb64 {
    type Word = u64;

    const WORD_BITS: u32 = 64;

    fn modulus(&self) -> Self::Word {
        self.modulus
    }

    fn canonicalize(&self, value: Self::Word) -> Self::Word {
        value % self.modulus
    }

    fn add_mod(&self, lhs: Self::Word, rhs: Self::Word) -> Self::Word {
        let q = u128::from(self.modulus);
        ((u128::from(lhs % self.modulus) + u128::from(rhs % self.modulus)) % q) as u64
    }

    fn sub_mod(&self, lhs: Self::Word, rhs: Self::Word) -> Self::Word {
        let lhs = lhs % self.modulus;
        let rhs = rhs % self.modulus;

        if lhs >= rhs {
            lhs - rhs
        } else {
            self.modulus - (rhs - lhs)
        }
    }

    fn neg_mod(&self, value: Self::Word) -> Self::Word {
        let value = value % self.modulus;
        if value == 0 {
            0
        } else {
            self.modulus - value
        }
    }

    fn mul_mod(&self, lhs: Self::Word, rhs: Self::Word) -> Self::Word {
        let q = u128::from(self.modulus);
        ((u128::from(lhs % self.modulus) * u128::from(rhs % self.modulus)) % q) as u64
    }

    fn pow_mod(&self, base: Self::Word, mut exponent: u128) -> Self::Word {
        let mut base = self.canonicalize(base);
        let mut result = 1 % self.modulus;

        while exponent != 0 {
            if exponent & 1 == 1 {
                result = self.mul_mod(result, base);
            }

            exponent >>= 1;

            if exponent != 0 {
                base = self.mul_mod(base, base);
            }
        }

        result
    }

    fn inv_mod(&self, value: Self::Word) -> Option<Self::Word> {
        inverse_u128(u128::from(value % self.modulus), u128::from(self.modulus))
            .map(|value| value as u64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limb128 {
    modulus: u128,
}

impl Limb128 {
    pub fn new(modulus: u128) -> Self {
        assert!(modulus > 1, "physical modulus must exceed one");
        Self { modulus }
    }
}

impl PhysicalLimbArithmetic for Limb128 {
    type Word = u128;

    const WORD_BITS: u32 = 128;

    fn modulus(&self) -> Self::Word {
        self.modulus
    }

    fn canonicalize(&self, value: Self::Word) -> Self::Word {
        value % self.modulus
    }

    fn add_mod(&self, lhs: Self::Word, rhs: Self::Word) -> Self::Word {
        let lhs = lhs % self.modulus;
        let rhs = rhs % self.modulus;

        if lhs >= self.modulus - rhs {
            lhs - (self.modulus - rhs)
        } else {
            lhs + rhs
        }
    }

    fn sub_mod(&self, lhs: Self::Word, rhs: Self::Word) -> Self::Word {
        let lhs = lhs % self.modulus;
        let rhs = rhs % self.modulus;

        if lhs >= rhs {
            lhs - rhs
        } else {
            self.modulus - (rhs - lhs)
        }
    }

    fn neg_mod(&self, value: Self::Word) -> Self::Word {
        let value = value % self.modulus;
        if value == 0 {
            0
        } else {
            self.modulus - value
        }
    }

    fn mul_mod(&self, lhs: Self::Word, rhs: Self::Word) -> Self::Word {
        let product = BigUint::from(lhs % self.modulus) * BigUint::from(rhs % self.modulus);
        let reduced = product % BigUint::from(self.modulus);

        reduced
            .to_u128()
            .expect("value reduced modulo u128 must fit u128")
    }

    fn pow_mod(&self, base: Self::Word, exponent: u128) -> Self::Word {
        let modulus = BigUint::from(self.modulus);

        BigUint::from(base % self.modulus)
            .modpow(&BigUint::from(exponent), &modulus)
            .to_u128()
            .expect("value reduced modulo u128 must fit u128")
    }

    fn inv_mod(&self, value: Self::Word) -> Option<Self::Word> {
        inverse_u128(value % self.modulus, self.modulus)
    }
}

fn inverse_u128(value: u128, modulus: u128) -> Option<u128> {
    if value == 0 || modulus <= 1 {
        return None;
    }

    use num_bigint::BigInt;
    use num_traits::{One, Signed};

    let mut old_r = BigInt::from(value);
    let mut r = BigInt::from(modulus);
    let mut old_s = BigInt::one();
    let mut s = BigInt::zero();

    while !r.is_zero() {
        let quotient = &old_r / &r;

        let next_r = &old_r - &quotient * &r;
        old_r = r;
        r = next_r;

        let next_s = &old_s - &quotient * &s;
        old_s = s;
        s = next_s;
    }

    if old_r != BigInt::one() {
        return None;
    }

    let modulus_big = BigInt::from(modulus);
    let mut result = old_s % &modulus_big;

    if result.is_negative() {
        result += &modulus_big;
    }

    result.to_u128()
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use num_traits::ToPrimitive;

    use super::{Limb128, Limb32, Limb64, PhysicalLimbArithmetic};

    fn oracle_add(lhs: u128, rhs: u128, modulus: u128) -> u128 {
        ((BigUint::from(lhs) + BigUint::from(rhs)) % BigUint::from(modulus))
            .to_u128()
            .unwrap()
    }

    fn oracle_sub(lhs: u128, rhs: u128, modulus: u128) -> u128 {
        let q = BigUint::from(modulus);
        let lhs = BigUint::from(lhs) % &q;
        let rhs = BigUint::from(rhs) % &q;

        if lhs >= rhs {
            (lhs - rhs).to_u128().unwrap()
        } else {
            (q.clone() - (rhs - lhs)).to_u128().unwrap()
        }
    }

    fn oracle_mul(lhs: u128, rhs: u128, modulus: u128) -> u128 {
        ((BigUint::from(lhs) * BigUint::from(rhs)) % BigUint::from(modulus))
            .to_u128()
            .unwrap()
    }

    #[test]
    fn limb32_matches_biguint_oracle() {
        let q = 4_294_967_291_u32;
        let limb = Limb32::new(q);

        let values = [0_u32, 1, 2, q / 2, q - 2, q - 1, 0xDEAD_BEEF_u32];

        for lhs in values {
            for rhs in values {
                assert_eq!(
                    u128::from(limb.add_mod(lhs, rhs)),
                    oracle_add(u128::from(lhs), u128::from(rhs), u128::from(q))
                );
                assert_eq!(
                    u128::from(limb.sub_mod(lhs, rhs)),
                    oracle_sub(u128::from(lhs), u128::from(rhs), u128::from(q))
                );
                assert_eq!(
                    u128::from(limb.mul_mod(lhs, rhs)),
                    oracle_mul(u128::from(lhs), u128::from(rhs), u128::from(q))
                );
            }
        }

        assert_eq!(Limb32::WORD_BITS, 32);
    }

    #[test]
    fn limb64_matches_biguint_oracle() {
        let q = 18_446_744_073_709_551_557_u64;
        let limb = Limb64::new(q);

        let values = [0_u64, 1, 2, q / 2, q - 2, q - 1, 0xDEAD_BEEF_CAFE_BABE_u64];

        for lhs in values {
            for rhs in values {
                assert_eq!(
                    u128::from(limb.add_mod(lhs, rhs)),
                    oracle_add(u128::from(lhs), u128::from(rhs), u128::from(q))
                );
                assert_eq!(
                    u128::from(limb.sub_mod(lhs, rhs)),
                    oracle_sub(u128::from(lhs), u128::from(rhs), u128::from(q))
                );
                assert_eq!(
                    u128::from(limb.mul_mod(lhs, rhs)),
                    oracle_mul(u128::from(lhs), u128::from(rhs), u128::from(q))
                );
            }
        }

        assert_eq!(Limb64::WORD_BITS, 64);
    }

    #[test]
    fn limb128_matches_biguint_oracle() {
        let q = (1_u128 << 127) - 1;
        let limb = Limb128::new(q);

        let values = [
            0_u128,
            1,
            2,
            q / 2,
            q - 2,
            q - 1,
            (1_u128 << 126) + 0x1234_5678_9ABC_DEF0_u128,
        ];

        for lhs in values {
            for rhs in values {
                assert_eq!(limb.add_mod(lhs, rhs), oracle_add(lhs, rhs, q));
                assert_eq!(limb.sub_mod(lhs, rhs), oracle_sub(lhs, rhs, q));
                assert_eq!(limb.mul_mod(lhs, rhs), oracle_mul(lhs, rhs, q));
            }
        }

        assert_eq!(Limb128::WORD_BITS, 128);
    }

    #[test]
    fn pow_and_inverse_match_across_widths() {
        let limb32 = Limb32::new(65_537);
        let limb64 = Limb64::new(4_294_967_291);
        let limb128 = Limb128::new((1_u128 << 127) - 1);

        assert_eq!(limb32.mul_mod(17, limb32.inv_mod(17).unwrap()), 1);

        let x64 = 1_000_003_u64;
        assert_eq!(limb64.mul_mod(x64, limb64.inv_mod(x64).unwrap()), 1);

        let x128 = (1_u128 << 100) + 0x12345;
        assert_eq!(limb128.mul_mod(x128, limb128.inv_mod(x128).unwrap()), 1);

        assert_eq!(
            u128::from(limb32.pow_mod(17, 12345)),
            oracle_mul(
                u128::from(limb32.pow_mod(17, 12344)),
                17,
                u128::from(limb32.modulus())
            )
        );
    }

    #[test]
    fn noninvertible_values_return_none() {
        let limb32 = Limb32::new(15);
        let limb64 = Limb64::new(21);
        let limb128 = Limb128::new(35);

        assert_eq!(limb32.inv_mod(5), None);
        assert_eq!(limb64.inv_mod(7), None);
        assert_eq!(limb128.inv_mod(5), None);
    }
}
