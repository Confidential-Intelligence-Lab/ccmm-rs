use num_bigint::BigUint;
use num_traits::One;

/// Deterministic Miller--Rabin probable-prime test for the parameter-generation
/// layer. For values below 2^64, the standard deterministic witness set is
/// used. Wider u128 candidates use a fixed, reproducible witness schedule and
/// are therefore reported as probable primes rather than proven primes.
pub fn is_ntt_prime_candidate(value: u128) -> bool {
    if value < 2 {
        return false;
    }

    const SMALL_PRIMES: [u128; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    for prime in SMALL_PRIMES {
        if value == prime {
            return true;
        }
        if value % prime == 0 {
            return false;
        }
    }

    let (s, d) = factor_twos(value - 1);

    if value <= u64::MAX as u128 {
        // Deterministic for every unsigned 64-bit integer.
        const WITNESSES_64: [u128; 7] = [2, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022];

        return WITNESSES_64
            .iter()
            .copied()
            .all(|witness| miller_rabin_round(value, s, d, witness));
    }

    // Reproducible high-confidence probable-prime screen for u128 candidates.
    // This is deliberately not labeled a deterministic primality proof.
    const WITNESSES_128: [u128; 20] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    ];

    WITNESSES_128
        .iter()
        .copied()
        .all(|witness| miller_rabin_round(value, s, d, witness))
}

pub fn is_ntt_compatible(modulus: u128, degree: usize) -> bool {
    if degree < 2 || !degree.is_power_of_two() || modulus <= 2 {
        return false;
    }

    let two_n = match (degree as u128).checked_mul(2) {
        Some(value) => value,
        None => return false,
    };

    (modulus - 1) % two_n == 0
}

/// Search downward for q = k * (2N) + 1 below the requested bit width.
///
/// For widths <= 64, primality is deterministic under the Miller--Rabin
/// witness set used above. Wider results are reproducible probable primes.
pub fn find_ntt_prime_below_bits(bits: u32, degree: usize) -> Option<u128> {
    if !(2..=128).contains(&bits) || degree < 2 || !degree.is_power_of_two() {
        return None;
    }

    let two_n = (degree as u128).checked_mul(2)?;

    let upper_exclusive = if bits == 128 {
        u128::MAX
    } else {
        1_u128.checked_shl(bits)?
    };

    let mut candidate = upper_exclusive - 1;
    candidate -= (candidate - 1) % two_n;

    loop {
        if candidate > 2
            && is_ntt_compatible(candidate, degree)
            && is_ntt_prime_candidate(candidate)
        {
            return Some(candidate);
        }

        if candidate <= two_n + 1 {
            return None;
        }
        candidate -= two_n;
    }
}

fn factor_twos(mut value: u128) -> (u32, u128) {
    let mut s = 0_u32;
    while value & 1 == 0 {
        value >>= 1;
        s += 1;
    }
    (s, value)
}

fn miller_rabin_round(value: u128, s: u32, d: u128, witness: u128) -> bool {
    let witness = witness % value;
    if witness == 0 {
        return true;
    }

    let modulus = BigUint::from(value);
    let mut x = BigUint::from(witness).modpow(&BigUint::from(d), &modulus);

    if x == BigUint::one() || x == &modulus - BigUint::one() {
        return true;
    }

    for _ in 1..s {
        x = (&x * &x) % &modulus;
        if x == &modulus - BigUint::one() {
            return true;
        }
        if x.is_one() {
            return false;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::{find_ntt_prime_below_bits, is_ntt_compatible, is_ntt_prime_candidate};

    #[test]
    fn rejects_known_composites_and_accepts_known_primes() {
        for composite in [0_u128, 1, 4, 9, 15, 21, 25, 341, 561, 1_105] {
            assert!(!is_ntt_prime_candidate(composite));
        }

        for prime in [
            2_u128,
            3,
            5,
            65_537,
            2_013_265_921,
            18_446_744_069_414_584_321,
        ] {
            assert!(is_ntt_prime_candidate(prime));
        }
    }

    #[test]
    fn compatibility_checks_two_n_divisibility() {
        assert!(is_ntt_compatible(65_537, 256));
        assert!(is_ntt_compatible(2_013_265_921, 4096));
        assert!(!is_ntt_compatible(65_537, 65_536));
        assert!(!is_ntt_compatible(65_537, 3));
    }

    #[test]
    fn discovers_native_32_bit_ntt_prime() {
        let degree = 4096;
        let q = find_ntt_prime_below_bits(32, degree).expect("32-bit NTT prime should exist");
        assert!(q <= u32::MAX as u128);
        assert!(q >= (1_u128 << 31));
        assert!(is_ntt_compatible(q, degree));
        assert!(is_ntt_prime_candidate(q));
    }

    #[test]
    fn discovers_native_64_bit_ntt_prime() {
        let degree = 4096;
        let q = find_ntt_prime_below_bits(64, degree).expect("64-bit NTT prime should exist");
        assert!(q <= u64::MAX as u128);
        assert!(q >= (1_u128 << 63));
        assert!(is_ntt_compatible(q, degree));
        assert!(is_ntt_prime_candidate(q));
    }

    #[test]
    fn discovers_genuine_wide_u128_ntt_probable_prime() {
        let degree = 4096;
        let q = find_ntt_prime_below_bits(120, degree)
            .expect("120-bit NTT probable prime should exist");
        assert!(q > u64::MAX as u128);
        assert!(q >= (1_u128 << 119));
        assert!(is_ntt_compatible(q, degree));
        assert!(is_ntt_prime_candidate(q));
    }
}
