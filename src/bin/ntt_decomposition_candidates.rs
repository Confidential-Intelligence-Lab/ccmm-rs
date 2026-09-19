use ccmm_rs::ring::{
    is_ntt_compatible, is_ntt_prime_candidate, CANONICAL_NTT_DEGREE, NTT128_MODULUS,
};
use num_bigint::BigUint;
use num_traits::One;

fn find_distinct_primes(bits: u32, count: usize, degree: usize) -> Vec<u128> {
    assert!((2..=128).contains(&bits));
    assert!(count > 0);
    assert!(degree >= 2 && degree.is_power_of_two());

    let two_n = 2_u128
        .checked_mul(degree as u128)
        .expect("degree too large");

    let upper_exclusive = if bits == 128 {
        u128::MAX
    } else {
        1_u128 << bits
    };

    let lower_inclusive = 1_u128 << (bits - 1);

    let mut candidate = upper_exclusive - 1;
    candidate -= (candidate - 1) % two_n;

    let mut primes = Vec::with_capacity(count);

    while candidate >= lower_inclusive && primes.len() < count {
        if is_ntt_compatible(candidate, degree) && is_ntt_prime_candidate(candidate) {
            primes.push(candidate);
        }

        if candidate <= two_n + 1 {
            break;
        }

        candidate -= two_n;
    }

    assert_eq!(
        primes.len(),
        count,
        "failed to find requested number of NTT primes"
    );

    primes
}

fn product_big(values: &[u128]) -> BigUint {
    values
        .iter()
        .fold(BigUint::one(), |acc, &value| acc * BigUint::from(value))
}

fn print_basis(name: &str, word_width: u32, primes: &[u128]) {
    let composite = product_big(primes);

    println!("BASIS={name}");
    println!("PHYSICAL_WORD_BITS={word_width}");
    println!("PHYSICAL_LIMBS={}", primes.len());
    println!(
        "PRIMES={}",
        primes
            .iter()
            .map(u128::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "PRIME_BITS={}",
        primes
            .iter()
            .map(|q| (128 - q.leading_zeros()).to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("COMPOSITE_MODULUS={composite}");
    println!("COMPOSITE_MODULUS_BITS={}", composite.bits());

    for &prime in primes {
        assert!(is_ntt_compatible(prime, CANONICAL_NTT_DEGREE));
        assert!(is_ntt_prime_candidate(prime));
    }

    println!("NTT_COMPATIBILITY=PASS");
    println!("REPOSITORY_PRIMALITY_SCREEN=PASS");
    println!();
}

fn main() {
    let basis32 = find_distinct_primes(30, 4, CANONICAL_NTT_DEGREE);
    let basis64 = find_distinct_primes(60, 2, CANONICAL_NTT_DEGREE);
    let basis128 = vec![NTT128_MODULUS];

    println!("R3_3D_1A_DECOMPOSITION_CANDIDATES_VERSION=1");
    println!("DEGREE={CANONICAL_NTT_DEGREE}");
    println!("TARGET_LOGICAL_MODULUS_BITS=120");
    println!();

    print_basis("W32_X4", 32, &basis32);
    print_basis("W64_X2", 64, &basis64);
    print_basis("W128_X1", 128, &basis128);

    let bits32 = product_big(&basis32).bits();
    let bits64 = product_big(&basis64).bits();
    let bits128 = product_big(&basis128).bits();

    let min_bits = bits32.min(bits64).min(bits128);
    let max_bits = bits32.max(bits64).max(bits128);

    println!("MIN_COMPOSITE_BITS={min_bits}");
    println!("MAX_COMPOSITE_BITS={max_bits}");
    println!("COMPOSITE_BIT_SPREAD={}", max_bits - min_bits);

    assert!(
        max_bits - min_bits <= 2,
        "candidate decompositions must stay within two bits of one another"
    );

    println!("EQUAL_CAPACITY_GATE=PASS");
    println!("R3_3D_1A_STATUS=PASS");
}
