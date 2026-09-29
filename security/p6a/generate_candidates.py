#!/usr/bin/env python3

"""Generate reproducible NTT-compatible candidate CKKS research chains.

These are parameter candidates, not security-bearing profiles.
Primality testing below 2^64 mirrors ccmm-rs/src/ring/ntt_prime.rs.
"""

from math import prod, log2


WITNESSES_64 = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def is_prime_u64(n: int) -> bool:
    if n < 2:
        return False

    for p in SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while d & 1 == 0:
        d >>= 1
        s += 1

    for witness in WITNESSES_64:
        a = witness % n
        if a == 0:
            continue

        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue

        for _ in range(1, s):
            x = (x * x) % n
            if x == n - 1:
                break
            if x == 1:
                return False
        else:
            return False

    return True


def primes_below_bits(bits: int, degree: int, count: int):
    two_n = 2 * degree
    upper = 1 << bits

    candidate = upper - 1
    candidate -= (candidate - 1) % two_n

    result = []

    while candidate > two_n + 1 and len(result) < count:
        if is_prime_u64(candidate):
            assert (candidate - 1) % two_n == 0
            result.append(candidate)
        candidate -= two_n

    if len(result) != count:
        raise RuntimeError(
            f"could not find {count} primes: N={degree}, bits={bits}"
        )

    return result


# Candidate design points.
#
# Continue the existing research-profile depth progression while staying
# comfortably within native u64 limbs. These are research-profile candidates;
# security claims remain separate from mechanical parameter generation.
#
# Existing:
#   4096  : 3 x ~35
#   8192  : 5 x ~40
#   16384 : 9 x ~45
#
# Candidates:
#   32768 : 13 x ~50
#   65536 : 17 x ~55
#
# The counts are engineering starting points, not security claims.
CANDIDATES = [
    ("research-32768-candidate", 32768, 50, 13, 50),
    ("research-65536-candidate", 65536, 55, 17, 55),
]


for name, degree, limb_bits, limb_count, scale_bits in CANDIDATES:
    moduli = primes_below_bits(limb_bits, degree, limb_count)
    q = prod(moduli)

    print("=" * 80)
    print(f"PROFILE={name}")
    print(f"N={degree}")
    print(f"SLOTS={degree // 2}")
    print(f"SCALE_BITS={scale_bits}")
    print(f"LIMB_COUNT={len(moduli)}")
    print(f"LIMB_BITS={','.join(str(x.bit_length()) for x in moduli)}")
    print(f"Q={q}")
    print(f"Q_BIT_LENGTH={q.bit_length()}")
    print(f"LOG2_Q={log2(q):.15f}")
    print("MODULI_DECIMAL=" + ",".join(str(x) for x in moduli))
    print("MODULI_HEX=" + ",".join(hex(x) for x in moduli))
    print("NTT_COMPATIBLE=YES")
    print("SECURITY_VALIDATED=NO")
    print()
