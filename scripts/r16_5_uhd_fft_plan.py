#!/usr/bin/env python3
"""R16.5: exact-UHD FFT structural and packing preflight."""

from collections import Counter
from math import ceil, prod

WIDTH = 3840
HEIGHT = 2160
CKKS_DEGREE = 16384
CKKS_SLOTS = CKKS_DEGREE // 2

def factorize(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def describe_axis(name, length):
    factors = factorize(length)
    assert prod(factors) == length
    assert set(factors) <= {2, 3, 5}

    print(f"R16_5_AXIS={name}")
    print(f"R16_5_{name}_LENGTH={length}")
    print(f"R16_5_{name}_FACTORS={','.join(map(str, factors))}")
    print(f"R16_5_{name}_RADIX_COUNTS={dict(sorted(Counter(factors).items()))}")
    print(f"R16_5_{name}_STAGE_COUNT={len(factors)}")
    return factors

def main():
    pixels = WIDTH * HEIGHT
    row_factors = describe_axis("ROWS", WIDTH)
    column_factors = describe_axis("COLUMNS", HEIGHT)

    assert pixels == 8_294_400
    assert prod(row_factors) == WIDTH
    assert prod(column_factors) == HEIGHT

    minimum_ciphertexts = ceil(pixels / CKKS_SLOTS)

    print(f"R16_5_UHD_WIDTH={WIDTH}")
    print(f"R16_5_UHD_HEIGHT={HEIGHT}")
    print(f"R16_5_UHD_PIXELS={pixels}")
    print(f"R16_5_CKKS_DEGREE={CKKS_DEGREE}")
    print(f"R16_5_CKKS_SLOTS={CKKS_SLOTS}")
    print(f"R16_5_MINIMUM_CIPHERTEXTS={minimum_ciphertexts}")

    # A conservative, implementation-independent structural count.
    # NOT a CKKS multiplicative-depth estimate.
    total_stages = len(row_factors) + len(column_factors)
    print(f"R16_5_MIXED_RADIX_STAGES={total_stages}")
    print("R16_5_CKKS_DEPTH=UNDETERMINED")
    print("R16_5_ROTATION_KEY_SET=UNDETERMINED")
    print("R16_5_CROSS_CIPHERTEXT_COMMUNICATION=REQUIRED")
    print("R16_5_SECURITY_STATUS=REQUIRES_ANALYSIS")

    # Two complete representations are a useful initial memory model,
    # not a proven lower bound on the implementation's peak memory.
    print(f"R16_5_TWO_BUFFER_CIPHERTEXT_COUNT={2 * minimum_ciphertexts}")
    print("R16_5_UHD_STRUCTURAL_PREFLIGHT=PASS")

if __name__ == "__main__":
    main()
