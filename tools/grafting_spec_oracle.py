#!/usr/bin/env python3

from fractions import Fraction


def centered(value: int, modulus: int) -> int:
    value %= modulus
    return value - modulus if value > modulus // 2 else value


def round_div_nearest(value: int, divisor: int) -> int:
    assert divisor > 0

    if value >= 0:
        return (value + divisor // 2) // divisor

    return -((-value + divisor // 2) // divisor)


def ckks_rescale(
    value: int,
    source_modulus: int,
    divisor: int,
    target_modulus: int,
) -> int:
    value_centered = centered(
        value,
        source_modulus,
    )

    quotient = round_div_nearest(
        value_centered,
        divisor,
    )

    return quotient % target_modulus


def negacyclic_mul(lhs, rhs):
    assert len(lhs) == len(rhs)

    degree = len(lhs)
    output = [Fraction(0) for _ in range(degree)]

    for i in range(degree):
        for j in range(degree):
            product = lhs[i] * rhs[j]

            if i + j < degree:
                output[i + j] += product
            else:
                output[i + j - degree] -= product

    return output


def main():
    q0 = 12_289
    q1 = 40_961
    q2 = 65_537

    Q0 = q0 * q1 * q2
    Q1 = q0 * q1
    Q2 = q0

    print("GRAFTING_SPEC_ORACLE_VERSION=1")

    # --------------------------------------------------------
    # Gate 1: centered CKKS rescale
    # --------------------------------------------------------

    vectors = [
        0,
        1,
        q2 // 2,
        q2 // 2 + 1,
        q2 - 1,
        Q0 - 1,
    ]

    rescaled = [
        ckks_rescale(
            value,
            Q0,
            q2,
            Q1,
        )
        for value in vectors
    ]

    print(
        "RESCALE_LEVEL0_VALUES="
        + ",".join(map(str, rescaled))
    )

    # Q0 - 1 represents -1.
    assert rescaled[-1] == 0

    # --------------------------------------------------------
    # Gate 2: modulus-chain structure
    # --------------------------------------------------------

    assert Q0 == q0 * q1 * q2
    assert Q1 == q0 * q1
    assert Q2 == q0

    print(f"LEVEL0_MODULUS={Q0}")
    print(f"LEVEL1_MODULUS={Q1}")
    print(f"LEVEL2_MODULUS={Q2}")

    # --------------------------------------------------------
    # Gate 3: scale evolution
    # --------------------------------------------------------

    level0_scale = Fraction(q2)

    level1_scale = (
        level0_scale
        * level0_scale
        / q2
    )

    level1_operand_scale = Fraction(q1)

    level2_scale = (
        level1_scale
        * level1_operand_scale
        / q1
    )

    assert level1_scale == q2
    assert level2_scale == q2

    print(f"LEVEL0_SCALE={level0_scale}")
    print(f"LEVEL1_SCALE={level1_scale}")
    print(f"LEVEL2_SCALE={level2_scale}")

    # --------------------------------------------------------
    # Gate 4: depth-2 plaintext semantics
    # --------------------------------------------------------

    lhs = [
        Fraction(1, 8),
        Fraction(-1, 16),
        Fraction(1, 32),
        0,
        0,
        0,
        0,
        0,
    ]

    rhs = [
        Fraction(1, 16),
        Fraction(1, 32),
        Fraction(-1, 16),
        0,
        0,
        0,
        0,
        0,
    ]

    third = [
        Fraction(1, 8),
        Fraction(-1, 16),
        Fraction(1, 32),
        0,
        0,
        0,
        0,
        0,
    ]

    level1_reference = negacyclic_mul(
        lhs,
        rhs,
    )

    level2_reference = negacyclic_mul(
        level1_reference,
        third,
    )

    print(
        "DEPTH2_REFERENCE="
        + ",".join(
            f"{float(value):.17g}"
            for value in level2_reference
        )
    )

    print("GRAFTING_SPEC_ORACLE_STATUS=PASS")


if __name__ == "__main__":
    main()
