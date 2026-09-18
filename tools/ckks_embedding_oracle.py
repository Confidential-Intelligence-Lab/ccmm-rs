#!/usr/bin/env python3

import cmath
import math
import random


def roots(degree):
    return [
        cmath.exp(
            1j
            * math.pi
            * (2 * index + 1)
            / degree
        )
        for index in range(degree)
    ]


def expand_slots(slots):
    degree = 2 * len(slots)
    values = [0j] * degree

    for index, slot in enumerate(slots):
        values[index] = slot
        values[degree - 1 - index] = slot.conjugate()

    return values


def slots_to_coefficients(slots):
    degree = 2 * len(slots)
    values = expand_slots(slots)
    xi = roots(degree)

    coefficients = []

    for k in range(degree):
        value = sum(
            values[j]
            * xi[j].conjugate() ** k
            for j in range(degree)
        ) / degree

        if abs(value.imag) > 1.0e-10 * (1.0 + abs(value.real)):
            raise AssertionError(
                f"non-real coefficient at {k}: {value}"
            )

        coefficients.append(value.real)

    return coefficients


def coefficients_to_slots(coefficients):
    degree = len(coefficients)
    xi = roots(degree)

    out = []

    for j in range(degree // 2):
        root = xi[j]
        value = 0j

        for coefficient in reversed(coefficients):
            value = value * root + coefficient

        out.append(value)

    return out


def quantize(coefficients, scale):
    return [
        round(value * scale) / scale
        for value in coefficients
    ]


def maximum_error(lhs, rhs):
    return max(
        abs(a - b)
        for a, b in zip(lhs, rhs)
    )


def main():
    print("CKKS_EMBEDDING_ORACLE_VERSION=1")

    for degree in [8, 16, 32]:
        for scale in [
            4096,
            65536,
            1048576,
        ]:
            tolerance = (
                degree
                / (2.0 * scale)
                + 1.0e-10
            )

            worst = 0.0

            for seed in range(32):
                rng = random.Random(
                    seed
                    ^ (degree << 16)
                    ^ scale
                )

                slots = [
                    complex(
                        rng.uniform(-2.0, 2.0),
                        rng.uniform(-2.0, 2.0),
                    )
                    for _ in range(degree // 2)
                ]

                coefficients = slots_to_coefficients(slots)
                quantized = quantize(coefficients, scale)
                recovered = coefficients_to_slots(quantized)

                error = maximum_error(
                    recovered,
                    slots,
                )

                worst = max(
                    worst,
                    error,
                )

                if error > tolerance:
                    raise AssertionError(
                        f"degree={degree}, "
                        f"scale={scale}, "
                        f"seed={seed}, "
                        f"error={error}, "
                        f"tolerance={tolerance}"
                    )

            print(
                f"DEGREE={degree},"
                f"SCALE={scale},"
                f"MAX_ERROR={worst:.17g},"
                "STATUS=PASS"
            )

    print("CKKS_EMBEDDING_ORACLE_STATUS=PASS")


if __name__ == "__main__":
    main()
