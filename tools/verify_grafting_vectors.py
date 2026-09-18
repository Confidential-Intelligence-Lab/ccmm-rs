#!/usr/bin/env python3

from fractions import Fraction
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent


def run_rust():
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "grafting_vectors",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    return result.stdout


def parse_transcript(text):
    data = {}

    for line in text.splitlines():
        line = line.strip()

        if not line or "=" not in line:
            continue

        key, value = line.split("=", 1)
        data[key] = value

    return data


def parse_fraction_vector(text):
    return [
        Fraction(value)
        for value in text.split(",")
    ]


def parse_float_vector(text):
    return [
        float(value)
        for value in text.split(",")
    ]


def negacyclic_mul(lhs, rhs):
    assert len(lhs) == len(rhs)

    degree = len(lhs)
    output = [
        Fraction(0)
        for _ in range(degree)
    ]

    for i in range(degree):
        for j in range(degree):
            term = lhs[i] * rhs[j]

            if i + j < degree:
                output[i + j] += term
            else:
                output[i + j - degree] -= term

    return output


def check_vector(
    name,
    actual,
    expected,
    tolerance,
):
    if len(actual) != len(expected):
        raise AssertionError(
            f"{name}: length mismatch "
            f"{len(actual)} != {len(expected)}"
        )

    maximum_error = 0.0

    for index, (observed, reference) in enumerate(
        zip(actual, expected)
    ):
        reference_float = float(reference)

        error = abs(
            observed - reference_float
        )

        maximum_error = max(
            maximum_error,
            error,
        )

        if error > tolerance:
            raise AssertionError(
                f"{name}[{index}]: "
                f"actual={observed}, "
                f"expected={reference_float}, "
                f"error={error}, "
                f"tolerance={tolerance}"
            )

    print(
        f"{name}_MAX_ERROR="
        f"{maximum_error:.17g}"
    )


def main():
    transcript = run_rust()

    print(transcript, end="")

    data = parse_transcript(
        transcript,
    )

    required = [
        "GRAFTING_VECTOR_VERSION",
        "DEGREE",
        "LEVEL0_MODULUS",
        "LEVEL1_MODULUS",
        "LEVEL2_MODULUS",
        "LEVEL0_SCALE",
        "LEVEL1_SCALE",
        "LEVEL1_OPERAND_SCALE",
        "LEVEL2_SCALE",
        "INPUT_LHS",
        "INPUT_RHS",
        "INPUT_THIRD",
        "LEVEL1_DECODED",
        "LEVEL2_DECODED",
        "GRAFTING_VECTOR_STATUS",
    ]

    for key in required:
        if key not in data:
            raise AssertionError(
                f"missing transcript field: {key}"
            )

    assert data["GRAFTING_VECTOR_VERSION"] == "1"
    assert data["GRAFTING_VECTOR_STATUS"] == "PASS"

    degree = int(data["DEGREE"])
    assert degree == 8

    q0 = 12_289
    q1 = 40_961
    q2 = 65_537

    expected_q0 = q0 * q1 * q2
    expected_q1 = q0 * q1
    expected_q2 = q0

    assert int(data["LEVEL0_MODULUS"]) == expected_q0
    assert int(data["LEVEL1_MODULUS"]) == expected_q1
    assert int(data["LEVEL2_MODULUS"]) == expected_q2

    level0_scale = Fraction(
        data["LEVEL0_SCALE"]
    )

    level1_scale = Fraction(
        data["LEVEL1_SCALE"]
    )

    level1_operand_scale = Fraction(
        data["LEVEL1_OPERAND_SCALE"]
    )

    level2_scale = Fraction(
        data["LEVEL2_SCALE"]
    )

    expected_level1_scale = (
        level0_scale
        * level0_scale
        / q2
    )

    expected_level2_scale = (
        expected_level1_scale
        * level1_operand_scale
        / q1
    )

    assert level1_scale == expected_level1_scale
    assert level2_scale == expected_level2_scale

    lhs = parse_fraction_vector(
        data["INPUT_LHS"]
    )

    rhs = parse_fraction_vector(
        data["INPUT_RHS"]
    )

    third = parse_fraction_vector(
        data["INPUT_THIRD"]
    )

    assert len(lhs) == degree
    assert len(rhs) == degree
    assert len(third) == degree

    level1_reference = negacyclic_mul(
        lhs,
        rhs,
    )

    level2_reference = negacyclic_mul(
        level1_reference,
        third,
    )

    level1_actual = parse_float_vector(
        data["LEVEL1_DECODED"]
    )

    level2_actual = parse_float_vector(
        data["LEVEL2_DECODED"]
    )

    check_vector(
        "LEVEL1_DIFF",
        level1_actual,
        level1_reference,
        0.001,
    )

    check_vector(
        "LEVEL2_DIFF",
        level2_actual,
        level2_reference,
        0.002,
    )

    print(
        "GRAFTING_RUST_PYTHON_DIFF_STATUS=PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(
            "GRAFTING_RUST_PYTHON_DIFF_STATUS=FAIL",
            file=sys.stderr,
        )
        raise
