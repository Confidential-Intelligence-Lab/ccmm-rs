#!/usr/bin/env python3

import importlib.util
import inspect
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "external" / "grafting-reference" / "ckks.py"


def load_reference():
    if not REFERENCE.exists():
        raise RuntimeError(
            f"external Grafting reference not found: {REFERENCE}"
        )

    spec = importlib.util.spec_from_file_location(
        "external_grafting_ckks",
        REFERENCE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("could not load external Grafting module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def negacyclic_mul(lhs, rhs):
    assert len(lhs) == len(rhs)

    n = len(lhs)
    out = [0.0] * n

    for i in range(n):
        for j in range(n):
            term = lhs[i] * rhs[j]

            if i + j < n:
                out[i + j] += term
            else:
                out[i + j - n] -= term

    return out


def max_error(actual, expected):
    assert len(actual) == len(expected)

    return max(
        abs(a - e)
        for a, e in zip(actual, expected)
    )


def call_encrypt(scheme, message):
    sig = inspect.signature(scheme.encrypt)

    params = list(sig.parameters.values())

    print(f"EXTERNAL_ENCRYPT_SIGNATURE={sig}")

    # Bound method: self is already removed.
    #
    # Prefer the implementation default whenever available.
    try:
        return scheme.encrypt(message)
    except TypeError:
        pass

    # If an explicit level is required, use all unit primes.
    level = len(scheme.unit_primes)

    return scheme.encrypt(
        message,
        level,
    )


def call_decrypt(scheme, ciphertext, count):
    sig = inspect.signature(scheme.decrypt)

    print(f"EXTERNAL_DECRYPT_SIGNATURE={sig}")

    try:
        return scheme.decrypt(
            ciphertext,
            count,
        )
    except TypeError:
        return scheme.decrypt(
            ciphertext,
        )[:count]


def main():
    module = load_reference()

    print("EXTERNAL_GRAFTING_CROSSCHECK_VERSION=1")
    print(f"REFERENCE_FILE={REFERENCE}")

    if not hasattr(module, "GraftingCKKS"):
        raise AssertionError(
            "external implementation has no GraftingCKKS class"
        )

    # Match the external implementation's own standalone
    # validation harness. Its main() seeds Python's global RNG
    # with 42 before constructing GraftingCKKS.
    module.random.seed(42)

    print("EXTERNAL_RANDOM_SEED=42")

    scheme = module.GraftingCKKS(
        N=128,
        unit_bits=18,
        num_units=4,
        delta_bits=12,
        dnum=2,
    )

    print(
        "EXTERNAL_MULTIPLY_SIGNATURE="
        f"{inspect.signature(scheme.multiply)}"
    )

    # --------------------------------------------------------
    # Gate 1: coefficient encoder semantics
    # --------------------------------------------------------

    coefficient_input = [
        0.5,
        -0.25,
        0.125,
        0.0,
    ]

    encoded = scheme._encode(
        coefficient_input
    )

    delta = scheme.delta

    expected_encoded = [
        round(value * delta)
        for value in coefficient_input
    ]

    assert (
        encoded[:len(expected_encoded)]
        == expected_encoded
    ), (
        "external coefficient encoder does not match "
        "round(m_i * Delta)"
    )

    print("EXTERNAL_COEFFICIENT_ENCODING=PASS")

    # --------------------------------------------------------
    # Gate 2: plaintext ring semantics
    #
    # Use the same semantics as our current coefficient-mode
    # implementation: polynomial multiplication is negacyclic.
    # --------------------------------------------------------

    lhs = [
        0.5,
        0.0,
        0.0,
        0.0,
    ]

    rhs = [
        0.6,
        0.0,
        0.0,
        0.0,
    ]

    expected = negacyclic_mul(
        lhs,
        rhs,
    )

    assert math.isclose(
        expected[0],
        0.3,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )

    print("EXTERNAL_PLAINTEXT_RING_REFERENCE=PASS")

    # --------------------------------------------------------
    # Gate 3: encrypt/decrypt independently
    # --------------------------------------------------------

    ct_lhs = call_encrypt(
        scheme,
        lhs,
    )

    ct_rhs = call_encrypt(
        scheme,
        rhs,
    )

    dec_lhs = call_decrypt(
        scheme,
        ct_lhs,
        len(lhs),
    )

    dec_rhs = call_decrypt(
        scheme,
        ct_rhs,
        len(rhs),
    )

    lhs_error = max_error(
        dec_lhs,
        lhs,
    )

    rhs_error = max_error(
        dec_rhs,
        rhs,
    )

    print(
        f"EXTERNAL_LHS_ROUNDTRIP_MAX_ERROR={lhs_error:.17g}"
    )

    print(
        f"EXTERNAL_RHS_ROUNDTRIP_MAX_ERROR={rhs_error:.17g}"
    )

    # Demo precision is intentionally small.
    assert lhs_error < 0.01
    assert rhs_error < 0.01

    print("EXTERNAL_ENCRYPT_DECRYPT=PASS")

    # --------------------------------------------------------
    # Gate 4: external encrypted multiply + Grafting rescale
    # --------------------------------------------------------

    product = scheme.multiply(
        ct_lhs,
        ct_rhs,
    )

    decoded = call_decrypt(
        scheme,
        product,
        len(expected),
    )

    product_error = max_error(
        decoded,
        expected,
    )

    print(
        "EXTERNAL_PRODUCT_DECODED="
        + ",".join(
            f"{value:.17g}"
            for value in decoded
        )
    )

    print(
        "EXTERNAL_PRODUCT_EXPECTED="
        + ",".join(
            f"{value:.17g}"
            for value in expected
        )
    )

    print(
        f"EXTERNAL_PRODUCT_MAX_ERROR={product_error:.17g}"
    )

    # The repository's own example reports ~2.36e-2 maximum
    # error for this exact 0.5 * 0.6 experiment.
    assert product_error < 0.04

    print("EXTERNAL_GRAFTING_MULTIPLY=PASS")

    print("EXTERNAL_GRAFTING_CROSSCHECK_STATUS=PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(
            "EXTERNAL_GRAFTING_CROSSCHECK_STATUS=FAIL",
            file=sys.stderr,
        )
        raise
