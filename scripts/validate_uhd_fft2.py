#!/usr/bin/env python3
"""Validate a native 3840x2160 encrypted FFT2 spectrum."""

import argparse
import numpy as np
from pathlib import Path

WIDTH = 3840
HEIGHT = 2160
PIXELS = WIDTH * HEIGHT


def read_pgm(path):
    data = Path(path).read_bytes()

    # The production image reader uses binary P5 with 8-bit pixels.
    import re

    match = re.match(
        rb"P5\s+(?:\#.*\n\s*)*(\d+)\s+"
        rb"(?:\#.*\n\s*)*(\d+)\s+"
        rb"(?:\#.*\n\s*)*(\d+)\s",
        data,
    )
    if match is None:
        raise ValueError("Unsupported PGM header")

    width, height, maximum = map(int, match.groups())
    assert (width, height) == (WIDTH, HEIGHT)
    assert maximum == 255

    payload = data[match.end():]
    assert len(payload) == PIXELS, (
        f"Expected {PIXELS} pixels, found {len(payload)}"
    )

    return np.frombuffer(payload, dtype=np.uint8).reshape(
        HEIGHT, WIDTH
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", default="data/uhd_fft_input.pgm"
    )
    parser.add_argument(
        "--spectrum",
        default="data/uhd_encrypted_fft2_spectrum.bin",
    )
    args = parser.parse_args()

    image = read_pgm(args.image)

    # Match the Rust UHD image scaling.
    signal = 0.001 * (
        image.astype(np.float64) / 255.0 - 0.5
    )

    spectrum_path = Path(args.spectrum)
    expected_bytes = PIXELS * 16

    assert spectrum_path.is_file(), (
        f"Spectrum file missing: {spectrum_path}"
    )
    assert spectrum_path.stat().st_size == expected_bytes, (
        f"Unexpected spectrum size: "
        f"{spectrum_path.stat().st_size}"
    )

    encrypted = np.memmap(
        spectrum_path,
        dtype="<c16",
        mode="r",
        shape=(HEIGHT, WIDTH),
    )

    reference = np.fft.fft2(signal)

    difference = encrypted - reference

    reference_norm = np.linalg.norm(reference.ravel())
    error_norm = np.linalg.norm(difference.ravel())

    relative_l2 = error_norm / reference_norm
    max_abs = np.max(np.abs(difference))
    rms = np.sqrt(np.mean(np.abs(difference) ** 2))

    print(f"R37_3_WIDTH={WIDTH}")
    print(f"R37_3_HEIGHT={HEIGHT}")
    print(f"R37_3_SPECTRAL_VALUES={PIXELS}")
    print(f"R37_3_REFERENCE_NORM={reference_norm:.12e}")
    print(f"R37_3_ERROR_NORM={error_norm:.12e}")
    print(f"R37_3_RELATIVE_L2={relative_l2:.12e}")
    print(f"R37_3_MAX_ABS={max_abs:.12e}")
    print(f"R37_3_RMS_ERROR={rms:.12e}")

    assert np.isfinite(relative_l2)
    assert np.isfinite(max_abs)

    print("R37_3_VALIDATION_STATUS=MEASURED")


if __name__ == "__main__":
    main()
