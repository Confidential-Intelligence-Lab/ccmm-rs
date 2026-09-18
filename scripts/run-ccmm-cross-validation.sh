#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NTT_RESILIENCE="${NTT_RESILIENCE:-$(cd "$ROOT/../ntt-resilience" 2>/dev/null && pwd || true)}"

if [[ -z "$NTT_RESILIENCE" || ! -d "$NTT_RESILIENCE" ]]; then
    echo "ERROR: ntt-resilience repository not found."
    echo "Set NTT_RESILIENCE=/absolute/path/to/ntt-resilience"
    exit 1
fi

HEAAN_GATE="$NTT_RESILIENCE/scripts/run-heaan-ccmm-oracle.sh"

if [[ ! -x "$HEAAN_GATE" ]]; then
    echo "ERROR: HEaaN CCMM oracle gate not found or not executable:"
    echo "  $HEAAN_GATE"
    exit 1
fi

rust_output="$(mktemp)"
heaan_output="$(mktemp)"

cleanup() {
    rm -f "$rust_output" "$heaan_output"
}
trap cleanup EXIT

echo "===== RUNNING NATIVE RUST CCMM ORACLE ====="
(
    cd "$ROOT"
    cargo run --release --bin ccmm_oracle
) | tee "$rust_output"

grep -q '^RUST_CCMM_STATUS=PASS$' "$rust_output" || {
    echo "ERROR: native Rust CCMM oracle failed"
    exit 1
}

echo
echo "===== RUNNING HEAaN CCMM ORACLE ====="
(
    cd "$NTT_RESILIENCE"
    "$HEAAN_GATE"
) | tee "$heaan_output"

grep -q '^HEAAN_CCMM_REFERENCE_GATE=PASS$' "$heaan_output" || {
    echo "ERROR: HEaaN CCMM reference gate failed"
    exit 1
}

python3 - "$rust_output" "$heaan_output" <<'PY'
from pathlib import Path
import math
import sys

rust_path = Path(sys.argv[1])
heaan_path = Path(sys.argv[2])

def parse(path):
    values = {}

    for raw in path.read_text().splitlines():
        line = raw.strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    return values

rust = parse(rust_path)
heaan = parse(heaan_path)

coordinates = ("00", "01", "10", "11")

cross_tolerance = 1.0e-3

rust_actual = {}
heaan_actual = {}

for coordinate in coordinates:
    rust_key = f"RUST_CCMM_ACTUAL_{coordinate}"
    heaan_key = f"ACTUAL_{coordinate}"

    if rust_key not in rust:
        raise SystemExit(
            f"ERROR: missing native Rust field {rust_key}"
        )

    if heaan_key not in heaan:
        raise SystemExit(
            f"ERROR: missing HEaaN field {heaan_key}"
        )

    rust_actual[coordinate] = float(rust[rust_key])
    heaan_actual[coordinate] = float(heaan[heaan_key])

rust_error = float(rust["RUST_CCMM_MAX_ABS_ERROR"])
rust_tolerance = float(rust["RUST_CCMM_TOLERANCE"])

heaan_error = float(heaan["HEAAN_CCMM_MAX_ABS_ERROR"])

if not math.isfinite(rust_error):
    raise SystemExit("ERROR: Rust error is not finite")

if not math.isfinite(heaan_error):
    raise SystemExit("ERROR: HEaaN error is not finite")

if rust_error > rust_tolerance:
    raise SystemExit(
        "ERROR: Rust oracle exceeds its declared tolerance: "
        f"{rust_error} > {rust_tolerance}"
    )

# Existing HEaaN oracle uses 1e-3 as its semantic acceptance bound.
heaan_tolerance = 1.0e-3

if heaan_error > heaan_tolerance:
    raise SystemExit(
        "ERROR: HEaaN oracle exceeds cross-validation acceptance bound: "
        f"{heaan_error} > {heaan_tolerance}"
    )

differences = {
    coordinate: abs(
        rust_actual[coordinate] - heaan_actual[coordinate]
    )
    for coordinate in coordinates
}

max_difference = max(differences.values())

print()
print("===== CCMM CROSS-VALIDATION REPORT =====")
print()

for coordinate in coordinates:
    print(
        f"RUST_HEAAN_ABS_DIFF_{coordinate}="
        f"{differences[coordinate]:.17e}"
    )

print(
    f"RUST_CCMM_MAX_ABS_ERROR={rust_error:.17e}"
)

print(
    f"HEAAN_CCMM_MAX_ABS_ERROR={heaan_error:.17e}"
)

print(
    f"RUST_HEAAN_MAX_ABS_DIFFERENCE={max_difference:.17e}"
)

print(
    f"CCMM_CROSS_VALIDATION_TOLERANCE="
    f"{cross_tolerance:.17e}"
)

if max_difference > cross_tolerance:
    print("CCMM_CROSS_VALIDATION=FAIL")
    raise SystemExit(
        "ERROR: Rust and HEaaN disagree beyond tolerance"
    )

print("CCMM_CROSS_VALIDATION=PASS")
PY
