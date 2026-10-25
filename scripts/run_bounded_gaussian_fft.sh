#!/usr/bin/env bash
set -euo pipefail

DIMENSION="${1:?usage: scripts/run_bounded_gaussian_fft.sh DIMENSION MILESTONE}"
MILESTONE="${2:?provide a unique milestone directory name}"
RANDOMNESS_MODE="${3:-deterministic}"

[[ "$RANDOMNESS_MODE" == "deterministic" ||
   "$RANDOMNESS_MODE" == "secure" ]] || {
    echo "ERROR: randomness mode must be deterministic or secure"
    exit 2
}

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

BIN="target/release/ckks_tiled_image_fft"
INPUT="results/image_filter/source/highres_bw_test.png"
OUT="results/fft_milestones/$MILESTONE"

[[ "$DIMENSION" =~ ^(64|128|256|512)$ ]] || {
    echo "ERROR: unsupported campaign dimension"
    exit 2
}

[[ "$MILESTONE" =~ ^[a-zA-Z0-9_-]+$ ]] || {
    echo "ERROR: invalid milestone name"
    exit 2
}

[[ -x "$BIN" && -s "$INPUT" ]] || {
    echo "ERROR: missing binary or source image"
    exit 2
}

[[ ! -e "$OUT" ]] || {
    echo "ERROR: milestone already exists: $OUT"
    exit 2
}

mkdir -p "$OUT"

export FFT_GALOIS_KEY_MODE=bounded_gaussian_direct
export FFT_RANDOMNESS_MODE="$RANDOMNESS_MODE"
export FHE_KEYGEN_PROFILE=0

"$BIN" "$INPUT" "$DIMENSION" 64 --plan-only \
    > "$OUT/plan.log" 2>&1 || {
        echo "ERROR: FFT preflight failed"
        exit 3
    }

grep -qx 'FHE_FFT_PLAN_ONLY_STATUS=PASS' "$OUT/plan.log" || {
    echo "ERROR: FFT plan did not pass"
    exit 3
}

EXPECTED_TILES=$(( (DIMENSION / 64) * (DIMENSION / 64) ))

grep -qx "FFT_PACKING_CIPHERTEXT_COUNT=$EXPECTED_TILES" "$OUT/plan.log" || {
    echo "ERROR: unexpected ciphertext count"
    exit 3
}

IMAGE_DIR="$OUT/images"

[[ ! -e "$IMAGE_DIR" ]] || {
    echo "ERROR: milestone image directory already exists: $IMAGE_DIR"
    exit 4
}

export FFT_OUTPUT_DIR="$IMAGE_DIR"

{
    echo "DIMENSION=$DIMENSION"
    echo "TILE_DIMENSION=64"
    echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "GIT_HEAD=$(git rev-parse HEAD)"
    echo "GIT_DIRTY=$(test -z "$(git status --porcelain)" && echo false || echo true)"
    echo "BIN_SHA256=$(shasum -a 256 "$BIN" | awk '{print $1}')"
    echo "INPUT_SHA256=$(shasum -a 256 "$INPUT" | awk '{print $1}')"
    echo "FFT_GALOIS_KEY_MODE=$FFT_GALOIS_KEY_MODE"
    echo "FFT_RANDOMNESS_MODE=$FFT_RANDOMNESS_MODE"
    echo "FFT_RANDOMNESS_SOURCE=$(
        if [[ "$FFT_RANDOMNESS_MODE" == "secure" ]]; then
            echo os_entropy_chacha20
        else
            echo deterministic_chacha20
        fi
    )"
    echo "FFT_SECURITY_BEARING=false"
    echo "IMAGE_OUTPUT_DIR=$IMAGE_DIR"
} > "$OUT/metadata.txt"

echo "R15_CAMPAIGN_START=$MILESTONE"
echo "R15_CAMPAIGN_LOG=$OUT/run.log"

set +e
/usr/bin/time -l "$BIN" "$INPUT" "$DIMENSION" 64 \
    > "$OUT/run.log" 2> "$OUT/resources.log"
STATUS=$?
set -e

{
    echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "PROCESS_EXIT_CODE=$STATUS"
} >> "$OUT/metadata.txt"

if [[ "$STATUS" -ne 0 ]]; then
    echo "R15_CAMPAIGN_STATUS=FAIL"
    exit "$STATUS"
fi

grep -qx 'FACTORED_FFT_RESULT_STATUS=PASS' "$OUT/run.log" || {
    echo "R15_CAMPAIGN_STATUS=FAIL"
    exit 5
}

grep -qx 'FFT_GLOBAL_SPECTRUM_EXPORT_STATUS=PASS' "$OUT/run.log" || {
    echo "R15_CAMPAIGN_STATUS=FAIL"
    exit 5
}

{
    echo "VALIDATION_STATUS=PASS"
    echo "OUTPUT_IMAGE_HASHES_BEGIN"
    for name in \
        03_clear_global_fft_log_magnitude.png \
        04_encrypted_global_fft_log_magnitude.png \
        05_global_fft_absolute_error.png
    do
        test -s "$IMAGE_DIR/$name" || exit 6
        shasum -a 256 "$IMAGE_DIR/$name"
    done
    echo "OUTPUT_IMAGE_HASHES_END"
} >> "$OUT/metadata.txt"

echo
echo "=== CORRECTNESS ==="
grep -E 'FACTORED_FFT_(REL_L2|MAX_ABS|RESULT_STATUS)=' "$OUT/run.log"

echo
echo "=== TIMING ==="
grep -E 'FFT_METRIC_(ENCRYPTED_FFT2_MS|END_TO_END_MS)=' "$OUT/run.log"

echo
echo "=== PEAK RSS ==="
grep 'maximum resident set size' "$OUT/resources.log"

echo "R15_CAMPAIGN_STATUS=PASS"
