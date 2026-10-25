#!/usr/bin/env bash
set -euo pipefail

DIMENSION="${1:?usage: $0 1024}"
INPUT="results/image_filter/source/highres_bw_test.png"
OUT="results/image_filter/encrypted_fft_campaign_20261009"
BIN="target/release/ckks_tiled_image_fft"

EXPECTED_BIN_SHA="9cd4294093c51264c3fdc255833cd707974ecf9ccb0a098f11d71a22dcfa4f4f"
EXPECTED_INPUT_SHA="a96e6eff4868bd7203e9bd00d5ef8d9af11e9356a9df2a921e4ec90de819e505"

if [[ "$DIMENSION" != "1024" ]]; then
    echo "ERROR: this 24-GiB machine is approved only for 1024x1024"
    exit 2
fi

test -x "$BIN"
test -s "$INPUT"
test -s "$OUT/${DIMENSION}_plan.log"

grep -qx 'FHE_FFT_PLAN_ONLY_STATUS=PASS' \
    "$OUT/${DIMENSION}_plan.log"

grep -qx 'FFT_RESOURCE_PLAN_END' \
    "$OUT/${DIMENSION}_plan.log"

ACTUAL_BIN_SHA="$(shasum -a 256 "$BIN" | awk '{print $1}')"
ACTUAL_INPUT_SHA="$(shasum -a 256 "$INPUT" | awk '{print $1}')"

if [[ "$ACTUAL_BIN_SHA" != "$EXPECTED_BIN_SHA" ]]; then
    echo "ERROR: release executable changed since preflight"
    exit 3
fi

if [[ "$ACTUAL_INPUT_SHA" != "$EXPECTED_INPUT_SHA" ]]; then
    echo "ERROR: source image changed since preflight"
    exit 3
fi

RUN_DIR="$OUT/${DIMENSION}_run"
IMAGE_DIR="results/image_filter/encrypted_frequency_filter/tiled_fft_${DIMENSION}x${DIMENSION}_tile_64x64"

if [[ -e "$RUN_DIR" ]]; then
    echo "ERROR: run directory already exists: $RUN_DIR"
    exit 4
fi

# The executable writes to a fixed image directory.
# Reject existing artifacts to avoid confusing runs.
for name in \
    03_clear_global_fft_log_magnitude.png \
    04_encrypted_global_fft_log_magnitude.png \
    05_global_fft_absolute_error.png
do
    if [[ -e "$IMAGE_DIR/$name" ]]; then
        echo "ERROR: existing spectral artifact: $IMAGE_DIR/$name"
        exit 4
    fi
done

mkdir -p "$RUN_DIR"

LOG="$RUN_DIR/run.log"
TIME_LOG="$RUN_DIR/resources.log"
META="$RUN_DIR/metadata.txt"

{
    echo "DIMENSION=$DIMENSION"
    echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "GIT_HEAD=$(git rev-parse HEAD)"
    echo "GIT_DIRTY=$(test -z "$(git status --porcelain)" && echo false || echo true)"
    echo "BIN_SHA256=$ACTUAL_BIN_SHA"
    echo "INPUT_SHA256=$ACTUAL_INPUT_SHA"
    echo "IMAGE_OUTPUT_DIR=$IMAGE_DIR"
    echo "FFT_SECURITY_BEARING=false"
} > "$META"

cp "$OUT/${DIMENSION}_plan.log" "$RUN_DIR/plan.log"

echo "Starting encrypted global FFT2 ${DIMENSION}x${DIMENSION}"
echo "Run directory: $RUN_DIR"

set +e
/usr/bin/time -l "$BIN" "$INPUT" "$DIMENSION" 64 \
    > "$LOG" 2> "$TIME_LOG"
STATUS=$?
set -e

{
    echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "PROCESS_EXIT_CODE=$STATUS"
} >> "$META"

if [[ "$STATUS" -ne 0 ]]; then
    echo "ERROR: FFT process exited with status $STATUS"
    echo "See $LOG and $TIME_LOG"
    exit "$STATUS"
fi

if ! grep -qx 'FACTORED_FFT_RESULT_STATUS=PASS' "$LOG"; then
    echo "ERROR: numerical correctness PASS marker missing"
    exit 5
fi

if ! grep -qx 'FFT_GLOBAL_SPECTRUM_EXPORT_STATUS=PASS' "$LOG"; then
    echo "ERROR: spectral export PASS marker missing"
    exit 5
fi

for name in \
    03_clear_global_fft_log_magnitude.png \
    04_encrypted_global_fft_log_magnitude.png \
    05_global_fft_absolute_error.png
do
    if [[ ! -s "$IMAGE_DIR/$name" ]]; then
        echo "ERROR: missing or empty image: $IMAGE_DIR/$name"
        exit 6
    fi
done

{
    echo "VALIDATION_STATUS=PASS"
    echo "OUTPUT_IMAGE_HASHES_BEGIN"
    shasum -a 256 \
        "$IMAGE_DIR/03_clear_global_fft_log_magnitude.png" \
        "$IMAGE_DIR/04_encrypted_global_fft_log_magnitude.png" \
        "$IMAGE_DIR/05_global_fft_absolute_error.png"
    echo "OUTPUT_IMAGE_HASHES_END"
} >> "$META"

echo
echo "=== FFT RESULTS ==="
rg 'FFT_TIMING_|FFT_METRIC_|FACTORED_FFT_REL_L2=|FACTORED_FFT_MAX_ABS=|FACTORED_FFT_RESULT_STATUS=|FFT_GLOBAL_SPECTRUM_EXPORT_STATUS=' \
    "$LOG" | tail -45

echo
echo "=== PEAK MEMORY ==="
rg 'maximum resident set size' "$TIME_LOG" || true

echo
echo "NIGHTTIME_FFT_CAMPAIGN_STATUS=PASS"
