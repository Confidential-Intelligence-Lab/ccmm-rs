#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <dimension> <tile-dimension> <tag>" >&2
    exit 2
fi

dimension="$1"
tile_dimension="$2"
tag="$3"

input="results/image_filter/source/highres_bw_test.png"
binary="target/release/ckks_tiled_image_fft"
outdir="results/fft_milestones/${tag}"

mkdir -p "$outdir"

echo "=== BUILD ==="
cargo build --locked --release --bin ckks_tiled_image_fft

echo "=== SOURCE STATE ==="
{
    echo "FFT_EVIDENCE_TAG=${tag}"
    echo "FFT_EVIDENCE_DIMENSION=${dimension}"
    echo "FFT_EVIDENCE_TILE_DIMENSION=${tile_dimension}"
    echo "FFT_EVIDENCE_GIT_HEAD=$(git rev-parse HEAD)"
    echo "FFT_EVIDENCE_GIT_BRANCH=$(git branch --show-current)"
    echo "FFT_EVIDENCE_TIMESTAMP_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "FFT_EVIDENCE_UNAME=$(uname -a)"
    echo "FFT_EVIDENCE_LOGICAL_CPUS=$(sysctl -n hw.logicalcpu 2>/dev/null || echo unknown)"
    echo "FFT_EVIDENCE_PHYSICAL_CPUS=$(sysctl -n hw.physicalcpu 2>/dev/null || echo unknown)"
    echo "FFT_EVIDENCE_MEMORY_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo unknown)"
} | tee "$outdir/environment.txt"

echo
echo "=== PLAN ==="
"$binary" \
    "$input" \
    "$dimension" \
    "$tile_dimension" \
    --plan-only \
    | tee "$outdir/plan.log"

echo
echo "=== EXACT ENCRYPTED FFT2 ==="
/usr/bin/time -l \
    "$binary" \
    "$input" \
    "$dimension" \
    "$tile_dimension" \
    2> "$outdir/resources.log" \
    | tee "$outdir/run.log"

echo
echo "=== RESULT SUMMARY ==="
grep -E \
'^(FFT_PROFILE|FFT_RING_DEGREE|FFT_SLOT_COUNT|FFT_CHAIN_LIMBS|FFT_TOTAL_MODULUS_BITS|FFT_EXPECTED_LEVELS_CONSUMED|FFT_RESOURCE_|FFT_STAGE_CHECK|FFT_TIMING_|FFT_METRIC_|FACTORED_FFT_)' \
    "$outdir/run.log" \
    | tee "$outdir/summary.log"

echo
echo "=== RESOURCE SUMMARY ==="
grep -E \
'maximum resident set size|user|system|real|page reclaims|page faults|swaps' \
    "$outdir/resources.log" \
    || true

echo
echo "FFT_MILESTONE_STATUS=COMPLETE"
echo "FFT_MILESTONE_OUTPUT=$outdir"
