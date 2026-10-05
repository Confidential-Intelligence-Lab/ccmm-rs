#!/usr/bin/env bash
set -euo pipefail

input="results/image_filter/source/highres_bw_test.png"
binary="target/release/ckks_tiled_image_fft"
outdir="results/fft_milestones/planning_ladder"

mkdir -p "$outdir"

cargo build --locked --release --bin ckks_tiled_image_fft

{
    echo "FFT_PLANNING_GIT_HEAD=$(git rev-parse HEAD)"
    echo "FFT_PLANNING_GIT_BRANCH=$(git branch --show-current)"
    echo "FFT_PLANNING_TIMESTAMP_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

    for dimension in 256 512 1024 2048 4096; do
        echo
        echo "=== ${dimension}x${dimension} ==="

        "$binary" \
            "$input" \
            "$dimension" 64 \
            --plan-only \
            | grep -E \
'FFT_CHAIN_LIMBS|FFT_TOTAL_MODULUS_BITS|FHE_FFT_PLAN_IMAGE_DIMENSION|FHE_FFT_PLAN_TILE_COUNT|FHE_FFT_PLAN_LOCAL_STAGES|FHE_FFT_PLAN_CROSS_TILE_STAGES|FHE_FFT_PLAN_TOTAL_EXACT_STAGES|FHE_FFT_PLAN_EXACT_EXECUTION_PROFILE|FHE_FFT_PLAN_EXACT_EXECUTION_CHAIN_LIMBS|FHE_FFT_PLAN_TERMINAL_GUARD_LIMBS|FHE_FFT_PLAN_ONLY_STATUS'
    done
} | tee "$outdir/planning_ladder.log"
