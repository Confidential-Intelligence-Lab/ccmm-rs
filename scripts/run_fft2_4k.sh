#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
MODE="${1:-preflight}"
[[ "$MODE" == preflight || "$MODE" == run ]] || { echo 'USAGE: preflight|run' >&2; exit 2; }
test -f src/bin/ckks_packed_uhd_fft_validation.rs || exit 2
IMAGE=results/image_filter/source/benchmarks/fhe_fft_architecture_gray_4096x4096.png
EXPECTED=7939621968e32052bbf47a67de3639825418b483d22a37b7139c01be3861f7b3
[[ -f "$IMAGE" ]] || { echo "MISSING_IMAGE=$IMAGE" >&2; exit 2; }
ACTUAL="$(shasum -a 256 "$IMAGE" | awk '{print $1}')"
[[ "$ACTUAL" == "$EXPECTED" ]] || { echo 'IMAGE_SHA256_MISMATCH' >&2; exit 2; }
for flag in R22_BENCH_FIRST_MACRO_ROW R22_EQUIVALENCE R22_4_EQUIVALENCE R22_5_EQUIVALENCE; do
  [[ "${!flag:-0}" != 1 ]] || { echo "PARTIAL_MODE_FORBIDDEN=$flag" >&2; exit 2; }
done
echo 'FFT2_4K_PREFLIGHT_STATUS=PASS'
[[ "$MODE" == preflight ]] && exit 0
export FHE_PACKED_1K_NTT_RESIDENT=1 R21_UHD_LOCAL_PARALLELISM=8 R21_UHD_INTER_CT_PARALLELISM=4
OUT="results/fft_v3/fft2_4k/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_head.txt"
git status --short > "$OUT/git_status.txt"
shasum -a 256 "$IMAGE" > "$OUT/input_sha256.txt"
cargo build --locked --release --bin ckks_packed_uhd_fft_validation
target/release/ckks_packed_uhd_fft_validation "$IMAGE" 2>&1 | tee "$OUT/run.log"
grep -qx 'R21_UHD_STATUS=PASS' "$OUT/run.log"
echo "FFT2_4K_STATUS=PASS OUTPUT=$OUT"
