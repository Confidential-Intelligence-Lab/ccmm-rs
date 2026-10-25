#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
MODE="${1:-preflight}"
[[ "$MODE" == preflight || "$MODE" == run ]] || { echo 'USAGE: preflight|run' >&2; exit 2; }
for bin in eblas_fft1_characterization eblas_packed_fft1_cp_validation; do
  test -f "src/bin/$bin.rs" || { echo "MISSING_BINARY_SOURCE=$bin" >&2; exit 2; }
done
echo 'FFT1_PREFLIGHT_STATUS=PASS'
[[ "$MODE" == preflight ]] && exit 0
OUT="results/fft_v3/fft1/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_head.txt"
git status --short > "$OUT/git_status.txt"
for bin in eblas_fft1_characterization eblas_packed_fft1_cp_validation; do
  cargo run --locked --release --bin "$bin" 2>&1 | tee "$OUT/$bin.log"
done
grep -q '^EBLAS_FFT1_CHARACTERIZATION_STATUS=PASS' "$OUT/eblas_fft1_characterization.log"
grep -q '^PACKED_FFT1_CP_STATUS=PASS' "$OUT/eblas_packed_fft1_cp_validation.log"
echo "FFT1_STATUS=PASS OUTPUT=$OUT"
