#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
MODE="${1:-preflight}"
[[ "$MODE" == preflight || "$MODE" == run ]] || { echo 'USAGE: preflight|run' >&2; exit 2; }
python3 - <<'PY_CHECK'
from pathlib import Path
import re
s=Path('src/bin/ckks_multi_tile_fft_validation.rs').read_text()
assert re.search(r'Fft2Shape::new\(\s*64\s*,\s*64\s*\)', s)
assert 'MULTI_TILE_FFT_STATUS=' in s
print('FFT2_64_PREFLIGHT_STATUS=PASS')
PY_CHECK
[[ "$MODE" == preflight ]] && exit 0
OUT="results/fft_v3/fft2_64/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_head.txt"
git status --short > "$OUT/git_status.txt"
cargo build --locked --release --bin ckks_multi_tile_fft_validation
set -o pipefail
target/release/ckks_multi_tile_fft_validation 2>&1 | tee "$OUT/run.log"
grep -qx 'MULTI_TILE_FFT_STATUS=PASS' "$OUT/run.log" || { echo 'FFT2_64_STATUS=FAIL_VALIDATION' >&2; exit 3; }
echo "FFT2_64_STATUS=PASS OUTPUT=$OUT"
