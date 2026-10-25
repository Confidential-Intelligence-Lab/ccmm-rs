#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
MODE="${1:-preflight}"
[[ "$MODE" == preflight || "$MODE" == run ]] || { echo 'USAGE: preflight|run' >&2; exit 2; }
test -f src/bin/ckks_packed_1k_fft_validation.rs || exit 2
python3 - <<'PY_CHECK'
from pathlib import Path
s=Path('src/bin/ckks_packed_1k_fft_validation.rs').read_text()
for marker in ('R17_1K_STATUS=', 'R17_1K_FFT_MS=', 'R17_1K_BENCHMARK'):
    assert marker in s, f'MISSING_CONTRACT={marker}'
print('FFT2_1K_PREFLIGHT_STATUS=PASS')
PY_CHECK
[[ "$MODE" == preflight ]] && exit 0
export FHE_PACKED_1K_NTT_RESIDENT=1 R17_1K_LOCAL_PARALLELISM=2 R17_1K_INTER_CT_PARALLELISM=4 R17_1K_BENCHMARK=1
OUT="results/fft_v3/fft2_1k/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_head.txt"
git status --short > "$OUT/git_status.txt"
cargo build --locked --release --bin ckks_packed_1k_fft_validation
target/release/ckks_packed_1k_fft_validation 2>&1 | tee "$OUT/run.log"
grep -qx 'R17_1K_STATUS=PASS' "$OUT/run.log"
echo "FFT2_1K_STATUS=PASS OUTPUT=$OUT"
