#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
MODE="${1:-preflight}"
[[ "$MODE" == preflight || "$MODE" == run ]] || { echo 'USAGE: preflight|run' >&2; exit 2; }
N=256
python3 - "$N" <<'PY_SMALL'
from pathlib import Path
import re,sys
n=int(sys.argv[1]); src=Path('src/bin')
rows=[]
for p in sorted(src.glob('*.rs')):
    if 'fft' not in p.name.lower(): continue
    s=p.read_text(errors='replace')
    matches=[]
    for pattern in (rf'Fft2Shape::new\(\s*{n}\s*,\s*{n}\s*\)', rf'\b{n}\s*[xX]\s*{n}\b', rf'\bFFT{n}\b', rf'\bfft{n}\b'):
        if re.search(pattern,s): matches.append(pattern)
    if matches: rows.append(p.name)
print(f'FFT2_{n}_CANDIDATE_SOURCES='+(','.join(rows) if rows else 'NONE'))
print(f'FFT2_{n}_EXECUTION_CONTRACT=NOT_YET_VERIFIED')
PY_SMALL
if [[ "$MODE" == run ]]; then
  echo "FFT2_${N}_STATUS=BLOCKED_UNVERIFIED_EXECUTION_CONTRACT" >&2
  exit 3
fi
echo "FFT2_${N}_PREFLIGHT_STATUS=REQUIRES_SOURCE_REVIEW"
