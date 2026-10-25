#!/usr/bin/env bash
set -euo pipefail
[[ -f Cargo.toml && -f src/bin/ckks_packed_uhd_fft_validation.rs ]] || {
  echo 'ERROR: execute from ccmm-rs repository root with the v3 4K validator' >&2; exit 2;
}
ROOT="$(pwd)"
BASE="$ROOT/results/eblas_fft_evidence/icpp_4k_baseline_r21"
IMAGE="$ROOT/results/image_filter/source/benchmarks/fhe_fft_architecture_gray_4096x4096.png"
[[ -f "$BASE/input_sha256.txt" && -f "$IMAGE" ]] || {
  echo 'ERROR: original R21 baseline image/hash unavailable' >&2; exit 2;
}
python3 - "$IMAGE" "$BASE/input_sha256.txt" <<'PY'
import hashlib,sys
from pathlib import Path
image=Path(sys.argv[1]); recorded=Path(sys.argv[2]).read_text()
h=hashlib.sha256(image.read_bytes()).hexdigest()
assert h in recorded, '4K input SHA-256 does not match R21 baseline'
print('V3_4K_IMAGE_IDENTITY=PASS')
PY
for flag in R22_BENCH_FIRST_MACRO_ROW R22_EQUIVALENCE R22_4_EQUIVALENCE R22_5_EQUIVALENCE; do
  if [[ "${!flag:-0}" == 1 ]]; then echo "ERROR: partial/equivalence mode enabled: $flag" >&2; exit 2; fi
done
export FHE_PACKED_1K_NTT_RESIDENT=1
export R21_UHD_LOCAL_PARALLELISM=8
export R21_UHD_INTER_CT_PARALLELISM=4
RUN_ID="${FHE_V3_4K_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: invalid run id' >&2; exit 2; }
OUT="$ROOT/results/eblas_fft_evidence/v3_4k_runs/$RUN_ID"
[[ ! -e "$OUT" ]] || { echo "ERROR: output already exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_head.txt"
git status --short > "$OUT/git_status.txt"
git diff --binary > "$OUT/working_tree.diff"
shasum -a 256 "$IMAGE" > "$OUT/input_sha256.txt"
{
  printf 'INPUT=%s\n' "$IMAGE"
  printf 'FHE_PACKED_1K_NTT_RESIDENT=%s\n' "$FHE_PACKED_1K_NTT_RESIDENT"
  printf 'R21_UHD_LOCAL_PARALLELISM=%s\n' "$R21_UHD_LOCAL_PARALLELISM"
  printf 'R21_UHD_INTER_CT_PARALLELISM=%s\n' "$R21_UHD_INTER_CT_PARALLELISM"
  rustc --version
  cargo --version
  uname -a
} > "$OUT/configuration.txt"
printf '%s\n' "target/release/ckks_packed_uhd_fft_validation $IMAGE" > "$OUT/command.txt"
cargo build --locked --release --bin ckks_packed_uhd_fft_validation
set +e
if [[ "$(uname)" == Darwin ]]; then
  /usr/bin/time -l target/release/ckks_packed_uhd_fft_validation "$IMAGE" > "$OUT/full_run.log" 2> "$OUT/resources.log"
else
  /usr/bin/time -v target/release/ckks_packed_uhd_fft_validation "$IMAGE" > "$OUT/full_run.log" 2> "$OUT/resources.log"
fi
RC=$?
set -e
printf 'EXIT_CODE=%s\n' "$RC" > "$OUT/exit_status.txt"
grep -E '^R21_UHD_(STATUS|FFT_MS|TOTAL_MS|REL_L2|MAX_ABS|CIPHERTEXT_COUNT|IMAGE_DIMENSION)=' "$OUT/full_run.log" || true
[[ "$RC" -eq 0 ]] || { echo "V3_4K_STATUS=FAIL EXIT_CODE=$RC OUTPUT=$OUT"; exit "$RC"; }
grep -qx 'R21_UHD_STATUS=PASS' "$OUT/full_run.log" || { echo 'V3_4K_STATUS=FAIL_VALIDATION' >&2; exit 3; }
echo 'V3_4K_STATUS=PASS'
echo "V3_4K_OUTPUT=$OUT"
