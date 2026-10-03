#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${EBLAS_BATCH_OUT_DIR:-results/eblas_batch_characterization}"
mkdir -p "$OUT_DIR"

RAW="$OUT_DIR/raw.log"
CSV="$OUT_DIR/summary.csv"
META="$OUT_DIR/metadata.txt"

: > "$RAW"

{
    echo "EBLAS_BATCH_CHARACTERIZATION_VERSION=1"
    echo "GIT_COMMIT=$(git rev-parse HEAD)"
    echo "GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)"
    echo "RUSTC=$(rustc --version)"
    echo "CARGO=$(cargo --version)"
    echo "LARGE_DEGREE=8192"
    echo "TIMING_REGION=EBLAS_OPERATION"
    echo "SETUP_INCLUDED_IN_TIMING=NO"
    echo "STATISTIC=single_authors_scale_run"
} | tee "$META"

run_case() {
    local mechanism="$1"
    local dimension="$2"
    local scalar_degree="$3"
    local batch_count="$4"
    local test_name="$5"

    local mechanism_lower
    mechanism_lower="$(printf '%s' "$mechanism" | tr '[:upper:]' '[:lower:]')"

    local case_log="$OUT_DIR/${mechanism_lower}_d${dimension}.log"

    printf '\n=== %s d=%s Ns=%s batch=%s ===\n' \
        "$mechanism" "$dimension" "$scalar_degree" "$batch_count" | tee -a "$RAW"

    cargo test --release "$test_name" --lib -- --nocapture \
        2>&1 | tee "$case_log" | tee -a "$RAW"

    local time_ms
    local error
    local status

    time_ms="$(
        grep -E "BATCH_${mechanism}_MATRIX_MULT_MS=" "$case_log" |
        tail -1 |
        cut -d= -f2
    )"

    error="$(
        grep -E "BATCH_${mechanism}_D${dimension}_RELATIVE_L2_ERROR=" "$case_log" |
        tail -1 |
        cut -d= -f2
    )"

    status="$(
        grep -E "BATCH_${mechanism}_D${dimension}_STATUS=" "$case_log" |
        tail -1 |
        cut -d= -f2
    )"

    if [[ -z "$time_ms" || -z "$error" || -z "$status" ]]; then
        echo "ERROR: failed to extract ${mechanism} d=${dimension} result" >&2
        exit 1
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$mechanism" \
        "$([[ "$mechanism" == "CPMM" ]] && echo CP || echo CC)" \
        "$dimension" \
        "$scalar_degree" \
        "8192" \
        "$batch_count" \
        "$time_ms" \
        "$error" \
        "$status" >> "$CSV"
}

cat > "$CSV" <<'EOFCSV'
mechanism,privacy,dimension,scalar_degree,large_degree,batch_count,execution_ms,relative_l2_error,status
EOFCSV

run_case \
    CPMM 64 256 128 \
    batch_cpmm_end_to_end_authors_d64

run_case \
    CPMM 128 128 64 \
    batch_cpmm_end_to_end_authors_d128

run_case \
    CCMM 64 128 64 \
    batch_ccmm_end_to_end_authors_d64

run_case \
    CCMM 128 64 32 \
    batch_ccmm_end_to_end_authors_d128

printf '\n=== EBLAS BATCH CHARACTERIZATION ===\n'
column -s, -t "$CSV" || cat "$CSV"

printf '\nRAW_LOG=%s\n' "$RAW"
printf 'SUMMARY_CSV=%s\n' "$CSV"
printf 'METADATA=%s\n' "$META"
