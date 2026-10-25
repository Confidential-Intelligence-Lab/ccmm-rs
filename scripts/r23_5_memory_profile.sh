#!/usr/bin/env bash
set -e
set -o pipefail

LOG="${R23_5_LOG:-/tmp/r23_5_memory_profile.log}"
TIME_LOG="${R23_5_TIME_LOG:-/tmp/r23_5_resource_usage.log}"

echo "R23_5_PROFILE=UHD_HORIZONTAL_FFT3840"
echo "R23_5_BRANCHES=15"
echo "R23_5_EXPECTED_OUTPUT_LEVEL=7"
echo "R23_5_MEMORY_MONITOR=MACOS_TIME_L"

# macOS /usr/bin/time -l reports maximum resident set size.
# Its measurement covers the complete child process, including
# key generation, production execution, and reference validation.

/usr/bin/time -l \
    env R23_4C4_COMPARE=1 CARGO_INCREMENTAL=0 \
    cargo test --release \
        --test r16_6c4_production_uhd_fft256 \
        r16_6c4_production_uhd_fft256 \
        -- --exact --ignored --nocapture \
    > "$LOG" 2> "$TIME_LOG"

cat "$LOG"

printf '\n=== RESOURCE USAGE ===\n'
cat "$TIME_LOG"

printf '\n=== VALIDATION ===\n'
grep 'R23_4C4_INTEGRATED_CIPHERTEXT_STATUS=PASS' "$LOG"
grep 'R23_4C4_BRANCHES_VERIFIED=15' "$LOG"
grep 'R23_4C4_OUTPUT_LEVEL=7' "$LOG"

printf '\nR23_5_MEMORY_PROFILE_STATUS=PASS\n'
