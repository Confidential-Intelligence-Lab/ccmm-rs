#!/usr/bin/env bash

set -u

LOG="${1:-results/fft_milestones/r15_g15b_bounded_gaussian_128/run.log}"
INTERVAL="${2:-10}"
PROCESS="${3:-ckks_tiled_image_fft}"

if ! [[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: refresh interval must be a positive integer\n' >&2
    exit 2
fi

printf 'FHE experiment monitor\n'
printf 'Log: %s\n' "$LOG"
printf 'Process: %s\n' "$PROCESS"
printf 'Refresh: %s seconds\n' "$INTERVAL"

while true; do
    clear

    printf '============================================================\n'
    printf ' FHE-rs | Experiment Monitor\n'
    printf '============================================================\n'
    date

    printf '\n=== EXPERIMENT ===\n'
    printf 'Log: %s\n' "$LOG"

    if [ -f "$LOG" ]; then
        COMPLETED=$(grep -c '^FFT_STAGE_END' "$LOG" || true)
        STARTED=$(grep -c '^FFT_STAGE_BEGIN' "$LOG" || true)

        printf 'FFT stages started:   %s\n' "$STARTED"
        printf 'FFT stages completed: %s\n' "$COMPLETED"

        printf '\n=== RECENT PROGRESS ===\n'
        grep -E \
            'FFT_STAGE_BEGIN|FFT_STAGE_END|R15_G12_KEY_PIPELINE_WALL_MS|FACTORED_FFT_RESULT_STATUS|FFT_METRIC_END_TO_END_MS' \
            "$LOG" | tail -10 || true

        printf '\n=== CORRECTNESS ===\n'
        grep -E \
            'FACTORED_FFT_REL_L2=|FACTORED_FFT_MAX_ABS=|FACTORED_FFT_RESULT_STATUS=' \
            "$LOG" | tail -5 || true
    else
        printf 'Waiting for experiment log...\n'
    fi

    printf '\n=== PROCESS ===\n'
    ps -axo pid,%cpu,rss,etime,command |
        grep -F "$PROCESS" |
        grep -v 'grep -F' |
        head -5 || true

    printf '\nRSS is reported in KiB.\n'

    printf '\n=== SYSTEM SWAP ===\n'
    sysctl vm.swapusage 2>/dev/null || true

    printf '\n=== CONTROLS ===\n'
    printf 'Ctrl+C: stop monitor only; experiment continues.\n'
    printf 'Refresh interval: %s seconds\n' "$INTERVAL"

    sleep "$INTERVAL"
done
