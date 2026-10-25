#!/bin/bash
# Read-only process and experiment-log monitor.
#
# Usage:
#   scripts/monitor_process.sh PROCESS_NAME LOG_PATH [WINDOW_PATTERN] [PROGRESS_PATTERN]

set -u

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    printf 'Usage: %s PROCESS_NAME LOG_PATH [WINDOW_PATTERN] [PROGRESS_PATTERN]\n' "$0"
    exit 2
fi

PROCESS_NAME="$1"
LOG="$2"
WINDOW_PATTERN="${3:-^R36_8D_WINDOW=}"
PROGRESS_PATTERN="${4:-R36_8D_WINDOW=|R36_8F_WINDOW=|R37_MEMORY_CHECKPOINT=|UHD_FFT2_EXECUTION_STATUS=}"

printf '\n=== PROCESS TREE: %s ===\n' "$PROCESS_NAME"

ps -axo pid,ppid,etime,rss,%cpu,command |
    awk -v name="$PROCESS_NAME" '
        NR == 1 {
            print
            next
        }
        index($0, name) &&
        $0 !~ /awk -v name=/ &&
        $0 !~ /monitor_process.sh/ {
            print
            found = 1
        }
        END {
            if (!found) {
                print "PROCESS_STATUS=NOT_FOUND"
            }
        }
    '

printf '\n=== COMPLETED WINDOWS ===\n'

if [ -f "$LOG" ]; then
    count=$(grep -Ec "$WINDOW_PATTERN" "$LOG" || true)
    printf 'COMPLETED_WINDOWS=%s\n' "$count"
else
    printf 'LOG_STATUS=NOT_FOUND\n'
    printf 'LOG_PATH=%s\n' "$LOG"
fi

printf '\n=== LATEST PROGRESS ===\n'

if [ -f "$LOG" ]; then
    grep -E "$PROGRESS_PATTERN" "$LOG" | tail -20 || true
fi

printf '\n=== MONITOR SUMMARY ===\n'
printf 'PROCESS_NAME=%s\n' "$PROCESS_NAME"
printf 'LOG_PATH=%s\n' "$LOG"
printf 'MONITOR_STATUS=PASS\n'
