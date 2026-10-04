#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

OUT="results/eblas_correlation_evidence"
mkdir -p "$OUT"

RAW="$OUT/raw.log"
SUMMARY="$OUT/summary.csv"
META="$OUT/metadata.txt"

printf '\n=== CORRELATION EVIDENCE RUN ===\n'

{
    echo "CORRELATION_EVIDENCE_SHA=$(git rev-parse HEAD)"
    echo "CORRELATION_EVIDENCE_DATE_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "CORRELATION_EVIDENCE_RUSTC=$(rustc --version)"
    echo "CORRELATION_EVIDENCE_CARGO=$(cargo --version)"
    echo "CORRELATION_EVIDENCE_UNAME=$(uname -a)"
    echo "CORRELATION_EVIDENCE_TEST_THREADS=1"
} > "$META"

RUST_TEST_THREADS=1 \
cargo test --release \
    eblas::correlation::tests:: \
    -- --nocapture 2>&1 | tee "$RAW"

python3 - "$RAW" "$SUMMARY" <<'PY'
from pathlib import Path
import csv
import re
import sys

raw_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])

lines = raw_path.read_text().splitlines()

case_1d = re.compile(
    r"CORRELATION_(CP|CC)_CASE=N(\d+) K(\d+) OUT(\d+)"
)
metric_1d = re.compile(
    r"CORRELATION_(CP|CC)_REL_L2=([0-9.eE+-]+) "
    r"CORRELATION_\1_MAX_ABS=([0-9.eE+-]+)"
)

case_2d = re.compile(
    r"CORRELATION_2D_(CP|CC)_CASE="
    r"H(\d+) W(\d+) C(\d+) "
    r"KH(\d+) KW(\d+) F(\d+) "
    r"M(\d+) K(\d+) N(\d+)"
)
metric_2d = re.compile(
    r"CORRELATION_2D_(CP|CC)_REL_L2=([0-9.eE+-]+) "
    r"CORRELATION_2D_\1_MAX_ABS=([0-9.eE+-]+)"
)

records = []
pending = {}

for line in lines:
    m = case_1d.search(line)
    if m:
        privacy, signal, kernel, output = m.groups()
        pending[("1d", privacy)] = {
            "operator": "correlation_1d",
            "privacy": privacy,
            "input_height_or_signal": signal,
            "input_width": "",
            "channels": "",
            "kernel_height_or_length": kernel,
            "kernel_width": "",
            "filters": "1",
            "m": output,
            "k": kernel,
            "n": "1",
            "output_extent": output,
        }
        continue

    m = metric_1d.search(line)
    if m:
        privacy, rel_l2, max_abs = m.groups()
        key = ("1d", privacy)
        if key not in pending:
            raise SystemExit(
                f"metric without preceding 1D case: {line}"
            )

        record = pending.pop(key)
        record["rel_l2"] = rel_l2
        record["max_abs"] = max_abs
        records.append(record)
        continue

    m = case_2d.search(line)
    if m:
        (
            privacy,
            height,
            width,
            channels,
            kernel_height,
            kernel_width,
            filters,
            logical_m,
            logical_k,
            logical_n,
        ) = m.groups()

        pending[("2d", privacy)] = {
            "operator": "correlation_2d",
            "privacy": privacy,
            "input_height_or_signal": height,
            "input_width": width,
            "channels": channels,
            "kernel_height_or_length": kernel_height,
            "kernel_width": kernel_width,
            "filters": filters,
            "m": logical_m,
            "k": logical_k,
            "n": logical_n,
            "output_extent": "",
        }
        continue

    m = metric_2d.search(line)
    if m:
        privacy, rel_l2, max_abs = m.groups()
        key = ("2d", privacy)
        if key not in pending:
            raise SystemExit(
                f"metric without preceding 2D case: {line}"
            )

        record = pending.pop(key)
        record["rel_l2"] = rel_l2
        record["max_abs"] = max_abs
        records.append(record)

if pending:
    raise SystemExit(f"unmatched evidence cases remain: {pending}")

if not records:
    raise SystemExit("no correlation numerical evidence found")

fields = [
    "operator",
    "privacy",
    "input_height_or_signal",
    "input_width",
    "channels",
    "kernel_height_or_length",
    "kernel_width",
    "filters",
    "m",
    "k",
    "n",
    "output_extent",
    "rel_l2",
    "max_abs",
]

with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)

print(f"CORRELATION_EVIDENCE_RECORDS={len(records)}")

counts = {}
for record in records:
    key = (record["operator"], record["privacy"])
    counts[key] = counts.get(key, 0) + 1

for (operator, privacy), count in sorted(counts.items()):
    print(
        f"CORRELATION_EVIDENCE_COUNT_"
        f"{operator.upper()}_{privacy}={count}"
    )
PY

grep -q \
    'test result: ok. 21 passed; 0 failed' \
    "$RAW"

printf '\n=== CORRELATION SUMMARY ===\n'
cat "$SUMMARY"

printf '\n=== CORRELATION METADATA ===\n'
cat "$META"

printf '\n=== ENGINEERING GATES ===\n'
cargo check --release --tests
cargo clippy --release --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check

printf '\nCORRELATION_EVIDENCE_STATUS=PASS\n'
