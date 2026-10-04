#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

OUT="$ROOT/results/eblas_fft_evidence"
RAW="$OUT/raw.log"
SUMMARY="$OUT/summary.csv"
META="$OUT/metadata.txt"

mkdir -p "$OUT"
: > "$RAW"

printf '\n=== EBLAS FFT EVIDENCE RUN ===\n'

{
    echo "EBLAS_FFT_EVIDENCE_VERSION=1"
    echo "GIT_COMMIT=$(git rev-parse HEAD)"
    echo "GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)"
    echo "RUSTC_VERSION=$(rustc --version)"
    echo "CARGO_VERSION=$(cargo --version)"
    echo "STRUCTURAL_COUNTS=implementation_derived"
    echo "MEASURED_FIELDS=validator_reported"
    echo "RUNTIME_POLICY=only_validator_reported_runtime_is_recorded"
} > "$META"

run_bin() {
    local bin="$1"

    {
        printf '\n=== BEGIN %s ===\n' "$bin"
        cargo run --release --bin "$bin"
        printf '=== END %s ===\n' "$bin"
    } 2>&1 | tee -a "$RAW"
}

run_bin eblas_fft1_characterization
run_bin eblas_packed_fft1_cp_validation
run_bin eblas_packed_fft2_stage_validation
run_bin eblas_packed_fft2_cp_validation

python3 - "$RAW" "$SUMMARY" <<'PY'
import csv
import math
import re
import sys
from pathlib import Path

raw_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])

columns = [
    "kind",
    "source",
    "operation",
    "representation",
    "direction",
    "n",
    "rows",
    "cols",
    "profile",
    "security_bearing",
    "row_stages",
    "column_stages",
    "stages",
    "ciphertexts",
    "rotations",
    "public_multiplies",
    "cc_multiplies",
    "relinearizations",
    "additions",
    "transposes",
    "levels_consumed",
    "rel_l2",
    "max_abs",
    "inactive_max_abs",
    "runtime_ns",
    "status",
    "raw_marker",
]

source = None
metadata = {}
rows = []

begin_re = re.compile(r"^=== BEGIN ([A-Za-z0-9_]+) ===$")


def parse_tokens(line):
    marker = ""
    values = {}

    for token in line.strip().split():
        if "=" not in token:
            continue

        key, value = token.split("=", 1)

        # Case records use forms such as:
        #
        # PACKED_FFT1_CP_CASE=DIRECTION=Forward
        #
        # Preserve the marker while recovering the nested first field.
        if key.endswith("_CASE") and "=" in value:
            marker = key
            nested_key, nested_value = value.split("=", 1)
            values[nested_key] = nested_value
        else:
            values[key] = value

    if not marker:
        for key in values:
            if key.endswith("_CASE"):
                marker = key
                break

    return marker, values


def first(values, *names):
    for name in names:
        value = values.get(name)
        if value is not None:
            return value
    return ""


def measured_runtime(values):
    return first(
        values,
        "RUNTIME_NS",
        "MEDIAN_NS",
        "TIME_NS",
        "ELAPSED_NS",
    )


def profile_for(src):
    prefixes = {
        "eblas_packed_fft1_cp_validation": "PACKED_FFT1_CP_PROFILE",
        "eblas_packed_fft2_stage_validation": "PACKED_FFT2_STAGE_PROFILE",
        "eblas_packed_fft2_cp_validation": "PACKED_FFT2_CP_PROFILE",
    }

    key = prefixes.get(src)
    return metadata.get(src, {}).get(key, "") if key else ""


def security_for(src):
    prefixes = {
        "eblas_packed_fft1_cp_validation": "PACKED_FFT1_CP_SECURITY_BEARING",
        "eblas_packed_fft2_stage_validation": "PACKED_FFT2_STAGE_SECURITY_BEARING",
        "eblas_packed_fft2_cp_validation": "PACKED_FFT2_CP_SECURITY_BEARING",
    }

    key = prefixes.get(src)
    return metadata.get(src, {}).get(key, "") if key else ""


def representation_for(src):
    if src == "eblas_fft1_characterization":
        return metadata.get(src, {}).get(
            "EBLAS_FFT1_REPRESENTATION",
            "one_ciphertext_per_logical_sample",
        )

    if src == "eblas_packed_fft1_cp_validation":
        return "packed_ckks_slots"

    if src in {
        "eblas_packed_fft2_stage_validation",
        "eblas_packed_fft2_cp_validation",
    }:
        return "packed_row_major_ckks_slots"

    return ""


def operation_for(marker, src):
    if "FFT2_STAGE" in marker:
        return "FFT2_STAGE"
    if "FFT2" in marker:
        return "FFT2"
    if "FFT1" in marker:
        return "FFT1"
    if src == "eblas_fft1_characterization":
        return "FFT1"
    return ""


def base_row(kind, src, marker, values):
    meta = metadata.get(src, {})

    n = first(values, "N", "LENGTH", "VECTOR_LENGTH")
    if not n and src == "eblas_packed_fft1_cp_validation":
        n = meta.get("PACKED_FFT1_CP_N", "")

    rows_value = first(values, "ROWS")
    cols_value = first(values, "COLS")

    if src == "eblas_packed_fft2_stage_validation":
        rows_value = rows_value or meta.get("PACKED_FFT2_STAGE_ROWS", "")
        cols_value = cols_value or meta.get("PACKED_FFT2_STAGE_COLS", "")

    if src == "eblas_packed_fft2_cp_validation":
        rows_value = rows_value or meta.get("PACKED_FFT2_CP_ROWS", "")
        cols_value = cols_value or meta.get("PACKED_FFT2_CP_COLS", "")

    row_stages = first(values, "ROW_STAGES")
    column_stages = first(values, "COLUMN_STAGES")
    stages = first(values, "STAGES")

    if not stages and row_stages and column_stages:
        stages = str(int(row_stages) + int(column_stages))

    return {
        "kind": kind,
        "source": src,
        "operation": operation_for(marker, src),
        "representation": representation_for(src),
        "direction": first(values, "DIRECTION"),
        "n": n,
        "rows": rows_value,
        "cols": cols_value,
        "profile": profile_for(src),
        "security_bearing": security_for(src),
        "row_stages": row_stages,
        "column_stages": column_stages,
        "stages": stages,
        "ciphertexts": "",
        "rotations": "",
        "public_multiplies": "",
        "cc_multiplies": "",
        "relinearizations": "",
        "additions": "",
        "transposes": first(values, "TRANSPOSES"),
        "levels_consumed": first(values, "LEVELS_CONSUMED"),
        "rel_l2": first(values, "REL_L2"),
        "max_abs": first(values, "MAX_ABS"),
        "inactive_max_abs": first(values, "INACTIVE_MAX_ABS"),
        "runtime_ns": measured_runtime(values),
        "status": first(values, "STATUS"),
        "raw_marker": marker,
    }


def structural_row(src, marker, values):
    row = base_row("STRUCTURAL", src, marker, values)

    direction = row["direction"]
    inverse = direction == "Inverse"

    if marker == "PACKED_FFT1_CP_CASE":
        stages = int(row["stages"])
        row.update(
            ciphertexts="1",
            rotations=str(2 * stages),
            public_multiplies=str(3 * stages + (1 if inverse else 0)),
            cc_multiplies="0",
            relinearizations="0",
            additions=str(2 * stages),
            transposes="0",
        )
        return row

    if marker == "PACKED_FFT2_STAGE_CASE":
        row.update(
            ciphertexts="1",
            rotations="2",
            public_multiplies="3",
            cc_multiplies="0",
            relinearizations="0",
            additions="2",
            transposes="0",
        )
        return row

    if marker == "PACKED_FFT2_CP_CASE":
        stages = int(row["row_stages"]) + int(row["column_stages"])
        row["stages"] = str(stages)
        row.update(
            ciphertexts="1",
            rotations=str(2 * stages),
            public_multiplies=str(3 * stages + (1 if inverse else 0)),
            cc_multiplies="0",
            relinearizations="0",
            additions=str(2 * stages),
            transposes="0",
        )
        return row

    return None


for raw_line in raw_path.read_text().splitlines():
    line = raw_line.strip()

    match = begin_re.match(line)
    if match:
        source = match.group(1)
        metadata.setdefault(source, {})
        continue

    if source is None:
        continue

    # Capture single-value provenance/status records.
    if line and " " not in line and "=" in line:
        key, value = line.split("=", 1)
        metadata[source][key] = value

    if line.startswith("FFT1_CHARACTERIZATION "):
        _, payload = line.split(" ", 1)
        values = {}

        for token in payload.split():
            if "=" in token:
                key, value = token.split("=", 1)
                values[key] = value

        measured = base_row(
            "MEASURED",
            source,
            "FFT1_CHARACTERIZATION",
            values,
        )

        measured["runtime_ns"] = ""
        measured["status"] = values.get("STATUS", "")
        rows.append(measured)
        continue

    if "CASE=" not in line:
        continue

    if "FFT1" not in line and "FFT2" not in line:
        continue

    marker, values = parse_tokens(line)

    if not marker:
        continue

    measured = base_row("MEASURED", source, marker, values)
    rows.append(measured)

    structural = structural_row(source, marker, values)
    if structural is not None:
        for field in (
            "levels_consumed",
            "rel_l2",
            "max_abs",
            "inactive_max_abs",
            "runtime_ns",
            "status",
        ):
            structural[field] = ""

        rows.append(structural)


with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()

    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})

print(f"FFT_EVIDENCE_ROWS={len(rows)}")
print(
    "FFT_EVIDENCE_MEASURED_ROWS="
    + str(sum(row["kind"] == "MEASURED" for row in rows))
)
print(
    "FFT_EVIDENCE_STRUCTURAL_ROWS="
    + str(sum(row["kind"] == "STRUCTURAL" for row in rows))
)

if not any(row["kind"] == "MEASURED" for row in rows):
    raise SystemExit("STOP: no measured FFT evidence rows produced")

if not any(row["kind"] == "STRUCTURAL" for row in rows):
    raise SystemExit("STOP: no structural FFT evidence rows produced")

scalar_fft1_rows = [
    row
    for row in rows
    if row["raw_marker"] == "FFT1_CHARACTERIZATION"
    and row["kind"] == "MEASURED"
]

if len(scalar_fft1_rows) != 4:
    raise SystemExit(
        f"STOP: expected 4 scalar FFT1 characterization rows, "
        f"found {len(scalar_fft1_rows)}"
    )

if not any(
    row["raw_marker"] == "PACKED_FFT1_CP_CASE"
    and row["kind"] == "MEASURED"
    for row in rows
):
    raise SystemExit("STOP: packed FFT1 measured evidence missing")

if not any(
    row["raw_marker"] == "PACKED_FFT2_CP_CASE"
    and row["kind"] == "MEASURED"
    for row in rows
):
    raise SystemExit("STOP: packed FFT2 measured evidence missing")
PY

printf '\n=== FFT SUMMARY ===\n'
cat "$SUMMARY"

printf '\n=== FFT METADATA ===\n'
cat "$META"

printf '\n=== ENGINEERING GATES ===\n'
cargo check --release --tests
cargo clippy --release --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check

printf '\nRAW_LOG=%s\n' "$RAW"
printf 'SUMMARY_CSV=%s\n' "$SUMMARY"
printf 'METADATA=%s\n' "$META"
printf 'EBLAS_FFT_EVIDENCE_STATUS=PASS\n'
