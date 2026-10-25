#!/usr/bin/env python3
"""R30.13c.3: validate and consolidate CKKS paired benchmark evidence."""

import argparse
import csv
import datetime as dt
import math
import statistics
from pathlib import Path

EXPECTED = {
    4096: (3, 30),
    8192: (5, 30),
    16384: (9, 30),
    32768: (13, 20),
    65536: (18, 10),
}

REUSES = (1, 2, 4, 8, 16, 32)

PAIRED_COLUMNS = [
    "degree", "limbs", "trial", "order",
    "ordinary_ms", "prepared_ms", "paired_speedup",
]

AMORT_COLUMNS = [
    "degree", "limbs", "reuses",
    "plan_median_ms", "preparation_median_ms",
    "ordinary_median_ms", "prepared_median_ms",
    "modeled_speedup",
]


def read_csv(path, expected_columns):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == expected_columns, (
            f"Unexpected schema: {path}: {reader.fieldnames}"
        )
        rows = list(reader)

    assert rows, f"Empty CSV: {path}"
    return rows


def close(actual, expected, label, rel=2e-6, abs_tol=2e-6):
    assert math.isfinite(actual), f"Non-finite {label}"
    assert math.isclose(
        actual, expected, rel_tol=rel, abs_tol=abs_tol
    ), f"{label}: {actual} != {expected}"


def p95(values):
    ordered = sorted(values)
    index = math.ceil(len(ordered) * 0.95) - 1
    return ordered[index]


def validate_profile(degree, paired_path, amort_path):
    limbs, trials = EXPECTED[degree]

    paired = read_csv(paired_path, PAIRED_COLUMNS)
    amort = read_csv(amort_path, AMORT_COLUMNS)

    assert len(paired) == trials, (
        f"Degree {degree}: expected {trials} trials, got {len(paired)}"
    )
    assert len(amort) == len(REUSES)

    ordinary = []
    prepared = []
    speedups = []

    for index, row in enumerate(paired):
        assert int(row["degree"]) == degree
        assert int(row["limbs"]) == limbs
        assert int(row["trial"]) == index

        expected_order = (
            "ordinary_first" if index % 2 == 0
            else "prepared_first"
        )
        assert row["order"] == expected_order

        o = float(row["ordinary_ms"])
        p = float(row["prepared_ms"])
        s = float(row["paired_speedup"])

        assert math.isfinite(o) and o > 0
        assert math.isfinite(p) and p > 0

        close(s, o / p, f"{degree} trial {index} speedup")

        ordinary.append(o)
        prepared.append(p)
        speedups.append(s)

    ordinary_median = statistics.median(ordinary)
    prepared_median = statistics.median(prepared)
    paired_median = statistics.median(speedups)

    setup_values = set()
    amortization = {}

    for row, reuse in zip(amort, REUSES):
        assert int(row["degree"]) == degree
        assert int(row["limbs"]) == limbs
        assert int(row["reuses"]) == reuse

        plan = float(row["plan_median_ms"])
        preparation = float(row["preparation_median_ms"])

        assert math.isfinite(plan) and plan > 0
        assert math.isfinite(preparation) and preparation > 0

        close(
            float(row["ordinary_median_ms"]),
            ordinary_median,
            f"{degree} ordinary median",
        )
        close(
            float(row["prepared_median_ms"]),
            prepared_median,
            f"{degree} prepared median",
        )

        setup_values.add((plan, preparation))

        modeled = (
            reuse * ordinary_median
            / (plan + preparation + reuse * prepared_median)
        )

        close(
            float(row["modeled_speedup"]),
            modeled,
            f"{degree} reuse {reuse}",
        )

        amortization[reuse] = modeled

    assert len(setup_values) == 1, (
        f"Degree {degree}: inconsistent setup measurements"
    )

    plan, preparation = next(iter(setup_values))

    return {
        "degree": degree,
        "limbs": limbs,
        "trials": trials,
        "ordinary_median_ms": ordinary_median,
        "ordinary_p95_ms": p95(ordinary),
        "prepared_median_ms": prepared_median,
        "prepared_p95_ms": p95(prepared),
        "execution_speedup": ordinary_median / prepared_median,
        "paired_speedup_median": paired_median,
        "plan_median_ms": plan,
        "preparation_median_ms": preparation,
        "setup_median_sum_ms": plan + preparation,
        "amortization": amortization,
        "paired_source": str(paired_path),
        "amortization_source": str(amort_path),
    }


def discover(root):
    selected = {}

    for degree, (limbs, _) in EXPECTED.items():
        name = f"paired_cp_n{degree}_l{limbs}.csv"

        candidates = sorted(root.glob(f"*/{name}"))

        assert candidates, f"Missing paired evidence for degree {degree}"

        valid_pairs = []

        for paired in candidates:
            amort = paired.with_name(
                f"amortization_n{degree}_l{limbs}.csv"
            )

            if amort.is_file():
                valid_pairs.append((paired, amort))

        assert valid_pairs, (
            f"No complete paired/amortization evidence for {degree}"
        )

        # Use the latest timestamped complete run.
        selected[degree] = valid_pairs[-1]

    return selected


def write_summary(rows, output):
    columns = [
        "degree", "limbs", "trials",
        "ordinary_median_ms", "ordinary_p95_ms",
        "prepared_median_ms", "prepared_p95_ms",
        "execution_speedup", "paired_speedup_median",
        "plan_median_ms", "preparation_median_ms",
        "setup_median_sum_ms",
        "speedup_reuse_1", "speedup_reuse_2",
        "speedup_reuse_4", "speedup_reuse_8",
        "speedup_reuse_16", "speedup_reuse_32",
        "security_classification",
        "security_qualified",
        "paired_source", "amortization_source",
    ]

    with output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            record = {
                key: row[key]
                for key in columns if key in row
            }

            for reuse in REUSES:
                record[f"speedup_reuse_{reuse}"] = (
                    f"{row['amortization'][reuse]:.9f}"
                )

            record["security_classification"] = "RESEARCH_ONLY"
            record["security_qualified"] = "false"

            writer.writerow(record)


def write_report(rows, output):
    lines = [
        "# R30.13c.3 — CKKS Prepared CP Evidence",
        "",
        f"Generated (UTC): {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "Five implemented CKKS research profiles, degrees 4096–65536.",
        "All configurations are RESEARCH_ONLY and are not security-qualified.",
        "",
        "The report validates CSV consistency and arithmetic relationships.",
        "Cryptographic correctness PASS status was established by the",
        "corresponding benchmark runs; it is not independently retested here.",
        "",
        "## Consolidated measurements",
        "",
        "| Degree | Limbs | Trials | Ordinary ms | Prepared ms | Execution speedup | Single-use speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['degree']} | {row['limbs']} | {row['trials']} "
            f"| {row['ordinary_median_ms']:.3f} "
            f"| {row['prepared_median_ms']:.3f} "
            f"| {row['execution_speedup']:.3f}x "
            f"| {row['amortization'][1]:.3f}x |"
        )

    lines += [
        "",
        "## Methodology",
        "",
        "- Ordinary and prepared multiplication were measured in alternating order.",
        "- Raw per-trial samples were retained in the source CSV files.",
        "- Medians use the conventional midpoint for even sample counts.",
        "- P95 uses the nearest-rank estimator.",
        "- Setup is the sum of separately measured preparation medians.",
        "- Modeled amortization charges setup once per reuse group.",
        "- Peak RSS is not included because it is not present in the CSV inputs.",
        "- Results are not hardware-independent performance guarantees.",
        "",
        "## Provenance",
        "",
    ]

    for row in rows:
        lines += [
            f"### Degree {row['degree']}",
            "",
            f"- Paired samples: `{row['paired_source']}`",
            f"- Amortization: `{row['amortization_source']}`",
            "",
        ]

    with output.open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/benchmarks/r30_13c2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    assert args.input_root.is_dir(), "Input root missing"
    assert not args.output_dir.exists(), (
        "Output directory already exists; refusing overwrite"
    )

    selected = discover(args.input_root)

    rows = []

    for degree, (paired, amort) in selected.items():
        result = validate_profile(degree, paired, amort)
        rows.append(result)

        print(
            f"R30_13C3_PROFILE_STATUS=PASS "
            f"DEGREE={degree} "
            f"SPEEDUP={result['execution_speedup']:.6f} "
            f"SINGLE_USE={result['amortization'][1]:.6f}"
        )

    args.output_dir.mkdir(parents=True)

    summary = args.output_dir / "ckks_prepared_cp_summary.csv"
    report = args.output_dir / "ckks_prepared_cp_report.md"

    write_summary(rows, summary)
    write_report(rows, report)

    print(f"R30_13C3_SUMMARY={summary}")
    print(f"R30_13C3_REPORT={report}")
    print("R30_13C3_STATUS=PASS")


if __name__ == "__main__":
    main()
