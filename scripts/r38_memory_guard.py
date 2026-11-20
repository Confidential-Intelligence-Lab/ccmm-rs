#!/usr/bin/env python3
"""R38 benchmark RSS watchdog for macOS.

This is a sampled memory guard, not a hard OS memory limit.
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def process_rss_bytes(pgid):
    """Sum RSS for processes in the guarded process group."""
    result = subprocess.run(
        ["ps", "-axo", "pid=,pgid=,rss="],
        capture_output=True,
        text=True,
        check=True,
    )

    total_kib = 0
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            _, group, rss_kib = map(int, fields)
        except ValueError:
            continue
        if group == pgid:
            total_kib += rss_kib

    return total_kib * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-gib", type=float, required=True)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not (0 < args.limit_gib <= 12):
        parser.error("limit must be > 0 and <= 12 GiB")
    if not (0.05 <= args.interval <= 2):
        parser.error("interval must be between 0.05 and 2 seconds")

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command")

    limit = int(args.limit_gib * 1024**3)
    peak = 0
    exceeded = False
    monitoring_error = None

    print(f"R38_3D_RSS_LIMIT_GIB={args.limit_gib:.3f}", flush=True)
    print(f"R38_3D_SAMPLE_INTERVAL_S={args.interval:.3f}", flush=True)

    child = subprocess.Popen(command, start_new_session=True)
    pgid = child.pid

    try:
        while True:
            try:
                rss = process_rss_bytes(pgid)
            except Exception as exc:
                monitoring_error = str(exc)
                break

            peak = max(peak, rss)

            if rss > limit:
                exceeded = True
                print("R38_3D_MEMORY_LIMIT_EXCEEDED=1", flush=True)
                break

            if child.poll() is not None:
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("R38_3D_INTERRUPTED=1", flush=True)
        monitoring_error = "interrupted"
    finally:
        if child.poll() is None and (exceeded or monitoring_error):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        returncode = child.wait()

    print(f"R38_3D_OBSERVED_PEAK_RSS_GIB={peak / 1024**3:.6f}")
    print(f"R38_3D_CHILD_EXIT_CODE={returncode}")

    if monitoring_error:
        print(f"R38_3D_MONITOR_ERROR={monitoring_error}")
        print("R38_3D_STATUS=FAIL")
        return 1

    if exceeded:
        print("R38_3D_STATUS=MEMORY_LIMIT_EXCEEDED")
        return 2

    if returncode != 0:
        print("R38_3D_STATUS=CHILD_FAILED")
        return 3

    print("R38_3D_STATUS=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
