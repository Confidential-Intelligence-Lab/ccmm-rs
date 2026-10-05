#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_fft_milestone.sh" 128 64 exact_128x128_tile_64
