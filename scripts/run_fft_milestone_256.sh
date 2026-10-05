#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_fft_milestone.sh" 256 64 exact_256x256_tile_64
