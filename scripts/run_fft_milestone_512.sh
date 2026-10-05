#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_fft_milestone.sh" 512 64 exact_512x512_tile_64
