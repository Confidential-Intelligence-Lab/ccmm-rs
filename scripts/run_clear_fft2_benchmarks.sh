#!/usr/bin/env bash
set -euo pipefail

root="results/image_filter/source/benchmarks"
out="results/image_filter/clear_fft2"

mkdir -p "$out"

cargo build --locked --release --bin clear_fft2_image

for dimension in 128 256 512 1024 2048 4096; do
    input="${root}/fhe_fft_architecture_gray_${dimension}x${dimension}.png"
    output="${out}/fhe_fft_architecture_fft2_${dimension}x${dimension}.png"
    log="${out}/fhe_fft_architecture_fft2_${dimension}x${dimension}.log"

    echo
    echo "=== CLEAR FFT2 ${dimension}x${dimension} ==="

    target/release/clear_fft2_image \
        "$input" \
        "$dimension" \
        "$output" \
        | tee "$log"
done

echo
echo "=== OUTPUT HASHES ==="

(
    cd "$out"
    shasum -a 256 *.png > SHA256SUMS
    cat SHA256SUMS
)
