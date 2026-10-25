#!/usr/bin/env bash
set -euo pipefail

root="results/image_filter/source/benchmarks"
original_dir="$root/original"
source="$original_dir/fhe_fft_architecture_source.jpg"

mkdir -p "$original_dir"

url='https://commons.wikimedia.org/wiki/Special:Redirect/file/Fethiye_Camii%2C_exterior%2C_Istanbul%2C_Turkey_-_View_from_southwest_-_MSBZ004-H57-0795_-_Dumbarton_Oaks.jpg'

echo "=== FETCH FFT BENCHMARK SOURCE ==="
curl --fail --location \
    --retry 3 \
    --output "$source" \
    "$url"

echo
echo "=== PREPARE FFT BENCHMARK PYRAMID ==="
cargo run --locked --release \
    --bin prepare_fft_benchmark_images -- \
    "$source" \
    "$root"

echo
echo "=== SHA-256 ==="
(
    cd "$root"

    shasum -a 256 \
        original/fhe_fft_architecture_source.jpg \
        fhe_fft_architecture_gray_128x128.png \
        fhe_fft_architecture_gray_256x256.png \
        fhe_fft_architecture_gray_512x512.png \
        fhe_fft_architecture_gray_1024x1024.png \
        fhe_fft_architecture_gray_2048x2048.png \
        fhe_fft_architecture_gray_4096x4096.png \
        > SHA256SUMS

    cat SHA256SUMS
)

echo
echo "FFT_BENCHMARK_FETCH_STATUS=PASS"
