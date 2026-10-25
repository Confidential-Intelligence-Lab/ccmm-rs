# FHE-rs v3.0: encrypted FFT experiment guide

This document records the experiment interfaces used for the R31.4/R31.5 evidence milestone. It does **not** present runtime or resident-memory figures. The research evidence is preserved locally in `results/eblas_fft_evidence/r31_5_frozen/`; generated evidence is not necessarily committed to Git.

## Scientific contract

The examples implement **encrypted-input, public-transform FFT**: the image samples are encrypted, while the mathematical FFT transform, public twiddle factors, masks, and index schedule are not secret. Encrypting twiddles is a separate future research question. The executed transform is separable, with packed radix-2 decimation-in-frequency (DIF) row/column stages and a bit-reversed physical output representation. The evaluator validates decrypted spectral output against a cleartext oracle and checks relative L2 and maximum absolute errors.

The application-facing stack is `image -> encode/encrypt -> encrypted FFT2 -> decrypt/decode -> cleartext spectral output`. This is an FFT-only example; it does **not** assert that an encrypted frequency-domain filter or encrypted inverse FFT was included in the large-image R31.5 run.

## FFT1 and FFT2 dimension ladder

All public scripts accept `preflight` and `run` (default: `preflight`). A successful preflight is **not** a numerical FFT result. An experiment is validated only when its encrypted executor runs and the expected PASS marker is observed.

| Transform | Script | Evidence and readiness |
|---|---|---|
| FFT1 | `scripts/run_fft1.sh` | Scalar FFT1 characterization and packed ciphertext FFT1 validator; two PASS markers required |
| FFT2 16 x 16 | `scripts/run_fft2_16.sh` | Execution contract not verified; run deliberately blocked |
| FFT2 64 x 64 | `scripts/run_fft2_64.sh` | `ckks_multi_tile_fft_validation`; fixed 64 x 64 shape, per-tile oracle comparisons and `MULTI_TILE_FFT_STATUS=PASS` |
| FFT2 128 x 128 | `scripts/run_fft2_128.sh` | No verified dimension-specific executor; run deliberately blocked |
| FFT2 256 x 256 | `scripts/run_fft2_256.sh` | Exact factored FFT2 path identified in `ckks_tiled_image_fft`; parameter/CLI contract still requires verification |
| FFT2 1024 x 1024 | `scripts/run_fft2_1k.sh` | 32-ciphertext encrypted FFT2; `R17_1K_STATUS=PASS` |
| FFT2 4096 x 4096 | `scripts/run_fft2_4k.sh` | 16 macrotiles, 512 ciphertexts, verified image hash; `R21_UHD_STATUS=PASS` |

The logical FFT2 image dimension, tile dimension, CKKS slot count, and number of ciphertexts are distinct quantities. The 64 x 64 multi-tile validator must not be presented as evidence of 128 x 128 support. The 256 x 256 research implementation is not yet a verified public run command.

**Publication boundary:** public examples omit runtime and resident-memory measurements; local raw logs may retain them for reproducibility. Numerical correctness does not imply security qualification.

## Prerequisites

From the repository root, use a pinned or compatible Rust toolchain, a locked build, and the research input:

```bash
cargo build --locked --release --bin ckks_packed_1k_fft_validation
cargo build --locked --release --bin ckks_packed_uhd_fft_validation
ls -l results/image_filter/source/benchmarks/fhe_fft_architecture_gray_4096x4096.png
```

The 4K runner enforces the SHA-256 of the baseline input. Running experiments writes to `results/`. Preserve the frozen `r31_5_frozen` directory rather than overwriting it. Avoid running other resource-intensive jobs concurrently for paper reproducibility. Parameter profiles in these runs are research profiles, not production-security configurations.

## Experiment index

| Experiment | Run command | Scope and validation |
|---|---|---|
| Scalar/packed FFT suite | `bash scripts/run_eblas_fft_evidence.sh` | FFT1 characterization; packed FFT1; FFT2 stage validation; full small packed FFT2; measured/structural evidence distinction |
| 1K preflight | `bash scripts/run_fft2_1k.sh preflight` | Verifies source-contract markers without encrypted execution |
| 1K full FFT2 | `FHE_PACKED_1K_NTT_RESIDENT=1 bash scripts/run_fft2_1k.sh run` | One complete 1024 x 1024 FFT2 using 32 ciphertexts, 32,768 slots each; decrypts and compares with cleartext FFT oracle |
| 4K full hierarchical FFT2 | `bash scripts/run_fft2_4k.sh` | Full 4096 x 4096 source image; 16 macrotiles of 1024 x 1024; 512 ciphertexts; decrypts and compares full spectrum |

The dedicated 1K entry point is `ckks_packed_1k_fft_validation`. The 4K hierarchical entry point is **`ckks_packed_uhd_fft_validation`** with its input image as the first argument. Despite the latter binary name, the R31.5 benchmark is the hierarchical FFT2 workload and must not be confused with a separately developed native UHD pipeline. Use only the dedicated hierarchical validator for the documented 4K workload.

## 1K reproduction

```bash
export FHE_PACKED_1K_NTT_RESIDENT=1
export R17_1K_LOCAL_PARALLELISM=2
export R17_1K_INTER_CT_PARALLELISM=4
export R17_1K_BENCHMARK=1
bash scripts/run_fft2_1k.sh preflight
bash scripts/run_fft2_1k.sh run
```

The public runner outputs a timestamped directory under `results/fft_v3/fft2_1k/<run-id>/`, recording `run.log`, `git_head.txt`, and `git_status.txt`. Historical research runners may record additional resource and summary files. The runner checks the expected degree, ciphertext count, depth/level contract, and status markers. The R31.4 profile is `research-65536-depth-aware`, degree 65536, 32 ciphertexts, 20 levels, and 29 distinct Galois keys. Its reported relative L2 and max-abs errors passed validation.

The earlier R21 1K reference and the dedicated R31.4 binary are not yet proven parameter/execution-contract equivalent. Do not publish a claimed 1K speedup solely from dividing those two execution times.

## 4K reproduction

Run the guarded wrapper, from the repository root:

```bash
bash scripts/run_fft2_4k.sh
```

The wrapper verifies the 4096 x 4096 image's SHA-256 against the frozen R21 baseline, refuses known partial-execution/equivalence modes, and builds/runs `ckks_packed_uhd_fft_validation` in release mode with the verified parallelism environment (`R21_UHD_LOCAL_PARALLELISM=8`, `R21_UHD_INTER_CT_PARALLELISM=4`, `FHE_PACKED_1K_NTT_RESIDENT=1`). It saves the log, status, configuration, image hash and Git provenance into a new (non-overwritten) output directory.

The R31.5 execution validated 512 ciphertexts and 16 packed macrotiles for the complete 4096 x 4096 input. The research profile must be labeled `SECURITY_BEARING=false`. The frozen result confirms complete execution and the same reported relative L2 and maximum absolute error as the previous R21 baseline, but does not attribute speedup to a single microoptimization.

## Techniques exercised or studied

- **Packing and decomposition:** tiles/macrotiles and CKKS slot layout; the full 4K image is not one monolithic ciphertext.
- **Separable transpose-free FFT2:** DIF row stages and strided column rotations with public masks/twiddles.
- **Prepared public operands:** reusable encoded/NTT-form plaintext operands; an earlier R19 microbenchmark established exact ciphertext equivalence and a prepared-operand amortization improvement.
- **NTT-resident execution:** keep compatible intermediate ciphertexts in the evaluation domain across operations when the selected executor supports it.
- **Prepared modular arithmetic and NTT butterflies:** separately differential-tested experimental optimizations. Do not claim every candidate is active in the large FFT2 executor absent a call-path inspection.
- **Parallelism:** the 1K and 4K experiment interfaces expose distinct local/inter-ciphertext parallelism variables. Report actual host, environment and provenance for any new campaign.

## Artifact and publication discipline

Local frozen package: `results/eblas_fft_evidence/r31_5_frozen/`; the package contains four baseline/optimized evidence sets, source snapshots, `working_tree.diff`, `summary.csv`, and SHA-256 manifest. Treat it as read-only; new measurements get new directories.

Paper claims must distinguish observed measurements from algorithmic operation counts, **FFT-only** from end-to-end setup/decryption, and functional correctness from security parameter qualification. Record the exact source tree, including uncommitted changes, and preserve the same input hash when claiming direct comparison. See [Reproducibility](REPRODUCIBILITY.md) and [Security and parameters](SECURITY_AND_PARAMETERS.md).
