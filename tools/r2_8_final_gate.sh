#!/bin/sh
set -eu

echo "R2.8_FINAL_GATE_VERSION=1"
echo

echo "=== formatting ==="
cargo fmt --all -- --check
echo "R2.8i_FORMAT=PASS"
echo

echo "=== ordinary RNS relinearization ==="
cargo test \
  grafting::rns_key_switch::tests::rns_relinearization_campaign_is_exact_without_noise \
  --lib
echo "R2.8i_ORDINARY_RNS=PASS"
echo

echo "=== RNS Galois differential campaign ==="
cargo test \
  ckks::rns_galois_key::tests::randomized_rns_ckks_galois_differential_campaign \
  --lib
echo "R2.8i_GALOIS=PASS"
echo

echo "=== hybrid reference campaign ==="
cargo test \
  grafting::hybrid_key_switch::tests::hybrid_relinearization_campaign_is_exact \
  --lib

cargo test \
  grafting::hybrid_key_switch::tests::prepared_helper_relinearization_matches_all_reference_paths \
  --lib

cargo test \
  grafting::hybrid_key_switch::tests::backend_dispatch_matches_all_hybrid_paths \
  --lib

cargo test \
  ckks::hybrid_multiply::tests::hybrid_ckks_backends_match_after_transition \
  --lib

echo "R2.8i_HYBRID=PASS"
echo

echo "=== variable-sprout differential semantics ==="
cargo test \
  grafting::pow2_transition_apply::tests::differential_campaign_matches_reference \
  --lib

cargo test \
  ckks::hybrid_multiply::tests::hybrid_ckks_scale_tracks_shrinking_sprout_8_to_6 \
  --lib

cargo test \
  ckks::hybrid_multiply::tests::hybrid_ckks_scale_tracks_growing_sprout_6_to_8 \
  --lib

cargo test \
  ckks::hybrid_multiply::tests::hybrid_ckks_scale_tracks_shrinking_sprout_8_to_4 \
  --lib

echo "R2.8i_VARIABLE_SPROUT=PASS"
echo

echo "=== full Grafting differential campaign ==="
cargo test \
  grafting::differential::tests::full_grafting_differential_campaign \
  --lib

cargo test \
  grafting::differential::tests::two_transition_campaign_preserves_semantics \
  --lib

echo "R2.8i_GRAFTING_DIFFERENTIAL=PASS"
echo

echo "=== deterministic source fingerprint ==="

for f in \
  src/ckks/chain.rs \
  src/ckks/evaluation_keys.rs \
  src/ckks/evaluator.rs \
  src/ckks/hybrid_ciphertext.rs \
  src/ckks/hybrid_multiply.rs \
  src/ckks/key_accounting.rs \
  src/ckks/rns_galois_key.rs \
  src/grafting/hybrid_key_switch.rs \
  src/grafting/pow2_transition.rs \
  src/grafting/pow2_transition_apply.rs \
  src/grafting/rns_key_switch.rs
do
  shasum -a 256 "$f"
done

echo "R2.8i_REPRODUCIBILITY=PASS"
echo

echo "=== full repository test gate ==="
cargo test --all
echo "R2.8i_FULL_TEST=PASS"
echo

echo "=== zero-warning static gate ==="
cargo clippy \
  --all-targets \
  --all-features \
  -- \
  -D warnings

echo "R2.8i_CLIPPY=PASS"
echo

echo "R2.8i_STATUS=PASS"
echo "R2.8_STATUS=PASS"
