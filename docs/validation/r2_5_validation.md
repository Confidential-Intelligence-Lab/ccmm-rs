# R2.5 Validation Evidence

This document records the validation boundary for the coefficient-domain
CKKS/Grafting implementation developed during Roadmap 2.0, R2.5.

## Validation scope

R2.5 establishes encrypted, multi-level coefficient-domain CKKS-style
computation over RNS together with Grafting modulus/scale transitions.

The implementation currently uses coefficient-wise fixed-point encoding.
It does not yet implement the canonical CKKS embedding or SIMD slot
semantics. Those are Roadmap R2.6.

## Independently validated semantics

The following semantics have independent validation evidence:

- coefficient-wise fixed-point encoding and decoding;
- negacyclic polynomial multiplication;
- RLWE encryption and decryption;
- RNS representation and CRT reconstruction;
- modulus-chain level transitions;
- centered nearest-rounding RNS rescaling;
- CKKS scale evolution across multiplication and rescaling;
- Grafting rational rescaling and modulus transitions;
- encrypted multiply -> relinearize -> rescale;
- consecutive encrypted multiplication depths.

### Independent mathematical oracle

`tools/grafting_spec_oracle.py` implements an independent mathematical
reference using Python rational arithmetic.

It validates:

- centered representatives;
- signed nearest division;
- CKKS rescaling;
- negacyclic multiplication;
- modulus-chain evolution;
- scale evolution; and
- a deterministic depth-2 plaintext reference.

The oracle is intentionally independent of the Rust implementation.

### Rust/Python differential validation

`src/bin/grafting_vectors.rs` emits deterministic observations from the
Rust encrypted-computation path.

`tools/verify_grafting_vectors.py` independently reconstructs the
corresponding mathematical references and compares the decoded Rust
results against them.

This validates the numerical semantics of the Rust path without requiring
ciphertext equality.

### External Grafting cross-check

`tools/external_grafting_crosscheck.py` cross-checks overlapping semantics
against the external `Sirraya-graft-ckks` implementation.

External revision:

`7a893201f6ce89a89869d561b3bf5bdd009c1e36`

The cross-check uses deterministic Python RNG seed 42, matching the
external implementation's standalone validation harness.

The external cross-check covers:

- coefficient encoding;
- negacyclic plaintext semantics;
- RLWE encrypt/decrypt behavior;
- encrypted multiplication;
- Grafting modulus selection;
- rational Q/Q' rescaling; and
- decoded numerical product semantics.

The external implementation uses validation-only parameters and simplified
relinearization based directly on the secret-square polynomial. Therefore
it is a secondary semantic cross-check, not a normative oracle for the
ccmm-rs key-switching architecture.

## Internally validated implementation mechanisms

The following mechanisms have extensive Rust unit, differential, and
invariant testing but are not independently validated by the external
Grafting implementation:

- digit-decomposed RNS relinearization;
- mixed-gadget decomposition;
- hybrid key switching;
- nonzero hybrid evaluation-key noise accounting;
- helper-prime NTT acceleration;
- prepared/cached helper-prime evaluation-key operands;
- prime/power-of-two Grafting transitions; and
- resurrection infrastructure.

These mechanisms must not be described as externally cross-validated
solely on the basis of the R2.5f external experiment.

## Current cryptographic claim

At R2.5, ccmm-rs implements genuine RLWE encryption/decryption, RNS
representation, coefficient-wise CKKS-style fixed-point encoding/decoding,
scale tracking, rescaling, relinearization, Grafting transitions, and
multi-depth encrypted computation.

The current plaintext multiplication semantics are negacyclic polynomial
convolution.

The implementation must not yet be described as complete canonical CKKS.

## Known limitations before R2.6+

R2.5 does not establish:

- canonical CKKS complex-slot encoding/decoding;
- SIMD element-wise slot multiplication;
- automorphisms, rotations, or conjugation;
- Galois-key switching;
- realistic security-bearing parameter profiles;
- an independently estimated RLWE security level;
- production constant-time guarantees; or
- production-scale performance.

Canonical CKKS encoding/decoding and SIMD slots are R2.6.
Automorphisms and Galois operations are R2.7.
Level-aware key switching is R2.8.
Realistic parameters, security, precision, assurance, and broader
independent validation are R2.9.

## Validation principle

Validation compares mathematical semantics rather than ciphertext bytes.

Independent implementations may legitimately differ in randomness, key
representation, RNS basis, gadget decomposition, relinearization strategy,
and internal representation while implementing the same plaintext
semantics.

R2.5f therefore establishes independent evidence for the shared
coefficient-domain and Grafting semantics while explicitly preserving the
boundary between independently validated behavior and internally tested
implementation mechanisms.
