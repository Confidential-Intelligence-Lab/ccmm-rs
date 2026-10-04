# Security Policy

FHE-rs / ccmm-rs is research software for privacy-preserving and encrypted
computing. Security-relevant defects are treated as engineering defects even
when they do not affect functional correctness.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability, secret
exposure, cryptographic weakness, unsafe parameter assumption, or
supply-chain compromise.

Use GitHub private vulnerability reporting when it is available for this
repository. Otherwise, contact the repository maintainers privately through
the Confidential Intelligence Lab / repository organization channel.

A useful report includes:

- affected commit or release;
- affected component and execution path;
- reproduction instructions or proof of concept;
- expected and observed behavior;
- security or confidentiality impact;
- whether secret data, keys, randomness, or parameter selection is involved.

Do not include real secret keys, credentials, access tokens, private datasets,
or other sensitive information in a report.

## Scope

Security reports are particularly relevant for:

- CKKS parameter and state handling;
- encryption, decryption, evaluation keys, and key switching;
- randomness and secret generation;
- RNS, NTT, modulus switching, rescaling, and relinearization;
- packed-slot rotations and encrypted linear algebra;
- serialization or future external interfaces;
- accelerator, GPU, FPGA, PIM, or FFI boundaries;
- dependency and build-system compromise;
- accidental logging or disclosure of secret-dependent material.

## Research and validation parameter sets

Not every parameter set in this repository is intended to make a production
security claim.

Validation-only profiles, including profiles explicitly marked
`SECURITY_BEARING=false`, exist to test numerical semantics, representation,
depth, and encrypted execution paths. They must not be represented as
production cryptographic parameter sets.

Research profiles likewise require an explicit security analysis before use
in a production security context.

## Supported code

Security fixes are applied to the current development line unless a release
is explicitly documented as supported separately.

## Disclosure

Please allow the maintainers a reasonable opportunity to investigate and
prepare a correction before public disclosure.
