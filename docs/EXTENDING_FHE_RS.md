# Extending FHE-rs

## New application

Prefer stable eBLAS/evaluator interfaces. Define threat/privacy model, public
and private inputs, parameter profile, keys/evaluation material, computation
graph, depth/level budget, cleartext oracle, tolerance, benchmark scope, and
retained evidence. Linear algebra should use eBLAS; a service such as proxy
re-encryption may use lower-level key/evaluator interfaces.

## New eBLAS operation

Keep mathematical operation, operand privacy, and execution backend separate.
Define semantics/shape/accounting first and reuse validated kernels where
possible. Test supported and unsupported privacy/backend combinations.

## New backend

Preserve the eBLAS contract; validate numerical equivalence; expose operation
counts; characterize the relevant frontier; retain explicit selection for
reproducibility; and document platform assumptions before policy dispatch.
Measure GPU/FPGA/packed backends before performance claims.

## New parameter profile

Record ring degree/slots, modulus chain, scale, secret/error distributions,
key-switch decomposition, auxiliary moduli, and estimator methodology if a
security claim is intended. Do not inherit `research-4096` security evidence.

## Future HE schemes and bootstrapping

Keep scheme-independent application/eBLAS semantics separate from
scheme-specific ciphertext/evaluator semantics and shared arithmetic. Integer/
discrete CKKS and additional HE schemes are future directions. Bootstrapping
should have an explicit semantic contract, key/parameter accounting, precision
characterization, depth-restoration tests, security assumptions, performance
measurements, and an application demonstrating need. Until then FHE-rs is
leveled CKKS.

## Resilience integration

Integrate with the existing NTT-resilience work rather than create a separate
fault-tolerance stack:

```text
fault -> modular arithmetic -> NTT -> RNS -> CKKS -> eBLAS/service -> application
```

Fault injection/detection/recovery hooks must not alter normal cryptographic
semantics when disabled.

## Documentation gate

Substantial extensions should update the README and relevant application,
eBLAS, assurance, security, reproducibility, and changelog documents. State
what was implemented, how to use it, what evidence supports it, and what is
not claimed.
