# RNS Architecture

**Status:** Roadmap 3 design contract  
**Scope:** generalized RNS representation for CKKS, composite scaling, bounded key switching, and Grafting.

## 1. Design principle

`ccmm-rs` separates the **logical ciphertext modulus** from its **physical residue representation**.

A CKKS computation is defined in terms of logical modulus levels. Those levels may be represented by one or more physical RNS limbs. Grafting adds separately managed modulus material without changing the meaning of a CKKS logical level. Arbitrary-precision arithmetic is the exact reference/oracle layer; it is not the intended hot execution path.

The architecture therefore has four distinct layers:

```text
CKKS semantics
    |
logical modulus levels
    |
composite-level / Grafting policy
    |
physical RNS basis
    |
32 / 64 / 128-bit limb arithmetic and NTT
```

`BigInt` / `BigUint` sit beside this path as exact CRT, differential-validation, and accounting oracles.

## 2. Physical limb

A **physical limb** is one residue polynomial modulo one pairwise-coprime modulus `q_i`.

A physical limb is the unit on which native modular arithmetic and an NTT backend operate. It is not, by itself, a CKKS level.

Required physical word classes are:

```text
32-bit limb:   modulus represented in u32
64-bit limb:   modulus represented in u64
128-bit limb:  modulus represented in u128
```

The corresponding exact multiplication requirements are:

```text
u32  x u32  -> u64 intermediate
u64  x u64  -> u128 intermediate
u128 x u128 -> 256-bit intermediate
```

The initial 128-bit correctness backend may use arbitrary precision as its multiplication/reduction oracle. A later optimized implementation may use an explicit `U256`, Montgomery reduction, Barrett reduction, or another verified equivalent.

## 3. Word width

**Word width** is a property of physical modular arithmetic, not of CKKS semantics.

Changing word width may change:

- the number of physical primes;
- modular multiplication and reduction;
- NTT implementation;
- memory layout and bandwidth;
- accelerator suitability;
- key and ciphertext storage.

Changing word width must not, by itself, change the logical plaintext computation.

The stable existing 64-bit implementation remains the baseline while 32-bit and 128-bit arithmetic are introduced incrementally. The repository must not require an all-at-once generic rewrite of every RNS type.

## 4. Physical RNS basis

A **physical RNS basis** is an ordered set of pairwise-coprime physical moduli:

```text
Q = {q_0, q_1, ..., q_k}
```

A polynomial represented in this basis contains one residue polynomial for every `q_i`.

The physical basis owns:

- residue storage;
- limb-local arithmetic;
- basis extension and basis reduction;
- NTT plans for its physical moduli;
- exact association between residue limbs and moduli.

The physical basis does **not** determine where CKKS logical level boundaries occur.

The composite integer

```text
Q = product(q_i)
```

may exceed `u128`. Exact composite-modulus construction, CRT reconstruction, and centered representatives therefore use the arbitrary-precision reference layer when whole-`Q` arithmetic is required.

## 5. Logical CKKS level

A **logical CKKS level** is the unit consumed by one logical rescale transition.

A level contains one or more physical RNS moduli:

```text
L_l = {q_l,0, q_l,1, ..., q_l,t_l-1}

Q_l = product_j q_l,j
```

The composition degree `t_l` is independent of the physical word width.

Consequently:

```text
one physical prime != necessarily one CKKS level
```

A conventional chain is the special case `t_l = 1`.

A composite-scaling chain may use, for example:

```text
32-bit physical basis:
[q00 q01] [q10 q11] [q20 q21]
    L0        L1        L2

64-bit physical basis:
[q0] [q1] [q2]
 L0   L1   L2
```

Both may implement the same logical CKKS level structure.

A CKKS level transition consumes a **logical group**, not an assumed single prime.

## 6. Composite modulus chain

A **composite modulus chain** binds an ordered physical RNS basis to explicit logical-level boundaries.

Conceptually it records:

```text
physical basis:
[q0, q1, q2, q3, q4, q5]

logical groups:
[0..2], [2..4], [4..6]
```

It owns:

- the mapping from physical moduli to logical levels;
- the active physical basis at each logical level;
- the composite modulus of each logical level;
- the physical moduli removed by a logical rescale;
- scale-transition metadata.

It does not own limb arithmetic.

The required invariant is:

> Equivalent logical chains may use different physical decompositions while preserving CKKS semantics within the scheme's approximation error.

## 7. Grafting sprout

A **Grafting sprout** is modulus material managed independently from the ordinary CKKS logical-level chain.

A sprout may contain one or more physical factors, but it is **not** a synonym for a composite CKKS level.

The distinction is:

```text
physical limb
    one native residue modulus

logical CKKS level
    one or more ordinary physical limbs consumed as one rescale unit

Grafting sprout
    independently managed modulus material governed by a Grafting transition
```

Conceptually:

```text
GraftedBasis
    |
    +-- ordinary basis
    |      |
    |      +-- logical CKKS level groups
    |             |
    |             +-- physical RNS limbs
    |
    +-- sprout
           |
           +-- independently managed physical factor(s)
```

Grafting transitions may activate, consume, replace, or resurrect sprout material according to Grafting policy. They must not silently redefine CKKS logical-level boundaries.

## 8. Transition ownership

Transitions are owned by the layer whose semantics they change.

| Transition | Owning layer |
|---|---|
| modular add/sub/mul | physical limb |
| NTT / inverse NTT | physical limb / NTT backend |
| CRT reconstruction | exact reference layer |
| basis extension/drop | physical RNS basis |
| bounded gadget decomposition | active composite RNS space |
| key switching / relinearization | cryptographic evaluation layer |
| logical rescale | composite CKKS chain |
| sprout transition | Grafting |
| CKKS scale/level metadata | CKKS |

A logical rescale may cause several physical limbs to disappear. That does not make basis dropping itself a CKKS semantic operation; the composite chain selects the group, and the RNS layer executes the corresponding physical transition.

## 9. Bounded key switching

Bounded-base gadget decomposition operates over the active composite modulus and therefore must not assume:

- the composite modulus fits in `u128`;
- one physical prime equals one logical level;
- one particular physical limb width.

The exact decomposition/reference path may use arbitrary precision. Actual gadget digits remain bounded signed integers determined by the configured radix.

Evaluation-key arithmetic is performed over the active physical RNS basis. Logical-level grouping is metadata above this arithmetic.

## 10. Composite rescaling

Rescaling is generalized from:

```text
drop one physical prime
```

to:

```text
consume one logical CKKS level
    -> remove all physical moduli in that level group
    -> update scale by the group's composite modulus
```

The implementation should remain RNS-native where possible. Full arbitrary-precision reconstruction is a reference oracle, not a required production rescale mechanism.

Sequential removal of the physical factors of a composite level is permitted only when it implements the defined logical transition and is validated against the exact reference semantics.

## 11. Width independence and validation contract

For equivalent logical parameters, 32-, 64-, and 128-bit physical representations should be different implementations of the same logical computation.

The minimum differential gates are:

```text
modular arithmetic:
    add / sub / neg / mul / pow / inverse
        == BigUint oracle

RNS:
    decomposition -> exact CRT reconstruction

polynomial arithmetic:
    RNS/NTT result == exact reference result modulo Q

CKKS:
    equivalent logical chains agree within the declared approximation tolerance

level transitions:
    consume identical logical levels even when the number of physical limbs differs

Grafting:
    preserves its transition semantics independently of physical word width
```

No security equivalence follows automatically from arithmetic equivalence. Every concrete parameterization must retain explicit security accounting and, where claimed, estimator validation.

## 12. Implementation order

The architecture is introduced incrementally:

```text
R3.3c.1  arbitrary-precision CRT reference layer             DONE
R3.3c.2  arbitrary-precision bounded decomposition           DONE
R3.3c.3  wide CKKS reconstruction/decode                     DONE

R3.3c.4  physical 32/64/128-bit limb arithmetic              NEXT
R3.3c.5  explicit logical/composite CKKS level abstraction
R3.3c.6  composite rescaling
R3.3c.7  generalized Grafting integration
R3.3c.8  cross-width differential validation
```

The existing 64-bit path remains operational throughout this sequence.

## 13. Non-goals for R3.3c

R3.3c does not require:

- replacing all existing types with generic types in one refactor;
- using arbitrary precision in performance-critical NTT arithmetic;
- treating Grafting sprouts as CKKS levels;
- claiming 32-, 64-, and 128-bit decompositions have identical performance;
- claiming a parameter set is security-bearing merely because its arithmetic is correct;
- optimizing the 128-bit backend before a correct differential reference exists.

## 14. Architectural invariant

The central invariant for the generalized implementation is:

> **CKKS defines the logical computation; the composite chain defines logical modulus consumption; Grafting defines independently managed modulus transitions; RNS defines the physical residue representation; and the limb backend defines how each physical modulus is executed.**

This separation is the contract that allows `ccmm-rs` to vary residue width, composite-scaling strategy, Grafting policy, and execution provider without redefining the encrypted computation.
