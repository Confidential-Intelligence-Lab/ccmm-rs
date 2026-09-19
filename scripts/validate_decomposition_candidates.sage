from sage.all import Integer, factor, is_prime, proof, version

DEGREE = Integer(4096)

BASIS_32 = [
    Integer(1073692673),
    Integer(1073668097),
    Integer(1073651713),
    Integer(1073643521),
]

BASIS_64 = [
    Integer(1152921504606830593),
    Integer(1152921504606748673),
]

BASIS_128 = [
    Integer(1329227995784915872903807060279713793),
]

BASES = [
    ("W32_X4", 32, BASIS_32),
    ("W64_X2", 64, BASIS_64),
    ("W128_X1", 128, BASIS_128),
]

proof.arithmetic(True)

print("R3_3D_1B_DECOMPOSITION_VALIDATION_VERSION=1")
print(f"SAGE_VERSION={version()}")
print("PRIMALITY_METHOD=SageMath is_prime with proof.arithmetic(True)")
print(f"DEGREE={DEGREE}")
print("TARGET_LOGICAL_MODULUS_BITS=120")
print()

all_pass = True
composite_bits = []

for name, word_bits, basis in BASES:
    composite = Integer(1)
    basis_pass = True

    print(f"BASIS={name}")
    print(f"PHYSICAL_WORD_BITS={word_bits}")
    print(f"PHYSICAL_LIMBS={len(basis)}")

    for index, q in enumerate(basis):
        prime = bool(is_prime(q))
        compatible = ((q - 1) % (2 * DEGREE)) == 0

        print(f"LIMB_{index}_MODULUS={q}")
        print(f"LIMB_{index}_BITS={q.nbits()}")
        print(f"LIMB_{index}_SAGE_PROVEN_PRIME={'true' if prime else 'false'}")
        print(f"LIMB_{index}_NTT_COMPATIBLE={'true' if compatible else 'false'}")
        print(f"LIMB_{index}_Q_MINUS_1_FACTORIZATION={factor(q - 1)}")

        basis_pass &= prime and compatible
        composite *= q

    print(f"COMPOSITE_MODULUS={composite}")
    print(f"COMPOSITE_MODULUS_BITS={composite.nbits()}")
    print(f"BASIS_STATUS={'PASS' if basis_pass else 'FAIL'}")
    print()

    composite_bits.append(composite.nbits())
    all_pass &= basis_pass

min_bits = min(composite_bits)
max_bits = max(composite_bits)

print(f"MIN_COMPOSITE_BITS={min_bits}")
print(f"MAX_COMPOSITE_BITS={max_bits}")
print(f"COMPOSITE_BIT_SPREAD={max_bits - min_bits}")

equal_capacity = (max_bits - min_bits) == 0
print(f"EQUAL_CAPACITY_GATE={'PASS' if equal_capacity else 'FAIL'}")

all_pass &= equal_capacity

print(f"R3_3D_1B_STATUS={'PASS' if all_pass else 'FAIL'}")

if not all_pass:
    raise SystemExit(1)
