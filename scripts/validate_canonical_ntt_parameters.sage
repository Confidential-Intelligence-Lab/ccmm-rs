from sage.all import Integer, factor, is_prime, proof, version

DEGREE = Integer(4096)

PARAMETERS = [
    ("32", Integer(4294828033), Integer(1953722822)),
    ("64", Integer(18446744073709436929), Integer(5975861664659593359)),
    (
        "128",
        Integer(1329227995784915872903807060279713793),
        Integer(84935201209993441529728139364756967),
    ),
]

print("R3_3D_0C_CANONICAL_NTT_PARAMETERS_VERSION=1")
print(f"SAGE_VERSION={version()}")
proof.arithmetic(True)
print("PRIMALITY_METHOD=SageMath is_prime with proof.arithmetic(True)")
print(f"DEGREE={DEGREE}")

all_pass = True

for width, q, psi in PARAMETERS:
    prime = bool(is_prime(q))
    compatible = ((q - 1) % (2 * DEGREE)) == 0
    root_n = pow(psi, DEGREE, q)
    root_2n = pow(psi, 2 * DEGREE, q)
    root_ok = root_n == q - 1 and root_2n == 1

    print()
    print(f"WIDTH={width}")
    print(f"MODULUS={q}")
    print(f"MODULUS_BITS={q.nbits()}")
    print(f"PSI={psi}")
    print(f"SAGE_PROVEN_PRIME={'true' if prime else 'false'}")
    print(f"NTT_COMPATIBLE={'true' if compatible else 'false'}")
    print(f"PSI_TO_N={root_n}")
    print(f"PSI_TO_2N={root_2n}")
    print(f"ROOT_ORDER_CHECK={'PASS' if root_ok else 'FAIL'}")
    print(f"Q_MINUS_1_FACTORIZATION={factor(q - 1)}")

    all_pass &= prime and compatible and root_ok

print()
print(f"R3_3D_0C_STATUS={'PASS' if all_pass else 'FAIL'}")

if not all_pass:
    raise SystemExit(1)
