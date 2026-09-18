from math import prod, log2

profiles = {
    "research-4096": {
        "N": 4096,
        "scale_bits": 37,
        "moduli": [
            0x0ffffee001,
            0x0ffffc4001,
            0x1ffffe0001,
        ],
    },
    "research-8192": {
        "N": 8192,
        "scale_bits": 40,
        "moduli": [
            0x07fffffd8001,
            0x07fffffc8001,
            0x0fffffffc001,
            0x0ffffff6c001,
            0x0fffffebc001,
        ],
    },
    "research-16384": {
        "N": 16384,
        "scale_bits": 45,
        "moduli": [
            0x0fffffffd8001,
            0x0fffffffa0001,
            0x0fffffff00001,
            0x1fffffff68001,
            0x1fffffff50001,
            0x1ffffffee8001,
            0x1ffffffea0001,
            0x1ffffffe88001,
            0x1ffffffe48001,
        ],
    },
}

table_52_128_ternary = {
    4096: 106,
    8192: 214,
    16384: 430,
}

for name, p in profiles.items():
    Q = prod(p["moduli"])
    exact_log_q = log2(Q)
    table_bound = table_52_128_ternary[p["N"]]

    print(f"PROFILE={name}")
    print(f"N={p['N']}")
    print(f"SLOTS={p['N']//2}")
    print(f"SECRET=uniform-ternary")
    print(f"ERROR=discrete-gaussian")
    print(f"SIGMA=3.19")
    print(f"TARGET_CLASSICAL_BITS=128")
    print(f"SCALE_BITS={p['scale_bits']}")
    print(f"LIMB_BITS={','.join(str(q.bit_length()) for q in p['moduli'])}")
    print(f"Q={Q}")
    print(f"Q_BIT_LENGTH={Q.bit_length()}")
    print(f"LOG2_Q={exact_log_q:.15f}")
    print(f"TABLE_5_2_MAX_LOG2_Q={table_bound}")
    print(f"TABLE_5_2_MARGIN_BITS={table_bound-exact_log_q:.6f}")
    print()
