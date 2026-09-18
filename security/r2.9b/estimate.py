from estimator import *
from sage.all import oo

SIGMA = 3.19

profiles = [
    (
        "research-4096",
        4096,
        649033470896967801447398927572993,
    ),
    (
        "research-8192",
        8192,
        421249101157150430150591791601812858371395928330411389778873040897,
    ),
    (
        "research-16384",
        16384,
        709803428473824464899716522243808726188912609449487857793115029420083157504007498781933901411951056604199361914151348163758737883137,
    ),
]

for name, n, q in profiles:
    print("=" * 72)
    print(f"PROFILE={name}")
    print(f"N={n}")
    print(f"Q={q}")
    print(f"SIGMA={SIGMA}")
    print("SECRET=uniform-ternary")
    print("COST_MODEL=RC.MATZOV")
    print("=" * 72)

    params = LWE.Parameters(
        n=n,
        q=q,
        Xs=ND.UniformMod(3),
        Xe=ND.DiscreteGaussian(SIGMA),
        m=oo,
    )

    print(params)

    print("\n--- primal_usvp ---")
    print(
        LWE.primal_usvp(
            params,
            red_cost_model=RC.MATZOV,
        )
    )

    print("\n--- primal_bdd ---")
    print(
        LWE.primal_bdd(
            params,
            red_cost_model=RC.MATZOV,
        )
    )

    print("\n--- dual_hybrid ---")
    print(
        LWE.dual_hybrid(
            params,
            red_cost_model=RC.MATZOV,
        )
    )

    if n <= 2**14:
        print("\n--- primal_hybrid ---")
        print(
            LWE.primal_hybrid(
                params,
                red_cost_model=RC.MATZOV,
            )
        )

    print()
