from estimator import *
from sage.all import oo

SIGMA = 3.19
N = 4096

CASES = [
    {
        "level": 0,
        "q": 40564045502413384804664457166849,
        "rlwe_eval_key_samples": 6,
    },
    {
        "level": 1,
        "q": 1180580361798806077441,
        "rlwe_eval_key_samples": 4,
    },
    {
        "level": 2,
        "q": 34359697409,
        "rlwe_eval_key_samples": 2,
    },
]

for case in CASES:
    level = case["level"]
    q = case["q"]
    rlwe_samples = case["rlwe_eval_key_samples"]

    print("=" * 80)
    print(f"LEVEL={level}")
    print(f"N={N}")
    print(f"Q={q}")
    print(f"SIGMA={SIGMA}")
    print("SECRET=uniform-ternary")
    print(f"OBSERVED_RLWE_EVALUATION_KEY_SAMPLES={rlwe_samples}")
    print("ESTIMATOR_M=oo")
    print("COST_MODEL=RC.MATZOV")
    print(
        "NOTE=RLWE evaluation-key sample count is recorded but is not "
        "mapped directly to scalar LWE m"
    )
    print("=" * 80)

    params = LWE.Parameters(
        n=N,
        q=q,
        Xs=ND.UniformMod(3),
        Xe=ND.DiscreteGaussian(SIGMA),
        m=oo,
    )

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

    try:
        print("\n--- primal_hybrid ---")
        print(
            LWE.primal_hybrid(
                params,
                red_cost_model=RC.MATZOV,
            )
        )
    except Exception as exc:
        print(
            f"PRIMAL_HYBRID_ERROR={type(exc).__name__}: {exc}"
        )
