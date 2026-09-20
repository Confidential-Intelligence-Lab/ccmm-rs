use ccmm_rs::eblas::{
    select_cc_backend, GemmBackend, GemmShape, GemmSpec, MatrixShape, PrivacyMode,
};

fn cc_spec(m: usize, k: usize, n: usize) -> GemmSpec {
    GemmSpec::new(
        GemmShape::new(MatrixShape::new(m, k), MatrixShape::new(k, n)),
        PrivacyMode::Cc,
    )
}

fn main() {
    let cases = [
        (1, 1, 1, GemmBackend::CcScalar),
        (1, 2, 1, GemmBackend::CcStructured),
        (1, 4, 1, GemmBackend::CcStructured),
        (1, 8, 1, GemmBackend::CcStructured),
        (1, 16, 1, GemmBackend::CcStructured),
        (2, 3, 4, GemmBackend::CcStructured),
        (4, 8, 4, GemmBackend::CcStructured),
        (8, 4, 8, GemmBackend::CcStructured),
    ];

    println!("R3_5I_EBLAS_BACKEND_POLICY_VALIDATION_VERSION=1");
    println!("POLICY_SOURCE=R3.5h-research-4096-characterization");
    println!("K_EQ_1_POLICY=CcScalar");
    println!("K_GE_2_POLICY=CcStructured");
    println!("EXPLICIT_BACKEND_SELECTION_RETAINED=YES");

    for (m, k, n, expected) in cases {
        let actual = select_cc_backend(cc_spec(m, k, n));
        assert_eq!(actual, expected);
        println!("CASE={m}x{k}x{n},SELECTED={actual:?},STATUS=PASS");
    }

    println!("R3_5I_EBLAS_BACKEND_POLICY_STATUS=PASS");
}
