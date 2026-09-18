use ccmm_rs::ccmm::{ccmm, decrypt_ckks_matrix, encrypt_ckks_matrix_with_rng};
use ccmm_rs::ckks::{project_secret_key_to_low, CkksParameters};
use ccmm_rs::eval::MultiplicationKey;
use ccmm_rs::matrix::BatchMatrix;
use ccmm_rs::rlwe::SecretKey;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn main() {
    let params = CkksParameters::new(8, 2_147_483_647, 65_537, 65_537.0, 1);

    let mut key_rng = ChaCha20Rng::seed_from_u64(0xC11_0001);

    let high_secret = SecretKey::generate_with_rng(params.high_rlwe(), &mut key_rng);

    let low_secret = project_secret_key_to_low(params, &high_secret);

    // Column-major:
    //
    // M1 = [[1, 2],
    //       [3, 4]]
    //
    // M2 = [[5, 6],
    //       [7, 8]]
    let lhs = BatchMatrix::from_vec_column_major(2, 2, 1, vec![1.0, 3.0, 2.0, 4.0]);

    let rhs = BatchMatrix::from_vec_column_major(2, 2, 1, vec![5.0, 7.0, 6.0, 8.0]);

    let expected = BatchMatrix::from_vec_column_major(2, 2, 1, vec![19.0, 43.0, 22.0, 50.0]);

    let mut lhs_rng = ChaCha20Rng::seed_from_u64(0xC11_0002);

    let mut rhs_rng = ChaCha20Rng::seed_from_u64(0xC11_0003);

    let lhs_ct = encrypt_ckks_matrix_with_rng(params, &high_secret, &lhs, &mut lhs_rng);

    let rhs_ct = encrypt_ckks_matrix_with_rng(params, &high_secret, &rhs, &mut rhs_rng);

    let mut eval_rng = ChaCha20Rng::seed_from_u64(0xC11_0004);

    let multiplication_key = MultiplicationKey::generate_with_rng(
        params.high_rlwe(),
        &high_secret,
        65_536,
        &mut eval_rng,
    );

    let result = ccmm(params, &lhs_ct, &rhs_ct, &multiplication_key);

    let actual = decrypt_ckks_matrix(params, &low_secret, &result);

    let coordinates = [("00", 0, 0), ("01", 0, 1), ("10", 1, 0), ("11", 1, 1)];

    let mut max_abs_error = 0.0_f64;

    println!("RUST_CCMM_ORACLE_VERSION=1");
    println!("RUST_CCMM_ROWS=2");
    println!("RUST_CCMM_COLS=2");
    println!("RUST_CCMM_LEVEL={}", result.level());
    println!("RUST_CCMM_SCALE={:.17e}", result.scale());

    for (name, row, col) in coordinates {
        let expected_value = *expected.get(0, row, col);

        let actual_value = *actual.get(0, row, col);

        let error = (actual_value - expected_value).abs();

        max_abs_error = max_abs_error.max(error);

        println!("RUST_CCMM_EXPECTED_{name}={expected_value:.17e}");

        println!("RUST_CCMM_ACTUAL_{name}={actual_value:.17e}");

        println!("RUST_CCMM_ABS_ERROR_{name}={error:.17e}");
    }

    println!("RUST_CCMM_MAX_ABS_ERROR={max_abs_error:.17e}");

    let tolerance = 1.0e-2;

    println!("RUST_CCMM_TOLERANCE={tolerance:.17e}");

    if max_abs_error <= tolerance {
        println!("RUST_CCMM_STATUS=PASS");
    } else {
        println!("RUST_CCMM_STATUS=FAIL");
        std::process::exit(1);
    }
}
