//! R23.4C: native FFT3840 frequency-index coverage.
//!
//! This checks the proposed mixed-radix output indexing independently
//! of CKKS arithmetic. Numerical correctness is a separate regression.

const BRANCHES: usize = 15;
const LOCAL_FFT: usize = 256;
const WIDTH: usize = BRANCHES * LOCAL_FFT;

fn natural_frequency(branch: usize, local: usize) -> usize {
    assert!(branch < BRANCHES);
    assert!(local < LOCAL_FFT);
    branch + BRANCHES * local
}

#[test]
fn r23_4c_fft3840_frequency_indexing() {
    let mut visited = vec![false; WIDTH];

    for branch in 0..BRANCHES {
        for local in 0..LOCAL_FFT {
            let frequency = natural_frequency(branch, local);

            assert!(frequency < WIDTH);
            assert!(!visited[frequency], "duplicate frequency {frequency}");

            visited[frequency] = true;

            assert_eq!(frequency % BRANCHES, branch);
            assert_eq!(frequency / BRANCHES, local);
        }
    }

    assert!(visited.iter().all(|&seen| seen));

    println!("R23_4C_FREQUENCY_COUNT={WIDTH}");
    println!("R23_4C_BRANCH_COUNT={BRANCHES}");
    println!("R23_4C_LOCAL_FFT_SIZE={LOCAL_FFT}");
    println!("R23_4C_FREQUENCY_INDEXING_STATUS=PASS");
}
