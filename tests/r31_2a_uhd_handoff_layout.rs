// R31.2A: complete UHD FFT3840 -> FFT2160 spectral layout contract.
// No encryption, key generation, or FFT evaluation.

use ccmm_rs::eblas::encrypted_fft2160::Fft135Plan;

const ROWS: usize = 2160;
const COLS: usize = 3840;
const BATCHES: usize = 135;
const GROUPS: usize = 2;
const TILES_PER_CT: usize = 8;
const ACTIVE_TILES: usize = 15;
const TILE_ROWS: usize = 16;
const TILE_COLS: usize = 256;
const TILE_SLOTS: usize = TILE_ROWS * TILE_COLS;
const CT_SLOTS: usize = TILES_PER_CT * TILE_SLOTS;

#[test]
fn r31_2a_uhd_handoff_layout() {
    assert_eq!(BATCHES * TILE_ROWS, ROWS);
    assert_eq!(ACTIVE_TILES * TILE_COLS, COLS);
    assert_eq!(GROUPS * CT_SLOTS, 65536);

    let mut seen = vec![false; ROWS * COLS];
    let mut count = 0usize;

    for vertical_branch in 0..BATCHES {
        let k_cross = Fft135Plan::natural_frequency(vertical_branch);

        for group in 0..GROUPS {
            let active_tiles = if group == 0 { 8 } else { 7 };

            for tile in 0..active_tiles {
                let horizontal_branch = group * TILES_PER_CT + tile;

                for local_row in 0..TILE_ROWS {
                    let ky = k_cross + BATCHES * local_row;

                    for local_frequency in 0..TILE_COLS {
                        let kx = horizontal_branch + ACTIVE_TILES * local_frequency;

                        let slot = tile * TILE_SLOTS + local_row * TILE_COLS + local_frequency;

                        assert!(slot < CT_SLOTS);
                        assert!(ky < ROWS);
                        assert!(kx < COLS);

                        let index = ky * COLS + kx;

                        assert!(!seen[index], "duplicate spectral bin ky={ky} kx={kx}");

                        seen[index] = true;
                        count += 1;
                    }
                }
            }
        }
    }

    assert_eq!(count, ROWS * COLS);
    assert!(seen.iter().all(|&present| present));

    println!("R31_2A_ROWS={ROWS}");
    println!("R31_2A_COLS={COLS}");
    println!("R31_2A_SPECTRAL_BINS={count}");
    println!("R31_2A_CIPHERTEXTS={}", BATCHES * GROUPS);
    println!("R31_2A_STATUS=PASS");
}
