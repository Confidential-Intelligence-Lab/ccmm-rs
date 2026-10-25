// R30_5_OPTIMIZED_UHD_LAYOUT
// R29 FFT3840 output representation, extended to full UHD FFT2.
// This test performs no encryption, key generation, or FFT evaluation.

const ROWS: usize = 2160;
const COLS: usize = 3840;
const TILE_ROWS: usize = 16;
const TILE_COLS: usize = 256;
const CROSS_ROWS: usize = ROWS / TILE_ROWS;
const H_BRANCHES: usize = COLS / TILE_COLS;
const TILE_SLOTS: usize = TILE_ROWS * TILE_COLS;
const CT_SLOTS: usize = 32768;
const TILES_PER_CT: usize = CT_SLOTS / TILE_SLOTS;
const CT_PER_BATCH: usize = H_BRANCHES.div_ceil(TILES_PER_CT);
const RADICES: [usize; 4] = [5, 3, 3, 3];

fn natural_cross_index(mut index: usize) -> usize {
    assert!(index < CROSS_ROWS);
    let mut digits = [0usize; 4];
    for (position, &radix) in RADICES.iter().rev().enumerate() {
        digits[3 - position] = index % radix;
        index /= radix;
    }
    assert_eq!(index, 0);
    digits[0] + RADICES[0] * (digits[1] + RADICES[1] * (digits[2] + RADICES[2] * digits[3]))
}

fn location(
    vertical_branch: usize,
    horizontal_branch: usize,
    local_row: usize,
    local_frequency: usize,
) -> (usize, usize) {
    assert!(vertical_branch < CROSS_ROWS);
    assert!(horizontal_branch < H_BRANCHES);
    assert!(local_row < TILE_ROWS);
    assert!(local_frequency < TILE_COLS);

    let ciphertext = vertical_branch * CT_PER_BATCH + horizontal_branch / TILES_PER_CT;
    let slot =
        (horizontal_branch % TILES_PER_CT) * TILE_SLOTS + local_row * TILE_COLS + local_frequency;
    (ciphertext, slot)
}

#[test]
fn r30_5_optimized_uhd_layout() {
    assert_eq!(CROSS_ROWS, 135);
    assert_eq!(H_BRANCHES, 15);
    assert_eq!(TILES_PER_CT, 8);
    assert_eq!(CT_PER_BATCH, 2);
    assert_eq!(CROSS_ROWS * CT_PER_BATCH, 270);

    let mut seen = vec![false; ROWS * COLS];
    let mut count = 0usize;

    for vertical_branch in 0..CROSS_ROWS {
        let cross_frequency = natural_cross_index(vertical_branch);

        for horizontal_branch in 0..H_BRANCHES {
            for local_row in 0..TILE_ROWS {
                let vertical_frequency = cross_frequency + CROSS_ROWS * local_row;

                for local_frequency in 0..TILE_COLS {
                    let horizontal_frequency = horizontal_branch + H_BRANCHES * local_frequency;

                    let index = vertical_frequency * COLS + horizontal_frequency;
                    assert!(!seen[index], "duplicate frequency bin");
                    seen[index] = true;
                    count += 1;

                    let (ct, slot) = location(
                        vertical_branch,
                        horizontal_branch,
                        local_row,
                        local_frequency,
                    );

                    assert!(ct < 270);
                    assert!(slot < CT_SLOTS);
                }
            }
        }
    }

    assert_eq!(count, ROWS * COLS);
    assert!(seen.iter().all(|&present| present));

    // FFT135 operates across ciphertexts at identical slot positions.
    for branch in 0..H_BRANCHES {
        for row in 0..TILE_ROWS {
            for frequency in 0..TILE_COLS {
                let (_, reference_slot) = location(0, branch, row, frequency);
                for vertical_branch in 1..CROSS_ROWS {
                    let (_, slot) = location(vertical_branch, branch, row, frequency);
                    assert_eq!(slot, reference_slot);
                }
            }
        }
    }

    // FFT16 rotation offsets stay inside each 4096-slot tile.
    let mut minimum = 0isize;
    let mut maximum = 0isize;

    for span in [4usize, 16] {
        let quarter = span / 4;
        for row in 0..TILE_ROWS {
            let within = row % span;
            let j = within % quarter;

            for q in 0..4 {
                let input_row = (row / span) * span + j + q * quarter;
                let delta = (input_row as isize - row as isize) * TILE_COLS as isize;

                minimum = minimum.min(delta);
                maximum = maximum.max(delta);

                for tile in 0..H_BRANCHES {
                    let tile_base = tile * TILE_SLOTS;
                    for frequency in 0..TILE_COLS {
                        let target = tile_base + row * TILE_COLS + frequency;
                        let source = (target as isize + delta) as usize;

                        assert_eq!(source / TILE_SLOTS, tile);
                        assert_eq!(source % TILE_COLS, frequency);
                        assert_eq!((source % TILE_SLOTS) / TILE_COLS, input_row);
                    }
                }
            }
        }
    }

    assert_eq!(minimum, -3072);
    assert_eq!(maximum, 3072);

    println!("R30_5_UHD_PIXELS={count}");
    println!("R30_5_HORIZONTAL_BATCHES={CROSS_ROWS}");
    println!("R30_5_HORIZONTAL_CIPHERTEXTS=270");
    println!("R30_5_OUTPUT_CIPHERTEXTS=270");
    println!("R30_5_ACTIVE_TILES_PER_BATCH=15");
    println!("R30_5_FFT135_SLOT_ALIGNMENT=PASS");
    println!("R30_5_FFT16_ROTATION_RANGE=-3072,3072");
    println!("R30_5_FREQUENCY_BIJECTION=PASS");
    println!("R30_5_STATUS=PASS");
}
