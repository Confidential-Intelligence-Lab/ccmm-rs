// R16.5.15b: UHD FFT2 ciphertext-layout schedule (no encryption).
// This test validates metadata mappings; it does not prove ciphertext
// transformations or a zero-depth homomorphic layout conversion.
const ROWS: usize = 2160;
const COLS: usize = 3840;
const TILE_ROWS: usize = 16;
const TILE_COLS: usize = 256;
const GRID_ROWS: usize = ROWS / TILE_ROWS;
const GRID_COLS: usize = COLS / TILE_COLS;
const TILE_SLOTS: usize = TILE_ROWS * TILE_COLS;
const CT_SLOTS: usize = 16384;
const TILES_PER_CT: usize = CT_SLOTS / TILE_SLOTS;
const CT_PER_TILE_ROW: usize = GRID_COLS.div_ceil(TILES_PER_CT);
const CT_COUNT: usize = GRID_ROWS * CT_PER_TILE_ROW;
const RADICES: [usize; 4] = [5, 3, 3, 3];

fn natural_cross_index(index: usize) -> usize {
    assert!(index < GRID_ROWS);
    let mut remaining = index;
    let mut digits = [0usize; 4];
    for (stage, &radix) in RADICES.iter().rev().enumerate() {
        digits[3 - stage] = remaining % radix;
        remaining /= radix;
    }
    assert_eq!(remaining, 0);
    digits[0] + RADICES[0] * (digits[1] + RADICES[1] * (digits[2] + RADICES[2] * digits[3]))
}

fn location(
    tile_row: usize,
    horizontal_branch: usize,
    local_row: usize,
    local_frequency: usize,
) -> (usize, usize) {
    assert!(tile_row < GRID_ROWS);
    assert!(horizontal_branch < GRID_COLS);
    assert!(local_row < TILE_ROWS);
    assert!(local_frequency < TILE_COLS);
    (
        tile_row * CT_PER_TILE_ROW + horizontal_branch / TILES_PER_CT,
        (horizontal_branch % TILES_PER_CT) * TILE_SLOTS + local_row * TILE_COLS + local_frequency,
    )
}

#[test]
fn r16_5_15b_uhd_fft2_ciphertext_layout() {
    assert_eq!((ROWS, COLS, GRID_ROWS, GRID_COLS), (2160, 3840, 135, 15));
    assert_eq!(
        (CT_SLOTS, TILES_PER_CT, CT_PER_TILE_ROW, CT_COUNT),
        (16384, 4, 4, 540)
    );
    let mut cross_seen = [false; GRID_ROWS];
    for branch in 0..GRID_ROWS {
        let k = natural_cross_index(branch);
        assert!(!cross_seen[k], "duplicate FFT135 natural index {k}");
        cross_seen[k] = true;
    }
    assert!(cross_seen.iter().all(|&v| v));

    // The horizontal output branch and FFT135 DIF output branch are
    // ciphertext metadata, not physically permuted slot indices.
    let mut seen = vec![false; ROWS * COLS];
    let mut seen_count = 0usize;
    for vertical_branch in 0..GRID_ROWS {
        let k_cross = natural_cross_index(vertical_branch);
        for horizontal_branch in 0..GRID_COLS {
            for local_vertical_frequency in 0..TILE_ROWS {
                let ky = k_cross + GRID_ROWS * local_vertical_frequency;
                for local_horizontal_frequency in 0..TILE_COLS {
                    let kx = horizontal_branch + GRID_COLS * local_horizontal_frequency;
                    let index = ky * COLS + kx;
                    assert!(!seen[index], "duplicate UHD frequency ({ky},{kx})");
                    seen[index] = true;
                    seen_count += 1;
                    let (ct, slot) = location(
                        vertical_branch,
                        horizontal_branch,
                        local_vertical_frequency,
                        local_horizontal_frequency,
                    );
                    assert!(ct < CT_COUNT && slot < CT_SLOTS);
                    assert_eq!(slot / TILE_SLOTS, horizontal_branch % TILES_PER_CT);
                    assert_eq!(
                        slot % TILE_SLOTS,
                        local_vertical_frequency * TILE_COLS + local_horizontal_frequency
                    );
                }
            }
        }
    }
    assert_eq!(seen_count, ROWS * COLS);
    assert!(seen.iter().all(|&v| v));

    // Each vertical FFT135 branch sees all 135 tile rows at the *same*
    // slot index, so its cross-ciphertext butterflies need no slot rotation.
    for horizontal_branch in 0..GRID_COLS {
        for local_row in 0..TILE_ROWS {
            for local_frequency in 0..TILE_COLS {
                let (_, expected_slot) = location(0, horizontal_branch, local_row, local_frequency);
                for tile_row in 1..GRID_ROWS {
                    let (ct, slot) =
                        location(tile_row, horizontal_branch, local_row, local_frequency);
                    assert_eq!(slot, expected_slot);
                    assert_eq!(
                        ct,
                        tile_row * CT_PER_TILE_ROW + horizontal_branch / TILES_PER_CT
                    );
                }
            }
        }
    }

    // FFT16 radix-4 DIT spans 4 and 16 rows. Signed rotation offsets
    // are measured in *slots*, not rows, and never leave a 4096-slot tile.
    let mut offsets = std::collections::BTreeSet::new();
    for span in [4usize, 16] {
        let quarter = span / 4;
        for local_row in 0..TILE_ROWS {
            let within = local_row % span;
            let j = within % quarter;
            for q in 0..4 {
                let input_row = (local_row / span) * span + j + q * quarter;
                let delta = (input_row as isize - local_row as isize) * TILE_COLS as isize;
                offsets.insert(delta);
                for tile in 0..TILES_PER_CT {
                    for frequency in 0..TILE_COLS {
                        let target = tile * TILE_SLOTS + local_row * TILE_COLS + frequency;
                        let source = (target as isize + delta) as usize;
                        assert_eq!(source / TILE_SLOTS, tile);
                        assert_eq!(source % TILE_COLS, frequency);
                        assert_eq!((source % TILE_SLOTS) / TILE_COLS, input_row);
                    }
                }
            }
        }
    }
    assert!(offsets.iter().any(|&offset| offset < 0));
    assert!(offsets.iter().any(|&offset| offset > 0));
    assert_eq!(offsets.iter().copied().max(), Some(3072));
    assert_eq!(offsets.iter().copied().min(), Some(-3072));

    // One padded tile per tile row, never containing an active branch.
    for tile_row in 0..GRID_ROWS {
        for branch in 0..GRID_COLS {
            let (ct, _) = location(tile_row, branch, 0, 0);
            assert_eq!(ct / CT_PER_TILE_ROW, tile_row);
        }
    }
    println!("R16_UHD_LAYOUT_PIXELS={seen_count}");
    println!("R16_UHD_LAYOUT_CIPHERTEXTS={CT_COUNT}");
    println!("R16_UHD_LAYOUT_TILES_PER_CIPHERTEXT={TILES_PER_CT}");
    println!("R16_UHD_LAYOUT_CIPHERTEXTS_PER_TILE_ROW={CT_PER_TILE_ROW}");
    println!("R16_UHD_LAYOUT_FFT135_SLOT_ALIGNMENT=PASS");
    println!("R16_UHD_LAYOUT_FFT16_MAX_ROTATION_SLOTS=3072");
    println!("R16_UHD_LAYOUT_FREQUENCY_BIJECTION=PASS");
    println!("R16_UHD_LAYOUT_STATUS=PASS");
}
