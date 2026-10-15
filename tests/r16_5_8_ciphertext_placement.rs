//! R16.5.8: concrete UHD ciphertext-slot placement contract.
//! This validates storage and addressing, not encrypted execution.

use ccmm_rs::eblas::physical_tiled_fft::PhysicalTileLayout;

const ROWS: usize = 2160;
const COLS: usize = 3840;
const TILE_ROWS: usize = 16;
const TILE_COLS: usize = 256;
const SLOTS: usize = 8192;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Packing {
    Compact,
    RowAligned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Address {
    ciphertext: usize,
    slot: usize,
}

#[derive(Debug, Clone, Copy)]
struct Placement {
    layout: PhysicalTileLayout,
    packing: Packing,
}

impl Placement {
    fn new(packing: Packing) -> Self {
        Self {
            layout: PhysicalTileLayout::new(ROWS, COLS, TILE_ROWS, TILE_COLS),
            packing,
        }
    }

    fn ciphertext_count(self) -> usize {
        match self.packing {
            Packing::Compact => self.layout.tile_count().div_ceil(2),
            Packing::RowAligned => self.layout.grid_rows * self.layout.grid_cols.div_ceil(2),
        }
    }

    fn address(self, row: usize, col: usize) -> Address {
        assert!(row < ROWS && col < COLS);

        let tile_row = row / TILE_ROWS;
        let tile_col = col / TILE_COLS;
        let local_row = row % TILE_ROWS;
        let local_col = col % TILE_COLS;

        let local_slot = local_row * TILE_COLS + local_col;

        let (ciphertext, half) = match self.packing {
            Packing::Compact => {
                let tile = tile_row * self.layout.grid_cols + tile_col;
                (tile / 2, tile % 2)
            }
            Packing::RowAligned => {
                let ct = tile_row * self.layout.grid_cols.div_ceil(2) + tile_col / 2;
                (ct, tile_col % 2)
            }
        };

        Address {
            ciphertext,
            slot: half * self.layout.tile_elements() + local_slot,
        }
    }
}

fn verify_buffer(placement: Placement) -> usize {
    let capacity = placement.ciphertext_count() * SLOTS;
    let mut occupied = vec![false; capacity];

    for row in 0..ROWS {
        for col in 0..COLS {
            let address = placement.address(row, col);

            assert!(address.ciphertext < placement.ciphertext_count());
            assert!(address.slot < SLOTS);

            let index = address.ciphertext * SLOTS + address.slot;

            assert!(!occupied[index], "Ciphertext-slot collision");
            occupied[index] = true;
        }
    }

    let used = occupied.iter().filter(|&&x| x).count();
    assert_eq!(used, ROWS * COLS);

    capacity - used
}

#[test]
fn r16_5_8_uhd_ciphertext_placement() {
    for packing in [Packing::Compact, Packing::RowAligned] {
        let placement = Placement::new(packing);

        // Both logical execution buffers use the same addressing
        // contract but must be allocated as separate ciphertext arrays.
        let input_unused = verify_buffer(placement);
        let output_unused = verify_buffer(placement);

        assert_eq!(input_unused, output_unused);

        let expected_cts = match packing {
            Packing::Compact => 1013,
            Packing::RowAligned => 1080,
        };

        assert_eq!(placement.ciphertext_count(), expected_cts);

        let expected_unused = match packing {
            Packing::Compact => 4096,
            Packing::RowAligned => 552960,
        };

        assert_eq!(input_unused, expected_unused);

        println!(
            "R16_5_8_PACKING={packing:?} \
             CIPHERTEXTS={} USED_SLOTS={} UNUSED_SLOTS={} \
             TWO_BUFFER_CIPHERTEXTS={}",
            placement.ciphertext_count(),
            ROWS * COLS,
            input_unused,
            placement.ciphertext_count() * 2,
        );
    }

    println!("R16_5_8_CIPHERTEXT_PLACEMENT=PASS");
}
