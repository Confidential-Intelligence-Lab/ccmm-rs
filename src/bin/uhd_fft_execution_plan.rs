//! R16.4a: exact UHD encrypted FFT planning, without execution.

const ROWS: usize = 2160;
const COLS: usize = 3840;
const TILE_ROWS: usize = 16;
const TILE_COLS: usize = 256;

fn factors(mut n: usize) -> Vec<usize> {
    let mut result = Vec::new();
    for radix in [2, 3, 5] {
        while n % radix == 0 {
            result.push(radix);
            n /= radix;
        }
    }
    assert_eq!(n, 1, "Unsupported FFT factor");
    result
}

fn main() {
    let grid_rows = ROWS / TILE_ROWS;
    let grid_cols = COLS / TILE_COLS;
    let tiles = grid_rows * grid_cols;
    let elements = TILE_ROWS * TILE_COLS;

    assert_eq!(ROWS % TILE_ROWS, 0);
    assert_eq!(COLS % TILE_COLS, 0);

    let row_radices = factors(ROWS);
    let col_radices = factors(COLS);
    let local_row_radices = factors(TILE_ROWS);
    let local_col_radices = factors(TILE_COLS);
    let global_row_radices = factors(grid_rows);
    let global_col_radices = factors(grid_cols);

    println!("R16_PLAN_STATUS=PASS");
    println!("R16_IMAGE_ROWS={ROWS}");
    println!("R16_IMAGE_COLS={COLS}");
    println!("R16_IMAGE_PIXELS={}", ROWS * COLS);
    println!("R16_TILE_ROWS={TILE_ROWS}");
    println!("R16_TILE_COLS={TILE_COLS}");
    println!("R16_TILE_ELEMENTS={elements}");
    println!("R16_GRID_ROWS={grid_rows}");
    println!("R16_GRID_COLS={grid_cols}");
    println!("R16_TILE_COUNT={tiles}");

    println!("R16_ROW_RADICES={row_radices:?}");
    println!("R16_COL_RADICES={col_radices:?}");
    println!("R16_LOCAL_ROW_RADICES={local_row_radices:?}");
    println!("R16_LOCAL_COL_RADICES={local_col_radices:?}");
    println!("R16_GLOBAL_ROW_RADICES={global_row_radices:?}");
    println!("R16_GLOBAL_COL_RADICES={global_col_radices:?}");

    let local_stages = local_row_radices.len() + local_col_radices.len();
    let global_stages = global_row_radices.len() + global_col_radices.len();

    println!("R16_LOCAL_STAGES={local_stages}");
    println!("R16_GLOBAL_STAGES={global_stages}");
    println!(
        "R16_TOTAL_MIXED_RADIX_STAGES={}",
        local_stages + global_stages
    );

    // Conservative planning assumption: one level per stage.
    // This is not yet a verified CKKS circuit-depth measurement.
    println!("R16_ASSUMED_LEVELS_PER_STAGE=1");
    println!(
        "R16_ESTIMATED_REQUIRED_LEVELS={}",
        local_stages + global_stages
    );

    // CKKS complex-slot capacities are hypothetical profile inputs.
    // Actual ciphertext packing and Galois-key requirements remain
    // subject to implementation and validation.
    for slots in [4096usize, 8192, 16384, 32768] {
        let packing = slots / elements;
        let ciphertexts = if packing == 0 {
            0
        } else {
            tiles.div_ceil(packing)
        };

        println!(
            "R16_PACKING_CANDIDATE slots={slots} tiles_per_ciphertext={packing} ciphertexts={ciphertexts}"
        );
    }

    println!("R16_ENCRYPTED_EXECUTION_PERFORMED=false");
    println!("R16_GALOIS_KEYS_GENERATED=false");
    println!("R16_CKKS_DEPTH_VALIDATED=false");
    println!("R16_SECURITY_BEARING=false");
}
