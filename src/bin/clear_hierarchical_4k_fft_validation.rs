use ccmm_rs::eblas::fft::{
    execute_packed_tile_dif_stage_pp, execute_repeated_packed_fft2_dif_stage_pp, fft2_pp,
    Fft2Shape, FftDirection, PackedFft2Axis, PackedFft2DifStageDiagonals,
    PackedTileDifStageDiagonals,
};
use num_complex::Complex64;
use std::f64::consts::PI;
use std::time::Instant;

const IMAGE_DIM: usize = 4096;
const MACRO_DIM: usize = 1024;
const MACROS_PER_AXIS: usize = 4;
const MACRO_COUNT: usize = 16;

const TILE_DIM: usize = 64;
const TILE_ELEMENTS: usize = TILE_DIM * TILE_DIM;
const TILES_PER_AXIS: usize = MACRO_DIM / TILE_DIM;

const TILES_PER_CIPHERTEXT: usize = 8;
const SLOT_COUNT: usize = TILE_ELEMENTS * TILES_PER_CIPHERTEXT;

const ROW_BLOCKS: usize = TILES_PER_AXIS / TILES_PER_CIPHERTEXT;
const CT_PER_MACRO: usize = ROW_BLOCKS * TILES_PER_AXIS;

fn bit_reverse(index: usize, length: usize) -> usize {
    if length <= 2 {
        return index;
    }

    let bits = length.trailing_zeros();
    index.reverse_bits() >> (usize::BITS - bits)
}

fn deterministic_black_white_image() -> Vec<Complex64> {
    (0..IMAGE_DIM)
        .flat_map(|row| {
            (0..IMAGE_DIM).map(move |col| {
                let white = ((row / 32) + (col / 32)) % 2 == 0;
                Complex64::new(if white { 1.0 } else { 0.0 }, 0.0)
            })
        })
        .collect()
}

fn packed_location(tile_row: usize, tile_col: usize) -> (usize, usize) {
    let row_block = tile_row / TILES_PER_CIPHERTEXT;
    let lane = tile_row % TILES_PER_CIPHERTEXT;
    let packed_index = row_block * TILES_PER_AXIS + tile_col;
    (packed_index, lane)
}

fn pack_macrotiles(image: &[Complex64]) -> Vec<Vec<Vec<Complex64>>> {
    assert_eq!(image.len(), IMAGE_DIM * IMAGE_DIM);

    let mut macrotiles = (0..MACRO_COUNT)
        .map(|_| vec![vec![Complex64::new(0.0, 0.0); SLOT_COUNT]; CT_PER_MACRO])
        .collect::<Vec<_>>();

    for macro_row in 0..MACROS_PER_AXIS {
        for macro_col in 0..MACROS_PER_AXIS {
            let macro_index = macro_row * MACROS_PER_AXIS + macro_col;

            for tile_row in 0..TILES_PER_AXIS {
                for tile_col in 0..TILES_PER_AXIS {
                    let (packed_index, lane) = packed_location(tile_row, tile_col);
                    let lane_base = lane * TILE_ELEMENTS;

                    for local_row in 0..TILE_DIM {
                        for local_col in 0..TILE_DIM {
                            let global_row =
                                macro_row * MACRO_DIM + tile_row * TILE_DIM + local_row;

                            let global_col =
                                macro_col * MACRO_DIM + tile_col * TILE_DIM + local_col;

                            let global_index = global_row * IMAGE_DIM + global_col;
                            let tile_slot = local_row * TILE_DIM + local_col;

                            macrotiles[macro_index][packed_index][lane_base + tile_slot] =
                                image[global_index];
                        }
                    }
                }
            }
        }
    }

    macrotiles
}

fn unpack_physical_image(macrotiles: &[Vec<Vec<Complex64>>]) -> Vec<Complex64> {
    assert_eq!(macrotiles.len(), MACRO_COUNT);

    let mut physical = vec![Complex64::new(0.0, 0.0); IMAGE_DIM * IMAGE_DIM];

    for macro_row in 0..MACROS_PER_AXIS {
        for macro_col in 0..MACROS_PER_AXIS {
            let macro_index = macro_row * MACROS_PER_AXIS + macro_col;

            for tile_row in 0..TILES_PER_AXIS {
                for tile_col in 0..TILES_PER_AXIS {
                    let (packed_index, lane) = packed_location(tile_row, tile_col);
                    let lane_base = lane * TILE_ELEMENTS;

                    for local_row in 0..TILE_DIM {
                        for local_col in 0..TILE_DIM {
                            let global_row =
                                macro_row * MACRO_DIM + tile_row * TILE_DIM + local_row;

                            let global_col =
                                macro_col * MACRO_DIM + tile_col * TILE_DIM + local_col;

                            let global_index = global_row * IMAGE_DIM + global_col;
                            let tile_slot = local_row * TILE_DIM + local_col;

                            physical[global_index] =
                                macrotiles[macro_index][packed_index][lane_base + tile_slot];
                        }
                    }
                }
            }
        }
    }

    physical
}

fn physical_to_logical(physical: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(physical.len(), IMAGE_DIM * IMAGE_DIM);

    let mut logical = vec![Complex64::new(0.0, 0.0); IMAGE_DIM * IMAGE_DIM];

    for row in 0..IMAGE_DIM {
        let physical_row = bit_reverse(row, IMAGE_DIM);

        for col in 0..IMAGE_DIM {
            let physical_col = bit_reverse(col, IMAGE_DIM);

            logical[row * IMAGE_DIM + col] = physical[physical_row * IMAGE_DIM + physical_col];
        }
    }

    logical
}

fn row_macro_twiddles(macro_offset: usize, ct_index: usize, span_macros: usize) -> Vec<Complex64> {
    let tile_col = ct_index % TILES_PER_AXIS;
    let global_span = span_macros * MACRO_DIM;

    let mut twiddles = vec![Complex64::new(0.0, 0.0); SLOT_COUNT];

    for lane in 0..TILES_PER_CIPHERTEXT {
        let lane_base = lane * TILE_ELEMENTS;

        for local_row in 0..TILE_DIM {
            for local_col in 0..TILE_DIM {
                let local_x = tile_col * TILE_DIM + local_col;
                let twiddle_index = macro_offset * MACRO_DIM + local_x;

                let angle = -2.0 * PI * twiddle_index as f64 / global_span as f64;

                twiddles[lane_base + local_row * TILE_DIM + local_col] =
                    Complex64::new(angle.cos(), angle.sin());
            }
        }
    }

    twiddles
}

fn column_macro_twiddles(
    macro_offset: usize,
    ct_index: usize,
    span_macros: usize,
) -> Vec<Complex64> {
    let row_block = ct_index / TILES_PER_AXIS;
    let global_span = span_macros * MACRO_DIM;

    let mut twiddles = vec![Complex64::new(0.0, 0.0); SLOT_COUNT];

    for lane in 0..TILES_PER_CIPHERTEXT {
        let lane_base = lane * TILE_ELEMENTS;

        for local_row in 0..TILE_DIM {
            let local_y = row_block * TILES_PER_CIPHERTEXT * TILE_DIM + lane * TILE_DIM + local_row;

            let twiddle_index = macro_offset * MACRO_DIM + local_y;

            let angle = -2.0 * PI * twiddle_index as f64 / global_span as f64;

            let w = Complex64::new(angle.cos(), angle.sin());

            for local_col in 0..TILE_DIM {
                twiddles[lane_base + local_row * TILE_DIM + local_col] = w;
            }
        }
    }

    twiddles
}

fn execute_macro_row_stage(macrotiles: &mut [Vec<Vec<Complex64>>], span_macros: usize) {
    let half = span_macros / 2;

    for macro_row in 0..MACROS_PER_AXIS {
        for group_start in (0..MACROS_PER_AXIS).step_by(span_macros) {
            for macro_offset in 0..half {
                let upper_macro = macro_row * MACROS_PER_AXIS + group_start + macro_offset;

                let lower_macro = upper_macro + half;

                let (upper_slice, lower_slice) = if upper_macro < lower_macro {
                    let (left, right) = macrotiles.split_at_mut(lower_macro);
                    (&mut left[upper_macro], &mut right[0])
                } else {
                    unreachable!();
                };

                for ct_index in 0..CT_PER_MACRO {
                    let twiddles = row_macro_twiddles(macro_offset, ct_index, span_macros);

                    for (slot, twiddle) in twiddles.iter().enumerate() {
                        let a = upper_slice[ct_index][slot];
                        let b = lower_slice[ct_index][slot];

                        upper_slice[ct_index][slot] = a + b;
                        lower_slice[ct_index][slot] = *twiddle * (a - b);
                    }
                }
            }
        }
    }
}

fn execute_macro_column_stage(macrotiles: &mut [Vec<Vec<Complex64>>], span_macros: usize) {
    let half = span_macros / 2;

    for macro_col in 0..MACROS_PER_AXIS {
        for group_start in (0..MACROS_PER_AXIS).step_by(span_macros) {
            for macro_offset in 0..half {
                let upper_macro = (group_start + macro_offset) * MACROS_PER_AXIS + macro_col;

                let lower_macro = (group_start + macro_offset + half) * MACROS_PER_AXIS + macro_col;

                let (upper_slice, lower_slice) = if upper_macro < lower_macro {
                    let (left, right) = macrotiles.split_at_mut(lower_macro);
                    (&mut left[upper_macro], &mut right[0])
                } else {
                    unreachable!();
                };

                for ct_index in 0..CT_PER_MACRO {
                    let twiddles = column_macro_twiddles(macro_offset, ct_index, span_macros);

                    for (slot, twiddle) in twiddles.iter().enumerate() {
                        let a = upper_slice[ct_index][slot];
                        let b = lower_slice[ct_index][slot];

                        upper_slice[ct_index][slot] = a + b;
                        lower_slice[ct_index][slot] = *twiddle * (a - b);
                    }
                }
            }
        }
    }
}

fn execute_1k_rows_clear(values: &mut Vec<Vec<Complex64>>) {
    let mut span_tiles = TILES_PER_AXIS;

    while span_tiles >= 2 {
        let half_tiles = span_tiles / 2;
        let global_span = span_tiles * TILE_DIM;

        let input = values.clone();
        let mut output = vec![vec![Complex64::new(0.0, 0.0); SLOT_COUNT]; CT_PER_MACRO];

        for row_block in 0..ROW_BLOCKS {
            for group_start in (0..TILES_PER_AXIS).step_by(span_tiles) {
                for tile_offset in 0..half_tiles {
                    let upper_col = group_start + tile_offset;
                    let lower_col = upper_col + half_tiles;

                    let upper_index = row_block * TILES_PER_AXIS + upper_col;
                    let lower_index = row_block * TILES_PER_AXIS + lower_col;

                    for lane in 0..TILES_PER_CIPHERTEXT {
                        let lane_base = lane * TILE_ELEMENTS;

                        for local_row in 0..TILE_DIM {
                            for local_col in 0..TILE_DIM {
                                let slot = lane_base + local_row * TILE_DIM + local_col;

                                let twiddle_index = tile_offset * TILE_DIM + local_col;

                                let angle = -2.0 * PI * twiddle_index as f64 / global_span as f64;

                                let w = Complex64::new(angle.cos(), angle.sin());

                                let a = input[upper_index][slot];
                                let b = input[lower_index][slot];

                                output[upper_index][slot] = a + b;
                                output[lower_index][slot] = w * (a - b);
                            }
                        }
                    }
                }
            }
        }

        *values = output;
        span_tiles /= 2;
    }

    let shape = Fft2Shape::new(TILE_DIM, TILE_DIM);
    let mut span = TILE_DIM;

    while span >= 2 {
        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Rows,
            span,
            FftDirection::Forward,
        );

        *values = values
            .iter()
            .map(|value| {
                execute_repeated_packed_fft2_dif_stage_pp(value, &diagonals, TILES_PER_CIPHERTEXT)
            })
            .collect();

        span /= 2;
    }
}

fn execute_1k_columns_clear(values: &mut Vec<Vec<Complex64>>) {
    /*
     * Inter-CT span-16 column stage.
     */
    {
        let span_tiles = TILES_PER_AXIS;
        let global_span = span_tiles * TILE_DIM;

        let input = values.clone();
        let mut output = vec![vec![Complex64::new(0.0, 0.0); SLOT_COUNT]; CT_PER_MACRO];

        for tile_col in 0..TILES_PER_AXIS {
            let upper_index = tile_col;
            let lower_index = TILES_PER_AXIS + tile_col;

            for lane in 0..TILES_PER_CIPHERTEXT {
                let lane_base = lane * TILE_ELEMENTS;

                for local_row in 0..TILE_DIM {
                    let twiddle_index = lane * TILE_DIM + local_row;

                    let angle = -2.0 * PI * twiddle_index as f64 / global_span as f64;

                    let w = Complex64::new(angle.cos(), angle.sin());

                    for local_col in 0..TILE_DIM {
                        let slot = lane_base + local_row * TILE_DIM + local_col;

                        let a = input[upper_index][slot];
                        let b = input[lower_index][slot];

                        output[upper_index][slot] = a + b;
                        output[lower_index][slot] = w * (a - b);
                    }
                }
            }
        }

        *values = output;
    }

    /*
     * Three intra-CT tile-block stages.
     */
    let mut span_tiles = 8usize;

    while span_tiles >= 2 {
        let diagonals = PackedTileDifStageDiagonals::new_column(
            TILE_DIM,
            TILE_DIM,
            TILES_PER_CIPHERTEXT,
            span_tiles,
            FftDirection::Forward,
        );

        *values = values
            .iter()
            .map(|value| execute_packed_tile_dif_stage_pp(value, &diagonals))
            .collect();

        span_tiles /= 2;
    }

    /*
     * Six local-column stages.
     */
    let shape = Fft2Shape::new(TILE_DIM, TILE_DIM);
    let mut span = TILE_DIM;

    while span >= 2 {
        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Columns,
            span,
            FftDirection::Forward,
        );

        *values = values
            .iter()
            .map(|value| {
                execute_repeated_packed_fft2_dif_stage_pp(value, &diagonals, TILES_PER_CIPHERTEXT)
            })
            .collect();

        span /= 2;
    }
}

fn error_metrics(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());

    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (actual, expected) in actual.iter().zip(expected) {
        let error = *actual - *expected;

        squared_error += error.norm_sqr();
        squared_reference += expected.norm_sqr();
        max_abs = max_abs.max(error.norm());
    }

    let rel_l2 = if squared_reference > 0.0 {
        (squared_error / squared_reference).sqrt()
    } else {
        squared_error.sqrt()
    };

    (rel_l2, max_abs)
}

fn main() {
    println!("R20_CLEAR_4K_BEGIN");
    println!("R20_CLEAR_4K_IMAGE_DIMENSION=4096x4096");
    println!("R20_CLEAR_4K_MACRO_GRID=4x4");
    println!("R20_CLEAR_4K_CT_EQUIVALENT_COUNT=512");

    let image_start = Instant::now();
    let image = deterministic_black_white_image();

    println!(
        "R20_CLEAR_4K_IMAGE_MS={}",
        image_start.elapsed().as_millis()
    );

    let oracle_start = Instant::now();

    let oracle = fft2_pp(
        Fft2Shape::new(IMAGE_DIM, IMAGE_DIM),
        FftDirection::Forward,
        &image,
    );

    println!(
        "R20_CLEAR_4K_ORACLE_MS={}",
        oracle_start.elapsed().as_millis()
    );

    let hierarchical_start = Instant::now();

    let mut macrotiles = pack_macrotiles(&image);

    for span_macros in [4usize, 2usize] {
        let start = Instant::now();
        execute_macro_row_stage(&mut macrotiles, span_macros);

        println!(
            "R20_CLEAR_4K_STAGE kind=macro-row span_macros={} elapsed_ms={}",
            span_macros,
            start.elapsed().as_millis()
        );
    }

    let start = Instant::now();

    for (macro_index, macrotile) in macrotiles.iter_mut().enumerate() {
        execute_1k_rows_clear(macrotile);

        println!("R20_CLEAR_4K_ROW_PROGRESS macro={}/16", macro_index + 1);
    }

    println!("R20_CLEAR_4K_ROW_PHASE_MS={}", start.elapsed().as_millis());

    for span_macros in [4usize, 2usize] {
        let start = Instant::now();
        execute_macro_column_stage(&mut macrotiles, span_macros);

        println!(
            "R20_CLEAR_4K_STAGE kind=macro-column span_macros={} elapsed_ms={}",
            span_macros,
            start.elapsed().as_millis()
        );
    }

    let start = Instant::now();

    for (macro_index, macrotile) in macrotiles.iter_mut().enumerate() {
        execute_1k_columns_clear(macrotile);

        println!("R20_CLEAR_4K_COLUMN_PROGRESS macro={}/16", macro_index + 1);
    }

    println!(
        "R20_CLEAR_4K_COLUMN_PHASE_MS={}",
        start.elapsed().as_millis()
    );

    let physical = unpack_physical_image(&macrotiles);
    let actual = physical_to_logical(&physical);

    let hierarchical_ms = hierarchical_start.elapsed().as_millis();

    let (rel_l2, max_abs) = error_metrics(&actual, &oracle);

    let tolerance = 1.0e-10;
    let status = if rel_l2 <= tolerance { "PASS" } else { "FAIL" };

    println!("R20_CLEAR_4K_RESULT_BEGIN");
    println!("R20_CLEAR_4K_HIERARCHICAL_MS={hierarchical_ms}");
    println!("R20_CLEAR_4K_REL_L2={rel_l2:.12e}");
    println!("R20_CLEAR_4K_MAX_ABS={max_abs:.12e}");
    println!("R20_CLEAR_4K_TOLERANCE={tolerance:.12e}");
    println!("R20_CLEAR_4K_STATUS={status}");
    println!("R20_CLEAR_4K_RESULT_END");

    assert_eq!(
        status, "PASS",
        "hierarchical 4K FFT must match the canonical clear FFT"
    );
}
