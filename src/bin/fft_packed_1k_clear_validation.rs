use ccmm_rs::eblas::fft::{
    execute_repeated_packed_fft2_dif_stage_pp, fft2_pp, Fft2Shape, FftDirection, PackedFft2Axis,
    PackedFft2DifStageDiagonals,
};
use num_complex::Complex64;
use std::f64::consts::PI;
use std::time::Instant;

const IMAGE_DIM: usize = 1024;
const TILE_DIM: usize = 64;
const TILE_ELEMENTS: usize = TILE_DIM * TILE_DIM;
const TILES_PER_AXIS: usize = IMAGE_DIM / TILE_DIM;
const TILES_PER_CIPHERTEXT: usize = 8;
const SLOT_COUNT: usize = TILE_ELEMENTS * TILES_PER_CIPHERTEXT;
const CIPHERTEXTS_PER_ROW_BLOCK: usize = TILES_PER_AXIS;
const ROW_BLOCKS: usize = TILES_PER_AXIS / TILES_PER_CIPHERTEXT;
const CIPHERTEXT_COUNT: usize = ROW_BLOCKS * CIPHERTEXTS_PER_ROW_BLOCK;

fn bit_reverse(index: usize, length: usize) -> usize {
    if length <= 2 {
        return index;
    }

    let bits = length.trailing_zeros();
    index.reverse_bits() >> (usize::BITS - bits)
}

fn packed_physical_to_logical(physical: &[Complex64], shape: Fft2Shape) -> Vec<Complex64> {
    assert_eq!(physical.len(), shape.elements());

    let mut logical = vec![Complex64::new(0.0, 0.0); shape.elements()];

    for row in 0..shape.rows() {
        for col in 0..shape.cols() {
            let logical_index = row * shape.cols() + col;

            let physical_row = bit_reverse(row, shape.rows());
            let physical_col = bit_reverse(col, shape.cols());

            let physical_index = physical_row * shape.cols() + physical_col;

            logical[logical_index] = physical[physical_index];
        }
    }

    logical
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

fn deterministic_image() -> Vec<Complex64> {
    (0..IMAGE_DIM)
        .flat_map(|row| {
            (0..IMAGE_DIM).map(move |col| {
                let real = ((17 * row + 29 * col + 11) % 251) as f64 / 251.0;

                let imag = ((13 * row + 19 * col + 7) % 127) as f64 / 127.0;

                Complex64::new(real, imag)
            })
        })
        .collect()
}

/*
 * R17 8x1 packing.
 *
 * One packed vector represents eight vertically adjacent 64x64 tiles
 * from the same tile column:
 *
 *   lane 0 -> tile row 0/8
 *   ...
 *   lane 7 -> tile row 7/15
 *
 * A 16x16 tile grid therefore maps to:
 *
 *   2 row blocks x 16 tile columns = 32 packed vectors.
 */
fn packed_location(tile_row: usize, tile_col: usize) -> (usize, usize) {
    let row_block = tile_row / TILES_PER_CIPHERTEXT;
    let lane = tile_row % TILES_PER_CIPHERTEXT;

    let packed_index = row_block * CIPHERTEXTS_PER_ROW_BLOCK + tile_col;

    (packed_index, lane)
}

fn pack_image_tiles(image: &[Complex64]) -> Vec<Vec<Complex64>> {
    assert_eq!(image.len(), IMAGE_DIM * IMAGE_DIM);

    let mut packed = vec![vec![Complex64::new(0.0, 0.0); SLOT_COUNT]; CIPHERTEXT_COUNT];

    for tile_row in 0..TILES_PER_AXIS {
        for tile_col in 0..TILES_PER_AXIS {
            let (packed_index, lane) = packed_location(tile_row, tile_col);

            for local_row in 0..TILE_DIM {
                for local_col in 0..TILE_DIM {
                    let global_row = tile_row * TILE_DIM + local_row;
                    let global_col = tile_col * TILE_DIM + local_col;

                    let global_index = global_row * IMAGE_DIM + global_col;

                    let tile_slot = local_row * TILE_DIM + local_col;

                    let packed_slot = lane * TILE_ELEMENTS + tile_slot;

                    packed[packed_index][packed_slot] = image[global_index];
                }
            }
        }
    }

    packed
}

fn reconstruct_physical_image(packed: &[Vec<Complex64>]) -> Vec<Complex64> {
    assert_eq!(packed.len(), CIPHERTEXT_COUNT);

    let mut physical = vec![Complex64::new(0.0, 0.0); IMAGE_DIM * IMAGE_DIM];

    for tile_row in 0..TILES_PER_AXIS {
        for tile_col in 0..TILES_PER_AXIS {
            let (packed_index, lane) = packed_location(tile_row, tile_col);

            for local_row in 0..TILE_DIM {
                for local_col in 0..TILE_DIM {
                    let tile_slot = local_row * TILE_DIM + local_col;

                    let packed_slot = lane * TILE_ELEMENTS + tile_slot;

                    let global_row = tile_row * TILE_DIM + local_row;
                    let global_col = tile_col * TILE_DIM + local_col;

                    let global_index = global_row * IMAGE_DIM + global_col;

                    physical[global_index] = packed[packed_index][packed_slot];
                }
            }
        }
    }

    physical
}

fn max_abs(values: &[Vec<Complex64>]) -> f64 {
    values
        .iter()
        .flat_map(|value| value.iter())
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max)
}

fn main() {
    assert_eq!(TILES_PER_AXIS, 16);
    assert_eq!(ROW_BLOCKS, 2);
    assert_eq!(CIPHERTEXT_COUNT, 32);
    assert_eq!(SLOT_COUNT, 32_768);

    let image_shape = Fft2Shape::new(IMAGE_DIM, IMAGE_DIM);
    let tile_shape = Fft2Shape::new(TILE_DIM, TILE_DIM);

    println!("R17_CLEAR_BEGIN");
    println!("R17_CLEAR_IMAGE_DIMENSION={}x{}", IMAGE_DIM, IMAGE_DIM);
    println!("R17_CLEAR_TILE_DIMENSION={}x{}", TILE_DIM, TILE_DIM);
    println!("R17_CLEAR_TILE_GRID={}x{}", TILES_PER_AXIS, TILES_PER_AXIS);
    println!("R17_CLEAR_TILES_PER_CIPHERTEXT={TILES_PER_CIPHERTEXT}");
    println!("R17_CLEAR_CIPHERTEXT_COUNT={CIPHERTEXT_COUNT}");
    println!("R17_CLEAR_SLOT_COUNT={SLOT_COUNT}");
    println!("R17_CLEAR_SLOT_UTILIZATION=1.000000");
    println!("R17_CLEAR_PACKING=8x1");

    let input = deterministic_image();

    let oracle_start = Instant::now();
    let oracle = fft2_pp(image_shape, FftDirection::Forward, &input);

    println!("R17_CLEAR_ORACLE_MS={}", oracle_start.elapsed().as_millis());

    let mut values = pack_image_tiles(&input);

    println!("R17_CLEAR_PACKED_VECTOR_MAX_ABS={:.12e}", max_abs(&values));

    let total_start = Instant::now();
    let mut level = 0usize;

    /*
     * ---------------------------------------------------------------
     * Global row stages.
     *
     * With vertical 8x1 packing, all four tile-column stages are
     * inter-container but lane aligned.
     * ---------------------------------------------------------------
     */
    let mut span_tiles = TILES_PER_AXIS;

    while span_tiles >= 2 {
        let stage_start = Instant::now();

        let half_tiles = span_tiles / 2;
        let global_span = span_tiles * TILE_DIM;

        let input_stage = values;
        let mut output = vec![vec![Complex64::new(0.0, 0.0); SLOT_COUNT]; CIPHERTEXT_COUNT];

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
                                let tile_slot = local_row * TILE_DIM + local_col;

                                let slot = lane_base + tile_slot;

                                let twiddle_index = tile_offset * TILE_DIM + local_col;

                                let angle = -2.0 * PI * twiddle_index as f64 / global_span as f64;

                                let w = Complex64::new(angle.cos(), angle.sin());

                                let a = input_stage[upper_index][slot];
                                let b = input_stage[lower_index][slot];

                                output[upper_index][slot] = a + b;
                                output[lower_index][slot] = w * (a - b);
                            }
                        }
                    }
                }
            }
        }

        values = output;
        level += 1;

        println!(
            "R17_CLEAR_STAGE level={} kind=global-row span_tiles={} mode=inter-ct ct_pairs=16 elapsed_ms={} max_abs={:.12e}",
            level,
            span_tiles,
            stage_start.elapsed().as_millis(),
            max_abs(&values)
        );

        span_tiles /= 2;
    }

    assert_eq!(level, 4);

    /*
     * ---------------------------------------------------------------
     * Local row stages.
     *
     * The existing repeated packed clear primitive performs the same
     * 64x64 DIF stage independently in all eight lane blocks.
     * ---------------------------------------------------------------
     */
    let mut span = TILE_DIM;

    while span >= 2 {
        let stage_start = Instant::now();

        let diagonals = PackedFft2DifStageDiagonals::new(
            tile_shape,
            PackedFft2Axis::Rows,
            span,
            FftDirection::Forward,
        );

        values = values
            .iter()
            .map(|value| {
                execute_repeated_packed_fft2_dif_stage_pp(value, &diagonals, TILES_PER_CIPHERTEXT)
            })
            .collect();

        level += 1;

        println!(
            "R17_CLEAR_STAGE level={} kind=local-row span={} mode=intra-ct-simd elapsed_ms={} max_abs={:.12e}",
            level,
            span,
            stage_start.elapsed().as_millis(),
            max_abs(&values)
        );

        span /= 2;
    }

    assert_eq!(level, 10);

    /*
     * ---------------------------------------------------------------
     * Global column span 16.
     *
     * This is the single global-column stage that remains inter-CT in
     * the 8x1 layout. Row block 0 contains tile rows 0..7 and row
     * block 1 contains tile rows 8..15, so lane k pairs directly with
     * lane k.
     * ---------------------------------------------------------------
     */
    {
        let stage_start = Instant::now();

        let span_tiles = 16usize;
        let global_span = span_tiles * TILE_DIM;

        let input_stage = values;
        let mut output = vec![vec![Complex64::new(0.0, 0.0); SLOT_COUNT]; CIPHERTEXT_COUNT];

        for tile_col in 0..TILES_PER_AXIS {
            let upper_index = tile_col;
            let lower_index = TILES_PER_AXIS + tile_col;

            for lane in 0..TILES_PER_CIPHERTEXT {
                let lane_base = lane * TILE_ELEMENTS;

                /*
                 * At span 16, the upper logical tile row is exactly
                 * tile_offset = lane.
                 */
                let tile_offset = lane;

                for local_row in 0..TILE_DIM {
                    let twiddle_index = tile_offset * TILE_DIM + local_row;

                    let angle = -2.0 * PI * twiddle_index as f64 / global_span as f64;

                    let w = Complex64::new(angle.cos(), angle.sin());

                    for local_col in 0..TILE_DIM {
                        let tile_slot = local_row * TILE_DIM + local_col;

                        let slot = lane_base + tile_slot;

                        let a = input_stage[upper_index][slot];
                        let b = input_stage[lower_index][slot];

                        output[upper_index][slot] = a + b;
                        output[lower_index][slot] = w * (a - b);
                    }
                }
            }
        }

        values = output;
        level += 1;

        println!(
            "R17_CLEAR_STAGE level={} kind=global-column span_tiles=16 mode=inter-ct ct_pairs=16 elapsed_ms={} max_abs={:.12e}",
            level,
            stage_start.elapsed().as_millis(),
            max_abs(&values)
        );
    }

    assert_eq!(level, 11);

    /*
     * ---------------------------------------------------------------
     * Remaining global column stages: spans 8, 4, 2.
     *
     * These tile-row butterflies are internal to each 8-lane packed
     * vector.
     * ---------------------------------------------------------------
     */
    span_tiles = 8;

    while span_tiles >= 2 {
        let stage_start = Instant::now();

        let half_tiles = span_tiles / 2;
        let global_span = span_tiles * TILE_DIM;

        for value in &mut values {
            let input_stage = value.clone();

            for group_start in (0..TILES_PER_CIPHERTEXT).step_by(span_tiles) {
                for tile_offset in 0..half_tiles {
                    let upper_lane = group_start + tile_offset;
                    let lower_lane = upper_lane + half_tiles;

                    let upper_base = upper_lane * TILE_ELEMENTS;
                    let lower_base = lower_lane * TILE_ELEMENTS;

                    for local_row in 0..TILE_DIM {
                        let twiddle_index = tile_offset * TILE_DIM + local_row;

                        let angle = -2.0 * PI * twiddle_index as f64 / global_span as f64;

                        let w = Complex64::new(angle.cos(), angle.sin());

                        for local_col in 0..TILE_DIM {
                            let tile_slot = local_row * TILE_DIM + local_col;

                            let upper_slot = upper_base + tile_slot;
                            let lower_slot = lower_base + tile_slot;

                            let a = input_stage[upper_slot];
                            let b = input_stage[lower_slot];

                            value[upper_slot] = a + b;
                            value[lower_slot] = w * (a - b);
                        }
                    }
                }
            }
        }

        level += 1;

        println!(
            "R17_CLEAR_STAGE level={} kind=global-column span_tiles={} mode=intra-ct tile_rotation_slots={} elapsed_ms={} max_abs={:.12e}",
            level,
            span_tiles,
            half_tiles * TILE_ELEMENTS,
            stage_start.elapsed().as_millis(),
            max_abs(&values)
        );

        span_tiles /= 2;
    }

    assert_eq!(level, 14);

    /*
     * ---------------------------------------------------------------
     * Local column stages.
     * ---------------------------------------------------------------
     */
    span = TILE_DIM;

    while span >= 2 {
        let stage_start = Instant::now();

        let diagonals = PackedFft2DifStageDiagonals::new(
            tile_shape,
            PackedFft2Axis::Columns,
            span,
            FftDirection::Forward,
        );

        values = values
            .iter()
            .map(|value| {
                execute_repeated_packed_fft2_dif_stage_pp(value, &diagonals, TILES_PER_CIPHERTEXT)
            })
            .collect();

        level += 1;

        println!(
            "R17_CLEAR_STAGE level={} kind=local-column span={} mode=intra-ct-simd elapsed_ms={} max_abs={:.12e}",
            level,
            span,
            stage_start.elapsed().as_millis(),
            max_abs(&values)
        );

        span /= 2;
    }

    assert_eq!(level, 20);

    let execution_ms = total_start.elapsed().as_millis();

    /*
     * Reconstruct the complete global DIF physical image and perform
     * exactly one global row/column bit-reversal conversion.
     */
    let physical = reconstruct_physical_image(&values);

    let actual = packed_physical_to_logical(&physical, image_shape);

    let (rel_l2, result_max_abs) = error_metrics(&actual, &oracle);

    let status = if rel_l2 <= 1.0e-10 && result_max_abs <= 1.0e-8 {
        "PASS"
    } else {
        "FAIL"
    };

    println!("R17_CLEAR_RESULT_BEGIN");
    println!("R17_CLEAR_TOTAL_LEVELS={level}");
    println!("R17_CLEAR_GLOBAL_ROW_STAGES=4");
    println!("R17_CLEAR_LOCAL_ROW_STAGES=6");
    println!("R17_CLEAR_GLOBAL_COLUMN_INTER_STAGES=1");
    println!("R17_CLEAR_GLOBAL_COLUMN_INTRA_STAGES=3");
    println!("R17_CLEAR_LOCAL_COLUMN_STAGES=6");
    println!("R17_CLEAR_INTER_CT_STAGE_COUNT=5");
    println!("R17_CLEAR_INTER_CT_PAIRS_PER_STAGE=16");
    println!("R17_CLEAR_INTER_CT_PHYSICAL_BUTTERFLIES=80");
    println!("R17_CLEAR_EXECUTION_MS={execution_ms}");
    println!("R17_CLEAR_REL_L2={rel_l2:.12e}");
    println!("R17_CLEAR_MAX_ABS={result_max_abs:.12e}");
    println!("R17_CLEAR_STATUS={status}");
    println!("R17_CLEAR_RESULT_END");

    assert_eq!(status, "PASS");
}
