#[derive(Debug, Clone, Copy)]
enum Axis {
    Rows,
    Columns,
}

#[derive(Debug, Clone, Copy)]
struct Packing {
    rows: usize,
    cols: usize,
}

impl Packing {
    fn new(rows: usize, cols: usize) -> Self {
        assert!(rows.is_power_of_two());
        assert!(cols.is_power_of_two());
        assert_eq!(rows * cols, 8);

        Self { rows, cols }
    }

    fn name(self) -> String {
        format!("{}x{}", self.rows, self.cols)
    }

    fn ciphertext_and_lane(self, tile_row: usize, tile_col: usize) -> (usize, usize) {
        const GRID: usize = 16;

        assert!(GRID % self.rows == 0);
        assert!(GRID % self.cols == 0);

        let ct_rows = GRID / self.rows;
        let ct_cols = GRID / self.cols;

        let ct_row = tile_row / self.rows;
        let ct_col = tile_col / self.cols;

        let local_row = tile_row % self.rows;
        let local_col = tile_col % self.cols;

        let ciphertext = ct_row * ct_cols + ct_col;
        let lane = local_row * self.cols + local_col;

        assert!(ciphertext < ct_rows * ct_cols);
        assert!(lane < 8);

        (ciphertext, lane)
    }
}

fn characterize_stage(packing: Packing, axis: Axis, offset: usize) -> (usize, usize, usize, usize) {
    const GRID: usize = 16;

    let mut intra = 0usize;
    let mut inter = 0usize;
    let mut lane_aligned_inter = 0usize;
    let mut intra_lane_distance = 0usize;

    match axis {
        Axis::Rows => {
            for row in 0..GRID {
                for col in 0..GRID {
                    if col & offset != 0 {
                        continue;
                    }

                    let partner_col = col + offset;

                    let (ct_a, lane_a) = packing.ciphertext_and_lane(row, col);
                    let (ct_b, lane_b) = packing.ciphertext_and_lane(row, partner_col);

                    if ct_a == ct_b {
                        intra += 1;
                        intra_lane_distance = intra_lane_distance.max(lane_a.abs_diff(lane_b));
                    } else {
                        inter += 1;

                        if lane_a == lane_b {
                            lane_aligned_inter += 1;
                        }
                    }
                }
            }
        }

        Axis::Columns => {
            for row in 0..GRID {
                if row & offset != 0 {
                    continue;
                }

                let partner_row = row + offset;

                for col in 0..GRID {
                    let (ct_a, lane_a) = packing.ciphertext_and_lane(row, col);
                    let (ct_b, lane_b) = packing.ciphertext_and_lane(partner_row, col);

                    if ct_a == ct_b {
                        intra += 1;
                        intra_lane_distance = intra_lane_distance.max(lane_a.abs_diff(lane_b));
                    } else {
                        inter += 1;

                        if lane_a == lane_b {
                            lane_aligned_inter += 1;
                        }
                    }
                }
            }
        }
    }

    assert_eq!(intra + inter, 128);

    (intra, inter, lane_aligned_inter, intra_lane_distance)
}

fn main() {
    const TILE_ELEMENTS: usize = 64 * 64;

    println!("FFT_PACKING_SCHEDULE_BEGIN");
    println!("FFT_PACKING_GRID=16x16");
    println!("FFT_PACKING_LOGICAL_TILES=256");
    println!("FFT_PACKING_TILES_PER_CIPHERTEXT=8");
    println!("FFT_PACKING_CIPHERTEXT_COUNT=32");
    println!("FFT_PACKING_TILE_ELEMENTS={TILE_ELEMENTS}");
    println!("FFT_PACKING_SLOT_COUNT={}", 8 * TILE_ELEMENTS);

    let packings = [
        Packing::new(1, 8),
        Packing::new(2, 4),
        Packing::new(4, 2),
        Packing::new(8, 1),
    ];

    for packing in packings {
        println!("FFT_PACKING_CANDIDATE_BEGIN packing={}", packing.name());

        let mut total_intra = 0usize;
        let mut total_inter = 0usize;
        let mut total_lane_aligned_inter = 0usize;

        /*
         * Tile-grid DIF stage offsets for a 16-element axis:
         *
         * span 16 -> offset 8
         * span  8 -> offset 4
         * span  4 -> offset 2
         * span  2 -> offset 1
         */
        for (axis, axis_name) in [(Axis::Rows, "row"), (Axis::Columns, "column")] {
            for offset in [8usize, 4, 2, 1] {
                let span = 2 * offset;

                let (intra, inter, lane_aligned_inter, intra_lane_distance) =
                    characterize_stage(packing, axis, offset);

                total_intra += intra;
                total_inter += inter;
                total_lane_aligned_inter += lane_aligned_inter;

                let intra_slot_rotation = intra_lane_distance * TILE_ELEMENTS;

                println!(
                    "FFT_PACKING_STAGE packing={} axis={} span_tiles={} offset_tiles={} intra_butterflies={} inter_butterflies={} lane_aligned_inter={} intra_lane_distance={} intra_slot_rotation={}",
                    packing.name(),
                    axis_name,
                    span,
                    offset,
                    intra,
                    inter,
                    lane_aligned_inter,
                    intra_lane_distance,
                    intra_slot_rotation,
                );
            }
        }

        println!(
            "FFT_PACKING_SUMMARY packing={} intra_butterflies={} inter_butterflies={} lane_aligned_inter={}",
            packing.name(),
            total_intra,
            total_inter,
            total_lane_aligned_inter,
        );

        println!("FFT_PACKING_CANDIDATE_END packing={}", packing.name());
    }

    println!("FFT_PACKING_SCHEDULE_END");
}
