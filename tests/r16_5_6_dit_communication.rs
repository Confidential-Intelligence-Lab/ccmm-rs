//! R16.5.6: actual recursive-DIT operand communication.
//! Structural analysis only; no ciphertext operations are executed.

use ccmm_rs::eblas::mixed_radix_fft::MixedRadixFft1Plan;

const GRID_ROWS: usize = 135;
const GRID_COLS: usize = 15;
const TILE_ROWS: usize = 16;
const TILE_COLS: usize = 256;
const TILE_ELEMENTS: usize = TILE_ROWS * TILE_COLS;

#[derive(Clone, Copy, Debug)]
enum Packing {
    Compact,
    RowAligned,
}

#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    same: u64,
    aligned: u64,
    realignment: u64,
}

impl Counts {
    fn total(self) -> u64 {
        self.same + self.aligned + self.realignment
    }
}

fn location(tile_row: usize, tile_col: usize, packing: Packing) -> (usize, usize) {
    assert!(tile_row < GRID_ROWS);
    assert!(tile_col < GRID_COLS);

    match packing {
        Packing::Compact => {
            let tile = tile_row * GRID_COLS + tile_col;
            (tile / 2, (tile % 2) * TILE_ELEMENTS)
        }
        Packing::RowAligned => (
            tile_row * GRID_COLS.div_ceil(2) + tile_col / 2,
            (tile_col % 2) * TILE_ELEMENTS,
        ),
    }
}

// Traverse the reference FFT's recursion tree.
// Each returned group identifies the branch-output positions consumed
// by one radix-r combine. The groups are emitted in execution order.
fn dit_groups(positions: &[usize], radices: &[usize], output: &mut Vec<(usize, Vec<usize>)>) {
    if positions.len() == 1 {
        assert!(radices.is_empty());
        return;
    }

    let radix = radices[0];
    let m = positions.len() / radix;
    assert_eq!(radix * m, positions.len());

    for j in 0..radix {
        let branch: Vec<usize> = (0..m).map(|k| positions[j + radix * k]).collect();

        dit_groups(&branch, &radices[1..], output);
    }

    for k in 0..m {
        let group: Vec<usize> = (0..radix).map(|j| positions[j + radix * k]).collect();

        output.push((radix, group));
    }
}

fn characterize(length: usize, horizontal: bool, packing: Packing) -> Vec<(usize, Counts)> {
    let plan = MixedRadixFft1Plan::new(length);
    let positions: Vec<usize> = (0..length).collect();
    let mut groups = Vec::new();

    dit_groups(&positions, plan.radices(), &mut groups);

    let mut results: Vec<(usize, Counts)> = Vec::new();

    for (radix, group) in groups {
        let locations: Vec<(usize, usize)> = if horizontal {
            // Examine a single physical tile row.
            // Every tile row has the same within-row classification.
            group
                .iter()
                .map(|&tile_col| location(0, tile_col, packing))
                .collect()
        } else {
            // Vertical classification can depend on tile-column parity.
            // Evaluate all tile columns below.
            Vec::new()
        };

        let classify = |locations: &[(usize, usize)]| -> usize {
            if locations.iter().all(|x| x.0 == locations[0].0) {
                0
            } else if locations.iter().all(|x| x.1 == locations[0].1) {
                1
            } else {
                2
            }
        };

        let repetitions = if horizontal {
            (GRID_ROWS * TILE_ELEMENTS) as u64
        } else {
            TILE_ELEMENTS as u64
        };

        let entry = if let Some((_, counts)) = results.iter_mut().find(|(r, _)| *r == radix) {
            counts
        } else {
            results.push((radix, Counts::default()));
            &mut results.last_mut().expect("entry inserted").1
        };

        let mut add = |class: usize| match class {
            0 => entry.same += repetitions,
            1 => entry.aligned += repetitions,
            _ => entry.realignment += repetitions,
        };

        if horizontal {
            add(classify(&locations));
        } else {
            for tile_col in 0..GRID_COLS {
                let locations: Vec<_> = group
                    .iter()
                    .map(|&tile_row| location(tile_row, tile_col, packing))
                    .collect();

                add(classify(&locations));
            }
        }
    }

    results
}

#[test]
fn r16_5_6_actual_dit_communication() {
    for packing in [Packing::Compact, Packing::RowAligned] {
        for (axis, length, horizontal) in [
            ("horizontal", GRID_COLS, true),
            ("vertical", GRID_ROWS, false),
        ] {
            let results = characterize(length, horizontal, packing);

            let mut totals = Counts::default();

            for (radix, counts) in &results {
                totals.same += counts.same;
                totals.aligned += counts.aligned;
                totals.realignment += counts.realignment;

                println!(
                    "R16_5_6_PACKING={packing:?} AXIS={axis} \
                     RADIX={radix} same={} aligned={} realignment={}",
                    counts.same, counts.aligned, counts.realignment
                );
            }

            let expected_groups = if horizontal { 4_423_680 } else { 9_953_280 };

            assert_eq!(totals.total(), expected_groups);

            if matches!(packing, Packing::RowAligned) && !horizontal {
                assert_eq!(totals.realignment, 0);
                assert_eq!(totals.aligned, expected_groups);
            }

            println!(
                "R16_5_6_TOTAL PACKING={packing:?} AXIS={axis} \
                 same={} aligned={} realignment={} total={}",
                totals.same,
                totals.aligned,
                totals.realignment,
                totals.total()
            );
        }
    }

    println!("R16_5_6_DIT_COMMUNICATION=PASS");
}
