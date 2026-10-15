//! R16.5.9: placement-aware FFT15 using simulated ciphertext slots.
//! Cleartext storage simulation only; no encryption or security claim.

use ccmm_rs::eblas::mixed_radix_fft::{mixed_radix_fft1, MixedRadixDirection, MixedRadixFft1Plan};
use num_complex::Complex64;
use std::f64::consts::PI;

const GRID_COLS: usize = 15;
const TILE_ELEMENTS: usize = 4096;
const SLOTS: usize = 8192;
const GRID_ROWS: usize = 135;

#[derive(Clone, Copy, Debug)]
enum Packing {
    Compact,
    RowAligned,
}

#[derive(Clone, Copy, Debug)]
struct Address {
    ct: usize,
    slot: usize,
}

fn address(packing: Packing, tile_row: usize, tile_col: usize, local_slot: usize) -> Address {
    assert!(tile_row < GRID_ROWS);
    assert!(tile_col < GRID_COLS);
    assert!(local_slot < TILE_ELEMENTS);

    let (ct, half) = match packing {
        Packing::Compact => {
            let tile = tile_row * GRID_COLS + tile_col;
            (tile / 2, tile % 2)
        }
        Packing::RowAligned => (
            tile_row * GRID_COLS.div_ceil(2) + tile_col / 2,
            tile_col % 2,
        ),
    };

    Address {
        ct,
        slot: half * TILE_ELEMENTS + local_slot,
    }
}

struct Buffer {
    data: Vec<Complex64>,
    packing: Packing,
}

impl Buffer {
    fn new(packing: Packing) -> Self {
        let ct_count = match packing {
            Packing::Compact => (GRID_ROWS * GRID_COLS).div_ceil(2),
            Packing::RowAligned => GRID_ROWS * GRID_COLS.div_ceil(2),
        };

        Self {
            data: vec![Complex64::new(0.0, 0.0); ct_count * SLOTS],
            packing,
        }
    }

    fn read(&self, tile_row: usize, tile_col: usize, local_slot: usize) -> Complex64 {
        let a = address(self.packing, tile_row, tile_col, local_slot);
        self.data[a.ct * SLOTS + a.slot]
    }

    fn write(&mut self, tile_row: usize, tile_col: usize, local_slot: usize, value: Complex64) {
        let a = address(self.packing, tile_row, tile_col, local_slot);
        self.data[a.ct * SLOTS + a.slot] = value;
    }
}

#[derive(Default)]
struct Metrics {
    combines: usize,
    twiddle_products: usize,
    output_writes: usize,
}

// Positions identify the physical tile columns currently owned by
// this recursion node. Each node produces a fresh output buffer.
// Output index t is stored at positions[t] in the parent layout.
fn execute_node(
    input: &Buffer,
    positions: &[usize],
    radices: &[usize],
    tile_row: usize,
    local_slot: usize,
    metrics: &mut Metrics,
) -> Buffer {
    let n = positions.len();
    let mut output = Buffer::new(input.packing);

    if n == 1 {
        assert!(radices.is_empty());
        output.write(
            tile_row,
            positions[0],
            local_slot,
            input.read(tile_row, positions[0], local_slot),
        );
        return output;
    }

    let radix = radices[0];
    assert_eq!(n % radix, 0);
    let m = n / radix;

    let branches: Vec<_> = (0..radix)
        .map(|j| {
            let branch_positions: Vec<_> = (0..m).map(|k| positions[j + radix * k]).collect();

            execute_node(
                input,
                &branch_positions,
                &radices[1..],
                tile_row,
                local_slot,
                metrics,
            )
        })
        .collect();

    for k in 0..m {
        let mut aligned = Vec::with_capacity(radix);

        for j in 0..radix {
            let source_position = positions[j + radix * k];
            let value = branches[j].read(tile_row, source_position, local_slot);

            let angle = -2.0 * PI * (j * k) as f64 / n as f64;
            let twiddle = Complex64::new(angle.cos(), angle.sin());

            if j != 0 && k != 0 {
                metrics.twiddle_products += 1;
            }

            aligned.push(value * twiddle);
        }

        for q in 0..radix {
            let mut sum = Complex64::new(0.0, 0.0);

            for (j, value) in aligned.iter().enumerate() {
                let angle = -2.0 * PI * (j * q) as f64 / radix as f64;
                sum += *value * Complex64::new(angle.cos(), angle.sin());
            }

            let destination = positions[k + m * q];
            output.write(tile_row, destination, local_slot, sum);
            metrics.output_writes += 1;
        }

        metrics.combines += 1;
    }

    output
}

#[test]
fn r16_5_9_placement_aware_fft15() {
    let plan = MixedRadixFft1Plan::new(GRID_COLS);
    assert_eq!(plan.radices(), &[3, 5]);

    for packing in [Packing::Compact, Packing::RowAligned] {
        for tile_row in [0, 1] {
            for local_slot in [0, 1, 255, 256, 4095] {
                let mut input = Buffer::new(packing);

                let values: Vec<_> = (0..GRID_COLS)
                    .map(|i| {
                        let x = i as f64 + local_slot as f64 * 0.001;
                        Complex64::new((0.17 * x).sin(), (0.11 * x).cos())
                    })
                    .collect();

                for (tile_col, &value) in values.iter().enumerate() {
                    input.write(tile_row, tile_col, local_slot, value);
                }

                let mut metrics = Metrics::default();
                let positions: Vec<_> = (0..GRID_COLS).collect();

                let output = execute_node(
                    &input,
                    &positions,
                    plan.radices(),
                    tile_row,
                    local_slot,
                    &mut metrics,
                );

                let expected = mixed_radix_fft1(&plan, MixedRadixDirection::Forward, &values);

                let max_error = (0..GRID_COLS)
                    .map(|col| (output.read(tile_row, col, local_slot) - expected[col]).norm())
                    .fold(0.0_f64, f64::max);

                assert!(
                    max_error < 1e-10,
                    "FFT15 mismatch packing={packing:?} \
                     tile_row={tile_row} local_slot={local_slot}: {max_error:e}"
                );

                assert_eq!(metrics.combines, 8);
                assert_eq!(metrics.output_writes, 30);

                println!(
                    "R16_5_9_PACKING={packing:?} ROW={tile_row} \
                     LOCAL_SLOT={local_slot} COMBINES={} \
                     WRITES={} MAX_ERROR={max_error:.9e}",
                    metrics.combines, metrics.output_writes,
                );
            }
        }
    }

    println!("R16_5_9_PLACEMENT_FFT15=PASS");
}
