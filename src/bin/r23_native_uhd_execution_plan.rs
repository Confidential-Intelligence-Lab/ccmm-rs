//! R23.2C: deterministic native UHD FFT scheduling model.
//!
//! Planning only. No encryption, key generation, or FFT execution.

const WIDTH: usize = 3840;
const HEIGHT: usize = 2160;
const TILE_WIDTH: usize = 256;
const TILE_HEIGHT: usize = 16;
const SLOTS: usize = 16384;

const H_BRANCHES: usize = WIDTH / TILE_WIDTH;
const V_GROUPS: usize = HEIGHT / TILE_HEIGHT;
const TILE_SLOTS: usize = TILE_WIDTH * TILE_HEIGHT;

#[derive(Debug, Clone, Copy)]
struct WorkUnit {
    horizontal_branch: usize,
    vertical_group: usize,
}

fn main() {
    assert_eq!(WIDTH % TILE_WIDTH, 0);
    assert_eq!(HEIGHT % TILE_HEIGHT, 0);
    assert_eq!(TILE_SLOTS, 4096);

    let total_samples = WIDTH * HEIGHT;
    let logical_tiles = H_BRANCHES * V_GROUPS;

    let units: Vec<WorkUnit> = (0..H_BRANCHES)
        .flat_map(|h| {
            (0..V_GROUPS).map(move |v| WorkUnit {
                horizontal_branch: h,
                vertical_group: v,
            })
        })
        .collect();

    assert_eq!(units.len(), logical_tiles);

    for (index, unit) in units.iter().enumerate() {
        assert_eq!(
            index,
            unit.horizontal_branch * V_GROUPS + unit.vertical_group
        );
    }

    let initial_ciphertexts_per_group = H_BRANCHES.div_ceil(2);
    let initial_ciphertexts = initial_ciphertexts_per_group * V_GROUPS;
    let minimum_ciphertexts = total_samples.div_ceil(SLOTS);

    let horizontal_output_ciphertexts = logical_tiles;
    let vertical_fft135_groups = H_BRANCHES;
    let vertical_fft16_ciphertexts = logical_tiles;

    println!("R23_2C_STATUS=PASS");
    println!("R23_2C_MODE=PLANNING_ONLY");
    println!("R23_2C_NATIVE_WIDTH={WIDTH}");
    println!("R23_2C_NATIVE_HEIGHT={HEIGHT}");
    println!("R23_2C_SAMPLES={total_samples}");
    println!("R23_2C_CKKS_SLOTS={SLOTS}");
    println!("R23_2C_TILE_SLOTS={TILE_SLOTS}");
    println!("R23_2C_HORIZONTAL_BRANCHES={H_BRANCHES}");
    println!("R23_2C_VERTICAL_GROUPS={V_GROUPS}");
    println!("R23_2C_LOGICAL_TILES={logical_tiles}");
    println!("R23_2C_INITIAL_CIPHERTEXTS={initial_ciphertexts}");
    println!("R23_2C_THEORETICAL_MIN_CIPHERTEXTS={minimum_ciphertexts}");
    println!("R23_2C_HORIZONTAL_OUTPUT_CIPHERTEXTS={horizontal_output_ciphertexts}");
    println!("R23_2C_VERTICAL_FFT135_GROUPS={vertical_fft135_groups}");
    println!("R23_2C_VERTICAL_FFT16_CIPHERTEXTS={vertical_fft16_ciphertexts}");
    println!("R23_2C_OUTPUT_BINS={}", logical_tiles * TILE_SLOTS);

    println!("R23_2C_STAGE_ORDER=PACK,FFT15,FFT256,FFT135,FFT16,OUTPUT");
    println!("R23_2C_PARALLEL_AXIS_HORIZONTAL=VERTICAL_GROUP");
    println!("R23_2C_PARALLEL_AXIS_VERTICAL=HORIZONTAL_BRANCH");
    println!("R23_2C_FFT15_SHARED_RADIX5=YES");
    println!("R23_2C_FFT256_BRANCH_SPECIFIC_OPERANDS=REQUIRED");
    println!("R23_2C_FFT16_SHARED_KEYS=REQUIRED");
    println!("R23_2C_PADDING=NONE");
    println!("R23_2C_ENCRYPTION_EXECUTED=NO");

    // R23.2D uses a symbolic limb count supplied by the caller.
    // Default 1 is a normalization unit, not the actual level-7 limb count.
    let limbs = std::env::var("R23_2D_HANDOFF_LIMBS")
        .ok()
        .map(|value| value.parse::<usize>().expect("positive limb count"))
        .unwrap_or(1);
    assert!(limbs > 0);

    let batch_size = std::env::var("R23_2D_BRANCH_BATCH")
        .ok()
        .map(|value| value.parse::<usize>().expect("valid batch size"))
        .unwrap_or(3);

    let parallel_workers = std::env::var("R23_2D_WORKERS")
        .ok()
        .map(|value| value.parse::<usize>().expect("valid worker count"))
        .unwrap_or(4);
    assert!(parallel_workers > 0);

    print_schedule("FULL", ScheduleStrategy::FullMaterialization, limbs);
    print_schedule(
        "BATCH_RECOMPUTE",
        ScheduleStrategy::BranchBatched {
            batch_size,
            retain_shared_fft15: false,
        },
        limbs,
    );
    print_schedule(
        "BATCH_RETAIN",
        ScheduleStrategy::BranchBatched {
            batch_size,
            retain_shared_fft15: true,
        },
        limbs,
    );

    println!("R23_2D_WORKERS={parallel_workers}");
    println!("R23_2D_BRANCH_BATCH_SIZE={batch_size}");
    println!("R23_2D_HANDOFF_LIMBS={limbs}");
    println!("R23_2D_MEMORY_MODEL=RLWE_PAYLOAD_ONLY");
    println!("R23_2D_RETENTION_MODEL=LOGICAL_NOT_IMPLEMENTED");
    println!("R23_2D_STATUS=PASS");
}

// R23.2D: logical scheduling and ciphertext-payload model.
// These counts are not measurements of peak process RSS.

const DEGREE: usize = 32768;
const RESIDUE_BYTES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScheduleStrategy {
    FullMaterialization,
    BranchBatched {
        batch_size: usize,
        retain_shared_fft15: bool,
    },
}

#[derive(Debug, Clone, Copy)]
struct ScheduleEstimate {
    branch_batches: usize,
    fft15_tile_row_executions: usize,
    fft256_executions: usize,
    fft135_executions: usize,
    fft16_executions: usize,
    peak_handoff_ciphertexts: usize,
    peak_live_ciphertexts_lower_bound: usize,
}

fn estimate_schedule(strategy: ScheduleStrategy) -> ScheduleEstimate {
    let batches = match strategy {
        ScheduleStrategy::FullMaterialization => 1,
        ScheduleStrategy::BranchBatched { batch_size, .. } => {
            assert!((1..=H_BRANCHES).contains(&batch_size));
            H_BRANCHES.div_ceil(batch_size)
        }
    };

    let fft15_tile_row_executions = match strategy {
        ScheduleStrategy::FullMaterialization => V_GROUPS,
        ScheduleStrategy::BranchBatched {
            retain_shared_fft15: true,
            ..
        } => V_GROUPS,
        ScheduleStrategy::BranchBatched {
            retain_shared_fft15: false,
            ..
        } => V_GROUPS * batches,
    };

    let peak_handoff_ciphertexts = match strategy {
        ScheduleStrategy::FullMaterialization => H_BRANCHES * V_GROUPS,
        ScheduleStrategy::BranchBatched { batch_size, .. } => batch_size * V_GROUPS,
    };

    ScheduleEstimate {
        branch_batches: batches,
        fft15_tile_row_executions,
        fft256_executions: H_BRANCHES * V_GROUPS,
        fft135_executions: H_BRANCHES,
        fft16_executions: H_BRANCHES * V_GROUPS,
        peak_handoff_ciphertexts,
        // A conservative scheduling floor, not total live ciphertexts.
        peak_live_ciphertexts_lower_bound: peak_handoff_ciphertexts,
    }
}

fn ciphertext_payload_bytes(rns_limbs: usize) -> usize {
    assert!(rns_limbs > 0);
    2 * DEGREE * rns_limbs * RESIDUE_BYTES
}

fn print_schedule(name: &str, strategy: ScheduleStrategy, limbs_at_handoff: usize) {
    let estimate = estimate_schedule(strategy);

    println!("R23_2D_{name}_BRANCH_BATCHES={}", estimate.branch_batches);
    println!(
        "R23_2D_{name}_FFT15_TILE_ROW_EXECUTIONS={}",
        estimate.fft15_tile_row_executions
    );
    println!(
        "R23_2D_{name}_FFT256_EXECUTIONS={}",
        estimate.fft256_executions
    );
    println!(
        "R23_2D_{name}_FFT135_EXECUTIONS={}",
        estimate.fft135_executions
    );
    println!(
        "R23_2D_{name}_FFT16_EXECUTIONS={}",
        estimate.fft16_executions
    );
    println!(
        "R23_2D_{name}_PEAK_HANDOFF_CIPHERTEXTS={}",
        estimate.peak_handoff_ciphertexts
    );
    println!(
        "R23_2D_{name}_PEAK_LIVE_CT_LOWER_BOUND={}",
        estimate.peak_live_ciphertexts_lower_bound
    );
    println!(
        "R23_2D_{name}_HANDOFF_PAYLOAD_MIB={:.3}",
        estimate.peak_handoff_ciphertexts as f64
            * ciphertext_payload_bytes(limbs_at_handoff) as f64
            / (1024.0 * 1024.0)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_dimensions_and_coverage() {
        assert_eq!(H_BRANCHES, 15);
        assert_eq!(V_GROUPS, 135);
        assert_eq!(H_BRANCHES * V_GROUPS * TILE_SLOTS, WIDTH * HEIGHT);
    }

    #[test]
    fn full_materialization_counts() {
        let estimate = estimate_schedule(ScheduleStrategy::FullMaterialization);
        assert_eq!(estimate.fft15_tile_row_executions, 135);
        assert_eq!(estimate.fft256_executions, 2025);
        assert_eq!(estimate.fft135_executions, 15);
        assert_eq!(estimate.fft16_executions, 2025);
        assert_eq!(estimate.peak_handoff_ciphertexts, 2025);
    }

    #[test]
    fn bounded_batch_recomputation_is_explicit() {
        let recompute = estimate_schedule(ScheduleStrategy::BranchBatched {
            batch_size: 3,
            retain_shared_fft15: false,
        });
        let retained = estimate_schedule(ScheduleStrategy::BranchBatched {
            batch_size: 3,
            retain_shared_fft15: true,
        });

        assert_eq!(recompute.branch_batches, 5);
        assert_eq!(recompute.fft15_tile_row_executions, 675);
        assert_eq!(retained.fft15_tile_row_executions, 135);
        assert_eq!(recompute.peak_handoff_ciphertexts, 405);
        assert_eq!(retained.peak_handoff_ciphertexts, 405);
    }

    #[test]
    fn payload_model_is_level_parameterized() {
        assert_eq!(ciphertext_payload_bytes(1), 524288);
        assert_eq!(ciphertext_payload_bytes(4), 2097152);
    }

    #[test]
    fn initial_ciphertext_accounting() {
        assert_eq!(H_BRANCHES.div_ceil(2) * V_GROUPS, 1080);
        assert_eq!((WIDTH * HEIGHT).div_ceil(SLOTS), 507);
    }
}
