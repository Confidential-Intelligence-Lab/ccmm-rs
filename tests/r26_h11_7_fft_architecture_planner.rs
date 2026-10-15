const NATIVE_N: usize = 3840;
const PADDED_N: usize = 4096;
const CKKS_SLOTS: usize = 32768;

#[derive(Debug, Clone, Copy)]
struct Architecture {
    name: &'static str,
    outer: usize,
    inner: usize,
    native: bool,
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

fn radix4_stages(mut n: usize) -> usize {
    let mut stages = 0;
    while n > 1 {
        assert_eq!(n % 4, 0, "inner dimension must be power of four");
        n /= 4;
        stages += 1;
    }
    stages
}

fn report(a: Architecture) {
    let length = a.outer * a.inner;
    let stages = radix4_stages(a.inner);

    // Preserve 16 transforms per lane when capacity permits.
    // Otherwise use the largest feasible power-of-two count
    // not exceeding 16.
    let max_transforms = CKKS_SLOTS / a.inner;
    assert!(max_transforms > 0);

    let transforms_per_lane = 16.min(max_transforms);
    assert!(transforms_per_lane.is_power_of_two());

    let lane_slots = transforms_per_lane * a.inner;
    assert_eq!(CKKS_SLOTS % lane_slots, 0);

    let lanes_per_ciphertext = CKKS_SLOTS / lane_slots;
    let branch_groups = ceil_div(a.outer, lanes_per_ciphertext);
    let branch_capacity = branch_groups * lanes_per_ciphertext;
    let occupancy = a.outer as f64 / branch_capacity as f64;

    // Each radix-4 stage has six nonzero offsets in the
    // currently validated diagonal construction.
    let rotations_per_stage = 6;
    let rotations_per_group = rotations_per_stage * stages;
    let total_rotations = rotations_per_group * branch_groups;

    // A radix-4 stage consumes one CKKS level in the
    // validated FFT256 implementation. This is only a
    // structural estimate for alternative kernels.
    let estimated_inner_levels = stages;

    println!("ARCH={}", a.name);
    println!("  LENGTH={length}");
    println!("  NATIVE={}", a.native);
    println!("  OUTER_BRANCHES={}", a.outer);
    println!("  INNER_FFT={}", a.inner);
    println!("  TRANSFORMS_PER_LANE={transforms_per_lane}");
    println!("  LANE_SLOTS={lane_slots}");
    println!("  LANES_PER_CIPHERTEXT={lanes_per_ciphertext}");
    println!("  BRANCH_GROUPS={branch_groups}");
    println!("  BRANCH_CAPACITY={branch_capacity}");
    println!("  BRANCH_OCCUPANCY={occupancy:.6}");
    println!("  RADIX4_STAGES={stages}");
    println!("  INNER_ROTATIONS_PER_GROUP={rotations_per_group}");
    println!("  INNER_TOTAL_ROTATIONS={total_rotations}");
    println!("  ESTIMATED_INNER_LEVELS={estimated_inner_levels}");
    println!("  OUTER_DEPTH=NOT_MODELED");
    println!("  PLACEMENT_COST=NOT_MODELED");
    println!("  KEYGEN_COST=NOT_MODELED");
    println!("  FULL_FFT_DEPTH=NOT_MODELED");
}

#[test]
fn compare_native_and_padded_fft_architectures() {
    let architectures = [
        Architecture {
            name: "native_15x256",
            outer: 15,
            inner: 256,
            native: true,
        },
        Architecture {
            name: "native_60x64",
            outer: 60,
            inner: 64,
            native: true,
        },
        Architecture {
            name: "padded_1x4096",
            outer: 1,
            inner: 4096,
            native: false,
        },
    ];

    for architecture in architectures {
        let expected_length = if architecture.native {
            NATIVE_N
        } else {
            PADDED_N
        };
        assert_eq!(architecture.outer * architecture.inner, expected_length);
        report(architecture);
    }

    println!(
        "PADDING_OVERHEAD={:.6}",
        PADDED_N as f64 / NATIVE_N as f64 - 1.0
    );
    println!("R26_H11_7_ARCHITECTURE_STATUS=PASS");
}
