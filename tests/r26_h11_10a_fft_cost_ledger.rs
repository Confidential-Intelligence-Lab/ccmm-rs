#[derive(Clone, Copy)]
struct Candidate {
    name: &'static str,
    outer: usize,
    inner: usize,
    outer_levels: usize,
    inner_levels: usize,
    inner_groups: usize,
    rotations_per_group: usize,
    outer_rotation_cost_known: bool,
    placement_cost_known: bool,
}

#[test]
fn compare_complete_transform_stage_budgets() {
    let candidates = [
        Candidate {
            name: "native_15x256",
            outer: 15,
            inner: 256,
            outer_levels: 2,
            inner_levels: 4,
            inner_groups: 2,
            rotations_per_group: 24,
            outer_rotation_cost_known: true,
            placement_cost_known: false,
        },
        Candidate {
            name: "native_60x64",
            outer: 60,
            inner: 64,
            outer_levels: 3,
            inner_levels: 3,
            inner_groups: 2,
            rotations_per_group: 18,
            outer_rotation_cost_known: false,
            placement_cost_known: false,
        },
    ];

    for candidate in candidates {
        assert_eq!(candidate.outer * candidate.inner, 3840);

        // One additional level for the mixed-radix interstage
        // twiddle in the currently validated 15x256 schedule.
        // The 60x64 value is a planning assumption, not a
        // validated encrypted implementation.
        let interstage_levels = 1;
        let total_levels = candidate.outer_levels + interstage_levels + candidate.inner_levels;

        let inner_rotations = candidate.inner_groups * candidate.rotations_per_group;

        println!("ARCH={}", candidate.name);
        println!("  OUTER_LEVELS={}", candidate.outer_levels);
        println!("  INTERSTAGE_LEVELS={interstage_levels}");
        println!("  INNER_LEVELS={}", candidate.inner_levels);
        println!("  TOTAL_PLANNED_LEVELS={total_levels}");
        println!("  INNER_ROTATIONS={inner_rotations}");
        println!(
            "  OUTER_ROTATION_COST_KNOWN={}",
            candidate.outer_rotation_cost_known
        );
        println!("  PLACEMENT_COST_KNOWN={}", candidate.placement_cost_known);

        assert_eq!(total_levels, 7);
    }

    println!("R26_H11_10A_DEPTH_PARITY=PASS");
    println!("R26_H11_10A_PERFORMANCE_WIN=NOT_ESTABLISHED");
    println!("R26_H11_10A_STATUS=PASS");
}
