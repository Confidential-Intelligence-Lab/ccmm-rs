use ccmm_rs::ckks::{research_profile_65536_for_levels, CkksParameterClass};

#[test]
fn r16_5_14b_vertical_depth_profile() {
    const HORIZONTAL_LEVELS: usize = 7;
    const VERTICAL_CROSS_TILE_LEVELS: usize = 4;
    const VERTICAL_LOCAL_LEVELS: usize = 2;
    const TERMINAL_GUARD_LIMBS: usize = 3;
    const REQUIRED_LEVELS: usize =
        HORIZONTAL_LEVELS + VERTICAL_CROSS_TILE_LEVELS + VERTICAL_LOCAL_LEVELS;

    let profile = research_profile_65536_for_levels(REQUIRED_LEVELS, TERMINAL_GUARD_LIMBS);
    let chain = profile.modulus_chain();
    let moduli = profile.modulus_values();

    assert_eq!(profile.degree(), 65_536);
    assert_eq!(profile.slot_count(), 32_768);
    assert_eq!(profile.parameter_class(), CkksParameterClass::Research);
    assert!(!profile.security_bearing());
    assert!(profile.security_model().is_none());
    assert_eq!(profile.initial_scale(), 2.0_f64.powi(55));
    assert_eq!(moduli.len(), REQUIRED_LEVELS + TERMINAL_GUARD_LIMBS);
    assert_eq!(chain.max_level(), moduli.len() - 1);
    assert!(chain.max_level() >= REQUIRED_LEVELS);

    // A level consumes one modulus limb; guard limbs remain at the endpoint.
    for level in 0..=REQUIRED_LEVELS {
        let basis = chain.level(level);
        assert_eq!(basis.moduli().len(), moduli.len() - level);
        assert_eq!(basis.moduli().len(), chain.top().moduli().len() - level);
    }
    assert_eq!(chain.level(HORIZONTAL_LEVELS).moduli().len(), 9);
    assert_eq!(
        chain
            .level(HORIZONTAL_LEVELS + VERTICAL_CROSS_TILE_LEVELS)
            .moduli()
            .len(),
        5
    );
    assert_eq!(
        chain.level(REQUIRED_LEVELS).moduli().len(),
        TERMINAL_GUARD_LIMBS
    );

    let total_bits: u32 = moduli.iter().map(|q| 64 - q.leading_zeros()).sum();
    assert_eq!(total_bits, profile.total_modulus_bits());

    println!("R16_VERTICAL_DEPTH_DEGREE={}", profile.degree());
    println!("R16_VERTICAL_DEPTH_SLOTS={}", profile.slot_count());
    println!("R16_VERTICAL_DEPTH_MODULUS_LIMBS={}", moduli.len());
    println!("R16_VERTICAL_DEPTH_TOTAL_MODULUS_BITS={total_bits}");
    println!(
        "R16_VERTICAL_DEPTH_INITIAL_SCALE={:.12e}",
        profile.initial_scale()
    );
    println!("R16_VERTICAL_DEPTH_HORIZONTAL_END={HORIZONTAL_LEVELS}");
    println!(
        "R16_VERTICAL_DEPTH_CROSS_TILE_END={}",
        HORIZONTAL_LEVELS + VERTICAL_CROSS_TILE_LEVELS
    );
    println!("R16_VERTICAL_DEPTH_FINAL_LEVEL={REQUIRED_LEVELS}");
    println!("R16_VERTICAL_DEPTH_FINAL_LIMBS={TERMINAL_GUARD_LIMBS}");
    println!("R16_VERTICAL_DEPTH_MAX_LEVEL={}", chain.max_level());
    println!(
        "R16_VERTICAL_DEPTH_SECURITY_BEARING={}",
        profile.security_bearing()
    );
    println!("R16_VERTICAL_DEPTH_STATUS=PASS");
}
