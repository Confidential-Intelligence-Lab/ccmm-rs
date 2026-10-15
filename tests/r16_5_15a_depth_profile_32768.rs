use ccmm_rs::ckks::{
    research_profile_32768, research_profile_32768_for_levels, CkksParameterClass,
};

#[test]
fn r16_5_15a_depth_aware_32768() {
    const HORIZONTAL: usize = 7;
    const CROSS_TILE: usize = 4;
    const LOCAL: usize = 2;
    const GUARD: usize = 3;
    const DEPTH: usize = HORIZONTAL + CROSS_TILE + LOCAL;
    let old = research_profile_32768();
    let profile = research_profile_32768_for_levels(DEPTH, GUARD);
    let chain = profile.modulus_chain();
    assert_eq!(old.modulus_values(), &profile.modulus_values()[..13]);
    assert_eq!(old.total_modulus_bits(), 650);
    assert_eq!(profile.degree(), 32768);
    assert_eq!(profile.slot_count(), 16384);
    assert_eq!(profile.initial_scale(), 2.0_f64.powi(50));
    assert_eq!(profile.parameter_class(), CkksParameterClass::Research);
    assert!(!profile.security_bearing());
    assert_eq!(profile.security_model(), None);
    assert_eq!(profile.modulus_values().len(), DEPTH + GUARD);
    assert_eq!(profile.total_modulus_bits(), 800);
    assert_eq!(chain.max_level(), 15);
    assert_eq!(chain.level(HORIZONTAL).moduli().len(), 9);
    assert_eq!(chain.level(HORIZONTAL + CROSS_TILE).moduli().len(), 5);
    assert_eq!(chain.level(DEPTH).moduli().len(), GUARD);
    for level in 0..=DEPTH {
        assert_eq!(chain.level(level).moduli().len(), 16 - level);
    }
    println!("R16_32768_DEGREE={}", profile.degree());
    println!("R16_32768_SLOTS={}", profile.slot_count());
    println!("R16_32768_LIMBS={}", profile.modulus_values().len());
    println!("R16_32768_MODULUS_BITS={}", profile.total_modulus_bits());
    println!("R16_32768_HORIZONTAL_END={HORIZONTAL}");
    println!("R16_32768_CROSS_TILE_END={}", HORIZONTAL + CROSS_TILE);
    println!("R16_32768_FINAL_LEVEL={DEPTH}");
    println!("R16_32768_TERMINAL_LIMBS={GUARD}");
    println!("R16_32768_SECURITY_BEARING={}", profile.security_bearing());
    println!("R16_32768_DEPTH_STATUS=PASS");
}
