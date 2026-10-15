use ccmm_rs::ckks::research_profile_32768_for_levels;
use ccmm_rs::grafting::BoundedGadgetLayout;
use ccmm_rs::ring::composite_modulus_big;

#[test]
fn r16_5_18e1_memory_preflight() {
    let profile = research_profile_32768_for_levels(13, 3);
    let degree = profile.degree();
    let basis = profile.modulus_chain().top().clone();

    let q_bits = composite_modulus_big(&basis).bits();
    let layout = BoundedGadgetLayout::new(basis.clone(), 8);
    let digits = layout.digit_count();
    let q_limbs = basis.moduli().len();

    println!("R16_5_18E1_DEGREE={degree}");
    println!("R16_5_18E1_Q_LIMBS={q_limbs}");
    println!("R16_5_18E1_Q_BITS={q_bits}");
    println!("R16_5_18E1_GADGET_DIGITS={digits}");

    for p_limbs in [2usize, 3, 4] {
        let qp_limbs = q_limbs + p_limbs;
        let bytes = (digits as u128) * 2 * (qp_limbs as u128) * (degree as u128) * 8;

        println!(
            "R16_5_18E1_P_LIMBS_{p_limbs}_KEY_GIB={:.6}",
            bytes as f64 / 1073741824.0
        );
    }

    println!("R16_5_18E1_STATUS=PASS");
}
