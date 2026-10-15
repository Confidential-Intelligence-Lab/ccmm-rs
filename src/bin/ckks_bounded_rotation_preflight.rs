use ccmm_rs::ckks::research_profile_65536_for_levels;
use ccmm_rs::grafting::BoundedGadgetLayout;

const DEGREE: u128 = 65536;

fn main() {
    println!("R22_3D_PREFLIGHT_BEGIN");

    for limbs in [14usize, 18, 22] {
        let profile = research_profile_65536_for_levels(limbs, 0);
        let basis = profile.modulus_chain().top().clone();

        let modulus_bits: u64 = basis
            .moduli()
            .iter()
            .map(|modulus| 64 - modulus.value().leading_zeros() as u64)
            .sum();

        for base_log in [4_u32, 8, 12] {
            let layout = BoundedGadgetLayout::new(basis.clone(), base_log);
            let digits = layout.digit_count() as u128;

            let key_bytes = digits * limbs as u128 * 2 * DEGREE * 8;

            let decomposition_bytes = digits * DEGREE * 16;

            let ciphertext_bytes = limbs as u128 * 2 * DEGREE * 8;

            println!(
                "R22_3D_PREFLIGHT limbs={} modulus_bits_upper={} base_log={} digits={} key_gib={:.3} decomposition_gib={:.3} ciphertext_mib={:.3}",
                limbs,
                modulus_bits,
                base_log,
                digits,
                key_bytes as f64 / 1024_f64.powi(3),
                decomposition_bytes as f64 / 1024_f64.powi(3),
                ciphertext_bytes as f64 / 1024_f64.powi(2),
            );
        }
    }

    println!("R22_3D_PREFLIGHT_END");
}
