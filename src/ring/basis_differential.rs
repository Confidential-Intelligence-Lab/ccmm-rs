use super::{
    convert_basis, drop_basis_prefix, extend_basis_prefix, Modulus, ModulusBasis, RnsNttPlan,
    RnsPolynomial,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn full_basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
            Modulus::new(114_689),
        ])
    }

    fn deterministic_coefficients(degree: usize, seed: u128, modulus: u128) -> Vec<u128> {
        (0..degree)
            .map(|index| {
                let i = index as u128;

                (seed * 97 + 13 * i + 7 * i * i + 3 * i * i * i + 11) % modulus
            })
            .collect()
    }

    fn reference_negacyclic_product(lhs: &[u128], rhs: &[u128], modulus: u128) -> Vec<u128> {
        assert_eq!(lhs.len(), rhs.len());

        let degree = lhs.len();
        let mut out = vec![0_u128; degree];

        for (i, &lhs_value) in lhs.iter().enumerate() {
            for (j, &rhs_value) in rhs.iter().enumerate() {
                let product = (lhs_value * rhs_value) % modulus;

                let raw = i + j;

                if raw < degree {
                    out[raw] = (out[raw] + product) % modulus;
                } else {
                    let index = raw - degree;

                    out[index] = (out[index] + modulus - product) % modulus;
                }
            }
        }

        out
    }

    #[test]
    fn full_stack_differential_campaign() {
        let full = full_basis();
        let mid = full.prefix(3);
        let low = full.prefix(2);

        for degree in [16_usize, 32, 64, 128] {
            let full_plan = RnsNttPlan::new(full.moduli().to_vec(), degree);

            let full_q = full.composite_modulus();

            let low_q = low.composite_modulus();

            for seed in 0_u128..16 {
                let lhs_coefficients = deterministic_coefficients(degree, seed + 1, full_q);

                let rhs_coefficients = deterministic_coefficients(degree, seed + 101, full_q);

                let lhs =
                    RnsPolynomial::from_coefficients(full.moduli().to_vec(), &lhs_coefficients);

                let rhs =
                    RnsPolynomial::from_coefficients(full.moduli().to_vec(), &rhs_coefficients);

                // -------------------------------------------------
                // 1. Multi-prime NTT product vs composite reference.
                // -------------------------------------------------

                let product = full_plan.negacyclic_mul(&lhs, &rhs);

                let expected_product =
                    reference_negacyclic_product(&lhs_coefficients, &rhs_coefficients, full_q);

                assert_eq!(
                    product.reconstruct_coefficients(),
                    expected_product,
                    "full-basis multiplication mismatch: \
                     degree={degree}, seed={seed}"
                );

                // -------------------------------------------------
                // 2. Fast drop vs reference basis conversion.
                // -------------------------------------------------

                let product_mid_fast = drop_basis_prefix(&product, &mid);

                let product_mid_reference = convert_basis(&product, &mid);

                assert_eq!(
                    product_mid_fast, product_mid_reference,
                    "mid-basis drop mismatch: \
                     degree={degree}, seed={seed}"
                );

                let product_low_fast = drop_basis_prefix(&product_mid_fast, &low);

                let product_low_reference = convert_basis(&product, &low);

                assert_eq!(
                    product_low_fast, product_low_reference,
                    "low-basis drop mismatch: \
                     degree={degree}, seed={seed}"
                );

                // -------------------------------------------------
                // 3. Fast extension vs reference canonical lift.
                // -------------------------------------------------

                let reextended_mid = extend_basis_prefix(&product_low_fast, &mid);

                let reextended_mid_reference = convert_basis(&product_low_fast, &mid);

                assert_eq!(
                    reextended_mid, reextended_mid_reference,
                    "mid-basis extension mismatch: \
                     degree={degree}, seed={seed}"
                );

                let reextended_full = extend_basis_prefix(&product_low_fast, &full);

                let reextended_full_reference = convert_basis(&product_low_fast, &full);

                assert_eq!(
                    reextended_full, reextended_full_reference,
                    "full-basis extension mismatch: \
                     degree={degree}, seed={seed}"
                );

                // -------------------------------------------------
                // 4. Low-basis canonical arithmetic semantics.
                // -------------------------------------------------

                let expected_low: Vec<u128> =
                    expected_product.iter().map(|value| value % low_q).collect();

                assert_eq!(
                    product_low_fast.reconstruct_coefficients(),
                    expected_low,
                    "low-basis canonical value mismatch: \
                     degree={degree}, seed={seed}"
                );
            }
        }
    }

    #[test]
    fn extend_drop_roundtrip_campaign() {
        let full = full_basis();
        let low = full.prefix(2);

        let low_q = low.composite_modulus();

        for degree in [16_usize, 32, 64, 128, 256] {
            for seed in 0_u128..32 {
                let coefficients = deterministic_coefficients(degree, seed, low_q);

                let original =
                    RnsPolynomial::from_coefficients(low.moduli().to_vec(), &coefficients);

                let extended = extend_basis_prefix(&original, &full);

                let recovered = drop_basis_prefix(&extended, &low);

                assert_eq!(
                    recovered, original,
                    "extend/drop roundtrip mismatch: \
                     degree={degree}, seed={seed}"
                );
            }
        }
    }

    #[test]
    fn chained_drops_match_direct_conversion() {
        let full = full_basis();
        let level1 = full.prefix(3);
        let level2 = full.prefix(2);
        let level3 = full.prefix(1);

        let full_q = full.composite_modulus();

        for seed in 0_u128..32 {
            let coefficients = deterministic_coefficients(64, seed, full_q);

            let polynomial =
                RnsPolynomial::from_coefficients(full.moduli().to_vec(), &coefficients);

            let chained = drop_basis_prefix(
                &drop_basis_prefix(&drop_basis_prefix(&polynomial, &level1), &level2),
                &level3,
            );

            let direct = convert_basis(&polynomial, &level3);

            assert_eq!(
                chained, direct,
                "chained/direct drop mismatch for seed {seed}"
            );
        }
    }
}
