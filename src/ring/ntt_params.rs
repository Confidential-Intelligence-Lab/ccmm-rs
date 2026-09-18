use super::{Modulus, NttPlan};

/// Finds a primitive `2N`-th root of unity for a power-of-two NTT degree.
///
/// For the current correctness-oriented implementation, candidates are
/// searched deterministically from `2` upward.
///
/// # Panics
///
/// Panics if:
/// - `degree` is zero or not a power of two;
/// - `2 * degree` does not divide `q - 1`;
/// - no suitable root is found.
///
/// The modulus is expected to be prime.
pub fn find_negacyclic_root(modulus: Modulus, degree: usize) -> u64 {
    assert!(
        degree > 0 && degree.is_power_of_two(),
        "NTT degree must be a positive power of two"
    );

    let two_n = degree.checked_mul(2).expect("NTT degree is too large");

    assert_eq!(
        (modulus.value() - 1) % two_n as u64,
        0,
        "2N must divide q - 1"
    );

    // If g is any generator candidate, then
    //
    // psi = g^((q-1)/(2N))
    //
    // lies in the subgroup whose order divides 2N.
    //
    // For power-of-two N, psi^N = -1 proves that the order is exactly 2N.
    let exponent = (modulus.value() - 1) / two_n as u64;

    for generator_candidate in 2..modulus.value() {
        let psi = modulus.pow(generator_candidate, exponent);

        if modulus.pow(psi, degree as u64) == modulus.value() - 1
            && modulus.pow(psi, two_n as u64) == 1
        {
            return psi;
        }
    }

    panic!("no primitive 2N-th root found");
}

/// Constructs an `NttPlan` by discovering a suitable negacyclic root.
pub fn make_ntt_plan(modulus: Modulus, degree: usize) -> NttPlan {
    let psi = find_negacyclic_root(modulus, degree);

    NttPlan::new(modulus, degree, psi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::Polynomial;

    fn deterministic_polynomial(modulus: Modulus, degree: usize, offset: u64) -> Polynomial {
        Polynomial::new(
            modulus,
            (0..degree)
                .map(|index| {
                    (offset + 17 * index as u64 + 3 * (index as u64).pow(2)) % modulus.value()
                })
                .collect(),
        )
    }

    #[test]
    fn discovers_root_for_original_small_plan() {
        let modulus = Modulus::new(97);
        let plan = make_ntt_plan(modulus, 8);

        let psi = plan.psi();

        assert_eq!(modulus.pow(psi, 16), 1);

        assert_eq!(modulus.pow(psi, 8), 96);
    }

    #[test]
    fn radix2_matches_reference_at_degree_16() {
        validate_degree(16);
    }

    #[test]
    fn radix2_matches_reference_at_degree_32() {
        validate_degree(32);
    }

    #[test]
    fn radix2_matches_reference_at_degree_64() {
        validate_degree(64);
    }

    #[test]
    fn radix2_matches_reference_at_degree_128() {
        validate_degree(128);
    }

    #[test]
    fn radix2_matches_reference_at_degree_256() {
        validate_degree(256);
    }

    fn validate_degree(degree: usize) {
        let modulus = Modulus::new(12_289);
        let plan = make_ntt_plan(modulus, degree);

        let lhs = deterministic_polynomial(modulus, degree, 5);

        let rhs = deterministic_polynomial(modulus, degree, 29);

        // Optimized forward transform must exactly match
        // the O(N^2) reference transform.
        assert_eq!(plan.forward_radix2(&lhs), plan.forward_reference(&lhs));

        assert_eq!(plan.forward_radix2(&rhs), plan.forward_reference(&rhs));

        // Round trip.
        let lhs_ntt = plan.forward_radix2(&lhs);

        assert_eq!(plan.inverse_radix2(&lhs_ntt), lhs);

        // Optimized multiplication must match the frozen
        // naive negacyclic multiplication oracle exactly.
        let product_ntt =
            plan.pointwise_mul(&plan.forward_radix2(&lhs), &plan.forward_radix2(&rhs));

        let optimized_product = plan.inverse_radix2(&product_ntt);

        let reference_product = lhs.negacyclic_mul(&rhs);

        assert_eq!(optimized_product, reference_product);
    }
}
