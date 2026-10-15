/// Applies a negacyclic Galois automorphism to natural-order NTT values.
///
/// For odd k modulo 2N, evaluation at psi^(2j+1) after the
/// automorphism equals the original evaluation at psi^(k(2j+1)).
pub fn permute_negacyclic_ntt_galois(values: &[u64], exponent: usize) -> Vec<u64> {
    let degree = values.len();
    assert!(degree.is_power_of_two() && degree >= 2);
    let modulus = degree.checked_mul(2).expect("NTT degree overflow");
    let exponent = exponent % modulus;
    assert_eq!(exponent % 2, 1, "Galois exponent must be odd");

    (0..degree)
        .map(|index| {
            let odd_index = 2 * index + 1;
            let rotated = (exponent * odd_index) % modulus;
            values[(rotated - 1) / 2]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::{Modulus, NttPlan, Polynomial, PreparedNttPlan};

    fn mod_pow(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
        let mut result = 1_u64;
        while exponent > 0 {
            if exponent & 1 != 0 {
                result = ((result as u128 * base as u128) % modulus as u128) as u64;
            }
            base = ((base as u128 * base as u128) % modulus as u128) as u64;
            exponent >>= 1;
        }
        result
    }

    #[test]
    fn galois_permutation_matches_coefficient_automorphism() {
        for (degree, modulus) in [(8usize, 12289u64), (16, 12289), (32, 12289), (64, 12289)] {
            let modulus = Modulus::new(modulus);
            let q = modulus.value();
            let psi = mod_pow(11, (q - 1) / (2 * degree as u64), q);
            assert_eq!(mod_pow(psi, degree as u64, q), q - 1);
            assert_eq!(mod_pow(psi, 2 * degree as u64, q), 1);
            let plan = NttPlan::new(modulus, degree, psi);
            let prepared = PreparedNttPlan::new(&plan);

            let polynomial = Polynomial::new(
                modulus,
                (0..degree)
                    .map(|i| ((i * i + 17 * i + 3) as u64) % modulus.value())
                    .collect(),
            );

            let original = prepared.forward(&polynomial);

            for exponent in (1..2 * degree).step_by(2) {
                let transformed = crate::ckks::apply_automorphism(&polynomial, exponent);
                let expected = prepared.forward(&transformed);
                let observed = permute_negacyclic_ntt_galois(&original, exponent);

                assert_eq!(observed, expected, "degree={degree}, exponent={exponent}");
            }
        }
    }
}
