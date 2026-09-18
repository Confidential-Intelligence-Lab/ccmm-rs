use crate::ring::Polynomial;

/// Applies the CKKS ring automorphism `sigma_k: X -> X^k`
/// in `Z_q[X] / (X^N + 1)`.
///
/// For power-of-two N, valid CKKS Galois automorphisms use odd
/// exponents k modulo 2N.
pub fn apply_automorphism(polynomial: &Polynomial, exponent: usize) -> Polynomial {
    let degree = polynomial.degree();

    assert!(
        degree >= 2 && degree.is_power_of_two(),
        "CKKS automorphism requires a power-of-two ring degree >= 2"
    );

    let two_n = 2 * degree;

    let exponent = exponent % two_n;

    assert!(
        exponent % 2 == 1,
        "CKKS automorphism exponent must be odd modulo 2N"
    );

    let modulus = polynomial.modulus();

    let mut output = vec![0_u64; degree];

    for (source_index, &coefficient) in polynomial.coefficients().iter().enumerate() {
        let mapped_exponent = (source_index * exponent) % two_n;

        let (target_index, negate) = if mapped_exponent < degree {
            (mapped_exponent, false)
        } else {
            (mapped_exponent - degree, true)
        };

        output[target_index] = if negate {
            modulus.neg(coefficient)
        } else {
            coefficient
        };
    }

    Polynomial::new(modulus, output)
}

/// Returns the inverse automorphism exponent modulo 2N.
///
/// For valid odd exponents and power-of-two N, the inverse always exists.
pub fn inverse_automorphism_exponent(degree: usize, exponent: usize) -> usize {
    assert!(
        degree >= 2 && degree.is_power_of_two(),
        "CKKS automorphism requires a power-of-two ring degree >= 2"
    );

    let modulus = 2 * degree;

    let exponent = exponent % modulus;

    assert!(
        exponent % 2 == 1,
        "CKKS automorphism exponent must be odd modulo 2N"
    );

    (1..modulus)
        .step_by(2)
        .find(|&candidate| (exponent * candidate) % modulus == 1)
        .expect("valid CKKS automorphism exponent must have an inverse modulo 2N")
}

/// Describes the action of a ring automorphism on one exposed
/// canonical CKKS slot.
///
/// `source_slot` identifies the exposed slot whose value appears at
/// the destination. If `conjugate` is true, its complex conjugate
/// appears instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CkksSlotAutomorphism {
    source_slot: usize,
    conjugate: bool,
}

impl CkksSlotAutomorphism {
    pub fn source_slot(&self) -> usize {
        self.source_slot
    }

    pub fn conjugates(&self) -> bool {
        self.conjugate
    }
}

/// Returns the canonical-slot action induced by `sigma_k`.
///
/// Entry `j` describes the value observed at exposed destination slot
/// `j` after applying the ring automorphism.
///
/// With canonical roots
///
/// `xi_j = zeta^(2j + 1)`,
///
/// the identity
///
/// `sigma_k(f)(xi_j) = f(xi_j^k)`
///
/// determines the source root. Roots in the second half correspond to
/// conjugates of the exposed first-half slots.
pub fn canonical_slot_automorphism(degree: usize, exponent: usize) -> Vec<CkksSlotAutomorphism> {
    assert!(
        degree >= 2 && degree.is_power_of_two(),
        "CKKS slot automorphism requires a power-of-two ring degree >= 2"
    );

    let two_n = 2 * degree;

    let exponent = exponent % two_n;

    assert!(
        exponent % 2 == 1,
        "CKKS automorphism exponent must be odd modulo 2N"
    );

    let slot_count = degree / 2;

    /*
     * Logical slot j corresponds to root exponent
     *
     *     r_j = 5^j mod 2N.
     *
     * Build that orbit explicitly so this slot-action routine uses
     * exactly the same convention as CkksCanonicalEmbedding.
     */
    let mut slot_exponents = Vec::with_capacity(slot_count);

    let mut root_exponent = 1_usize;

    for _ in 0..slot_count {
        slot_exponents.push(root_exponent);

        root_exponent = (root_exponent * 5) % two_n;
    }

    slot_exponents
        .iter()
        .map(|&destination_root| {
            let source_root = (destination_root * exponent) % two_n;

            for (source_slot, &logical_root) in slot_exponents.iter().enumerate() {
                if source_root == logical_root {
                    return CkksSlotAutomorphism {
                        source_slot,
                        conjugate: false,
                    };
                }

                if source_root == (two_n - logical_root) % two_n {
                    return CkksSlotAutomorphism {
                        source_slot,
                        conjugate: true,
                    };
                }
            }

            unreachable!("valid CKKS Galois action must map to a logical slot or its conjugate")
        })
        .collect()
}

/// Returns the Galois exponent implementing a logical CKKS
/// left rotation by `steps` slots.
///
/// Logical slots are ordered by the root orbit
///
/// `1, 5, 5^2, ..., 5^(N/2-1) mod 2N`.
///
/// Therefore `sigma_(5^r)` rotates the logical slot vector left
/// by `r` positions.
pub fn rotation_exponent_left(degree: usize, steps: usize) -> usize {
    assert!(
        degree >= 2 && degree.is_power_of_two(),
        "CKKS degree must be a power of two at least 2"
    );

    let slot_count = degree / 2;

    let steps = steps % slot_count;

    modular_pow(5, steps, 2 * degree)
}

/// Returns the Galois exponent implementing a logical CKKS
/// right rotation by `steps` slots.
pub fn rotation_exponent_right(degree: usize, steps: usize) -> usize {
    assert!(
        degree >= 2 && degree.is_power_of_two(),
        "CKKS degree must be a power of two at least 2"
    );

    let slot_count = degree / 2;

    let steps = steps % slot_count;

    let left_steps = (slot_count - steps) % slot_count;

    rotation_exponent_left(degree, left_steps)
}

fn modular_pow(mut base: usize, mut exponent: usize, modulus: usize) -> usize {
    let mut result = 1_usize;

    base %= modulus;

    while exponent > 0 {
        if exponent & 1 == 1 {
            result = (result * base) % modulus;
        }

        base = (base * base) % modulus;

        exponent >>= 1;
    }

    result
}

/// Returns the Galois exponent implementing logical CKKS complex
/// conjugation.
///
/// For ring degree `N`, complex conjugation corresponds to
///
/// `sigma_{-1}`, i.e. exponent `2N - 1` modulo `2N`.
pub fn conjugation_exponent(degree: usize) -> usize {
    assert!(
        degree >= 2 && degree.is_power_of_two(),
        "CKKS degree must be a power of two at least 2"
    );

    2 * degree - 1
}

#[cfg(test)]
mod tests {
    use crate::ring::Modulus;

    use super::*;

    fn polynomial() -> Polynomial {
        Polynomial::new(Modulus::new(97), vec![1, 2, 3, 4, 5, 6, 7, 8])
    }

    #[test]
    fn identity_automorphism_is_exact() {
        let input = polynomial();

        assert_eq!(apply_automorphism(&input, 1,), input);
    }

    #[test]
    fn exponent_is_reduced_modulo_2n() {
        let input = polynomial();

        assert_eq!(
            apply_automorphism(&input, 3,),
            apply_automorphism(&input, 19,)
        );
    }

    #[test]
    fn automorphism_matches_manual_negacyclic_mapping() {
        let input = polynomial();

        let modulus = input.modulus();

        /*
         * N = 8, k = 3.
         *
         * source exponent i -> 3i mod 16:
         *
         * 0 -> 0
         * 1 -> 3
         * 2 -> 6
         * 3 -> 9  -> -X^1
         * 4 -> 12 -> -X^4
         * 5 -> 15 -> -X^7
         * 6 -> 2
         * 7 -> 5
         */
        let expected = Polynomial::new(
            modulus,
            vec![
                1,
                modulus.neg(4),
                7,
                2,
                modulus.neg(5),
                8,
                3,
                modulus.neg(6),
            ],
        );

        assert_eq!(apply_automorphism(&input, 3,), expected);
    }

    #[test]
    fn every_valid_exponent_is_permutation_up_to_sign() {
        let degree = 16;

        let modulus = Modulus::new(97);

        let input = Polynomial::new(modulus, (1..=degree).map(|value| value as u64).collect());

        for exponent in (1..2 * degree).step_by(2) {
            let output = apply_automorphism(&input, exponent);

            let mut magnitudes: Vec<_> = output
                .coefficients()
                .iter()
                .map(|&value| {
                    if value > modulus.value() / 2 {
                        modulus.value() - value
                    } else {
                        value
                    }
                })
                .collect();

            magnitudes.sort_unstable();

            assert_eq!(
                magnitudes,
                (1..=degree)
                    .map(|value| { value as u64 })
                    .collect::<Vec<_>>(),
                "exponent={exponent}"
            );
        }
    }

    #[test]
    fn composition_matches_exponent_product() {
        let input = polynomial();

        let degree = input.degree();

        let first = apply_automorphism(&input, 3);

        let composed = apply_automorphism(&first, 5);

        let direct = apply_automorphism(&input, (3 * 5) % (2 * degree));

        assert_eq!(composed, direct);
    }

    #[test]
    fn inverse_exponent_recovers_original_polynomial() {
        let input = polynomial();

        for exponent in [1_usize, 3, 5, 7, 9, 11, 13, 15] {
            let inverse = inverse_automorphism_exponent(input.degree(), exponent);

            let mapped = apply_automorphism(&input, exponent);

            let recovered = apply_automorphism(&mapped, inverse);

            assert_eq!(recovered, input, "exponent={exponent}, inverse={inverse}");
        }
    }

    #[test]
    fn inverse_exponent_is_correct_modulo_2n() {
        let degree = 32;

        for exponent in (1..2 * degree).step_by(2) {
            let inverse = inverse_automorphism_exponent(degree, exponent);

            assert_eq!((exponent * inverse) % (2 * degree), 1);
        }
    }

    #[test]
    #[should_panic(expected = "CKKS automorphism exponent must be odd modulo 2N")]
    fn rejects_even_exponent() {
        let input = polynomial();

        let _ = apply_automorphism(&input, 2);
    }

    #[test]
    #[should_panic(expected = "CKKS automorphism exponent must be odd modulo 2N")]
    fn inverse_rejects_even_exponent() {
        let _ = inverse_automorphism_exponent(8, 6);
    }

    #[test]
    fn classifies_galois_actions_on_current_slot_order() {
        for degree in [8_usize, 16, 32] {
            for exponent in (1..2 * degree).step_by(2) {
                let mapping = canonical_slot_automorphism(degree, exponent);

                let conjugated = mapping.iter().filter(|action| action.conjugates()).count();

                println!(
                    "degree={degree}, exponent={exponent}, \
                     conjugated={conjugated}/{}, mapping={mapping:?}",
                    degree / 2,
                );
            }
        }
    }

    #[test]
    fn rotation_exponents_match_power_of_five_orbit() {
        assert_eq!(rotation_exponent_left(8, 0,), 1);

        assert_eq!(rotation_exponent_left(8, 1,), 5);

        assert_eq!(rotation_exponent_left(8, 2,), 9);

        assert_eq!(rotation_exponent_left(8, 3,), 13);
    }

    #[test]
    fn rotation_steps_reduce_modulo_slot_count() {
        for degree in [8_usize, 16, 32] {
            let slots = degree / 2;

            for steps in 0..2 * slots {
                assert_eq!(
                    rotation_exponent_left(degree, steps,),
                    rotation_exponent_left(degree, steps % slots,)
                );

                assert_eq!(
                    rotation_exponent_right(degree, steps,),
                    rotation_exponent_right(degree, steps % slots,)
                );
            }
        }
    }

    #[test]
    fn left_and_right_rotation_exponents_are_inverses() {
        for degree in [8_usize, 16, 32] {
            let modulus = 2 * degree;

            for steps in 0..degree / 2 {
                let left = rotation_exponent_left(degree, steps);

                let right = rotation_exponent_right(degree, steps);

                assert_eq!(
                    (left * right) % modulus,
                    1,
                    "degree={degree}, steps={steps}"
                );
            }
        }
    }

    #[test]
    fn left_rotation_exponent_has_expected_slot_action() {
        for degree in [8_usize, 16, 32] {
            let slot_count = degree / 2;

            for steps in 0..slot_count {
                let exponent = rotation_exponent_left(degree, steps);

                let action = canonical_slot_automorphism(degree, exponent);

                for (destination, mapping) in action.iter().enumerate() {
                    assert!(
                        !mapping.conjugates(),
                        "rotation unexpectedly conjugates: \
                         degree={degree}, steps={steps}, \
                         destination={destination}"
                    );

                    assert_eq!(
                        mapping.source_slot(),
                        (destination + steps) % slot_count,
                        "degree={degree}, steps={steps}, \
                         destination={destination}"
                    );
                }
            }
        }
    }

    #[test]
    fn conjugation_exponent_is_minus_one_modulo_2n() {
        for degree in [2_usize, 4, 8, 16, 32] {
            assert_eq!(conjugation_exponent(degree,), 2 * degree - 1);
        }
    }

    #[test]
    fn conjugation_slot_action_is_componentwise_conjugation() {
        for degree in [8_usize, 16, 32] {
            let action = canonical_slot_automorphism(degree, conjugation_exponent(degree));

            for (destination, mapping) in action.iter().enumerate() {
                assert_eq!(mapping.source_slot(), destination);

                assert!(
                    mapping.conjugates(),
                    "degree={degree}, destination={destination}"
                );
            }
        }
    }

    #[test]
    fn conjugation_is_self_inverse() {
        for degree in [8_usize, 16, 32] {
            let exponent = conjugation_exponent(degree);

            assert_eq!((exponent * exponent) % (2 * degree), 1);
        }
    }
}
