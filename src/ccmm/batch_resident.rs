//! Internal NTT-resident Batch CCMM primitives.
//!
//! R30.17g: mechanically extracted from the existing
//! exact-equivalence-tested experimental implementations.
//!
//! This module does not change the production CCMM dispatch.

fn batch_inverse_cyclic_ntt_radix2(values: &mut [u64], modulus: crate::ring::Modulus, root: u64) {
    let dimension = values.len();

    assert!(
        dimension > 0 && dimension.is_power_of_two(),
        "cross-frequency inverse NTT dimension must be a positive power of two"
    );

    // Bit-reverse permutation.
    let bits = dimension.trailing_zeros();

    for index in 0..dimension {
        let reversed = index.reverse_bits() >> (usize::BITS - bits);

        if reversed > index {
            values.swap(index, reversed);
        }
    }

    let root_inverse = modulus.inverse_prime(root);

    let mut len = 2_usize;

    while len <= dimension {
        let half = len / 2;

        let root_step = modulus.pow(root_inverse, (dimension / len) as u64);

        let mut twiddles = Vec::with_capacity(half);

        let mut twiddle = 1_u64;

        for _ in 0..half {
            twiddles.push(twiddle);

            twiddle = modulus.mul(twiddle, root_step);
        }

        for start in (0..dimension).step_by(len) {
            for (offset, &twiddle) in twiddles.iter().enumerate() {
                let even = values[start + offset];

                let odd = modulus.mul(values[start + offset + half], twiddle);

                values[start + offset] = modulus.add_canonical(even, odd);

                values[start + offset + half] = modulus.sub_canonical(even, odd);
            }
        }

        len *= 2;
    }

    let dimension_inverse = modulus.inverse_prime(dimension as u64);

    for value in values {
        *value = modulus.mul(*value, dimension_inverse);
    }
}

pub(super) fn batch_large_ntt_to_scalar_ntt_radix2(
    large_ntt: &crate::ring::RnsNttPolynomial,
    dimension: usize,
    large_plan: &crate::ring::RnsNttPlan,
    scalar_plan: &crate::ring::RnsNttPlan,
) -> Vec<crate::ring::RnsNttPolynomial> {
    assert!(dimension > 0);
    assert!(dimension.is_power_of_two());

    let large_degree = large_ntt.degree();

    assert_eq!(
        large_degree,
        large_plan.degree(),
        "large NTT polynomial degree must match large plan"
    );

    assert_eq!(
        large_degree % dimension,
        0,
        "large degree must be divisible by Batch dimension"
    );

    let scalar_degree = large_degree / dimension;

    assert_eq!(
        scalar_degree,
        scalar_plan.degree(),
        "scalar plan degree must match N / d"
    );

    assert_eq!(
        large_plan.moduli(),
        scalar_plan.moduli(),
        "large and scalar plans must use the same RNS basis"
    );

    let mut output_values =
        vec![vec![vec![0_u64; scalar_degree]; large_plan.moduli().len()]; dimension];

    for limb_index in 0..large_plan.moduli().len() {
        let modulus = large_plan.moduli()[limb_index];

        let large_limb_plan = large_plan.plan(limb_index);

        let scalar_limb_plan = scalar_plan.plan(limb_index);

        let psi_large = large_limb_plan.psi();

        let psi_scalar = scalar_limb_plan.psi();

        let derived_scalar = modulus.pow(psi_large, dimension as u64);

        let two_ns = 2 * scalar_degree;

        let root_exponent = (0..two_ns)
            .step_by(2)
            .map(|offset| offset + 1)
            .find(|&candidate| modulus.pow(derived_scalar, candidate as u64) == psi_scalar)
            .expect("scalar NTT root must belong to derived large-root subgroup");

        let rho = modulus.pow(psi_large, (2 * scalar_degree) as u64);

        let large_values = large_ntt.residue(limb_index).values();

        for scalar_frequency in 0..scalar_degree {
            let odd = (root_exponent * (2 * scalar_frequency + 1)) % (2 * scalar_degree);

            assert_eq!(
                odd % 2,
                1,
                "mapped negacyclic evaluation exponent must remain odd"
            );

            let k0 = (odd - 1) / 2;

            let z0 = modulus.pow(psi_large, (2 * k0 + 1) as u64);

            let mut cross_frequency = Vec::with_capacity(dimension);

            for m in 0..dimension {
                cross_frequency.push(large_values[k0 + m * scalar_degree]);
            }

            batch_inverse_cyclic_ntt_radix2(&mut cross_frequency, modulus, rho);

            let z0_inverse = modulus.inverse_prime(z0);

            let mut untwiddle = 1_u64;

            for (row, row_output) in output_values.iter_mut().enumerate() {
                if row > 0 {
                    untwiddle = modulus.mul(untwiddle, z0_inverse);
                }

                row_output[limb_index][scalar_frequency] =
                    modulus.mul(cross_frequency[row], untwiddle);
            }
        }
    }

    output_values
        .into_iter()
        .map(|row_limbs| {
            let residues = row_limbs
                .into_iter()
                .enumerate()
                .map(|(limb_index, values)| {
                    crate::ring::NttPolynomial::from_canonical_values(
                        scalar_plan.plan(limb_index),
                        values,
                    )
                })
                .collect();

            crate::ring::RnsNttPolynomial::from_residues(residues)
        })
        .collect()
}

pub(super) fn batch_ccmm_slice_ranked_resident_ntt(
    ciphertexts: &[crate::grafting::RnsNttRlweCiphertext],
    dimension: usize,
    large_plan: &crate::ring::RnsNttPlan,
    scalar_plan: &crate::ring::RnsNttPlan,
) -> Vec<Vec<crate::ring::RnsNttPolynomial>> {
    assert!(!ciphertexts.is_empty());
    assert!(dimension > 0);

    let sliced: Vec<_> = ciphertexts
        .iter()
        .map(|ciphertext| {
            assert_eq!(
                ciphertext.degree(),
                large_plan.degree(),
                "resident Batch CCMM ciphertext degree must match large NTT plan"
            );

            assert_eq!(
                ciphertext.moduli(),
                large_plan.moduli(),
                "resident Batch CCMM ciphertext basis must match large NTT plan"
            );

            let b_rows = batch_large_ntt_to_scalar_ntt_radix2(
                ciphertext.b(),
                dimension,
                large_plan,
                scalar_plan,
            );

            let a_rows = batch_large_ntt_to_scalar_ntt_radix2(
                ciphertext.a(),
                dimension,
                large_plan,
                scalar_plan,
            );

            assert_eq!(b_rows.len(), dimension);
            assert_eq!(a_rows.len(), dimension);

            (b_rows, a_rows)
        })
        .collect();

    (0..2 * dimension)
        .map(|ranked_row| {
            sliced
                .iter()
                .map(|(b_rows, a_rows)| {
                    if ranked_row < dimension {
                        b_rows[ranked_row].clone()
                    } else {
                        a_rows[ranked_row - dimension].clone()
                    }
                })
                .collect()
        })
        .collect()
}

pub(super) fn batch_ccmm_modular_product_delayed_ntt_rhs_resident(
    lhs: &[Vec<crate::ring::RnsPolynomial>],
    rhs_transposed_ntt: &[Vec<crate::ring::RnsNttPolynomial>],
    plan: &crate::ring::PreparedRnsNttPlan,
) -> Vec<Vec<crate::ring::RnsPolynomial>> {
    let block = std::env::var("CCMM_GEMM_BLOCK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(128);

    assert!(block > 0, "CCMM GEMM block size must be nonzero");

    assert!(!lhs.is_empty(), "Batch CCMM lhs must not be empty");
    assert!(
        !rhs_transposed_ntt.is_empty(),
        "Batch CCMM rhs must not be empty"
    );

    let output_rows = lhs.len();
    let inner = lhs[0].len();

    assert!(inner > 0);
    assert!(lhs.iter().all(|row| row.len() == inner));

    assert_eq!(
        rhs_transposed_ntt.len(),
        inner,
        "Batch CCMM modular inner dimensions must match"
    );

    let output_columns = rhs_transposed_ntt[0].len();

    assert!(output_columns > 0);
    assert!(rhs_transposed_ntt
        .iter()
        .all(|row| row.len() == output_columns));

    let degree = plan.degree();
    let limb_count = plan.moduli().len();

    // Transform each polynomial matrix entry exactly once.
    let forward_start = std::time::Instant::now();

    let lhs_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = lhs
        .iter()
        .map(|row| row.iter().map(|entry| plan.forward(entry)).collect())
        .collect();

    let rhs_ntt = rhs_transposed_ntt;

    let forward_elapsed = forward_start.elapsed();

    let output_layer_size = output_rows * output_columns;

    // [limb][coordinate][row * output_columns + column]
    let mut output_layers = vec![vec![vec![0_u64; output_layer_size]; degree]; limb_count];

    let mut materialize_elapsed = std::time::Duration::ZERO;
    let mut gemm_elapsed = std::time::Duration::ZERO;

    let mut limb_materialize_elapsed = vec![std::time::Duration::ZERO; limb_count];

    let mut limb_gemm_elapsed = vec![std::time::Duration::ZERO; limb_count];

    let mut accumulate_elapsed = std::time::Duration::ZERO;

    let mut reduction_elapsed = std::time::Duration::ZERO;

    let mut limb_accumulate_elapsed = vec![std::time::Duration::ZERO; limb_count];

    let mut limb_reduction_elapsed = vec![std::time::Duration::ZERO; limb_count];

    for (limb_index, output_limb_layers) in output_layers.iter_mut().enumerate().take(limb_count) {
        let modulus = plan.moduli()[limb_index];
        let q = u128::from(modulus.value());

        let max_product = (q - 1) * (q - 1);

        max_product
            .checked_mul(inner as u128)
            .expect("CCMM delayed-reduction dot product must fit in u128");

        for (coordinate, output_layer) in output_limb_layers.iter_mut().enumerate().take(degree) {
            let materialize_start = std::time::Instant::now();

            let mut lhs_layer = vec![0_u64; output_rows * inner];

            let mut rhs_layer = vec![0_u64; inner * output_columns];

            for row in 0..output_rows {
                for k in 0..inner {
                    lhs_layer[row * inner + k] =
                        lhs_ntt[row][k].residue(limb_index).values()[coordinate];
                }
            }

            for k in 0..inner {
                for column in 0..output_columns {
                    rhs_layer[k * output_columns + column] =
                        rhs_ntt[k][column].residue(limb_index).values()[coordinate];
                }
            }

            let materialize_coordinate_elapsed = materialize_start.elapsed();

            materialize_elapsed += materialize_coordinate_elapsed;

            limb_materialize_elapsed[limb_index] += materialize_coordinate_elapsed;

            let mut wide = vec![0_u128; output_layer_size];

            let gemm_start = std::time::Instant::now();

            let accumulate_start = std::time::Instant::now();

            // Dense blocked i-k-j multiplication with no modular
            // reduction in the cubic loop.
            for ii in (0..output_rows).step_by(block) {
                let i_end = (ii + block).min(output_rows);

                for kk in (0..inner).step_by(block) {
                    let k_end = (kk + block).min(inner);

                    for jj in (0..output_columns).step_by(block) {
                        let j_end = (jj + block).min(output_columns);

                        for row in ii..i_end {
                            let lhs_base = row * inner;

                            let out_base = row * output_columns;

                            for k in kk..k_end {
                                let lhs_value = u128::from(lhs_layer[lhs_base + k]);

                                let rhs_base = k * output_columns;

                                for column in jj..j_end {
                                    let rhs_value = u128::from(rhs_layer[rhs_base + column]);

                                    wide[out_base + column] += lhs_value * rhs_value;
                                }
                            }
                        }
                    }
                }
            }

            let accumulate_coordinate_elapsed = accumulate_start.elapsed();

            accumulate_elapsed += accumulate_coordinate_elapsed;

            limb_accumulate_elapsed[limb_index] += accumulate_coordinate_elapsed;

            let reduction_start = std::time::Instant::now();

            // One exact reduction per output matrix entry.
            for index in 0..output_layer_size {
                output_layer[index] = (wide[index] % q) as u64;
            }

            let reduction_coordinate_elapsed = reduction_start.elapsed();

            reduction_elapsed += reduction_coordinate_elapsed;

            limb_reduction_elapsed[limb_index] += reduction_coordinate_elapsed;

            let gemm_coordinate_elapsed = gemm_start.elapsed();

            gemm_elapsed += gemm_coordinate_elapsed;

            limb_gemm_elapsed[limb_index] += gemm_coordinate_elapsed;
        }
    }

    for limb_index in 0..limb_count {
        println!(
            "BATCH_CCMM_DELAYED_LIMB_{}_MATERIALIZE_MS={:.3}",
            limb_index,
            limb_materialize_elapsed[limb_index].as_secs_f64() * 1.0e3
        );

        println!(
            "BATCH_CCMM_DELAYED_LIMB_{}_GEMM_MS={:.3}",
            limb_index,
            limb_gemm_elapsed[limb_index].as_secs_f64() * 1.0e3
        );

        println!(
            "BATCH_CCMM_DELAYED_LIMB_{}_ACCUMULATE_MS={:.3}",
            limb_index,
            limb_accumulate_elapsed[limb_index].as_secs_f64() * 1.0e3
        );

        println!(
            "BATCH_CCMM_DELAYED_LIMB_{}_REDUCTION_MS={:.3}",
            limb_index,
            limb_reduction_elapsed[limb_index].as_secs_f64() * 1.0e3
        );
    }

    let assemble_start = std::time::Instant::now();

    let output_ntt: Vec<Vec<crate::ring::RnsNttPolynomial>> = (0..output_rows)
        .map(|row| {
            (0..output_columns)
                .map(|column| {
                    let residues = (0..limb_count)
                        .map(|limb_index| {
                            let values: Vec<u64> = (0..degree)
                                .map(|coordinate| {
                                    output_layers[limb_index][coordinate]
                                        [row * output_columns + column]
                                })
                                .collect();

                            crate::ring::NttPolynomial::from_prepared_values(
                                plan.plan(limb_index),
                                values,
                            )
                        })
                        .collect();

                    crate::ring::RnsNttPolynomial::from_residues(residues)
                })
                .collect()
        })
        .collect();

    let assemble_elapsed = assemble_start.elapsed();

    let inverse_start = std::time::Instant::now();

    let output: Vec<Vec<crate::ring::RnsPolynomial>> = output_ntt
        .iter()
        .map(|row| row.iter().map(|ntt| plan.inverse(ntt)).collect())
        .collect();

    let inverse_elapsed = inverse_start.elapsed();

    let reconstruct_elapsed = assemble_elapsed + inverse_elapsed;

    println!(
        "BATCH_CCMM_DELAYED_PHASE_FORWARD_NTT_RHS_RESIDENT_MS={:.3}",
        forward_elapsed.as_secs_f64() * 1.0e3
    );
    println!(
        "BATCH_CCMM_DELAYED_PHASE_LAYER_MATERIALIZE_MS={:.3}",
        materialize_elapsed.as_secs_f64() * 1.0e3
    );
    println!(
        "BATCH_CCMM_DELAYED_PHASE_ACCUMULATE_MS={:.3}",
        accumulate_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CCMM_DELAYED_PHASE_REDUCTION_MS={:.3}",
        reduction_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CCMM_DELAYED_PHASE_GEMM_MS={:.3}",
        gemm_elapsed.as_secs_f64() * 1.0e3
    );
    println!(
        "BATCH_CCMM_DELAYED_PHASE_ASSEMBLE_NTT_MS={:.3}",
        assemble_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CCMM_DELAYED_PHASE_INVERSE_NTT_MS={:.3}",
        inverse_elapsed.as_secs_f64() * 1.0e3
    );

    println!(
        "BATCH_CCMM_DELAYED_PHASE_RECONSTRUCT_INVERSE_MS={:.3}",
        reconstruct_elapsed.as_secs_f64() * 1.0e3
    );

    output
}

/// Inverse Batch CRT directly on NTT-resident RLWE ciphertexts.
///
/// Preserves the NTT representation across the large DFT and
/// the final signed monomial multiplication.
pub(super) fn batch_large_inv_crt_ntt_resident(
    ciphertexts: &[crate::grafting::RnsNttRlweCiphertext],
    scalar_degree: usize,
    plan: &crate::ring::RnsNttPlan,
) -> Vec<crate::grafting::RnsNttRlweCiphertext> {
    assert!(!ciphertexts.is_empty());
    assert!(ciphertexts.len().is_power_of_two());

    let dimension = ciphertexts.len();
    assert_eq!(
        scalar_degree.checked_mul(dimension),
        Some(plan.degree()),
        "Batch resident inverse CRT geometry must satisfy Ns * d = N"
    );

    for ciphertext in ciphertexts {
        assert_eq!(ciphertext.degree(), plan.degree());
        assert_eq!(ciphertext.moduli(), plan.moduli());
    }

    // Match the coefficient-domain iterative DFT:
    // bit-reversed input followed by radix-2 butterflies.
    let mut output = ciphertexts.to_vec();

    let bits = dimension.trailing_zeros();
    for index in 0..dimension {
        let reversed = index.reverse_bits() >> (usize::BITS - bits);
        if reversed > index {
            output.swap(index, reversed);
        }
    }

    let multiplier = -((2 * scalar_degree) as i64);
    let mut len = 2_usize;

    while len <= dimension {
        let half = len / 2;
        let stage_multiplier = multiplier * (dimension / len) as i64;

        for start in (0..dimension).step_by(len) {
            for k in 0..half {
                let upper_index = start + k;
                let lower_index = upper_index + half;

                let twiddled = crate::grafting::RnsNttRlweCiphertext::new(
                    output[lower_index]
                        .b()
                        .mul_monomial_signed(plan, stage_multiplier * k as i64),
                    output[lower_index]
                        .a()
                        .mul_monomial_signed(plan, stage_multiplier * k as i64),
                );

                let upper = output[upper_index].add(&twiddled);
                let lower = output[upper_index].sub(&twiddled);

                output[upper_index] = upper;
                output[lower_index] = lower;
            }
        }

        len *= 2;
    }

    output
        .into_iter()
        .enumerate()
        .map(|(column, ciphertext)| {
            let exponent = -(column as i64);

            crate::grafting::RnsNttRlweCiphertext::new(
                ciphertext.b().mul_monomial_signed(plan, exponent),
                ciphertext.a().mul_monomial_signed(plan, exponent),
            )
        })
        .collect()
}

/// Applies Batch C-MT scrambled automorphisms while retaining NTT output.
pub(super) fn batch_scrambled_auto_resident(
    ciphertexts: &[crate::grafting::RnsRlweCiphertext],
    scalar_degree: usize,
    galois_keys: &[crate::ckks::PreparedRnsGaloisKey],
    plan: &crate::ring::PreparedRnsNttPlan,
) -> Vec<crate::grafting::RnsNttRlweCiphertext> {
    assert!(!ciphertexts.is_empty());

    let dimension = ciphertexts.len();
    let large_degree = ciphertexts[0].degree();

    assert!(dimension.is_power_of_two());
    assert_eq!(scalar_degree * dimension, large_degree);
    assert_eq!(plan.degree(), large_degree);

    let two_n = 2 * large_degree;
    let mut output = Vec::with_capacity(dimension);

    for index in 0..dimension {
        let exponent = 2 * scalar_degree * index + 1;

        // Multiplicative inverse modulo 2N.
        let mut t = 0_i128;
        let mut new_t = 1_i128;
        let mut r = two_n as i128;
        let mut new_r = exponent as i128;

        while new_r != 0 {
            let quotient = r / new_r;

            let next_t = t - quotient * new_t;
            t = new_t;
            new_t = next_t;

            let next_r = r - quotient * new_r;
            r = new_r;
            new_r = next_r;
        }

        assert_eq!(r, 1);

        let inverse = t.rem_euclid(two_n as i128) as usize;

        assert_eq!(
            (inverse - 1) % (2 * scalar_degree),
            0,
            "Resident C-MT inverse exponent must remain in subgroup"
        );

        let inverse_index = (inverse - 1) / (2 * scalar_degree);

        if exponent == 1 {
            output.push(crate::grafting::RnsNttRlweCiphertext::from_coefficient(
                &ciphertexts[inverse_index],
                plan,
            ));
        } else {
            let key = galois_keys
                .iter()
                .find(|key| key.exponent() == exponent)
                .unwrap_or_else(|| {
                    panic!("Missing resident C-MT Galois key for exponent {exponent}")
                });

            output.push(
                crate::ckks::apply_rns_galois_automorphism_with_prepared_dynamic_ntt_resident(
                    &ciphertexts[inverse_index],
                    key,
                    plan,
                ),
            );
        }
    }

    output
}

/// Computes Batch ciphertext matrix transpose with NTT-resident output.
///
/// The initial CRT remains coefficient-domain. The Galois results
/// and inverse CRT remain NTT-resident.
pub(super) fn batch_ciphertext_transpose_resident(
    ciphertexts: &[crate::grafting::RnsRlweCiphertext],
    scalar_degree: usize,
    galois_keys: &[crate::ckks::PreparedRnsGaloisKey],
    prepared_plan: &crate::ring::PreparedRnsNttPlan,
    canonical_plan: &crate::ring::RnsNttPlan,
) -> Vec<crate::grafting::RnsNttRlweCiphertext> {
    assert_eq!(prepared_plan.degree(), canonical_plan.degree());
    assert_eq!(prepared_plan.moduli(), canonical_plan.moduli());

    let (crt, _, _) = super::batch::batch_large_crt_profiled(ciphertexts, scalar_degree);

    let transformed =
        batch_scrambled_auto_resident(&crt, scalar_degree, galois_keys, prepared_plan);

    batch_large_inv_crt_ntt_resident(&transformed, scalar_degree, canonical_plan)
}
