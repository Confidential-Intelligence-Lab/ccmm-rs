//! P6c.1 Park Mod-PP-MM backend characterization.
//!
//! Measures only plaintext modular matrix multiplication. Homomorphic setup,
//! C-MT, key switching, relinearization, and CKKS work are excluded.

use std::hint::black_box;
use std::time::Instant;

use ccmm_rs::ccmm::park::{park_mod_pp_mm, ParkModPpMmBackend};
use ccmm_rs::ckks::research_profile_4096;

fn matrix(rows: usize, cols: usize, modulus: u64, salt: u64) -> Vec<Vec<u64>> {
    (0..rows)
        .map(|row| {
            (0..cols)
                .map(|col| {
                    salt.wrapping_add(17 * row as u64)
                        .wrapping_add(29 * col as u64)
                        .wrapping_add(5 * row as u64 * col as u64)
                        % modulus
                })
                .collect()
        })
        .collect()
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn benchmark(
    lhs: &[Vec<u64>],
    rhs: &[Vec<u64>],
    modulus: u64,
    backend: ParkModPpMmBackend,
    warmups: usize,
    repeats: usize,
) -> u128 {
    for _ in 0..warmups {
        black_box(park_mod_pp_mm(
            black_box(lhs),
            black_box(rhs),
            black_box(modulus),
            backend,
        ));
    }

    let mut samples = Vec::with_capacity(repeats);

    for _ in 0..repeats {
        let start = Instant::now();

        let output = park_mod_pp_mm(black_box(lhs), black_box(rhs), black_box(modulus), backend);

        black_box(output);

        samples.push(start.elapsed().as_nanos());
    }

    median(&mut samples)
}

fn main() {
    let profile = research_profile_4096();
    let modulus = profile.modulus_basis().moduli()[0].value();

    let max_n = std::env::var("PARK_MODPP_MAX_N")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(512);

    let sizes = [64_usize, 128, 256, 512, 1024];

    println!("PARK_MOD_PP_MM_CHARACTERIZATION_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("MODULUS={modulus}");
    println!("MAX_N={max_n}");
    println!("SETUP_INCLUDED_IN_TIMING=NO");
    println!("TIMING_STATISTIC=median");
    println!("CSV_HEADER=n,backend,median_us,speedup_vs_reference");

    for n in sizes.into_iter().filter(|&n| n <= max_n) {
        let lhs = matrix(n, n, modulus, 0x5041_524b ^ n as u64);

        let rhs = matrix(n, n, modulus, 0x4d4f_4450 ^ n as u64);

        let warmups = 1;
        let repeats = if n <= 256 { 5 } else { 3 };

        let reference_ns = benchmark(
            &lhs,
            &rhs,
            modulus,
            ParkModPpMmBackend::Reference,
            warmups,
            repeats,
        );

        let transposed_ns = benchmark(
            &lhs,
            &rhs,
            modulus,
            ParkModPpMmBackend::TransposedRhs,
            warmups,
            repeats,
        );

        let wide_ns = benchmark(
            &lhs,
            &rhs,
            modulus,
            ParkModPpMmBackend::WideAccumulator,
            warmups,
            repeats,
        );

        let reference_us = reference_ns as f64 / 1.0e3;

        let transposed_us = transposed_ns as f64 / 1.0e3;

        let wide_us = wide_ns as f64 / 1.0e3;

        println!("RESULT,{n},Reference,{reference_us:.3},1.000000");

        println!(
            "RESULT,{n},TransposedRhs,{transposed_us:.3},{:.6}",
            reference_ns as f64 / transposed_ns as f64,
        );

        println!(
            "RESULT,{n},WideAccumulator,{wide_us:.3},{:.6}",
            reference_ns as f64 / wide_ns as f64,
        );

        // Characterization must never benchmark an incorrect backend.
        let reference = park_mod_pp_mm(&lhs, &rhs, modulus, ParkModPpMmBackend::Reference);

        let wide = park_mod_pp_mm(&lhs, &rhs, modulus, ParkModPpMmBackend::WideAccumulator);

        assert_eq!(wide, reference, "WideAccumulator mismatch at N={n}");
    }

    println!("PARK_MOD_PP_MM_CHARACTERIZATION_STATUS=PASS");
}
