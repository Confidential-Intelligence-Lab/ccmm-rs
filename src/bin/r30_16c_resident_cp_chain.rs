use std::fs::{self, OpenOptions};
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use ccmm_rs::ckks::{
    multiply_ntt_resident_rlwe_with_prepared_plaintext,
    multiply_ntt_resident_rlwe_with_prepared_plaintext_resident, research_profile_16384,
    PreparedRnsPlaintextNtt,
};
use ccmm_rs::grafting::{RnsNttRlweCiphertext, RnsRlweCiphertext};
use ccmm_rs::ring::{Modulus, Polynomial, PreparedRnsNttPlan, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::RlweCiphertext;

fn setting(name: &str, default: usize, max: usize) -> usize {
    let value = std::env::var(name)
        .ok()
        .map(|text| text.parse::<usize>().expect("invalid benchmark setting"))
        .unwrap_or(default);
    assert!(value > 0 && value <= max, "{name} outside allowed range");
    value
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn elapsed_ms<F, T>(operation: F) -> f64
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let output = operation();
    let elapsed = start.elapsed().as_secs_f64() * 1.0e3;
    black_box(output);
    elapsed
}

fn polynomial(modulus: Modulus, degree: usize, seed: u64) -> Polynomial {
    let q = modulus.value();
    let mut state = seed;
    let coefficients = (0..degree)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 16) % q
        })
        .collect();
    Polynomial::new(modulus, coefficients)
}

fn reference(
    input: &RnsRlweCiphertext,
    first: &PreparedRnsPlaintextNtt,
    second: &PreparedRnsPlaintextNtt,
    plan: &PreparedRnsNttPlan,
) -> RnsRlweCiphertext {
    let initial = RnsNttRlweCiphertext::from_coefficient(input, plan);
    let intermediate = multiply_ntt_resident_rlwe_with_prepared_plaintext(&initial, first, plan);
    let intermediate_ntt = RnsNttRlweCiphertext::from_coefficient(&intermediate, plan);
    multiply_ntt_resident_rlwe_with_prepared_plaintext(&intermediate_ntt, second, plan)
}

fn resident(
    input: &RnsRlweCiphertext,
    first: &PreparedRnsPlaintextNtt,
    second: &PreparedRnsPlaintextNtt,
    plan: &PreparedRnsNttPlan,
) -> RnsRlweCiphertext {
    let initial = RnsNttRlweCiphertext::from_coefficient(input, plan);
    let intermediate =
        multiply_ntt_resident_rlwe_with_prepared_plaintext_resident(&initial, first, plan);
    let final_ntt =
        multiply_ntt_resident_rlwe_with_prepared_plaintext_resident(&intermediate, second, plan);
    final_ntt.to_coefficient(plan)
}

fn main() {
    let warmups = setting("R30_16C_WARMUPS", 3, 100);
    let trials = setting("R30_16C_TRIALS", 12, 1000);

    let profile = research_profile_16384();
    assert!(!profile.security_bearing());

    let degree = profile.degree();
    let basis = profile.modulus_chain().top().clone();
    let moduli = basis.moduli();

    let ordinary_plan = RnsNttPlan::new(moduli.to_vec(), degree);
    let plan = PreparedRnsNttPlan::new(&ordinary_plan);

    let input = RnsRlweCiphertext::from_limbs(
        moduli
            .iter()
            .copied()
            .enumerate()
            .map(|(index, modulus)| {
                RlweCiphertext::new(
                    polynomial(modulus, degree, 100 + index as u64),
                    polynomial(modulus, degree, 200 + index as u64),
                )
            })
            .collect(),
    );

    let first_plain = RnsPolynomial::from_residues(
        moduli
            .iter()
            .copied()
            .enumerate()
            .map(|(index, modulus)| polynomial(modulus, degree, 300 + index as u64))
            .collect(),
    );
    let second_plain = RnsPolynomial::from_residues(
        moduli
            .iter()
            .copied()
            .enumerate()
            .map(|(index, modulus)| polynomial(modulus, degree, 400 + index as u64))
            .collect(),
    );

    let first = PreparedRnsPlaintextNtt::new(&first_plain, &plan);
    let second = PreparedRnsPlaintextNtt::new(&second_plain, &plan);

    assert_eq!(
        reference(&input, &first, &second, &plan),
        resident(&input, &first, &second, &plan),
        "reference and resident ciphertexts differ"
    );
    println!("R30_16C_EXACT_CIPHERTEXT_EQUALITY=PASS");

    let output_dir = PathBuf::from(
        std::env::var("R30_16C_OUTPUT_DIR").unwrap_or_else(|_| "results/benchmarks/r30_16c".into()),
    );
    fs::create_dir_all(&output_dir).expect("cannot create output directory");

    let csv_path = output_dir.join(format!("resident_cp_chain_n{degree}_l{}.csv", moduli.len()));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&csv_path)
        .expect("benchmark CSV already exists or cannot be created");
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "degree,limbs,trial,order,reference_ms,resident_ms,speedup"
    )
    .unwrap();

    for trial in 0..warmups {
        if trial % 2 == 0 {
            black_box(reference(&input, &first, &second, &plan));
            black_box(resident(&input, &first, &second, &plan));
        } else {
            black_box(resident(&input, &first, &second, &plan));
            black_box(reference(&input, &first, &second, &plan));
        }
    }

    let mut reference_samples = Vec::with_capacity(trials);
    let mut resident_samples = Vec::with_capacity(trials);

    for trial in 0..trials {
        let reference_first = trial % 2 == 0;

        let (reference_ms, resident_ms) = if reference_first {
            (
                elapsed_ms(|| reference(black_box(&input), &first, &second, &plan)),
                elapsed_ms(|| resident(black_box(&input), &first, &second, &plan)),
            )
        } else {
            let resident_ms = elapsed_ms(|| resident(black_box(&input), &first, &second, &plan));
            let reference_ms = elapsed_ms(|| reference(black_box(&input), &first, &second, &plan));
            (reference_ms, resident_ms)
        };

        reference_samples.push(reference_ms);
        resident_samples.push(resident_ms);

        writeln!(
            writer,
            "{degree},{},{trial},{},{reference_ms:.6},{resident_ms:.6},{:.6}",
            moduli.len(),
            if reference_first {
                "reference_first"
            } else {
                "resident_first"
            },
            reference_ms / resident_ms,
        )
        .unwrap();
    }

    writer.flush().expect("cannot flush benchmark CSV");

    let reference_median = median(&reference_samples);
    let resident_median = median(&resident_samples);

    println!("R30_16C_DEGREE={degree}");
    println!("R30_16C_LIMBS={}", moduli.len());
    println!("R30_16C_WARMUPS={warmups}");
    println!("R30_16C_TRIALS={trials}");
    println!("R30_16C_REFERENCE_MEDIAN_MS={reference_median:.6}");
    println!("R30_16C_RESIDENT_MEDIAN_MS={resident_median:.6}");
    println!("R30_16C_SPEEDUP={:.6}", reference_median / resident_median);
    println!("R30_16C_SECURITY_CLASSIFICATION=RESEARCH_ONLY");
    println!("R30_16C_CSV={}", csv_path.display());
    println!("R30_16C_STATUS=PASS");
}
