use std::fs::{self, File};
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use ccmm_rs::ckks::{research_profile_16384, research_profile_32768, research_profile_65536};
use ccmm_rs::ring::{Polynomial, PreparedNttPlan, RnsNttPlan};

fn env_usize(name: &str, default: usize) -> usize {
    let value = std::env::var(name)
        .ok()
        .map(|v| v.parse::<usize>().expect("Invalid integer"))
        .unwrap_or(default);

    assert!(value > 0);
    value
}

fn elapsed_ms<F, T>(operation: F) -> f64
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    black_box(operation());
    start.elapsed().as_secs_f64() * 1000.0
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();

    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn main() {
    let degree = env_usize("R30_14B_DEGREE", 16384);

    let profile = match degree {
        16384 => research_profile_16384(),
        32768 => research_profile_32768(),
        65536 => research_profile_65536(),
        _ => panic!("Unsupported degree"),
    };

    assert!(!profile.security_bearing());

    let chain = profile.modulus_chain();
    let basis = chain.top();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

    let warmups = env_usize("R30_14B_WARMUPS", 5);
    let trials = env_usize("R30_14B_TRIALS", 30);

    let output_dir = PathBuf::from(
        std::env::var("R30_14B_OUTPUT_DIR").unwrap_or_else(|_| "results/benchmarks/r30_14b".into()),
    );

    fs::create_dir_all(&output_dir).unwrap();

    let output_path = output_dir.join(format!("forward_n{degree}.csv"));

    assert!(!output_path.exists(), "Choose a fresh output directory");

    let file = File::create(&output_path).unwrap();
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "degree,limb_index,trial,order,forward_ms,owned_ms,owned_transform_only_ms"
    )
    .unwrap();

    let mut forward_samples = Vec::new();
    let mut owned_samples = Vec::new();
    let mut transform_samples = Vec::new();

    for limb in 0..basis.len() {
        let modulus = basis.moduli()[limb];
        let prepared = PreparedNttPlan::new(plan.plan(limb));

        let coefficients: Vec<u64> = (0..degree)
            .map(|i| (i as u64).wrapping_mul(37).wrapping_add(19) % modulus.value())
            .collect();

        let polynomial = Polynomial::new(modulus, coefficients.clone());

        let reference = prepared.forward(&polynomial);
        let owned = prepared.forward_owned(coefficients.clone());

        assert_eq!(
            reference, owned,
            "NTT forward equality failed at limb {limb}"
        );

        for trial in 0..(warmups + trials) {
            let ordinary_first = trial % 2 == 0;

            let measure_forward = || elapsed_ms(|| prepared.forward(black_box(&polynomial)));

            let measure_owned =
                || elapsed_ms(|| prepared.forward_owned(black_box(coefficients.clone())));

            let (forward_ms, owned_ms) = if ordinary_first {
                (measure_forward(), measure_owned())
            } else {
                let owned = measure_owned();
                let forward = measure_forward();
                (forward, owned)
            };

            // This measurement excludes the input clone.
            let owned_input = coefficients.clone();
            let transform_only_ms = elapsed_ms(|| prepared.forward_owned(black_box(owned_input)));

            if trial >= warmups {
                let index = trial - warmups;

                forward_samples.push(forward_ms);
                owned_samples.push(owned_ms);
                transform_samples.push(transform_only_ms);

                writeln!(
                    writer,
                    "{degree},{limb},{index},{},{forward_ms:.9},{owned_ms:.9},{transform_only_ms:.9}",
                    if ordinary_first {
                        "forward_first"
                    } else {
                        "owned_first"
                    }
                ).unwrap();
            }
        }
    }

    writer.flush().unwrap();

    let forward_median = median(&forward_samples);
    let owned_median = median(&owned_samples);
    let transform_median = median(&transform_samples);

    println!("R30_14B_DEGREE={degree}");
    println!("R30_14B_LIMBS={}", basis.len());
    println!("R30_14B_FORWARD_MEDIAN_MS={forward_median:.6}");
    println!("R30_14B_OWNED_MEDIAN_MS={owned_median:.6}");
    println!("R30_14B_OWNED_TRANSFORM_ONLY_MEDIAN_MS={transform_median:.6}");
    println!("R30_14B_OWNED_SPEEDUP={:.6}", forward_median / owned_median);
    println!("R30_14B_EXACT_EQUALITY=PASS");
    println!("R30_14B_SECURITY_CLASSIFICATION=RESEARCH_ONLY");
    println!("R30_14B_SECURITY_QUALIFIED=false");
    println!("R30_14B_CSV={}", output_path.display());
    println!("R30_14B_STATUS=PASS");
}
