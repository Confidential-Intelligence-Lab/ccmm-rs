use super::{Modulus, NttPlan, Polynomial};
use crate::ring::modulus::PreparedModulus;

/// Prepared radix-2 execution schedule for a canonical `NttPlan`.
///
/// The canonical `NttPlan` remains the lightweight mathematical description.
/// This type precomputes transform-invariant execution data:
///
/// - forward twist powers;
/// - inverse untwist powers;
/// - forward radix-2 stage twiddles;
/// - inverse radix-2 stage twiddles.
///
/// The prepared transform is mathematically identical to
/// `NttPlan::{forward,inverse}_radix2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedNttPlan {
    modulus: Modulus,
    prepared_modulus: PreparedModulus,
    degree: usize,
    degree_inverse: u64,
    forward_twists: Vec<u64>,
    inverse_twists: Vec<u64>,
    forward_stage_twiddles: Vec<Vec<u64>>,
    inverse_stage_twiddles: Vec<Vec<u64>>,
}

impl PreparedNttPlan {
    pub fn new(plan: &NttPlan) -> Self {
        let modulus = plan.modulus();
        let degree = plan.degree();
        let psi = plan.psi();
        let psi_inverse = plan.psi_inverse();

        let forward_twists = powers(modulus, psi, degree);
        let inverse_twists = powers(modulus, psi_inverse, degree);

        let omega = modulus.mul(psi, psi);
        let omega_inverse = modulus.mul(psi_inverse, psi_inverse);

        let forward_stage_twiddles = stage_twiddles(modulus, degree, omega);
        let inverse_stage_twiddles = stage_twiddles(modulus, degree, omega_inverse);

        Self {
            modulus,
            prepared_modulus: PreparedModulus::new(modulus),
            degree,
            degree_inverse: plan.degree_inverse(),
            forward_twists,
            inverse_twists,
            forward_stage_twiddles,
            inverse_stage_twiddles,
        }
    }

    pub fn modulus(&self) -> Modulus {
        self.modulus
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn forward(&self, polynomial: &Polynomial) -> Vec<u64> {
        assert_eq!(
            polynomial.modulus(),
            self.modulus,
            "polynomial modulus must match prepared NTT modulus"
        );
        assert_eq!(
            polynomial.degree(),
            self.degree,
            "polynomial degree must match prepared NTT degree"
        );

        let mut values = polynomial.coefficients().to_vec();

        for (value, &twist) in values.iter_mut().zip(&self.forward_twists) {
            *value = self.prepared_modulus.mul_canonical(*value, twist);
        }

        cyclic_ntt_prepared(&mut values, self.modulus, &self.forward_stage_twiddles);

        values
    }

    /// Applies the prepared forward negacyclic NTT to a batch of
    /// coefficient-domain polynomials using a shared stage-major schedule.
    ///
    /// This is mathematically identical to calling `forward()` on each
    /// polynomial independently. The execution order is reorganized so
    /// that each radix-2 stage is completed across the entire batch before
    /// advancing to the next stage.
    pub fn forward_batch(&self, polynomials: &[Polynomial]) -> Vec<Vec<u64>> {
        if polynomials.is_empty() {
            return Vec::new();
        }

        for polynomial in polynomials {
            assert_eq!(
                polynomial.modulus(),
                self.modulus,
                "polynomial modulus must match prepared NTT modulus"
            );

            assert_eq!(
                polynomial.degree(),
                self.degree,
                "polynomial degree must match prepared NTT degree"
            );
        }

        let mut batch: Vec<Vec<u64>> = polynomials
            .iter()
            .map(|polynomial| polynomial.coefficients().to_vec())
            .collect();

        // Negacyclic forward twist.
        for values in &mut batch {
            for (value, &twist) in values.iter_mut().zip(&self.forward_twists) {
                *value = self.prepared_modulus.mul_canonical(*value, twist);
            }
        }

        // The cyclic radix-2 transform begins with the same permutation
        // for every polynomial.
        for values in &mut batch {
            bit_reverse_permute(values);
        }

        // Stage-major traversal:
        //
        //     stage -> polynomial -> block -> butterfly
        //
        // rather than completing every stage for one polynomial before
        // moving to the next polynomial.
        let mut len = 2_usize;

        for twiddles in &self.forward_stage_twiddles {
            let half = len / 2;

            debug_assert_eq!(
                twiddles.len(),
                half,
                "prepared NTT twiddle count must match stage width"
            );

            for values in &mut batch {
                for start in (0..self.degree).step_by(len) {
                    for (offset, &twiddle) in twiddles.iter().enumerate() {
                        let even = values[start + offset];

                        let odd = self
                            .prepared_modulus
                            .mul_canonical(values[start + offset + half], twiddle);

                        values[start + offset] = self.modulus.add_canonical(even, odd);

                        values[start + offset + half] = self.modulus.sub_canonical(even, odd);
                    }
                }
            }

            len *= 2;
        }

        batch
    }

    pub fn inverse(&self, values: &[u64]) -> Polynomial {
        assert_eq!(
            values.len(),
            self.degree,
            "NTT value count must match prepared NTT degree"
        );

        let mut coefficients = values.to_vec();

        cyclic_ntt_prepared(
            &mut coefficients,
            self.modulus,
            &self.inverse_stage_twiddles,
        );

        for (coefficient, &untwist) in coefficients.iter_mut().zip(&self.inverse_twists) {
            *coefficient = self
                .prepared_modulus
                .mul_canonical(*coefficient, self.degree_inverse);
            *coefficient = self.prepared_modulus.mul_canonical(*coefficient, untwist);
        }

        Polynomial::new(self.modulus, coefficients)
    }
}

fn powers(modulus: Modulus, root: u64, degree: usize) -> Vec<u64> {
    let mut values = Vec::with_capacity(degree);
    let mut value = 1_u64;

    for _ in 0..degree {
        values.push(value);
        value = modulus.mul(value, root);
    }

    values
}

fn stage_twiddles(modulus: Modulus, degree: usize, root: u64) -> Vec<Vec<u64>> {
    let mut stages = Vec::with_capacity(degree.trailing_zeros() as usize);
    let mut len = 2_usize;

    while len <= degree {
        let half = len / 2;
        let root_step = modulus.pow(root, (degree / len) as u64);

        let mut twiddles = Vec::with_capacity(half);
        let mut twiddle = 1_u64;

        for _ in 0..half {
            twiddles.push(twiddle);
            twiddle = modulus.mul(twiddle, root_step);
        }

        stages.push(twiddles);
        len *= 2;
    }

    stages
}

fn cyclic_ntt_prepared(values: &mut [u64], modulus: Modulus, stage_twiddles: &[Vec<u64>]) {
    assert!(
        !values.is_empty() && values.len().is_power_of_two(),
        "radix-2 NTT length must be a positive power of two"
    );

    assert_eq!(
        stage_twiddles.len(),
        values.len().trailing_zeros() as usize,
        "prepared NTT stage count must match transform degree"
    );

    bit_reverse_permute(values);

    let mut len = 2_usize;

    for twiddles in stage_twiddles {
        let half = len / 2;

        assert_eq!(
            twiddles.len(),
            half,
            "prepared NTT twiddle count must match stage width"
        );

        for start in (0..values.len()).step_by(len) {
            for (offset, &twiddle) in twiddles.iter().enumerate() {
                let even = values[start + offset];
                let odd = modulus.mul(values[start + offset + half], twiddle);

                values[start + offset] = modulus.add_canonical(even, odd);
                values[start + offset + half] = modulus.sub_canonical(even, odd);
            }
        }

        len *= 2;
    }
}

fn bit_reverse_permute(values: &mut [u64]) {
    let n = values.len();
    let mut j = 0_usize;

    for i in 1..n {
        let mut bit = n >> 1;

        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if i < j {
            values.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> NttPlan {
        NttPlan::new(Modulus::new(97), 8, 8)
    }

    #[test]
    fn prepared_forward_matches_radix2_exactly() {
        let plan = plan();
        let prepared = PreparedNttPlan::new(&plan);
        let modulus = plan.modulus();

        let vectors = [
            vec![0, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![96, 95, 94, 93, 92, 91, 90, 89],
            vec![11, 22, 33, 44, 55, 66, 77, 88],
        ];

        for vector in vectors {
            let polynomial = Polynomial::new(modulus, vector);

            assert_eq!(
                prepared.forward(&polynomial),
                plan.forward_radix2(&polynomial)
            );
        }

        println!("PREPARED_NTT_FORWARD_EQUIVALENCE=PASS");
    }

    #[test]
    #[ignore]
    fn sd3b_forward_ntt_microbench_degree_8192() {
        const DEGREE: usize = 8192;
        const REPEATS: usize = 9;

        for modulus_value in [268_238_849_u64, 68_712_923_137_u64] {
            let modulus = Modulus::new(modulus_value);

            let plan = crate::ring::make_ntt_plan(modulus, DEGREE);

            let prepared = PreparedNttPlan::new(&plan);

            let polynomial = Polynomial::new(
                modulus,
                (0..DEGREE)
                    .map(|index| {
                        let x = index as u64;
                        (11 + 13 * x + 3 * x * x) % modulus.value()
                    })
                    .collect(),
            );

            // Warm-up.
            std::hint::black_box(plan.forward_radix2(&polynomial));

            std::hint::black_box(prepared.forward(&polynomial));

            let mut canonical_samples = Vec::with_capacity(REPEATS);

            let mut prepared_samples = Vec::with_capacity(REPEATS);

            for _ in 0..REPEATS {
                let start = std::time::Instant::now();

                let result = plan.forward_radix2(&polynomial);

                canonical_samples.push(start.elapsed().as_secs_f64());

                std::hint::black_box(result);

                let start = std::time::Instant::now();

                let result = prepared.forward(&polynomial);

                prepared_samples.push(start.elapsed().as_secs_f64());

                std::hint::black_box(result);
            }

            canonical_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

            prepared_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let canonical = canonical_samples[REPEATS / 2];

            let prepared_time = prepared_samples[REPEATS / 2];

            println!("SD3B_NTT_MODULUS={modulus_value}");

            println!("SD3B_NTT_CANONICAL_MS={:.6}", canonical * 1.0e3);

            println!("SD3B_NTT_PREPARED_MS={:.6}", prepared_time * 1.0e3);

            println!(
                "SD3B_NTT_PREPARED_SPEEDUP={:.3}x",
                canonical / prepared_time
            );
        }
    }

    #[test]
    #[ignore]
    fn prepared_batch_forward_microbench_degree_8192() {
        let degree = 8192;

        // Same modulus family used in the authors-scale CCMM path.
        let modulus = Modulus::new(268_238_849);

        let plan = crate::ring::make_ntt_plan(modulus, degree);

        let prepared = PreparedNttPlan::new(&plan);

        for batch_size in [1_usize, 2, 4, 8, 16, 32, 63] {
            let polynomials: Vec<_> = (0..batch_size)
                .map(|seed| {
                    Polynomial::new(
                        modulus,
                        (0..degree)
                            .map(|index| {
                                let x = index as u64;

                                (11 + 17 * seed as u64 + 13 * x + 7 * seed as u64 * x + 3 * x * x)
                                    % modulus.value()
                            })
                            .collect(),
                    )
                })
                .collect();

            // Warm both paths.
            let _ = polynomials
                .iter()
                .map(|polynomial| prepared.forward(polynomial))
                .collect::<Vec<_>>();

            let _ = prepared.forward_batch(&polynomials);

            const REPEATS: usize = 5;

            let mut individual_samples = Vec::with_capacity(REPEATS);

            let mut batch_samples = Vec::with_capacity(REPEATS);

            for _ in 0..REPEATS {
                let start = std::time::Instant::now();

                let individual: Vec<_> = polynomials
                    .iter()
                    .map(|polynomial| prepared.forward(polynomial))
                    .collect();

                let elapsed = start.elapsed().as_secs_f64();

                std::hint::black_box(individual);

                individual_samples.push(elapsed);

                let start = std::time::Instant::now();

                let batched = prepared.forward_batch(&polynomials);

                let elapsed = start.elapsed().as_secs_f64();

                std::hint::black_box(batched);

                batch_samples.push(elapsed);
            }

            individual_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

            batch_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let individual = individual_samples[REPEATS / 2];

            let batched = batch_samples[REPEATS / 2];

            println!("PREPARED_NTT_BATCH_BENCH_BATCH_SIZE={batch_size}");

            println!(
                "PREPARED_NTT_BATCH_BENCH_INDIVIDUAL_MS={:.3}",
                individual * 1.0e3
            );

            println!(
                "PREPARED_NTT_BATCH_BENCH_STAGE_MAJOR_MS={:.3}",
                batched * 1.0e3
            );

            println!(
                "PREPARED_NTT_BATCH_BENCH_SPEEDUP={:.3}x",
                individual / batched
            );
        }
    }

    #[test]
    fn prepared_batch_forward_matches_individual_exactly() {
        let plan = plan();
        let prepared = PreparedNttPlan::new(&plan);
        let modulus = plan.modulus();

        let polynomials: Vec<_> = (0..17_u64)
            .map(|seed| {
                Polynomial::new(
                    modulus,
                    (0..plan.degree())
                        .map(|index| {
                            let x = index as u64;
                            (11 + 17 * seed + 13 * x + 7 * seed * x + 3 * x * x) % modulus.value()
                        })
                        .collect(),
                )
            })
            .collect();

        let reference: Vec<_> = polynomials
            .iter()
            .map(|polynomial| prepared.forward(polynomial))
            .collect();

        let actual = prepared.forward_batch(&polynomials);

        assert_eq!(actual, reference);

        println!("PREPARED_NTT_BATCH_FORWARD_EQUIVALENCE=PASS");
    }

    #[test]
    fn prepared_inverse_matches_radix2_exactly() {
        let plan = plan();
        let prepared = PreparedNttPlan::new(&plan);
        let modulus = plan.modulus();

        let vectors = [
            vec![0, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![96, 95, 94, 93, 92, 91, 90, 89],
            vec![11, 22, 33, 44, 55, 66, 77, 88],
        ];

        for vector in vectors {
            let polynomial = Polynomial::new(modulus, vector);
            let transformed = plan.forward_radix2(&polynomial);

            assert_eq!(
                prepared.inverse(&transformed),
                plan.inverse_radix2(&transformed)
            );
        }

        println!("PREPARED_NTT_INVERSE_EQUIVALENCE=PASS");
    }

    #[test]
    fn prepared_roundtrip_recovers_polynomial() {
        let plan = plan();
        let prepared = PreparedNttPlan::new(&plan);
        let modulus = plan.modulus();

        let polynomial = Polynomial::new(modulus, vec![9, 17, 25, 33, 41, 49, 57, 65]);

        let transformed = prepared.forward(&polynomial);
        let recovered = prepared.inverse(&transformed);

        assert_eq!(recovered, polynomial);

        println!("PREPARED_NTT_ROUNDTRIP=PASS");
    }
}
