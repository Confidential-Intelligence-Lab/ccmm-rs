//! Error distributions for RLWE encryption.
//!
//! Correctness/reference paths may use bounded-uniform error. Security-bearing
//! profiles use an integer discrete Gaussian and sample one logical error
//! polynomial before projection into an RNS basis.

use rand::{CryptoRng, Rng, RngCore};

use crate::ring::{Modulus, Polynomial};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorDistribution {
    BoundedUniform { bound: i64 },
    DiscreteGaussian { sigma: f64 },
}

impl ErrorDistribution {
    pub fn validate(self) {
        match self {
            Self::BoundedUniform { bound } => {
                assert!(bound >= 0, "error bound must be nonnegative");
            }
            Self::DiscreteGaussian { sigma } => {
                assert!(
                    sigma.is_finite() && sigma > 0.0,
                    "discrete-Gaussian sigma must be positive and finite"
                );
            }
        }
    }
}

pub fn sample_error_coefficients<R>(
    degree: usize,
    distribution: ErrorDistribution,
    rng: &mut R,
) -> Vec<i64>
where
    R: RngCore + CryptoRng,
{
    distribution.validate();

    match distribution {
        ErrorDistribution::BoundedUniform { bound } => {
            if bound == 0 {
                vec![0; degree]
            } else {
                (0..degree).map(|_| rng.gen_range(-bound..=bound)).collect()
            }
        }
        ErrorDistribution::DiscreteGaussian { sigma } => (0..degree)
            .map(|_| sample_discrete_gaussian(sigma, rng))
            .collect(),
    }
}

/// Samples the integer distribution proportional to
///
/// ```text
/// exp(-x^2 / (2 sigma^2)), x in Z.
/// ```
///
/// The tail is truncated only after its omitted probability mass is
/// negligible relative to 64-bit sampling precision.
fn sample_discrete_gaussian<R>(sigma: f64, rng: &mut R) -> i64
where
    R: RngCore + CryptoRng,
{
    /*
     * At 16 sigma the omitted two-sided Gaussian tail is far below the
     * resolution of a 64-bit uniform draw. The table is deliberately
     * constructed directly from the discrete probability mass rather than
     * rounding samples from a continuous normal distribution.
     */
    let tail = (16.0 * sigma).ceil() as i64;

    let mut weights = Vec::with_capacity((2 * tail + 1) as usize);
    let mut total = 0.0_f64;

    for x in -tail..=tail {
        let xf = x as f64;
        let weight = (-(xf * xf) / (2.0 * sigma * sigma)).exp();
        weights.push((x, weight));
        total += weight;
    }

    let target = uniform_open_unit(rng) * total;
    let mut cumulative = 0.0_f64;

    for (x, weight) in weights {
        cumulative += weight;
        if target < cumulative {
            return x;
        }
    }

    tail
}

fn uniform_open_unit<R>(rng: &mut R) -> f64
where
    R: RngCore + CryptoRng,
{
    /*
     * Use the upper 53 random bits so every representable draw used here
     * has the same probability.
     */
    let value = rng.next_u64() >> 11;
    (value as f64 + 0.5) * (1.0 / ((1_u64 << 53) as f64))
}

pub fn project_error(modulus: Modulus, coefficients: &[i64]) -> Polynomial {
    Polynomial::new(
        modulus,
        coefficients
            .iter()
            .copied()
            .map(|value| signed_to_mod(value, modulus.value()))
            .collect(),
    )
}

fn signed_to_mod(value: i64, modulus: u64) -> u64 {
    if value >= 0 {
        (value as u64) % modulus
    } else {
        let magnitude = value.unsigned_abs() % modulus;
        if magnitude == 0 {
            0
        } else {
            modulus - magnitude
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use super::*;

    #[test]
    fn zero_bounded_error_is_exactly_zero() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let values =
            sample_error_coefficients(32, ErrorDistribution::BoundedUniform { bound: 0 }, &mut rng);
        assert_eq!(values, vec![0; 32]);
    }

    #[test]
    fn bounded_error_stays_inside_requested_support() {
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        let values = sample_error_coefficients(
            4096,
            ErrorDistribution::BoundedUniform { bound: 3 },
            &mut rng,
        );
        assert!(values.iter().all(|&value| (-3..=3).contains(&value)));
    }

    #[test]
    fn gaussian_sampler_has_expected_empirical_moments() {
        let sigma = 3.19;
        let mut rng = ChaCha20Rng::seed_from_u64(3);
        let values = sample_error_coefficients(
            100_000,
            ErrorDistribution::DiscreteGaussian { sigma },
            &mut rng,
        );

        let n = values.len() as f64;
        let mean = values.iter().map(|&x| x as f64).sum::<f64>() / n;
        let variance = values
            .iter()
            .map(|&x| {
                let delta = x as f64 - mean;
                delta * delta
            })
            .sum::<f64>()
            / n;

        assert!(mean.abs() < 0.05, "empirical mean = {mean}");
        assert!(
            (variance.sqrt() - sigma).abs() < 0.05,
            "empirical sigma = {}",
            variance.sqrt()
        );
    }

    #[test]
    fn projection_preserves_signed_coefficients_mod_q() {
        let modulus = Modulus::new(12_289);
        let values = [-7, -1, 0, 1, 7];
        let projected = project_error(modulus, &values);

        assert_eq!(projected.coefficients(), &[12_282, 12_288, 0, 1, 7]);
    }
}
