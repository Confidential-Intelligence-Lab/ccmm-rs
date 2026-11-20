//! Unbiased uniform sampling for modular polynomial coefficients.

use rand::{CryptoRng, RngCore};

#[cfg(test)]
const BUFFER_SIZE: usize = 64 * 1024;

/// Samples `degree` independent uniform residues in `[0, modulus)`.
///
/// Randomness is obtained from the caller-provided cryptographic RNG.
/// Rejection sampling avoids modulo bias.
pub(crate) fn sample_uniform_residues<R>(degree: usize, modulus: u64, rng: &mut R) -> Vec<u64>
where
    R: RngCore + CryptoRng,
{
    assert!(modulus >= 2, "modulus must be at least 2");

    let threshold = modulus.wrapping_neg() % modulus;
    let mut coefficients = Vec::with_capacity(degree);

    while coefficients.len() < degree {
        let candidate = rng.next_u64();

        if candidate >= threshold {
            coefficients.push(candidate % modulus);
        }
    }

    coefficients
}

#[cfg(test)]
fn buffered_reference_sample<R>(degree: usize, modulus: u64, rng: &mut R) -> Vec<u64>
where
    R: RngCore + CryptoRng,
{
    assert!(modulus >= 2, "modulus must be at least 2");

    let threshold = modulus.wrapping_neg() % modulus;

    let mut buffer = [0u8; BUFFER_SIZE];
    let mut position = BUFFER_SIZE;
    let mut coefficients = Vec::with_capacity(degree);

    while coefficients.len() < degree {
        if position + 8 > BUFFER_SIZE {
            rng.fill_bytes(&mut buffer);
            position = 0;
        }

        let candidate = u64::from_le_bytes(
            buffer[position..position + 8]
                .try_into()
                .expect("eight-byte random candidate"),
        );
        position += 8;

        if candidate >= threshold {
            coefficients.push(candidate % modulus);
        }
    }

    coefficients
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn residues_are_within_modulus() {
        for modulus in [2, 3, 17, 257, 65537, u64::MAX] {
            let mut rng = ChaCha20Rng::seed_from_u64(modulus);
            let values = sample_uniform_residues(4096, modulus, &mut rng);

            assert_eq!(values.len(), 4096);
            assert!(values.iter().all(|&value| value < modulus));
        }
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let mut lhs = ChaCha20Rng::seed_from_u64(0xE10);
        let mut rhs = ChaCha20Rng::seed_from_u64(0xE10);

        let a = sample_uniform_residues(16384, 65537, &mut lhs);
        let b = sample_uniform_residues(16384, 65537, &mut rhs);

        assert_eq!(a, b);
    }

    #[test]
    fn empty_sample_is_empty() {
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        assert!(sample_uniform_residues(0, 17, &mut rng).is_empty());
    }
}

#[cfg(test)]
mod n4_e11r5_tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn direct_rng_matches_buffered_reference() {
        for modulus in [65537_u64, 1125899906826241, 1152921504606846883] {
            for seed in [0xE110_A001_u64, 0xE110_A002, 0xE110_A003] {
                let mut reference_rng = ChaCha20Rng::seed_from_u64(seed);
                let mut optimized_rng = ChaCha20Rng::seed_from_u64(seed);

                let reference = buffered_reference_sample(16384, modulus, &mut reference_rng);

                let optimized = sample_uniform_residues(16384, modulus, &mut optimized_rng);

                assert_eq!(reference, optimized);
                assert_eq!(reference_rng.next_u64(), optimized_rng.next_u64());

                println!("N4_E11R5_MODULUS={} SEED={} STATUS=PASS", modulus, seed);
            }
        }
    }
}
