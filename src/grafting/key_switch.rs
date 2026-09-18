use crate::eval::MultiplicationKey;
use crate::ring::Polynomial;
use crate::rlwe::{RlweCiphertext, RlweParameters, RlweQuadraticCiphertext};

/// Scheduling description for grouping conventional base-B gadget
/// digits into logical blocks.
///
/// This does not change the gadget decomposition itself. It provides
/// the scheduling seam required for later Grafting/RNS gadget
/// resurrection while preserving exact parity with the existing
/// relinearization implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GadgetDigitSchedule {
    block_sizes: Vec<usize>,
}

impl GadgetDigitSchedule {
    pub fn new(block_sizes: Vec<usize>, digit_count: usize) -> Self {
        assert!(
            !block_sizes.is_empty(),
            "gadget digit schedule must contain at least one block"
        );

        assert!(
            block_sizes.iter().all(|&size| size > 0),
            "gadget digit blocks must be nonempty"
        );

        assert_eq!(
            block_sizes.iter().sum::<usize>(),
            digit_count,
            "gadget digit blocks must cover every evaluation-key digit exactly"
        );

        Self { block_sizes }
    }

    pub fn block_sizes(&self) -> &[usize] {
        &self.block_sizes
    }

    pub fn block_count(&self) -> usize {
        self.block_sizes.len()
    }

    pub fn digit_count(&self) -> usize {
        self.block_sizes.iter().sum()
    }

    pub fn digit_ranges(&self) -> Vec<std::ops::Range<usize>> {
        let mut start = 0;

        self.block_sizes
            .iter()
            .map(|&size| {
                let end = start + size;
                let range = start..end;
                start = end;
                range
            })
            .collect()
    }
}

/// Relinearizes through a block-aware gadget-digit schedule.
///
/// Cryptographic semantics are intentionally identical to the existing
/// conventional base-B relinearization path. Only evaluation scheduling
/// is changed.
///
/// This gives Grafting a correctness-preserving integration seam before
/// introducing an RNS/resurrected gadget decomposition.
pub fn relinearize_scheduled(
    params: RlweParameters,
    product: &RlweQuadraticCiphertext,
    multiplication_key: &MultiplicationKey,
    schedule: &GadgetDigitSchedule,
) -> RlweCiphertext {
    assert_eq!(
        product.c0().modulus(),
        params.modulus(),
        "quadratic ciphertext modulus must match RLWE parameters"
    );

    assert_eq!(
        product.c0().degree(),
        params.degree(),
        "quadratic ciphertext degree must match RLWE parameters"
    );

    assert_eq!(
        schedule.digit_count(),
        multiplication_key.digit_count(),
        "schedule must cover multiplication-key digits"
    );

    let digits = gadget_decompose_reference(
        product.c2(),
        multiplication_key.base(),
        multiplication_key.digit_count(),
    );

    let mut b = product.c0().clone();

    let mut a = product.c1().clone();

    for range in schedule.digit_ranges() {
        for index in range {
            let digit = &digits[index];

            let evaluation_key = &multiplication_key.digits()[index];

            b = b.add(&digit.negacyclic_mul(evaluation_key.b()));

            a = a.add(&digit.negacyclic_mul(evaluation_key.a()));
        }
    }

    RlweCiphertext::new(b, a)
}

/// Local copy of the conventional base-B decomposition.
///
/// Keeping this private and local makes the differential implementation
/// independent of the private helper in `eval::relinearization`.
fn gadget_decompose_reference(
    polynomial: &Polynomial,
    base: u64,
    digit_count: usize,
) -> Vec<Polynomial> {
    assert!(base >= 2, "gadget base must be at least 2");

    assert!(digit_count > 0, "gadget digit count must be positive");

    let degree = polynomial.degree();

    let modulus = polynomial.modulus();

    let mut digit_coefficients = vec![vec![0_u64; degree]; digit_count];

    for (coefficient_index, &coefficient) in polynomial.coefficients().iter().enumerate() {
        let mut value = coefficient;

        for digit in digit_coefficients.iter_mut() {
            digit[coefficient_index] = value % base;

            value /= base;
        }

        assert_eq!(value, 0, "gadget decomposition has insufficient digits");
    }

    digit_coefficients
        .into_iter()
        .map(|coefficients| Polynomial::new(modulus, coefficients))
        .collect()
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::eval::{relinearize, MultiplicationKey};
    use crate::ring::Modulus;
    use crate::rlwe::{
        decrypt_raw, encrypt_with_rng, tensor, RlweParameters, RlwePlaintext, SecretKey,
    };

    use super::*;

    fn params() -> RlweParameters {
        RlweParameters::new(8, Modulus::new(12_289), 16, 1)
    }

    #[test]
    fn schedule_ranges_cover_digits_exactly() {
        let schedule = GadgetDigitSchedule::new(vec![1, 2, 1], 4);

        assert_eq!(schedule.digit_ranges(), vec![0..1, 1..3, 3..4]);
    }

    #[test]
    fn one_block_matches_conventional_relinearization_exactly() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(1);

        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(2);

        let mut rhs_rng = ChaCha20Rng::seed_from_u64(3);

        let lhs = encrypt_with_rng(params, &secret, &plaintext, &mut lhs_rng);

        let rhs = encrypt_with_rng(params, &secret, &plaintext, &mut rhs_rng);

        let product = tensor(&lhs, &rhs);

        let mut eval_rng = ChaCha20Rng::seed_from_u64(4);

        let key = MultiplicationKey::generate_with_rng(params, &secret, 16, &mut eval_rng);

        let schedule = GadgetDigitSchedule::new(vec![key.digit_count()], key.digit_count());

        assert_eq!(
            relinearize_scheduled(params, &product, &key, &schedule,),
            relinearize(params, &product, &key,)
        );
    }

    #[test]
    fn different_blockings_produce_identical_ciphertext() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(10);

        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let plaintext = RlwePlaintext::new(params, vec![8, 7, 6, 5, 4, 3, 2, 1]);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(11);

        let mut rhs_rng = ChaCha20Rng::seed_from_u64(12);

        let product = tensor(
            &encrypt_with_rng(params, &secret, &plaintext, &mut lhs_rng),
            &encrypt_with_rng(params, &secret, &plaintext, &mut rhs_rng),
        );

        let mut eval_rng = ChaCha20Rng::seed_from_u64(13);

        let key = MultiplicationKey::generate_with_rng(params, &secret, 16, &mut eval_rng);

        assert_eq!(key.digit_count(), 4);

        let schedules = [
            vec![4],
            vec![1, 3],
            vec![2, 2],
            vec![1, 1, 2],
            vec![1, 1, 1, 1],
        ];

        let conventional = relinearize(params, &product, &key);

        for block_sizes in schedules {
            let schedule = GadgetDigitSchedule::new(block_sizes, key.digit_count());

            assert_eq!(
                relinearize_scheduled(params, &product, &key, &schedule,),
                conventional
            );
        }
    }

    #[test]
    fn scheduled_relinearization_matches_conventional_campaign() {
        let params = params();

        for seed in 0_u64..32 {
            let mut key_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x1111);

            let secret = SecretKey::generate_with_rng(params, &mut key_rng);

            let lhs_plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);

            let rhs_plaintext = RlwePlaintext::new(params, vec![8, 7, 6, 5, 4, 3, 2, 1]);

            let mut lhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x2222);

            let mut rhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x3333);

            let lhs = encrypt_with_rng(params, &secret, &lhs_plaintext, &mut lhs_rng);

            let rhs = encrypt_with_rng(params, &secret, &rhs_plaintext, &mut rhs_rng);

            let product = tensor(&lhs, &rhs);

            let mut eval_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x4444);

            let key = MultiplicationKey::generate_with_rng(params, &secret, 16, &mut eval_rng);

            let schedule = GadgetDigitSchedule::new(vec![1, 1, 2], key.digit_count());

            let conventional = relinearize(params, &product, &key);

            let scheduled = relinearize_scheduled(params, &product, &key, &schedule);

            // Strongest possible gate: same evaluation key,
            // same decomposition and same arithmetic order imply
            // identical ciphertext components.
            assert_eq!(
                scheduled, conventional,
                "ciphertext parity failed for seed {seed}"
            );

            assert_eq!(
                decrypt_raw(params, &secret, &scheduled,),
                decrypt_raw(params, &secret, &conventional,),
                "decryption parity failed for seed {seed}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "cover every")]
    fn rejects_incomplete_schedule() {
        let _ = GadgetDigitSchedule::new(vec![1, 2], 4);
    }

    #[test]
    #[should_panic(expected = "cover every")]
    fn rejects_oversized_schedule() {
        let _ = GadgetDigitSchedule::new(vec![2, 3], 4);
    }

    #[test]
    #[should_panic(expected = "nonempty")]
    fn rejects_empty_block() {
        let _ = GadgetDigitSchedule::new(vec![2, 0, 2], 4);
    }
}
