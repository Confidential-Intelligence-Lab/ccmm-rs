use std::collections::BTreeSet;

use crate::ring::{Modulus, ModulusBasis, ModulusChain, RnsNttPlan};

use super::CkksChainState;

/// Intended use of a CKKS parameter profile.
///
/// This classification is descriptive. In particular, `Research` does not
/// imply a cryptographic security level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CkksParameterClass {
    CorrectnessOriented,
    Research,
}

/// Coherent CKKS ring, modulus-chain, and scale configuration.
///
/// Security-bearing status is explicit rather than inferred from the ring
/// degree or modulus size. Profiles introduced before independent security
/// estimation must set `security_bearing` to false.
#[derive(Debug, Clone, PartialEq)]
pub struct CkksParameterProfile {
    name: &'static str,
    degree: usize,
    modulus_values: &'static [u64],
    initial_scale: f64,
    class: CkksParameterClass,
    security_bearing: bool,
}

impl CkksParameterProfile {
    pub fn new(
        name: &'static str,
        degree: usize,
        modulus_values: &'static [u64],
        initial_scale: f64,
        class: CkksParameterClass,
        security_bearing: bool,
    ) -> Self {
        let profile = Self {
            name,
            degree,
            modulus_values,
            initial_scale,
            class,
            security_bearing,
        };
        profile.validate();
        profile
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn slot_count(&self) -> usize {
        self.degree / 2
    }

    pub fn modulus_values(&self) -> &'static [u64] {
        self.modulus_values
    }

    pub fn initial_scale(&self) -> f64 {
        self.initial_scale
    }

    pub fn parameter_class(&self) -> CkksParameterClass {
        self.class
    }

    pub fn security_bearing(&self) -> bool {
        self.security_bearing
    }

    pub fn modulus_basis(&self) -> ModulusBasis {
        ModulusBasis::new(
            self.modulus_values
                .iter()
                .copied()
                .map(Modulus::new)
                .collect(),
        )
    }

    pub fn modulus_chain(&self) -> ModulusChain {
        ModulusChain::from_top_basis(self.modulus_basis())
    }

    pub fn top_state(&self) -> CkksChainState {
        let chain = self.modulus_chain();
        CkksChainState::top(&chain, self.initial_scale)
    }

    pub fn rns_ntt_plan(&self) -> RnsNttPlan {
        RnsNttPlan::new(self.modulus_basis().moduli().to_vec(), self.degree)
    }

    /// Sum of the binary lengths of the individual RNS moduli.
    ///
    /// This is the conventional RNS modulus-chain bit budget, not the exact
    /// bit length of the composite modulus product.
    pub fn total_modulus_bits(&self) -> u32 {
        self.modulus_values
            .iter()
            .map(|&q| u64::BITS - q.leading_zeros())
            .sum()
    }

    pub fn validate(&self) {
        assert!(!self.name.is_empty(), "CKKS profile name must be nonempty");
        assert!(
            self.degree >= 2 && self.degree.is_power_of_two(),
            "CKKS profile degree must be a power of two of at least 2"
        );
        assert!(
            !self.modulus_values.is_empty(),
            "CKKS profile requires at least one modulus"
        );
        assert!(
            self.initial_scale.is_finite() && self.initial_scale > 0.0,
            "CKKS profile scale must be positive and finite"
        );

        let two_n = self
            .degree
            .checked_mul(2)
            .expect("CKKS profile degree is too large") as u64;

        let mut seen = BTreeSet::new();
        for &q in self.modulus_values {
            assert!(q >= 3, "CKKS profile modulus must be at least 3");
            assert!(
                seen.insert(q),
                "CKKS profile moduli must be pairwise distinct"
            );
            assert_eq!(
                (q - 1) % two_n,
                0,
                "CKKS profile modulus must satisfy 2N | q - 1"
            );
        }

        // Constructing the plan validates that every modulus actually admits
        // the negacyclic NTT required by the implementation.
        let _ = self.rns_ntt_plan();

        assert!(
            !self.security_bearing,
            "security-bearing CKKS profiles require R2.9b security validation"
        );
    }
}

/// Existing small profile used throughout the RNS/canonical CKKS correctness
/// tests. It is deliberately not security-bearing.
pub fn correctness_profile_8() -> CkksParameterProfile {
    CkksParameterProfile::new(
        "correctness-8",
        8,
        &[12_289, 40_961, 65_537],
        65_537.0,
        CkksParameterClass::CorrectnessOriented,
        false,
    )
}

/// Large-N research profile derived from the Microsoft SEAL tc128
/// coefficient-modulus table for polynomial degree 4096.
///
/// This profile is intentionally not marked security-bearing. R2.9b
/// performs the independent RLWE security analysis.
pub fn research_profile_4096() -> CkksParameterProfile {
    CkksParameterProfile::new(
        "research-4096",
        4096,
        &[0x0ffffee001, 0x0ffffc4001, 0x1ffffe0001],
        2.0_f64.powi(30),
        CkksParameterClass::Research,
        false,
    )
}

/// Large-N research profile derived from the Microsoft SEAL tc128
/// coefficient-modulus table for polynomial degree 8192.
pub fn research_profile_8192() -> CkksParameterProfile {
    CkksParameterProfile::new(
        "research-8192",
        8192,
        &[
            0x07fffffd8001,
            0x07fffffc8001,
            0x0fffffffc001,
            0x0ffffff6c001,
            0x0fffffebc001,
        ],
        2.0_f64.powi(40),
        CkksParameterClass::Research,
        false,
    )
}

/// Large-N research profile derived from the Microsoft SEAL tc128
/// coefficient-modulus table for polynomial degree 16384.
pub fn research_profile_16384() -> CkksParameterProfile {
    CkksParameterProfile::new(
        "research-16384",
        16384,
        &[
            0x0fffffffd8001,
            0x0fffffffa0001,
            0x0fffffff00001,
            0x1fffffff68001,
            0x1fffffff50001,
            0x1ffffffee8001,
            0x1ffffffea0001,
            0x1ffffffe88001,
            0x1ffffffe48001,
        ],
        2.0_f64.powi(45),
        CkksParameterClass::Research,
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correctness_profile_metadata_is_exact() {
        let profile = correctness_profile_8();

        assert_eq!(profile.name(), "correctness-8");
        assert_eq!(profile.degree(), 8);
        assert_eq!(profile.slot_count(), 4);
        assert_eq!(profile.modulus_values(), &[12_289, 40_961, 65_537]);
        assert_eq!(profile.initial_scale(), 65_537.0);
        assert_eq!(
            profile.parameter_class(),
            CkksParameterClass::CorrectnessOriented
        );
        assert!(!profile.security_bearing());
    }

    #[test]
    fn correctness_profile_builds_expected_chain() {
        let profile = correctness_profile_8();
        let chain = profile.modulus_chain();

        assert_eq!(chain.len(), 3);
        assert_eq!(chain.level(0), &profile.modulus_basis());
        assert_eq!(
            chain.level(1).moduli(),
            &[Modulus::new(12_289), Modulus::new(40_961)]
        );
        assert_eq!(chain.level(2).moduli(), &[Modulus::new(12_289)]);
    }

    #[test]
    fn correctness_profile_builds_top_state() {
        let profile = correctness_profile_8();
        let chain = profile.modulus_chain();
        let state = profile.top_state();

        assert_eq!(state.level(), 0);
        assert_eq!(state.basis(), chain.top());
        assert_eq!(state.scale(), profile.initial_scale());
    }

    #[test]
    fn correctness_profile_builds_ntt_plan() {
        let profile = correctness_profile_8();
        let plan = profile.rns_ntt_plan();

        assert_eq!(plan.degree(), 8);
        assert_eq!(plan.moduli(), profile.modulus_basis().moduli());

        for &modulus in plan.moduli() {
            let two_n = (2 * profile.degree()) as u64;
            let psi = plan
                .moduli()
                .iter()
                .position(|candidate| *candidate == modulus)
                .map(|index| plan.plan(index).psi())
                .expect("plan modulus must be present");

            assert_eq!(modulus.pow(psi, two_n), 1);
            assert_eq!(
                modulus.pow(psi, profile.degree() as u64),
                modulus.value() - 1
            );
        }
    }

    #[test]
    fn modulus_bit_budget_is_sum_of_limb_widths() {
        let profile = correctness_profile_8();
        let expected = [12_289_u64, 40_961, 65_537]
            .iter()
            .map(|&q| u64::BITS - q.leading_zeros())
            .sum::<u32>();

        assert_eq!(profile.total_modulus_bits(), expected);
    }

    #[test]
    fn research_4096_canonical_embedding_roundtrips_sparse_slots() {
        use num_complex::Complex64;

        let profile = research_profile_4096();
        let embedding = super::super::CkksCanonicalEmbedding::new(profile.degree());

        assert_eq!(embedding.degree(), profile.degree());
        assert_eq!(embedding.slot_count(), profile.slot_count());

        let mut slots = vec![Complex64::new(0.0, 0.0); profile.slot_count()];

        // Deterministic sparse vector keeps the test semantically meaningful
        // without making its expected values depend on random generation.
        slots[0] = Complex64::new(1.25, -0.50);
        slots[1] = Complex64::new(-0.75, 0.25);
        slots[17] = Complex64::new(0.125, 0.375);
        slots[257] = Complex64::new(-1.50, -0.625);
        slots[1023] = Complex64::new(2.0, 0.125);
        slots[2047] = Complex64::new(-0.25, 1.0);

        let coefficients = embedding.slots_to_coefficients(&slots);
        let recovered = embedding.coefficients_to_slots(&coefficients);

        let max_error = recovered
            .iter()
            .zip(&slots)
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0_f64, f64::max);

        assert!(
            max_error <= 1.0e-9,
            "research-4096 canonical roundtrip max_error={max_error:e}"
        );
    }

    #[test]
    fn research_profiles_support_large_n_rns_ntt_roundtrips() {
        use crate::ring::Polynomial;

        for profile in [
            research_profile_4096(),
            research_profile_8192(),
            research_profile_16384(),
        ] {
            let plan = profile.rns_ntt_plan();
            let basis = profile.modulus_basis();

            assert_eq!(plan.degree(), profile.degree());
            assert_eq!(plan.moduli(), basis.moduli());

            for (index, &modulus) in basis.moduli().iter().enumerate() {
                let polynomial = Polynomial::new(
                    modulus,
                    (0..profile.degree())
                        .map(|coefficient| {
                            let i = coefficient as u64;
                            (7 + 13 * i + 5 * i * i) % modulus.value()
                        })
                        .collect(),
                );

                let transformed = plan.plan(index).forward_radix2(&polynomial);
                let recovered = plan.plan(index).inverse_radix2(&transformed);

                assert_eq!(
                    recovered,
                    polynomial,
                    "large-N NTT roundtrip failed for {} limb {}",
                    profile.name(),
                    index
                );
            }
        }
    }

    #[test]
    fn research_4096_supports_ntt_backed_rns_encrypt_decrypt() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        use crate::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_ntt_rng};
        use crate::ring::RnsPolynomial;

        let profile = research_profile_4096();
        let basis = profile.modulus_basis();
        let plan = profile.rns_ntt_plan();

        let secret: Vec<i8> = (0..profile.degree())
            .map(|index| match index % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();

        let coefficients: Vec<u128> = (0..profile.degree())
            .map(|index| {
                if index < 16 {
                    (17 + 13 * index) as u128
                } else {
                    0
                }
            })
            .collect();

        let message = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &coefficients);

        let mut rng = ChaCha20Rng::seed_from_u64(0x290A_300E);

        /*
         * Zero noise makes this a strict functional smoke rather than
         * a precision/noise experiment. R2.9c characterizes noise.
         */
        let ciphertext = encrypt_rns_raw_with_ntt_rng(&message, 2, 0, &secret, &plan, &mut rng);

        let recovered = decrypt_rns_raw_with_ntt(&ciphertext, &secret, &plan);

        assert_eq!(recovered, message);
    }

    #[test]
    #[should_panic(expected = "power of two of at least 2")]
    fn rejects_invalid_degree() {
        let _ = CkksParameterProfile::new(
            "bad-degree",
            12,
            &[12_289],
            64.0,
            CkksParameterClass::Research,
            false,
        );
    }

    #[test]
    #[should_panic(expected = "at least one modulus")]
    fn rejects_empty_modulus_chain() {
        let _ =
            CkksParameterProfile::new("empty", 8, &[], 64.0, CkksParameterClass::Research, false);
    }

    #[test]
    #[should_panic(expected = "pairwise distinct")]
    fn rejects_duplicate_moduli() {
        let _ = CkksParameterProfile::new(
            "duplicate",
            8,
            &[12_289, 12_289],
            64.0,
            CkksParameterClass::Research,
            false,
        );
    }

    #[test]
    #[should_panic(expected = "2N | q - 1")]
    fn rejects_ntt_incompatible_modulus() {
        let _ = CkksParameterProfile::new(
            "bad-ntt",
            8,
            &[101],
            64.0,
            CkksParameterClass::Research,
            false,
        );
    }

    #[test]
    #[should_panic(expected = "positive and finite")]
    fn rejects_invalid_scale() {
        let _ = CkksParameterProfile::new(
            "bad-scale",
            8,
            &[12_289],
            0.0,
            CkksParameterClass::Research,
            false,
        );
    }

    #[test]
    #[should_panic(expected = "R2.9b security validation")]
    fn refuses_unvalidated_security_claim() {
        let _ = CkksParameterProfile::new(
            "premature-security-claim",
            8,
            &[12_289],
            64.0,
            CkksParameterClass::Research,
            true,
        );
    }
}
