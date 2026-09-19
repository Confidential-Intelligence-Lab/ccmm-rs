use super::parameters::{CkksErrorDistribution, CkksSecretDistribution};
use super::CkksParameterProfile;
use crate::grafting::BoundedGadgetLayout;

#[derive(Debug, Clone, PartialEq)]
pub struct CkksSecurityExposure {
    profile_name: &'static str,
    degree: usize,
    slot_count: usize,
    level: usize,
    modulus_values: Vec<u64>,
    composite_modulus: u128,
    composite_modulus_bits: u32,
    secret_distribution: CkksSecretDistribution,
    error_distribution: CkksErrorDistribution,
    auxiliary_modulus_present: bool,
    bounded_base_log: u32,
    bounded_base: u128,
    bounded_digit_count: usize,
    multiplication_key_rlwe_samples: usize,
    multiplication_key_logical_bytes: usize,
}

impl CkksSecurityExposure {
    pub fn bounded_reference(
        profile: &CkksParameterProfile,
        level: usize,
        base_log: u32,
        secret_distribution: CkksSecretDistribution,
        error_distribution: CkksErrorDistribution,
    ) -> Self {
        let chain = profile.modulus_chain();

        assert!(
            level < chain.len(),
            "security exposure level must exist in CKKS chain"
        );

        let basis = chain.level(level).clone();
        let layout = BoundedGadgetLayout::new(basis.clone(), base_log);

        let composite_modulus = basis.composite_modulus();
        let composite_modulus_bits = u128::BITS - composite_modulus.leading_zeros();

        let digit_count = layout.digit_count();

        let logical_bytes =
            digit_count * basis.len() * 2 * profile.degree() * core::mem::size_of::<u64>();

        Self {
            profile_name: profile.name(),
            degree: profile.degree(),
            slot_count: profile.slot_count(),
            level,
            modulus_values: basis
                .moduli()
                .iter()
                .map(|modulus| modulus.value())
                .collect(),
            composite_modulus,
            composite_modulus_bits,
            secret_distribution,
            error_distribution,
            auxiliary_modulus_present: false,
            bounded_base_log: base_log,
            bounded_base: layout.base(),
            bounded_digit_count: digit_count,
            multiplication_key_rlwe_samples: digit_count,
            multiplication_key_logical_bytes: logical_bytes,
        }
    }

    pub fn profile_name(&self) -> &'static str {
        self.profile_name
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn modulus_values(&self) -> &[u64] {
        &self.modulus_values
    }

    pub fn composite_modulus(&self) -> u128 {
        self.composite_modulus
    }

    pub fn composite_modulus_bits(&self) -> u32 {
        self.composite_modulus_bits
    }

    pub fn bounded_base_log(&self) -> u32 {
        self.bounded_base_log
    }

    pub fn bounded_base(&self) -> u128 {
        self.bounded_base
    }

    pub fn bounded_digit_count(&self) -> usize {
        self.bounded_digit_count
    }

    pub fn multiplication_key_rlwe_samples(&self) -> usize {
        self.multiplication_key_rlwe_samples
    }

    pub fn multiplication_key_logical_bytes(&self) -> usize {
        self.multiplication_key_logical_bytes
    }

    pub fn auxiliary_modulus_present(&self) -> bool {
        self.auxiliary_modulus_present
    }

    pub fn to_report_string(&self) -> String {
        let secret = match self.secret_distribution {
            CkksSecretDistribution::UniformTernary => "uniform-ternary",
        };

        let (error_name, sigma) = match self.error_distribution {
            CkksErrorDistribution::DiscreteGaussian { sigma } => ("discrete-gaussian", sigma),
        };

        let moduli = self
            .modulus_values
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");

        format!(
            concat!(
                "CKKS_SECURITY_EXPOSURE_VERSION=1\n",
                "PROFILE={}\n",
                "RING_DEGREE={}\n",
                "SLOT_COUNT={}\n",
                "LEVEL={}\n",
                "RNS_LIMBS={}\n",
                "MODULI={}\n",
                "COMPOSITE_Q={}\n",
                "COMPOSITE_Q_BITS={}\n",
                "SECRET_DISTRIBUTION={}\n",
                "ERROR_DISTRIBUTION={}\n",
                "ERROR_SIGMA={}\n",
                "AUXILIARY_MODULUS_PRESENT={}\n",
                "BOUNDED_BASE_LOG={}\n",
                "BOUNDED_BASE={}\n",
                "BOUNDED_DIGITS={}\n",
                "MULTIPLICATION_KEY_RLWE_SAMPLES={}\n",
                "MULTIPLICATION_KEY_LOGICAL_BYTES={}\n"
            ),
            self.profile_name,
            self.degree,
            self.slot_count,
            self.level,
            self.modulus_values.len(),
            moduli,
            self.composite_modulus,
            self.composite_modulus_bits,
            secret,
            error_name,
            sigma,
            self.auxiliary_modulus_present,
            self.bounded_base_log,
            self.bounded_base,
            self.bounded_digit_count,
            self.multiplication_key_rlwe_samples,
            self.multiplication_key_logical_bytes,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ckks::research_profile_4096;

    #[test]
    fn research_4096_top_level_bounded_exposure_is_exact() {
        let profile = research_profile_4096();

        let exposure = CkksSecurityExposure::bounded_reference(
            &profile,
            0,
            20,
            CkksSecretDistribution::UniformTernary,
            CkksErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        );

        assert_eq!(exposure.profile_name(), "research-4096");
        assert_eq!(exposure.degree(), 4096);
        assert_eq!(exposure.slot_count(), 2048);
        assert_eq!(exposure.level(), 0);
        assert_eq!(exposure.modulus_values(), profile.modulus_values());
        assert_eq!(exposure.bounded_base_log(), 20);
        assert_eq!(exposure.bounded_base(), 1_u128 << 20);
        assert_eq!(exposure.bounded_digit_count(), 6);
        assert_eq!(exposure.multiplication_key_rlwe_samples(), 6);
        assert_eq!(exposure.multiplication_key_logical_bytes(), 1_179_648);
        assert!(!exposure.auxiliary_modulus_present());
    }

    #[test]
    fn exact_composite_bit_length_is_not_limb_width_sum() {
        let profile = research_profile_4096();

        let exposure = CkksSecurityExposure::bounded_reference(
            &profile,
            0,
            20,
            CkksSecretDistribution::UniformTernary,
            CkksErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        );

        assert_eq!(profile.total_modulus_bits(), 105);
        assert!(exposure.composite_modulus_bits() <= profile.total_modulus_bits());
    }

    #[test]
    fn research_4096_all_levels_have_expected_bounded_exposure() {
        let profile = research_profile_4096();

        let expected = [
            (0_usize, 3_usize, 105_u32, 6_usize, 1_179_648_usize),
            (1_usize, 2_usize, 70_u32, 4_usize, 524_288_usize),
            (2_usize, 1_usize, 35_u32, 2_usize, 131_072_usize),
        ];

        for (level, limbs, q_bits, digits, logical_bytes) in expected {
            let exposure = CkksSecurityExposure::bounded_reference(
                &profile,
                level,
                20,
                CkksSecretDistribution::UniformTernary,
                CkksErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            );

            assert_eq!(exposure.level(), level);
            assert_eq!(exposure.modulus_values().len(), limbs);
            assert_eq!(exposure.composite_modulus_bits(), q_bits);
            assert_eq!(exposure.bounded_digit_count(), digits);
            assert_eq!(exposure.multiplication_key_rlwe_samples(), digits);
            assert_eq!(exposure.multiplication_key_logical_bytes(), logical_bytes);
            assert!(!exposure.auxiliary_modulus_present());
        }
    }

    #[test]
    fn report_contains_complete_security_exposure_contract() {
        let profile = research_profile_4096();

        let report = CkksSecurityExposure::bounded_reference(
            &profile,
            0,
            20,
            CkksSecretDistribution::UniformTernary,
            CkksErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        )
        .to_report_string();

        for expected in [
            "CKKS_SECURITY_EXPOSURE_VERSION=1",
            "PROFILE=research-4096",
            "RING_DEGREE=4096",
            "SLOT_COUNT=2048",
            "LEVEL=0",
            "RNS_LIMBS=3",
            "SECRET_DISTRIBUTION=uniform-ternary",
            "ERROR_DISTRIBUTION=discrete-gaussian",
            "ERROR_SIGMA=3.19",
            "AUXILIARY_MODULUS_PRESENT=false",
            "BOUNDED_BASE_LOG=20",
            "BOUNDED_BASE=1048576",
            "BOUNDED_DIGITS=6",
            "MULTIPLICATION_KEY_RLWE_SAMPLES=6",
            "MULTIPLICATION_KEY_LOGICAL_BYTES=1179648",
        ] {
            assert!(
                report.contains(expected),
                "security exposure report missing {expected}"
            );
        }
    }
}
