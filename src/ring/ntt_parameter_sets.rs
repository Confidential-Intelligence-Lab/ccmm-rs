use super::{
    is_ntt_compatible, is_ntt_prime_candidate, Limb128, Limb32, Limb64, LimbNttPlan,
    PhysicalLimbArithmetic,
};

pub const CANONICAL_NTT_DEGREE: usize = 4096;

pub const NTT32_MODULUS: u32 = 4_294_828_033;
pub const NTT32_PSI: u32 = 1_953_722_822;

pub const NTT64_MODULUS: u64 = 18_446_744_073_709_436_929;
pub const NTT64_PSI: u64 = 5_975_861_664_659_593_359;

pub const NTT128_MODULUS: u128 = 1_329_227_995_784_915_872_903_807_060_279_713_793;
pub const NTT128_PSI: u128 = 84_935_201_209_993_441_529_728_139_364_756_967;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalNttWidth {
    Bits32,
    Bits64,
    Bits128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalNttParameter {
    pub width: CanonicalNttWidth,
    pub modulus: u128,
    pub modulus_bits: u32,
    pub degree: usize,
    pub psi: u128,
}

pub const CANONICAL_NTT_PARAMETERS: [CanonicalNttParameter; 3] = [
    CanonicalNttParameter {
        width: CanonicalNttWidth::Bits32,
        modulus: NTT32_MODULUS as u128,
        modulus_bits: 32,
        degree: CANONICAL_NTT_DEGREE,
        psi: NTT32_PSI as u128,
    },
    CanonicalNttParameter {
        width: CanonicalNttWidth::Bits64,
        modulus: NTT64_MODULUS as u128,
        modulus_bits: 64,
        degree: CANONICAL_NTT_DEGREE,
        psi: NTT64_PSI as u128,
    },
    CanonicalNttParameter {
        width: CanonicalNttWidth::Bits128,
        modulus: NTT128_MODULUS,
        modulus_bits: 120,
        degree: CANONICAL_NTT_DEGREE,
        psi: NTT128_PSI,
    },
];

pub fn validate_canonical_ntt_parameters() {
    for parameter in CANONICAL_NTT_PARAMETERS {
        assert!(
            is_ntt_prime_candidate(parameter.modulus),
            "canonical NTT modulus must pass the repository primality screen"
        );
        assert!(
            is_ntt_compatible(parameter.modulus, parameter.degree),
            "canonical NTT modulus must support the configured degree"
        );
    }

    let plan32 = LimbNttPlan::new(Limb32::new(NTT32_MODULUS), CANONICAL_NTT_DEGREE);
    assert_eq!(u128::from(plan32.psi()), u128::from(NTT32_PSI));

    let plan64 = LimbNttPlan::new(Limb64::new(NTT64_MODULUS), CANONICAL_NTT_DEGREE);
    assert_eq!(u128::from(plan64.psi()), u128::from(NTT64_PSI));

    let plan128 = LimbNttPlan::new(Limb128::new(NTT128_MODULUS), CANONICAL_NTT_DEGREE);
    assert_eq!(plan128.psi(), NTT128_PSI);

    validate_root32(&plan32);
    validate_root64(&plan64);
    validate_root128(&plan128);
}

fn validate_root32(plan: &LimbNttPlan<Limb32>) {
    let psi = plan.psi();
    assert_eq!(
        plan.arithmetic().pow_mod(psi, CANONICAL_NTT_DEGREE as u128),
        NTT32_MODULUS - 1
    );
    assert_eq!(
        plan.arithmetic()
            .pow_mod(psi, (2 * CANONICAL_NTT_DEGREE) as u128),
        1
    );
}

fn validate_root64(plan: &LimbNttPlan<Limb64>) {
    let psi = plan.psi();
    assert_eq!(
        plan.arithmetic().pow_mod(psi, CANONICAL_NTT_DEGREE as u128),
        NTT64_MODULUS - 1
    );
    assert_eq!(
        plan.arithmetic()
            .pow_mod(psi, (2 * CANONICAL_NTT_DEGREE) as u128),
        1
    );
}

fn validate_root128(plan: &LimbNttPlan<Limb128>) {
    let psi = plan.psi();
    assert_eq!(
        plan.arithmetic().pow_mod(psi, CANONICAL_NTT_DEGREE as u128),
        NTT128_MODULUS - 1
    );
    assert_eq!(
        plan.arithmetic()
            .pow_mod(psi, (2 * CANONICAL_NTT_DEGREE) as u128),
        1
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_parameter_contract_is_exact() {
        validate_canonical_ntt_parameters();
    }

    #[test]
    fn canonical_widths_are_native_and_distinct() {
        assert!(NTT32_MODULUS as u128 <= u32::MAX as u128);
        assert!(NTT64_MODULUS as u128 <= u64::MAX as u128);
        assert!(NTT128_MODULUS > u64::MAX as u128);

        assert_eq!(32 - NTT32_MODULUS.leading_zeros(), 32);
        assert_eq!(64 - NTT64_MODULUS.leading_zeros(), 64);
        assert_eq!(128 - NTT128_MODULUS.leading_zeros(), 120);
    }
}
