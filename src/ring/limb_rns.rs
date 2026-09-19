use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};

use super::{LimbNttPlan, LimbPolynomial, NttLimbArithmetic};

/// Width-parametric RNS polynomial.
///
/// All physical limbs use the same word backend `A`, while every limb has its
/// own pairwise-coprime modulus. The logical composite modulus is exact and
/// may exceed `u128`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LimbRnsPolynomial<A>
where
    A: NttLimbArithmetic,
{
    residues: Vec<LimbPolynomial<A>>,
}

impl<A> LimbRnsPolynomial<A>
where
    A: NttLimbArithmetic,
{
    pub fn from_big_coefficients(arithmetic: Vec<A>, coefficients: &[BigUint]) -> Self {
        assert!(
            !arithmetic.is_empty(),
            "width-parametric RNS basis must contain at least one limb"
        );
        assert!(
            !coefficients.is_empty(),
            "width-parametric RNS polynomial degree must be positive"
        );

        assert_pairwise_coprime(&arithmetic);

        let residues = arithmetic
            .into_iter()
            .map(|limb| {
                let modulus_u128 = A::word_to_u128(limb.modulus());
                let modulus = BigUint::from(modulus_u128);

                let reduced = coefficients
                    .iter()
                    .map(|coefficient| {
                        let value = (coefficient % &modulus)
                            .to_u128()
                            .expect("coefficient reduced modulo u128 must fit");

                        A::word_from_u128(value)
                    })
                    .collect();

                LimbPolynomial::new(limb, reduced)
            })
            .collect();

        Self { residues }
    }

    pub fn from_residues(residues: Vec<LimbPolynomial<A>>) -> Self {
        assert!(
            !residues.is_empty(),
            "width-parametric RNS basis must contain at least one limb"
        );

        let degree = residues[0].degree();

        assert!(
            residues.iter().all(|residue| residue.degree() == degree),
            "all width-parametric RNS residue polynomials must have the same degree"
        );

        let arithmetic: Vec<A> = residues
            .iter()
            .map(|residue| residue.arithmetic().clone())
            .collect();

        assert_pairwise_coprime(&arithmetic);

        Self { residues }
    }

    pub fn residues(&self) -> &[LimbPolynomial<A>] {
        &self.residues
    }

    pub fn residue(&self, index: usize) -> &LimbPolynomial<A> {
        &self.residues[index]
    }

    pub fn limb_count(&self) -> usize {
        self.residues.len()
    }

    pub fn degree(&self) -> usize {
        self.residues[0].degree()
    }

    pub fn composite_modulus_big(&self) -> BigUint {
        self.residues
            .iter()
            .fold(BigUint::one(), |product, residue| {
                product * BigUint::from(A::word_to_u128(residue.arithmetic().modulus()))
            })
    }

    /// Exact arbitrary-precision CRT reconstruction via incremental Garner.
    pub fn reconstruct_coefficients_big(&self) -> Vec<BigUint> {
        (0..self.degree())
            .map(|coefficient_index| {
                let mut value = BigUint::zero();
                let mut accumulated_modulus = BigUint::one();

                for residue in &self.residues {
                    let modulus = A::word_to_u128(residue.arithmetic().modulus());
                    let residue_value = A::word_to_u128(residue.coefficients()[coefficient_index]);

                    let value_mod = (&value % modulus)
                        .to_u128()
                        .expect("BigUint modulo u128 must fit u128");

                    let accumulated_modulus_mod = (&accumulated_modulus % modulus)
                        .to_u128()
                        .expect("BigUint modulo u128 must fit u128");

                    let inverse = inverse_mod_u128(accumulated_modulus_mod, modulus)
                        .expect("RNS moduli must be pairwise coprime");

                    let delta = if residue_value >= value_mod {
                        residue_value - value_mod
                    } else {
                        modulus - (value_mod - residue_value)
                    };

                    let correction = mul_mod_u128(delta, inverse, modulus);

                    value += &accumulated_modulus * BigUint::from(correction);
                    accumulated_modulus *= modulus;
                }

                value
            })
            .collect()
    }

    pub fn add(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        Self::from_residues(
            self.residues
                .iter()
                .zip(rhs.residues.iter())
                .map(|(lhs, rhs)| lhs.add(rhs))
                .collect(),
        )
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        Self::from_residues(
            self.residues
                .iter()
                .zip(rhs.residues.iter())
                .map(|(lhs, rhs)| lhs.sub(rhs))
                .collect(),
        )
    }

    pub fn neg(&self) -> Self {
        Self::from_residues(self.residues.iter().map(LimbPolynomial::neg).collect())
    }

    fn assert_compatible(&self, rhs: &Self) {
        assert_eq!(
            self.limb_count(),
            rhs.limb_count(),
            "width-parametric RNS limb counts must match"
        );
        assert_eq!(
            self.degree(),
            rhs.degree(),
            "width-parametric RNS polynomial degrees must match"
        );

        for (lhs, rhs) in self.residues.iter().zip(rhs.residues.iter()) {
            assert!(
                lhs.arithmetic() == rhs.arithmetic(),
                "width-parametric RNS physical moduli must match"
            );
        }
    }
}

/// One NTT plan per physical limb of a width-parametric RNS basis.
#[derive(Clone, PartialEq, Eq)]
pub struct LimbRnsNttPlan<A>
where
    A: NttLimbArithmetic,
{
    plans: Vec<LimbNttPlan<A>>,
    degree: usize,
}

impl<A> LimbRnsNttPlan<A>
where
    A: NttLimbArithmetic,
{
    pub fn new(arithmetic: Vec<A>, degree: usize) -> Self {
        assert!(
            !arithmetic.is_empty(),
            "width-parametric RNS NTT requires at least one physical limb"
        );

        assert_pairwise_coprime(&arithmetic);

        let plans = arithmetic
            .into_iter()
            .map(|limb| LimbNttPlan::new(limb, degree))
            .collect();

        Self { plans, degree }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn limb_count(&self) -> usize {
        self.plans.len()
    }

    pub fn negacyclic_mul(
        &self,
        lhs: &LimbRnsPolynomial<A>,
        rhs: &LimbRnsPolynomial<A>,
    ) -> LimbRnsPolynomial<A> {
        assert_eq!(
            lhs.degree(),
            self.degree,
            "left RNS polynomial degree must match NTT plan"
        );
        assert_eq!(
            rhs.degree(),
            self.degree,
            "right RNS polynomial degree must match NTT plan"
        );
        assert_eq!(
            lhs.limb_count(),
            self.limb_count(),
            "left RNS polynomial limb count must match NTT plan"
        );
        assert_eq!(
            rhs.limb_count(),
            self.limb_count(),
            "right RNS polynomial limb count must match NTT plan"
        );

        let residues = self
            .plans
            .iter()
            .enumerate()
            .map(|(index, plan)| {
                assert!(
                    lhs.residue(index).arithmetic() == plan.arithmetic(),
                    "left RNS limb arithmetic must match NTT plan"
                );
                assert!(
                    rhs.residue(index).arithmetic() == plan.arithmetic(),
                    "right RNS limb arithmetic must match NTT plan"
                );

                plan.negacyclic_mul(lhs.residue(index), rhs.residue(index))
            })
            .collect();

        LimbRnsPolynomial::from_residues(residues)
    }
}

fn assert_pairwise_coprime<A>(arithmetic: &[A])
where
    A: NttLimbArithmetic,
{
    for lhs in 0..arithmetic.len() {
        for rhs in (lhs + 1)..arithmetic.len() {
            let lhs_modulus = A::word_to_u128(arithmetic[lhs].modulus());
            let rhs_modulus = A::word_to_u128(arithmetic[rhs].modulus());

            assert_eq!(
                gcd_u128(lhs_modulus, rhs_modulus),
                1,
                "width-parametric RNS moduli must be pairwise coprime"
            );
        }
    }
}

fn gcd_u128(mut lhs: u128, mut rhs: u128) -> u128 {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }

    lhs
}

fn inverse_mod_u128(value: u128, modulus: u128) -> Option<u128> {
    if value == 0 || modulus <= 1 {
        return None;
    }

    use num_bigint::BigInt;
    use num_traits::{One, Signed};

    let mut old_r = BigInt::from(value);
    let mut r = BigInt::from(modulus);
    let mut old_s = BigInt::one();
    let mut s = BigInt::zero();

    while !r.is_zero() {
        let quotient = &old_r / &r;

        let next_r = &old_r - &quotient * &r;
        old_r = r;
        r = next_r;

        let next_s = &old_s - &quotient * &s;
        old_s = s;
        s = next_s;
    }

    if old_r != BigInt::one() {
        return None;
    }

    let modulus_big = BigInt::from(modulus);
    let mut result = old_s % &modulus_big;

    if result.is_negative() {
        result += &modulus_big;
    }

    result.to_u128()
}

fn mul_mod_u128(lhs: u128, rhs: u128, modulus: u128) -> u128 {
    ((BigUint::from(lhs) * BigUint::from(rhs)) % BigUint::from(modulus))
        .to_u128()
        .expect("value reduced modulo u128 must fit u128")
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use num_traits::Zero;

    use crate::ring::{Limb128, Limb32, Limb64};

    use super::{LimbRnsNttPlan, LimbRnsPolynomial};

    const MODULI: [u128; 3] = [12_289, 40_961, 65_537];

    fn coefficients(degree: usize, offset: u128) -> Vec<BigUint> {
        let composite = MODULI.iter().product::<u128>();

        (0..degree)
            .map(|index| {
                let i = index as u128;
                BigUint::from((offset + 17 * i + 5 * i * i + 3 * i * i * i) % composite)
            })
            .collect()
    }

    fn backends32() -> Vec<Limb32> {
        MODULI.iter().map(|&q| Limb32::new(q as u32)).collect()
    }

    fn backends64() -> Vec<Limb64> {
        MODULI.iter().map(|&q| Limb64::new(q as u64)).collect()
    }

    fn backends128() -> Vec<Limb128> {
        MODULI.iter().map(|&q| Limb128::new(q)).collect()
    }

    #[test]
    fn same_rns_basis_roundtrips_exactly_across_all_word_widths() {
        let degree = 64;
        let values = coefficients(degree, 7);

        let rns32 = LimbRnsPolynomial::from_big_coefficients(backends32(), &values);
        let rns64 = LimbRnsPolynomial::from_big_coefficients(backends64(), &values);
        let rns128 = LimbRnsPolynomial::from_big_coefficients(backends128(), &values);

        assert_eq!(rns32.reconstruct_coefficients_big(), values);
        assert_eq!(rns64.reconstruct_coefficients_big(), values);
        assert_eq!(rns128.reconstruct_coefficients_big(), values);

        assert_eq!(rns32.composite_modulus_big(), rns64.composite_modulus_big());
        assert_eq!(
            rns64.composite_modulus_big(),
            rns128.composite_modulus_big()
        );
    }

    #[test]
    fn rns_add_sub_neg_are_exact_across_all_word_widths() {
        let degree = 64;
        let lhs_values = coefficients(degree, 5);
        let rhs_values = coefficients(degree, 29);

        let lhs32 = LimbRnsPolynomial::from_big_coefficients(backends32(), &lhs_values);
        let rhs32 = LimbRnsPolynomial::from_big_coefficients(backends32(), &rhs_values);

        let lhs64 = LimbRnsPolynomial::from_big_coefficients(backends64(), &lhs_values);
        let rhs64 = LimbRnsPolynomial::from_big_coefficients(backends64(), &rhs_values);

        let lhs128 = LimbRnsPolynomial::from_big_coefficients(backends128(), &lhs_values);
        let rhs128 = LimbRnsPolynomial::from_big_coefficients(backends128(), &rhs_values);

        assert_eq!(
            lhs32.add(&rhs32).reconstruct_coefficients_big(),
            lhs64.add(&rhs64).reconstruct_coefficients_big()
        );
        assert_eq!(
            lhs64.add(&rhs64).reconstruct_coefficients_big(),
            lhs128.add(&rhs128).reconstruct_coefficients_big()
        );

        assert_eq!(
            lhs32.sub(&rhs32).reconstruct_coefficients_big(),
            lhs64.sub(&rhs64).reconstruct_coefficients_big()
        );
        assert_eq!(
            lhs64.sub(&rhs64).reconstruct_coefficients_big(),
            lhs128.sub(&rhs128).reconstruct_coefficients_big()
        );

        assert_eq!(
            lhs32.neg().reconstruct_coefficients_big(),
            lhs64.neg().reconstruct_coefficients_big()
        );
        assert_eq!(
            lhs64.neg().reconstruct_coefficients_big(),
            lhs128.neg().reconstruct_coefficients_big()
        );
    }

    #[test]
    fn ntt_rns_product_is_exact_across_32_64_128_bit_backends() {
        let degree = 64;
        let lhs_values = coefficients(degree, 11);
        let rhs_values = coefficients(degree, 37);

        let lhs32 = LimbRnsPolynomial::from_big_coefficients(backends32(), &lhs_values);
        let rhs32 = LimbRnsPolynomial::from_big_coefficients(backends32(), &rhs_values);
        let plan32 = LimbRnsNttPlan::new(backends32(), degree);

        let lhs64 = LimbRnsPolynomial::from_big_coefficients(backends64(), &lhs_values);
        let rhs64 = LimbRnsPolynomial::from_big_coefficients(backends64(), &rhs_values);
        let plan64 = LimbRnsNttPlan::new(backends64(), degree);

        let lhs128 = LimbRnsPolynomial::from_big_coefficients(backends128(), &lhs_values);
        let rhs128 = LimbRnsPolynomial::from_big_coefficients(backends128(), &rhs_values);
        let plan128 = LimbRnsNttPlan::new(backends128(), degree);

        let product32 = plan32
            .negacyclic_mul(&lhs32, &rhs32)
            .reconstruct_coefficients_big();

        let product64 = plan64
            .negacyclic_mul(&lhs64, &rhs64)
            .reconstruct_coefficients_big();

        let product128 = plan128
            .negacyclic_mul(&lhs128, &rhs128)
            .reconstruct_coefficients_big();

        assert_eq!(product32, product64);
        assert_eq!(product64, product128);
    }

    #[test]
    fn ntt_rns_product_matches_biguint_negacyclic_oracle() {
        let degree = 64;
        let lhs_values = coefficients(degree, 13);
        let rhs_values = coefficients(degree, 41);

        let lhs = LimbRnsPolynomial::from_big_coefficients(backends64(), &lhs_values);
        let rhs = LimbRnsPolynomial::from_big_coefficients(backends64(), &rhs_values);
        let plan = LimbRnsNttPlan::new(backends64(), degree);

        let actual = plan
            .negacyclic_mul(&lhs, &rhs)
            .reconstruct_coefficients_big();

        let modulus = lhs.composite_modulus_big();
        let mut expected = vec![BigUint::zero(); degree];

        for (lhs_index, lhs_value) in lhs_values.iter().enumerate() {
            for (rhs_index, rhs_value) in rhs_values.iter().enumerate() {
                let product = (lhs_value * rhs_value) % &modulus;
                let raw_index = lhs_index + rhs_index;

                if raw_index < degree {
                    expected[raw_index] = (&expected[raw_index] + &product) % &modulus;
                } else {
                    let index = raw_index - degree;

                    expected[index] = if expected[index] >= product {
                        &expected[index] - &product
                    } else {
                        &modulus - (&product - &expected[index])
                    };
                }
            }
        }

        assert_eq!(actual, expected);
    }

    #[test]
    fn composite_modulus_can_exceed_u128() {
        let arithmetic = vec![
            Limb128::new((1_u128 << 127) - 1),
            Limb128::new((1_u128 << 127) - 3),
        ];

        let values = vec![
            BigUint::from(0_u8),
            BigUint::from(1_u8),
            (BigUint::from(1_u8) << 180_usize) + BigUint::from(17_u8),
        ];

        let rns = LimbRnsPolynomial::from_big_coefficients(arithmetic, &values);

        assert!(rns.composite_modulus_big().bits() > 128);
        assert_eq!(rns.reconstruct_coefficients_big(), values);
    }
}
