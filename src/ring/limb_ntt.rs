use super::{Limb128, Limb32, Limb64, LimbPolynomial, PhysicalLimbArithmetic};

/// Conversion surface needed by the generic NTT planner.
///
/// This remains deliberately local to the NTT layer. The base physical-limb
/// arithmetic trait does not need representation-conversion methods.
pub trait NttLimbArithmetic: PhysicalLimbArithmetic + Clone + PartialEq + Eq {
    fn word_from_u128(value: u128) -> Self::Word;
    fn word_to_u128(value: Self::Word) -> u128;
}

impl NttLimbArithmetic for Limb32 {
    fn word_from_u128(value: u128) -> Self::Word {
        u32::try_from(value).expect("value must fit u32 physical word")
    }

    fn word_to_u128(value: Self::Word) -> u128 {
        u128::from(value)
    }
}

impl NttLimbArithmetic for Limb64 {
    fn word_from_u128(value: u128) -> Self::Word {
        u64::try_from(value).expect("value must fit u64 physical word")
    }

    fn word_to_u128(value: Self::Word) -> u128 {
        u128::from(value)
    }
}

impl NttLimbArithmetic for Limb128 {
    fn word_from_u128(value: u128) -> Self::Word {
        value
    }

    fn word_to_u128(value: Self::Word) -> u128 {
        value
    }
}

/// Width-parametric negacyclic radix-2 NTT plan for one physical limb.
///
/// The plan uses the standard twist construction:
///
/// - discover a primitive `2N`-th root `psi`;
/// - twist coefficient `a_j` by `psi^j`;
/// - apply a cyclic radix-2 NTT with `omega = psi^2`;
/// - inverse NTT and untwist by `psi^-j`.
///
/// All modular arithmetic is delegated to the selected physical-limb backend.
#[derive(Clone, PartialEq, Eq)]
pub struct LimbNttPlan<A>
where
    A: NttLimbArithmetic,
{
    arithmetic: A,
    degree: usize,
    psi: A::Word,
    psi_inverse: A::Word,
    omega: A::Word,
    omega_inverse: A::Word,
    degree_inverse: A::Word,
}

impl<A> LimbNttPlan<A>
where
    A: NttLimbArithmetic,
{
    pub fn new(arithmetic: A, degree: usize) -> Self {
        assert!(
            degree >= 2 && degree.is_power_of_two(),
            "NTT degree must be a power of two >= 2"
        );

        let modulus = A::word_to_u128(arithmetic.modulus());
        let two_n = (degree as u128)
            .checked_mul(2)
            .expect("NTT degree is too large");

        assert_eq!(
            (modulus - 1) % two_n,
            0,
            "2 * degree must divide modulus - 1"
        );

        let psi = find_negacyclic_root(&arithmetic, degree);
        let psi_inverse = arithmetic
            .inv_mod(psi)
            .expect("primitive NTT root must be invertible");

        let omega = arithmetic.mul_mod(psi, psi);
        let omega_inverse = arithmetic
            .inv_mod(omega)
            .expect("cyclic NTT root must be invertible");

        let degree_word = A::word_from_u128(degree as u128);
        let degree_inverse = arithmetic
            .inv_mod(degree_word)
            .expect("NTT degree must be invertible modulo physical modulus");

        Self {
            arithmetic,
            degree,
            psi,
            psi_inverse,
            omega,
            omega_inverse,
            degree_inverse,
        }
    }

    pub fn arithmetic(&self) -> &A {
        &self.arithmetic
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn psi(&self) -> A::Word {
        self.psi
    }

    pub fn forward(&self, polynomial: &LimbPolynomial<A>) -> Vec<A::Word> {
        self.assert_polynomial(polynomial);

        let mut values = polynomial.coefficients().to_vec();

        let mut twist = A::word_from_u128(1);
        for value in &mut values {
            *value = self.arithmetic.mul_mod(*value, twist);
            twist = self.arithmetic.mul_mod(twist, self.psi);
        }

        radix2_cyclic_ntt(&self.arithmetic, &mut values, self.omega);
        values
    }

    pub fn inverse(&self, values: &[A::Word]) -> LimbPolynomial<A> {
        assert_eq!(
            values.len(),
            self.degree,
            "NTT value count must match plan degree"
        );

        let mut coefficients = values.to_vec();

        radix2_cyclic_ntt(&self.arithmetic, &mut coefficients, self.omega_inverse);

        let mut untwist = A::word_from_u128(1);

        for value in &mut coefficients {
            *value = self.arithmetic.mul_mod(*value, self.degree_inverse);
            *value = self.arithmetic.mul_mod(*value, untwist);
            untwist = self.arithmetic.mul_mod(untwist, self.psi_inverse);
        }

        LimbPolynomial::new(self.arithmetic.clone(), coefficients)
    }

    pub fn negacyclic_mul(
        &self,
        lhs: &LimbPolynomial<A>,
        rhs: &LimbPolynomial<A>,
    ) -> LimbPolynomial<A> {
        self.assert_polynomial(lhs);
        self.assert_polynomial(rhs);

        let lhs_ntt = self.forward(lhs);
        let rhs_ntt = self.forward(rhs);

        let product: Vec<A::Word> = lhs_ntt
            .into_iter()
            .zip(rhs_ntt)
            .map(|(lhs, rhs)| self.arithmetic.mul_mod(lhs, rhs))
            .collect();

        self.inverse(&product)
    }

    fn assert_polynomial(&self, polynomial: &LimbPolynomial<A>) {
        assert!(
            polynomial.arithmetic() == &self.arithmetic,
            "polynomial arithmetic backend must match NTT plan"
        );

        assert_eq!(
            polynomial.degree(),
            self.degree,
            "polynomial degree must match NTT plan"
        );
    }
}

fn find_negacyclic_root<A>(arithmetic: &A, degree: usize) -> A::Word
where
    A: NttLimbArithmetic,
{
    let modulus = A::word_to_u128(arithmetic.modulus());
    let two_n = 2_u128 * degree as u128;
    let exponent = (modulus - 1) / two_n;
    let minus_one = A::word_from_u128(modulus - 1);

    for candidate in 2_u128..modulus {
        let candidate_word = A::word_from_u128(candidate);
        let psi = arithmetic.pow_mod(candidate_word, exponent);

        if arithmetic.pow_mod(psi, degree as u128) == minus_one
            && arithmetic.pow_mod(psi, two_n) == A::word_from_u128(1)
        {
            return psi;
        }
    }

    panic!("no primitive 2N-th root found");
}

fn radix2_cyclic_ntt<A>(arithmetic: &A, values: &mut [A::Word], root: A::Word)
where
    A: NttLimbArithmetic,
{
    let degree = values.len();
    bit_reverse_permute(values);

    let mut length = 2_usize;

    while length <= degree {
        let step = degree / length;
        let root_step = arithmetic.pow_mod(root, step as u128);

        for start in (0..degree).step_by(length) {
            let mut twiddle = A::word_from_u128(1);
            let half = length / 2;

            for offset in 0..half {
                let even = values[start + offset];
                let odd = arithmetic.mul_mod(values[start + offset + half], twiddle);

                values[start + offset] = arithmetic.add_mod(even, odd);
                values[start + offset + half] = arithmetic.sub_mod(even, odd);

                twiddle = arithmetic.mul_mod(twiddle, root_step);
            }
        }

        length *= 2;
    }
}

fn bit_reverse_permute<T>(values: &mut [T]) {
    let degree = values.len();
    let bits = degree.trailing_zeros();

    for index in 0..degree {
        let reversed = index.reverse_bits() >> (usize::BITS - bits);

        if reversed > index {
            values.swap(index, reversed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Limb128, Limb32, Limb64, LimbNttPlan, LimbPolynomial, NttLimbArithmetic,
        PhysicalLimbArithmetic,
    };

    fn deterministic_u128(degree: usize, modulus: u128, offset: u128) -> Vec<u128> {
        (0..degree)
            .map(|index| {
                let i = index as u128;
                (offset + 17 * i + 5 * i * i + 3 * i * i * i) % modulus
            })
            .collect()
    }

    fn polynomial32(modulus: u32, degree: usize, offset: u128) -> LimbPolynomial<Limb32> {
        LimbPolynomial::new(
            Limb32::new(modulus),
            deterministic_u128(degree, u128::from(modulus), offset)
                .into_iter()
                .map(|value| value as u32)
                .collect(),
        )
    }

    fn polynomial64(modulus: u64, degree: usize, offset: u128) -> LimbPolynomial<Limb64> {
        LimbPolynomial::new(
            Limb64::new(modulus),
            deterministic_u128(degree, u128::from(modulus), offset)
                .into_iter()
                .map(|value| value as u64)
                .collect(),
        )
    }

    fn polynomial128(modulus: u128, degree: usize, offset: u128) -> LimbPolynomial<Limb128> {
        LimbPolynomial::new(
            Limb128::new(modulus),
            deterministic_u128(degree, modulus, offset),
        )
    }

    #[test]
    fn shared_ntt_modulus_roundtrips_across_all_word_widths() {
        let modulus = 65_537_u128;
        let degree = 64;

        let p32 = polynomial32(modulus as u32, degree, 5);
        let p64 = polynomial64(modulus as u64, degree, 5);
        let p128 = polynomial128(modulus, degree, 5);

        let plan32 = LimbNttPlan::new(Limb32::new(modulus as u32), degree);
        let plan64 = LimbNttPlan::new(Limb64::new(modulus as u64), degree);
        let plan128 = LimbNttPlan::new(Limb128::new(modulus), degree);

        assert_eq!(plan32.inverse(&plan32.forward(&p32)), p32);
        assert_eq!(plan64.inverse(&plan64.forward(&p64)), p64);
        assert_eq!(plan128.inverse(&plan128.forward(&p128)), p128);
    }

    #[test]
    fn shared_modulus_ntt_product_is_exact_across_all_word_widths() {
        let modulus = 65_537_u128;
        let degree = 64;

        let lhs32 = polynomial32(modulus as u32, degree, 7);
        let rhs32 = polynomial32(modulus as u32, degree, 29);
        let lhs64 = polynomial64(modulus as u64, degree, 7);
        let rhs64 = polynomial64(modulus as u64, degree, 29);
        let lhs128 = polynomial128(modulus, degree, 7);
        let rhs128 = polynomial128(modulus, degree, 29);

        let plan32 = LimbNttPlan::new(Limb32::new(modulus as u32), degree);
        let plan64 = LimbNttPlan::new(Limb64::new(modulus as u64), degree);
        let plan128 = LimbNttPlan::new(Limb128::new(modulus), degree);

        let product32 = plan32.negacyclic_mul(&lhs32, &rhs32);
        let product64 = plan64.negacyclic_mul(&lhs64, &rhs64);
        let product128 = plan128.negacyclic_mul(&lhs128, &rhs128);

        assert_eq!(product32, lhs32.negacyclic_mul(&rhs32));
        assert_eq!(product64, lhs64.negacyclic_mul(&rhs64));
        assert_eq!(product128, lhs128.negacyclic_mul(&rhs128));

        let canonical32: Vec<u128> = product32
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect();

        let canonical64: Vec<u128> = product64
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect();

        let canonical128 = product128.coefficients().to_vec();

        assert_eq!(canonical32, canonical64);
        assert_eq!(canonical64, canonical128);
    }

    #[test]
    fn native_32_bit_ntt_prime_matches_reference() {
        let modulus = 2_013_265_921_u32;
        let degree = 64;

        let lhs = polynomial32(modulus, degree, 11);
        let rhs = polynomial32(modulus, degree, 37);
        let plan = LimbNttPlan::new(Limb32::new(modulus), degree);

        assert_eq!(plan.negacyclic_mul(&lhs, &rhs), lhs.negacyclic_mul(&rhs));
    }

    #[test]
    fn native_64_bit_ntt_prime_matches_reference() {
        // Goldilocks prime: 2^64 - 2^32 + 1.
        let modulus = 18_446_744_069_414_584_321_u64;
        let degree = 64;

        let lhs = polynomial64(modulus, degree, 13);
        let rhs = polynomial64(modulus, degree, 41);
        let plan = LimbNttPlan::new(Limb64::new(modulus), degree);

        assert_eq!(plan.negacyclic_mul(&lhs, &rhs), lhs.negacyclic_mul(&rhs));
    }

    #[test]
    fn u128_backend_executes_ntt_semantics_exactly() {
        // This first NTT milestone validates the u128 execution backend using
        // an NTT-friendly modulus representable in all three word classes.
        // Selection and characterization of >64-bit NTT primes is a separate
        // parameter-generation milestone.
        let modulus = 18_446_744_069_414_584_321_u128;
        let degree = 64;

        let lhs = polynomial128(modulus, degree, 17);
        let rhs = polynomial128(modulus, degree, 43);
        let plan = LimbNttPlan::new(Limb128::new(modulus), degree);

        assert_eq!(plan.negacyclic_mul(&lhs, &rhs), lhs.negacyclic_mul(&rhs));
    }

    #[test]
    fn discovered_roots_have_exact_negacyclic_order() {
        let degree = 64;

        let backends = [
            (Limb128::new(65_537_u128), 65_537_u128),
            (Limb128::new(2_013_265_921_u128), 2_013_265_921_u128),
            (
                Limb128::new(18_446_744_069_414_584_321_u128),
                18_446_744_069_414_584_321_u128,
            ),
        ];

        for (backend, modulus) in backends {
            let plan = LimbNttPlan::new(backend, degree);
            let psi = plan.psi();

            assert_eq!(
                plan.arithmetic().pow_mod(psi, degree as u128),
                <Limb128 as NttLimbArithmetic>::word_from_u128(modulus - 1)
            );

            assert_eq!(
                plan.arithmetic().pow_mod(psi, (2 * degree) as u128),
                <Limb128 as NttLimbArithmetic>::word_from_u128(1)
            );
        }
    }

    #[test]
    fn degrees_16_through_256_match_reference_on_shared_modulus() {
        let modulus = 65_537_u64;

        for degree in [16_usize, 32, 64, 128, 256] {
            let lhs = polynomial64(modulus, degree, 5);
            let rhs = polynomial64(modulus, degree, 29);
            let plan = LimbNttPlan::new(Limb64::new(modulus), degree);

            assert_eq!(
                plan.negacyclic_mul(&lhs, &rhs),
                lhs.negacyclic_mul(&rhs),
                "width-parametric NTT mismatch at degree {degree}"
            );
        }
    }
}
