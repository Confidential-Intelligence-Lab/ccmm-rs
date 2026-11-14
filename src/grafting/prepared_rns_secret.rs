use zeroize::Zeroizing;

use crate::ring::{Modulus, PreparedRnsNttPlan, RnsNttPlan};

/// Owned NTT-domain representation of an RNS ternary secret.
///
/// The transformed coefficients are secret key material.
///
/// This type intentionally does not implement Clone or Debug.
/// Its owned coefficient buffers are zeroized on destruction.
pub struct PreparedRnsSecret {
    degree: usize,
    moduli: Vec<Modulus>,
    transforms: Vec<Zeroizing<Vec<u64>>>,
}

impl PreparedRnsSecret {
    /// Prepares a ternary secret for a specific ordered RNS basis.
    ///
    /// The caller retains ownership of the original secret.
    pub fn new(
        secret_coefficients: &[i8],
        plan: &RnsNttPlan,
        prepared_plan: &PreparedRnsNttPlan,
    ) -> Self {
        assert_eq!(
            secret_coefficients.len(),
            plan.degree(),
            "secret degree must match RNS NTT plan"
        );

        assert_eq!(
            prepared_plan.degree(),
            plan.degree(),
            "prepared NTT degree must match RNS NTT plan"
        );

        assert_eq!(
            prepared_plan.moduli(),
            plan.moduli(),
            "prepared NTT basis must match RNS NTT basis"
        );

        assert!(
            secret_coefficients
                .iter()
                .all(|&value| matches!(value, -1..=1)),
            "RNS secret coefficients must be ternary"
        );

        let transforms = plan
            .moduli()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, modulus)| {
                let q = modulus.value();

                let projection = Zeroizing::new(
                    secret_coefficients
                        .iter()
                        .map(|&value| match value {
                            -1 => q - 1,
                            0 => 0,
                            1 => 1,
                            _ => unreachable!(),
                        })
                        .collect::<Vec<u64>>(),
                );

                // The prepared transform returns a newly owned vector.
                // The coefficient-domain projection is erased on scope exit.
                prepared_plan
                    .plan(index)
                    .forward_zeroizing_owned(projection)
            })
            .collect();

        Self {
            degree: plan.degree(),
            moduli: plan.moduli().to_vec(),
            transforms,
        }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn moduli(&self) -> &[Modulus] {
        &self.moduli
    }

    /// Returns a read-only view of one transformed secret limb.
    ///
    /// The returned coefficients must be treated as secret material.
    pub fn limb(&self, index: usize) -> &[u64] {
        &self.transforms[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ring::{Modulus, Polynomial};

    #[test]
    fn prepared_secret_matches_independent_transforms() {
        let moduli = vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ];

        let plan = RnsNttPlan::new(moduli.clone(), 8);
        let prepared_plan = PreparedRnsNttPlan::new(&plan);

        let secret = [-1_i8, 0, 1, 1, -1, 0, 1, -1];

        let cached = PreparedRnsSecret::new(&secret, &plan, &prepared_plan);

        assert_eq!(cached.degree(), 8);
        assert_eq!(cached.moduli(), moduli.as_slice());

        for (index, modulus) in moduli.iter().copied().enumerate() {
            let coefficients = secret
                .iter()
                .map(|&value| match value {
                    -1 => modulus.value() - 1,
                    0 => 0,
                    1 => 1,
                    _ => unreachable!(),
                })
                .collect();

            let polynomial = Polynomial::new(modulus, coefficients);

            let expected = prepared_plan.plan(index).forward(&polynomial);

            assert_eq!(cached.limb(index), expected.as_slice());
        }

        println!("N4_E11S4B_TRANSFORM_EQUIVALENCE=PASS");
    }
}
