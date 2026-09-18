use rand::{CryptoRng, Rng, RngCore};

use crate::grafting::RnsGadgetLayout;
use crate::ring::{ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use crate::rlwe::{
    decrypt_raw_with_ntt, encrypt_raw_with_ntt_rng, encrypt_raw_with_rng, project_error,
    sample_error_coefficients, ErrorDistribution, RlweCiphertext, RlweParameters,
    RlweQuadraticCiphertext, SecretKey,
};

/// One RLWE ciphertext per RNS modulus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsRlweCiphertext {
    basis: ModulusBasis,
    limbs: Vec<RlweCiphertext>,
}

impl RnsRlweCiphertext {
    pub fn from_limbs(limbs: Vec<RlweCiphertext>) -> Self {
        assert!(
            !limbs.is_empty(),
            "RNS RLWE ciphertext requires at least one limb"
        );

        let degree = limbs[0].b().degree();

        for limb in &limbs {
            assert_eq!(
                limb.b().degree(),
                degree,
                "RNS RLWE ciphertext limbs must have the same degree"
            );
        }

        let basis = ModulusBasis::new(limbs.iter().map(|limb| limb.b().modulus()).collect());

        Self { basis, limbs }
    }

    pub fn basis(&self) -> &ModulusBasis {
        &self.basis
    }

    pub fn limbs(&self) -> &[RlweCiphertext] {
        &self.limbs
    }

    pub fn limb(&self, index: usize) -> &RlweCiphertext {
        &self.limbs[index]
    }

    pub fn degree(&self) -> usize {
        self.limbs[0].b().degree()
    }
}

/// NTT-backed raw RNS RLWE encryption.
///
/// Each RNS residue is encrypted under the same logical ternary secret,
/// projected into that residue modulus. Polynomial multiplication uses the
/// corresponding per-limb NTT plan.
pub fn encrypt_rns_raw_with_ntt_rng<R>(
    message: &RnsPolynomial,
    plaintext_modulus: u64,
    noise_bound: i64,
    secret_coefficients: &[i8],
    plan: &RnsNttPlan,
    rng: &mut R,
) -> RnsRlweCiphertext
where
    R: RngCore + CryptoRng,
{
    assert_eq!(
        secret_coefficients.len(),
        message.degree(),
        "secret coefficient count must match RNS message degree"
    );
    assert_eq!(
        plan.degree(),
        message.degree(),
        "RNS NTT plan degree must match message degree"
    );
    assert_eq!(
        plan.moduli(),
        message.moduli(),
        "RNS NTT plan basis must match message basis"
    );
    assert!(
        secret_coefficients
            .iter()
            .all(|&value| matches!(value, -1..=1)),
        "RNS secret coefficients must be ternary"
    );

    let limbs = message
        .moduli()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, modulus)| {
            let params =
                RlweParameters::new(message.degree(), modulus, plaintext_modulus, noise_bound);

            let secret = project_secret(modulus, secret_coefficients);

            encrypt_raw_with_ntt_rng(
                params,
                &secret,
                message.residue(index),
                plan.plan(index),
                rng,
            )
        })
        .collect();

    RnsRlweCiphertext::from_limbs(limbs)
}

/// NTT-backed raw RNS RLWE encryption with an explicit logical error
/// distribution.
///
/// One integer error polynomial is sampled in ``Z[X]/(X^N + 1)`` and projected
/// into every RNS limb. This preserves the semantics of one logical RLWE
/// sample represented in CRT/RNS form.
pub fn encrypt_rns_raw_with_distribution_ntt_rng<R>(
    message: &RnsPolynomial,
    plaintext_modulus: u64,
    distribution: ErrorDistribution,
    secret_coefficients: &[i8],
    plan: &RnsNttPlan,
    rng: &mut R,
) -> RnsRlweCiphertext
where
    R: RngCore + CryptoRng,
{
    assert_eq!(
        secret_coefficients.len(),
        message.degree(),
        "secret coefficient count must match RNS message degree"
    );

    assert_eq!(
        plan.degree(),
        message.degree(),
        "RNS NTT plan degree must match message degree"
    );

    assert_eq!(
        plan.moduli(),
        message.moduli(),
        "RNS NTT plan basis must match message basis"
    );

    assert!(
        secret_coefficients
            .iter()
            .all(|&value| matches!(value, -1..=1)),
        "RNS secret coefficients must be ternary"
    );

    distribution.validate();

    let logical_error = sample_error_coefficients(message.degree(), distribution, rng);

    let limbs = message
        .moduli()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, modulus)| {
            /*
             * RlweParameters remains the correctness-oriented substrate.
             * noise_bound=0 is intentional here because the security-bearing
             * error is supplied explicitly below.
             */
            let params = RlweParameters::new(message.degree(), modulus, plaintext_modulus, 0);

            let secret = project_secret(modulus, secret_coefficients);

            let q = modulus.value();
            let a = Polynomial::new(
                modulus,
                (0..message.degree()).map(|_| rng.gen_range(0..q)).collect(),
            );

            let error = project_error(modulus, &logical_error);

            let a_times_s = plan.plan(index).negacyclic_mul(&a, secret.polynomial());

            let b = message.residue(index).add(&error).sub(&a_times_s);

            let _ = params;

            RlweCiphertext::new(b, a)
        })
        .collect();

    RnsRlweCiphertext::from_limbs(limbs)
}

/// NTT-backed raw RNS RLWE decryption.
///
/// This is semantically identical to `decrypt_rns_raw`, but evaluates `a*s`
/// using the supplied per-limb NTT plans.
pub fn decrypt_rns_raw_with_ntt(
    ciphertext: &RnsRlweCiphertext,
    secret_coefficients: &[i8],
    plan: &RnsNttPlan,
) -> RnsPolynomial {
    assert_eq!(
        secret_coefficients.len(),
        ciphertext.degree(),
        "secret coefficient count must match ciphertext degree"
    );
    assert_eq!(
        plan.degree(),
        ciphertext.degree(),
        "RNS NTT plan degree must match ciphertext degree"
    );
    assert_eq!(
        plan.moduli(),
        ciphertext.basis().moduli(),
        "RNS NTT plan basis must match ciphertext basis"
    );

    let residues = ciphertext
        .basis()
        .moduli()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, modulus)| {
            let secret = project_secret(modulus, secret_coefficients);

            /*
             * plaintext_modulus is irrelevant to raw decryption, but the
             * parameter object still requires a valid value.
             */
            let params = RlweParameters::new(ciphertext.degree(), modulus, 2, 0);

            decrypt_raw_with_ntt(params, &secret, ciphertext.limb(index), plan.plan(index))
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

#[derive(Clone)]
pub struct RnsKeygenConfig<'a> {
    pub degree: usize,
    pub plaintext_modulus: u64,
    pub noise_bound: i64,
    pub layout: RnsGadgetLayout,
    pub plan: &'a RnsNttPlan,
}

/// Generic RNS evaluation key for switching one logical ternary secret
/// to another.
///
/// For gadget block `i`, the corresponding entry encrypts
///
/// `E_i * s_source`
///
/// under `s_target`, where `E_i` is the CRT idempotent associated with
/// the block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsKeySwitchKey {
    layout: RnsGadgetLayout,
    entries: Vec<RnsRlweCiphertext>,
}

impl RnsKeySwitchKey {
    pub fn generate_with_rng<R>(
        degree: usize,
        plaintext_modulus: u64,
        noise_bound: i64,
        source_secret_coefficients: &[i8],
        target_secret_coefficients: &[i8],
        layout: RnsGadgetLayout,
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        assert_eq!(
            source_secret_coefficients.len(),
            degree,
            "source secret coefficient count must match RLWE degree"
        );

        assert_eq!(
            target_secret_coefficients.len(),
            degree,
            "target secret coefficient count must match RLWE degree"
        );

        assert!(
            source_secret_coefficients
                .iter()
                .all(|&value| matches!(value, -1..=1)),
            "RNS source secret coefficients must be ternary"
        );

        assert!(
            target_secret_coefficients
                .iter()
                .all(|&value| matches!(value, -1..=1)),
            "RNS target secret coefficients must be ternary"
        );

        let basis = layout.full_basis().clone();

        let mut entries = Vec::with_capacity(layout.block_count());

        for block_index in 0..layout.block_count() {
            let idempotent = layout.crt_idempotent(block_index);

            let mut limbs = Vec::with_capacity(basis.len());

            for &modulus in basis.moduli() {
                let params = RlweParameters::new(degree, modulus, plaintext_modulus, noise_bound);

                let source = project_secret(modulus, source_secret_coefficients);

                let target = project_secret(modulus, target_secret_coefficients);

                let factor = (idempotent % u128::from(modulus.value())) as u64;

                let message = source.polynomial().scalar_mul(factor);

                limbs.push(encrypt_raw_with_rng(params, &target, &message, rng));
            }

            entries.push(RnsRlweCiphertext::from_limbs(limbs));
        }

        Self { layout, entries }
    }

    /// Generates an RNS key-switch key using per-limb NTT-backed RLWE
    /// encryption.
    pub fn generate_with_ntt_rng<R>(
        config: RnsKeygenConfig<'_>,
        source_secret_coefficients: &[i8],
        target_secret_coefficients: &[i8],
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        assert_eq!(
            source_secret_coefficients.len(),
            config.degree,
            "source secret coefficient count must match RLWE degree"
        );
        assert_eq!(
            target_secret_coefficients.len(),
            config.degree,
            "target secret coefficient count must match RLWE degree"
        );
        assert!(
            source_secret_coefficients
                .iter()
                .all(|&value| matches!(value, -1..=1)),
            "RNS source secret coefficients must be ternary"
        );
        assert!(
            target_secret_coefficients
                .iter()
                .all(|&value| matches!(value, -1..=1)),
            "RNS target secret coefficients must be ternary"
        );

        let basis = config.layout.full_basis().clone();

        assert_eq!(
            config.plan.moduli(),
            basis.moduli(),
            "RNS NTT plan basis must match key-switch basis"
        );
        assert_eq!(
            config.plan.degree(),
            config.degree,
            "RNS NTT plan degree must match key-switch degree"
        );

        let mut entries = Vec::with_capacity(config.layout.block_count());

        for block_index in 0..config.layout.block_count() {
            let idempotent = config.layout.crt_idempotent(block_index);
            let mut limbs = Vec::with_capacity(basis.len());

            for (limb_index, &modulus) in basis.moduli().iter().enumerate() {
                let params = RlweParameters::new(
                    config.degree,
                    modulus,
                    config.plaintext_modulus,
                    config.noise_bound,
                );
                let source = project_secret(modulus, source_secret_coefficients);
                let target = project_secret(modulus, target_secret_coefficients);
                let factor = (idempotent % u128::from(modulus.value())) as u64;
                let message = source.polynomial().scalar_mul(factor);

                limbs.push(encrypt_raw_with_ntt_rng(
                    params,
                    &target,
                    &message,
                    config.plan.plan(limb_index),
                    rng,
                ));
            }

            entries.push(RnsRlweCiphertext::from_limbs(limbs));
        }

        Self {
            layout: config.layout,
            entries,
        }
    }

    /// Generates an NTT-backed RNS key-switch key with an explicit
    /// logical error distribution.
    ///
    /// Each gadget entry receives an independently sampled logical error
    /// polynomial. Within that entry, the same integer error polynomial is
    /// projected into every RNS limb.
    pub fn generate_with_distribution_ntt_rng<R>(
        config: RnsKeygenConfig<'_>,
        source_secret_coefficients: &[i8],
        target_secret_coefficients: &[i8],
        distribution: ErrorDistribution,
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        assert_eq!(
            source_secret_coefficients.len(),
            config.degree,
            "source secret coefficient count must match RLWE degree"
        );
        assert_eq!(
            target_secret_coefficients.len(),
            config.degree,
            "target secret coefficient count must match RLWE degree"
        );
        assert!(
            source_secret_coefficients
                .iter()
                .all(|&value| matches!(value, -1..=1)),
            "RNS source secret coefficients must be ternary"
        );
        assert!(
            target_secret_coefficients
                .iter()
                .all(|&value| matches!(value, -1..=1)),
            "RNS target secret coefficients must be ternary"
        );
        assert!(
            source_secret_coefficients.iter().any(|&value| value != 0),
            "RNS source secret must be nonzero"
        );
        assert!(
            target_secret_coefficients.iter().any(|&value| value != 0),
            "RNS target secret must be nonzero"
        );

        distribution.validate();

        let basis = config.layout.full_basis().clone();

        assert_eq!(
            config.plan.moduli(),
            basis.moduli(),
            "RNS NTT plan basis must match key-switch basis"
        );
        assert_eq!(
            config.plan.degree(),
            config.degree,
            "RNS NTT plan degree must match key-switch degree"
        );

        let mut entries = Vec::with_capacity(config.layout.block_count());

        for block_index in 0..config.layout.block_count() {
            let idempotent = config.layout.crt_idempotent(block_index);

            let message_residues = basis
                .moduli()
                .iter()
                .copied()
                .map(|modulus| {
                    let source = project_secret(modulus, source_secret_coefficients);
                    let factor = (idempotent % u128::from(modulus.value())) as u64;

                    source.polynomial().scalar_mul(factor)
                })
                .collect();

            let message = RnsPolynomial::from_residues(message_residues);

            entries.push(encrypt_rns_raw_with_distribution_ntt_rng(
                &message,
                config.plaintext_modulus,
                distribution,
                target_secret_coefficients,
                config.plan,
                rng,
            ));
        }

        Self {
            layout: config.layout,
            entries,
        }
    }

    pub fn layout(&self) -> &RnsGadgetLayout {
        &self.layout
    }

    pub fn entries(&self) -> &[RnsRlweCiphertext] {
        &self.entries
    }

    pub fn entry(&self, index: usize) -> &RnsRlweCiphertext {
        &self.entries[index]
    }
}

/// One evaluation-key ciphertext for every gadget block and RNS limb.
///
/// Block `i` encrypts
///
/// ```text
/// E_i * s^2
/// ```
///
/// where E_i is the CRT idempotent of gadget block i.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsMultiplicationKey {
    layout: RnsGadgetLayout,
    entries: Vec<RnsRlweCiphertext>,
}

impl RnsMultiplicationKey {
    pub fn generate_with_rng<R>(
        degree: usize,
        plaintext_modulus: u64,
        noise_bound: i64,
        secret_coefficients: &[i8],
        layout: RnsGadgetLayout,
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        assert_eq!(
            secret_coefficients.len(),
            degree,
            "secret coefficient count must match RLWE degree"
        );

        assert!(
            secret_coefficients
                .iter()
                .all(|&value| matches!(value, -1..=1)),
            "RNS secret coefficients must be ternary"
        );

        assert!(
            secret_coefficients.iter().any(|&value| value != 0),
            "RNS secret must be nonzero"
        );

        let basis = layout.full_basis().clone();

        let mut entries = Vec::with_capacity(layout.block_count());

        for block_index in 0..layout.block_count() {
            let idempotent = layout.crt_idempotent(block_index);

            let mut limbs = Vec::with_capacity(basis.len());

            for &modulus in basis.moduli() {
                let params = RlweParameters::new(degree, modulus, plaintext_modulus, noise_bound);

                let secret = project_secret(modulus, secret_coefficients);

                let secret_squared = secret.polynomial().negacyclic_mul(secret.polynomial());

                let factor = (idempotent % u128::from(modulus.value())) as u64;

                let target = secret_squared.scalar_mul(factor);

                limbs.push(encrypt_raw_with_rng(params, &secret, &target, rng));
            }

            entries.push(RnsRlweCiphertext::from_limbs(limbs));
        }

        Self { layout, entries }
    }

    /// Generates an RNS multiplication key using per-limb NTT-backed
    /// secret squaring and RLWE encryption.
    pub fn generate_with_ntt_rng<R>(
        config: RnsKeygenConfig<'_>,
        secret_coefficients: &[i8],
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        assert_eq!(
            secret_coefficients.len(),
            config.degree,
            "secret coefficient count must match RLWE degree"
        );
        assert!(
            secret_coefficients
                .iter()
                .all(|&value| matches!(value, -1..=1)),
            "RNS secret coefficients must be ternary"
        );
        assert!(
            secret_coefficients.iter().any(|&value| value != 0),
            "RNS secret must be nonzero"
        );

        let basis = config.layout.full_basis().clone();

        assert_eq!(
            config.plan.moduli(),
            basis.moduli(),
            "RNS NTT plan basis must match multiplication-key basis"
        );
        assert_eq!(
            config.plan.degree(),
            config.degree,
            "RNS NTT plan degree must match multiplication-key degree"
        );

        let mut entries = Vec::with_capacity(config.layout.block_count());

        for block_index in 0..config.layout.block_count() {
            let idempotent = config.layout.crt_idempotent(block_index);
            let mut limbs = Vec::with_capacity(basis.len());

            for (limb_index, &modulus) in basis.moduli().iter().enumerate() {
                let params = RlweParameters::new(
                    config.degree,
                    modulus,
                    config.plaintext_modulus,
                    config.noise_bound,
                );
                let secret = project_secret(modulus, secret_coefficients);
                let limb_plan = config.plan.plan(limb_index);

                let secret_squared =
                    limb_plan.negacyclic_mul(secret.polynomial(), secret.polynomial());

                let factor = (idempotent % u128::from(modulus.value())) as u64;
                let target = secret_squared.scalar_mul(factor);

                limbs.push(encrypt_raw_with_ntt_rng(
                    params, &secret, &target, limb_plan, rng,
                ));
            }

            entries.push(RnsRlweCiphertext::from_limbs(limbs));
        }

        Self {
            layout: config.layout,
            entries,
        }
    }

    /// Generates an RNS multiplication key with an explicit logical
    /// error distribution.
    ///
    /// Each evaluation-key entry samples one error polynomial over the
    /// integers and projects that same polynomial into every RNS limb.
    /// This is the security-bearing counterpart of `generate_with_ntt_rng`;
    /// the existing bounded-noise generator remains unchanged for
    /// correctness and differential testing.
    pub fn generate_with_distribution_ntt_rng<R>(
        config: RnsKeygenConfig<'_>,
        secret_coefficients: &[i8],
        distribution: ErrorDistribution,
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        assert_eq!(
            secret_coefficients.len(),
            config.degree,
            "secret coefficient count must match RLWE degree"
        );

        assert!(
            secret_coefficients
                .iter()
                .all(|&value| matches!(value, -1..=1)),
            "RNS secret coefficients must be ternary"
        );

        assert!(
            secret_coefficients.iter().any(|&value| value != 0),
            "RNS secret must be nonzero"
        );

        distribution.validate();

        let basis = config.layout.full_basis().clone();

        assert_eq!(
            config.plan.moduli(),
            basis.moduli(),
            "RNS NTT plan basis must match multiplication-key basis"
        );

        assert_eq!(
            config.plan.degree(),
            config.degree,
            "RNS NTT plan degree must match multiplication-key degree"
        );

        let mut entries = Vec::with_capacity(config.layout.block_count());

        for block_index in 0..config.layout.block_count() {
            let idempotent = config.layout.crt_idempotent(block_index);

            let target_residues = basis
                .moduli()
                .iter()
                .copied()
                .enumerate()
                .map(|(limb_index, modulus)| {
                    let secret = project_secret(modulus, secret_coefficients);
                    let limb_plan = config.plan.plan(limb_index);

                    let secret_squared =
                        limb_plan.negacyclic_mul(secret.polynomial(), secret.polynomial());

                    let factor = (idempotent % u128::from(modulus.value())) as u64;

                    secret_squared.scalar_mul(factor)
                })
                .collect();

            let target = RnsPolynomial::from_residues(target_residues);

            entries.push(encrypt_rns_raw_with_distribution_ntt_rng(
                &target,
                config.plaintext_modulus,
                distribution,
                secret_coefficients,
                config.plan,
                rng,
            ));
        }

        Self {
            layout: config.layout,
            entries,
        }
    }

    pub fn layout(&self) -> &RnsGadgetLayout {
        &self.layout
    }

    pub fn entries(&self) -> &[RnsRlweCiphertext] {
        &self.entries
    }

    pub fn entry(&self, index: usize) -> &RnsRlweCiphertext {
        &self.entries[index]
    }
}

/// Projects one logical ternary secret into one coefficient modulus.
fn project_secret(modulus: crate::ring::Modulus, coefficients: &[i8]) -> SecretKey {
    let values = coefficients
        .iter()
        .map(|&value| match value {
            -1 => modulus.value() - 1,
            0 => 0,
            1 => 1,
            _ => unreachable!("secret coefficients validated as ternary"),
        })
        .collect();

    SecretKey::from_polynomial(Polynomial::new(modulus, values))
}

/// Projects an existing ternary secret into signed {-1,0,1} form.
pub fn ternary_secret_coefficients(secret: &SecretKey) -> Vec<i8> {
    let q = secret.polynomial().modulus().value();

    secret
        .polynomial()
        .coefficients()
        .iter()
        .map(|&value| {
            if value == 0 {
                0
            } else if value == 1 {
                1
            } else if value == q - 1 {
                -1
            } else {
                panic!("source secret is not ternary");
            }
        })
        .collect()
}

/// RNS representation of a degree-2 ciphertext.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsQuadraticCiphertext {
    c0: RnsPolynomial,
    c1: RnsPolynomial,
    c2: RnsPolynomial,
}

impl RnsQuadraticCiphertext {
    pub fn from_rns_polynomials(c0: RnsPolynomial, c1: RnsPolynomial, c2: RnsPolynomial) -> Self {
        assert_eq!(
            c0.basis(),
            c1.basis(),
            "RNS quadratic c0/c1 bases must match"
        );

        assert_eq!(
            c0.basis(),
            c2.basis(),
            "RNS quadratic c0/c2 bases must match"
        );

        assert_eq!(
            c0.degree(),
            c1.degree(),
            "RNS quadratic c0/c1 degrees must match"
        );

        assert_eq!(
            c0.degree(),
            c2.degree(),
            "RNS quadratic c0/c2 degrees must match"
        );

        Self { c0, c1, c2 }
    }

    pub fn from_components(c0: RnsPolynomial, c1: RnsPolynomial, c2: RnsPolynomial) -> Self {
        assert_eq!(
            c0.basis(),
            c1.basis(),
            "quadratic RNS c0/c1 bases must match"
        );

        assert_eq!(
            c0.basis(),
            c2.basis(),
            "quadratic RNS c0/c2 bases must match"
        );

        assert_eq!(
            c0.degree(),
            c1.degree(),
            "quadratic RNS c0/c1 degrees must match"
        );

        assert_eq!(
            c0.degree(),
            c2.degree(),
            "quadratic RNS c0/c2 degrees must match"
        );

        Self { c0, c1, c2 }
    }

    pub fn from_coefficient_ciphertext(
        product: &RlweQuadraticCiphertext,
        basis: &ModulusBasis,
    ) -> Self {
        let c0 = product
            .c0()
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect::<Vec<_>>();

        let c1 = product
            .c1()
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect::<Vec<_>>();

        let c2 = product
            .c2()
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect::<Vec<_>>();

        Self {
            c0: RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &c0),
            c1: RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &c1),
            c2: RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &c2),
        }
    }

    pub fn c0(&self) -> &RnsPolynomial {
        &self.c0
    }

    pub fn c1(&self) -> &RnsPolynomial {
        &self.c1
    }

    pub fn c2(&self) -> &RnsPolynomial {
        &self.c2
    }
}

/// Multiplies two RNS RLWE ciphertexts using per-limb NTT plans.
pub fn rns_tensor_with_ntt(
    lhs: &RnsRlweCiphertext,
    rhs: &RnsRlweCiphertext,
    plan: &RnsNttPlan,
) -> RnsQuadraticCiphertext {
    assert_eq!(lhs.basis(), rhs.basis(), "RNS ciphertext bases must match");
    assert_eq!(
        lhs.degree(),
        rhs.degree(),
        "RNS ciphertext degrees must match"
    );
    assert_eq!(
        plan.moduli(),
        lhs.basis().moduli(),
        "RNS NTT plan basis must match ciphertext basis"
    );
    assert_eq!(
        plan.degree(),
        lhs.degree(),
        "RNS NTT plan degree must match ciphertext degree"
    );

    let mut c0 = Vec::with_capacity(lhs.basis().len());
    let mut c1 = Vec::with_capacity(lhs.basis().len());
    let mut c2 = Vec::with_capacity(lhs.basis().len());

    for limb_index in 0..lhs.basis().len() {
        let quadratic = crate::rlwe::tensor_with_ntt(
            lhs.limb(limb_index),
            rhs.limb(limb_index),
            plan.plan(limb_index),
        );

        c0.push(quadratic.c0().clone());
        c1.push(quadratic.c1().clone());
        c2.push(quadratic.c2().clone());
    }

    RnsQuadraticCiphertext::from_rns_polynomials(
        RnsPolynomial::from_residues(c0),
        RnsPolynomial::from_residues(c1),
        RnsPolynomial::from_residues(c2),
    )
}

/// Relinearizes a degree-2 RNS ciphertext using CRT gadget blocks.
pub fn rns_relinearize(
    product: &RnsQuadraticCiphertext,
    multiplication_key: &RnsMultiplicationKey,
) -> RnsRlweCiphertext {
    let layout = multiplication_key.layout();

    assert_eq!(
        product.c0().basis(),
        layout.full_basis(),
        "quadratic ciphertext basis must match RNS multiplication key"
    );

    assert_eq!(
        product.c1().basis(),
        layout.full_basis(),
        "quadratic ciphertext basis must match RNS multiplication key"
    );

    assert_eq!(
        product.c2().basis(),
        layout.full_basis(),
        "quadratic ciphertext basis must match RNS multiplication key"
    );

    let decomposition = layout.decompose(product.c2());

    let mut output_limbs = Vec::with_capacity(layout.full_basis().len());

    for limb_index in 0..layout.full_basis().len() {
        let modulus = layout.full_basis().modulus(limb_index);

        let mut b = product.c0().residue(limb_index).clone();

        let mut a = product.c1().residue(limb_index).clone();

        for block_index in 0..layout.block_count() {
            let digit = Polynomial::new(
                modulus,
                decomposition
                    .digit(block_index)
                    .iter()
                    .map(|&value| (value % u128::from(modulus.value())) as u64)
                    .collect(),
            );

            let evaluation_key = multiplication_key.entry(block_index).limb(limb_index);

            b = b.add(&digit.negacyclic_mul(evaluation_key.b()));

            a = a.add(&digit.negacyclic_mul(evaluation_key.a()));
        }

        output_limbs.push(RlweCiphertext::new(b, a));
    }

    RnsRlweCiphertext::from_limbs(output_limbs)
}

/// NTT-backed RNS relinearization using CRT gadget blocks.
///
/// Semantics are identical to `rns_relinearize`; gadget-digit products use
/// the corresponding per-limb negacyclic NTT plan.
pub fn rns_relinearize_with_ntt(
    product: &RnsQuadraticCiphertext,
    multiplication_key: &RnsMultiplicationKey,
    plan: &RnsNttPlan,
) -> RnsRlweCiphertext {
    let layout = multiplication_key.layout();

    assert_eq!(
        product.c0().basis(),
        layout.full_basis(),
        "quadratic ciphertext basis must match RNS multiplication key"
    );
    assert_eq!(
        product.c1().basis(),
        layout.full_basis(),
        "quadratic ciphertext basis must match RNS multiplication key"
    );
    assert_eq!(
        product.c2().basis(),
        layout.full_basis(),
        "quadratic ciphertext basis must match RNS multiplication key"
    );
    assert_eq!(
        plan.moduli(),
        layout.full_basis().moduli(),
        "RNS NTT plan basis must match multiplication-key basis"
    );
    assert_eq!(
        plan.degree(),
        product.c0().degree(),
        "RNS NTT plan degree must match quadratic ciphertext degree"
    );

    let decomposition = layout.decompose(product.c2());
    let mut output_limbs = Vec::with_capacity(layout.full_basis().len());

    for limb_index in 0..layout.full_basis().len() {
        let modulus = layout.full_basis().modulus(limb_index);
        let limb_plan = plan.plan(limb_index);

        let mut b = product.c0().residue(limb_index).clone();
        let mut a = product.c1().residue(limb_index).clone();

        for block_index in 0..layout.block_count() {
            let digit = Polynomial::new(
                modulus,
                decomposition
                    .digit(block_index)
                    .iter()
                    .map(|&value| (value % u128::from(modulus.value())) as u64)
                    .collect(),
            );

            let evaluation_key = multiplication_key.entry(block_index).limb(limb_index);

            b = b.add(&limb_plan.negacyclic_mul(&digit, evaluation_key.b()));
            a = a.add(&limb_plan.negacyclic_mul(&digit, evaluation_key.a()));
        }

        output_limbs.push(RlweCiphertext::new(b, a));
    }

    RnsRlweCiphertext::from_limbs(output_limbs)
}

/// Switches an RNS RLWE ciphertext from the source secret encoded by
/// `key_switch_key` to its target secret.
///
/// The input is interpreted limbwise as
///
/// `b + a * s_source`.
///
/// The returned ciphertext decrypts under `s_target` to the same
/// RNS plaintext, up to evaluation-key noise.
pub fn rns_key_switch(
    ciphertext: &RnsRlweCiphertext,
    key_switch_key: &RnsKeySwitchKey,
) -> RnsRlweCiphertext {
    let layout = key_switch_key.layout();

    assert_eq!(
        ciphertext.basis(),
        layout.full_basis(),
        "RNS ciphertext basis must match key-switch layout"
    );

    let a = RnsPolynomial::from_residues(
        ciphertext
            .limbs()
            .iter()
            .map(|limb| limb.a().clone())
            .collect(),
    );

    let decomposition = layout.decompose(&a);

    let mut output_limbs = Vec::with_capacity(layout.full_basis().len());

    for limb_index in 0..layout.full_basis().len() {
        let modulus = layout.full_basis().modulus(limb_index);

        let mut b = ciphertext.limb(limb_index).b().clone();

        let mut a_out = Polynomial::zero(modulus, ciphertext.degree());

        for block_index in 0..layout.block_count() {
            let digit = Polynomial::new(
                modulus,
                decomposition
                    .digit(block_index)
                    .iter()
                    .map(|&value| (value % u128::from(modulus.value())) as u64)
                    .collect(),
            );

            let evaluation_key = key_switch_key.entry(block_index).limb(limb_index);

            b = b.add(&digit.negacyclic_mul(evaluation_key.b()));

            a_out = a_out.add(&digit.negacyclic_mul(evaluation_key.a()));
        }

        output_limbs.push(RlweCiphertext::new(b, a_out));
    }

    RnsRlweCiphertext::from_limbs(output_limbs)
}

/// NTT-backed generic RNS key switching.
///
/// Semantics are identical to `rns_key_switch`; gadget-digit products use
/// the corresponding per-limb negacyclic NTT plan.
pub fn rns_key_switch_with_ntt(
    ciphertext: &RnsRlweCiphertext,
    key_switch_key: &RnsKeySwitchKey,
    plan: &RnsNttPlan,
) -> RnsRlweCiphertext {
    let layout = key_switch_key.layout();

    assert_eq!(
        ciphertext.basis(),
        layout.full_basis(),
        "RNS ciphertext basis must match key-switch layout"
    );
    assert_eq!(
        plan.moduli(),
        layout.full_basis().moduli(),
        "RNS NTT plan basis must match key-switch basis"
    );
    assert_eq!(
        plan.degree(),
        ciphertext.degree(),
        "RNS NTT plan degree must match ciphertext degree"
    );

    let a = RnsPolynomial::from_residues(
        ciphertext
            .limbs()
            .iter()
            .map(|limb| limb.a().clone())
            .collect(),
    );

    let decomposition = layout.decompose(&a);
    let mut output_limbs = Vec::with_capacity(layout.full_basis().len());

    for limb_index in 0..layout.full_basis().len() {
        let modulus = layout.full_basis().modulus(limb_index);
        let limb_plan = plan.plan(limb_index);

        let mut b = ciphertext.limb(limb_index).b().clone();
        let mut a_out = Polynomial::zero(modulus, ciphertext.degree());

        for block_index in 0..layout.block_count() {
            let digit = Polynomial::new(
                modulus,
                decomposition
                    .digit(block_index)
                    .iter()
                    .map(|&value| (value % u128::from(modulus.value())) as u64)
                    .collect(),
            );

            let evaluation_key = key_switch_key.entry(block_index).limb(limb_index);

            b = b.add(&limb_plan.negacyclic_mul(&digit, evaluation_key.b()));
            a_out = a_out.add(&limb_plan.negacyclic_mul(&digit, evaluation_key.a()));
        }

        output_limbs.push(RlweCiphertext::new(b, a_out));
    }

    RnsRlweCiphertext::from_limbs(output_limbs)
}

/// Raw RNS decryption using one logical ternary secret.
pub fn decrypt_rns_raw(
    ciphertext: &RnsRlweCiphertext,
    secret_coefficients: &[i8],
) -> RnsPolynomial {
    assert_eq!(
        secret_coefficients.len(),
        ciphertext.degree(),
        "secret coefficient count must match ciphertext degree"
    );

    let residues = ciphertext
        .basis()
        .moduli()
        .iter()
        .copied()
        .zip(ciphertext.limbs())
        .map(|(modulus, limb)| {
            let secret = project_secret(modulus, secret_coefficients);

            limb.b().add(&limb.a().negacyclic_mul(secret.polynomial()))
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

/// Raw degree-2 RNS decryption.
pub fn decrypt_rns_quadratic_raw(
    ciphertext: &RnsQuadraticCiphertext,
    secret_coefficients: &[i8],
) -> RnsPolynomial {
    let residues = ciphertext
        .c0()
        .basis()
        .moduli()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, modulus)| {
            let secret = project_secret(modulus, secret_coefficients);

            let s = secret.polynomial();

            let s_squared = s.negacyclic_mul(s);

            ciphertext
                .c0()
                .residue(index)
                .add(&ciphertext.c1().residue(index).negacyclic_mul(s))
                .add(&ciphertext.c2().residue(index).negacyclic_mul(&s_squared))
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ring::{Modulus, ModulusBasis};
    use crate::rlwe::{encrypt_with_rng, tensor, RlweParameters, RlwePlaintext, SecretKey};

    use super::*;

    fn basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ])
    }

    fn source_params() -> RlweParameters {
        RlweParameters::new(8, Modulus::new(12_289), 16, 0)
    }

    #[test]
    fn shared_gaussian_rns_error_projects_identically_across_limbs() {
        let degree = 8;
        let basis = basis();
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();

        let message = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &[0_u128; 8]);

        let mut rng = ChaCha20Rng::seed_from_u64(0x29B4_0001);

        let ciphertext = encrypt_rns_raw_with_distribution_ntt_rng(
            &message,
            2,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            &secret,
            &plan,
            &mut rng,
        );

        let decrypted = decrypt_rns_raw_with_ntt(&ciphertext, &secret, &plan);

        /*
         * A single logical error was sampled before RNS projection.
         * Therefore every limb must represent the same signed integer
         * coefficient modulo its corresponding q_i.
         */
        for coefficient_index in 0..degree {
            let reference = decrypted.residue(0).coefficient(coefficient_index);

            let q0 = basis.modulus(0).value();
            let signed = if reference <= q0 / 2 {
                reference as i128
            } else {
                reference as i128 - q0 as i128
            };

            for limb_index in 1..basis.len() {
                let q = basis.modulus(limb_index).value() as i128;
                let expected = ((signed % q) + q) % q;

                assert_eq!(
                    decrypted.residue(limb_index).coefficient(coefficient_index) as i128,
                    expected,
                    "RNS Gaussian error projection mismatch at coefficient {} limb {}",
                    coefficient_index,
                    limb_index
                );
            }
        }
    }

    #[test]
    fn shared_gaussian_rns_encryption_is_seed_reproducible() {
        let degree = 8;
        let basis = basis();
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();

        let message = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &[3_u128; 8]);

        let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(0x29B4_0002);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(0x29B4_0002);

        let lhs = encrypt_rns_raw_with_distribution_ntt_rng(
            &message,
            2,
            distribution,
            &secret,
            &plan,
            &mut lhs_rng,
        );

        let rhs = encrypt_rns_raw_with_distribution_ntt_rng(
            &message,
            2,
            distribution,
            &secret,
            &plan,
            &mut rhs_rng,
        );

        assert_eq!(lhs, rhs);
    }

    #[test]
    fn ternary_secret_projects_identically_across_moduli() {
        let params = source_params();

        let mut rng = ChaCha20Rng::seed_from_u64(1);

        let secret = SecretKey::generate_with_rng(params, &mut rng);

        let ternary = ternary_secret_coefficients(&secret);

        for &modulus in basis().moduli() {
            let projected = project_secret(modulus, &ternary);

            let recovered = ternary_secret_coefficients(&projected);

            assert_eq!(recovered, ternary);
        }
    }

    #[test]
    fn evaluation_key_targets_crt_idempotent_times_s_squared() {
        let params = source_params();

        let mut secret_rng = ChaCha20Rng::seed_from_u64(10);

        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let ternary = ternary_secret_coefficients(&secret);

        let layout = RnsGadgetLayout::new(basis(), vec![1, 2]);

        let mut eval_rng = ChaCha20Rng::seed_from_u64(11);

        let key = RnsMultiplicationKey::generate_with_rng(
            params.degree(),
            params.plaintext_modulus(),
            0,
            &ternary,
            layout.clone(),
            &mut eval_rng,
        );

        for block_index in 0..layout.block_count() {
            let e = layout.crt_idempotent(block_index);

            for (limb_index, &modulus) in layout.full_basis().moduli().iter().enumerate() {
                let projected = project_secret(modulus, &ternary);

                let s_squared = projected
                    .polynomial()
                    .negacyclic_mul(projected.polynomial());

                let factor = (e % u128::from(modulus.value())) as u64;

                let expected = s_squared.scalar_mul(factor);

                let actual = key.entry(block_index).limb(limb_index).b().add(
                    &key.entry(block_index)
                        .limb(limb_index)
                        .a()
                        .negacyclic_mul(projected.polynomial()),
                );

                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn ntt_rns_raw_encryption_matches_reference_exactly() {
        use crate::ring::RnsNttPlan;

        let basis = basis();
        let degree = 8;

        let secret = [-1, 0, 1, 1, 0, -1, 1, 0];

        let message = RnsPolynomial::from_coefficients(
            basis.moduli().to_vec(),
            &[3_u128, 1, 4, 1, 5, 9, 2, 6],
        );

        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        for seed in 0_u64..32 {
            let mut reference_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xA900);

            let reference_limbs = basis
                .moduli()
                .iter()
                .copied()
                .enumerate()
                .map(|(index, modulus)| {
                    let params = RlweParameters::new(degree, modulus, 2, 1);

                    let projected_secret = project_secret(modulus, &secret);

                    encrypt_raw_with_rng(
                        params,
                        &projected_secret,
                        message.residue(index),
                        &mut reference_rng,
                    )
                })
                .collect();

            let reference = RnsRlweCiphertext::from_limbs(reference_limbs);

            let mut optimized_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xA900);

            let optimized =
                encrypt_rns_raw_with_ntt_rng(&message, 2, 1, &secret, &plan, &mut optimized_rng);

            assert_eq!(
                optimized, reference,
                "NTT RNS encryption diverged for seed {seed}"
            );
        }
    }

    #[test]
    fn ntt_rns_raw_decryption_matches_reference_exactly() {
        use crate::ring::RnsNttPlan;

        let basis = basis();
        let degree = 8;

        let secret = [-1, 0, 1, 1, 0, -1, 1, 0];

        let message = RnsPolynomial::from_coefficients(
            basis.moduli().to_vec(),
            &[2_u128, 7, 1, 8, 2, 8, 1, 8],
        );

        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        for seed in 0_u64..32 {
            let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0xAA00);

            let ciphertext = encrypt_rns_raw_with_ntt_rng(&message, 2, 1, &secret, &plan, &mut rng);

            assert_eq!(
                decrypt_rns_raw_with_ntt(&ciphertext, &secret, &plan,),
                decrypt_rns_raw(&ciphertext, &secret,),
                "NTT RNS decryption diverged for seed {seed}"
            );
        }
    }

    #[test]
    fn ntt_rns_tensor_matches_reference_exactly() {
        let basis = basis();
        let degree = 8;
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        let secret = [-1, 0, 1, 1, 0, -1, 1, 0];

        for seed in 0_u64..32 {
            let lhs_message = RnsPolynomial::from_coefficients(
                basis.moduli().to_vec(),
                &[1_u128, 2, 3, 4, 5, 6, 7, 8],
            );
            let rhs_message = RnsPolynomial::from_coefficients(
                basis.moduli().to_vec(),
                &[8_u128, 7, 6, 5, 4, 3, 2, 1],
            );

            let mut lhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xAB10);
            let mut rhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xAB20);

            let lhs =
                encrypt_rns_raw_with_ntt_rng(&lhs_message, 2, 1, &secret, &plan, &mut lhs_rng);

            let rhs =
                encrypt_rns_raw_with_ntt_rng(&rhs_message, 2, 1, &secret, &plan, &mut rhs_rng);

            let optimized = rns_tensor_with_ntt(&lhs, &rhs, &plan);

            let mut c0 = Vec::new();
            let mut c1 = Vec::new();
            let mut c2 = Vec::new();

            for limb_index in 0..basis.len() {
                let reference = tensor(lhs.limb(limb_index), rhs.limb(limb_index));

                c0.push(reference.c0().clone());
                c1.push(reference.c1().clone());
                c2.push(reference.c2().clone());
            }

            let reference = RnsQuadraticCiphertext::from_rns_polynomials(
                RnsPolynomial::from_residues(c0),
                RnsPolynomial::from_residues(c1),
                RnsPolynomial::from_residues(c2),
            );

            assert_eq!(
                optimized, reference,
                "NTT RNS tensor diverged for seed {seed}"
            );
        }
    }

    #[test]
    fn ntt_rns_relinearization_matches_reference_exactly() {
        let basis = basis();
        let degree = 8;
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        let secret = [-1, 0, 1, 1, 0, -1, 1, 0];
        let layout = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xAC00);
        let key =
            RnsMultiplicationKey::generate_with_rng(degree, 2, 0, &secret, layout, &mut key_rng);

        for seed in 0_u64..32 {
            let lhs_message = RnsPolynomial::from_coefficients(
                basis.moduli().to_vec(),
                &[1_u128, 3, 5, 7, 9, 11, 13, 15],
            );
            let rhs_message = RnsPolynomial::from_coefficients(
                basis.moduli().to_vec(),
                &[2_u128, 4, 6, 8, 10, 12, 14, 16],
            );

            let mut lhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xAC10);
            let mut rhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xAC20);

            let lhs =
                encrypt_rns_raw_with_ntt_rng(&lhs_message, 2, 0, &secret, &plan, &mut lhs_rng);

            let rhs =
                encrypt_rns_raw_with_ntt_rng(&rhs_message, 2, 0, &secret, &plan, &mut rhs_rng);

            let product = rns_tensor_with_ntt(&lhs, &rhs, &plan);

            assert_eq!(
                rns_relinearize_with_ntt(&product, &key, &plan),
                rns_relinearize(&product, &key),
                "NTT RNS relinearization diverged for seed {seed}"
            );
        }
    }

    #[test]
    fn gaussian_rns_multiplication_key_is_seed_reproducible() {
        let basis = basis();
        let degree = 8;
        let secret = [-1, 0, 1, 1, 0, -1, 1, 0];

        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        let lhs_layout = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);
        let rhs_layout = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(0x29B4_1001);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(0x29B4_1001);

        let lhs = RnsMultiplicationKey::generate_with_distribution_ntt_rng(
            RnsKeygenConfig {
                degree,
                plaintext_modulus: 2,
                noise_bound: 0,
                layout: lhs_layout,
                plan: &plan,
            },
            &secret,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            &mut lhs_rng,
        );

        let rhs = RnsMultiplicationKey::generate_with_distribution_ntt_rng(
            RnsKeygenConfig {
                degree,
                plaintext_modulus: 2,
                noise_bound: 0,
                layout: rhs_layout,
                plan: &plan,
            },
            &secret,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            &mut rhs_rng,
        );

        assert_eq!(lhs, rhs);
    }

    #[test]
    fn ntt_rns_multiplication_key_generation_matches_reference_exactly() {
        let basis = basis();
        let degree = 8;
        let secret = [-1, 0, 1, 1, 0, -1, 1, 0];

        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        for seed in 0_u64..32 {
            let layout_reference = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);

            let layout_optimized = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);

            let mut reference_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xAE10);

            let reference = RnsMultiplicationKey::generate_with_rng(
                degree,
                2,
                1,
                &secret,
                layout_reference,
                &mut reference_rng,
            );

            let mut optimized_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xAE10);

            let optimized = RnsMultiplicationKey::generate_with_ntt_rng(
                RnsKeygenConfig {
                    degree,
                    plaintext_modulus: 2,
                    noise_bound: 1,
                    layout: layout_optimized,
                    plan: &plan,
                },
                &secret,
                &mut optimized_rng,
            );

            assert_eq!(
                optimized, reference,
                "NTT multiplication-key generation diverged for seed {seed}"
            );
        }
    }

    #[test]
    fn rns_relinearization_preserves_quadratic_semantics_without_noise() {
        let params = source_params();

        let mut secret_rng = ChaCha20Rng::seed_from_u64(20);

        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let ternary = ternary_secret_coefficients(&secret);

        let lhs_plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let rhs_plaintext = RlwePlaintext::new(params, vec![8, 7, 6, 5, 4, 3, 2, 1]);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(21);

        let mut rhs_rng = ChaCha20Rng::seed_from_u64(22);

        let lhs = encrypt_with_rng(params, &secret, &lhs_plaintext, &mut lhs_rng);

        let rhs = encrypt_with_rng(params, &secret, &rhs_plaintext, &mut rhs_rng);

        let product = tensor(&lhs, &rhs);

        let layout = RnsGadgetLayout::new(basis(), vec![1, 2]);

        let rns_product =
            RnsQuadraticCiphertext::from_coefficient_ciphertext(&product, layout.full_basis());

        let mut eval_rng = ChaCha20Rng::seed_from_u64(23);

        let key = RnsMultiplicationKey::generate_with_rng(
            params.degree(),
            params.plaintext_modulus(),
            0,
            &ternary,
            layout,
            &mut eval_rng,
        );

        let relinearized = rns_relinearize(&rns_product, &key);

        assert_eq!(
            decrypt_rns_raw(&relinearized, &ternary,),
            decrypt_rns_quadratic_raw(&rns_product, &ternary,)
        );
    }

    #[test]
    fn rns_relinearization_campaign_is_exact_without_noise() {
        let params = source_params();

        for seed in 0_u64..32 {
            let mut secret_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x1111);

            let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

            let ternary = ternary_secret_coefficients(&secret);

            let lhs_plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);

            let rhs_plaintext = RlwePlaintext::new(params, vec![8, 7, 6, 5, 4, 3, 2, 1]);

            let mut lhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x2222);

            let mut rhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x3333);

            let product = tensor(
                &encrypt_with_rng(params, &secret, &lhs_plaintext, &mut lhs_rng),
                &encrypt_with_rng(params, &secret, &rhs_plaintext, &mut rhs_rng),
            );

            for block_sizes in [vec![3], vec![1, 2], vec![2, 1], vec![1, 1, 1]] {
                let layout = RnsGadgetLayout::new(basis(), block_sizes);

                let rns_product = RnsQuadraticCiphertext::from_coefficient_ciphertext(
                    &product,
                    layout.full_basis(),
                );

                let mut eval_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x4444);

                let key = RnsMultiplicationKey::generate_with_rng(
                    params.degree(),
                    params.plaintext_modulus(),
                    0,
                    &ternary,
                    layout,
                    &mut eval_rng,
                );

                let actual = decrypt_rns_raw(&rns_relinearize(&rns_product, &key), &ternary);

                let expected = decrypt_rns_quadratic_raw(&rns_product, &ternary);

                assert_eq!(
                    actual, expected,
                    "RNS relinearization mismatch for seed {seed}"
                );
            }
        }
    }

    #[test]
    fn generic_rns_key_switch_preserves_raw_decryption_without_noise() {
        let basis = basis();

        let degree = 8;

        let source_secret = [-1, 0, 1, 1, 0, -1, 1, 0];

        let target_secret = [1, 0, -1, 0, 1, 1, 0, -1];

        let message = [3_u128, 1, 4, 1, 5, 9, 2, 6];

        let mut limbs = Vec::new();

        for (limb_index, &modulus) in basis.moduli().iter().enumerate() {
            let params = RlweParameters::new(degree, modulus, 2, 0);

            let source = project_secret(modulus, &source_secret);

            let plaintext = Polynomial::new(
                modulus,
                message
                    .iter()
                    .map(|&value| (value % u128::from(modulus.value())) as u64)
                    .collect(),
            );

            let mut rng = ChaCha20Rng::seed_from_u64(0xA100 ^ limb_index as u64);

            limbs.push(encrypt_raw_with_rng(params, &source, &plaintext, &mut rng));
        }

        let ciphertext = RnsRlweCiphertext::from_limbs(limbs);

        let layout = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xA200);

        let key = RnsKeySwitchKey::generate_with_rng(
            degree,
            2,
            0,
            &source_secret,
            &target_secret,
            layout,
            &mut key_rng,
        );

        let before = decrypt_rns_raw(&ciphertext, &source_secret);

        let switched = rns_key_switch(&ciphertext, &key);

        let after = decrypt_rns_raw(&switched, &target_secret);

        assert_eq!(after, before);
    }

    #[test]
    fn ntt_rns_key_switch_matches_reference_exactly() {
        let basis = basis();
        let degree = 8;
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        let source_secret = [-1, 0, 1, 1, 0, -1, 1, 0];
        let target_secret = [1, 0, -1, 0, 1, 1, 0, -1];

        let layout = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xAD00);

        let key = RnsKeySwitchKey::generate_with_rng(
            degree,
            2,
            0,
            &source_secret,
            &target_secret,
            layout,
            &mut key_rng,
        );

        let message = RnsPolynomial::from_coefficients(
            basis.moduli().to_vec(),
            &[3_u128, 1, 4, 1, 5, 9, 2, 6],
        );

        for seed in 0_u64..32 {
            let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0xAD10);

            let ciphertext =
                encrypt_rns_raw_with_ntt_rng(&message, 2, 0, &source_secret, &plan, &mut rng);

            assert_eq!(
                rns_key_switch_with_ntt(&ciphertext, &key, &plan,),
                rns_key_switch(&ciphertext, &key,),
                "NTT RNS key switch diverged for seed {seed}"
            );
        }
    }

    #[test]
    fn gaussian_rns_key_switch_key_is_seed_reproducible() {
        let degree = 8;
        let basis = basis();
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        let source = [-1, 0, 1, 1, 0, -1, 1, 0];
        let target = [1, -1, 0, 1, -1, 0, 0, 1];

        let lhs_layout = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);
        let rhs_layout = RnsGadgetLayout::new(basis, vec![1, 2]);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(0x29B4_2001);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(0x29B4_2001);

        let lhs = RnsKeySwitchKey::generate_with_distribution_ntt_rng(
            RnsKeygenConfig {
                degree,
                plaintext_modulus: 2,
                noise_bound: 0,
                layout: lhs_layout,
                plan: &plan,
            },
            &source,
            &target,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            &mut lhs_rng,
        );

        let rhs = RnsKeySwitchKey::generate_with_distribution_ntt_rng(
            RnsKeygenConfig {
                degree,
                plaintext_modulus: 2,
                noise_bound: 0,
                layout: rhs_layout,
                plan: &plan,
            },
            &source,
            &target,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            &mut rhs_rng,
        );

        assert_eq!(lhs, rhs);
    }

    #[test]
    fn ntt_rns_key_switch_key_generation_matches_reference_exactly() {
        let basis = basis();
        let degree = 8;

        let source_secret = [-1, 0, 1, 1, 0, -1, 1, 0];
        let target_secret = [1, 0, -1, 0, 1, 1, 0, -1];

        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        for seed in 0_u64..32 {
            let layout_reference = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);

            let layout_optimized = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);

            let mut reference_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xAF10);

            let reference = RnsKeySwitchKey::generate_with_rng(
                degree,
                2,
                1,
                &source_secret,
                &target_secret,
                layout_reference,
                &mut reference_rng,
            );

            let mut optimized_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xAF10);

            let optimized = RnsKeySwitchKey::generate_with_ntt_rng(
                RnsKeygenConfig {
                    degree,
                    plaintext_modulus: 2,
                    noise_bound: 1,
                    layout: layout_optimized,
                    plan: &plan,
                },
                &source_secret,
                &target_secret,
                &mut optimized_rng,
            );

            assert_eq!(
                optimized, reference,
                "NTT key-switch-key generation diverged for seed {seed}"
            );
        }
    }

    #[test]
    fn generic_rns_key_switch_preserves_basis() {
        let basis = basis();

        let degree = 8;

        let source_secret = [-1, 0, 1, 1, 0, -1, 1, 0];

        let target_secret = [1, 0, -1, 0, 1, 1, 0, -1];

        let layout = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0xA300);

        let key = RnsKeySwitchKey::generate_with_rng(
            degree,
            2,
            0,
            &source_secret,
            &target_secret,
            layout,
            &mut key_rng,
        );

        assert_eq!(key.layout().full_basis(), &basis);

        for entry in key.entries() {
            assert_eq!(entry.basis(), &basis);
        }
    }
}

#[cfg(test)]
mod r3_evaluation_key_noise_diagnostics {
    use super::*;

    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ckks::research_profile_4096;
    use crate::rlwe::ErrorDistribution;

    fn centered_u64(value: u64, modulus: u64) -> i128 {
        if value <= modulus / 2 {
            value as i128
        } else {
            value as i128 - modulus as i128
        }
    }

    fn bit_length_u128(value: u128) -> u32 {
        if value == 0 {
            0
        } else {
            128 - value.leading_zeros()
        }
    }

    #[test]
    fn r3_1a_gaussian_evaluation_key_noise_accounting_is_exact() {
        let profile = research_profile_4096();
        let chain = profile.modulus_chain();
        let degree = profile.degree();
        let basis = chain.top().clone();
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        assert_eq!(degree, 4096);
        assert_eq!(basis.len(), 3);

        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 1,
            })
            .collect();

        let zero = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &vec![0_u128; degree]);

        /*
         * Zero ciphertext error deliberately isolates evaluation-key noise.
         * The random RLWE `a` components still produce a realistic, large c2
         * after tensoring.
         */
        let mut lhs_rng = ChaCha20Rng::seed_from_u64(0x31A0_0001);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(0x31A0_0002);

        let lhs = encrypt_rns_raw_with_ntt_rng(&zero, 2, 0, &secret, &plan, &mut lhs_rng);
        let rhs = encrypt_rns_raw_with_ntt_rng(&zero, 2, 0, &secret, &plan, &mut rhs_rng);

        let quadratic = rns_tensor_with_ntt(&lhs, &rhs, &plan);

        let layout = RnsGadgetLayout::new(basis.clone(), vec![1, 2]);
        let decomposition = layout.decompose(quadratic.c2());

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x31A0_0003);
        let key = RnsMultiplicationKey::generate_with_distribution_ntt_rng(
            RnsKeygenConfig {
                degree,
                plaintext_modulus: 2,
                noise_bound: 0,
                layout: layout.clone(),
                plan: &plan,
            },
            &secret,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            &mut key_rng,
        );

        println!("R3_1A_PROFILE={}", profile.name());
        println!("R3_1A_RING_DEGREE={degree}");
        println!("R3_1A_BLOCK_COUNT={}", layout.block_count());

        for block_index in 0..layout.block_count() {
            let block_modulus = layout.blocks()[block_index].composite_modulus();
            let max_digit = decomposition
                .digit(block_index)
                .iter()
                .copied()
                .max()
                .unwrap_or(0);

            println!(
                "R3_1A_BLOCK_{block_index}_MODULUS_BITS={}",
                bit_length_u128(block_modulus)
            );
            println!(
                "R3_1A_BLOCK_{block_index}_MAX_DIGIT_BITS={}",
                bit_length_u128(max_digit)
            );
            println!("R3_1A_BLOCK_{block_index}_MAX_DIGIT={max_digit}");
        }

        /*
         * Recover each key-entry error:
         *
         *   e_i = Dec(EK_i) - E_i*s^2
         *
         * Then predict the relinearization error:
         *
         *   e_ks = sum_i d_i * e_i.
         */
        let mut predicted_noise_residues = Vec::with_capacity(basis.len());

        for limb_index in 0..basis.len() {
            let modulus = basis.modulus(limb_index);
            let limb_plan = plan.plan(limb_index);
            let projected_secret = project_secret(modulus, &secret);
            let s = projected_secret.polynomial();
            let s_squared = limb_plan.negacyclic_mul(s, s);

            let mut predicted = Polynomial::zero(modulus, degree);

            for block_index in 0..layout.block_count() {
                let idempotent = layout.crt_idempotent(block_index);
                let factor = (idempotent % u128::from(modulus.value())) as u64;

                let expected_target = s_squared.scalar_mul(factor);
                let evaluation_key = key.entry(block_index).limb(limb_index);

                let decrypted_entry = evaluation_key
                    .b()
                    .add(&limb_plan.negacyclic_mul(evaluation_key.a(), s));

                let entry_error = decrypted_entry.sub(&expected_target);

                let digit = Polynomial::new(
                    modulus,
                    decomposition
                        .digit(block_index)
                        .iter()
                        .map(|&value| (value % u128::from(modulus.value())) as u64)
                        .collect(),
                );

                let contribution = limb_plan.negacyclic_mul(&digit, &entry_error);

                predicted = predicted.add(&contribution);

                let max_entry_error = entry_error
                    .coefficients()
                    .iter()
                    .map(|&value| centered_u64(value, modulus.value()).abs())
                    .max()
                    .unwrap_or(0);

                let max_contribution = contribution
                    .coefficients()
                    .iter()
                    .map(|&value| centered_u64(value, modulus.value()).abs())
                    .max()
                    .unwrap_or(0);

                println!(
                    "R3_1A_LIMB_{limb_index}_BLOCK_{block_index}_MAX_KEY_ERROR={max_entry_error}"
                );
                println!(
                    "R3_1A_LIMB_{limb_index}_BLOCK_{block_index}_MAX_NOISE_CONTRIBUTION={max_contribution}"
                );
            }

            let max_predicted = predicted
                .coefficients()
                .iter()
                .map(|&value| centered_u64(value, modulus.value()).abs())
                .max()
                .unwrap_or(0);

            println!("R3_1A_LIMB_{limb_index}_MAX_PREDICTED_KS_NOISE={max_predicted}");

            predicted_noise_residues.push(predicted);
        }

        let predicted_noise = RnsPolynomial::from_residues(predicted_noise_residues);

        /*
         * Actual relinearization delta:
         *
         *   Dec(relinearized) - Dec(quadratic).
         */
        let relinearized = rns_relinearize_with_ntt(&quadratic, &key, &plan);

        let actual_linear = decrypt_rns_raw_with_ntt(&relinearized, &secret, &plan);

        let expected_quadratic = decrypt_rns_quadratic_raw(&quadratic, &secret);

        let actual_delta_residues = basis
            .moduli()
            .iter()
            .copied()
            .enumerate()
            .map(|(limb_index, _)| {
                actual_linear
                    .residue(limb_index)
                    .sub(expected_quadratic.residue(limb_index))
            })
            .collect();

        let actual_delta = RnsPolynomial::from_residues(actual_delta_residues);

        assert_eq!(
            actual_delta, predicted_noise,
            "predicted sum(d_i * e_i) does not equal actual relinearization noise"
        );

        let q = predicted_noise.composite_modulus();

        let reconstructed = predicted_noise.reconstruct_coefficients();

        let max_centered = reconstructed
            .iter()
            .map(|&value| {
                let centered = if value <= q / 2 {
                    value as i128
                } else {
                    value as i128 - q as i128
                };
                centered.abs()
            })
            .max()
            .unwrap_or(0);

        println!("R3_1A_COMPOSITE_MODULUS={q}");
        println!("R3_1A_MAX_CENTERED_KS_NOISE={max_centered}");
        println!("R3_1A_NOISE_IDENTITY=PASS");
    }

    #[test]
    fn r3_1a_crt_gadget_layout_noise_amplification_sweep() {
        let profile = research_profile_4096();
        let chain = profile.modulus_chain();
        let degree = profile.degree();
        let basis = chain.top().clone();
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        let secret: Vec<i8> = (0..degree)
            .map(|index| match index % 4 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 1,
            })
            .collect();

        let zero = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &vec![0_u128; degree]);

        /*
         * Freeze the quadratic ciphertext across all layouts so the only
         * experimental variable is the gadget partition.
         */
        let mut lhs_rng = ChaCha20Rng::seed_from_u64(0x31A1_0001);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(0x31A1_0002);

        let lhs = encrypt_rns_raw_with_ntt_rng(&zero, 2, 0, &secret, &plan, &mut lhs_rng);
        let rhs = encrypt_rns_raw_with_ntt_rng(&zero, 2, 0, &secret, &plan, &mut rhs_rng);

        let quadratic = rns_tensor_with_ntt(&lhs, &rhs, &plan);
        let expected_quadratic = decrypt_rns_quadratic_raw(&quadratic, &secret);

        let layouts = [
            ("3", vec![3]),
            ("1_2", vec![1, 2]),
            ("2_1", vec![2, 1]),
            ("1_1_1", vec![1, 1, 1]),
        ];

        println!("R3_1A_SWEEP_PROFILE={}", profile.name());
        println!("R3_1A_SWEEP_RING_DEGREE={degree}");
        println!(
            "R3_1A_SWEEP_COMPOSITE_MODULUS={}",
            basis.composite_modulus()
        );

        for (layout_name, block_sizes) in layouts {
            let layout = RnsGadgetLayout::new(basis.clone(), block_sizes);

            let decomposition = layout.decompose(quadratic.c2());

            /*
             * Use the same key seed for every layout. The number of entries
             * differs, so this does not make the sampled keys identical; it
             * simply makes every layout experiment deterministic.
             */
            let mut key_rng = ChaCha20Rng::seed_from_u64(0x31A1_1000);

            let key = RnsMultiplicationKey::generate_with_distribution_ntt_rng(
                RnsKeygenConfig {
                    degree,
                    plaintext_modulus: 2,
                    noise_bound: 0,
                    layout: layout.clone(),
                    plan: &plan,
                },
                &secret,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                &mut key_rng,
            );

            println!("R3_1A_LAYOUT={layout_name}");
            println!(
                "R3_1A_LAYOUT_{layout_name}_BLOCK_COUNT={}",
                layout.block_count()
            );

            for block_index in 0..layout.block_count() {
                let block_modulus = layout.blocks()[block_index].composite_modulus();

                let max_digit = decomposition
                    .digit(block_index)
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0);

                println!(
                    "R3_1A_LAYOUT_{layout_name}_BLOCK_{block_index}_MODULUS_BITS={}",
                    bit_length_u128(block_modulus)
                );

                println!(
                    "R3_1A_LAYOUT_{layout_name}_BLOCK_{block_index}_MAX_DIGIT_BITS={}",
                    bit_length_u128(max_digit)
                );
            }

            let mut predicted_residues = Vec::with_capacity(basis.len());

            let mut global_max_key_error = 0_i128;
            let mut global_max_contribution = 0_i128;
            let mut global_max_limb_noise = 0_i128;

            for limb_index in 0..basis.len() {
                let modulus = basis.modulus(limb_index);
                let limb_plan = plan.plan(limb_index);

                let projected_secret = project_secret(modulus, &secret);
                let s = projected_secret.polynomial();

                let s_squared = limb_plan.negacyclic_mul(s, s);

                let mut predicted = Polynomial::zero(modulus, degree);

                for block_index in 0..layout.block_count() {
                    let idempotent = layout.crt_idempotent(block_index);

                    let factor = (idempotent % u128::from(modulus.value())) as u64;

                    let target = s_squared.scalar_mul(factor);

                    let evaluation_key = key.entry(block_index).limb(limb_index);

                    let decrypted_entry = evaluation_key
                        .b()
                        .add(&limb_plan.negacyclic_mul(evaluation_key.a(), s));

                    let entry_error = decrypted_entry.sub(&target);

                    let digit = Polynomial::new(
                        modulus,
                        decomposition
                            .digit(block_index)
                            .iter()
                            .map(|&value| (value % u128::from(modulus.value())) as u64)
                            .collect(),
                    );

                    let contribution = limb_plan.negacyclic_mul(&digit, &entry_error);

                    predicted = predicted.add(&contribution);

                    let max_key_error = entry_error
                        .coefficients()
                        .iter()
                        .map(|&value| centered_u64(value, modulus.value()).abs())
                        .max()
                        .unwrap_or(0);

                    let max_contribution = contribution
                        .coefficients()
                        .iter()
                        .map(|&value| centered_u64(value, modulus.value()).abs())
                        .max()
                        .unwrap_or(0);

                    global_max_key_error = global_max_key_error.max(max_key_error);

                    global_max_contribution = global_max_contribution.max(max_contribution);
                }

                let max_limb_noise = predicted
                    .coefficients()
                    .iter()
                    .map(|&value| centered_u64(value, modulus.value()).abs())
                    .max()
                    .unwrap_or(0);

                global_max_limb_noise = global_max_limb_noise.max(max_limb_noise);

                predicted_residues.push(predicted);
            }

            let predicted = RnsPolynomial::from_residues(predicted_residues);

            /*
             * Verify the accounting identity independently for every layout.
             */
            let relinearized = rns_relinearize_with_ntt(&quadratic, &key, &plan);

            let actual = decrypt_rns_raw_with_ntt(&relinearized, &secret, &plan);

            let delta_residues = basis
                .moduli()
                .iter()
                .enumerate()
                .map(|(limb_index, _)| {
                    actual
                        .residue(limb_index)
                        .sub(expected_quadratic.residue(limb_index))
                })
                .collect();

            let delta = RnsPolynomial::from_residues(delta_residues);

            assert_eq!(
                delta, predicted,
                "noise accounting mismatch for layout {layout_name}"
            );

            let q = predicted.composite_modulus();

            let max_composite_noise = predicted
                .reconstruct_coefficients()
                .iter()
                .map(|&value| {
                    if value <= q / 2 {
                        value as i128
                    } else {
                        value as i128 - q as i128
                    }
                    .abs()
                })
                .max()
                .unwrap_or(0);

            println!("R3_1A_LAYOUT_{layout_name}_MAX_KEY_ERROR={global_max_key_error}");

            println!(
                "R3_1A_LAYOUT_{layout_name}_MAX_SINGLE_CONTRIBUTION={global_max_contribution}"
            );

            println!("R3_1A_LAYOUT_{layout_name}_MAX_LIMB_NOISE={global_max_limb_noise}");

            println!("R3_1A_LAYOUT_{layout_name}_MAX_COMPOSITE_NOISE={max_composite_noise}");

            println!("R3_1A_LAYOUT_{layout_name}_IDENTITY=PASS");
        }

        println!("R3_1A_LAYOUT_SWEEP_STATUS=PASS");
    }
}
