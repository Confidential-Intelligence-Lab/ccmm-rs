use ccmm_rs::ckks::{
    rotation_exponent_left, rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState,
    RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsCkksLevelKeys, RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{execute_packed_fft1_dif_cp, fft1_pp, Fft1Shape, FftDirection};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout,
};
use ccmm_rs::ring::{Modulus, ModulusBasis, ModulusChain, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const SIGMA: f64 = 3.19;
const DEGREE: usize = 4096;
const INITIAL_SCALE: f64 = (1_u64 << 30) as f64;
const N: usize = 4;
const TOLERANCE: f64 = 5.0e-3;

const VALIDATION_MODULI: [u64; 4] = [0x3fff4001, 0x3ffee001, 0x3ffea001, 0x3ffe8001];

fn validation_chain() -> ModulusChain {
    let basis = ModulusBasis::new(VALIDATION_MODULI.into_iter().map(Modulus::new).collect());

    ModulusChain::from_top_basis(basis)
}

fn block_sizes(limb_count: usize) -> Vec<usize> {
    vec![1; limb_count]
}

fn encode_slots_rns(
    slots: &[Complex64],
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    assert_eq!(slots.len(), embedding.slot_count());

    let coefficients = embedding.slots_to_coefficients(slots);

    let residues = basis
        .moduli()
        .iter()
        .copied()
        .map(|modulus| {
            let q = i128::from(modulus.value());

            Polynomial::new(
                modulus,
                coefficients
                    .iter()
                    .map(|&coefficient| {
                        ((coefficient * scale).round() as i128).rem_euclid(q) as u64
                    })
                    .collect(),
            )
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

fn centered(value: u128, modulus: u128) -> i128 {
    if value > modulus / 2 {
        value as i128 - modulus as i128
    } else {
        value as i128
    }
}

fn decrypt_slots(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let plan = RnsNttPlan::new(
        ciphertext.basis().moduli().to_vec(),
        ciphertext.rlwe().degree(),
    );

    let plaintext = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);
    let modulus = plaintext.composite_modulus();

    let coefficients: Vec<f64> = plaintext
        .reconstruct_coefficients()
        .into_iter()
        .map(|value| centered(value, modulus) as f64 / ciphertext.scale())
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

fn bit_reversed_physical(physical: &[Complex64], logical_index: usize, n: usize) -> Complex64 {
    if n <= 2 {
        return physical[logical_index];
    }

    let bits = n.trailing_zeros();
    let physical_index = logical_index.reverse_bits() >> (usize::BITS - bits);

    physical[physical_index]
}

fn error_metrics(actual_physical: &[Complex64], expected_logical: &[Complex64]) -> (f64, f64) {
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (logical_index, &expected) in expected_logical.iter().enumerate() {
        let actual = bit_reversed_physical(actual_physical, logical_index, N);
        let error = actual - expected;

        squared_error += error.norm_sqr();
        squared_reference += expected.norm_sqr();
        max_abs = max_abs.max(error.norm());
    }

    let rel_l2 = if squared_reference > 0.0 {
        (squared_error / squared_reference).sqrt()
    } else {
        squared_error.sqrt()
    };

    (rel_l2, max_abs)
}

fn main() {
    let chain = validation_chain();
    let degree = DEGREE;
    let scale = INITIAL_SCALE;
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slot_count = embedding.slot_count();
    let shape = Fft1Shape::new(N);

    let logical_input: Vec<Complex64> = (0..N)
        .map(|index| {
            let re = ((index * 7 + 3) % 17) as f64 - 8.0;
            let im = ((index * 11 + 5) % 19) as f64 - 9.0;
            Complex64::new(re / 256.0, im / 256.0)
        })
        .collect();

    let mut physical_input = vec![Complex64::new(0.0, 0.0); slot_count];
    physical_input[..N].copy_from_slice(&logical_input);

    let top_basis = chain.top().clone();
    let plaintext = encode_slots_rns(&physical_input, &embedding, &top_basis, scale);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5041_434b_4646_5401);

    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let top_plan = RnsNttPlan::new(top_basis.moduli().to_vec(), degree);

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x5041_434b_4646_5402);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &top_plan,
        &mut encryption_rng,
    );

    let ciphertext = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

    let mut evaluation_keys = RnsCkksEvaluationKeys::new();

    let rotations = [N / 2, N / 4];

    for (level, &rotation) in rotations.iter().enumerate() {
        let basis = chain.level(level).clone();
        let mut level_keys = RnsCkksLevelKeys::new(level, basis.clone());

        for (direction_tag, exponent) in [
            (0u64, rotation_exponent_left(degree, rotation)),
            (1u64, rotation_exponent_right(degree, rotation)),
        ] {
            let mut rng = ChaCha20Rng::seed_from_u64(
                0x5041_434b_4646_5500 ^ ((level as u64) << 8) ^ direction_tag,
            );

            let key = RnsGaloisKey::generate_with_rng(
                degree,
                2,
                0,
                &secret,
                exponent,
                RnsGadgetLayout::new(basis.clone(), block_sizes(basis.len())),
                &mut rng,
            );

            level_keys.insert_galois_key(key);
        }

        evaluation_keys.insert_level(level_keys);
    }

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    println!("PACKED_FFT1_CP_PROFILE=functional-4096-4x30");
    println!("PACKED_FFT1_CP_SECURITY_BEARING=false");
    println!("PACKED_FFT1_CP_RING_DEGREE={degree}");
    println!("PACKED_FFT1_CP_CHAIN_LEVELS={}", chain.len());
    println!("PACKED_FFT1_CP_TOP_MODULUS_BITS=120");
    println!("PACKED_FFT1_CP_N={N}");
    println!("PACKED_FFT1_CP_SLOT_COUNT={slot_count}");
    println!("PACKED_FFT1_CP_INPUT_LEVEL={}", ciphertext.level());

    for direction in [FftDirection::Forward, FftDirection::Inverse] {
        let expected = fft1_pp(shape, direction, &logical_input);

        let encrypted = execute_packed_fft1_dif_cp(
            shape,
            direction,
            &ciphertext,
            &evaluator,
            &embedding,
            &chain,
        );

        let actual = decrypt_slots(&encrypted, &secret, &embedding);

        let (rel_l2, max_abs) = error_metrics(&actual, &expected);

        let inactive_max_abs = actual[N..]
            .iter()
            .map(|value| value.norm())
            .fold(0.0_f64, f64::max);

        let stage_count = shape.stages() as usize;

        let expected_levels = match direction {
            FftDirection::Forward => stage_count,
            FftDirection::Inverse => stage_count + 1,
        };

        assert_eq!(encrypted.level(), ciphertext.level() + expected_levels);
        assert!(rel_l2 <= TOLERANCE);
        assert!(max_abs <= TOLERANCE);
        assert!(inactive_max_abs <= TOLERANCE);

        println!(
            "PACKED_FFT1_CP_CASE=DIRECTION={direction:?} STAGES={} OUTPUT_LEVEL={} LEVELS_CONSUMED={} REL_L2={rel_l2:.12e} MAX_ABS={max_abs:.12e} INACTIVE_MAX_ABS={inactive_max_abs:.12e} STATUS=PASS",
            shape.stages(),
            encrypted.level(),
            encrypted.level() - ciphertext.level(),
        );
    }

    println!("PACKED_FFT1_CP_STATUS=PASS");
}
