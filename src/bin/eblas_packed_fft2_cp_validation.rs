use ccmm_rs::ckks::{
    rotation_exponent_left, rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState,
    RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsCkksLevelKeys, RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{execute_packed_fft2_dif_cp, fft2_pp, Fft2Shape, FftDirection};
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
const TOLERANCE: f64 = 5.0e-3;

const VALIDATION_MODULI: [u64; 4] = [0x3fff4001, 0x3ffee001, 0x3ffea001, 0x3ffe8001];

fn validation_chain() -> ModulusChain {
    ModulusChain::from_top_basis(ModulusBasis::new(
        VALIDATION_MODULI.into_iter().map(Modulus::new).collect(),
    ))
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

fn bit_reverse_index(index: usize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }

    let bits = length.trailing_zeros();
    index.reverse_bits() >> (usize::BITS - bits)
}

fn error_metrics(
    actual_physical: &[Complex64],
    expected_logical: &[Complex64],
    shape: Fft2Shape,
) -> (f64, f64) {
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for row in 0..shape.rows() {
        for col in 0..shape.cols() {
            let physical_row = bit_reverse_index(row, shape.rows());
            let physical_col = bit_reverse_index(col, shape.cols());

            let physical_index = physical_row * shape.cols() + physical_col;

            let logical_index = row * shape.cols() + col;

            let error = actual_physical[physical_index] - expected_logical[logical_index];

            squared_error += error.norm_sqr();
            squared_reference += expected_logical[logical_index].norm_sqr();
            max_abs = max_abs.max(error.norm());
        }
    }

    let rel_l2 = if squared_reference > 0.0 {
        (squared_error / squared_reference).sqrt()
    } else {
        squared_error.sqrt()
    };

    (rel_l2, max_abs)
}

fn main() {
    let shape = Fft2Shape::new(2, 2);
    let chain = validation_chain();
    let degree = DEGREE;
    let scale = INITIAL_SCALE;

    let basis = chain.top().clone();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slot_count = embedding.slot_count();
    let top_plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

    let logical_input: Vec<Complex64> = (0..shape.elements())
        .map(|index| {
            let re = ((index * 7 + 3) % 17) as f64 - 8.0;
            let im = ((index * 11 + 5) % 19) as f64 - 9.0;
            Complex64::new(re / 256.0, im / 256.0)
        })
        .collect();

    let mut physical_input = vec![Complex64::new(0.0, 0.0); slot_count];

    physical_input[..shape.elements()].copy_from_slice(&logical_input);

    let plaintext = encode_slots_rns(&physical_input, &embedding, &basis, scale);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4654_5432_4350_0001);

    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x4654_5432_4350_1001);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &top_plan,
        &mut encryption_rng,
    );

    let ciphertext = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

    /*
     * Complete 2x2 FFT2:
     *
     * level 0: row rotation 1
     * level 1: column rotation 2
     *
     * inverse then normalizes at level 2 without rotations.
     */
    let level_rotations = [(0usize, 1usize), (1usize, 2usize)];

    let mut evaluation_keys = RnsCkksEvaluationKeys::new();

    for (level, rotation) in level_rotations {
        let level_basis = chain.level(level).clone();

        let mut level_keys = RnsCkksLevelKeys::new(level, level_basis.clone());

        for (tag, exponent) in [
            (0_u64, rotation_exponent_left(degree, rotation)),
            (1_u64, rotation_exponent_right(degree, rotation)),
        ] {
            let mut rng =
                ChaCha20Rng::seed_from_u64(0x4654_5432_4350_2001 ^ ((level as u64) << 8) ^ tag);

            let key = RnsGaloisKey::generate_with_rng(
                degree,
                2,
                0,
                &secret,
                exponent,
                RnsGadgetLayout::new(level_basis.clone(), block_sizes(level_basis.len())),
                &mut rng,
            );

            level_keys.insert_galois_key(key);
        }

        evaluation_keys.insert_level(level_keys);
    }

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    println!("PACKED_FFT2_CP_PROFILE=functional-4096-4x30");
    println!("PACKED_FFT2_CP_SECURITY_BEARING=false");
    println!("PACKED_FFT2_CP_ROWS={}", shape.rows());
    println!("PACKED_FFT2_CP_COLS={}", shape.cols());
    println!("PACKED_FFT2_CP_ELEMENTS={}", shape.elements());
    println!("PACKED_FFT2_CP_SLOT_COUNT={slot_count}");
    println!("PACKED_FFT2_CP_TRANSPOSES=0");
    println!("PACKED_FFT2_CP_INPUT_LEVEL={}", ciphertext.level());

    for direction in [FftDirection::Forward, FftDirection::Inverse] {
        let expected = fft2_pp(shape, direction, &logical_input);

        let encrypted = execute_packed_fft2_dif_cp(
            shape,
            direction,
            &ciphertext,
            &evaluator,
            &embedding,
            &chain,
        );

        let actual = decrypt_slots(&encrypted, &secret, &embedding);

        let (rel_l2, max_abs) = error_metrics(&actual, &expected, shape);

        let inactive_max_abs = actual[shape.elements()..]
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
            "PACKED_FFT2_CP_CASE=DIRECTION={direction:?} \
             ROW_STAGES={} COLUMN_STAGES={} OUTPUT_LEVEL={} \
             LEVELS_CONSUMED={} TRANSPOSES=0 REL_L2={rel_l2:.12e} \
             MAX_ABS={max_abs:.12e} \
             INACTIVE_MAX_ABS={inactive_max_abs:.12e} STATUS=PASS",
            shape.row_stages(),
            shape.column_stages(),
            encrypted.level(),
            encrypted.level() - ciphertext.level(),
        );
    }

    println!("PACKED_FFT2_CP_STATUS=PASS");
}
