use ccmm_rs::ckks::{
    rotation_exponent_left, rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState,
    RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsCkksLevelKeys, RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{
    execute_packed_fft2_dif_stage_cp, execute_packed_fft2_dif_stage_pp, Fft2Shape, FftDirection,
    PackedFft2Axis, PackedFft2DifStageDiagonals,
};
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

fn main() {
    let shape = Fft2Shape::new(2, 4);
    let chain = validation_chain();
    let basis = chain.top().clone();
    let embedding = CkksCanonicalEmbedding::new(DEGREE);
    let slot_count = embedding.slot_count();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), DEGREE);

    let logical_input: Vec<Complex64> = (0..shape.elements())
        .map(|index| {
            let re = ((index * 7 + 3) % 17) as f64 - 8.0;
            let im = ((index * 11 + 5) % 19) as f64 - 9.0;
            Complex64::new(re / 256.0, im / 256.0)
        })
        .collect();

    let mut physical_input = vec![Complex64::new(0.0, 0.0); slot_count];
    physical_input[..shape.elements()].copy_from_slice(&logical_input);

    let plaintext = encode_slots_rns(&physical_input, &embedding, &basis, INITIAL_SCALE);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4654_5432_5354_0001);

    let mut secret: Vec<i8> = (0..DEGREE)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x4654_5432_5354_1001);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &plan,
        &mut encryption_rng,
    );

    let ciphertext =
        RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, INITIAL_SCALE), &chain);

    /*
     * Shape 2x4 requires rotations:
     *
     * row span 4 -> 2
     * row span 2 -> 1
     * col span 2 -> 4
     */
    let rotations = [1usize, 2, 4];

    let mut level_keys = RnsCkksLevelKeys::new(0, basis.clone());

    for (index, rotation) in rotations.into_iter().enumerate() {
        for (tag, exponent) in [
            (0_u64, rotation_exponent_left(DEGREE, rotation)),
            (1_u64, rotation_exponent_right(DEGREE, rotation)),
        ] {
            let mut rng =
                ChaCha20Rng::seed_from_u64(0x4654_5432_5354_2001 ^ ((index as u64) << 8) ^ tag);

            let key = RnsGaloisKey::generate_with_rng(
                DEGREE,
                2,
                0,
                &secret,
                exponent,
                RnsGadgetLayout::new(basis.clone(), block_sizes(basis.len())),
                &mut rng,
            );

            level_keys.insert_galois_key(key);
        }
    }

    let mut evaluation_keys = RnsCkksEvaluationKeys::new();
    evaluation_keys.insert_level(level_keys);

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    println!("PACKED_FFT2_STAGE_PROFILE=functional-4096-4x30");
    println!("PACKED_FFT2_STAGE_SECURITY_BEARING=false");
    println!("PACKED_FFT2_STAGE_ROWS={}", shape.rows());
    println!("PACKED_FFT2_STAGE_COLS={}", shape.cols());
    println!("PACKED_FFT2_STAGE_SLOT_COUNT={slot_count}");
    println!("PACKED_FFT2_STAGE_INPUT_LEVEL={}", ciphertext.level());

    let cases = [
        (PackedFft2Axis::Rows, 4usize),
        (PackedFft2Axis::Rows, 2usize),
        (PackedFft2Axis::Columns, 2usize),
    ];

    for (axis, span) in cases {
        let diagonals = PackedFft2DifStageDiagonals::new(shape, axis, span, FftDirection::Forward);

        let expected = execute_packed_fft2_dif_stage_pp(&logical_input, &diagonals);

        let encrypted = execute_packed_fft2_dif_stage_cp(
            &ciphertext,
            &diagonals,
            &evaluator,
            &embedding,
            &chain,
            &plan,
        );

        let actual = decrypt_slots(&encrypted, &secret, &embedding);

        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for index in 0..shape.elements() {
            let error = actual[index] - expected[index];

            squared_error += error.norm_sqr();
            squared_reference += expected[index].norm_sqr();
            max_abs = max_abs.max(error.norm());
        }

        let rel_l2 = if squared_reference > 0.0 {
            (squared_error / squared_reference).sqrt()
        } else {
            squared_error.sqrt()
        };

        let inactive_max_abs = actual[shape.elements()..]
            .iter()
            .map(|value| value.norm())
            .fold(0.0_f64, f64::max);

        assert_eq!(
            encrypted.level(),
            ciphertext.level() + 1,
            "packed FFT2 DIF stage must consume exactly one level"
        );

        assert!(
            (encrypted.scale() / ciphertext.scale() - 1.0).abs() < 1.0e-12,
            "packed FFT2 DIF stage must preserve nominal scale"
        );

        assert!(rel_l2 <= TOLERANCE);
        assert!(max_abs <= TOLERANCE);
        assert!(inactive_max_abs <= TOLERANCE);

        println!(
            "PACKED_FFT2_STAGE_CASE=AXIS={axis:?} SPAN={span} ROTATION={} \
             OUTPUT_LEVEL={} LEVELS_CONSUMED={} REL_L2={rel_l2:.12e} \
             MAX_ABS={max_abs:.12e} INACTIVE_MAX_ABS={inactive_max_abs:.12e} \
             STATUS=PASS",
            diagonals.rotation(),
            encrypted.level(),
            encrypted.level() - ciphertext.level(),
        );
    }

    println!("PACKED_FFT2_STAGE_STATUS=PASS");
}
