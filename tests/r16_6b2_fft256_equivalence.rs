//! R16.6b.2-Lite: exact prepared FFT256 equivalence.
//! Reduced research parameters; no cryptographic security claim.

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next,
    rotate_left_hybrid_rns_ckks_with_prepared_ntt, rotate_right_hybrid_rns_ckks_with_prepared_ntt,
    rotation_exponent_left, rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState,
    PreparedHybridRnsGaloisKey, PreparedRnsPlaintextNtt, RnsCkksCiphertext, RnsCkksEvaluationKeys,
    RnsCkksEvaluator,
};
use ccmm_rs::grafting::{encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout};
use ccmm_rs::ring::{Modulus, ModulusBasis, ModulusChain, PreparedRnsNttPlan, RnsNttPlan};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::time::Instant;

type Diagonals = BTreeMap<isize, Vec<Complex64>>;
type RotationKeys = BTreeMap<isize, PreparedHybridRnsGaloisKey>;

fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * exponent as f64 / n as f64)
}
fn reverse_base4_4(mut i: usize) -> usize {
    let mut reversed = 0;
    for _ in 0..4 {
        reversed = reversed * 4 + i % 4;
        i /= 4;
    }
    reversed
}

// Radix-4 DIT: spans 4, 16, 64, 256. Signed offsets prevent FFT256 block crossings.
fn stage_diagonals(slots: usize, span: usize, k_cross: usize) -> Diagonals {
    assert!([4, 16, 64, 256].contains(&span));
    let quarter = span / 4;
    let mut diagonals = Diagonals::new();
    for i in 0..slots {
        let within = i % span;
        let p = within / quarter;
        let j = within % quarter;
        for q in 0..4 {
            let input_within = j + q * quarter;
            let offset = input_within as isize - within as isize;
            let mut coefficient = root(span, q * j) * root(4, p * q);
            if span == 4 {
                // The input is base-4 digit-reversed. Twiddle exponents refer
                // to natural local sample positions, not packed slot indices.
                let packed_input = i as isize + offset;
                assert!(packed_input >= 0 && (packed_input as usize) < slots);
                let natural_local = reverse_base4_4((packed_input as usize) % 256);
                coefficient *= root(3840, k_cross * natural_local);
            }
            diagonals
                .entry(offset)
                .or_insert_with(|| vec![Complex64::new(0.0, 0.0); slots])[i] = coefficient;
        }
    }
    diagonals
}

struct StageContext<'a> {
    keys: &'a RotationKeys,
    evaluator: &'a RnsCkksEvaluator<'a>,
    chain: &'a ModulusChain,
    qp_plan: &'a RnsNttPlan,
    prepared: &'a PreparedRnsNttPlan,
}

struct PreparedFft256Stage {
    level: usize,
    plain_scale: f64,
    operands: Vec<(isize, PreparedRnsPlaintextNtt)>,
}

impl PreparedFft256Stage {
    fn new(
        level: usize,
        diagonals: &Diagonals,
        embedding: &CkksCanonicalEmbedding,
        chain: &ModulusChain,
        prepared: &PreparedRnsNttPlan,
    ) -> Self {
        let basis = chain.level(level);
        let plain_scale = chain
            .dropped_modulus(level)
            .expect("rescale level available")
            .value() as f64;

        assert_eq!(basis.moduli(), prepared.moduli());

        let operands = diagonals
            .iter()
            .map(|(&offset, diagonal)| {
                let plain = encode_rns(diagonal, embedding, basis, plain_scale);

                (offset, PreparedRnsPlaintextNtt::new(&plain, prepared))
            })
            .collect();

        Self {
            level,
            plain_scale,
            operands,
        }
    }
}

fn apply_stage(
    input: &RnsCkksCiphertext,
    stage: &PreparedFft256Stage,
    ctx: &StageContext<'_>,
) -> RnsCkksCiphertext {
    assert_eq!(input.level(), stage.level);
    assert_eq!(input.basis().moduli(), ctx.prepared.moduli());
    let mut accumulator: Option<RnsCkksCiphertext> = None;
    for (offset, prepared_plain) in &stage.operands {
        let offset = *offset;
        let rotated = if offset > 0 {
            Some(rotate_left_hybrid_rns_ckks_with_prepared_ntt(
                input,
                offset as usize,
                ctx.keys.get(&offset).expect("left rotation key"),
                ctx.chain,
                ctx.qp_plan,
            ))
        } else if offset < 0 {
            Some(rotate_right_hybrid_rns_ckks_with_prepared_ntt(
                input,
                (-offset) as usize,
                ctx.keys.get(&offset).expect("right rotation key"),
                ctx.chain,
                ctx.qp_plan,
            ))
        } else {
            None
        };
        let source = rotated.as_ref().unwrap_or(input);
        let term = multiply_plain_rns_ckks_with_prepared_ntt(
            source,
            prepared_plain,
            stage.plain_scale,
            ctx.chain,
            ctx.prepared,
        );
        accumulator = Some(match accumulator {
            Some(previous) => ctx.evaluator.add(&previous, &term),
            None => term,
        });
    }
    rescale_rns_ckks_to_next(&accumulator.expect("nonempty stage"), ctx.chain)
}

// R16.6b.2: baseline hybrid FFT256 stage for exact equivalence.
fn apply_stage_baseline(
    input: &RnsCkksCiphertext,
    diagonals: &Diagonals,
    ctx: &StageContext<'_>,
    embedding: &CkksCanonicalEmbedding,
) -> RnsCkksCiphertext {
    let plain_scale = ctx
        .chain
        .dropped_modulus(input.level())
        .expect("rescale level available")
        .value() as f64;

    let mut accumulator: Option<RnsCkksCiphertext> = None;

    for (&offset, diagonal) in diagonals {
        let rotated = if offset > 0 {
            Some(rotate_left_hybrid_rns_ckks_with_prepared_ntt(
                input,
                offset as usize,
                ctx.keys.get(&offset).expect("left rotation key"),
                ctx.chain,
                ctx.qp_plan,
            ))
        } else if offset < 0 {
            Some(rotate_right_hybrid_rns_ckks_with_prepared_ntt(
                input,
                (-offset) as usize,
                ctx.keys.get(&offset).expect("right rotation key"),
                ctx.chain,
                ctx.qp_plan,
            ))
        } else {
            None
        };

        let source = rotated.as_ref().unwrap_or(input);
        let plain = encode_rns(diagonal, embedding, input.basis(), plain_scale);
        let prepared_plain = PreparedRnsPlaintextNtt::new(&plain, ctx.prepared);

        let term = multiply_plain_rns_ckks_with_prepared_ntt(
            source,
            &prepared_plain,
            plain_scale,
            ctx.chain,
            ctx.prepared,
        );

        accumulator = Some(match accumulator {
            Some(previous) => ctx.evaluator.add(&previous, &term),
            None => term,
        });
    }

    rescale_rns_ckks_to_next(&accumulator.expect("nonempty stage"), ctx.chain)
}

#[test]
fn r16_6b2_fft256_equivalence() {
    let degree = 1024usize;
    let scale = (1u64 << 25) as f64;

    let profile = ccmm_rs::ckks::research_profile_32768_for_levels(13, 3);
    let validated_q = profile.modulus_chain().top().clone();

    let q = ModulusBasis::new(validated_q.moduli()[..3].to_vec());

    let p = ModulusBasis::new(vec![
        Modulus::new(1_125_899_908_022_273),
        Modulus::new(1_125_899_908_612_097),
    ]);

    for modulus in q.moduli().iter().chain(p.moduli()) {
        assert_eq!(modulus.value() % (2 * degree as u64), 1);
    }

    let chain = ModulusChain::from_top_basis(q.clone());
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    assert_eq!(slots, 512);
    assert_eq!(slots % 256, 0);

    let plan = RnsNttPlan::new(q.moduli().to_vec(), degree);
    let prepared_plan = PreparedRnsNttPlan::new(&plan);

    let mut qp_moduli = q.moduli().to_vec();
    qp_moduli.extend_from_slice(p.moduli());
    let qp_plan = RnsNttPlan::new(qp_moduli, degree);

    let input: Vec<Complex64> = (0..slots)
        .map(|i| {
            Complex64::new(
                ((i % 17) as f64 - 8.0) / 128.0,
                ((i % 11) as f64 - 5.0) / 256.0,
            )
        })
        .collect();

    let plaintext = encode_rns(&input, &embedding, &q, scale);

    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x166b_2001);

    let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        &secret,
        &plan,
        &mut rng,
    );

    let ciphertext = RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, scale), &chain);

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let diagonals = stage_diagonals(slots, 4, 0);
    let mut keys = RotationKeys::new();

    let keygen_started = Instant::now();

    for &offset in diagonals.keys().filter(|&&offset| offset != 0) {
        let exponent = if offset > 0 {
            rotation_exponent_left(degree, offset as usize)
        } else {
            rotation_exponent_right(degree, (-offset) as usize)
        };

        let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
            BoundedGadgetLayout::new(q.clone(), 8),
            p.clone(),
            &secret,
            exponent,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            2,
            &qp_plan,
            &mut rng,
        );

        keys.insert(offset, key);
    }

    let keygen_ms = keygen_started.elapsed().as_secs_f64() * 1000.0;

    let ctx = StageContext {
        keys: &keys,
        evaluator: &evaluator,
        chain: &chain,
        qp_plan: &qp_plan,
        prepared: &prepared_plan,
    };

    let preparation_started = Instant::now();

    let prepared_stage =
        PreparedFft256Stage::new(0, &diagonals, &embedding, &chain, &prepared_plan);

    let preparation_ms = preparation_started.elapsed().as_secs_f64() * 1000.0;

    let baseline_started = Instant::now();

    let baseline = apply_stage_baseline(&ciphertext, &diagonals, &ctx, &embedding);

    let baseline_ms = baseline_started.elapsed().as_secs_f64() * 1000.0;

    let optimized_started = Instant::now();

    let optimized = apply_stage(&ciphertext, &prepared_stage, &ctx);

    let optimized_ms = optimized_started.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(baseline, optimized);
    assert_eq!(optimized.level(), 1);
    assert_eq!(optimized.basis(), chain.level(1));

    println!("R16_6B2_LITE_DEGREE={degree}");
    println!("R16_6B2_LITE_SLOTS={slots}");
    println!("R16_6B2_LITE_DIAGONALS={}", diagonals.len());
    println!("R16_6B2_LITE_KEYGEN_MS={keygen_ms:.3}");
    println!("R16_6B2_LITE_PREPARATION_MS={preparation_ms:.3}");
    println!("R16_6B2_LITE_BASELINE_MS={baseline_ms:.3}");
    println!("R16_6B2_LITE_OPTIMIZED_MS={optimized_ms:.3}");
    println!("R16_6B2_LITE_EXACT_EQUALITY=PASS");
    println!("R16_6B2_LITE_STATUS=PASS");
}
