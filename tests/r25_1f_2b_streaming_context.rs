//! R25.1F.2B: reduced 15-branch prepared FFT256 streaming equivalence.
//! Deterministic research-only fixture; no cryptographic security claim.

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    mod_switch_rns_ckks_to_next, rotation_exponent_left, rotation_exponent_right,
    CkksCanonicalEmbedding, CkksChainState, PreparedHybridRnsGaloisKey, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::prepared_radix4_fft256::{
    apply_fft256_all_branches_prepared, apply_fft256_all_branches_streaming,
    apply_fft256_all_branches_streaming_with_context, apply_fft256_multirow_prepared_with_context,
    PreparedFft256BranchBank, PreparedFft256ExecutionContext, StageContext,
};
use ccmm_rs::grafting::{encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout};
use ccmm_rs::ring::{Modulus, ModulusBasis, ModulusChain, PreparedRnsNttPlan, RnsNttPlan};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::BTreeMap;
use std::f64::consts::PI;

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

fn collect_streaming<F>(mut run: F) -> [RnsCkksCiphertext; 15]
where
    F: FnMut(&mut dyn FnMut(usize, RnsCkksCiphertext)),
{
    let mut collected: Vec<(usize, RnsCkksCiphertext)> = Vec::with_capacity(15);
    run(&mut |branch, output| collected.push((branch, output)));
    assert_eq!(collected.len(), 15, "streaming must return 15 branches");
    for (expected, (actual, _)) in collected.iter().enumerate() {
        assert_eq!(*actual, expected, "streaming branch order mismatch");
    }
    collected
        .into_iter()
        .map(|(_, ciphertext)| ciphertext)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap_or_else(|_| panic!("exactly 15 streaming outputs"))
}

fn assert_exact(
    reference: &[RnsCkksCiphertext; 15],
    actual: &[RnsCkksCiphertext; 15],
    label: &str,
) {
    for branch in 0..15 {
        assert_eq!(
            reference[branch].rlwe(),
            actual[branch].rlwe(),
            "{label}: RLWE mismatch at branch {branch}"
        );
        assert_eq!(
            reference[branch].state(),
            actual[branch].state(),
            "{label}: CKKS state mismatch at branch {branch}"
        );
        assert_eq!(
            actual[branch].level(),
            7,
            "{label}: final level mismatch at branch {branch}"
        );
    }
    println!("R25_1F_2B_{label}=PASS branches=15 level=7");
}

#[test]
fn r25_1f_2b_streaming_context_exact_equivalence() {
    let degree = 1024usize;
    let scale = (1u64 << 25) as f64;
    let profile = ccmm_rs::ckks::research_profile_32768_for_levels(13, 3);
    let validated_q = profile.modulus_chain().top().clone();
    let q = ModulusBasis::new(validated_q.moduli()[..8].to_vec());
    let p = ModulusBasis::new(vec![
        Modulus::new(1_125_899_908_022_273),
        Modulus::new(1_125_899_908_612_097),
    ]);
    for modulus in q.moduli().iter().chain(p.moduli()) {
        assert_eq!(modulus.value() % (2 * degree as u64), 1);
    }
    let chain = ModulusChain::from_top_basis(q.clone());
    assert_eq!(chain.max_level(), 7);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();
    assert_eq!(slots, 512);

    let plan0 = RnsNttPlan::new(q.moduli().to_vec(), degree);
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
    let mut rng = ChaCha20Rng::seed_from_u64(0x251f_2b01);
    let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        &secret,
        &plan0,
        &mut rng,
    );
    let mut ciphertext =
        RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, scale), &chain);
    for _ in 0..3 {
        ciphertext = mod_switch_rns_ckks_to_next(&ciphertext, &chain);
    }
    assert_eq!(ciphertext.level(), 3);

    let stage_bases: Vec<ModulusBasis> = (3..7).map(|level| chain.level(level).clone()).collect();
    let stage_plans: Vec<RnsNttPlan> = stage_bases
        .iter()
        .map(|basis| RnsNttPlan::new(basis.moduli().to_vec(), degree))
        .collect();
    let prepared_plans: Vec<PreparedRnsNttPlan> =
        stage_plans.iter().map(PreparedRnsNttPlan::new).collect();
    let qp_plans: Vec<RnsNttPlan> = stage_bases
        .iter()
        .map(|basis| {
            let mut moduli = basis.moduli().to_vec();
            moduli.extend_from_slice(p.moduli());
            RnsNttPlan::new(moduli, degree)
        })
        .collect();

    let spans = [4usize, 16, 64, 256];
    let mut stage_keys: Vec<RotationKeys> = Vec::with_capacity(4);
    for (stage, &span) in spans.iter().enumerate() {
        let diagonals = stage_diagonals(slots, span, 0);
        let mut keys = RotationKeys::new();
        for &offset in diagonals.keys().filter(|&&offset| offset != 0) {
            let exponent = if offset > 0 {
                rotation_exponent_left(degree, offset as usize)
            } else {
                rotation_exponent_right(degree, (-offset) as usize)
            };
            let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(stage_bases[stage].clone(), 8),
                p.clone(),
                &secret,
                exponent,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                2,
                &qp_plans[stage],
                &mut rng,
            );
            keys.insert(offset, key);
        }
        assert_eq!(keys.len(), 6, "six nonzero rotations per stage");
        stage_keys.push(keys);
    }
    println!("R25_1F_2B_KEYGEN=COMPLETE levels=3..6 keys=24");

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);
    let contexts: [StageContext<'_>; 4] = std::array::from_fn(|stage| StageContext {
        keys: &stage_keys[stage],
        evaluator: &evaluator,
        chain: &chain,
        qp_plan: &qp_plans[stage],
        prepared: &prepared_plans[stage],
    });
    let context_refs: [&StageContext<'_>; 4] = std::array::from_fn(|stage| &contexts[stage]);
    let bank = PreparedFft256BranchBank::new(
        |span, branch| stage_diagonals(slots, span, branch),
        &embedding,
        &chain,
        std::array::from_fn(|stage| &prepared_plans[stage]),
    );
    assert_eq!(bank.prepared_stage_count(), 18);
    let persistent = PreparedFft256ExecutionContext::new(context_refs);
    let inputs = std::array::from_fn(|_| ciphertext.clone());

    // The independent sequential API is the reference.
    let reference = apply_fft256_all_branches_prepared(inputs.clone(), &bank, context_refs);
    println!("R25_1F_2B_REFERENCE=COMPLETE");

    // Environment mutation is confined to this single integration test process.
    // Run this target with --test-threads=1; no other tests execute concurrently.
    unsafe {
        std::env::set_var("FHE_RS_WORKERS", "1");
    }
    let compatibility = collect_streaming(|consume| {
        apply_fft256_all_branches_streaming(inputs.clone(), &bank, context_refs, consume);
    });
    assert_exact(&reference, &compatibility, "COMPATIBILITY_1_WORKER");

    let persistent_one = collect_streaming(|consume| {
        apply_fft256_all_branches_streaming_with_context(
            inputs.clone(),
            &bank,
            context_refs,
            &persistent,
            consume,
        );
    });
    assert_exact(&reference, &persistent_one, "PERSISTENT_1_WORKER");

    unsafe {
        std::env::set_var("FHE_RS_WORKERS", "2");
    }
    let persistent_two = collect_streaming(|consume| {
        apply_fft256_all_branches_streaming_with_context(
            inputs.clone(),
            &bank,
            context_refs,
            &persistent,
            consume,
        );
    });
    assert_exact(&reference, &persistent_two, "PERSISTENT_2_WORKERS");

    // R25.2C.1: validate production-candidate parallelism.
    unsafe {
        std::env::set_var("FHE_RS_WORKERS", "10");
    }
    let persistent_ten = collect_streaming(|consume| {
        apply_fft256_all_branches_streaming_with_context(
            inputs.clone(),
            &bank,
            context_refs,
            &persistent,
            consume,
        );
    });
    assert_exact(&reference, &persistent_ten, "PERSISTENT_10_WORKERS");

    let repeated = collect_streaming(|consume| {
        apply_fft256_all_branches_streaming_with_context(
            inputs.clone(),
            &bank,
            context_refs,
            &persistent,
            consume,
        );
    });
    assert_exact(&persistent_two, &repeated, "REPEATED_PERSISTENT");

    // R25_3B_2_MULTIROW_EXACT_EQUIVALENCE
    //
    // Validate two independently encrypted ciphertext rows against
    // the existing sequential FFT256 reference.

    let second_encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        &secret,
        &plan0,
        &mut rng,
    );

    let mut second_ciphertext =
        RnsCkksCiphertext::new(second_encrypted, CkksChainState::top(&chain, scale), &chain);

    for _ in 0..3 {
        second_ciphertext = mod_switch_rns_ckks_to_next(&second_ciphertext, &chain);
    }

    assert_eq!(second_ciphertext.level(), 3);

    assert_ne!(
        ciphertext.rlwe(),
        second_ciphertext.rlwe(),
        "independently encrypted rows must differ"
    );

    let second_inputs = std::array::from_fn(|_| second_ciphertext.clone());

    let reference_row0 = apply_fft256_all_branches_prepared(inputs.clone(), &bank, context_refs);

    let reference_row1 =
        apply_fft256_all_branches_prepared(second_inputs.clone(), &bank, context_refs);

    for workers in [1usize, 2, 10] {
        unsafe {
            std::env::set_var("FHE_RS_WORKERS", workers.to_string());
        }

        let actual = apply_fft256_multirow_prepared_with_context(
            vec![inputs.clone(), second_inputs.clone()],
            &bank,
            context_refs,
            &persistent,
        );

        assert_eq!(actual.len(), 2);

        assert_exact(
            &reference_row0,
            &actual[0],
            &format!("R25_3B_ROW0_{workers}_WORKERS"),
        );

        assert_exact(
            &reference_row1,
            &actual[1],
            &format!("R25_3B_ROW1_{workers}_WORKERS"),
        );

        println!("R25_3B_2_GLOBAL_SCHEDULER_{workers}_WORKERS=PASS rows=2 branches=30 level=7");
    }

    println!("R25_3B_2_MULTIROW_EXACT_EQUIVALENCE=PASS");

    println!("R25_1F_2B_EXACT_STREAMING_EQUIVALENCE=PASS");
}
