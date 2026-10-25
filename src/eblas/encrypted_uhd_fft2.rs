//! R31.1: ciphertext-only UHD forward FFT2 composition.
//!
//! Connects 135 encrypted horizontal FFT3840 output pairs to the
//! existing encrypted vertical FFT2160 evaluator.
//!
//! No encryption, decryption, decoding, or global transpose occurs here.

use crate::ckks::{CkksCanonicalEmbedding, RnsCkksCiphertext, RnsCkksEvaluator};
use crate::eblas::encrypted_fft2160::{execute_encrypted_fft2160_observed, FFT135_BRANCHES};
use crate::eblas::prepared_radix4_fft256::StageContext;
use crate::ring::{ModulusChain, PreparedRnsNttPlan};

pub const UHD_HORIZONTAL_BATCHES: usize = FFT135_BRANCHES;
pub const UHD_CIPHERTEXTS: usize = 2 * UHD_HORIZONTAL_BATCHES;
pub const UHD_HORIZONTAL_LEVEL: usize = 7;
pub const UHD_FINAL_LEVEL: usize = 13;

/// Regroup horizontal outputs without changing their encrypted contents.
///
/// Batch order is the vertical spatial order required by FFT135.
/// Within each batch, group 0 contains eight horizontal frequency
/// tiles; group 1 contains seven tiles and one padding tile.
pub fn regroup_horizontal_spectra(
    horizontal: Vec<[RnsCkksCiphertext; 2]>,
) -> [Vec<RnsCkksCiphertext>; 2] {
    assert_eq!(
        horizontal.len(),
        UHD_HORIZONTAL_BATCHES,
        "UHD FFT2 requires exactly 135 horizontal batches"
    );

    let mut groups = [
        Vec::with_capacity(UHD_HORIZONTAL_BATCHES),
        Vec::with_capacity(UHD_HORIZONTAL_BATCHES),
    ];

    for [left, right] in horizontal {
        assert_eq!(left.level(), UHD_HORIZONTAL_LEVEL);
        assert_eq!(right.level(), UHD_HORIZONTAL_LEVEL);

        groups[0].push(left);
        groups[1].push(right);
    }

    assert_eq!(groups[0].len(), UHD_HORIZONTAL_BATCHES);
    assert_eq!(groups[1].len(), UHD_HORIZONTAL_BATCHES);

    groups
}

/// Execute the vertical FFT on the encrypted horizontal spectra.
///
/// The supplied contexts must contain the validated hybrid-QP
/// FFT16 rotation keys at CKKS levels 11 and 12.
#[allow(clippy::too_many_arguments)]
pub fn execute_encrypted_uhd_vertical_handoff(
    horizontal: Vec<[RnsCkksCiphertext; 2]>,
    embedding: &CkksCanonicalEmbedding,
    evaluator: &RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
    prepared: [&PreparedRnsNttPlan; 2],
    contexts: [&StageContext<'_>; 2],
) -> Vec<[RnsCkksCiphertext; 2]> {
    execute_encrypted_uhd_vertical_handoff_observed(
        horizontal, embedding, evaluator, chain, prepared, contexts, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn execute_encrypted_uhd_vertical_handoff_observed(
    horizontal: Vec<[RnsCkksCiphertext; 2]>,
    embedding: &CkksCanonicalEmbedding,
    evaluator: &RnsCkksEvaluator<'_>,
    chain: &ModulusChain,
    prepared: [&PreparedRnsNttPlan; 2],
    contexts: [&StageContext<'_>; 2],
    observer: Option<&dyn Fn(&str)>,
) -> Vec<[RnsCkksCiphertext; 2]> {
    let [group0, group1] = regroup_horizontal_spectra(horizontal);

    let observe_group0 = |label: &str| {
        if let Some(callback) = observer {
            callback(&format!("GROUP0_{label}"));
        }
    };
    let observe_group1 = |label: &str| {
        if let Some(callback) = observer {
            callback(&format!("GROUP1_{label}"));
        }
    };

    let outputs0 = execute_encrypted_fft2160_observed(
        group0,
        0,
        embedding,
        evaluator,
        chain,
        prepared,
        contexts,
        observer.map(|_| &observe_group0 as &dyn Fn(&str)),
    );

    let outputs1 = execute_encrypted_fft2160_observed(
        group1,
        1,
        embedding,
        evaluator,
        chain,
        prepared,
        contexts,
        observer.map(|_| &observe_group1 as &dyn Fn(&str)),
    );

    assert_eq!(outputs0.len(), UHD_HORIZONTAL_BATCHES);
    assert_eq!(outputs1.len(), UHD_HORIZONTAL_BATCHES);

    let output: Vec<[RnsCkksCiphertext; 2]> = outputs0
        .into_iter()
        .zip(outputs1)
        .map(|(left, right)| [left, right])
        .collect();

    assert_eq!(output.len(), UHD_HORIZONTAL_BATCHES);
    assert!(
        output.iter().all(|pair| {
            pair[0].level() == UHD_FINAL_LEVEL && pair[1].level() == UHD_FINAL_LEVEL
        }),
        "UHD FFT2 outputs must be at CKKS level 13"
    );

    output
}
