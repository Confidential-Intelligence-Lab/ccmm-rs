// R30_9A_SHARED_PROFILE_CONTRACT
//
// Lightweight compatibility checks for the encrypted UHD 2D FFT.
// No key generation, encryption, or expensive FFT execution.

use ccmm_rs::ckks::{research_profile_65536_for_levels, CkksCanonicalEmbedding};
use ccmm_rs::eblas::encrypted_fft2160::{
    CIPHERTEXTS_PER_BATCH, FFT135_BRANCHES, ROW_STRIDE, SLOTS_PER_CIPHERTEXT, TILE_SLOTS, UHD_COLS,
    UHD_ROWS,
};

#[test]
fn r30_9a_shared_profile_contract() {
    let profile = research_profile_65536_for_levels(13, 3);
    let degree = profile.degree();
    let chain = profile.modulus_chain();
    let embedding = CkksCanonicalEmbedding::new(degree);

    assert_eq!(degree, 65536);
    assert_eq!(embedding.slot_count(), 32768);
    assert_eq!(chain.top().moduli().len(), 16);
    assert_eq!(chain.max_level(), 15);

    // Horizontal FFT3840 consumes levels 0..7.
    // Vertical FFT2160 consumes levels 7..13.
    let horizontal_output = chain.level(7);
    let vertical_input = chain.level(7);

    assert_eq!(horizontal_output, vertical_input);
    assert_eq!(horizontal_output.moduli().len(), 9);

    assert_eq!(SLOTS_PER_CIPHERTEXT, 32768);
    assert_eq!(TILE_SLOTS, 4096);
    assert_eq!(ROW_STRIDE, 256);
    assert_eq!(FFT135_BRANCHES, 135);
    assert_eq!(CIPHERTEXTS_PER_BATCH, 2);
    assert_eq!(UHD_COLS, 3840);
    assert_eq!(UHD_ROWS, 2160);

    assert_eq!(UHD_ROWS / 16, FFT135_BRANCHES);
    assert_eq!(15 * ROW_STRIDE, UHD_COLS);
    assert_eq!(SLOTS_PER_CIPHERTEXT / TILE_SLOTS, 8);

    // Reserve two additional validated primes for hybrid-QP keys.
    let extended = research_profile_65536_for_levels(15, 3);
    let extended_chain = extended.modulus_chain();
    let extended_moduli = extended_chain.top().moduli();

    assert_eq!(extended_moduli.len(), 18);
    assert_eq!(&extended_moduli[..16], chain.top().moduli());

    for modulus in &extended_moduli[16..18] {
        assert_eq!((modulus.value() - 1) % (2 * degree as u64), 0);
        assert!(!chain.top().moduli().contains(modulus));
    }

    println!("R30_9A_DEGREE={degree}");
    println!("R30_9A_SLOTS={}", embedding.slot_count());
    println!("R30_9A_Q_LIMBS={}", chain.top().moduli().len());
    println!(
        "R30_9A_HANDOFF_Q_LIMBS={}",
        horizontal_output.moduli().len()
    );
    println!("R30_9A_HANDOFF_LEVEL=7");
    println!("R30_9A_VERTICAL_OUTPUT_LEVEL=13");
    println!("R30_9A_HORIZONTAL_BATCHES=135");
    println!("R30_9A_VERTICAL_PACKING_GROUPS=2");
    println!("R30_9A_STATUS=PASS");
}
