//! R16.5.2: ciphertext-aware physical FFT planning.
//!
//! This module models indexing and communication, not encrypted execution.
//! No cryptographic security claim follows from this planner.

use super::physical_tiled_fft::PhysicalTileLayout;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CiphertextSlot {
    pub ciphertext: usize,
    pub slot: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationClass {
    SameCiphertext,
    AlignedCiphertexts,
    SlotRealignmentRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftPhase {
    HorizontalCrossTile,
    HorizontalLocal,
    VerticalCrossTile,
    VerticalLocal,
}

#[derive(Debug, Clone, Copy)]
pub struct UhdCiphertextLayout {
    pub physical: PhysicalTileLayout,
    pub slots_per_ciphertext: usize,
    pub tiles_per_ciphertext: usize,
    pub ciphertext_count: usize,
}

impl UhdCiphertextLayout {
    pub fn new(physical: PhysicalTileLayout, slots_per_ciphertext: usize) -> Self {
        let tile_elements = physical.tile_elements();

        assert!(slots_per_ciphertext >= tile_elements);
        assert_eq!(slots_per_ciphertext % tile_elements, 0);

        let tiles_per_ciphertext = slots_per_ciphertext / tile_elements;
        let ciphertext_count = physical.tile_count().div_ceil(tiles_per_ciphertext);

        Self {
            physical,
            slots_per_ciphertext,
            tiles_per_ciphertext,
            ciphertext_count,
        }
    }

    pub fn uhd_16384() -> Self {
        Self::new(PhysicalTileLayout::new(2160, 3840, 16, 256), 8192)
    }

    pub fn locate(&self, row: usize, col: usize) -> CiphertextSlot {
        let physical_index = self.physical.physical_index(row, col);

        CiphertextSlot {
            ciphertext: physical_index / self.slots_per_ciphertext,
            slot: physical_index % self.slots_per_ciphertext,
        }
    }

    pub fn classify(&self, first: CiphertextSlot, second: CiphertextSlot) -> CommunicationClass {
        if first.ciphertext == second.ciphertext {
            CommunicationClass::SameCiphertext
        } else if first.slot == second.slot {
            CommunicationClass::AlignedCiphertexts
        } else {
            CommunicationClass::SlotRealignmentRequired
        }
    }

    pub fn classify_logical(
        &self,
        first: (usize, usize),
        second: (usize, usize),
    ) -> CommunicationClass {
        self.classify(
            self.locate(first.0, first.1),
            self.locate(second.0, second.1),
        )
    }

    pub fn phase_length(&self, phase: FftPhase) -> usize {
        match phase {
            FftPhase::HorizontalCrossTile => self.physical.grid_cols,
            FftPhase::HorizontalLocal => self.physical.tile_cols,
            FftPhase::VerticalCrossTile => self.physical.grid_rows,
            FftPhase::VerticalLocal => self.physical.tile_rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r16_5_2_uhd_layout_coverage() {
        let layout = UhdCiphertextLayout::uhd_16384();

        assert_eq!(layout.physical.tile_count(), 2025);
        assert_eq!(layout.physical.tile_elements(), 4096);
        assert_eq!(layout.tiles_per_ciphertext, 2);
        assert_eq!(layout.ciphertext_count, 1013);

        let mut visited = vec![false; 2160 * 3840];

        for row in 0..2160 {
            for col in 0..3840 {
                let location = layout.locate(row, col);

                assert!(location.ciphertext < layout.ciphertext_count);
                assert!(location.slot < layout.slots_per_ciphertext);

                let index = location.ciphertext * layout.slots_per_ciphertext + location.slot;

                assert!(index < visited.len());
                assert!(!visited[index]);
                visited[index] = true;
            }
        }

        assert!(visited.iter().all(|value| *value));

        println!("R16_5_2_UHD_PIXELS=8294400");
        println!("R16_5_2_UHD_CIPHERTEXTS=1013");
        println!("R16_5_2_UHD_COVERAGE=PASS");
    }

    #[test]
    fn r16_5_2_uhd_phase_lengths() {
        let layout = UhdCiphertextLayout::uhd_16384();

        assert_eq!(layout.phase_length(FftPhase::HorizontalCrossTile), 15);
        assert_eq!(layout.phase_length(FftPhase::HorizontalLocal), 256);
        assert_eq!(layout.phase_length(FftPhase::VerticalCrossTile), 135);
        assert_eq!(layout.phase_length(FftPhase::VerticalLocal), 16);

        println!("R16_5_2_PHASES=15,256,135,16");
        println!("R16_5_2_PHASE_LENGTHS=PASS");
    }

    #[test]
    fn r16_5_2_uhd_communication_classes() {
        let layout = UhdCiphertextLayout::uhd_16384();

        // Two locations in the same physical tile.
        assert_eq!(
            layout.classify_logical((0, 0), (0, 1)),
            CommunicationClass::SameCiphertext
        );

        // Fifteen tiles per physical tile row is odd.
        // Adjacent tile rows therefore alternate ciphertext halves.
        assert_eq!(
            layout.classify_logical((0, 0), (16, 0)),
            CommunicationClass::SlotRealignmentRequired
        );

        // Two tile rows apart restores the same ciphertext slot offset.
        assert_eq!(
            layout.classify_logical((0, 0), (32, 0)),
            CommunicationClass::AlignedCiphertexts
        );

        // Corresponding local offsets in adjacent tile columns.
        // Adjacent tiles are packed into different slot halves.
        assert_eq!(
            layout.classify_logical((0, 0), (0, 256)),
            CommunicationClass::SameCiphertext
        );

        assert_eq!(
            layout.classify_logical((0, 0), (0, 512)),
            CommunicationClass::AlignedCiphertexts
        );

        println!("R16_5_2_COMMUNICATION_CLASSES=PASS");
    }
}

#[cfg(test)]
mod r16_5_3_tests {
    use super::*;

    #[derive(Default)]
    struct GroupCounts {
        same_ciphertext: u64,
        aligned_ciphertexts: u64,
        realignment_required: u64,
    }

    fn classify_group(
        layout: &UhdCiphertextLayout,
        coordinates: &[(usize, usize)],
    ) -> CommunicationClass {
        let locations: Vec<_> = coordinates
            .iter()
            .map(|&(row, col)| layout.locate(row, col))
            .collect();

        let first = locations[0];

        if locations
            .iter()
            .all(|location| location.ciphertext == first.ciphertext)
        {
            return CommunicationClass::SameCiphertext;
        }

        if locations.iter().all(|location| location.slot == first.slot) {
            return CommunicationClass::AlignedCiphertexts;
        }

        CommunicationClass::SlotRealignmentRequired
    }

    fn count_phase(
        layout: &UhdCiphertextLayout,
        phase: FftPhase,
        radices: &[usize],
    ) -> GroupCounts {
        let physical = &layout.physical;
        let length = layout.phase_length(phase);

        assert_eq!(radices.iter().product::<usize>(), length);

        let mut counts = GroupCounts::default();
        let mut stride = 1;

        for &radix in radices {
            let block = stride * radix;

            for base in (0..length).step_by(block) {
                for offset in 0..stride {
                    let indices: Vec<usize> = (0..radix)
                        .map(|digit| base + offset + digit * stride)
                        .collect();

                    let repetitions = match phase {
                        FftPhase::HorizontalCrossTile => physical.rows * physical.tile_cols,
                        FftPhase::HorizontalLocal => physical.rows * physical.grid_cols,
                        FftPhase::VerticalCrossTile => physical.cols * physical.tile_rows,
                        FftPhase::VerticalLocal => physical.cols * physical.grid_rows,
                    };

                    for repetition in 0..repetitions {
                        let coordinates: Vec<(usize, usize)> = indices
                            .iter()
                            .map(|&index| match phase {
                                FftPhase::HorizontalCrossTile => {
                                    let row = repetition / physical.tile_cols;
                                    let local_col = repetition % physical.tile_cols;
                                    (row, index * physical.tile_cols + local_col)
                                }
                                FftPhase::HorizontalLocal => {
                                    let row = repetition / physical.grid_cols;
                                    let tile_col = repetition % physical.grid_cols;
                                    (row, tile_col * physical.tile_cols + index)
                                }
                                FftPhase::VerticalCrossTile => {
                                    let col = repetition / physical.tile_rows;
                                    let local_row = repetition % physical.tile_rows;
                                    (index * physical.tile_rows + local_row, col)
                                }
                                FftPhase::VerticalLocal => {
                                    let col = repetition / physical.grid_rows;
                                    let tile_row = repetition % physical.grid_rows;
                                    (tile_row * physical.tile_rows + index, col)
                                }
                            })
                            .collect();

                        match classify_group(layout, &coordinates) {
                            CommunicationClass::SameCiphertext => counts.same_ciphertext += 1,
                            CommunicationClass::AlignedCiphertexts => {
                                counts.aligned_ciphertexts += 1
                            }
                            CommunicationClass::SlotRealignmentRequired => {
                                counts.realignment_required += 1
                            }
                        }
                    }
                }
            }

            stride = block;
        }

        counts
    }

    #[test]
    fn r16_5_3_uhd_butterfly_communication() {
        let layout = UhdCiphertextLayout::uhd_16384();

        let phases = [
            (FftPhase::HorizontalCrossTile, vec![3, 5]),
            (FftPhase::HorizontalLocal, vec![2; 8]),
            (FftPhase::VerticalCrossTile, vec![3, 3, 3, 5]),
            (FftPhase::VerticalLocal, vec![2; 4]),
        ];

        for (phase, radices) in phases {
            let counts = count_phase(&layout, phase, &radices);

            let total =
                counts.same_ciphertext + counts.aligned_ciphertexts + counts.realignment_required;

            assert!(total > 0);

            println!(
                "R16_5_3_PHASE={phase:?} \
                 same_ct={} aligned_ct={} realignment={} total={}",
                counts.same_ciphertext,
                counts.aligned_ciphertexts,
                counts.realignment_required,
                total
            );
        }

        println!("R16_5_3_UHD_COMMUNICATION_CHARACTERIZATION=PASS");
    }
}
