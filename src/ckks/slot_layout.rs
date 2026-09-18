/// Describes how logical values occupy the canonical CKKS SIMD slots.
///
/// The canonical embedding provides `N / 2` physical complex slots for
/// ring degree `N`. A slot layout maps an application-visible logical
/// index to one of those physical slots.
///
/// R2.6 defines the deterministic dense layout. More specialized
/// layouts, including matrix/diagonal layouts, can build on this
/// abstraction once rotations are available.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CkksSlotLayout {
    ring_degree: usize,
    logical_slots: usize,
}

impl CkksSlotLayout {
    /// Creates a dense slot layout using the first `logical_slots`
    /// canonical CKKS slots.
    pub fn dense(ring_degree: usize, logical_slots: usize) -> Self {
        assert!(
            ring_degree >= 2 && ring_degree.is_power_of_two(),
            "CKKS slot layout requires a power-of-two ring degree >= 2"
        );

        let capacity = ring_degree / 2;

        assert!(
            logical_slots <= capacity,
            "logical CKKS slot count exceeds physical slot capacity"
        );

        Self {
            ring_degree,
            logical_slots,
        }
    }

    /// Creates a layout occupying every available canonical CKKS slot.
    pub fn fully_packed(ring_degree: usize) -> Self {
        Self::dense(ring_degree, ring_degree / 2)
    }

    pub fn ring_degree(&self) -> usize {
        self.ring_degree
    }

    pub fn capacity(&self) -> usize {
        self.ring_degree / 2
    }

    pub fn logical_slots(&self) -> usize {
        self.logical_slots
    }

    pub fn unused_slots(&self) -> usize {
        self.capacity() - self.logical_slots
    }

    pub fn utilization(&self) -> f64 {
        if self.capacity() == 0 {
            return 0.0;
        }

        self.logical_slots as f64 / self.capacity() as f64
    }

    /// Maps a logical index to its canonical physical slot.
    ///
    /// The R2.6 dense convention is identity mapping:
    ///
    /// logical j -> canonical slot j.
    pub fn physical_slot(&self, logical_index: usize) -> usize {
        assert!(
            logical_index < self.logical_slots,
            "logical CKKS slot index out of range"
        );

        logical_index
    }

    pub fn is_fully_packed(&self) -> bool {
        self.logical_slots == self.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fully_packed_layout_uses_all_slots() {
        let layout = CkksSlotLayout::fully_packed(16);

        assert_eq!(layout.ring_degree(), 16);
        assert_eq!(layout.capacity(), 8);
        assert_eq!(layout.logical_slots(), 8);
        assert_eq!(layout.unused_slots(), 0);
        assert_eq!(layout.utilization(), 1.0);
        assert!(layout.is_fully_packed());
    }

    #[test]
    fn partial_dense_layout_tracks_utilization() {
        let layout = CkksSlotLayout::dense(16, 3);

        assert_eq!(layout.capacity(), 8);
        assert_eq!(layout.logical_slots(), 3);
        assert_eq!(layout.unused_slots(), 5);
        assert_eq!(layout.utilization(), 3.0 / 8.0);
        assert!(!layout.is_fully_packed());
    }

    #[test]
    fn dense_layout_has_identity_mapping() {
        let layout = CkksSlotLayout::dense(16, 6);

        for index in 0..6 {
            assert_eq!(layout.physical_slot(index), index);
        }
    }

    #[test]
    fn zero_logical_slots_are_valid() {
        let layout = CkksSlotLayout::dense(8, 0);

        assert_eq!(layout.capacity(), 4);
        assert_eq!(layout.logical_slots(), 0);
        assert_eq!(layout.unused_slots(), 4);
        assert_eq!(layout.utilization(), 0.0);
    }

    #[test]
    #[should_panic(expected = "logical CKKS slot count exceeds physical slot capacity")]
    fn rejects_layout_larger_than_capacity() {
        let _ = CkksSlotLayout::dense(8, 5);
    }

    #[test]
    #[should_panic(expected = "CKKS slot layout requires a power-of-two ring degree >= 2")]
    fn rejects_invalid_ring_degree() {
        let _ = CkksSlotLayout::dense(12, 4);
    }

    #[test]
    #[should_panic(expected = "logical CKKS slot index out of range")]
    fn rejects_out_of_range_logical_index() {
        let layout = CkksSlotLayout::dense(8, 3);

        let _ = layout.physical_slot(3);
    }
}
