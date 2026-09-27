//! Device-neutral execution characterization.
//!
//! This module defines the vocabulary used to describe encrypted computation
//! independently of the device that executes it.  It intentionally contains
//! no timing model and no CPU-specific state.

/// Execution target used for characterization.
///
/// CPU is the only implemented execution target today.  The additional
/// variants reserve vocabulary for future execution and simulation without
/// changing the meaning of recorded operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionDevice {
    Cpu,
    Gpu,
    Simulator,
}

/// Semantic encrypted-computing operation.
///
/// These events describe *what* computation occurs, not how long a particular
/// implementation takes to execute it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionEventKind {
    // eBLAS semantic operations.
    EblasDot,
    EblasGemv,
    EblasGemm,

    // Matrix-level encrypted algorithms.
    Cpmm,
    CcmmScalar,
    CcmmStructured,

    // Cryptographic operations.
    CiphertextPlaintextMultiply,
    CiphertextCiphertextMultiply,
    Add,
    Relinearize,
    Rescale,
    ModSwitch,
    Automorphism,
    Rotate,

    // Polynomial/RNS transforms.
    NttForward,
    NttInverse,
}

/// Metadata associated with one logical execution event.
///
/// Fields are optional because not every abstraction level naturally exposes
/// every piece of cryptographic state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionEvent {
    pub kind: ExecutionEventKind,
    pub ring_degree: Option<usize>,

    /// Input CKKS level when known.
    pub level: Option<usize>,

    /// Output CKKS level when the event changes level.
    pub level_after: Option<usize>,

    /// Input RNS width when known.
    pub rns_limbs: Option<usize>,

    /// Output RNS width when the event changes basis width.
    pub rns_limbs_after: Option<usize>,
}

impl ExecutionEvent {
    /// Constructs an event without representation-specific metadata.
    pub const fn new(kind: ExecutionEventKind) -> Self {
        Self {
            kind,
            ring_degree: None,
            level: None,
            level_after: None,
            rns_limbs: None,
            rns_limbs_after: None,
        }
    }

    /// Adds ring-dimension metadata.
    pub const fn with_ring_degree(mut self, ring_degree: usize) -> Self {
        self.ring_degree = Some(ring_degree);
        self
    }

    /// Adds CKKS level metadata.
    pub const fn with_level(mut self, level: usize) -> Self {
        self.level = Some(level);
        self
    }

    /// Adds active RNS-width metadata.
    pub const fn with_rns_limbs(mut self, rns_limbs: usize) -> Self {
        self.rns_limbs = Some(rns_limbs);
        self
    }

    /// Adds output CKKS-level metadata for a state transition.
    pub const fn with_level_after(mut self, level_after: usize) -> Self {
        self.level_after = Some(level_after);
        self
    }

    /// Adds output RNS-width metadata for a state transition.
    pub const fn with_rns_limbs_after(mut self, rns_limbs_after: usize) -> Self {
        self.rns_limbs_after = Some(rns_limbs_after);
        self
    }
}

/// Logical batching/packing strategy used by an encrypted representation.
///
/// `ScalarPerCiphertext` describes the current eBLAS matrix representation:
/// each logical matrix element occupies its own ciphertext/plaintext object,
/// even though the underlying CKKS ciphertext has additional SIMD capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchingStrategy {
    ScalarPerCiphertext,
    PackedSimd,
}

/// Compact representation-level characterization.
///
/// These fields describe logical encrypted objects and packing. They do not
/// represent measured allocations or memory traffic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepresentationProfile {
    pub batching: BatchingStrategy,
    pub input_ciphertexts: usize,
    pub input_plaintexts: usize,
    pub output_ciphertexts: usize,
    pub output_plaintexts: usize,
    pub slots_used_per_ciphertext: usize,
    pub slots_available: usize,
}

impl RepresentationProfile {
    /// Fraction of available CKKS slots carrying logically distinct values in
    /// one ciphertext under the selected representation.
    pub fn packing_utilization(self) -> f64 {
        if self.slots_available == 0 {
            return 0.0;
        }

        self.slots_used_per_ciphertext as f64 / self.slots_available as f64
    }
}

/// Compact aggregate characterization of logical encrypted execution.
///
/// Unlike `ExecutionTrace`, this representation is O(1) in workload size:
/// it stores counts rather than materializing one event per operation.
///
/// Timing is deliberately excluded.  A profile describes computation; a
/// measurement describes the performance of that computation on a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExecutionProfile {
    pub ciphertext_plaintext_multiplies: usize,
    pub ciphertext_ciphertext_multiplies: usize,
    pub additions: usize,
    pub relinearizations: usize,
    pub rescales: usize,
    pub modulus_switches: usize,
    pub automorphisms: usize,
    pub rotations: usize,
    pub ntt_forwards: usize,
    pub ntt_inverses: usize,
}

impl ExecutionProfile {
    /// Returns the aggregate count for an event kind represented by this
    /// profile.
    ///
    /// Semantic container events such as GEMM/CPMM/CCMM are intentionally not
    /// counted here; this profile describes the aggregate computational work.
    pub const fn count(self, kind: ExecutionEventKind) -> usize {
        match kind {
            ExecutionEventKind::CiphertextPlaintextMultiply => self.ciphertext_plaintext_multiplies,
            ExecutionEventKind::CiphertextCiphertextMultiply => {
                self.ciphertext_ciphertext_multiplies
            }
            ExecutionEventKind::Add => self.additions,
            ExecutionEventKind::Relinearize => self.relinearizations,
            ExecutionEventKind::Rescale => self.rescales,
            ExecutionEventKind::ModSwitch => self.modulus_switches,
            ExecutionEventKind::Automorphism => self.automorphisms,
            ExecutionEventKind::Rotate => self.rotations,
            ExecutionEventKind::NttForward => self.ntt_forwards,
            ExecutionEventKind::NttInverse => self.ntt_inverses,

            ExecutionEventKind::EblasDot
            | ExecutionEventKind::EblasGemv
            | ExecutionEventKind::EblasGemm
            | ExecutionEventKind::Cpmm
            | ExecutionEventKind::CcmmScalar
            | ExecutionEventKind::CcmmStructured => 0,
        }
    }

    /// Total number of represented computational events.
    pub const fn total_operations(self) -> usize {
        self.ciphertext_plaintext_multiplies
            + self.ciphertext_ciphertext_multiplies
            + self.additions
            + self.relinearizations
            + self.rescales
            + self.modulus_switches
            + self.automorphisms
            + self.rotations
            + self.ntt_forwards
            + self.ntt_inverses
    }
}

/// Ordered logical execution trace.
///
/// Timing, hardware counters, and simulator cycle information deliberately do
/// not live here.  They can later be attached by measurement/model layers
/// without changing the semantic trace format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionTrace {
    device: ExecutionDevice,
    events: Vec<ExecutionEvent>,
}

impl ExecutionTrace {
    pub const fn new(device: ExecutionDevice) -> Self {
        Self {
            device,
            events: Vec::new(),
        }
    }

    pub const fn device(&self) -> ExecutionDevice {
        self.device
    }

    pub fn record(&mut self, event: ExecutionEvent) {
        self.events.push(event);
    }

    pub fn events(&self) -> &[ExecutionEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn count(&self, kind: ExecutionEventKind) -> usize {
        self.events
            .iter()
            .filter(|event| event.kind == kind)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecutionDevice, ExecutionEvent, ExecutionEventKind, ExecutionTrace};

    #[test]
    fn execution_trace_preserves_order_and_device() {
        let mut trace = ExecutionTrace::new(ExecutionDevice::Cpu);

        trace.record(ExecutionEvent::new(ExecutionEventKind::EblasGemm));
        trace.record(
            ExecutionEvent::new(ExecutionEventKind::CcmmStructured)
                .with_ring_degree(16)
                .with_level(2)
                .with_rns_limbs(3),
        );
        trace.record(ExecutionEvent::new(ExecutionEventKind::Relinearize));
        trace.record(ExecutionEvent::new(ExecutionEventKind::Rescale));

        assert_eq!(trace.device(), ExecutionDevice::Cpu);
        assert_eq!(trace.len(), 4);
        assert_eq!(trace.events()[0].kind, ExecutionEventKind::EblasGemm);
        assert_eq!(trace.events()[1].kind, ExecutionEventKind::CcmmStructured);
        assert_eq!(trace.events()[1].ring_degree, Some(16));
        assert_eq!(trace.events()[1].level, Some(2));
        assert_eq!(trace.events()[1].rns_limbs, Some(3));
        assert_eq!(trace.events()[1].level_after, None);
        assert_eq!(trace.events()[1].rns_limbs_after, None);
    }

    #[test]
    fn execution_trace_counts_event_kinds() {
        let mut trace = ExecutionTrace::new(ExecutionDevice::Cpu);

        trace.record(ExecutionEvent::new(
            ExecutionEventKind::CiphertextCiphertextMultiply,
        ));
        trace.record(ExecutionEvent::new(
            ExecutionEventKind::CiphertextCiphertextMultiply,
        ));
        trace.record(ExecutionEvent::new(ExecutionEventKind::Relinearize));
        trace.record(ExecutionEvent::new(ExecutionEventKind::Rescale));

        assert_eq!(
            trace.count(ExecutionEventKind::CiphertextCiphertextMultiply),
            2
        );
        assert_eq!(trace.count(ExecutionEventKind::Relinearize), 1);
        assert_eq!(trace.count(ExecutionEventKind::Rescale), 1);
        assert_eq!(trace.count(ExecutionEventKind::Rotate), 0);
    }
}
