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
    pub level: Option<usize>,
    pub rns_limbs: Option<usize>,
}

impl ExecutionEvent {
    /// Constructs an event without representation-specific metadata.
    pub const fn new(kind: ExecutionEventKind) -> Self {
        Self {
            kind,
            ring_degree: None,
            level: None,
            rns_limbs: None,
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
