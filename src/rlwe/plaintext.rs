use super::RlweParameters;

/// Canonical coefficient message for the minimal RLWE layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlwePlaintext {
    coefficients: Vec<u64>,
}

impl RlwePlaintext {
    pub fn new(params: RlweParameters, coefficients: Vec<u64>) -> Self {
        assert_eq!(
            coefficients.len(),
            params.degree(),
            "plaintext length must match RLWE degree"
        );

        let coefficients = coefficients
            .into_iter()
            .map(|value| value % params.plaintext_modulus())
            .collect();

        Self { coefficients }
    }

    pub fn coefficients(&self) -> &[u64] {
        &self.coefficients
    }
}
