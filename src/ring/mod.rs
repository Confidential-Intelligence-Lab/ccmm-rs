//! Ring, modulus, RNS, polynomial, and NTT arithmetic.
//!
//! This module provides the arithmetic substrate shared by the reference and
//! optimized Roadmap-2 implementations.
//!
pub mod modulus;
pub mod ntt;
pub mod ntt_params;
pub mod ntt_polynomial;
pub mod ntt_radix2;
pub mod polynomial;

pub use modulus::Modulus;
pub use ntt::NttPlan;
pub use polynomial::Polynomial;

pub use ntt_params::{find_negacyclic_root, make_ntt_plan};

pub use ntt_polynomial::NttPolynomial;

pub mod rns;
pub use rns::RnsPolynomial;

pub mod rns_ntt;
pub use rns_ntt::{RnsNttPlan, RnsNttPolynomial};

pub mod modulus_basis;
pub use modulus_basis::ModulusBasis;

pub mod modulus_chain;
pub use modulus_chain::ModulusChain;

pub mod basis_conversion;
pub use basis_conversion::convert_basis;

pub mod basis_transition;
pub use basis_transition::{drop_basis_prefix, extend_basis_prefix};

#[cfg(test)]
mod basis_differential;

pub mod wide_crt;

pub use wide_crt::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big,
    rns_from_big_coefficients,
};
