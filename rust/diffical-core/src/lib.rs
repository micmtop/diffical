//! Core signal and model plumbing for the Diffical backend.
//! Stage 1: mel frontend parity with the Python reference.

pub mod mel;

pub use mel::{MelFrontend, MelParams};

pub mod inference;
