//! Core signal and model plumbing for the Diffical backend.
//!
//! Pipeline: source wav -> hop-128 log-mel -> per-phone stretch -> PointerGen
//! segment assembler (ONNX) -> pc-nsf-hifigan vocoder (ONNX) -> phrase
//! gap gate / duration alignment / edge fade / RMS normalization.

pub mod assemble;
pub mod biquad;
pub mod config;
pub mod effects;
pub mod flags;
pub mod hnsep;
pub mod inference;
pub mod mel;
pub mod phone;
pub mod pipeline;
pub mod pitch;
pub mod protocol;
pub mod runtime;
pub mod source;
pub mod stretch;
pub mod vocoder;
pub mod wav;

pub use config::AudioConfig;
pub use mel::{MelFrontend, MelParams};
pub use pipeline::{PhoneStage, RenderEngine, Rendered};
pub use protocol::{parse_request, HealthResponse, RenderRequest, RenderResultResponse};

pub mod ep;
