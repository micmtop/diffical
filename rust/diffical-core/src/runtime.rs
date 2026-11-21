//! Process-wide ONNX Runtime environment.
//!
//! `ort` allows exactly one environment per process, so every session in a
//! render or parity command must share it. Re-exported here so the CLI never
//! names the `ort` crate directly.

use anyhow::{Context, Result};

pub use ort::environment::Environment;

pub fn environment() -> Result<Environment> {
    ort::environment::init()
        .build()
        .context("create ort environment")
}
