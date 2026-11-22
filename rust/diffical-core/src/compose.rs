//! Envelope-timeline composition.
//!
//! Each phone's stretched mel is placed at its own start frame, gated by the
//! OpenUtau envelope, and overlapping phones are summed in the power domain.
//! This is the same geometry the Python backend uses (`backend/utau_geometry.py`):
//! placement is defined by the oto window and the envelope span, not by the
//! next phone's audio start.
//!
//! Use `DIFFICAL_GEOM_V3=0` to fall back to the onnx segment assembler.

use anyhow::{Context, Result};

use crate::phone::PhoneFeature;
use crate::protocol::{EnvelopePoint, RenderPhone};
use crate::source::MelMatrix;

pub const FRAME_MS: f64 = 512.0 / 44100.0 * 1000.0;
const QUIET: f32 = -30.0;
const MEL_FLOOR: f64 = -30.0;

/// Linear gain of a piecewise-linear envelope at `t_ms` (relative to the note
/// start). OpenUtau sends `y` either as 0..1 or as 0..100 percent; both are
/// accepted by normalising on the maximum.
pub fn envelope_gain(envelope: &[EnvelopePoint], t_ms: f64) -> f64 {
    if envelope.len() < 2 {
        return 1.0;
    }
    let max_y = envelope.iter().map(|p| p.y()).fold(0.0_f64, f64::max);
    let scale = if max_y > 1.5 { 100.0 } else { 1.0 };
    let first = &envelope[0];
    let last = &envelope[envelope.len() - 1];
    if t_ms <= first.x() {
        return first.y() / scale;
    }
    if t_ms >= last.x() {
        return last.y() / scale;
    }
    for pair in envelope.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if t_ms <= b.x() {
            let dx = b.x() - a.x();
            if dx.abs() < 1e-9 {
                return b.y() / scale;
            }
            let w = (t_ms - a.x()) / dx;
            return (a.y() + (b.y() - a.y()) * w) / scale;
        }
    }
    last.y() / scale
}

/// Per-frame gain of one phone window.
fn window_gains(feature: &PhoneFeature, phone: &RenderPhone) -> Vec<f32> {
    let start = feature.start_frame_exact.round() as i64;
    let anchor_ms = start as f64 * FRAME_MS + feature.preutter_ms;
    let frames = feature.mel.frames;
    let mut out = vec![1.0_f32; frames];
    for (k, slot) in out.iter_mut().enumerate() {
        let t_ms = (start + k as i64) as f64 * FRAME_MS - anchor_ms;
        *slot = envelope_gain(&phone.envelope, t_ms) as f32;
    }
    out
}

/// Compose the phrase mel from per-phone windows.
pub fn compose_envelope(
    features: &[PhoneFeature],
    phones: &[RenderPhone],
    total_frames: usize,
    n_mels: usize,
) -> Result<MelMatrix> {
    let mut power = vec![0.0_f64; n_mels * total_frames];
    let mut touched = false;
    for feature in features {
        let phone = phones
            .get(feature.index)
            .with_context(|| format!("phone feature {} has no request entry", feature.index))?;
        if feature.mel.frames == 0 {
            continue;
        }
        let start = feature.start_frame_exact.round() as i64;
        let gains = window_gains(feature, phone);
        for k in 0..feature.mel.frames {
            let t = start + k as i64;
            if t < 0 || t >= total_frames as i64 {
                continue;
            }
            let gain = gains[k] as f64;
            let t = t as usize;
            for bin in 0..n_mels {
                let value = feature.mel.data[bin * feature.mel.frames + k] as f64;
                let clamped = value.clamp(MEL_FLOOR, 2.0);
                power[bin * total_frames + t] += (2.0 * clamped).exp() * gain;
            }
            touched = true;
        }
    }
    let mut data = vec![QUIET; n_mels * total_frames];
    if touched {
        for (i, p) in power.iter().enumerate() {
            data[i] = (0.5 * p.max(1e-24).ln()) as f32;
        }
    }
    Ok(MelMatrix {
        n_mels,
        frames: total_frames,
        data,
    })
}

/// Env switch: composition is on unless explicitly disabled.
pub fn geom_v3_from_env() -> bool {
    match std::env::var("DIFFICAL_GEOM_V3") {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no"),
        Err(_) => true,
    }
}
