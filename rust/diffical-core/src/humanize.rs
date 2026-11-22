//! HUM (Humanization): slow, note-driven phrase-level undulation.
//!
//! Not random noise and not pitch jitter. The envelope is derived from the note
//! structure (duration, melodic direction) and scaled by the HUM curve, then
//! applied to the assembled mel right before the vocoder:
//!
//! * level channel  — slow gain undulation on top of DYN
//! * timbre channel — the same envelope drives a spectral tilt, so louder parts
//!   read brighter/tenser and softer parts darker/breathier
//!
//! Same math as `backend/humanize.py`, so both backends agree; HUM = 0 is a
//! bit-exact no-op.

use crate::protocol::{CurveInput, RenderNote};
use crate::source::MelMatrix;

pub const ABBR: &str = "HUM";
const FRAME_MS: f64 = 512.0 / 44100.0 * 1000.0;
const MAX_GAIN_DB: f64 = 6.0;
const MAX_TILT_DB: f64 = 3.0;

pub fn default_hum() -> f64 {
    std::env::var("DIFFICAL_HUM_DEFAULT")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// HUM curve resampled to per-frame values in 0..100; missing -> `default`.
pub fn hum_per_frame(curves: &[CurveInput], total_frames: usize, default: f64) -> Vec<f32> {
    let t = total_frames.max(1);
    for curve in curves {
        let (abbr, values) = match curve {
            CurveInput::Tuple((a, v)) => (a.as_str(), v.as_slice()),
            CurveInput::Object { abbr, values } => (abbr.as_str(), values.as_slice()),
        };
        if abbr.eq_ignore_ascii_case(ABBR) && !values.is_empty() {
            let n = values.len();
            let mut out = Vec::with_capacity(t);
            for i in 0..t {
                let pos = if t <= 1 {
                    0.0
                } else {
                    i as f64 * (n - 1) as f64 / (t - 1) as f64
                };
                let i0 = pos.floor() as usize;
                let i1 = (i0 + 1).min(n - 1);
                let w = pos - i0 as f64;
                let v = values[i0] * (1.0 - w) + values[i1] * w;
                out.push(v.clamp(0.0, 100.0) as f32);
            }
            return out;
        }
    }
    vec![default.clamp(0.0, 100.0) as f32; t]
}

/// Note-derived slow envelope in roughly [-1, 1].
pub fn phrase_envelope(notes: &[RenderNote], total_frames: usize, lead_ms: f64) -> Vec<f32> {
    let t = total_frames.max(1);
    let mut env = vec![0.0_f64; t];
    if notes.is_empty() {
        return vec![0.0_f32; t];
    }
    let base = notes
        .iter()
        .map(|n| n.position_ms)
        .fold(f64::INFINITY, f64::min)
        - lead_ms.max(0.0);
    let mut prev_tone: Option<f64> = None;
    for note in notes {
        let dur = note.duration_ms;
        let start = note.position_ms - base;
        let tone = note.tone as f64;
        if dur <= 0.0 {
            continue;
        }
        let depth = (dur / 900.0).clamp(0.3, 1.0);
        let slope = match prev_tone {
            None => 0.0,
            Some(p) => ((tone - p) / 2.0).clamp(-1.0, 1.0),
        };
        prev_tone = Some(tone);
        let i0 = (start / FRAME_MS).floor().max(0.0) as usize;
        let i1 = ((start + dur) / FRAME_MS).ceil().max(0.0) as usize;
        let i0 = i0.min(t);
        let i1 = i1.min(t);
        if i1 <= i0 {
            continue;
        }
        for i in i0..i1 {
            let u = ((i as f64 * FRAME_MS - start) / dur).clamp(0.0, 1.0);
            let rise = (std::f64::consts::PI * (u / 0.7).min(1.0)).sin();
            let decay = (-3.0 * u).exp();
            let mut shape = rise * (0.55 + 0.45 * decay);
            shape *= 1.0 + 0.35 * slope * u;
            env[i] += depth * shape;
        }
    }
    // smooth over ~250 ms so it stays phrase-level
    let win = (250.0 / FRAME_MS).round().max(1.0) as usize;
    if win > 1 && t > win {
        let mut smoothed = vec![0.0_f64; t];
        let half = win / 2;
        for i in 0..t {
            let a = i.saturating_sub(half);
            let b = (i + half + 1).min(t);
            let sum: f64 = env[a..b].iter().sum();
            smoothed[i] = sum / (b - a) as f64;
        }
        env = smoothed;
    }
    // remove the DC so the modulation rides on top of DYN instead of lifting it
    let mean = env.iter().sum::<f64>() / t as f64;
    for v in env.iter_mut() {
        *v -= mean;
    }
    let peak = env.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if peak > 1e-9 {
        for v in env.iter_mut() {
            *v /= peak;
        }
    }
    env.into_iter().map(|v| v as f32).collect()
}

/// Apply HUM to the assembled mel: level gain plus a low/high tilt.
pub fn apply(mel: &mut MelMatrix, hum: &[f32], env: &[f32]) {
    let t = mel.frames.min(hum.len()).min(env.len());
    if t == 0 {
        return;
    }
    let n = mel.n_mels;
    if n == 0 {
        return;
    }
    let mut any = false;
    for i in 0..t {
        let h = (hum[i] as f64).clamp(0.0, 100.0) / 100.0;
        let drive = h * env[i] as f64;
        if drive.abs() <= 1e-9 {
            continue;
        }
        any = true;
        let gain_db = drive * MAX_GAIN_DB;
        let tilt_db = drive * MAX_TILT_DB;
        for bin in 0..n {
            let weight = if n > 1 {
                -1.0 + 2.0 * bin as f64 / (n - 1) as f64
            } else {
                0.0
            };
            let delta = ((gain_db + tilt_db * weight) / 8.686) as f32;
            mel.data[bin * mel.frames + i] += delta;
        }
    }
    let _ = any;
}
