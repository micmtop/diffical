//! Phrase F0 curve (`backend/renderer.py::PhraseRenderer._build_pitch_curve`).
//!
//! Two input conventions exist:
//! - per-frame: `pitches` is already one absolute-cent value per vocoder frame;
//! - 5-tick: `pitches` is sampled every 5 ticks and is interpolated by frame.

use crate::config::AudioConfig;
use crate::protocol::RenderRequest;
use anyhow::{bail, Result};

pub struct PitchCurve {
    pub total_frames: usize,
    pub f0: Vec<f32>,
}

fn midi_to_hz(midi: f64) -> f64 {
    440.0 * 2f64.powf((midi - 69.0) / 12.0)
}

fn finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn phone_preutter(phone: &crate::protocol::RenderPhone) -> f64 {
    finite_or(phone.preutter_ms, 0.0)
}

/// `frame_idx -> pitches[]` float index for the 5-tick convention.
fn frame_to_pitch_pos(req: &RenderRequest, cfg: &AudioConfig, frame_idx: usize) -> f64 {
    if req.phones.is_empty() {
        return 0.0;
    }
    let abs_ms =
        (req.position_ms - req.leading_ms) + frame_idx as f64 * cfg.frame_ms();
    for phone in &req.phones {
        let preutter_ms = phone_preutter(phone);
        let dur_ms = if phone.duration_ms != 0.0 {
            phone.duration_ms
        } else {
            1.0
        };
        if phone.position_ms - preutter_ms <= abs_ms && abs_ms < phone.end_ms {
            let rel_ms = abs_ms - phone.position_ms;
            let tick = phone.position as f64 + rel_ms * phone.duration as f64 / dur_ms;
            return (tick + req.leading as f64) / 5.0;
        }
    }
    let phone = req.phones.last().expect("checked non-empty");
    let dur_ms = if phone.duration_ms != 0.0 {
        phone.duration_ms
    } else {
        1.0
    };
    let rel_ms = abs_ms - phone.position_ms;
    let tick = phone.position as f64 + rel_ms * phone.duration as f64 / dur_ms;
    (tick + req.leading as f64) / 5.0
}

pub fn build_pitch_curve(req: &RenderRequest, cfg: &AudioConfig) -> Result<PitchCurve> {
    let total_ms = req.duration_ms + req.leading_ms;
    anyhow::ensure!(total_ms.is_finite(), "phrase duration is not finite");
    let mut total_frames = (total_ms / cfg.frame_ms()).ceil() as i64;
    total_frames = total_frames.max(1);
    anyhow::ensure!(
        total_frames < 4_000_000,
        "phrase too long for frame curve ({total_frames} frames)"
    );
    let total_frames = total_frames as usize;

    let mut voiced = vec![false; total_frames];
    for phone in &req.phones {
        let preutter_ms = phone_preutter(phone);
        let audio_start_ms =
            phone.position_ms - req.position_ms + req.leading_ms - preutter_ms;
        let note_end_ms = phone.end_ms - req.position_ms + req.leading_ms;
        let start_frame = (audio_start_ms / cfg.frame_ms()) as i64;
        let end_frame = (note_end_ms / cfg.frame_ms()).ceil() as i64;
        let start = start_frame.max(0) as usize;
        let end = end_frame.min(total_frames as i64).max(0) as usize;
        if start < end {
            voiced[start..end].fill(true);
        }
    }

    if req.pitches.is_empty() {
        if req.phones.is_empty() {
            return Ok(PitchCurve {
                total_frames,
                f0: vec![0.0; total_frames],
            });
        }
        bail!("request pitch curve is missing; backend will not invent F0");
    }

    let pitches: Vec<f64> = req.pitches.clone();
    let per_frame = pitches.len() as f64 >= total_frames as f64 * 0.9
        && pitches.len() as f64 <= total_frames as f64 * 1.1;
    let mut f0 = vec![0f32; total_frames];

    if per_frame {
        for i in 0..total_frames {
            if !voiced[i] {
                continue;
            }
            let mut base_tone = 60i64;
            let frame_ms_val = i as f64 * cfg.frame_ms();
            for phone in &req.phones {
                let preutter_ms = phone_preutter(phone);
                let audio_start_ms =
                    phone.position_ms - req.position_ms + req.leading_ms - preutter_ms;
                let note_end_ms = phone.end_ms - req.position_ms + req.leading_ms;
                if audio_start_ms <= frame_ms_val && frame_ms_val < note_end_ms {
                    base_tone = phone.tone;
                    break;
                }
            }
            let hz = if i < pitches.len() {
                midi_to_hz(pitches[i] / 100.0)
            } else {
                midi_to_hz(base_tone as f64)
            };
            f0[i] = hz as f32;
        }
    } else {
        for (i, slot) in f0.iter_mut().enumerate() {
            if !voiced[i] {
                continue;
            }
            let pos = frame_to_pitch_pos(req, cfg, i).clamp(0.0, pitches.len() as f64 - 1.0);
            let idx0 = pos.floor() as usize;
            let idx1 = (idx0 + 1).min(pitches.len() - 1);
            let frac = pos - idx0 as f64;
            let cent = pitches[idx0] * (1.0 - frac) + pitches[idx1] * frac;
            *slot = midi_to_hz(cent / 100.0) as f32;
        }
    }

    for value in &mut f0 {
        if *value > 0.0 {
            *value = value.clamp(50.0, 1500.0);
        } else {
            *value = 0.0;
        }
    }
    Ok(PitchCurve { total_frames, f0 })
}
