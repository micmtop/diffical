//! pc-nsf-hifigan vocoder ONNX and the phrase waveform post-processing from
//! `PhraseRenderer._synthesize_waveform` / `_synthesize_per_phone:`
//! 6-frame context pad, voiced masking, context trim, gap gate, duration
//! alignment, 5.8 ms edge fades and phrase RMS normalization.

use crate::config::AudioConfig;
use crate::phone::{MelMatrix, PhoneFeature};
use anyhow::{Context, Result};
use ort::environment::{init as env_init, Environment};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

const CONTEXT_FRAMES: usize = 6;

pub struct Vocoder {
    _env: Environment,
    session: Session,
    n_mels: usize,
}

impl Vocoder {
    pub fn load(path: &Path, n_mels: usize) -> Result<Self> {
        let env = env_init().with_execution_providers(crate::ep::providers()).build().context("create ort environment")?;
        Self::load_with_env(&env, path, n_mels)
    }

    pub fn load_with_env(
        env: &Environment,
        path: &Path,
        n_mels: usize,
    ) -> Result<Self> {
        let session = Session::builder(env)?
            .with_execution_providers(crate::ep::providers())
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_auto_device(crate::ep::auto_policy())
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_execution_providers(crate::ep::providers())
            .unwrap_or_else(|e| e.recover())
            .with_optimization_level(GraphOptimizationLevel::All)
            .unwrap_or_else(|e| e.recover())
            .with_intra_threads(4)
            .unwrap_or_else(|e| e.recover())
            .with_inter_threads(2)
            .unwrap_or_else(|e| e.recover())
            .commit_from_file(path)
            .with_context(|| format!("load vocoder {path:?}"))?;
        Ok(Self {
            _env: env.clone(),
            session,
            n_mels,
        })
    }

    /// `mel`: log-mel frames `(n_mels, frames)`, `f0`: Hz `(frames)`,
    /// already context-padded and masked. Returns one waveform.
    fn vocode(&mut self, mel: &MelMatrix, f0: &[f32]) -> Result<Vec<f32>> {
        anyhow::ensure!(mel.frames == f0.len(), "mel/f0 frame mismatch");
        anyhow::ensure!(mel.frames > 0, "empty vocoder input");
        let mut mel_frames = Vec::with_capacity(mel.frames * self.n_mels);
        for t in 0..mel.frames {
            for band in 0..self.n_mels {
                mel_frames.push(mel.get(band, t));
            }
        }
        let mel_tensor = Tensor::from_array((
            vec![1i64, mel.frames as i64, self.n_mels as i64],
            mel_frames,
        ))?;
        let f0_tensor = Tensor::from_array((vec![1i64, f0.len() as i64], f0.to_vec()))?;
        let outputs = self
            .session
            .run(ort::inputs!["mel" => mel_tensor, "f0" => f0_tensor])?;
        let (shape, data) = outputs["waveform"].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            shape.len() == 2 && shape[0] == 1,
            "unexpected vocoder output shape {:?}",
            shape
        );
        anyhow::ensure!(
            shape[1] as usize == data.len(),
            "vocoder output shape/size mismatch"
        );
        Ok(data.to_vec())
    }

    /// Whole phrase vocode with the 6-frame repeated context, then trim it.
    pub fn synthesize_phrase(
        &mut self,
        mel_phrase: &MelMatrix,
        f0_concat: &[f32],
        total_frames: usize,
        cfg: &AudioConfig,
    ) -> Result<Vec<f32>> {
        let n = mel_phrase.frames.min(f0_concat.len());
        if n == 0 {
            return Ok(vec![0.0; total_frames * cfg.hop_size]);
        }
        let n_mels = self.n_mels;
        let pad = if n > CONTEXT_FRAMES {
            CONTEXT_FRAMES
        } else {
            0
        };
        let padded = n + 2 * pad;
        let mut mel_pad = MelMatrix {
            n_mels,
            frames: padded,
            data: vec![0f32; n_mels * padded],
        };
        let mut f0_pad = vec![0f32; padded];
        if pad > 0 {
            for k in 0..pad {
                for band in 0..n_mels {
                    let first = mel_phrase.get(band, 0);
                    let last = mel_phrase.get(band, n - 1);
                    mel_pad.data[band * padded + k] = first;
                    mel_pad.data[band * padded + pad + n + k] = last;
                }
                f0_pad[k] = f0_concat[0];
                f0_pad[pad + n + k] = f0_concat[n - 1];
            }
        }
        for t in 0..n {
            for band in 0..n_mels {
                mel_pad.data[band * padded + pad + t] = mel_phrase.get(band, t);
            }
            f0_pad[pad + t] = f0_concat[t];
        }

        for value in &mut f0_pad {
            if !value.is_finite() || *value < 0.0 {
                *value = 0.0;
            }
        }

        // A frame is voiced when it carries pitch or audible energy; pure
        // empty frames are forced to the mel floor (silence).
        let mut voiced = vec![false; padded];
        for (t, slot) in voiced.iter_mut().enumerate() {
            let mut energy = 0f64;
            for band in 0..n_mels {
                energy += (mel_pad.data[band * padded + t] as f64).exp();
            }
            energy /= n_mels as f64;
            *slot = f0_pad[t] > 50.0 || energy > 0.001;
        }
        for t in 0..padded {
            if !voiced[t] {
                for band in 0..n_mels {
                    mel_pad.data[band * padded + t] = -80.0;
                }
            }
        }

        let with_context = self.vocode(&mel_pad, &f0_pad)?;
        let context_samples = pad * cfg.hop_size;
        let mut samples = if with_context.len() > context_samples * 2 {
            with_context[context_samples..with_context.len() - context_samples].to_vec()
        } else {
            with_context
        };

        // Anti-click fades inside the vocoder helper (128 samples).
        let fade_samples = 128.min(samples.len() / 4);
        apply_linear_fades(&mut samples, fade_samples);
        let mut out = vec![0f32; total_frames * cfg.hop_size];
        let copy = samples.len().min(out.len());
        out[..copy].copy_from_slice(&samples[..copy]);
        Ok(out)
    }
}

/// First `fade` samples ramp 0->1, last `fade` samples ramp 1->0.
fn apply_linear_fades(samples: &mut [f32], fade: usize) {
    if fade == 0 || samples.is_empty() {
        return;
    }
    for i in 0..fade.min(samples.len()) {
        let w = if fade > 1 {
            i as f32 / (fade - 1) as f32
        } else {
            0.0
        };
        samples[i] *= w;
    }
    let len = samples.len();
    for i in 0..fade.min(len) {
        let w = if fade > 1 {
            i as f32 / (fade - 1) as f32
        } else {
            1.0
        };
        samples[len - 1 - i] *= w;
    }
}

pub fn sanitize(samples: &mut [f32]) -> usize {
    let mut bad = 0;
    for value in samples {
        if !value.is_finite() {
            *value = 0.0;
            bad += 1;
        }
    }
    bad
}

/// Silence the rests between phone segments, with 10 ms ramps at both edges.
pub fn gap_gate(
    samples: &mut [f32],
    features: &[PhoneFeature],
    total_frames: usize,
    cfg: &AudioConfig,
) {
    let mut ordered: Vec<&PhoneFeature> = features.iter().collect();
    ordered.sort_by_key(|f| f.start_frame);
    let hop = cfg.hop_size;
    let fade_n = (0.010 * cfg.sample_rate as f64) as usize;
    for pair in ordered.windows(2) {
        let prev = pair[0];
        let curr = pair[1];
        let prev_start = prev.start_frame.max(0) as usize;
        let prev_end = (prev_start + prev.mel.frames).min(total_frames);
        let curr_start = curr.start_frame.max(0) as usize;
        if curr_start <= prev_end {
            continue;
        }
        let s0 = (prev_end * hop).min(samples.len());
        let s1 = (curr_start * hop).min(samples.len());
        if s1 <= s0 {
            continue;
        }
        let fade_in_out = fade_n.min(s0);
        let fade_in = fade_n.min(samples.len() - s1);
        if fade_in_out > 0 {
            for i in 0..fade_in_out {
                let w = if fade_in_out > 1 {
                    1.0 - i as f32 / (fade_in_out - 1) as f32
                } else {
                    1.0
                };
                samples[s0 - fade_in_out + i] *= w;
            }
        }
        samples[s0..s1].fill(0.0);
        if fade_in > 0 {
            for i in 0..fade_in {
                let w = if fade_in > 1 {
                    i as f32 / (fade_in - 1) as f32
                } else {
                    0.0
                };
                samples[s1 + i] *= w;
            }
        }
    }
}

/// Trim/pad to the exact client-visible phrase length.
pub fn align_duration(samples: &mut Vec<f32>, expected_samples: usize) {
    if samples.len() > expected_samples {
        samples.truncate(expected_samples);
    } else if samples.len() < expected_samples {
        samples.resize(expected_samples, 0.0);
    }
}

/// Final 5.8 ms phrase-edge fade (256 samples at 44.1 kHz).
pub fn phrase_edge_fade(samples: &mut [f32]) {
    let fade = 256.min(samples.len() / 4);
    apply_linear_fades(samples, fade);
}

#[derive(Debug, Clone)]
pub struct NormalizationTrace {
    pub reason: &'static str,
    pub gain: f64,
    pub activity_threshold: f64,
    pub active_rms_before: f64,
    pub peak_before: f64,
    pub active_windows: usize,
    pub total_windows: usize,
}

/// `backend/loudness.py::normalize_phrase_rms` with the renderer's arguments.
pub fn normalize_phrase_rms(
    samples: &mut [f32],
    sample_rate: u32,
    target_rms: f64,
    peak_ceiling: f64,
) -> NormalizationTrace {
    let empty_trace = NormalizationTrace {
        reason: "empty",
        gain: 1.0,
        activity_threshold: 0.0,
        active_rms_before: 0.0,
        peak_before: 0.0,
        active_windows: 0,
        total_windows: 0,
    };
    if samples.is_empty() {
        return empty_trace;
    }
    let peak = samples.iter().fold(0f64, |acc, &v| acc.max(v.abs() as f64));
    if peak <= 1e-12 {
        return NormalizationTrace {
            reason: "silent",
            peak_before: peak,
            ..empty_trace
        };
    }
    let window = ((sample_rate as f64 * 0.020).round() as usize).max(1);
    let hop = ((sample_rate as f64 * 0.010).round() as usize).max(1);
    let mut starts: Vec<usize> = if samples.len() <= window {
        vec![0]
    } else {
        (0..=samples.len() - window).step_by(hop).collect()
    };
    if samples.len() > window {
        let final_start = samples.len() - window;
        if starts.last() != Some(&final_start) {
            starts.push(final_start);
        }
    }
    let frame_rms: Vec<f64> = starts
        .iter()
        .map(|&start| {
            let sum: f64 = samples[start..start + window]
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum();
            (sum / window as f64).sqrt()
        })
        .collect();
    let threshold = 1e-5_f64.max(peak * 10f64.powf(-45.0 / 20.0));
    let active: Vec<f64> = frame_rms
        .iter()
        .copied()
        .filter(|v| *v >= threshold)
        .collect();
    if active.is_empty() {
        return NormalizationTrace {
            reason: "no_active_windows",
            peak_before: peak,
            activity_threshold: threshold,
            total_windows: frame_rms.len(),
            ..empty_trace
        };
    }
    let measured =
        (active.iter().map(|v| v * v).sum::<f64>() / active.len() as f64).sqrt();
    let requested_gain = target_rms / measured.max(1e-12);
    let peak_gain = peak_ceiling / peak;
    let gain = requested_gain.min(peak_gain);
    for value in samples.iter_mut() {
        *value = (*value as f64 * gain) as f32;
    }
    NormalizationTrace {
        reason: if gain < requested_gain {
            "peak_limited"
        } else {
            "target_reached"
        },
        gain,
        activity_threshold: threshold,
        active_rms_before: measured,
        peak_before: peak,
        active_windows: active.len(),
        total_windows: frame_rms.len(),
    }
}
