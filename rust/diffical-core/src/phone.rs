//! Per-phone source window, stretch mapping and mel construction.
//!
//! Port of `backend/renderer.py::PhoneProcessor.process` for the model path
//! with flags/curves disabled: peak guard, hop-128 source mel, oto window,
//! velocity handling, envelope span, vowel stretch mapping, tail silence.

use crate::config::AudioConfig;
use crate::mel::MelFrontend;
use crate::protocol::{RenderPhone, RenderRequest};
use crate::wav::read_wav_mono;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Row-major `n_mels x frames` log-mel.
#[derive(Debug, Clone)]
pub struct MelMatrix {
    pub n_mels: usize,
    pub frames: usize,
    pub data: Vec<f32>,
}

impl MelMatrix {
    pub fn empty(n_mels: usize) -> Self {
        Self {
            n_mels,
            frames: 0,
            data: Vec::new(),
        }
    }

    pub fn from_vec(n_mels: usize, data: Vec<f32>) -> Result<Self> {
        anyhow::ensure!(
            n_mels > 0 && data.len() % n_mels == 0,
            "mel data length {} is not a multiple of {n_mels}",
            data.len()
        );
        Ok(Self {
            n_mels,
            frames: data.len() / n_mels,
            data,
        })
    }

    #[inline]
    pub fn get(&self, band: usize, frame: usize) -> f32 {
        self.data[band * self.frames + frame]
    }
}

#[derive(Debug, Clone)]
pub struct PhoneDebug {
    pub source_wave: Vec<f32>,
    pub mel_orig: MelMatrix,
}

#[derive(Debug, Clone)]
pub struct PhoneFeature {
    pub index: usize,
    pub phoneme: String,
    pub start_frame: i64,
    pub start_frame_exact: f64,
    pub mel: MelMatrix,
    pub f0: Vec<f32>,
    pub preutter_ms: f64,
    pub overlap_ms: f64,
    pub pre_vel_ms: f64,
    pub source_skip_ms: f64,
    pub consonant_output_ms: f64,
    pub envelope_span_ms: f64,
    pub segment_len_ms: f64,
    pub tail_silence_frames: usize,
    pub pre_scale: f64,
    pub debug: Option<PhoneDebug>,
}

/// Loads voicebank wav files, caching per render request like `AudioLoader`.
pub struct SourceLoader {
    sample_rate: u32,
    singer_path: String,
    cache: HashMap<String, Arc<Vec<f32>>>,
}

impl SourceLoader {
    pub fn new(sample_rate: u32, singer_path: &str) -> Self {
        Self {
            sample_rate,
            singer_path: singer_path.to_string(),
            cache: HashMap::new(),
        }
    }

    fn resolve(&self, oto_file: &str) -> Option<PathBuf> {
        if oto_file.is_empty() {
            return None;
        }
        let direct = Path::new(oto_file);
        if direct.is_absolute() {
            return direct.is_file().then(|| direct.to_path_buf());
        }
        let joined = Path::new(&self.singer_path).join(oto_file);
        joined.is_file().then_some(joined)
    }

    /// Missing/unreadable sources degrade to ~100 ms of silence, matching
    /// `PhoneProcessor._load_wave`; a render must not fail for one bad phone.
    pub fn load(&mut self, oto_file: &str) -> Arc<Vec<f32>> {
        if let Some(cached) = self.cache.get(oto_file) {
            return Arc::clone(cached);
        }
        let wave = match self.resolve(oto_file) {
            Some(path) => match read_wav_mono(&path, self.sample_rate) {
                Ok(samples) => samples,
                Err(err) => {
                    eprintln!("warning: failed to load {path:?}: {err:#}; using silence");
                    silence(self.sample_rate)
                }
            },
            None => {
                if !oto_file.is_empty() {
                    eprintln!("warning: voicebank file not found: {oto_file}; using silence");
                }
                silence(self.sample_rate)
            }
        };
        let wave = Arc::new(wave);
        self.cache.insert(oto_file.to_string(), Arc::clone(&wave));
        wave
    }
}

fn silence(sample_rate: u32) -> Vec<f32> {
    vec![0.0; (0.1 * sample_rate as f64) as usize]
}

fn finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

struct PhoneContext {
    preutter_ms: f64,
    overlap_ms: f64,
    start_frame: i64,
    start_frame_exact: f64,
    pre_vel_ms: f64,
    source_skip_ms: f64,
    total_span_ms: f64,
    f0_slice: Vec<f32>,
}

pub struct PhoneProcessor<'a> {
    cfg: &'a AudioConfig,
    analyzer: &'a MelFrontend,
    loader: SourceLoader,
}

impl<'a> PhoneProcessor<'a> {
    pub fn new(
        cfg: &'a AudioConfig,
        analyzer: &'a MelFrontend,
        singer_path: &str,
    ) -> Self {
        Self {
            cfg,
            analyzer,
            loader: SourceLoader::new(cfg.sample_rate, singer_path),
        }
    }

    pub fn features(
        &mut self,
        req: &RenderRequest,
        f0_phrase: &[f32],
        capture_debug: bool,
    ) -> Result<Vec<PhoneFeature>> {
        let mut out = Vec::with_capacity(req.phones.len());
        for (index, phone) in req.phones.iter().enumerate() {
            let ctx = self.context(req, f0_phrase, index)?;
            out.push(self.process(index, phone, ctx, capture_debug)?);
        }
        Ok(out)
    }

    fn context(
        &self,
        req: &RenderRequest,
        f0_phrase: &[f32],
        index: usize,
    ) -> Result<PhoneContext> {
        let phone = &req.phones[index];
        let frame_ms = self.cfg.frame_ms();
        let target_start_ms = phone.position_ms - req.position_ms + req.leading_ms;
        let target_duration_ms = phone.duration_ms;

        let mut preutter_ms = finite_or(phone.preutter_ms, 0.0);
        if preutter_ms < 0.0 {
            preutter_ms = 0.0;
        }
        let preutter_max = 5000.0_f64.max(target_duration_ms);
        if preutter_ms > preutter_max {
            preutter_ms = preutter_max;
        }
        let overlap_ms = finite_or(phone.overlap_ms, 0.0);
        let velocity = finite_or(phone.velocity, 1.0);
        let vel = 2f64.powf(1.0 - velocity);

        let (tail_intrude_ms, tail_overlap_ms) = if index + 1 < req.phones.len() {
            let next = &req.phones[index + 1];
            let next_pre = finite_or(next.preutter_ms, 0.0).max(0.0);
            let next_ovl = finite_or(next.overlap_ms, 0.0).max(0.0);
            (next_pre.max(next_pre - next_ovl), next_ovl.max(0.0))
        } else {
            (0.0, 0.0)
        };

        let audio_start_ms = target_start_ms - preutter_ms;
        let start_frame_exact = audio_start_ms / frame_ms;
        let pre_vel_ms = target_start_ms - audio_start_ms;
        let start_frame = (audio_start_ms / frame_ms) as i64;

        let mut total_span_ms = target_duration_ms + pre_vel_ms - tail_intrude_ms + tail_overlap_ms;
        let oto_consonant = finite_or(phone.oto_consonant, 0.0);
        total_span_ms = total_span_ms.max(oto_consonant * vel + frame_ms);

        let mut target_n_frames = ((total_span_ms + 80.0) / frame_ms).round() as i64;
        target_n_frames = target_n_frames.max(1);
        anyhow::ensure!(
            target_n_frames < 200_000,
            "phone {} is implausibly long ({target_n_frames} frames)",
            phone.phoneme
        );

        let mut f0_slice = vec![0f32; target_n_frames as usize];
        let src_start = start_frame.max(0);
        let src_end = (start_frame + target_n_frames).min(f0_phrase.len() as i64);
        if src_end > src_start {
            let len = (src_end - src_start) as usize;
            let dst = (src_start - start_frame) as usize;
            f0_slice[dst..dst + len]
                .copy_from_slice(&f0_phrase[src_start as usize..src_end as usize]);
        }

        let source_skip_ms =
            finite_or(phone.oto_preutter, 0.0).max(0.0) * vel - pre_vel_ms;
        Ok(PhoneContext {
            preutter_ms,
            overlap_ms,
            start_frame,
            start_frame_exact,
            pre_vel_ms,
            source_skip_ms,
            total_span_ms,
            f0_slice,
        })
    }

    fn process(
        &mut self,
        index: usize,
        phone: &RenderPhone,
        ctx: PhoneContext,
        capture_debug: bool,
    ) -> Result<PhoneFeature> {
        let cfg = self.cfg;
        let n_mels = cfg.n_mels;
        let frame_ms = cfg.frame_ms();
        let origin_frame_ms = cfg.origin_frame_ms();

        let mut wave = self.loader.load(&phone.oto_file);
        if wave.is_empty() {
            return Ok(PhoneFeature {
                index,
                phoneme: phone.phoneme.clone(),
                start_frame: ctx.start_frame,
                start_frame_exact: ctx.start_frame_exact,
                mel: MelMatrix::empty(n_mels),
                f0: Vec::new(),
                preutter_ms: ctx.preutter_ms,
                overlap_ms: ctx.overlap_ms,
                pre_vel_ms: ctx.pre_vel_ms,
                source_skip_ms: ctx.source_skip_ms,
                consonant_output_ms: 0.0,
                envelope_span_ms: 0.0,
                segment_len_ms: ctx.total_span_ms,
                tail_silence_frames: 0,
                pre_scale: 1.0,
                debug: None,
            });
        }

        // Peak guard only when close to clipping (no unconditional pre_scale).
        let peak = wave.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
        let pre_scale = if peak > 0.98 {
            (0.98 / peak) as f64
        } else {
            1.0
        };
        if pre_scale != 1.0 {
            let scale = pre_scale as f32;
            Arc::make_mut(&mut wave).iter_mut().for_each(|v| *v *= scale);
        }

        let params = self.analyzer.params();
        let padded_len = wave.len() + params.pad_left + params.pad_right;
        if padded_len >= params.win_size {
            let source_frames = 1 + (padded_len - params.win_size) / params.hop;
            anyhow::ensure!(
                source_frames < 500_000,
                "source wav {:?} is too long for mel analysis ({source_frames} frames)",
                phone.oto_file
            );
        }
        let mel_orig =
            MelMatrix::from_vec(n_mels, self.analyzer.compute(&wave)?).context("source mel")?;
        let orig_n_frames = mel_orig.frames;
        if orig_n_frames == 0 {
            return Ok(PhoneFeature {
                index,
                phoneme: phone.phoneme.clone(),
                start_frame: ctx.start_frame,
                start_frame_exact: ctx.start_frame_exact,
                mel: MelMatrix::empty(n_mels),
                f0: Vec::new(),
                preutter_ms: ctx.preutter_ms,
                overlap_ms: ctx.overlap_ms,
                pre_vel_ms: ctx.pre_vel_ms,
                source_skip_ms: ctx.source_skip_ms,
                consonant_output_ms: 0.0,
                envelope_span_ms: 0.0,
                segment_len_ms: ctx.total_span_ms,
                tail_silence_frames: 0,
                pre_scale,
                debug: capture_debug.then(|| PhoneDebug {
                    source_wave: wave.as_ref().clone(),
                    mel_orig: mel_orig.clone(),
                }),
            });
        }

        let offset_ms = finite_or(phone.oto_offset, 0.0);
        let consonant_ms = finite_or(phone.oto_consonant, 0.0);
        let cutoff_ms = finite_or(phone.oto_cutoff, 0.0);
        let total_time_ms = orig_n_frames as f64 * origin_frame_ms;
        let t_orig_ms: Vec<f64> = (0..orig_n_frames)
            .map(|i| i as f64 * origin_frame_ms + origin_frame_ms / 2.0)
            .collect();

        let mut start_ms = offset_ms;
        let con_ms = start_ms + consonant_ms;
        let mut end_ms = if cutoff_ms < 0.0 {
            start_ms + cutoff_ms.abs()
        } else if cutoff_ms > 0.0 {
            total_time_ms - cutoff_ms
        } else {
            total_time_ms
        };
        end_ms = end_ms.min(total_time_ms).max(con_ms);
        start_ms = start_ms.max(0.0);

        let velocity = finite_or(phone.velocity, 1.0);
        let vel = 2f64.powf(1.0 - velocity);
        let pre_vel_ms = ctx.pre_vel_ms.max(0.0);
        let source_skip_ms = ctx.source_skip_ms;
        let con_vel = consonant_ms * vel;
        let con_output_ms = con_vel - source_skip_ms;
        let has_envelope = !phone.envelope.is_empty();
        let envelope_span_ms = if has_envelope {
            pre_vel_ms
                + phone
                    .envelope
                    .iter()
                    .map(|p| finite_or(p.x(), 0.0))
                    .fold(f64::NEG_INFINITY, f64::max)
        } else {
            pre_vel_ms + phone.duration_ms
        };
        let total_span_ms = frame_ms
            .max(envelope_span_ms)
            .max(con_output_ms + frame_ms);

        let stretch_length_ms = (end_ms - con_ms).max(0.0);
        let vowel_output_ms = frame_ms.max(total_span_ms - con_output_ms);
        let scaling_ratio = 1.0_f64.max(
            vowel_output_ms / stretch_length_ms.max(origin_frame_ms),
        );
        let mut target_n_frames = ((total_span_ms / frame_ms).ceil() as i64).max(1);
        anyhow::ensure!(
            target_n_frames < 200_000,
            "phone {} stretch is implausibly long ({target_n_frames} frames)",
            phone.phoneme
        );

        let t_target_ms: Vec<f64> = (0..target_n_frames)
            .map(|i| i as f64 * frame_ms + frame_ms / 2.0)
            .collect();
        let source_clock: Vec<f64> = t_target_ms.iter().map(|t| t + source_skip_ms).collect();
        let mut mapped = vec![0f64; source_clock.len()];
        let mut present = vec![false; source_clock.len()];

        let use_hold = stretch_length_ms > 1.0
            && vowel_output_ms > stretch_length_ms * 2.5;
        let onset_src = stretch_length_ms * 0.45;
        let onset_tgt = onset_src * 2.5;
        for (i, &clock) in source_clock.iter().enumerate() {
            let is_consonant = clock < con_vel;
            let consonant_mapped = start_ms + clock / vel;
            let mut vowel_mapped = con_ms + (clock - con_vel) / scaling_ratio;
            if use_hold {
                let rel = clock - con_vel;
                let in_onset = rel < onset_tgt;
                let onset_map = con_ms + (rel / onset_tgt.max(1e-9)) * onset_src;
                let hold_tgt = (vowel_output_ms - onset_tgt).max(1e-9);
                let hold_rel = (rel - onset_tgt).max(0.0);
                let drift = (hold_rel / hold_tgt).min(1.0);
                let hold_map = con_ms + stretch_length_ms * (0.45 + drift * 0.52);
                vowel_mapped = if in_onset { onset_map } else { hold_map };
            }
            let value = if is_consonant {
                consonant_mapped
            } else {
                vowel_mapped
            };
            let hi = start_ms.max(end_ms - origin_frame_ms);
            mapped[i] = value.clamp(start_ms, hi);
            present[i] = clock >= 0.0;
        }

        let target_n = target_n_frames as usize;
        let mut mel = vec![0f32; n_mels * target_n];
        for band in 0..n_mels {
            let row = &mel_orig.data[band * orig_n_frames..(band + 1) * orig_n_frames];
            for (frame, &value) in mapped.iter().enumerate() {
                if present[frame] {
                    mel[band * target_n + frame] = interp(value, &t_orig_ms, row);
                }
            }
        }
        for frame in 0..target_n {
            if !present[frame] {
                for band in 0..n_mels {
                    mel[band * target_n + frame] = -80.0;
                }
            }
        }

        // ~80 ms tail silence so vocoder boundary artifacts land outside content.
        let extra_frames = (80.0 / frame_ms).round() as usize;
        if extra_frames > 0 {
            let mut extended = Vec::with_capacity(n_mels * (target_n + extra_frames));
            for band in 0..n_mels {
                extended.extend_from_slice(&mel[band * target_n..(band + 1) * target_n]);
                extended.extend(std::iter::repeat_n(-80.0, extra_frames));
            }
            let fade_n = extra_frames.min(3);
            let last: Vec<f32> = (0..n_mels).map(|b| mel[b * target_n + target_n - 1]).collect();
            for k in 0..fade_n {
                let w = if fade_n > 1 {
                    1.0 - k as f32 / (fade_n - 1) as f32
                } else {
                    1.0
                };
                for band in 0..n_mels {
                    extended[band * (target_n + extra_frames) + target_n + k] =
                        last[band] * w + (-80.0) * (1.0 - w);
                }
            }
            mel = extended;
            target_n_frames += extra_frames as i64;
        }
        let mel = MelMatrix {
            n_mels,
            frames: target_n_frames as usize,
            data: mel,
        };

        let mut f0 = vec![0f32; target_n_frames as usize];
        let copy_n = ctx.f0_slice.len().min(target_n);
        f0[..copy_n].copy_from_slice(&ctx.f0_slice[..copy_n]);
        if extra_frames > 0 {
            let tail = target_n_frames as usize - extra_frames;
            f0[tail..].fill(0.0);
        }

        Ok(PhoneFeature {
            index,
            phoneme: phone.phoneme.clone(),
            start_frame: ctx.start_frame,
            start_frame_exact: ctx.start_frame_exact,
            mel,
            f0,
            preutter_ms: ctx.preutter_ms,
            overlap_ms: ctx.overlap_ms,
            pre_vel_ms: ctx.pre_vel_ms,
            source_skip_ms: ctx.source_skip_ms,
            consonant_output_ms: con_output_ms,
            envelope_span_ms,
            segment_len_ms: total_span_ms + extra_frames as f64 * frame_ms,
            tail_silence_frames: extra_frames,
            pre_scale,
            debug: capture_debug.then(|| PhoneDebug {
                source_wave: wave.as_ref().clone(),
                mel_orig,
            }),
        })
    }
}

/// `np.interp` for a monotonically increasing `xp`.
fn interp(x: f64, xp: &[f64], fp: &[f32]) -> f32 {
    if xp.is_empty() {
        return 0.0;
    }
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    let mut lo = 0usize;
    let mut hi = xp.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xp[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    (fp[lo] as f64 * (1.0 - t) + fp[hi] as f64 * t) as f32
}

#[cfg(test)]
mod tests {
    use super::interp;

    #[test]
    fn interp_matches_numpy_endpoints() {
        let xp = [0.5, 1.5, 2.5];
        let fp = [10.0, 20.0, 30.0];
        assert_eq!(interp(-1.0, &xp, &fp), 10.0);
        assert_eq!(interp(0.5, &xp, &fp), 10.0);
        assert_eq!(interp(1.0, &xp, &fp), 15.0);
        assert_eq!(interp(2.5, &xp, &fp), 30.0);
        assert_eq!(interp(9.0, &xp, &fp), 30.0);
    }
}
