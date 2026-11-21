//! Per-phone source window, stretch mapping and mel construction.
//!
//! Port of `backend/renderer.py::PhoneProcessor.process` for the model path
//! with flags/curves disabled: peak guard, hop-128 source mel, oto window,
//! velocity handling, envelope span, vowel stretch mapping, tail silence.

use crate::config::AudioConfig;
use crate::effects::tension_tilt;
use crate::flags::{HnsepParams, PhoneFlags};
use crate::hnsep::{Hnsep, HnsepCache, Separation, SourceKey};
use crate::mel::MelFrontend;
use crate::protocol::{RenderPhone, RenderRequest};
pub use crate::source::{interp, MelMatrix};
use crate::source::SourceLoader;
use crate::stretch::StretchMap;
use anyhow::{Context, Result};
use std::sync::Arc;

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
    /// Parsed standard flags for this phone.
    pub flags: PhoneFlags,
    /// `g` formant shift in semitones applied to the source mel.
    pub key_shift: f64,
    /// `t` pitch offset in cents applied to the phrase F0.
    pub pitch_cents: f64,
    /// Frames this phone covers in the phrase for phrase-wide effects
    /// (`HG` growl, `A` amplitude), excluding the 80 ms tail.
    pub effect_span_frames: usize,
    pub debug: Option<PhoneDebug>,
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
    hnsep: Option<&'a mut Hnsep>,
    hnsep_cache: Option<&'a HnsepCache>,
}

impl<'a> PhoneProcessor<'a> {
    pub fn new(
        cfg: &'a AudioConfig,
        analyzer: &'a MelFrontend,
        singer_path: &str,
        hnsep: Option<&'a mut Hnsep>,
        hnsep_cache: Option<&'a HnsepCache>,
    ) -> Self {
        Self {
            cfg,
            analyzer,
            loader: SourceLoader::new(cfg.sample_rate, singer_path),
            hnsep,
            hnsep_cache,
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
        let flags = PhoneFlags::from_values(&phone.flags);
        let key_shift = flags.gender_key_shift();
        let pitch_cents = flags.pitch_cents();
        let effect_span_frames = ((ctx.total_span_ms / frame_ms).round() as i64).max(1) as usize;

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
                flags: flags.clone(),
                key_shift,
                pitch_cents,
                effect_span_frames,
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
        let source_frames = if padded_len >= params.win_size {
            let frames = 1 + (padded_len - params.win_size) / params.hop;
            anyhow::ensure!(
                frames < 500_000,
                "source wav {:?} is too long for mel analysis ({frames} frames)",
                phone.oto_file
            );
            frames
        } else {
            0
        };

        // Source geometry of this phone. The stretch mapping only reads source
        // times in [start_ms, end_ms], so this is also the wave window that
        // HN-SEP has to process (plus one STFT window of context on each side).
        let offset_ms = finite_or(phone.oto_offset, 0.0);
        let consonant_ms = finite_or(phone.oto_consonant, 0.0);
        let cutoff_ms = finite_or(phone.oto_cutoff, 0.0);
        let total_time_ms = source_frames as f64 * origin_frame_ms;
        let t_orig_ms: Vec<f64> = (0..source_frames)
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

        // HN-SEP (`Hb`/`Hv`/`Ht`) works on the source window the phone reads,
        // before the mel: start/end of the stretch window plus one STFT window
        // of context on each side. Windows are cached by (path, mtime, start,
        // end); a cache hit reuses the shared harmonic/noise for that window.
        //
        // Production default is window-only separation: it separates just the
        // window the phone reads, which is ~7x faster on first render. Compared
        // with whole-file separation the output waveform differs by NCC 0.99992
        // and 0.11% RMS, so the shorter mask context is not audible.
        // `DIFFICAL_HNSEP_WHOLE=1` restores whole-file separation for exact
        // comparison against the Python reference (slower, more memory).
        let hnsep_params = flags.hnsep();
        let tension = flags.tension();
        let mut processed: Option<Vec<f32>> = None;
        if hnsep_params.is_some() {
            let Some(session) = self.hnsep.as_deref_mut() else {
                anyhow::bail!(
                    "phone {:?} requests HN-SEP flags but no HN-SEP model is configured (--hnsep <onnx>)",
                    phone.phoneme
                );
            };
            let whole_mode = std::env::var("DIFFICAL_HNSEP_WHOLE")
                .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
                .unwrap_or(false);
            let (w0, w1) = if whole_mode {
                (0, wave.len())
            } else {
                let sr = cfg.sample_rate as f64;
                let first = ((start_ms / 1000.0) * sr).floor().max(0.0) as usize;
                let last =
                    (((end_ms / 1000.0) * sr).ceil().max(0.0) as usize).min(wave.len());
                // One STFT window of context each side covers every mel frame
                // the stretch mapping reads.
                let guard = std::env::var("DIFFICAL_HNSEP_GUARD")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(cfg.n_fft);
                (first.saturating_sub(guard), (last + guard).min(wave.len()))
            };
            if w1 > w0 {
                let key = self
                    .loader
                    .resolved_path(&phone.oto_file)
                    .and_then(|path| SourceKey::new(&path, w0, w1, cfg.sample_rate));
                let separation = session.separate_cached(
                    self.hnsep_cache,
                    key.as_ref(),
                    &wave[w0..w1],
                )?;
                let mixed = mix_hnsep(&separation, hnsep_params.expect("checked"));
                if w0 == 0 && w1 == wave.len() {
                    processed = Some(mixed);
                } else {
                    let mut out = wave.as_ref().clone();
                    out[w0..w1].copy_from_slice(&mixed);
                    processed = Some(out);
                }
            }
        }
        if tension != 0.0 {
            let base = processed.take().unwrap_or_else(|| wave.as_ref().clone());
            processed = Some(tension_tilt(&base, tension, cfg.sample_rate));
        }
        let analyzer_wave: &[f32] = processed.as_deref().unwrap_or_else(|| wave.as_ref());
        let mel_orig = MelMatrix::from_vec(
            n_mels,
            self.analyzer.compute_with_key_shift(analyzer_wave, key_shift)?,
        )
        .context("source mel")?;
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
                flags: flags.clone(),
                key_shift,
                pitch_cents,
                effect_span_frames,
                debug: capture_debug.then(|| PhoneDebug {
                    source_wave: analyzer_wave.to_vec(),
                    mel_orig: mel_orig.clone(),
                }),
            });
        }


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
        // `He`: replace the frozen long-note hold with a ping-pong traversal of
        // the stable part of the source vowel. Active exactly where the hold
        // would be used and only when the source vowel is long enough (>= 200 ms).
        let use_loop = flags.long_note_loop() && use_hold && stretch_length_ms >= 200.0;
        let mapping = StretchMap {
            vel,
            start_ms,
            con_ms,
            con_vel,
            scaling_ratio,
            stretch_length_ms,
            vowel_output_ms,
            use_hold,
            use_loop,
            loop_start_ms: con_ms + stretch_length_ms * 0.35,
            loop_span_ms: stretch_length_ms * 0.57,
            loop_cycle_ms: 2.0 * stretch_length_ms.max(1500.0),
            onset_src: stretch_length_ms * 0.45,
            onset_tgt: stretch_length_ms * 0.45 * 2.5,
        };
        let hi = start_ms.max(end_ms - origin_frame_ms);
        for (i, &clock) in source_clock.iter().enumerate() {
            mapped[i] = mapping.source_ms(clock).clamp(start_ms, hi);
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
            flags,
            key_shift,
            pitch_cents,
            effect_span_frames,
            debug: capture_debug.then(|| PhoneDebug {
                source_wave: analyzer_wave.to_vec(),
                mel_orig,
            }),
        })
    }
}

/// Reference HN-SEP mix: `(1 - breath/100) * harmonic + (voicing/100) * noise`.
/// At default parameters this is `wave`.
fn mix_hnsep(separation: &Separation, params: HnsepParams) -> Vec<f32> {
    let harmonic_gain = 1.0 - params.breath / 100.0;
    let noise_gain = params.voicing / 100.0;
    separation
        .harmonic
        .iter()
        .zip(separation.noise.iter())
        .map(|(h, n)| (harmonic_gain * *h as f64 + noise_gain * *n as f64) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::interp;

}
