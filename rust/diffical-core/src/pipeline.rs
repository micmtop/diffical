//! Model-path render orchestration:
//! pitch -> per-phone mel -> assembler -> vocoder -> phrase post-processing.

use crate::assemble::{build_inputs, seg_norm_from_env, Assembler};
use crate::config::AudioConfig;
use crate::mel::MelFrontend;
use crate::phone::{MelMatrix, PhoneFeature, PhoneProcessor};
use crate::pitch::{build_pitch_curve, PitchCurve};
use crate::protocol::RenderRequest;
use crate::vocoder::{
    align_duration, gap_gate, normalize_phrase_rms, phrase_edge_fade, sanitize, NormalizationTrace,
    Vocoder,
};
use anyhow::{Context, Result};
use std::path::Path;

/// Pitch curve + source mel construction, usable without any ONNX model.
pub struct PhoneStage {
    pub cfg: AudioConfig,
    analyzer: MelFrontend,
}

impl PhoneStage {
    pub fn new(cfg: AudioConfig) -> Result<Self> {
        let analyzer = MelFrontend::new(
            cfg.sample_rate,
            cfg.n_fft,
            cfg.win_size,
            cfg.origin_hop_size,
            cfg.n_mels,
            cfg.mel_fmin,
            cfg.mel_fmax,
        )?;
        Ok(Self { cfg, analyzer })
    }

    pub fn build(
        &self,
        req: &RenderRequest,
        capture_debug: bool,
    ) -> Result<(PitchCurve, Vec<PhoneFeature>)> {
        let pitch = build_pitch_curve(req, &self.cfg)?;
        let mut processor = PhoneProcessor::new(&self.cfg, &self.analyzer, &req.singer_path);
        let features = processor.features(req, &pitch.f0, capture_debug)?;
        Ok((pitch, features))
    }
}

pub struct Rendered {
    pub samples: Vec<f32>,
    pub normalization: NormalizationTrace,
    pub total_frames: usize,
    pub mel_phrase: MelMatrix,
    pub f0: Vec<f32>,
    pub features: Vec<PhoneFeature>,
}

pub struct RenderEngine {
    pub cfg: AudioConfig,
    pub phone: PhoneStage,
    assembler: Assembler,
    vocoder: Vocoder,
    seg_norm: bool,
}

impl RenderEngine {
    pub fn new(assembler_path: &Path, vocoder_path: &Path) -> Result<Self> {
        let cfg = AudioConfig::default();
        Self::with_config(cfg, assembler_path, vocoder_path)
    }

    pub fn with_config(
        cfg: AudioConfig,
        assembler_path: &Path,
        vocoder_path: &Path,
    ) -> Result<Self> {
        let phone = PhoneStage::new(cfg.clone())?;
        let env = ort::environment::init()
            .build()
            .context("create ort environment")?;
        let assembler = Assembler::load_with_env(&env, assembler_path, cfg.n_mels)?;
        let vocoder = Vocoder::load_with_env(&env, vocoder_path, cfg.n_mels)?;
        Ok(Self {
            cfg,
            phone,
            assembler,
            vocoder,
            seg_norm: seg_norm_from_env(),
        })
    }

    pub fn render(&mut self, req: &RenderRequest, capture_debug: bool) -> Result<Rendered> {
        let timing = std::env::var_os("DIFFICAL_TIMING").is_some();
        let stage = std::time::Instant::now();

        let (pitch, features) = self.phone.build(req, capture_debug)?;
        if timing {
            eprintln!("[timing] phones: {:.1} ms", stage.elapsed().as_secs_f64() * 1000.0);
        }
        let stage = std::time::Instant::now();
        let total_frames = pitch.total_frames;
        let inputs = build_inputs(
            &features,
            total_frames,
            &pitch.f0,
            self.cfg.n_mels,
            self.seg_norm,
        )?;
        let mel_phrase = self.assembler.assemble(inputs, total_frames)?;
        if timing {
            eprintln!(
                "[timing] assemble: {:.1} ms",
                stage.elapsed().as_secs_f64() * 1000.0
            );
        }

        let stage = std::time::Instant::now();
        let mut samples = self
            .vocoder
            .synthesize_phrase(&mel_phrase, &pitch.f0, total_frames, &self.cfg)?;
        if timing {
            eprintln!(
                "[timing] vocode: {:.1} ms",
                stage.elapsed().as_secs_f64() * 1000.0
            );
        }
        let stage = std::time::Instant::now();
        sanitize(&mut samples);
        gap_gate(&mut samples, &features, total_frames, &self.cfg);
        sanitize(&mut samples);

        let expected_samples =
            (((req.duration_ms + req.leading_ms) / 1000.0) * self.cfg.sample_rate as f64)
                .round() as usize;
        align_duration(&mut samples, expected_samples);
        phrase_edge_fade(&mut samples);

        let volume_reference = req.phone_volume_reference();
        let normalization = if volume_reference > 0.0 {
            normalize_phrase_rms(
                &mut samples,
                self.cfg.sample_rate,
                0.15 * volume_reference,
                0.95,
            )
        } else {
            samples.fill(0.0);
            NormalizationTrace {
                reason: "zero_volume",
                gain: 0.0,
                activity_threshold: 0.0,
                active_rms_before: 0.0,
                peak_before: 0.0,
                active_windows: 0,
                total_windows: 0,
            }
        };
        if timing {
            eprintln!(
                "[timing] post: {:.1} ms",
                stage.elapsed().as_secs_f64() * 1000.0
            );
        }

        Ok(Rendered {
            samples,
            normalization,
            total_frames,
            mel_phrase,
            f0: pitch.f0,
            features,
        })
    }
}
