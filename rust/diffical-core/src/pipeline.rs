//! Model-path render orchestration:
//! pitch -> per-phone mel (flags: HN-SEP / g / t / He) -> assembler -> vocoder
//! -> phrase effects (HG / A) -> post-processing.

use crate::assemble::{build_inputs, seg_norm_from_env, Assembler};
use crate::compose::{compose_envelope, geom_v3_from_env};
use crate::config::AudioConfig;
use crate::effects;
use crate::flags::PhoneFlags;
use crate::hnsep::{Hnsep, HnsepCache};
use crate::mel::MelFrontend;
use crate::phone::{MelMatrix, PhoneFeature, PhoneProcessor};
use crate::pitch::{build_pitch_curve, PitchCurve};
use crate::protocol::RenderRequest;
use crate::vocoder::{
    align_duration, gap_gate, normalize_phrase_rms, phrase_edge_fade, sanitize, NormalizationTrace,
    Vocoder,
};
use anyhow::{Context, Result};
use ort::environment::{init as env_init, Environment};
use std::path::{Path, PathBuf};

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
        &mut self,
        req: &RenderRequest,
        capture_debug: bool,
    ) -> Result<(PitchCurve, Vec<PhoneFeature>)> {
        self.build_with_hnsep(req, capture_debug, None, None)
    }

    /// `hnsep` is passed in (rather than owned) so the session survives across
    /// renders without making the flag-free path load a 59 MB model.
    pub fn build_with_hnsep(
        &mut self,
        req: &RenderRequest,
        capture_debug: bool,
        mut hnsep: Option<&mut Hnsep>,
        hnsep_cache: Option<&HnsepCache>,
    ) -> Result<(PitchCurve, Vec<PhoneFeature>)> {
        let pitch = build_pitch_curve(req, &self.cfg)?;
        let mut processor = PhoneProcessor::new(
            &self.cfg,
            &self.analyzer,
            &req.singer_path,
            hnsep.as_deref_mut(),
            hnsep_cache,
        );
        let mut features = processor.features(req, &pitch.f0, capture_debug)?;
        let mut f0 = pitch.f0.clone();
        apply_pitch_cents(&mut features, &mut f0);
        Ok((
            PitchCurve {
                total_frames: pitch.total_frames,
                f0,
            },
            features,
        ))
    }
}

/// `t`: add the phone's cent offset to its voiced F0 frames. Overlaps are
/// resolved in favour of the later phone (it owns the content there).
fn apply_pitch_cents(features: &mut [PhoneFeature], f0: &mut [f32]) {
    let mut cents = vec![0f64; f0.len()];
    let mut any = false;
    for feat in features.iter() {
        if feat.pitch_cents == 0.0 || !feat.pitch_cents.is_finite() {
            continue;
        }
        any = true;
        let start = feat.start_frame.max(0) as usize;
        if start >= cents.len() {
            continue;
        }
        let end = (start + feat.effect_span_frames).min(cents.len());
        cents[start..end].fill(feat.pitch_cents);
    }
    if !any {
        return;
    }
    for (i, value) in f0.iter_mut().enumerate() {
        if cents[i] != 0.0 && *value > 0.0 {
            *value = (*value as f64 * 2f64.powf(cents[i] / 1200.0)) as f32;
        }
    }
    for feat in features.iter_mut() {
        if feat.pitch_cents == 0.0 {
            continue;
        }
        let ratio = 2f64.powf(feat.pitch_cents / 1200.0) as f32;
        for value in &mut feat.f0 {
            if *value > 0.0 {
                *value *= ratio;
            }
        }
    }
}

/// Per-frame `HG` / `A` curves from the phone flags (later phones win overlaps).
fn phrase_effect_curves(
    features: &[PhoneFeature],
    total_frames: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut growl = vec![0f32; total_frames];
    let mut amplitude = vec![0f32; total_frames];
    for feat in features {
        let flags = &feat.flags;
        let start = feat.start_frame.max(0) as usize;
        if start >= total_frames {
            continue;
        }
        let end = (start + feat.effect_span_frames).min(total_frames);
        let growl_value = flags.growl() as f32;
        if growl_value > 0.5 {
            for slot in &mut growl[start..end] {
                *slot = slot.max(growl_value);
            }
        }
        let amplitude_value = flags.amplitude_mod() as f32;
        if amplitude_value != 0.0 {
            amplitude[start..end].fill(amplitude_value);
        }
    }
    (growl, amplitude)
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
    env: Environment,
    hnsep_path: Option<PathBuf>,
    hnsep: Option<Hnsep>,
    hnsep_cache: Option<HnsepCache>,
}

impl RenderEngine {
    pub fn new(assembler_path: &Path, vocoder_path: &Path) -> Result<Self> {
        Self::with_config_hnsep(AudioConfig::default(), assembler_path, vocoder_path, None)
    }

    pub fn with_config(
        cfg: AudioConfig,
        assembler_path: &Path,
        vocoder_path: &Path,
    ) -> Result<Self> {
        Self::with_config_hnsep(cfg, assembler_path, vocoder_path, None)
    }

    /// `hnsep_path` is the exported HN-SEP mask network; it is only loaded on
    /// the first render whose phones actually use `Hb`/`Hv`/`Ht`.
    pub fn with_hnsep(
        assembler_path: &Path,
        vocoder_path: &Path,
        hnsep_path: Option<PathBuf>,
    ) -> Result<Self> {
        Self::with_config_hnsep(
            AudioConfig::default(),
            assembler_path,
            vocoder_path,
            hnsep_path,
        )
    }

    pub fn with_config_hnsep(
        cfg: AudioConfig,
        assembler_path: &Path,
        vocoder_path: &Path,
        hnsep_path: Option<PathBuf>,
    ) -> Result<Self> {
        let phone = PhoneStage::new(cfg.clone())?;
        let env = env_init()
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
            env,
            hnsep_path,
            hnsep: None,
            hnsep_cache: None,
        })
    }

    fn ensure_hnsep(&mut self, req: &RenderRequest) -> Result<()> {
        if self.hnsep.is_some() {
            return Ok(());
        }
        let needs = req
            .phones
            .iter()
            .any(|p| PhoneFlags::from_values(&p.flags).hnsep().is_some());
        if !needs {
            return Ok(());
        }
        let path = self.hnsep_path.as_ref().context(
            "phone flags request HN-SEP (Hb/Hv/Ht) but no model path was given \
             (--hnsep <onnx> or DIFFICAL_HNSEP)",
        )?;
        let started = std::time::Instant::now();
        self.hnsep = Some(Hnsep::load_with_env(&self.env, path)?);
        self.hnsep_cache = Some(HnsepCache::default());
        if std::env::var_os("DIFFICAL_TIMING").is_some() {
            eprintln!(
                "[timing] hnsep load: {:.1} ms",
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(())
    }

    pub fn render(&mut self, req: &RenderRequest, capture_debug: bool) -> Result<Rendered> {
        let timing = std::env::var_os("DIFFICAL_TIMING").is_some();
        let stage = std::time::Instant::now();
        self.ensure_hnsep(req)?;

        let (pitch, features) = self.phone.build_with_hnsep(
            req,
            capture_debug,
            self.hnsep.as_mut(),
            self.hnsep_cache.as_ref(),
        )?;
        if timing {
            eprintln!(
                "[timing] phones: {:.1} ms",
                stage.elapsed().as_secs_f64() * 1000.0
            );
        }
        let stage = std::time::Instant::now();
        let total_frames = pitch.total_frames;
        let mel_phrase = if geom_v3_from_env() {
            compose_envelope(&features, &req.phones, total_frames, self.cfg.n_mels)?
        } else {
            let inputs = build_inputs(
                &features,
                total_frames,
                &pitch.f0,
                self.cfg.n_mels,
                self.seg_norm,
            )?;
            self.assembler.assemble(inputs, total_frames)?
        };
        if std::env::var_os("DIFFICAL_DUMP_STAGES").is_some() {
            dump_stage("mel", &mel_phrase.data, &pitch.f0, mel_phrase.frames, self.cfg.sample_rate)?;
        }
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
        if std::env::var_os("DIFFICAL_DUMP_STAGES").is_some() {
            dump_stage("voc", &samples, &[], 0, self.cfg.sample_rate)?;
        }
        let stage = std::time::Instant::now();
        sanitize(&mut samples);
        gap_gate(&mut samples, &features, total_frames, &self.cfg);
        sanitize(&mut samples);

        // Phrase effects: HG growl, then A pitch-following amplitude. The
        // flag-free path skips the curve allocation entirely.
        let has_effects = features
            .iter()
            .any(|f| f.flags.growl() > 0.5 || f.flags.amplitude_mod() != 0.0);
        if has_effects {
            let (growl_curve, amplitude_curve) = phrase_effect_curves(&features, total_frames);
            if let Some(processed) = effects::growl(
                &samples,
                &growl_curve,
                &pitch.f0,
                self.cfg.sample_rate,
                self.cfg.hop_size,
            ) {
                samples = processed;
            }
            if let Some(processed) = effects::pitch_amplitude(
                &samples,
                &amplitude_curve,
                &pitch.f0,
                self.cfg.sample_rate,
                self.cfg.hop_size,
            ) {
                samples = processed;
            }
        }

        let expected_samples =
            (((req.duration_ms + req.leading_ms) / 1000.0) * self.cfg.sample_rate as f64)
                .round() as usize;
        align_duration(&mut samples, expected_samples);
        phrase_edge_fade(&mut samples);

        let volume_reference = req.phone_volume_reference();
        let rms_target = std::env::var("DIFFICAL_RMS_TARGET")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| *v > 0.0 && *v < 1.0)
            .unwrap_or(0.13);
        let normalization = if volume_reference > 0.0 {
            normalize_phrase_rms(
                &mut samples,
                self.cfg.sample_rate,
                rms_target * volume_reference,
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

fn dump_stage(kind: &str, data: &[f32], f0: &[f32], frames: usize, sr: u32) -> Result<()> {
    let dir = std::env::temp_dir().join("diffical_stages");
    std::fs::create_dir_all(&dir)?;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let base = dir.join(format!("{ts}_{kind}"));
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(base.with_extension("f32"), bytes)?;
    if !f0.is_empty() {
        let fbytes: Vec<u8> = f0.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(base.with_extension("f0.f32"), fbytes)?;
    }
    let meta = format!("{{\"kind\":\"{kind}\",\"len\":{},\"frames\":{frames},\"sample_rate\":{sr}}}", data.len());
    std::fs::write(base.with_extension("json"), meta)?;
    Ok(())
}
