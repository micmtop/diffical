use anyhow::{Context, Result};
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct MelParams {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_size: usize,
    pub hop: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
    pub pad_left: usize,
    pub pad_right: usize,
}

pub struct MelFrontend {
    params: MelParams,
    basis: Vec<f32>, // n_mels x (n_fft/2 + 1), row-major
    /// First/last non-zero bin of each band. The triangular filterbank is
    /// sparse (~2% dense), and the dense projection dominated render time.
    spans: Vec<(usize, usize)>,
    window: Vec<f32>,
}

fn basis_spans(basis: &[f32], n_mels: usize, bins: usize) -> Vec<(usize, usize)> {
    (0..n_mels)
        .map(|m| {
            let row = &basis[m * bins..(m + 1) * bins];
            let start = row.iter().position(|v| *v != 0.0).unwrap_or(0);
            let end = row.iter().rposition(|v| *v != 0.0).map_or(0, |i| i + 1);
            (start, end)
        })
        .collect()
}

/// Slaney mel scale, matching `librosa.hz_to_mel` with `htk=False`.
fn hz_to_mel(hz: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = 6.4_f64.ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        min_log_mel + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        hz / F_SP
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = 6.4_f64.ln() / 27.0;
    if mel >= min_log_mel {
        MIN_LOG_HZ * (logstep * (mel - min_log_mel)).exp()
    } else {
        F_SP * mel
    }
}

/// `librosa.filters.mel(..., htk=False, norm="slaney")`, cast to float32.
pub fn slaney_mel_basis(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let mel_lo = hz_to_mel(fmin as f64);
    let mel_hi = hz_to_mel(fmax as f64);
    let mut mel_f = Vec::with_capacity(n_mels + 2);
    for i in 0..=n_mels + 1 {
        let t = i as f64 / (n_mels + 1) as f64;
        mel_f.push(mel_to_hz(mel_lo + (mel_hi - mel_lo) * t));
    }
    let fft_freqs: Vec<f64> = (0..bins)
        .map(|k| k as f64 * sample_rate as f64 / n_fft as f64)
        .collect();
    let mut basis = vec![0f32; n_mels * bins];
    for m in 0..n_mels {
        let left = mel_f[m];
        let center = mel_f[m + 1];
        let right = mel_f[m + 2];
        let lower_slope = center - left;
        let upper_slope = right - center;
        let enorm = 2.0 / (right - left);
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - left) / lower_slope;
            let upper = (right - freq) / upper_slope;
            let weight = lower.min(upper).max(0.0) * enorm;
            basis[m * bins + k] = weight as f32;
        }
    }
    basis
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos())
        .collect()
}

fn reflect_pad(x: &[f32], left: usize, right: usize) -> Result<Vec<f32>> {
    let n = x.len();
    if n == 0 {
        anyhow::bail!("empty input");
    }
    if left >= n || right >= n {
        anyhow::bail!("reflect pad larger than signal (n={n}, left={left}, right={right})");
    }
    let mut out = Vec::with_capacity(n + left + right);
    for i in 0..left {
        out.push(x[left - i]);
    }
    out.extend_from_slice(x);
    for i in left + n..left + n + right {
        out.push(x[2 * n - 2 - (i - left)]);
    }
    Ok(out)
}

impl MelFrontend {
    /// Load a filterbank dumped by the Python reference plus its parameters.
    pub fn load(basis_path: &Path, params_path: &Path) -> Result<Self> {
        let params: MelParams = serde_json::from_str(
            &fs::read_to_string(params_path).with_context(|| format!("read {params_path:?}"))?,
        )?;
        let bytes = fs::read(basis_path).with_context(|| format!("read {basis_path:?}"))?;
        let basis: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let bins = params.n_fft / 2 + 1;
        anyhow::ensure!(
            basis.len() == params.n_mels * bins,
            "basis size {} != {} x {}",
            basis.len(),
            params.n_mels,
            bins
        );
        let window = hann_window(params.win_size);
        let spans = basis_spans(&basis, params.n_mels, params.n_fft / 2 + 1);
        Ok(Self {
            params,
            basis,
            spans,
            window,
        })
    }

    /// Build an analyzer from the signal parameters instead of dumped goldens.
    /// `hop` selects the time resolution (512 for the vocoder, 128 for source mel).
    pub fn new(
        sample_rate: u32,
        n_fft: usize,
        win_size: usize,
        hop: usize,
        n_mels: usize,
        fmin: f32,
        fmax: f32,
    ) -> Result<Self> {
        anyhow::ensure!(win_size <= n_fft, "win_size {win_size} > n_fft {n_fft}");
        anyhow::ensure!(hop > 0 && win_size > hop, "invalid hop {hop} for win {win_size}");
        let basis = slaney_mel_basis(sample_rate, n_fft, n_mels, fmin, fmax);
        let params = MelParams {
            sample_rate,
            n_fft,
            win_size,
            hop,
            n_mels,
            fmin,
            fmax,
            pad_left: (win_size - hop) / 2,
            pad_right: (win_size - hop + 1) / 2,
        };
        let window = hann_window(win_size);
        let spans = basis_spans(&basis, n_mels, n_fft / 2 + 1);
        Ok(Self {
            params,
            basis,
            spans,
            window,
        })
    }

    pub fn params(&self) -> &MelParams {
        &self.params
    }

    /// Log-mel (natural log, floor 1e-9), row-major `n_mels x frames`.
    pub fn compute(&self, wav: &[f32]) -> Result<Vec<f32>> {
        let p = &self.params;
        let padded = reflect_pad(wav, p.pad_left, p.pad_right)?;
        if padded.len() < p.win_size {
            anyhow::bail!("signal too short for one frame");
        }
        let frames = 1 + (padded.len() - p.win_size) / p.hop;
        let bins = p.n_fft / 2 + 1;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(p.n_fft);
        let mut spec = vec![Complex32::default(); p.n_fft];
        let mut mags = vec![0f32; frames * bins];
        for t in 0..frames {
            let start = t * p.hop;
            for i in 0..p.win_size {
                spec[i] = Complex32::new(padded[start + i] * self.window[i], 0.0);
            }
            for i in p.win_size..p.n_fft {
                spec[i] = Complex32::default();
            }
            fft.process(&mut spec);
            let row = &mut mags[t * bins..(t + 1) * bins];
            for (k, slot) in row.iter_mut().enumerate() {
                *slot = spec[k].norm();
            }
        }
        self.project(&mags, frames, bins)
    }

    /// `g` formant shift: the reference scales the STFT window/FFT size by
    /// `2^(key_shift/12)`, crops/pads the magnitude spectrum back to the
    /// vocoder's 1025 bins and compensates amplitude by
    /// `win_size / win_size_new`. `key_shift == 0` is bit-identical to
    /// [`compute`](Self::compute).
    pub fn compute_with_key_shift(&self, wav: &[f32], key_shift: f64) -> Result<Vec<f32>> {
        if !key_shift.is_finite() || key_shift == 0.0 {
            return self.compute(wav);
        }
        let p = &self.params;
        anyhow::ensure!(
            key_shift.abs() <= 12.0,
            "key shift {key_shift} semitones is out of range"
        );
        let factor = 2f64.powf(key_shift / 12.0);
        let n_fft_new = (p.n_fft as f64 * factor).round().max(8.0) as usize;
        let win_new = (p.win_size as f64 * factor).round().max(8.0) as usize;
        anyhow::ensure!(win_new <= n_fft_new, "resized window exceeds FFT size");
        let pad_left = (win_new - p.hop) / 2;
        let pad_right = (win_new - p.hop + 1) / 2;
        let padded = reflect_pad(wav, pad_left, pad_right)?;
        anyhow::ensure!(
            padded.len() >= win_new,
            "signal too short for one {win_new}-sample window"
        );
        let frames = 1 + (padded.len() - win_new) / p.hop;
        let window = hann_window(win_new);
        let bins_new = n_fft_new / 2 + 1;
        let bins = p.n_fft / 2 + 1;
        let scale = p.win_size as f32 / win_new as f32;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft_new);
        let mut spec = vec![Complex32::default(); n_fft_new];
        let mut mags = vec![0f32; frames * bins];
        for t in 0..frames {
            let start = t * p.hop;
            for i in 0..win_new {
                spec[i] = Complex32::new(padded[start + i] * window[i], 0.0);
            }
            for i in win_new..n_fft_new {
                spec[i] = Complex32::default();
            }
            fft.process(&mut spec);
            let row = &mut mags[t * bins..(t + 1) * bins];
            for (k, slot) in row.iter_mut().enumerate() {
                *slot = if k < bins_new { spec[k].norm() * scale } else { 0.0 };
            }
        }
        self.project(&mags, frames, bins)
    }

    fn project(&self, mags: &[f32], frames: usize, bins: usize) -> Result<Vec<f32>> {
        let p = &self.params;
        let mut mel = vec![0f32; p.n_mels * frames];
        for t in 0..frames {
            let row = &mags[t * bins..(t + 1) * bins];
            for m in 0..p.n_mels {
                let (start, end) = self.spans[m];
                let weights = &self.basis[m * bins + start..m * bins + end];
                let mut acc = 0f32;
                for (w, mag) in weights.iter().zip(&row[start..end]) {
                    acc += w * mag;
                }
                mel[m * frames + t] = acc.max(1e-9).ln();
            }
        }
        Ok(mel)
    }
}

pub fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {path:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
