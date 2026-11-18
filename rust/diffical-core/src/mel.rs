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
    window: Vec<f32>,
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
        let window = (0..params.win_size)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / params.win_size as f32).cos())
            .collect();
        Ok(Self { params, basis, window })
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
        let mut mags = vec![0f32; bins * frames];
        for t in 0..frames {
            let start = t * p.hop;
            for i in 0..p.win_size {
                spec[i] = Complex32::new(padded[start + i] * self.window[i], 0.0);
            }
            for i in p.win_size..p.n_fft {
                spec[i] = Complex32::default();
            }
            fft.process(&mut spec);
            for k in 0..bins {
                mags[k * frames + t] = spec[k].norm();
            }
        }
        let mut mel = vec![0f32; p.n_mels * frames];
        for m in 0..p.n_mels {
            let row = &self.basis[m * bins..(m + 1) * bins];
            for t in 0..frames {
                let mut acc = 0f32;
                for k in 0..bins {
                    acc += row[k] * mags[k * frames + t];
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
