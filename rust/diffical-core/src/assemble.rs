//! Segment packing for the PointerGen assembler ONNX
//! (`backend/model_assembly.py::assemble` + `_process_phones` frame map).

use crate::inference::{IntTensorData, PointerOnnx, TensorData};
use crate::phone::{MelMatrix, PhoneFeature};
use anyhow::{Context, Result};
use std::path::Path;

const BOUNDARY_W: i64 = 4;
const MEL_FLOOR: f32 = -30.0;

pub struct AssemblerInputs {
    pub segs: Vec<f32>,
    pub seg_count: usize,
    pub lmax: usize,
    pub seg_len: Vec<i64>,
    pub seg_idx: Vec<i64>,
    pub progress: Vec<f32>,
    pub boundary: Vec<f32>,
    pub f0: Vec<f32>,
    pub total_frames: usize,
}

/// DIFFICAL_SEG_NORM (default on): trim each segment to the span `seg_idx`
/// actually uses and run `progress` from 0 to 1 over that span.
pub fn seg_norm_from_env() -> bool {
    let raw = std::env::var("DIFFICAL_SEG_NORM").unwrap_or_else(|_| "1".into());
    !matches!(raw.to_ascii_lowercase().as_str(), "0" | "false" | "no")
}

pub fn build_inputs(
    features: &[PhoneFeature],
    total_frames: usize,
    f0: &[f32],
    n_mels: usize,
    seg_norm: bool,
) -> Result<AssemblerInputs> {
    anyhow::ensure!(total_frames >= 1, "phrase has no frames");
    let mut ordered: Vec<&PhoneFeature> = features.iter().collect();
    ordered.sort_by_key(|f| f.start_frame);

    struct Prepared<'a> {
        a: usize,
        b: usize,
        mel: &'a MelMatrix,
    }
    let mut prepared: Vec<Prepared> = Vec::new();
    for feat in &ordered {
        if feat.mel.frames == 0 {
            continue;
        }
        if feat.mel.n_mels != n_mels {
            anyhow::bail!(
                "mel band count {} != {n_mels}",
                feat.mel.n_mels
            );
        }
        let a = (feat.start_frame_exact.round() as i64).clamp(0, total_frames as i64 - 1) as usize;
        let b = (a as i64 + feat.mel.frames as i64).min(total_frames as i64).max(0) as usize;
        if b.saturating_sub(a) < 2 {
            continue;
        }
        prepared.push(Prepared {
            a,
            b,
            mel: &feat.mel,
        });
    }
    if prepared.is_empty() {
        return Ok(AssemblerInputs {
            segs: Vec::new(),
            seg_count: 0,
            lmax: 0,
            seg_len: Vec::new(),
            seg_idx: vec![0; total_frames],
            progress: vec![0.0; total_frames],
            boundary: vec![0.0; total_frames],
            f0: sanitized_f0(f0, total_frames),
            total_frames,
        });
    }

    let mut segs: Vec<&MelMatrix> = Vec::with_capacity(prepared.len());
    let mut lens: Vec<usize> = Vec::with_capacity(prepared.len());
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(prepared.len());
    for (i, entry) in prepared.iter().enumerate() {
        // 完整窗口：不再裁到下一音素的音频起点。截断会让每个音素丢掉
        // 自己内容的大半，重叠区内容在进模型前就没了。
        let used = entry.b - entry.a;
        segs.push(entry.mel);
        lens.push(used);
        spans.push((entry.a, entry.a + used));
    }

    let seg_count = segs.len();
    let lmax = *lens.iter().max().context("segment lengths")?;
    let mut arr = vec![MEL_FLOOR; seg_count * n_mels * lmax];
    for (i, (mel, &len)) in segs.iter().zip(&lens).enumerate() {
        for band in 0..n_mels {
            let src = &mel.data[band * mel.frames..band * mel.frames + len];
            let dst = &mut arr[i * n_mels * lmax + band * lmax..i * n_mels * lmax + band * lmax + len];
            dst.copy_from_slice(src);
        }
    }

    let mut seg_idx = vec![0i64; total_frames];
    let mut progress = vec![0f32; total_frames];
    for (i, &(a, b)) in spans.iter().enumerate() {
        let len = b - a;
        if len == 0 {
            continue;
        }
        seg_idx[a..b].fill(i as i64 + 1);
        let denom = (len - 1).max(1) as f64;
        for (k, slot) in progress[a..b].iter_mut().enumerate() {
            *slot = (k as f64 / denom) as f32;
        }
        if len == 1 {
            progress[a] = 0.0;
        }
    }

    let mut boundary = vec![0f32; total_frames];
    for &(a, _) in spans.iter().skip(1) {
        let start = (a as i64 - BOUNDARY_W).max(0) as usize;
        let end = ((a as i64 + BOUNDARY_W + 1).min(total_frames as i64)).max(0) as usize;
        if start < end {
            boundary[start..end].fill(1.0);
        }
    }

    Ok(AssemblerInputs {
        segs: arr,
        seg_count,
        lmax,
        seg_len: lens.iter().map(|v| *v as i64).collect(),
        seg_idx,
        progress,
        boundary,
        f0: sanitized_f0(f0, total_frames),
        total_frames,
    })
}

fn sanitized_f0(f0: &[f32], total_frames: usize) -> Vec<f32> {
    let mut out = vec![0f32; total_frames];
    let n = f0.len().min(total_frames);
    out[..n].copy_from_slice(&f0[..n]);
    for value in &mut out {
        if *value > 0.0 && *value < 55.0 {
            *value = 0.0;
        }
    }
    out
}

pub struct Assembler {
    onnx: PointerOnnx,
    n_mels: usize,
}

impl Assembler {
    pub fn load(path: &Path, n_mels: usize) -> Result<Self> {
        Ok(Self {
            onnx: PointerOnnx::load(path)?,
            n_mels,
        })
    }

    pub fn load_with_env(
        env: &ort::environment::Environment,
        path: &Path,
        n_mels: usize,
    ) -> Result<Self> {
        Ok(Self {
            onnx: PointerOnnx::load_with_env(env, path)?,
            n_mels,
        })
    }

    pub fn assemble(
        &mut self,
        inputs: AssemblerInputs,
        total_frames: usize,
    ) -> Result<MelMatrix> {
        if inputs.seg_count == 0 {
            return Ok(MelMatrix {
                n_mels: self.n_mels,
                frames: total_frames,
                data: vec![MEL_FLOOR; self.n_mels * total_frames],
            });
        }
        let segs = TensorData {
            shape: vec![
                1,
                inputs.seg_count as i64,
                self.n_mels as i64,
                inputs.lmax as i64,
            ],
            data: inputs.segs,
        };
        let seg_len = IntTensorData {
            shape: vec![1, inputs.seg_count as i64],
            data: inputs.seg_len,
        };
        let seg_idx = IntTensorData {
            shape: vec![1, inputs.total_frames as i64],
            data: inputs.seg_idx,
        };
        let progress = TensorData {
            shape: vec![1, inputs.total_frames as i64],
            data: inputs.progress,
        };
        let boundary = TensorData {
            shape: vec![1, inputs.total_frames as i64],
            data: inputs.boundary,
        };
        let f0 = TensorData {
            shape: vec![1, inputs.total_frames as i64],
            data: inputs.f0,
        };
        let out = self
            .onnx
            .run(segs, seg_len, seg_idx, progress, boundary, f0)
            .context("assembler onnx run")?;
        anyhow::ensure!(
            out.shape.len() == 3 && out.shape[1] == self.n_mels as i64,
            "unexpected assembler output shape {:?}",
            out.shape
        );
        let frames = out.shape[2] as usize;
        anyhow::ensure!(
            frames == total_frames,
            "assembler returned {frames} frames, expected {total_frames}"
        );
        anyhow::ensure!(
            out.data.len() == self.n_mels * frames,
            "assembler output size mismatch"
        );
        Ok(MelMatrix {
            n_mels: self.n_mels,
            frames,
            data: out.data,
        })
    }
}
