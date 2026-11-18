use anyhow::{bail, Context, Result};
use diffical_core::inference::{IntTensorData, PointerOnnx, TensorData};
use diffical_core::mel::read_f32;
use std::path::PathBuf;

fn read_int_tensor(dir: &std::path::Path, name: &str) -> Result<IntTensorData> {
    let shape: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json")))?)?;
    let shape: Vec<i64> = shape["shape"]
        .as_array()
        .context("shape")?
        .iter()
        .map(|v| v.as_i64().context("dim"))
        .collect::<Result<_>>()?;
    let data = read_f32(&dir.join(format!("{name}.f32")))?
        .into_iter()
        .map(|v| v as i64)
        .collect();
    Ok(IntTensorData { shape, data })
}

fn read_tensor(dir: &std::path::Path, name: &str) -> Result<TensorData> {
    let shape: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json")))?)?;
    let shape: Vec<i64> = shape["shape"]
        .as_array()
        .context("shape")?
        .iter()
        .map(|v| v.as_i64().context("dim"))
        .collect::<Result<_>>()?;
    let data = read_f32(&dir.join(format!("{name}.f32")))?;
    Ok(TensorData { shape, data })
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    match cmd.as_str() {
        "mel-parity" => {
            let dir = PathBuf::from(args.next().context("missing golden dir")?);
            let frontend = diffical_core::MelFrontend::load(&dir.join("basis.f32"), &dir.join("params.json"))?;
            let wav = read_f32(&dir.join("wav.f32"))?;
            let got = frontend.compute(&wav)?;
            let want = read_f32(&dir.join("mel.f32"))?;
            if got.len() != want.len() {
                bail!("length mismatch: got {} want {}", got.len(), want.len());
            }
            let max = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let sum: f64 = got.iter().zip(&want).map(|(a, b)| (a - b).abs() as f64).sum();
            println!("values={} max_abs_diff={max:.3e} mean_abs_diff={:.3e}", got.len(), sum / got.len() as f64);
            if max > 1e-4 {
                bail!("mel parity failed: {max:.3e}");
            }
        }
        "onnx-parity" => {
            let golden = PathBuf::from(args.next().context("missing golden_onnx dir")?);
            let model = PathBuf::from(args.next().context("missing model path")?);
            let mut onnx = PointerOnnx::load(&model)?;
            let out = onnx.run(
                read_tensor(&golden, "segs")?,
                read_int_tensor(&golden, "seg_len")?,
                read_int_tensor(&golden, "seg_idx")?,
                read_tensor(&golden, "progress")?,
                read_tensor(&golden, "boundary")?,
                read_tensor(&golden, "f0")?,
            )?;
            let want = read_f32(&golden.join("mel.f32"))?;
            if out.data.len() != want.len() {
                bail!("length mismatch: got {} want {}", out.data.len(), want.len());
            }
            let max = out.data.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let sum: f64 = out.data.iter().zip(&want).map(|(a, b)| (a - b).abs() as f64).sum();
            println!("shape={:?} max_abs_diff={max:.3e} mean_abs_diff={:.3e}", out.shape, sum / out.data.len() as f64);
            if max > 1e-4 {
                bail!("onnx parity failed: {max:.3e}");
            }
        }
        _ => bail!("usage: diffical-cli [mel-parity|onnx-parity] ..."),
    }
    Ok(())
}
