use anyhow::{bail, Context, Result};
use diffical_core::inference::{IntTensorData, PointerOnnx, TensorData};
use diffical_core::mel::read_f32;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

mod parity;
mod serve;

fn read_int_tensor(dir: &Path, name: &str) -> Result<IntTensorData> {
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

fn read_tensor(dir: &Path, name: &str) -> Result<TensorData> {
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

fn flags(args: &[String]) -> Result<HashMap<String, String>> {
    let mut out = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let Some(name) = arg.strip_prefix("--") else {
            bail!("unexpected positional argument {arg:?}");
        };
        let value = args
            .get(i + 1)
            .with_context(|| format!("missing value for --{name}"))?;
        out.insert(name.to_string(), value.clone());
        i += 2;
    }
    Ok(out)
}

fn model_path(flags: &HashMap<String, String>, name: &str, env: &str) -> Result<PathBuf> {
    flags
        .get(name)
        .map(PathBuf::from)
        .or_else(|| std::env::var(env).ok().map(PathBuf::from))
        .with_context(|| format!("missing --{name} (or ${env})"))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or_default();
    let rest = &args[1..];
    match cmd {
        "mel-parity" => {
            let dir = PathBuf::from(rest.first().context("missing golden dir")?);
            let frontend =
                diffical_core::MelFrontend::load(&dir.join("basis.f32"), &dir.join("params.json"))?;
            let wav = read_f32(&dir.join("wav.f32"))?;
            let got = frontend.compute(&wav)?;
            let want = read_f32(&dir.join("mel.f32"))?;
            if got.len() != want.len() {
                bail!("length mismatch: got {} want {}", got.len(), want.len());
            }
            let max = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let sum: f64 = got.iter().zip(&want).map(|(a, b)| (a - b).abs() as f64).sum();
            println!(
                "values={} max_abs_diff={max:.3e} mean_abs_diff={:.3e}",
                got.len(),
                sum / got.len() as f64
            );
            if max > 1e-4 {
                bail!("mel parity failed: {max:.3e}");
            }
        }
        "onnx-parity" => {
            let golden = PathBuf::from(rest.first().context("missing golden_onnx dir")?);
            let model = PathBuf::from(rest.get(1).context("missing model path")?);
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
            let max = out
                .data
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let sum: f64 = out
                .data
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs() as f64)
                .sum();
            println!(
                "shape={:?} max_abs_diff={max:.3e} mean_abs_diff={:.3e}",
                out.shape,
                sum / out.data.len() as f64
            );
            if max > 1e-4 {
                bail!("onnx parity failed: {max:.3e}");
            }
        }
        "inspect" => {
            parity::inspect(rest.first().context("usage: inspect <request.json>")?)?;
        }
        "phone-parity" => {
            parity::phone_parity(
                rest.first().context("usage: phone-parity <request.json> <golden-dir>")?,
                rest.get(1).context("missing golden dir")?,
            )?;
        }
        "assemble-parity" => {
            parity::assemble_parity(
                rest.first()
                    .context("usage: assemble-parity <request.json> <golden-dir> <assembler.onnx>")?,
                rest.get(1).context("missing golden dir")?,
                rest.get(2).context("missing assembler model")?,
            )?;
        }
        "e2e-parity" => {
            parity::e2e_parity(
                rest.first().context(
                    "usage: e2e-parity <request.json> <golden-dir> <assembler.onnx> <vocoder.onnx>",
                )?,
                rest.get(1).context("missing golden dir")?,
                rest.get(2).context("missing assembler model")?,
                rest.get(3).context("missing vocoder model")?,
            )?;
        }
        "response-parity" => {
            parity::response_parity(
                rest.first()
                    .context("usage: response-parity <render-response.json> <golden-dir>")?,
                rest.get(1).context("missing golden dir")?,
            )?;
        }
        "render" => {
            parity::render_to_wav(
                rest.first()
                    .context("usage: render <request.json> <out.wav> <assembler.onnx> <vocoder.onnx>")?,
                rest.get(1).context("missing output wav")?,
                rest.get(2).context("missing assembler model")?,
                rest.get(3).context("missing vocoder model")?,
            )?;
        }
        "serve" => {
            let opts = flags(rest)?;
            let host = opts
                .get("host")
                .cloned()
                .unwrap_or_else(|| "127.0.0.1".to_string());
            let port: u16 = opts
                .get("port")
                .map(|v| v.parse().context("--port must be a u16"))
                .transpose()?
                .unwrap_or(8573);
            let assembler = model_path(&opts, "assembler", "DIFFICAL_ASSEMBLER")?;
            let vocoder = model_path(&opts, "vocoder", "DIFFICAL_VOCODER")?;
            serve::run(&host, port, &assembler, &vocoder)?;
        }
        "" | "-h" | "--help" => {
            println!(
                "usage:
  diffical-cli mel-parity <golden-dir>
  diffical-cli onnx-parity <golden_onnx-dir> <assembler.onnx>
  diffical-cli inspect <request.json>
  diffical-cli phone-parity <request.json> <golden-dir>
  diffical-cli assemble-parity <request.json> <golden-dir> <assembler.onnx>
  diffical-cli e2e-parity <request.json> <golden-dir> <assembler.onnx> <vocoder.onnx>
  diffical-cli response-parity <render-response.json> <golden-dir>
  diffical-cli render <request.json> <out.wav> <assembler.onnx> <vocoder.onnx>
  diffical-cli serve [--host 127.0.0.1] [--port 8573] --assembler <onnx> --vocoder <onnx>"
            );
        }
        other => bail!("unknown command {other:?}; run diffical-cli --help"),
    }
    Ok(())
}
