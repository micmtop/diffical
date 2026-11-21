use anyhow::{bail, Context, Result};
use diffical_core::inference::{IntTensorData, PointerOnnx, TensorData};
use diffical_core::mel::read_f32;
use std::path::{Path, PathBuf};

mod flags_cmd;
mod inspect_cmd;
mod parity;
mod serve;
mod stats;

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

/// Remove `--name value` from the argument list and return the value.
fn take_value(args: &mut Vec<String>, name: &str) -> Option<String> {
    let flag = format!("--{name}");
    let pos = args.iter().position(|a| *a == flag)?;
    args.remove(pos);
    if pos < args.len() {
        Some(args.remove(pos))
    } else {
        None
    }
}

/// Remove a value-less switch.
fn take_switch(args: &mut Vec<String>, name: &str) -> bool {
    let flag = format!("--{name}");
    if let Some(pos) = args.iter().position(|a| *a == flag) {
        args.remove(pos);
        true
    } else {
        false
    }
}

fn take_positional(args: &mut Vec<String>, index: usize, usage: &str) -> Result<String> {
    if args.len() <= index {
        bail!("usage: {usage}");
    }
    Ok(args.remove(index))
}

fn model_path(explicit: Option<String>, env: &str) -> Result<PathBuf> {
    explicit
        .map(PathBuf::from)
        .or_else(|| std::env::var(env).ok().map(PathBuf::from))
        .with_context(|| format!("missing model path (CLI flag or ${env})"))
}

/// `--hnsep`, then `$DIFFICAL_HNSEP`, then `hnsep.onnx` next to the assembler.
fn resolve_hnsep(explicit: Option<String>, assembler: &Path) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("DIFFICAL_HNSEP") {
        return Some(PathBuf::from(path));
    }
    let candidate = assembler.parent()?.join("hnsep.onnx");
    candidate.is_file().then_some(candidate)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or_default();
    let mut rest = args[1..].to_vec();
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
            inspect_cmd::inspect(rest.first().context("usage: inspect <request.json>")?)?;
        }
        "phone-parity" => {
            let hnsep_flag = take_value(&mut rest, "hnsep");
            let usage = "phone-parity <request.json> <golden-dir> [--hnsep <onnx>]";
            let request = take_positional(&mut rest, 0, usage)?;
            let golden = take_positional(&mut rest, 0, usage)?;
            if !rest.is_empty() {
                bail!("usage: {usage}");
            }
            let hnsep = hnsep_flag.map(PathBuf::from);
            parity::phone_parity(&request, &golden, hnsep)?;
        }
        "assemble-parity" => {
            let hnsep_flag = take_value(&mut rest, "hnsep");
            let usage = "assemble-parity <request.json> <golden-dir> <assembler.onnx> \
                         [--hnsep <onnx>]";
            let request = take_positional(&mut rest, 0, usage)?;
            let golden = take_positional(&mut rest, 0, usage)?;
            let assembler = take_positional(&mut rest, 0, usage)?;
            if !rest.is_empty() {
                bail!("usage: {usage}");
            }
            let hnsep = hnsep_flag.map(PathBuf::from);
            parity::assemble_parity(&request, &golden, &assembler, hnsep)?;
        }
        "e2e-parity" => {
            let hnsep_flag = take_value(&mut rest, "hnsep");
            let usage = "e2e-parity <request.json> <golden-dir> <assembler.onnx> \
                         <vocoder.onnx> [--hnsep <onnx>]";
            let request = take_positional(&mut rest, 0, usage)?;
            let golden = take_positional(&mut rest, 0, usage)?;
            let assembler = take_positional(&mut rest, 0, usage)?;
            let vocoder = take_positional(&mut rest, 0, usage)?;
            if !rest.is_empty() {
                bail!("usage: {usage}");
            }
            parity::e2e_parity(
                &request,
                &golden,
                &assembler,
                &vocoder,
                hnsep_flag.map(PathBuf::from),
            )?;
        }
        "hnsep-parity" => {
            let golden = take_positional(&mut rest, 0, "hnsep-parity <golden-dir> <hnsep.onnx>")?;
            let model = take_positional(&mut rest, 0, "hnsep-parity <golden-dir> <hnsep.onnx>")?;
            if !rest.is_empty() {
                bail!("unexpected arguments: {rest:?}");
            }
            flags_cmd::hnsep_parity(&golden, &model)?;
        }
        "flag-parity" => {
            let inject = take_value(&mut rest, "inject-flags");
            let allow_mel_diff = take_switch(&mut rest, "allow-mel-diff");
            let allow_wav_diff = take_switch(&mut rest, "allow-wav-diff");
            let hnsep_flag = take_value(&mut rest, "hnsep");
            let usage = "flag-parity <request.json> <golden-dir> <assembler.onnx> \
                         <vocoder.onnx> [hnsep.onnx] [--inject-flags \"Hb200g100\"] \
                         [--allow-mel-diff] [--allow-wav-diff]";
            let request = take_positional(&mut rest, 0, usage)?;
            let golden = take_positional(&mut rest, 0, usage)?;
            let assembler = take_positional(&mut rest, 0, usage)?;
            let vocoder = take_positional(&mut rest, 0, usage)?;
            if rest.len() > 1 {
                bail!("usage: {usage}");
            }
            let hnsep = hnsep_flag.or_else(|| rest.first().cloned());
            let hnsep = resolve_hnsep(hnsep, Path::new(&assembler));
            flags_cmd::flag_parity(
                &request,
                &golden,
                &assembler,
                &vocoder,
                hnsep,
                inject.as_deref(),
                allow_mel_diff,
                allow_wav_diff,
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
            let hnsep_flag = take_value(&mut rest, "hnsep");
            let usage =
                "render <request.json> <out.wav> <assembler.onnx> <vocoder.onnx> [--hnsep <onnx>]";
            let request = take_positional(&mut rest, 0, usage)?;
            let out = take_positional(&mut rest, 0, usage)?;
            let assembler = take_positional(&mut rest, 0, usage)?;
            let vocoder = take_positional(&mut rest, 0, usage)?;
            if !rest.is_empty() {
                bail!("usage: {usage}");
            }
            let hnsep = resolve_hnsep(hnsep_flag, Path::new(&assembler));
            parity::render_to_wav(&request, &out, &assembler, &vocoder, hnsep)?;
        }
        "serve" => {
            let host = take_value(&mut rest, "host").unwrap_or_else(|| "127.0.0.1".to_string());
            let port: u16 = take_value(&mut rest, "port")
                .map(|v| v.parse().context("--port must be a u16"))
                .transpose()?
                .unwrap_or(8573);
            let assembler = model_path(take_value(&mut rest, "assembler"), "DIFFICAL_ASSEMBLER")?;
            let vocoder = model_path(take_value(&mut rest, "vocoder"), "DIFFICAL_VOCODER")?;
            let hnsep = resolve_hnsep(take_value(&mut rest, "hnsep"), &assembler);
            if !rest.is_empty() {
                bail!("unexpected arguments: {rest:?}");
            }
            serve::run(&host, port, &assembler, &vocoder, hnsep)?;
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
  diffical-cli hnsep-parity <golden-dir> <hnsep.onnx>
  diffical-cli flag-parity <request.json> <golden-dir> <assembler.onnx> <vocoder.onnx> \\
      [hnsep.onnx] [--inject-flags \\\"Hb200Hv120Ht30g100Dt-50\\\"] \\
      [--allow-mel-diff] [--allow-wav-diff]
  diffical-cli render <request.json> <out.wav> <assembler.onnx> <vocoder.onnx> [--hnsep <onnx>]
  diffical-cli serve [--host 127.0.0.1] [--port 8573] --assembler <onnx> --vocoder <onnx> \\
      [--hnsep <onnx>]

HN-SEP model path: --hnsep, $DIFFICAL_HNSEP, or hnsep.onnx next to the assembler."
            );
        }
        other => bail!("unknown command {other:?}; run diffical-cli --help"),
    }
    Ok(())
}
