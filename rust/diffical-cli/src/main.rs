use anyhow::{bail, Context, Result};
use diffical_core::{mel::read_f32, MelFrontend};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    if cmd != "mel-parity" {
        bail!("usage: diffical-cli mel-parity <golden-dir>");
    }
    let dir = PathBuf::from(args.next().context("missing golden dir")?);
    let frontend = MelFrontend::load(&dir.join("basis.f32"), &dir.join("params.json"))?;
    let wav = read_f32(&dir.join("wav.f32"))?;
    let got = frontend.compute(&wav)?;
    let want = read_f32(&dir.join("mel.f32"))?;
    if got.len() != want.len() {
        bail!("length mismatch: got {} want {}", got.len(), want.len());
    }
    let mut max = 0f32;
    let mut sum = 0f64;
    let mut worst = 0usize;
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max {
            max = d;
            worst = i;
        }
        sum += d as f64;
    }
    let frames = frontend.params().n_mels;
    println!(
        "frames_axis_mels={frames} values={} max_abs_diff={max:.3e} mean_abs_diff={:.3e} worst_index={worst}",
        got.len(),
        sum / got.len() as f64
    );
    if max > 1e-4 {
        bail!("mel parity failed: max abs diff {max:.3e} > 1e-4");
    }
    Ok(())
}
