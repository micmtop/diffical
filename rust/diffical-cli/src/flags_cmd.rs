//! `flag-parity` and `hnsep-parity` commands.

use anyhow::{bail, Context, Result};
use diffical_core::flags::{inject_into_request, parse_flag_string};
use diffical_core::hnsep::Hnsep;
use diffical_core::mel::read_f32;
use diffical_core::pipeline::RenderEngine;
use diffical_core::protocol::parse_request;
use std::path::{Path, PathBuf};

fn read_text(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("read {path:?}"))
}

use crate::stats::diff_f32;

fn wav_stats(got: &[f32], want: &[f32]) -> (f64, f64) {
    let dot: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let na: f64 = got.iter().map(|v| *v as f64 * *v as f64).sum();
    let nb: f64 = want.iter().map(|v| *v as f64 * *v as f64).sum();
    let ncc = dot / (na.sqrt() * nb.sqrt()).max(1e-20);
    let ratio = (na / got.len() as f64).sqrt() / (nb / want.len() as f64).sqrt().max(1e-20);
    (ncc, ratio)
}

/// `flag-parity`: render a request (optionally with an injected flags string)
/// and compare per-phone mel and the wav against a Python reference dump.
///
/// Gates: comparable per-phone mel max abs diff < 1e-3; wav NCC > 0.99 and RMS
/// ratio within 10 %. `allow_mel_diff` / `allow_wav_diff` turn a gate off for
/// flags whose documented effect the Python reference does not implement.
pub fn flag_parity(
    request_path: &str,
    golden_dir: &str,
    assembler_path: &str,
    vocoder_path: &str,
    hnsep_path: Option<PathBuf>,
    inject: Option<&str>,
    allow_mel_diff: bool,
    allow_wav_diff: bool,
) -> Result<()> {
    let golden = PathBuf::from(golden_dir);
    let mut req = parse_request(&read_text(Path::new(request_path))?)?;
    let mut injected = Vec::new();
    if let Some(text) = inject {
        let entries = parse_flag_string(text);
        if entries.is_empty() {
            bail!("--inject-flags {text:?} contains no known flags");
        }
        println!("injecting flags on all {} phones: {text}", req.phones.len());
        for entry in &entries {
            match entry.value {
                Some(v) => injected.push(format!("{}={v}", entry.name)),
                None => injected.push(entry.name.to_string()),
            }
        }
        inject_into_request(&mut req, &entries);
    }

    let mut engine = RenderEngine::with_hnsep(
        Path::new(assembler_path),
        Path::new(vocoder_path),
        hnsep_path,
    )?;
    let rendered = engine.render(&req, false)?;

    let mut mel_max = 0f32;
    let mut mel_mean = 0f64;
    let mut mel_count = 0usize;
    let mut f0_max = 0f32;
    let mut comparable = 0usize;
    let mut length_mismatch = Vec::new();
    for (i, feat) in rendered.features.iter().enumerate() {
        let path = golden.join(format!("phone_{i:02}_mel.f32"));
        if !path.is_file() {
            bail!("missing golden {path:?}");
        }
        let f0_path = golden.join(format!("phone_{i:02}_f0.f32"));
        if f0_path.is_file() {
            let f0_want = read_f32(&f0_path)?;
            if f0_want.len() == feat.f0.len() {
                for (a, b) in feat.f0.iter().zip(&f0_want) {
                    f0_max = f0_max.max((a - b).abs());
                }
            }
        }
        let want = read_f32(&path)?;
        if want.len() != feat.mel.data.len() {
            length_mismatch.push(i);
            println!(
                "phone {i:>2} {:>8}: mel frames {} vs golden {} (not comparable)",
                format!("{:?}", feat.phoneme),
                feat.mel.frames,
                want.len() / feat.mel.n_mels.max(1)
            );
            continue;
        }
        comparable += 1;
        let mut local = 0f32;
        for (a, b) in feat.mel.data.iter().zip(&want) {
            let diff = (a - b).abs();
            local = local.max(diff);
            mel_max = mel_max.max(diff);
            mel_mean += diff as f64;
            mel_count += 1;
        }
        println!(
            "phone {i:>2} {:>8}: mel_max={:.3e} frames={} active=[{}]",
            format!("{:?}", feat.phoneme),
            local,
            feat.mel.frames,
            feat.flags.active().join(" ")
        );
    }
    if mel_count > 0 {
        mel_mean /= mel_count as f64;
    }

    let want = read_f32(&golden.join("samples.f32"))?;
    let (ncc, ratio) = if want.len() == rendered.samples.len() {
        let stats = wav_stats(&rendered.samples, &want);
        println!(
            "wav: {} samples max_abs_diff={:.3e} ncc={:.9} rms_ratio={:.6}",
            rendered.samples.len(),
            diff_f32(&rendered.samples, &want, "wav")?.max,
            stats.0,
            stats.1
        );
        stats
    } else {
        println!(
            "wav: {} samples vs golden {} (not comparable)",
            rendered.samples.len(),
            want.len()
        );
        (0.0, 0.0)
    };
    println!(
        "mel: {comparable}/{} phones comparable, max_abs_diff={mel_max:.3e} mean_abs_diff={mel_mean:.3e}; f0_max={f0_max:.3e} (diagnostic)",
        rendered.features.len()
    );
    if !injected.is_empty() {
        println!("effective flags: {}", injected.join(" "));
    }

    let mut failures = Vec::new();
    if !length_mismatch.is_empty() && !allow_mel_diff {
        failures.push(format!(
            "phone mel length mismatch on {length_mismatch:?} (compare with --allow-mel-diff)"
        ));
    }
    if !allow_mel_diff && mel_count > 0 && mel_max >= 1e-3 {
        failures.push(format!("mel max abs diff {mel_max:.3e} >= 1e-3"));
    }
    if !allow_wav_diff {
        if ncc <= 0.99 {
            failures.push(format!("wav NCC {ncc:.6} <= 0.99"));
        }
        if (ratio - 1.0).abs() > 0.10 {
            failures.push(format!("wav RMS ratio {ratio:.6} outside 10%"));
        }
    }
    if !failures.is_empty() {
        bail!("flag parity FAIL: {}", failures.join("; "));
    }
    println!("flag parity PASS");
    Ok(())
}

/// `hnsep-parity`: compare the Rust STFT + ONNX mask + ISTFT chain against a
/// torch reference harmonic dumped by `tools/hnsep_parity.py`.
pub fn hnsep_parity(golden_dir: &str, model_path: &str) -> Result<()> {
    let dir = PathBuf::from(golden_dir);
    let meta: serde_json::Value =
        serde_json::from_str(&read_text(&dir.join("meta.json"))?).context("parse meta.json")?;
    let rate = meta.get("sample_rate").and_then(|v| v.as_u64()).unwrap_or(44_100);
    anyhow::ensure!(rate == 44_100, "golden sample_rate {rate} != 44100");
    let wav = read_f32(&dir.join("wav.f32"))?;
    let want = read_f32(&dir.join("harmonic.f32"))?;
    anyhow::ensure!(wav.len() == want.len(), "golden wav/harmonic length mismatch");
    let mut session = Hnsep::load(Path::new(model_path))?;
    let got = session.separate(&wav)?;
    let stats = diff_f32(&got, &want, "hnsep harmonic")?;
    let torch_peak = want.iter().fold(0f32, |a, b| a.max(b.abs()));
    println!(
        "hnsep: {} samples, harmonic max_abs_diff={:.3e} mean_abs_diff={:.3e} (torch peak {torch_peak:.6})",
        got.len(),
        stats.max,
        stats.mean
    );
    if stats.max >= 1e-4 {
        bail!("HN-SEP parity failed: max abs diff {:.3e} >= 1e-4", stats.max);
    }
    println!("HN-SEP parity PASS (gate 1e-4)");
    Ok(())
}