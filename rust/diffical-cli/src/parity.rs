use anyhow::{bail, Context, Result};
use crate::stats::{diff_f32, DiffStats};
use diffical_core::assemble::{build_inputs, Assembler};
use diffical_core::config::AudioConfig;
use diffical_core::hnsep::{Hnsep, HnsepCache};
use diffical_core::mel::read_f32;
use diffical_core::pipeline::{PhoneStage, RenderEngine};
use diffical_core::protocol::parse_request;
use diffical_core::wav::write_wav_f32;
use std::path::{Path, PathBuf};

fn read_text(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("read {path:?}"))
}

fn read_meta(dir: &Path) -> Result<serde_json::Value> {
    serde_json::from_str(&read_text(&dir.join("meta.json"))?).context("parse meta.json")
}

fn load_golden_f32(dir: &Path, name: &str) -> Result<Vec<f32>> {
    let path = dir.join(name);
    anyhow::ensure!(path.is_file(), "missing golden file {path:?}");
    read_f32(&path)
}

fn phone_meta(meta: &serde_json::Value, index: usize) -> Option<&serde_json::Value> {
    meta.get("phones")?.get(index)
}

fn f64_at(meta: &serde_json::Value, index: usize, key: &str) -> Option<f64> {
    phone_meta(meta, index)?.get(key)?.as_f64()
}

fn i64_at(meta: &serde_json::Value, index: usize, key: &str) -> Option<i64> {
    phone_meta(meta, index)?.get(key)?.as_i64()
}

pub fn phone_parity(
    request_path: &str,
    golden_dir: &str,
    hnsep_path: Option<PathBuf>,
) -> Result<()> {
    let req = parse_request(&read_text(Path::new(request_path))?)?;
    let golden = PathBuf::from(golden_dir);
    let meta = read_meta(&golden)?;
    let cfg = AudioConfig::default();
    let mut stage = PhoneStage::new(cfg.clone())?;
    let mut hnsep = match hnsep_path.as_ref() {
        Some(path) => Some(Hnsep::load(path)?),
        None => None,
    };
    let cache = hnsep.as_ref().map(|_| HnsepCache::default());
    let (pitch, features) =
        stage.build_with_hnsep(&req, true, hnsep.as_mut(), cache.as_ref())?;

    let mut mel_all = DiffStats::default();
    let mut f0_all = DiffStats::default();
    let mut mel_orig_all = DiffStats::default();
    let mut src_all = DiffStats::default();
    let mut meta_mismatches = Vec::new();
    let mut worst = (0usize, 0f32);

    for (i, feat) in features.iter().enumerate() {
        let mel_want = load_golden_f32(&golden, &format!("phone_{i:02}_mel.f32"))?;
        if mel_want.len() != feat.mel.data.len() {
            bail!(
                "phone {i} {}: mel length got {} want {}",
                feat.phoneme,
                feat.mel.data.len(),
                mel_want.len()
            );
        }
        let local = diff_f32(&feat.mel.data, &mel_want, &format!("phone {i} mel"))?;
        if local.max > worst.1 {
            worst = (i, local.max);
        }
        for (a, b) in feat.mel.data.iter().zip(&mel_want) {
            mel_all.push((a - b).abs());
        }
        let f0_want = load_golden_f32(&golden, &format!("phone_{i:02}_f0.f32"))?;
        let local_f0 = diff_f32(&feat.f0, &f0_want, &format!("phone {i} f0"))?;
        for (a, b) in feat.f0.iter().zip(&f0_want) {
            f0_all.push((a - b).abs());
        }

        let mut orig_note = String::new();
        if let Some(debug) = &feat.debug {
            let orig_want = load_golden_f32(&golden, &format!("phone_{i:02}_mel_orig.f32"))?;
            let local_orig = diff_f32(
                &debug.mel_orig.data,
                &orig_want,
                &format!("phone {i} mel_orig"),
            )?;
            for (a, b) in debug.mel_orig.data.iter().zip(&orig_want) {
                mel_orig_all.push((a - b).abs());
            }
            orig_note = format!(" mel_orig_max={:.2e}", local_orig.max);
            let src_want = load_golden_f32(&golden, &format!("phone_{i:02}_src.f32"))?;
            let local_src = diff_f32(&debug.source_wave, &src_want, &format!("phone {i} src"))?;
            for (a, b) in debug.source_wave.iter().zip(&src_want) {
                src_all.push((a - b).abs());
            }
            orig_note += &format!(" src_max={:.2e}", local_src.max);
        }

        for (key, got) in [
            ("start_frame", feat.start_frame as f64),
            ("start_frame_exact", feat.start_frame_exact),
            ("end_frame", (feat.start_frame + feat.mel.frames as i64) as f64),
            ("mel_frames", feat.mel.frames as f64),
            ("pre_vel_ms", feat.pre_vel_ms),
            ("source_skip_ms", feat.source_skip_ms),
            ("envelope_span_ms", feat.envelope_span_ms),
            ("segment_len_ms", feat.segment_len_ms),
            ("tail_silence_frames", feat.tail_silence_frames as f64),
            ("pre_scale", feat.pre_scale),
        ] {
            let want = if key == "mel_frames" {
                i64_at(&meta, i, "mel_frames").map(|v| v as f64)
            } else {
                f64_at(&meta, i, key)
            };
            if let Some(want) = want {
                let tol = if key.ends_with("_frame") || key == "mel_frames" {
                    0.0
                } else {
                    1e-6
                };
                if (got - want).abs() > tol {
                    meta_mismatches.push(format!(
                        "phone {i} {key}: got {got:.6} want {want:.6}"
                    ));
                }
            }
        }
        if feat.mel.frames != i64_at(&meta, i, "mel_frames").unwrap_or(feat.mel.frames as i64) as usize {
            bail!("phone {i}: mel frame count mismatch");
        }

        println!(
            "phone {i:>2} {:>8}: mel_max={:.3e} f0_max={:.3e} frames={}{}",
            format!("{:?}", feat.phoneme),
            local.max,
            local_f0.max,
            feat.mel.frames,
            orig_note
        );
    }

    let mel_all = mel_all.finish();
    let f0_all = f0_all.finish();
    let mel_orig_all = mel_orig_all.finish();
    let src_all = src_all.finish();
    println!(
        "total_frames={} (golden {}) start mismatch checks: {}",
        pitch.total_frames,
        meta.get("total_frames").and_then(|v| v.as_i64()).unwrap_or(-1),
        if meta_mismatches.is_empty() {
            "all match".to_string()
        } else {
            format!("{} mismatches", meta_mismatches.len())
        }
    );
    for line in meta_mismatches.iter().take(10) {
        println!("  meta: {line}");
    }
    println!(
        "per-phone mel: max_abs_diff={:.3e} mean_abs_diff={:.3e} (gate 1e-3, worst phone {})",
        mel_all.max, mel_all.mean, worst.0
    );
    println!(
        "diagnostics: f0_max={:.3e} mel_orig_max={:.3e} src_max={:.3e}",
        f0_all.max, mel_orig_all.max, src_all.max
    );
    if mel_all.max >= 1e-3 {
        bail!("phone parity failed: mel max diff {:.3e} >= 1e-3", mel_all.max);
    }
    println!("phone parity PASS");
    Ok(())
}

pub fn assemble_parity(
    request_path: &str,
    golden_dir: &str,
    assembler_path: &str,
    hnsep_path: Option<PathBuf>,
) -> Result<()> {
    let req = parse_request(&read_text(Path::new(request_path))?)?;
    let golden = PathBuf::from(golden_dir);
    let meta = read_meta(&golden)?;
    let cfg = AudioConfig::default();
    let mut stage = PhoneStage::new(cfg.clone())?;
    let env = diffical_core::runtime::environment()?;
    let mut hnsep = match hnsep_path.as_ref() {
        Some(path) => Some(Hnsep::load_with_env(&env, path)?),
        None => None,
    };
    let cache = hnsep.as_ref().map(|_| HnsepCache::default());
    let (pitch, features) =
        stage.build_with_hnsep(&req, false, hnsep.as_mut(), cache.as_ref())?;

    let mut f0_want = load_golden_f32(&golden, "f0_phrase.f32")?;
    if f0_want.len() < pitch.f0.len() {
        f0_want.resize(pitch.f0.len(), 0.0);
    }
    let f0_stats = diff_f32(&pitch.f0, &f0_want[..pitch.f0.len()], "f0_phrase")?;
    println!(
        "f0_phrase: max_abs_diff={:.3e} mean_abs_diff={:.3e}",
        f0_stats.max, f0_stats.mean
    );

    let inputs = build_inputs(
        &features,
        pitch.total_frames,
        &pitch.f0,
        cfg.n_mels,
        diffical_core::assemble::seg_norm_from_env(),
    )?;
    let mut checks: Vec<(&str, DiffStats)> = Vec::new();
    let golden_segs = load_golden_f32(&golden, "segs.f32")?;
    checks.push((
        "segs",
        diff_f32(&inputs.segs, &golden_segs, "segs")?,
    ));
    let golden_len: Vec<i64> = load_golden_f32(&golden, "seg_len.f32")?
        .into_iter()
        .map(|v| v as i64)
        .collect();
    anyhow::ensure!(
        golden_len.len() == inputs.seg_len.len(),
        "seg_len length got {} want {}",
        inputs.seg_len.len(),
        golden_len.len()
    );
    let len_max = inputs
        .seg_len
        .iter()
        .zip(&golden_len)
        .map(|(a, b)| (*a - *b).unsigned_abs())
        .max()
        .unwrap_or(0);
    let golden_idx: Vec<i64> = load_golden_f32(&golden, "seg_idx.f32")?
        .into_iter()
        .map(|v| v as i64)
        .collect();
    anyhow::ensure!(
        golden_idx.len() == inputs.seg_idx.len(),
        "seg_idx length got {} want {}",
        inputs.seg_idx.len(),
        golden_idx.len()
    );
    let idx_max = inputs
        .seg_idx
        .iter()
        .zip(&golden_idx)
        .map(|(a, b)| (*a - *b).unsigned_abs())
        .max()
        .unwrap_or(0);
    checks.push((
        "progress",
        diff_f32(
            &inputs.progress,
            &load_golden_f32(&golden, "progress.f32")?,
            "progress",
        )?,
    ));
    checks.push((
        "boundary",
        diff_f32(
            &inputs.boundary,
            &load_golden_f32(&golden, "boundary.f32")?,
            "boundary",
        )?,
    ));
    checks.push((
        "assembler_f0",
        diff_f32(
            &inputs.f0,
            &load_golden_f32(&golden, "assembler_f0.f32")?,
            "assembler_f0",
        )?,
    ));
    for (name, stats) in &checks {
        println!(
            "input {name}: max_abs_diff={:.3e} mean_abs_diff={:.3e}",
            stats.max, stats.mean
        );
    }
    println!("input seg_len: max_diff={len_max}; seg_idx: max_diff={idx_max}");
    let seg_count = meta
        .get("seg_count")
        .and_then(|v| v.as_i64())
        .unwrap_or(inputs.seg_count as i64);
    println!(
        "segments: {} (golden {}) lmax={} frames={}",
        inputs.seg_count, seg_count, inputs.lmax, pitch.total_frames
    );

    let mut assembler = Assembler::load_with_env(&env, Path::new(assembler_path), cfg.n_mels)?;
    let mel = assembler.assemble(inputs, pitch.total_frames)?;
    let mel_want = load_golden_f32(&golden, "mel_phrase.f32")?;
    let mel_stats = diff_f32(&mel.data, &mel_want, "mel_phrase")?;
    println!(
        "mel_phrase: {}x{} max_abs_diff={:.3e} mean_abs_diff={:.3e} (gate 1e-3)",
        mel.n_mels,
        mel.frames,
        mel_stats.max,
        mel_stats.mean
    );
    if mel_stats.max >= 1e-3 {
        bail!("assemble parity failed: mel max diff {:.3e}", mel_stats.max);
    }
    println!("assemble parity PASS");
    Ok(())
}

fn compare_samples(got: &[f32], want: &[f32], label: &str) -> Result<()> {
    anyhow::ensure!(
        got.len() == want.len(),
        "sample count mismatch: got {} want {}",
        got.len(),
        want.len()
    );
    let stats = diff_f32(got, want, label)?;
    let dot: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let na: f64 = got.iter().map(|v| *v as f64 * *v as f64).sum();
    let nb: f64 = want.iter().map(|v| *v as f64 * *v as f64).sum();
    let ncc = dot / (na.sqrt() * nb.sqrt()).max(1e-20);
    let rms_got = (na / got.len() as f64).sqrt();
    let rms_want = (nb / want.len() as f64).sqrt();
    let ratio = rms_got / rms_want.max(1e-20);
    println!(
        "samples={} max_abs_diff={:.3e} ncc={ncc:.9} rms_got={rms_got:.6} rms_ref={rms_want:.6} ratio={ratio:.6}",
        got.len(),
        stats.max
    );
    if ncc <= 0.99 {
        bail!("{label} parity failed: cross-correlation {ncc:.6} <= 0.99");
    }
    if (ratio - 1.0).abs() > 0.05 {
        bail!("{label} parity failed: RMS ratio {ratio:.6} outside 5%");
    }
    println!("{label} parity PASS");
    Ok(())
}

pub fn e2e_parity(
    request_path: &str,
    golden_dir: &str,
    assembler_path: &str,
    vocoder_path: &str,
    hnsep_path: Option<PathBuf>,
) -> Result<()> {
    let req = parse_request(&read_text(Path::new(request_path))?)?;
    let golden = PathBuf::from(golden_dir);
    let mut engine = RenderEngine::with_hnsep(
        Path::new(assembler_path),
        Path::new(vocoder_path),
        hnsep_path,
    )?;
    let rendered = engine.render(&req, false)?;
    let want = load_golden_f32(&golden, "samples.f32")?;
    let trace = &rendered.normalization;
    println!(
        "normalization: reason={} gain={:.9} rms_before={:.6} peak_before={:.6} active_windows={}/{}",
        trace.reason,
        trace.gain,
        trace.active_rms_before,
        trace.peak_before,
        trace.active_windows,
        trace.total_windows
    );
    if let Some(py_gain) = read_meta(&golden)?
        .get("normalization_trace")
        .and_then(|t| t.get("gain"))
        .and_then(|v| v.as_f64())
    {
        println!("python normalization gain: {py_gain:.9}");
    }
    compare_samples(&rendered.samples, &want, "e2e")
}

/// Validate the JSON body returned by `POST /render` against the Python wav.
pub fn response_parity(response_path: &str, golden_dir: &str) -> Result<()> {
    let value: serde_json::Value =
        serde_json::from_str(&read_text(Path::new(response_path))?).context("parse response")?;
    let samples: Vec<f32> = value
        .get("samples")
        .and_then(|v| v.as_array())
        .context("response has no samples array")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN) as f32)
        .collect();
    let rate = value.get("sample_rate").and_then(|v| v.as_u64());
    println!(
        "response: {} samples, sample_rate={:?}, leading_ms={:?}, estimated_length_ms={:?}",
        samples.len(),
        rate,
        value.get("leading_ms"),
        value.get("estimated_length_ms")
    );
    anyhow::ensure!(rate == Some(44_100), "unexpected sample_rate {rate:?}");
    let want = load_golden_f32(&PathBuf::from(golden_dir), "samples.f32")?;
    compare_samples(&samples, &want, "server response")
}

pub fn render_to_wav(
    request_path: &str,
    out_path: &str,
    assembler_path: &str,
    vocoder_path: &str,
    hnsep_path: Option<PathBuf>,
) -> Result<()> {
    let mut req = parse_request(&read_text(Path::new(request_path))?)?;
    let _ = &mut req;
    let mut engine = RenderEngine::with_hnsep(
        Path::new(assembler_path),
        Path::new(vocoder_path),
        hnsep_path,
    )?;
    let rendered = engine.render(&req, false)?;
    write_wav_f32(Path::new(out_path), &rendered.samples, engine.cfg.sample_rate)?;
    println!(
        "wrote {out_path}: {} samples @ {} Hz, normalization {} (gain {:.6})",
        rendered.samples.len(),
        engine.cfg.sample_rate,
        rendered.normalization.reason,
        rendered.normalization.gain
    );
    Ok(())
}
