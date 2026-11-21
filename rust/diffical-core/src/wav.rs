//! Minimal wav IO matching `soundfile.read(dtype="float32")` for the mono,
//! 16/24-bit PCM voicebanks used by the host editor.
//!
//! Sample-rate conversion is intentionally rejected: every current voicebank
//! file is 44.1 kHz, and silently resampling with a different filter than
//! `resampy` would break parity. This is recorded as a gap in PROGRESS.md.

use anyhow::{bail, Context, Result};
use std::path::Path;

pub fn read_wav_mono(path: &Path, target_sr: u32) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open wav {path:?}"))?;
    let spec = reader.spec();
    if spec.sample_rate != target_sr {
        bail!(
            "unsupported sample rate {} Hz in {path:?} (need {target_sr} Hz)",
            spec.sample_rate
        );
    }
    if spec.channels == 0 || spec.channels > 8 {
        bail!("unsupported channel count {} in {path:?}", spec.channels);
    }
    let channels = spec.channels as usize;
    let mut interleaved: Vec<f32> = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Float => {
            if spec.bits_per_sample != 32 {
                bail!(
                    "unsupported float bit depth {} in {path:?}",
                    spec.bits_per_sample
                );
            }
            for sample in reader.samples::<f32>() {
                interleaved.push(sample.with_context(|| format!("read {path:?}"))?);
            }
        }
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            if !(8..=32).contains(&bits) {
                bail!("unsupported PCM bit depth {bits} in {path:?}");
            }
            let scale = (1i64 << (bits - 1)) as f32;
            for sample in reader.samples::<i32>() {
                interleaved.push(sample.with_context(|| format!("read {path:?}"))? as f32 / scale);
            }
        }
    }
    if interleaved.is_empty() {
        return Ok(Vec::new());
    }
    if channels == 1 {
        return Ok(interleaved);
    }
    let frames = interleaved.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for frame in interleaved.chunks_exact(channels) {
        let sum: f32 = frame.iter().sum();
        mono.push(sum / channels as f32);
    }
    Ok(mono)
}

pub fn write_wav_f32(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create wav {path:?}"))?;
    for sample in samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "diffical_wav_{}_{tag}.wav",
            std::process::id()
        ))
    }

    #[test]
    fn reads_pcm16_and_downmixes() {
        let path = temp_path("pcm16");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        {
            let mut writer = hound::WavWriter::create(&path, spec).unwrap();
            for sample in [0i16, 16384, -32768, 32767, 1000, 3000] {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        let mono = read_wav_mono(&path, 44_100).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.25).abs() < 1e-6); // (0 + 16384) / 2 / 32768
        assert!((mono[1] + 0.5 / 32768.0).abs() < 1e-6); // (-32768 + 32767) / 2 / 32768
        assert!((mono[2] - 2000.0 / 32768.0).abs() < 1e-6);
    }

    #[test]
    fn reads_pcm24() {
        let path = temp_path("pcm24");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44_100,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        };
        {
            let mut writer = hound::WavWriter::create(&path, spec).unwrap();
            for sample in [0i32, 4_194_304, -8_388_608, 8_388_607] {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        let mono = read_wav_mono(&path, 44_100).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(mono.len(), 4);
        assert!((mono[1] - 0.5).abs() < 1e-6);
        assert!((mono[2] + 1.0).abs() < 1e-6);
        assert!((mono[3] - 8_388_607.0 / 8_388_608.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_other_sample_rates() {
        let path = temp_path("rate");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        {
            let mut writer = hound::WavWriter::create(&path, spec).unwrap();
            writer.write_sample(0i16).unwrap();
            writer.finalize().unwrap();
        }
        let err = read_wav_mono(&path, 44_100).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.to_string().contains("sample rate"));
    }
}
