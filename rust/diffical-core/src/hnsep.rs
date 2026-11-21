//! HN-SEP harmonic/noise separation (CascadedNet mask network, ONNX Runtime).
//!
//! The Python reference keeps STFT, the mask network and ISTFT in torch. Here
//! the mask network runs from `export/hnsep.onnx` and the STFT/ISTFT are
//! reimplemented to match `torch.stft`/`torch.istft` with a periodic Hann
//! window, `center=True`, `pad_mode="constant"` and the reference's
//! "pad to a whole segment" preprocessing. Output frames are therefore always
//! a multiple of 32, which the exported graph's decoder expects.

use anyhow::{Context, Result};
use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

pub const N_FFT: usize = 2048;
pub const HOP: usize = 512;
/// `CascadedNet.seg_length`: 32 * hop.
pub const SEG_LENGTH: usize = 32 * HOP;
const SPEC_BINS: usize = N_FFT / 2 + 1;
const MAX_BIN: usize = N_FFT / 2;

/// Default bounded separation cache: enough for a phrase with many distinct
/// sources, while keeping at most ~256 MiB of `harmonic` + `noise` buffers.
pub const DEFAULT_CACHE_FILES: usize = 64;
pub const DEFAULT_CACHE_BYTES: usize = 256 << 20;

fn periodic_hann(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * i as f32 / size as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

pub struct Hnsep {
    _env: Environment,
    session: Session,
    window: Vec<f32>,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
}

impl Hnsep {
    /// Load for a standalone process (the `hnsep-parity` check). The render
    /// pipeline reuses the engine's environment via `load_with_env` instead,
    /// because ONNX Runtime allows a single environment per process.
    pub fn load(path: &Path) -> Result<Self> {
        let env = crate::runtime::environment()?;
        Self::load_with_env(&env, path)
    }

    pub fn load_with_env(env: &Environment, path: &Path) -> Result<Self> {
        let session = Session::builder(env)?
            .with_execution_providers([ort::ep::CPU::default().build()])
            .unwrap_or_else(|e| e.recover())
            .with_optimization_level(GraphOptimizationLevel::All)
            .unwrap_or_else(|e| e.recover())
            .with_intra_threads(4)
            .unwrap_or_else(|e| e.recover())
            .with_inter_threads(2)
            .unwrap_or_else(|e| e.recover())
            .commit_from_file(path)
            .with_context(|| format!("load HN-SEP model {path:?}"))?;
        let mut planner = FftPlanner::<f32>::new();
        Ok(Self {
            _env: env.clone(),
            session,
            window: periodic_hann(N_FFT),
            forward: planner.plan_fft_forward(N_FFT),
            inverse: planner.plan_fft_inverse(N_FFT),
        })
    }

    /// Reference preprocessing: zero-pad so `T + hop` fills whole segments,
    /// split the pad evenly at hop granularity.
    fn pad_plan(&self, samples: usize) -> (usize, usize) {
        let t1 = samples + HOP;
        let target = SEG_LENGTH * t1.div_ceil(SEG_LENGTH);
        let t_pad = target - t1;
        let left = (t_pad / 2) / HOP * HOP;
        (left, samples + t_pad)
    }

    /// `torch.stft(center=True, pad_mode="constant", onesided=True)`.
    fn analyze(&self, padded: &[f32], frames: usize) -> (Vec<f32>, Vec<f32>) {
        let mut re = vec![0f32; SPEC_BINS * frames];
        let mut im = vec![0f32; SPEC_BINS * frames];
        let mut buf = vec![Complex32::default(); N_FFT];
        let half = (N_FFT / 2) as isize;
        let len = padded.len() as isize;
        for frame in 0..frames {
            let start = (frame * HOP) as isize - half;
            for (i, slot) in buf.iter_mut().enumerate() {
                let idx = start + i as isize;
                let sample = if idx >= 0 && idx < len {
                    padded[idx as usize]
                } else {
                    0.0
                };
                *slot = Complex32::new(sample * self.window[i], 0.0);
            }
            self.forward.process(&mut buf);
            for bin in 0..SPEC_BINS {
                re[bin * frames + frame] = buf[bin].re;
                im[bin * frames + frame] = buf[bin].im;
            }
        }
        (re, im)
    }

    /// `torch.istft(center=True)`: overlap-add, window-squared envelope
    /// normalization, then trim `n_fft / 2` from both ends.
    fn synthesize(
        &self,
        spec_re: &[f32],
        spec_im: &[f32],
        mask: &[f32],
        frames: usize,
        left_pad: usize,
        out_samples: usize,
    ) -> Vec<f32> {
        let full_len = N_FFT + HOP * (frames - 1);
        let mut acc = vec![0f64; full_len];
        let mut env = vec![0f64; full_len];
        let mut buf = vec![Complex32::default(); N_FFT];
        for frame in 0..frames {
            for bin in 0..SPEC_BINS {
                let mre = mask[bin * frames + frame];
                let mim = mask[SPEC_BINS * frames + bin * frames + frame];
                let sre = spec_re[bin * frames + frame];
                let sim = spec_im[bin * frames + frame];
                buf[bin] = Complex32::new(sre * mre - sim * mim, sre * mim + sim * mre);
            }
            for bin in 1..MAX_BIN {
                buf[N_FFT - bin] = buf[bin].conj();
            }
            buf[0].im = 0.0;
            buf[MAX_BIN].im = 0.0;
            self.inverse.process(&mut buf);
            let scale = 1.0 / N_FFT as f32;
            for (i, slot) in buf.iter().enumerate() {
                acc[frame * HOP + i] += (slot.re * scale) as f64 * self.window[i] as f64;
                env[frame * HOP + i] += (self.window[i] as f64).powi(2);
            }
        }
        let mut out = vec![0f32; out_samples];
        let center = N_FFT / 2;
        for (k, slot) in out.iter_mut().enumerate() {
            let idx = center + left_pad + k;
            if idx < acc.len() && env[idx] > 1e-11 {
                *slot = (acc[idx] / env[idx]) as f32;
            }
        }
        out
    }

    /// Harmonic component of `wave`; callers derive noise as `wave - harmonic`.
    pub fn separate(&mut self, wave: &[f32]) -> Result<Vec<f32>> {
        let samples = wave.len();
        if samples == 0 {
            return Ok(Vec::new());
        }
        let (left, padded_len) = self.pad_plan(samples);
        anyhow::ensure!(
            samples < 60 * 44_100,
            "HN-SEP input too long ({samples} samples, limit 60 s)"
        );
        let frames = padded_len / HOP + 1;
        anyhow::ensure!(frames % 32 == 0, "HN-SEP frame count {frames} is not 32-aligned");
        let mut padded = vec![0f32; padded_len];
        padded[left..left + samples].copy_from_slice(wave);

        let (spec_re, spec_im) = self.analyze(&padded, frames);
        let mut input = vec![0f32; 2 * SPEC_BINS * frames];
        input[..SPEC_BINS * frames].copy_from_slice(&spec_re);
        input[SPEC_BINS * frames..].copy_from_slice(&spec_im);

        let shape = vec![1i64, 2, SPEC_BINS as i64, frames as i64];
        let mask = {
            let outputs = self.session.run(ort::inputs![
                "spec" => Tensor::from_array((shape, input)).context("HN-SEP input tensor")?
            ])?;
            let (mask_shape, data) = outputs["mask"]
                .try_extract_tensor::<f32>()
                .context("HN-SEP mask output")?;
            anyhow::ensure!(
                data.len() == 2 * SPEC_BINS * frames,
                "HN-SEP mask shape {mask_shape:?} does not match {frames} frames"
            );
            data.to_vec()
        };
        Ok(self.synthesize(&spec_re, &spec_im, &mask, frames, left, samples))
    }
}

/// Identity of one source *window* at a point in time: canonical path, mtime,
/// and the half-open sample range `[start_sample, end_sample)`.
///
/// A changed file changes the mtime, so a cache hit can never reuse audio from
/// another version of the same path; a different window is a different key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceKey {
    path: PathBuf,
    mtime_ns: u128,
    start_sample: usize,
    end_sample: usize,
    sample_rate: u32,
}

impl SourceKey {
    pub fn new(
        path: &Path,
        start_sample: usize,
        end_sample: usize,
        sample_rate: u32,
    ) -> Option<Self> {
        if end_sample <= start_sample {
            return None;
        }
        let metadata = std::fs::metadata(path).ok()?;
        let mtime_ns = metadata
            .modified()
            .ok()?
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_nanos();
        let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        Some(Self {
            path,
            mtime_ns,
            start_sample,
            end_sample,
            sample_rate,
        })
    }

    fn samples(&self) -> usize {
        self.end_sample - self.start_sample
    }

    fn bytes(&self) -> usize {
        2 * self.samples() * std::mem::size_of::<f32>()
    }
}

/// One full-file separation. Both components are shared with the phones that
/// use the file and with later renders.
#[derive(Debug, Clone)]
pub struct Separation {
    pub harmonic: Arc<Vec<f32>>,
    pub noise: Arc<Vec<f32>>,
}

struct CacheState {
    entries: HashMap<SourceKey, (Separation, u64)>,
    tick: u64,
    bytes: usize,
}

/// Bounded LRU cache of whole-file separations, safe to share between threads.
///
/// The engine serializes renders behind one lock today, but the cache itself
/// uses a mutex so a future concurrent renderer cannot corrupt it.
pub struct HnsepCache {
    state: Mutex<CacheState>,
    max_entries: usize,
    max_bytes: usize,
}

impl Default for HnsepCache {
    fn default() -> Self {
        Self::new(DEFAULT_CACHE_FILES, DEFAULT_CACHE_BYTES)
    }
}

impl HnsepCache {
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            state: Mutex::new(CacheState {
                entries: HashMap::new(),
                tick: 0,
                bytes: 0,
            }),
            max_entries: max_entries.max(1),
            max_bytes: max_bytes.max(1),
        }
    }

    /// `(entries, stored bytes)`; used by tests and timing output.
    pub fn stats(&self) -> (usize, usize) {
        let state = self.state.lock().expect("HN-SEP cache mutex poisoned");
        (state.entries.len(), state.bytes)
    }

    fn get(&self, key: &SourceKey) -> Option<Separation> {
        let mut state = self.state.lock().expect("HN-SEP cache mutex poisoned");
        let tick = state.tick;
        state.tick += 1;
        let entry = state.entries.get_mut(key)?;
        entry.1 = tick;
        Some(entry.0.clone())
    }

    fn put(&self, key: SourceKey, separation: Separation) {
        let mut state = self.state.lock().expect("HN-SEP cache mutex poisoned");
        let tick = state.tick;
        state.tick += 1;
        let bytes = key.bytes();
        if let Some((old, _)) = state.entries.remove(&key) {
            let old_size = old.harmonic.len() * std::mem::size_of::<f32>() * 2;
            state.bytes = state.bytes.saturating_sub(old_size);
        }
        while (!state.entries.is_empty())
            && (state.entries.len() >= self.max_entries
                || state.bytes.saturating_add(bytes) > self.max_bytes)
        {
            let Some(oldest) = state
                .entries
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some((old, _)) = state.entries.remove(&oldest) {
                let old_size = old.harmonic.len() * std::mem::size_of::<f32>() * 2;
                state.bytes = state.bytes.saturating_sub(old_size);
            }
        }
        state.bytes = state.bytes.saturating_add(bytes);
        state.entries.insert(key, (separation, tick));
    }
}

impl Hnsep {
    /// Separate `wave`, reusing a cached full-file result when `key` matches.
    ///
    /// Cache keys include the canonical path, mtime, sample count and sample
    /// rate, so a file rewrite invalidates the entry. `cache = None` or
    /// `key = None` (missing/unreadable source) simply skips caching.
    pub fn separate_cached(
        &mut self,
        cache: Option<&HnsepCache>,
        key: Option<&SourceKey>,
        wave: &[f32],
    ) -> Result<Separation> {
        let timing = std::env::var_os("DIFFICAL_TIMING").is_some();
        if let (Some(cache), Some(key)) = (cache, key) {
            if let Some(hit) = cache.get(key) {
                if hit.harmonic.len() == wave.len() {
                    if timing {
                        let (entries, bytes) = cache.stats();
                        eprintln!(
                            "[timing] hnsep cache hit ({} samples, {entries} entries, {:.1} MB)",
                            wave.len(),
                            bytes as f64 / 1e6
                        );
                    }
                    return Ok(hit);
                }
            }
        }
        let started = std::time::Instant::now();
        let harmonic = self.separate(wave)?;
        if timing {
            eprintln!(
                "[timing] hnsep separate: {:.1} ms ({} samples)",
                started.elapsed().as_secs_f64() * 1000.0,
                wave.len()
            );
        }
        let noise = wave
            .iter()
            .zip(&harmonic)
            .map(|(w, h)| w - h)
            .collect::<Vec<f32>>();
        let separation = Separation {
            harmonic: Arc::new(harmonic),
            noise: Arc::new(noise),
        };
        if let (Some(cache), Some(key)) = (cache, key) {
            cache.put(key.clone(), separation.clone());
            if timing {
                let (entries, bytes) = cache.stats();
                eprintln!(
                    "[timing] hnsep cache miss ({} samples, {entries} entries, {:.1} MB)",
                    wave.len(),
                    bytes as f64 / 1e6
                );
            }
        }
        Ok(separation)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        periodic_hann, HnsepCache, Separation, SourceKey, HOP, N_FFT, SEG_LENGTH,
    };
    use std::sync::Arc;

    #[test]
    fn hann_matches_torch_periodic() {
        let w = periodic_hann(N_FFT);
        assert!((w[0]).abs() < 1e-9);
        assert!((w[N_FFT / 2] - 1.0).abs() < 1e-6);
        assert!((w[1] - 0.5 * (1.0 - (2.0 * std::f32::consts::PI / N_FFT as f32).cos())).abs() < 1e-7);
    }

    fn temp_key(dir: &std::path::Path, name: &str, start: usize, end: usize) -> SourceKey {
        let path = dir.join(name);
        std::fs::write(&path, [0u8; 4]).expect("write temp source");
        SourceKey::new(&path, start, end, 44_100).expect("cache key")
    }

    fn separation(n: usize) -> Separation {
        Separation {
            harmonic: Arc::new(vec![0.0; n]),
            noise: Arc::new(vec![0.0; n]),
        }
    }

    #[test]
    fn cache_evicts_least_recently_used() {
        let dir = std::env::temp_dir().join(format!("diffical_hnsep_lru_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let keys = [
            temp_key(&dir, "a.f32", 0, 3),
            temp_key(&dir, "b.f32", 0, 3),
            temp_key(&dir, "c.f32", 0, 3),
        ];
        let cache = HnsepCache::new(2, 1 << 30);
        cache.put(keys[0].clone(), separation(3));
        cache.put(keys[1].clone(), separation(3));
        assert!(cache.get(&keys[0]).is_some()); // keys[1] becomes oldest
        cache.put(keys[2].clone(), separation(3));
        assert_eq!(cache.stats().0, 2);
        assert!(cache.get(&keys[0]).is_some());
        assert!(cache.get(&keys[1]).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_respects_byte_bound() {
        let dir = std::env::temp_dir().join(format!("diffical_hnsep_bytes_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let keys = [
            temp_key(&dir, "a.f32", 0, 100),
            temp_key(&dir, "b.f32", 0, 100),
        ];
        // Each entry accounts 2 * 100 * 4 = 800 bytes; allow only one.
        let cache = HnsepCache::new(8, 900);
        cache.put(keys[0].clone(), separation(100));
        cache.put(keys[1].clone(), separation(100));
        assert_eq!(cache.stats().0, 1);
        assert!(cache.get(&keys[1]).is_some());
        assert!(cache.get(&keys[0]).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_keys_track_file_identity() {
        let dir = std::env::temp_dir().join(format!("diffical_hnsep_key_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let a = temp_key(&dir, "a.f32", 0, 10);
        let b = temp_key(&dir, "b.f32", 0, 10);
        // Same file, same window (do not rewrite it: mtime is part of the key).
        let same = SourceKey::new(&dir.join("a.f32"), 0, 10, 44_100).expect("key");
        assert_eq!(a, same);
        assert_ne!(a, b);
        let other_window = SourceKey::new(&dir.join("a.f32"), 1, 10, 44_100).expect("key");
        assert_ne!(a, other_window);
        let other_rate = SourceKey::new(&dir.join("a.f32"), 0, 10, 48_000).expect("key");
        assert_ne!(a, other_rate);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pad_plan_is_32_frame_aligned() {
        // Mirrors the reference pad arithmetic for a range of lengths.
        for samples in [1usize, 512, 20_000, 70_560, 220_500] {
            let t1 = samples + HOP;
            let target = SEG_LENGTH * t1.div_ceil(SEG_LENGTH);
            let t_pad = target - t1;
            let left = (t_pad / 2) / HOP * HOP;
            let padded_len = samples + t_pad;
            assert_eq!(padded_len % HOP, 0);
            assert_eq!(left + (t_pad - left), t_pad);
            assert_eq!(padded_len / HOP + 1, target / HOP);
            assert_eq!((padded_len / HOP + 1) % 32, 0);
        }
    }
}
