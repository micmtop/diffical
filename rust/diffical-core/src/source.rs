//! Mel matrix type and voicebank wav loading for the phone stage.

use crate::wav::read_wav_mono;
use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Row-major `n_mels x frames` log-mel.
#[derive(Debug, Clone)]
pub struct MelMatrix {
    pub n_mels: usize,
    pub frames: usize,
    pub data: Vec<f32>,
}

impl MelMatrix {
    pub fn empty(n_mels: usize) -> Self {
        Self {
            n_mels,
            frames: 0,
            data: Vec::new(),
        }
    }

    pub fn from_vec(n_mels: usize, data: Vec<f32>) -> Result<Self> {
        anyhow::ensure!(
            n_mels > 0 && data.len() % n_mels == 0,
            "mel data length {} is not a multiple of {n_mels}",
            data.len()
        );
        Ok(Self {
            n_mels,
            frames: data.len() / n_mels,
            data,
        })
    }

    #[inline]
    pub fn get(&self, band: usize, frame: usize) -> f32 {
        self.data[band * self.frames + frame]
    }
}

/// Loads voicebank wav files, caching per render request like `AudioLoader`.
pub struct SourceLoader {
    sample_rate: u32,
    singer_path: String,
    cache: HashMap<String, Arc<Vec<f32>>>,
}

impl SourceLoader {
    /// Resolved path of a phone wav, if it exists (used as the HN-SEP cache
    /// identity together with mtime/sample count).
    pub fn resolved_path(&self, oto_file: &str) -> Option<PathBuf> {
        self.resolve(oto_file)
    }

    pub fn new(sample_rate: u32, singer_path: &str) -> Self {
        Self {
            sample_rate,
            singer_path: singer_path.to_string(),
            cache: HashMap::new(),
        }
    }

    fn resolve(&self, oto_file: &str) -> Option<PathBuf> {
        if oto_file.is_empty() {
            return None;
        }
        let direct = Path::new(oto_file);
        if direct.is_absolute() {
            return direct.is_file().then(|| direct.to_path_buf());
        }
        let joined = Path::new(&self.singer_path).join(oto_file);
        joined.is_file().then_some(joined)
    }

    /// Missing/unreadable sources degrade to ~100 ms of silence, matching
    /// `PhoneProcessor._load_wave`; a render must not fail for one bad phone.
    pub fn load(&mut self, oto_file: &str) -> Arc<Vec<f32>> {
        if let Some(cached) = self.cache.get(oto_file) {
            return Arc::clone(cached);
        }
        let wave = match self.resolve(oto_file) {
            Some(path) => match read_wav_mono(&path, self.sample_rate) {
                Ok(samples) => samples,
                Err(err) => {
                    eprintln!("warning: failed to load {path:?}: {err:#}; using silence");
                    silence(self.sample_rate)
                }
            },
            None => {
                if !oto_file.is_empty() {
                    eprintln!("warning: voicebank file not found: {oto_file}; using silence");
                }
                silence(self.sample_rate)
            }
        };
        let wave = Arc::new(wave);
        self.cache.insert(oto_file.to_string(), Arc::clone(&wave));
        wave
    }
}

fn silence(sample_rate: u32) -> Vec<f32> {
    vec![0.0; (0.1 * sample_rate as f64) as usize]
}

/// `np.interp` for a monotonically increasing `xp`.
pub fn interp(x: f64, xp: &[f64], fp: &[f32]) -> f32 {
    if xp.is_empty() {
        return 0.0;
    }
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    let mut lo = 0usize;
    let mut hi = xp.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xp[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    (fp[lo] as f64 * (1.0 - t) + fp[hi] as f64 * t) as f32
}


#[cfg(test)]
mod tests {
    use super::interp;

    #[test]
    fn interp_matches_numpy_endpoints() {
        let xp = [0.5, 1.5, 2.5];
        let fp = [10.0, 20.0, 30.0];
        assert_eq!(interp(-1.0, &xp, &fp), 10.0);
        assert_eq!(interp(0.5, &xp, &fp), 10.0);
        assert_eq!(interp(1.0, &xp, &fp), 15.0);
        assert_eq!(interp(2.5, &xp, &fp), 30.0);
        assert_eq!(interp(9.0, &xp, &fp), 30.0);
    }
}
