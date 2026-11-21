//! Shared f32 array comparison helpers for the parity commands.

use anyhow::Result;

#[derive(Debug, Clone, Copy, Default)]
pub struct DiffStats {
    pub count: usize,
    pub max: f32,
    pub mean: f64,
}

impl DiffStats {
    pub(crate) fn push(&mut self, diff: f32) {
        self.count += 1;
        self.max = self.max.max(diff);
        self.mean += diff as f64;
    }

    pub(crate) fn finish(mut self) -> Self {
        if self.count > 0 {
            self.mean /= self.count as f64;
        }
        self
    }
}

pub fn diff_f32(got: &[f32], want: &[f32], label: &str) -> Result<DiffStats> {
    anyhow::ensure!(
        got.len() == want.len(),
        "{label}: length mismatch got {} want {}",
        got.len(),
        want.len()
    );
    let mut stats = DiffStats::default();
    for (a, b) in got.iter().zip(want) {
        stats.push((a - b).abs());
    }
    Ok(stats.finish())
}

