//! Audio/frame geometry shared by the model render path.
//!
//! Values mirror `docs/config.yaml`. They are intentionally a struct rather
//! than scattered constants so a future YAML loader can replace `Default`.

#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub hop_size: usize,
    pub origin_hop_size: usize,
    pub n_fft: usize,
    pub win_size: usize,
    pub n_mels: usize,
    pub mel_fmin: f32,
    pub mel_fmax: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            hop_size: 512,
            origin_hop_size: 128,
            n_fft: 2048,
            win_size: 2048,
            n_mels: 128,
            mel_fmin: 40.0,
            mel_fmax: 16_000.0,
        }
    }
}

impl AudioConfig {
    /// Vocoder frame duration in milliseconds (hop 512 at 44.1 kHz).
    pub fn frame_ms(&self) -> f64 {
        self.hop_size as f64 / self.sample_rate as f64 * 1000.0
    }

    /// Source mel frame duration in milliseconds (hop 128 at 44.1 kHz).
    pub fn origin_frame_ms(&self) -> f64 {
        self.origin_hop_size as f64 / self.sample_rate as f64 * 1000.0
    }
}
