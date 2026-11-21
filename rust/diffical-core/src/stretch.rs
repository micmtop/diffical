//! Source-time mapping for the per-phone stretch, including the `He` loop.
//!
//! Consonant frames are mapped at native speed; vowel frames either stretch
//! linearly, freeze near the end of the source (long-note hold), or ping-pong
//! through the stable part of the source vowel when `He` asks for it.

/// Everything the mapping needs, computed once per phone.
pub struct StretchMap {
    pub vel: f64,
    pub start_ms: f64,
    pub con_ms: f64,
    pub con_vel: f64,
    pub scaling_ratio: f64,
    pub stretch_length_ms: f64,
    pub vowel_output_ms: f64,
    pub use_hold: bool,
    pub use_loop: bool,
    pub loop_start_ms: f64,
    pub loop_span_ms: f64,
    pub loop_cycle_ms: f64,
    pub onset_src: f64,
    pub onset_tgt: f64,
}

impl StretchMap {
    /// Source time (ms) for a target clock (ms since the source origin).
    pub fn source_ms(&self, clock: f64) -> f64 {
        if clock < self.con_vel {
            return self.start_ms + clock / self.vel;
        }
        if self.use_loop {
            let rel = (clock - self.con_vel).max(0.0);
            let phase = rel.rem_euclid(self.loop_cycle_ms) / self.loop_cycle_ms;
            let tri = if phase <= 0.5 {
                phase * 2.0
            } else {
                2.0 - phase * 2.0
            };
            return self.loop_start_ms + tri * self.loop_span_ms;
        }
        let linear = self.con_ms + (clock - self.con_vel) / self.scaling_ratio;
        if !self.use_hold {
            return linear;
        }
        let rel = clock - self.con_vel;
        if rel < self.onset_tgt {
            self.con_ms + (rel / self.onset_tgt.max(1e-9)) * self.onset_src
        } else {
            let hold_tgt = (self.vowel_output_ms - self.onset_tgt).max(1e-9);
            let drift = ((rel - self.onset_tgt).max(0.0) / hold_tgt).min(1.0);
            self.con_ms + self.stretch_length_ms * (0.45 + drift * 0.52)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StretchMap;

    fn map(use_hold: bool, use_loop: bool) -> StretchMap {
        StretchMap {
            vel: 1.0,
            start_ms: 0.0,
            con_ms: 100.0,
            con_vel: 100.0,
            scaling_ratio: 2.0,
            stretch_length_ms: 400.0,
            vowel_output_ms: 1800.0,
            use_hold,
            use_loop,
            loop_start_ms: 240.0,
            loop_span_ms: 228.0,
            loop_cycle_ms: 3000.0,
            onset_src: 180.0,
            onset_tgt: 450.0,
        }
    }

    #[test]
    fn consonant_is_not_stretched() {
        let m = map(true, false);
        assert!((m.source_ms(50.0) - 50.0).abs() < 1e-9);
    }

    #[test]
    fn linear_vowel_stretches() {
        let m = map(false, false);
        assert!((m.source_ms(300.0) - (100.0 + 100.0)).abs() < 1e-9);
    }

    #[test]
    fn hold_freezes_near_the_source_end() {
        let m = map(true, false);
        let a = m.source_ms(600.0);
        let b = m.source_ms(1500.0);
        assert!(a >= 100.0 + 0.45 * 400.0 - 1e-9);
        assert!(b <= 100.0 + 0.97 * 400.0 + 1e-9);
        assert!(b > a);
    }

    #[test]
    fn he_loop_ping_pongs_inside_the_stable_vowel() {
        let m = map(true, true);
        let quarter = m.source_ms(100.0 + 750.0);
        let half = m.source_ms(100.0 + 1500.0);
        assert!((m.source_ms(100.0) - 240.0).abs() < 1e-9);
        assert!((quarter - (240.0 + 114.0)).abs() < 1e-9);
        assert!((half - (240.0 + 228.0)).abs() < 1e-9);
        // Back to the start at the end of the cycle, continuous everywhere.
        assert!((m.source_ms(100.0 + 3000.0) - 240.0).abs() < 1e-9);
        let mut prev = m.source_ms(100.0);
        for step in 0..3000 {
            let value = m.source_ms(100.0 + step as f64);
            assert!((value - prev).abs() < 1.0, "loop must be continuous");
            assert!(value >= 240.0 - 1e-9 && value <= 240.0 + 228.0 + 1e-9);
            prev = value;
        }
    }
}
