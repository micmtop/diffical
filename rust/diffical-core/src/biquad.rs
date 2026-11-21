//! Second-order sections and Butterworth designs for the flag effects.
//!
//! Coefficients are the standard bilinear-transform biquads; the cascade
//! response matches a scipy `butter(2|4, ..., output="sos")` design (section
//! gain split differs, the product is the same rational function). Processing
//! runs in `f64` and casts back, like scipy does on `float32` input.

const Q2: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// Butterworth pole Qs for a 4th-order cascade.
const Q4: [f64; 2] = [0.541_196_100_146_196_9, 1.306_562_964_876_376_6];

#[derive(Debug, Clone)]
pub struct Sos {
    /// `[b0, b1, b2, a1, a2]`, normalized so `a0 = 1`.
    sections: Vec<[f64; 5]>,
}

fn biquad(kind: Kind, fc: f64, q: f64, fs: f64) -> [f64; 5] {
    let nyq = fs / 2.0;
    let fc = fc.clamp(0.01, nyq * 0.99);
    let w0 = 2.0 * std::f64::consts::PI * fc / fs;
    let (sin, cos) = w0.sin_cos();
    let alpha = sin / (2.0 * q);
    let a0 = 1.0 + alpha;
    let (b0, b1, b2) = match kind {
        Kind::LowPass => ((1.0 - cos) / 2.0, 1.0 - cos, (1.0 - cos) / 2.0),
        Kind::HighPass => ((1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0),
    };
    [b0 / a0, b1 / a0, b2 / a0, -2.0 * cos / a0, (1.0 - alpha) / a0]
}

#[derive(Clone, Copy)]
enum Kind {
    LowPass,
    HighPass,
}

impl Sos {
    fn build(kind: Kind, order: usize, fc: f64, fs: f64) -> Self {
        let qs: Vec<f64> = match order {
            2 => vec![Q2],
            4 => Q4.to_vec(),
            other => {
                // General even-order Butterworth pole Qs.
                (0..other / 2)
                    .map(|k| {
                        let theta = (2 * k + 1) as f64 * std::f64::consts::PI / (2.0 * other as f64);
                        1.0 / (2.0 * theta.cos())
                    })
                    .collect()
            }
        };
        Self {
            sections: qs.into_iter().map(|q| biquad(kind, fc, q, fs)).collect(),
        }
    }

    pub fn highpass(order: usize, fc: f64, fs: f64) -> Self {
        Self::build(Kind::HighPass, order.max(2), fc, fs)
    }

    pub fn lowpass(order: usize, fc: f64, fs: f64) -> Self {
        Self::build(Kind::LowPass, order.max(2), fc, fs)
    }

    /// `sosfilt`: one causal pass, section by section.
    pub fn apply(&self, input: &[f32]) -> Vec<f32> {
        let mut buf: Vec<f64> = input.iter().map(|v| *v as f64).collect();
        for [b0, b1, b2, a1, a2] in &self.sections {
            let (mut z1, mut z2) = (0.0f64, 0.0f64);
            for x in &mut buf {
                let y = b0 * *x + z1;
                z1 = b1 * *x - a1 * y + z2;
                z2 = b2 * *x - a2 * y;
                *x = y;
            }
        }
        buf.into_iter().map(|v| v as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Sos;

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
    }

    #[test]
    fn highpass_keeps_highs_and_removes_dc() {
        let fs = 44_100.0;
        let n = 44_100;
        let low: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / fs as f32).sin())
            .collect();
        let high: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 4000.0 * i as f32 / fs as f32).sin())
            .collect();
        let hp = Sos::highpass(4, 400.0, fs);
        assert!(rms(&hp.apply(&low)) < 0.02, "100 Hz should be attenuated");
        assert!((rms(&hp.apply(&high)) - rms(&high)).abs() < 0.01);
        // DC is removed (steady state).
        let dc = vec![0.5f32; 20_000];
        let out = hp.apply(&dc);
        assert!(out[19_999].abs() < 1e-3);
    }

    #[test]
    fn lowpass_keeps_lows_and_removes_highs() {
        let fs = 44_100.0;
        let n = 44_100;
        let low: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / fs as f32).sin())
            .collect();
        let high: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 4000.0 * i as f32 / fs as f32).sin())
            .collect();
        let lp = Sos::lowpass(2, 1000.0, fs);
        assert!((rms(&lp.apply(&low)) - rms(&low)).abs() < 0.02);
        assert!(rms(&lp.apply(&high)) < 0.1);
    }
}
