//! Waveform-domain flag effects: `Ht` spectral tilt, `HG` growl, `A` pitch AM.
//!
//! Only `HG` has a Python counterpart (`backend/growl.py`); it is reimplemented
//! here with the same signal chain (400 Hz 4th-order high-pass, square LFO
//! following 0.25 * F0, +/-100 cent band modulation, drift high-pass, per-band
//! RMS restore). `Ht` and `A` are our own documented semantics: see FLAGS.md.

use crate::biquad::Sos;

const MAX_VIBRATO_CENTS: f64 = 100.0;
const GROWL_LOW_HZ: f64 = 400.0;
const DRIFT_HP_HZ: f64 = 20.0;
const GROWL_FOLLOW: f64 = 0.25;
const GROWL_LFO_MIN_HZ: f64 = 30.0;
const GROWL_LFO_MAX_HZ: f64 = 200.0;
const GROWL_LFO_FALLBACK_HZ: f64 = 80.0;

/// Linear interpolation over frame values with linear extrapolation at both
/// ends (the sample clock runs from 0 to `frames - 1` in frame units).
fn interp_frames(values: &[f32], pos: f64) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return values[0] as f64;
    }
    let last = (n - 1) as f64;
    if pos <= 0.0 {
        return values[0] as f64 + (values[1] as f64 - values[0] as f64) * pos;
    }
    if pos >= last {
        let t = pos - last;
        return values[n - 1] as f64 + (values[n - 1] as f64 - values[n - 2] as f64) * t;
    }
    let i = pos.floor() as usize;
    let frac = pos - i as f64;
    if i >= n - 1 {
        return values[n - 1] as f64;
    }
    values[i] as f64 * (1.0 - frac) + values[i + 1] as f64 * frac
}

fn interp_samples(values: &[f64], pos: f64) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return values[0];
    }
    if pos <= 0.0 {
        return values[0];
    }
    if pos >= (n - 1) as f64 {
        return values[n - 1];
    }
    let i = pos.floor() as usize;
    let frac = pos - i as f64;
    values[i] * (1.0 - frac) + values[i + 1] * frac
}

fn rms(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().map(|v| (*v as f64).powi(2)).sum();
    (sum / values.len() as f64).sqrt()
}

/// `Ht` tension as a dynamic spectral tilt around a 400 Hz crossover.
///
/// Positive Ht crossfades towards a 2nd-order Butterworth high-pass, negative
/// Ht towards the complementary low-pass, at
/// `cutoff = 150 + 250 * |Ht|/100` Hz (the reference's f0-less cutoff formula).
/// The crossfade weight is `|Ht|/100`, so Ht = 0 returns the input untouched.
pub fn tension_tilt(input: &[f32], tension: f64, sample_rate: u32) -> Vec<f32> {
    if input.is_empty() || tension == 0.0 || !tension.is_finite() {
        return input.to_vec();
    }
    let strength = (tension.abs() / 100.0).clamp(0.0, 1.0);
    let fs = sample_rate as f64;
    let cutoff = (150.0 + strength * 250.0).clamp(60.0, fs * 0.45);
    let filtered = if tension > 0.0 {
        Sos::highpass(2, cutoff, fs).apply(input)
    } else {
        Sos::lowpass(2, cutoff, fs).apply(input)
    };
    input
        .iter()
        .zip(&filtered)
        .map(|(dry, wet)| (*dry as f64 * (1.0 - strength) + *wet as f64 * strength) as f32)
        .collect()
}

/// `HG` growl: square-LFO pitch modulation of the > 400 Hz band.
///
/// Returns `None` when the flag is inactive (max level < 0.5) so callers can
/// keep the phrase bit-identical.
pub fn growl(
    samples: &[f32],
    growl_per_frame: &[f32],
    f0: &[f32],
    sample_rate: u32,
    hop: usize,
) -> Option<Vec<f32>> {
    let n = samples.len();
    if n == 0 || growl_per_frame.is_empty() || hop == 0 {
        return None;
    }
    let peak = growl_per_frame.iter().fold(0f32, |a, b| a.max(*b));
    if peak < 0.5 {
        return None;
    }
    let fs = sample_rate as f64;

    // Per-sample strength and LFO frequency from the frame curves.
    let mut strength = vec![0f64; n];
    let mut lfo_freq = vec![GROWL_LFO_FALLBACK_HZ; n];
    for (i, (s, f)) in strength.iter_mut().zip(lfo_freq.iter_mut()).enumerate() {
        let pos = i as f64 / hop as f64;
        *s = (interp_frames(growl_per_frame, pos) / 100.0).clamp(0.0, 1.0);
        let f0_hz = interp_frames(f0, pos);
        if f0_hz > 50.0 {
            *f = (f0_hz * GROWL_FOLLOW).clamp(GROWL_LFO_MIN_HZ, GROWL_LFO_MAX_HZ);
        }
    }
    if strength.iter().fold(0f64, |a, b| a.max(*b)) < 0.01 {
        return None;
    }

    // Split at 400 Hz; only the high band is modulated.
    let high = Sos::highpass(4, GROWL_LOW_HZ, fs).apply(samples);
    let low: Vec<f32> = samples
        .iter()
        .zip(&high)
        .map(|(s, h)| s - h)
        .collect();

    // Square LFO: sign(sin(cumsum(2*pi*f)/sr)).
    let mut phase = 0f64;
    let lfo: Vec<f64> = lfo_freq
        .iter()
        .map(|f| {
            phase += 2.0 * std::f64::consts::PI * f;
            let s = (phase / fs).sin();
            if s > 0.0 {
                1.0
            } else if s < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect();

    // Band pitch modulation with a low-frequency drift compensation.
    let ratio: Vec<f64> = lfo
        .iter()
        .zip(&strength)
        .map(|(l, s)| 2f64.powf(l * s * MAX_VIBRATO_CENTS / 1200.0))
        .collect();
    let mean_ratio = ratio.iter().sum::<f64>() / n as f64;
    let mut cumsum = 0f64;
    let mut drift = vec![0f64; n];
    for (i, r) in ratio.iter().enumerate() {
        cumsum += r;
        drift[i] = cumsum - ratio[0] - i as f64 * mean_ratio;
    }
    if n > 100 {
        drift = Sos::highpass(2, DRIFT_HP_HZ, fs)
            .apply(&drift.iter().map(|v| *v as f32).collect::<Vec<_>>())
            .into_iter()
            .map(|v| v as f64)
            .collect();
    }
    let high_f64: Vec<f64> = high.iter().map(|v| *v as f64).collect();
    let mut modulated = vec![0f32; n];
    for (i, slot) in modulated.iter_mut().enumerate() {
        let idx = (i as f64 + drift[i]).clamp(0.0, (n - 1) as f64);
        *slot = interp_samples(&high_f64, idx) as f32;
    }
    let orig = rms(&high);
    let new = rms(&modulated);
    if new > 1e-10 {
        let gain = orig / new;
        for v in &mut modulated {
            *v = (*v as f64 * gain) as f32;
        }
    }
    Some(
        low.iter()
            .zip(&modulated)
            .map(|(l, h)| l + h)
            .collect(),
    )
}

/// `A`: level-preserving amplitude modulation following the local pitch slope.
///
/// For every voiced frame the slope `s` of `log2(f0)` is measured in octaves
/// per second (central difference, one-sided at the phrase edges, 0 for
/// unvoiced frames). The raw per-frame gain is
/// `1 + (A/100) * 0.5 * clamp(s / 2, -1, 1)`, so |slope| >= 2 oct/s reaches
/// +/-50 % at A = +/-100. The gain curve is interpolated linearly to samples.
///
/// The stage is level-preserving: one scalar compensation restores the
/// voiced-region RMS to its pre-`A` value, and the shaped signal is then
/// peak-limited to the pre-`A` peak (documented limiter). Keeping the peak
/// equal matters because the downstream phrase RMS normalization is
/// peak-limited on loud material; with both RMS and peak unchanged the final
/// phrase level is unchanged too (measured RMS change < 1 %). The limiter only
/// engages on the loudest samples whose gain is above 1.
///
/// If the phrase has no voiced samples the input is returned. `A = 0` returns
/// `None` (the caller never allocates a curve).
pub fn pitch_amplitude(
    samples: &[f32],
    amount_per_frame: &[f32],
    f0: &[f32],
    sample_rate: u32,
    hop: usize,
) -> Option<Vec<f32>> {
    if samples.is_empty()
        || amount_per_frame.is_empty()
        || f0.len() < 2
        || hop == 0
        || amount_per_frame.iter().all(|v| *v == 0.0)
    {
        return None;
    }
    let frame_sec = hop as f64 / sample_rate as f64;
    let mut gain = vec![0f64; f0.len()];
    for i in 0..f0.len() {
        let amount = amount_per_frame.get(i).copied().unwrap_or(0.0) as f64;
        if amount == 0.0 || !amount.is_finite() {
            gain[i] = 1.0;
            continue;
        }
        let here = f0[i] as f64;
        if here <= 50.0 {
            gain[i] = 1.0;
            continue;
        }
        let prev = if i > 0 { f0[i - 1] as f64 } else { 0.0 };
        let next = if i + 1 < f0.len() { f0[i + 1] as f64 } else { 0.0 };
        let slope = if prev > 50.0 && next > 50.0 {
            (next.log2() - prev.log2()) / (2.0 * frame_sec)
        } else if next > 50.0 {
            (next.log2() - here.log2()) / frame_sec
        } else if prev > 50.0 {
            (here.log2() - prev.log2()) / frame_sec
        } else {
            0.0
        };
        let normalized = (slope / 2.0).clamp(-1.0, 1.0);
        gain[i] = 1.0 + (amount / 100.0) * 0.5 * normalized;
    }
    let gain_f32: Vec<f32> = gain.iter().map(|v| *v as f32).collect();
    let n = samples.len();
    let frame_of = |i: usize| (i / hop).min(f0.len() - 1);
    let mut out = Vec::with_capacity(n);
    let mut before_sq = 0f64;
    let mut after_sq = 0f64;
    for i in 0..n {
        let shaped = samples[i] as f64 * interp_frames(&gain_f32, i as f64 / hop as f64);
        if f0[frame_of(i)] > 50.0 {
            before_sq += (samples[i] as f64).powi(2);
            after_sq += shaped * shaped;
        }
        out.push(shaped as f32);
    }
    if before_sq <= 0.0 || after_sq <= 1e-20 {
        return Some(samples.to_vec());
    }
    // Restore the voiced RMS, then keep the peak at the pre-A level.
    let compensation = (before_sq / after_sq).sqrt() as f32;
    let ceiling = samples.iter().fold(0f32, |a, v| a.max(v.abs()));
    for value in &mut out {
        let scaled = *value * compensation;
        *value = scaled.clamp(-ceiling, ceiling);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{growl, pitch_amplitude, rms, tension_tilt};

    #[test]
    fn tension_zero_is_identity() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
        assert_eq!(tension_tilt(&input, 0.0, 44_100), input);
    }

    #[test]
    fn tension_positive_removes_low_energy() {
        let fs = 44_100u32;
        let n = 44_100;
        let low: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / fs as f32).sin())
            .collect();
        let out = tension_tilt(&low, 100.0, fs);
        let e_in: f64 = low.iter().map(|v| (*v as f64).powi(2)).sum();
        let e_out: f64 = out.iter().map(|v| (*v as f64).powi(2)).sum();
        assert!(e_out < e_in * 0.1, "low band should be attenuated");
    }

    #[test]
    fn growl_inactive_below_threshold() {
        let samples = vec![0.1f32; 1000];
        assert!(growl(&samples, &[0.0, 0.4], &[220.0, 220.0], 44_100, 512).is_none());
        assert!(growl(&samples, &[], &[], 44_100, 512).is_none());
    }

    #[test]
    fn growl_changes_pitch_not_level() {
        let fs = 44_100u32;
        let n = 44_100;
        let samples: Vec<f32> = (0..n)
            .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / fs as f32).sin())
            .collect();
        let out = growl(&samples, &vec![80.0; 90], &vec![220.0; 90], fs, 512).unwrap();
        assert_eq!(out.len(), samples.len());
        let r_in = (samples.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
        let r_out = (out.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
        assert!((r_out / r_in - 1.0).abs() < 0.05, "RMS should be restored");
        let diff: f64 = samples
            .iter()
            .zip(&out)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .sum::<f64>();
        assert!(diff > 1.0, "growl should change the waveform");
    }

    #[test]
    fn amplitude_follows_slope_and_preserves_level_and_peak() {
        let fs = 44_100u32;
        let frames = 72;
        let half_frames = frames / 2;
        let flat_then = |rise: bool| -> Vec<f32> {
            (0..frames)
                .map(|i| {
                    if i < half_frames {
                        220.0
                    } else {
                        let t = (i - half_frames) as f32 / (half_frames - 1) as f32;
                        if rise {
                            220.0 * 2f32.powf(t)
                        } else {
                            440.0 * 2f32.powf(-t)
                        }
                    }
                })
                .collect()
        };
        // Quiet second half: the global peak sits in the flat (gain 1) region,
        // so the peak guard does not have to remove the modulation.
        let n = frames * 512;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let amp = if i < n / 2 { 0.5 } else { 0.2 };
                amp * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / fs as f32).sin()
            })
            .collect();
        let amount = vec![100.0f32; frames];
        let rising = flat_then(true);
        let out = pitch_amplitude(&samples, &amount, &rising, fs, 512).unwrap();
        let level_in = rms(&samples);
        let level_out = rms(&out);
        assert!(
            (level_out / level_in - 1.0).abs() < 0.01,
            "level should be restored: {level_out} vs {level_in}"
        );
        let peak_in = samples.iter().fold(0f32, |a, v| a.max(v.abs()));
        let peak_out = out.iter().fold(0f32, |a, v| a.max(v.abs()));
        assert!(
            peak_out <= peak_in * 1.001,
            "peak must not grow: {peak_out} vs {peak_in}"
        );
        let half = n / 2;
        let gain_low = rms(&out[..half]) / rms(&samples[..half]);
        let gain_high = rms(&out[half..]) / rms(&samples[half..]);
        assert!(
            gain_high > gain_low * 1.15,
            "rising second half should be louder: {gain_high} vs {gain_low}"
        );
        let falling = flat_then(false);
        let down = pitch_amplitude(&samples, &amount, &falling, fs, 512).unwrap();
        let fall_low = rms(&down[..half]) / rms(&samples[..half]);
        let fall_high = rms(&down[half..]) / rms(&samples[half..]);
        assert!(
            fall_high < fall_low * 0.87,
            "falling second half should be quieter: {fall_high} vs {fall_low}"
        );
        // A = 0 is inert (caller never invokes it, but the function must agree).
        assert!(pitch_amplitude(&samples, &vec![0.0; frames], &rising, fs, 512).is_none());
    }
}