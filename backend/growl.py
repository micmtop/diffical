"""咆哮效果处理器（参考 hifisampler / HiFiUTAU-engine）

真正的咆哮 = 高频段（>400Hz）的方波 LFO 音高调制（强度 × ±100 cents），
而不是幅度调制。旧实现用 ±4% AM，听感几乎无效果（曲线"画了等于没画"）。
"""

import numpy as np
from scipy.signal import butter, sosfilt

_MAX_VIBRATO_CENTS = 100.0
_HP_CUTOFF_HZ = 20.0
_GROWL_LOW_HZ = 400.0
_MIN_NYQ_FRAC = 0.01


def _highpass(audio: np.ndarray, sr: int, cutoff: float) -> tuple:
    nyq = sr / 2
    norm = np.clip(cutoff / nyq, _MIN_NYQ_FRAC, 0.99)
    sos = butter(4, norm, "high", output="sos")
    high = sosfilt(sos, audio)
    return np.asarray(high), np.asarray(audio - high)


def _square_lfo(n_samples: int, sr: int, freq: np.ndarray) -> np.ndarray:
    t = np.arange(n_samples) / sr
    phase = np.cumsum(2 * np.pi * freq) / sr
    return np.sign(np.sin(phase))


def _apply_pitch_modulation(band: np.ndarray, sr: int,
                            lfo: np.ndarray, strength: np.ndarray) -> np.ndarray:
    cents = lfo * strength * _MAX_VIBRATO_CENTS
    ratio = 2 ** (cents / 1200.0)
    cumsum = np.cumsum(ratio) - ratio[0]
    ideal = np.arange(len(band)) * np.mean(ratio)
    drift = cumsum - ideal
    if len(band) > 100:
        hp_norm = _HP_CUTOFF_HZ / (sr / 2)
        drift = sosfilt(butter(2, hp_norm, "high", output="sos"), drift)
    idx = np.clip(np.arange(len(band)) + drift, 0, len(band) - 1)
    modulated = np.interp(idx, np.arange(len(band)), band)
    rms_orig = np.sqrt(np.mean(band ** 2))
    rms_new = np.sqrt(np.mean(modulated ** 2))
    if rms_new > 1e-10:
        modulated *= rms_orig / rms_new
    return modulated


def apply_growl(audio: np.ndarray, growl_level: np.ndarray, sample_rate: int,
                f0_curve: np.ndarray = None, hop_length: int = 512) -> np.ndarray:
    """应用咆哮效果（高频段音高调制，hifisampler 式）

    Args:
        audio: 输入音频 (n_samples,)
        growl_level: 咆哮等级曲线 (n_frames,) 0-100
        sample_rate: 采样率
        f0_curve: F0 曲线 (n_frames,)，用于调制咆哮频率（跟随音高）
        hop_length: 参数曲线的帧移

    Returns:
        应用咆哮效果后的音频
    """
    if len(audio) == 0 or len(growl_level) == 0:
        return audio.copy()
    if np.allclose(growl_level, 0, atol=0.5):
        return audio.copy()

    n_samples = len(audio)
    strength = _interpolate_parameter(growl_level, n_samples, hop_length, sample_rate)
    strength = np.clip(strength / 100.0, 0.0, 1.0)
    if np.max(strength) < 0.01:
        return audio.copy()

    # LFO 频率：跟随 F0（0.25×f0，限 30-200Hz）或固定 80Hz
    if f0_curve is not None and len(f0_curve) > 0:
        f0_samples = _interpolate_parameter(f0_curve, n_samples, hop_length, sample_rate)
        f0_samples = np.where(f0_samples > 50, f0_samples, 80.0)
        lfo_freq = np.clip(f0_samples * 0.25, 30.0, 200.0)
    else:
        lfo_freq = np.full(n_samples, 80.0, dtype=np.float32)

    high, low = _highpass(audio, sample_rate, _GROWL_LOW_HZ)
    lfo = _square_lfo(n_samples, sample_rate, lfo_freq)
    high_mod = _apply_pitch_modulation(high, sample_rate, lfo, strength)

    return (low + high_mod).astype(np.float32)


def _interpolate_parameter(param_frames: np.ndarray, n_samples: int,
                           hop_length: int, sample_rate: int) -> np.ndarray:
    """将帧级参数插值到样本级"""
    from scipy.interpolate import interp1d

    if len(param_frames) == 0:
        return np.zeros(n_samples, dtype=np.float32)
    if len(param_frames) == 1:
        return np.full(n_samples, param_frames[0], dtype=np.float32)

    frame_times = np.arange(len(param_frames)) * hop_length / sample_rate
    sample_times = np.arange(n_samples) / sample_rate

    f = interp1d(frame_times, param_frames, kind='cubic',
                 bounds_error=False, fill_value='extrapolate')
    return f(sample_times).astype(np.float32)
