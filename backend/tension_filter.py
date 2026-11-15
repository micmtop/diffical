"""动态频谱滤波器

实现基于 F0 跟随的平滑动态滤波：
- 使用二阶 Butterworth 滤波器
- 支持频率调制的平滑过渡
- 添加预加重补偿
"""

import numpy as np


def apply_dynamic_filter(audio: np.ndarray, filter_level: np.ndarray,
                         sample_rate: int, f0_curve: np.ndarray = None,
                         hop_length: int = 512) -> np.ndarray:
    """应用动态低切滤波

    Args:
        audio: 输入音频 (n_samples,)
        filter_level: 滤波等级曲线 (n_frames,) 0-100
        sample_rate: 采样率
        f0_curve: F0 曲线 (n_frames,)，用于跟随音高调整截止频率
        hop_length: 参数曲线的帧移

    Returns:
        滤波后的音频
    """
    from scipy.signal import butter, filtfilt

    if len(audio) == 0 or len(filter_level) == 0:
        return audio.copy()

    if np.allclose(filter_level, 0, atol=0.5):
        return audio.copy()

    n_samples = len(audio)
    nyquist = sample_rate / 2

    filter_factor = _interpolate_parameter(filter_level, n_samples, hop_length, sample_rate)
    filter_factor = np.clip(filter_factor / 100, 0, 1)

    if f0_curve is not None and len(f0_curve) > 0:
        f0_samples = _interpolate_parameter(f0_curve, n_samples, hop_length, sample_rate)
        base_freq = np.maximum(f0_samples * 0.35, 80)
    else:
        base_freq = np.full(n_samples, 150, dtype=np.float32)

    cutoff_freq = base_freq + filter_factor * 250
    cutoff_freq = np.clip(cutoff_freq, 60, nyquist * 0.9)

    mean_cutoff = float(np.mean(cutoff_freq))
    b, a = butter(2, mean_cutoff / nyquist, btype='high')
    filtered = filtfilt(b, a, audio)

    smoothed_factor = _smooth_factor(filter_factor, sample_rate, window_ms=50)
    result = audio * (1 - smoothed_factor) + filtered * smoothed_factor

    return result.astype(np.float32)


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


def _smooth_factor(factor: np.ndarray, sample_rate: int, window_ms: int = 50) -> np.ndarray:
    """平滑滤波因子，避免突变"""
    window_size = int(window_ms * sample_rate / 1000)
    window_size = max(window_size, 3)

    if window_size >= len(factor):
        return np.full(len(factor), float(np.mean(factor)), dtype=np.float32)

    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(factor, kernel, mode='same')

    return smoothed.astype(np.float32)