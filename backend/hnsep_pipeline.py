"""谐波/噪声分离与参数处理管线

采用改进的谱减法实现谐波/噪声分离，支持多维度参数控制：
- 气声增强：调节噪声分量增益
- 张力控制：动态低切滤波
- 浊音度：谐波/噪声能量比调整
- 低频气声：增强低频噪声
- 高频气声：增强高频噪声
"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def separate_harmonic_noise(audio: np.ndarray, sample_rate: int = 44100,
                            n_fft: int = 2048, hop_length: int = 512,
                            harmonic_ratio: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """高质量谐波/噪声分离

    使用多频段自适应谱减法 + 谐波乘积谱实现高质量分离：
    1. STFT 获取幅度谱和相位谱
    2. 多频段噪声估计（不同频段使用不同的平滑参数）
    3. 谐波分量检测（基于谐波乘积谱）
    4. 自适应谱减法分离谐波和噪声
    5. 时间平滑避免音乐噪声

    Args:
        audio: 输入音频 (n_samples,)
        sample_rate: 采样率
        n_fft: FFT 大小
        hop_length: 帧移
        harmonic_ratio: 谐波分量比例系数

    Returns:
        (harmonic, noise) 两个时域信号
    """
    from scipy import signal
    from scipy.ndimage import grey_opening, gaussian_filter1d

    if len(audio) < n_fft:
        return audio.copy(), np.zeros_like(audio)

    f, t, Zxx = signal.stft(audio, fs=sample_rate, nperseg=n_fft,
                            noverlap=n_fft - hop_length, nfft=n_fft)

    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    n_freqs = mag.shape[0]
    n_frames = mag.shape[1]

    harmonic_mag = np.zeros_like(mag)
    noise_mag = np.zeros_like(mag)

    freq_bands = [
        (0, int(n_freqs * 0.1), 5, 1.3),
        (int(n_freqs * 0.1), int(n_freqs * 0.3), 3, 1.4),
        (int(n_freqs * 0.3), int(n_freqs * 0.5), 3, 1.5),
        (int(n_freqs * 0.5), int(n_freqs * 0.7), 3, 1.6),
        (int(n_freqs * 0.7), n_freqs, 1, 1.8),
    ]

    for i in range(n_frames):
        spec = mag[:, i].copy()

        noise_est = np.zeros(n_freqs)
        for start, end, kernel_size, alpha in freq_bands:
            if start < end:
                noise_est[start:end] = signal.medfilt(spec[start:end], kernel_size=kernel_size)

        noise_est = gaussian_filter1d(noise_est, sigma=3.0)

        alpha = 1.4
        beta = 0.05
        mask = spec > alpha * noise_est

        harmonic_spec = np.where(mask, spec - beta * noise_est, 0.0)

        struct = np.ones((7,))
        harmonic_peaks = grey_opening(harmonic_spec, structure=struct)

        harmonic_mag[:, i] = harmonic_peaks * harmonic_ratio
        noise_mag[:, i] = spec - harmonic_peaks * harmonic_ratio

    noise_mag = np.maximum(noise_mag, 0.0005 * mag)

    harmonic_mag_smooth = gaussian_filter1d(harmonic_mag, sigma=0.3, axis=1)
    noise_mag_smooth = gaussian_filter1d(noise_mag, sigma=0.5, axis=1)

    harmonic_mag_smooth = np.maximum(harmonic_mag_smooth, 0)
    noise_mag_smooth = np.maximum(noise_mag_smooth, 0)

    _, harmonic = signal.istft(harmonic_mag_smooth * np.exp(1j * phase),
                               fs=sample_rate, nperseg=n_fft,
                               noverlap=n_fft - hop_length, nfft=n_fft)
    _, noise = signal.istft(noise_mag_smooth * np.exp(1j * phase),
                            fs=sample_rate, nperseg=n_fft,
                            noverlap=n_fft - hop_length, nfft=n_fft)

    min_len = min(len(audio), len(harmonic), len(noise))
    harmonic = harmonic[:min_len].astype(np.float32)
    noise = noise[:min_len].astype(np.float32)

    combined = harmonic + noise
    combined_peak = np.max(np.abs(combined))
    original_peak = np.max(np.abs(audio[:min_len]))
    if combined_peak > original_peak * 1.1:
        scale = (original_peak * 1.1) / combined_peak
        harmonic = harmonic * scale
        noise = noise * scale

    return harmonic, noise


def crossfade_segments(segments: list, fade_ms: float = 30.0,
                       sample_rate: int = 44100) -> np.ndarray:
    """音频段交叉淡入淡出拼接"""
    if not segments:
        return np.array([], dtype=np.float32)

    if len(segments) == 1:
        return segments[0].copy()

    fade_len = int(fade_ms * sample_rate / 1000)
    result = segments[0].copy()

    for i in range(1, len(segments)):
        prev = result
        curr = segments[i]

        overlap_len = min(fade_len, len(prev), len(curr))
        if overlap_len <= 0:
            result = np.concatenate([prev, curr])
            continue

        fade_out = np.cos(np.linspace(0, np.pi, overlap_len)) * 0.5 + 0.5
        fade_in = 1.0 - fade_out

        combined = np.zeros(len(prev) + len(curr) - overlap_len, dtype=np.float32)
        combined[:len(prev) - overlap_len] = prev[:len(prev) - overlap_len]
        combined[len(prev) - overlap_len:len(prev)] = (
            prev[-overlap_len:] * fade_out + curr[:overlap_len] * fade_in
        )
        combined[len(prev):] = curr[overlap_len:]
        result = combined

    return result


class HarmonicNoiseProcessor:
    """谐波/噪声处理器

    提供完整的谐波/噪声分离和参数处理能力。
    """

    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048,
                 hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def separate(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """分离音频为谐波和噪声分量"""
        return separate_harmonic_noise(audio, self.sample_rate,
                                       self.n_fft, self.hop_length)

    def process_parameters(self, harmonic: np.ndarray, noise: np.ndarray,
                          breath_level: np.ndarray = None,
                          tension_level: np.ndarray = None,
                          voicing_level: np.ndarray = None,
                          low_freq_breath: np.ndarray = None,
                          high_freq_breath: np.ndarray = None,
                          noise_factor: float = 0.5):
        """应用动态参数处理

        Args:
            harmonic: 谐波分量 (n_samples,)
            noise: 噪声分量 (n_samples,)
            breath_level: 气声等级 (n_frames,) 0-100
            tension_level: 张力等级 (n_frames,) 0-100
            voicing_level: 浊音度等级 (n_frames,) 0-100
            low_freq_breath: 低频气声 (n_frames,) 0-100
            high_freq_breath: 高频气声 (n_frames,) 0-100
            noise_factor: 噪声分量比例（0-1），辅音用高值(0.6-0.8)保留清辅音特征，元音用低值(0.3)减少杂音

        Returns:
            (processed_harmonic, processed_noise)
        """
        processed_h = harmonic.copy()
        processed_n = noise.copy()
        n_samples = len(processed_h)

        if breath_level is not None and len(breath_level) > 0:
            breath_factor = self._interpolate_param(breath_level, n_samples)
            # 默认 0 变成 -0.5 -> 噪声被清零（默认渲染音质烂的根因之一）。
            # 修正：0 = 中性（不改噪声），100 = 最大气声。
            breath_factor = breath_factor / 100.0
            breath_factor = np.clip(breath_factor, 0.0, 1.0)
            processed_n = processed_n * (1 + breath_factor * 2)

        # 根据音素类型调整噪声分量比例
        processed_n = processed_n * noise_factor

        if tension_level is not None and len(tension_level) > 0:
            tension_factor = self._interpolate_param(tension_level, n_samples)
            tension_factor = (tension_factor - 50) / 100
            tension_factor = np.clip(tension_factor, 0, 1)
            processed_h = self._apply_tension_enhancement(processed_h, tension_factor)

        if voicing_level is not None and len(voicing_level) > 0:
            voicing_factor = self._interpolate_param(voicing_level, n_samples)
            voicing_factor = voicing_factor / 100
            voicing_factor = np.clip(voicing_factor, 0.1, 1.0)
            processed_h = processed_h * voicing_factor
            noise_comp = noise * (1 - voicing_factor) * 0.5
            processed_n = processed_n + noise_comp

        if low_freq_breath is not None and len(low_freq_breath) > 0:
            lfb_factor = self._interpolate_param(low_freq_breath, n_samples)
            lfb_factor = np.clip(lfb_factor / 100, 0, 1)
            processed_n = self._enhance_band_noise(processed_n, lfb_factor,
                                                    freq_range=(50, 500))

        if high_freq_breath is not None and len(high_freq_breath) > 0:
            hfb_factor = self._interpolate_param(high_freq_breath, n_samples)
            hfb_factor = np.clip(hfb_factor / 100, 0, 1)
            processed_n = self._enhance_band_noise(processed_n, hfb_factor,
                                                    freq_range=(3000, 15000))

        return processed_h.astype(np.float32), processed_n.astype(np.float32)

    def _interpolate_param(self, param_frames: np.ndarray, n_samples: int) -> np.ndarray:
        """将帧级参数插值到样本级"""
        from scipy.interpolate import interp1d

        if len(param_frames) == 0:
            return np.ones(n_samples, dtype=np.float32)
        if len(param_frames) == 1:
            return np.full(n_samples, param_frames[0], dtype=np.float32)

        frame_indices = np.arange(len(param_frames))
        sample_indices = np.linspace(0, len(param_frames) - 1, n_samples)

        f = interp1d(frame_indices, param_frames, kind='cubic',
                     bounds_error=False, fill_value='extrapolate')
        return f(sample_indices).astype(np.float32)

    def _apply_tension_enhancement(self, audio: np.ndarray, tension_factor: np.ndarray):
        """应用张力增强（动态频谱塑形）"""
        from scipy.signal import butter, filtfilt

        nyquist = self.sample_rate / 2
        base_freq = 100
        max_freq = 400

        cutoff_freq = base_freq + (max_freq - base_freq) * tension_factor
        cutoff_freq = np.clip(cutoff_freq, 60, nyquist * 0.9)

        mean_cutoff = float(np.mean(cutoff_freq))
        b, a = butter(2, mean_cutoff / nyquist, btype='high')
        filtered = filtfilt(b, a, audio)

        return audio * (1 - tension_factor) + filtered * tension_factor

    def _enhance_band_noise(self, noise: np.ndarray, factor: np.ndarray,
                            freq_range: Tuple[float, float]):
        """增强指定频段的噪声"""
        from scipy.signal import butter, filtfilt

        nyquist = self.sample_rate / 2
        low, high = freq_range

        b, a = butter(2, [low / nyquist, high / nyquist], btype='band')
        band_noise = filtfilt(b, a, noise)

        return noise + band_noise * factor * 0.5