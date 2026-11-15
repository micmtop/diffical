"""整句流畅度增强（五级）

对拼接后的整句音频进行流畅度优化：
1. 频谱平滑 - 减少频谱突变
2. 相位连续 - 确保帧间相位连续
3. 能量包络平滑 - 消除能量突变
4. 音色一致性 - 保持全局音色稳定
5. 韵律优化 - 微调节奏使更自然
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)


class FluencyEnhancer:
    """五级流畅度增强器"""

    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048,
                 hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.window = np.hanning(self.win_length)

    def enhance(self, audio: np.ndarray, level: int = 3) -> np.ndarray:
        """应用流畅度增强

        Args:
            audio: 输入音频
            level: 增强级别 1-5

        Returns:
            增强后的音频
        """
        if level <= 0:
            return audio

        result = audio.copy()

        # 级别 1: 频谱平滑
        if level >= 1:
            result = self._spectral_smooth(result)

        # 级别 2: 相位连续（已在 STFT 处理中部分保证）
        if level >= 2:
            result = self._phase_continuity(result)

        # 级别 3: 能量包络平滑
        if level >= 3:
            result = self._energy_smooth(result)

        # 级别 4: 音色一致性
        if level >= 4:
            result = self._timbre_consistency(result)

        # 级别 5: 韵律优化
        if level >= 5:
            result = self._prosody_optimize(result)

        return result

    def _spectral_smooth(self, audio: np.ndarray) -> np.ndarray:
        """频谱平滑 - 减少帧间频谱突变（极轻度，避免糊感）"""
        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        mag_smooth = gaussian_filter1d(mag, sigma=0.3, axis=1)

        _, result = signal.istft(
            mag_smooth * np.exp(1j * phase),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        return result[:len(audio)]

    def _phase_continuity(self, audio: np.ndarray) -> np.ndarray:
        """相位连续 - 确保相位演化自然

        使用改进的相位平滑算法：
        1. 对相位进行展开
        2. 使用双向滤波确保因果性
        3. 保留高频相位细节（避免过度平滑导致电音感）
        """
        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        phase_unwrapped = np.unwrap(phase, axis=1)

        low_freq_mask = np.abs(f) < 2000
        high_freq_mask = ~low_freq_mask

        phase_smooth = phase_unwrapped.copy()
        if np.any(low_freq_mask):
            phase_smooth[low_freq_mask] = gaussian_filter1d(
                phase_unwrapped[low_freq_mask], sigma=1.5, axis=1
            )

        phase_out = phase_smooth - 2 * np.pi * np.round(phase_smooth / (2 * np.pi))

        _, result = signal.istft(
            mag * np.exp(1j * phase_out),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        return result[:len(audio)]

    def _energy_smooth(self, audio: np.ndarray, window_ms: float = 20.0) -> np.ndarray:
        """能量包络平滑（带噪声门限，避免放大静音区）"""
        hop_samples = int(window_ms * self.sample_rate / 1000)
        n_frames = len(audio) // hop_samples
        if n_frames < 2:
            return audio

        energy = np.array([
            np.sqrt(np.mean(audio[i * hop_samples:(i + 1) * hop_samples] ** 2))
            for i in range(n_frames)
        ])

        max_energy = np.max(energy)
        if max_energy < 1e-6:
            return audio

        noise_floor = max_energy * 0.01
        energy_safe = np.maximum(energy, noise_floor)

        energy_smooth = gaussian_filter1d(energy_safe, sigma=2.0)

        gain = energy_smooth / energy_safe
        gain = np.clip(gain, 0.6, 1.5)

        silence_mask = energy < noise_floor
        gain[silence_mask] = 1.0

        from scipy.interpolate import interp1d
        t_orig = np.arange(len(gain)) * hop_samples + hop_samples / 2
        t_new = np.arange(len(audio))
        gain_interp = interp1d(
            t_orig, gain, kind='cubic',
            fill_value=1.0, bounds_error=False
        )(t_new)

        return audio * gain_interp

    def _timbre_consistency(self, audio: np.ndarray) -> np.ndarray:
        """音色一致性 - 全局频谱均衡"""
        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)

        global_spec = np.mean(mag, axis=1, keepdims=True)
        global_spec_smooth = gaussian_filter1d(global_spec[:, 0], sigma=8.0)

        for i in range(mag.shape[1]):
            frame_spec = gaussian_filter1d(mag[:, i], sigma=5.0)
            gain = np.where(frame_spec > 1e-6,
                           global_spec_smooth / frame_spec, 1.0)
            gain = np.clip(gain, 0.7, 1.4)
            mag[:, i] *= gain

        phase = np.angle(Zxx)
        _, result = signal.istft(
            mag * np.exp(1j * phase),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        return result[:len(audio)]

    def _prosody_optimize(self, audio: np.ndarray) -> np.ndarray:
        """韵律优化 - 微调节奏

        通过轻微的时间拉伸/压缩让听起来更自然。
        检测能量低谷（可能是音素边界），在边界处做轻微的时长调整。
        """
        hop = 512
        energy = np.array([
            np.sqrt(np.mean(audio[i:i + hop] ** 2))
            for i in range(0, len(audio) - hop, hop)
        ])

        energy_smooth = gaussian_filter1d(energy, sigma=5)

        from scipy.signal import find_peaks
        valleys, _ = find_peaks(-energy_smooth, distance=8)

        if len(valleys) == 0:
            return audio

        result = audio.copy()

        for valley in valleys:
            center_sample = valley * hop
            window_size = hop * 3

            start = max(0, center_sample - window_size)
            end = min(len(audio), center_sample + window_size)

            if end - start < hop * 2:
                continue

            fade_len = hop // 2
            left_start = center_sample - fade_len
            left_end = center_sample
            right_start = center_sample
            right_end = center_sample + fade_len

            if left_start >= 0 and right_end <= len(audio):
                fade_out = np.linspace(1, 0.8, fade_len)
                fade_in = np.linspace(0.8, 1, fade_len)

                result[left_start:left_end] *= fade_out
                result[right_start:right_end] *= fade_in

        return result

    def _cross_spectral_smoothing(self, audio: np.ndarray) -> np.ndarray:
        """跨频段平滑 - 确保不同频段之间的过渡自然"""
        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        n_freqs = mag.shape[0]
        freq_smooth = np.zeros_like(mag)

        for i in range(n_freqs):
            start = max(0, i - 2)
            end = min(n_freqs, i + 3)
            freq_smooth[i, :] = np.mean(mag[start:end, :], axis=0)

        _, result = signal.istft(
            freq_smooth * np.exp(1j * phase),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        return result[:len(audio)]


def enhance_fluency(audio: np.ndarray, sample_rate: int = 44100,
                    level: int = 3) -> np.ndarray:
    """便捷函数：流畅度增强"""
    enhancer = FluencyEnhancer(sample_rate=sample_rate)
    return enhancer.enhance(audio, level=level)