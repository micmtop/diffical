"""智能去噪系统

实现高质量的音频降噪处理，在去除背景噪声的同时保留自然气息感：
1. 自适应噪声门控：根据音频能量动态调整阈值
2. 多频段噪声抑制：不同频段不同降噪强度
3. 谐波/噪声分离后的噪声重整：保留自然气息感
4. 智能噪声检测：区分背景噪声和人声自然噪声
"""

import numpy as np
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class SmartDenoiser:
    """智能去噪器

    采用多阶段降噪策略，在有效去除背景噪声的同时，
    最大程度保留人声的自然气息感和细节。
    """

    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048,
                 hop_length: int = 512):
        """初始化智能去噪器

        Args:
            sample_rate: 采样率，默认 44100Hz
            n_fft: FFT 大小
            hop_length: 帧移
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self._noise_profile: Dict[str, np.ndarray] = {}

    def adaptive_noise_gate(self, audio: np.ndarray, threshold_db: float = -50,
                            ratio: float = 2.0, attack_ms: float = 5,
                            release_ms: float = 50) -> np.ndarray:
        """自适应噪声门

        根据音频能量动态调整门控阈值，在静音段有效抑制噪声，
        在有声段保持自然过渡，避免门控咔哒声。

        Args:
            audio: 输入音频 (n_samples,)
            threshold_db: 初始阈值（dB），相对于满刻度
            ratio: 压缩比（高于阈值部分的衰减比例）
            attack_ms: 启动时间（毫秒）
            release_ms: 释放时间（毫秒）

        Returns:
            处理后的音频
        """
        from scipy.signal import filtfilt, butter

        if len(audio) == 0:
            return audio.copy()

        audio = audio.astype(np.float64)
        n_samples = len(audio)

        envelope = np.abs(audio)

        attack_samples = int(attack_ms * self.sample_rate / 1000)
        release_samples = int(release_ms * self.sample_rate / 1000)
        attack_samples = max(1, attack_samples)
        release_samples = max(1, release_samples)

        alpha_a = np.exp(-1.0 / attack_samples)
        alpha_r = np.exp(-1.0 / release_samples)

        smoothed_envelope = np.zeros(n_samples)
        prev = 0.0
        for i in range(n_samples):
            curr = envelope[i]
            if curr > prev:
                prev = alpha_a * prev + (1 - alpha_a) * curr
            else:
                prev = alpha_r * prev + (1 - alpha_r) * curr
            smoothed_envelope[i] = prev

        ref = 1.0
        threshold_linear = ref * (10 ** (threshold_db / 20.0))

        frame_size = 512
        n_frames = n_samples // frame_size
        if n_frames < 2:
            n_frames = 2
            frame_size = n_samples // n_frames

        local_threshold = np.zeros(n_samples)
        for i in range(n_frames):
            start = i * frame_size
            end = min(start + frame_size, n_samples)
            frame_env = smoothed_envelope[start:end]
            frame_median = np.median(frame_env)

            adaptive_threshold = max(
                threshold_linear,
                frame_median * 0.5
            )
            local_threshold[start:end] = adaptive_threshold

        if n_samples % frame_size != 0:
            start = n_frames * frame_size
            local_threshold[start:] = local_threshold[start - 1]

        b, a = butter(2, 0.01, btype='low')
        local_threshold = filtfilt(b, a, local_threshold)

        gain = np.ones(n_samples)
        above_threshold = smoothed_envelope > local_threshold
        gain[above_threshold] = 1.0

        below = ~above_threshold
        if np.any(below):
            ratio_env = smoothed_envelope[below] / (local_threshold[below] + 1e-10)
            gain[below] = ratio_env ** (1.0 / ratio - 1.0)

        gain = np.clip(gain, 0.0, 1.0)

        result = audio * gain

        return result.astype(np.float32)

    def multiband_denoise(self, audio: np.ndarray, strength: float = 0.5,
                          n_bands: int = 6) -> np.ndarray:
        """多频段噪声抑制

        将音频分成多个频段，每个频段根据其信噪比特性
        应用不同的降噪强度，低频段更保守，高频段更积极。

        Args:
            audio: 输入音频 (n_samples,)
            strength: 整体降噪强度 (0~1)
            n_bands: 频段数量

        Returns:
            处理后的音频
        """
        from scipy import signal
        from scipy.ndimage import gaussian_filter1d

        if len(audio) < self.n_fft or strength <= 0:
            return audio.copy()

        audio = audio.astype(np.float64)

        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        n_freqs = mag.shape[0]
        n_frames = mag.shape[1]

        band_edges = np.linspace(0, n_freqs - 1, n_bands + 1).astype(int)

        band_strengths = np.linspace(0.4, 0.3, n_bands)
        band_strengths = band_strengths * strength

        mag_out = mag.copy()

        for band_idx in range(n_bands):
            start_freq = band_edges[band_idx]
            end_freq = band_edges[band_idx + 1]
            band_mag = mag[start_freq:end_freq, :]

            if band_mag.size == 0:
                continue

            noise_est = np.median(band_mag, axis=1, keepdims=True)
            noise_est = gaussian_filter1d(noise_est.squeeze(), sigma=2).reshape(-1, 1)

            snr = band_mag / (noise_est + 1e-10)

            alpha = band_strengths[band_idx]

            oversubtraction = 1.5 + alpha * 1.5
            flooring = 0.01 + alpha * 0.02

            gain = np.ones_like(band_mag)
            high_snr = snr > oversubtraction
            gain[high_snr] = 1.0

            low_snr = snr < 1.0
            gain[low_snr] = flooring

            mid_mask = ~high_snr & ~low_snr
            if np.any(mid_mask):
                mid_snr = snr[mid_mask]
                gain[mid_mask] = (
                    flooring +
                    (1.0 - flooring) *
                    (mid_snr - 1.0) / (oversubtraction - 1.0)
                )

            gain_smooth = gaussian_filter1d(gain, sigma=1, axis=0)
            gain_smooth = gaussian_filter1d(gain_smooth, sigma=0.5, axis=1)

            mag_out[start_freq:end_freq, :] = band_mag * gain_smooth

        _, result = signal.istft(
            mag_out * np.exp(1j * phase),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )

        result = result[:len(audio)]
        return result.astype(np.float32)

    def noise_reshaping(self, harmonic: np.ndarray, noise: np.ndarray,
                        breathiness: float = 0.3) -> np.ndarray:
        """噪声重整（保留气息感）

        对谐波/噪声分离后的噪声分量进行频谱整形，
        在降低整体噪声水平的同时，保留人声自然的气息感。

        Args:
            harmonic: 谐波分量 (n_samples,)
            noise: 噪声分量 (n_samples,)
            breathiness: 气息感保留程度 (0~1)，越大越保留气息

        Returns:
            重整后的噪声分量
        """
        from scipy import signal
        from scipy.ndimage import gaussian_filter1d

        if len(noise) == 0:
            return noise.copy()

        noise = noise.astype(np.float64)
        harmonic = harmonic.astype(np.float64)

        f, t, Zxx_noise = signal.stft(
            noise, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag_noise = np.abs(Zxx_noise)
        phase_noise = np.angle(Zxx_noise)

        n_freqs = mag_noise.shape[0]
        freq_axis = f / (self.sample_rate / 2)

        _, _, Zxx_harm = signal.stft(
            harmonic, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag_harm = np.abs(Zxx_harm)

        harm_energy = np.sum(mag_harm, axis=0)
        max_harm_energy = np.max(harm_energy) + 1e-10
        voicing = harm_energy / max_harm_energy
        voicing = gaussian_filter1d(voicing, sigma=2)
        voicing = np.clip(voicing, 0, 1)

        noise_spectrum_avg = np.mean(mag_noise, axis=1)

        breath_band_low = int(0.1 * n_freqs)
        breath_band_high = int(0.6 * n_freqs)

        spectral_shape = np.ones(n_freqs)

        low_freq_reduction = 0.3 + (1 - breathiness) * 0.4
        spectral_shape[:breath_band_low] = low_freq_reduction

        mid_freq_boost = 0.5 + breathiness * 0.3
        breath_range = breath_band_high - breath_band_low
        mid_shape = np.linspace(low_freq_reduction, mid_freq_boost, breath_range)
        spectral_shape[breath_band_low:breath_band_high] = mid_shape

        high_freq_rolloff = 0.2 + (1 - breathiness) * 0.2
        high_range = n_freqs - breath_band_high
        high_shape = np.linspace(mid_freq_boost, high_freq_rolloff, high_range)
        spectral_shape[breath_band_high:] = high_shape

        spectral_shape = gaussian_filter1d(spectral_shape, sigma=3)

        temporal_mask = 0.3 + breathiness * 0.5 + voicing * (1 - breathiness) * 0.4
        temporal_mask = np.clip(temporal_mask, 0.2, 1.0)

        gain = spectral_shape.reshape(-1, 1) * temporal_mask.reshape(1, -1)

        mag_out = mag_noise * gain

        _, result = signal.istft(
            mag_out * np.exp(1j * phase_noise),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )

        result = result[:len(noise)]
        return result.astype(np.float32)

    def detect_noise_profile(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """噪声轮廓检测

        分析音频的噪声特征，包括：
        - 频谱噪声轮廓
        - 噪声能量随时间的分布
        - 各频段的噪声水平

        Args:
            audio: 输入音频 (n_samples,)

        Returns:
            包含噪声轮廓信息的字典
        """
        from scipy import signal
        from scipy.ndimage import gaussian_filter1d

        if len(audio) < self.n_fft:
            return {
                'spectral_profile': np.zeros(self.n_fft // 2 + 1),
                'noise_floor_db': -60.0,
                'band_levels': np.zeros(6),
            }

        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)

        frame_energy = np.sum(mag, axis=0)
        sorted_energy = np.sort(frame_energy)
        low_20_idx = int(len(sorted_energy) * 0.2)
        low_energy_frames = frame_energy <= sorted_energy[max(0, low_20_idx)]

        if np.any(low_energy_frames):
            noise_spectrum = np.mean(mag[:, low_energy_frames], axis=1)
        else:
            noise_spectrum = np.median(mag, axis=1)

        noise_spectrum = gaussian_filter1d(noise_spectrum, sigma=3)

        ref = np.max(mag) + 1e-10
        noise_floor_db = 20 * np.log10(np.mean(noise_spectrum) / ref + 1e-10)

        n_bands = 6
        band_edges = np.linspace(0, len(f) - 1, n_bands + 1).astype(int)
        band_levels = np.zeros(n_bands)

        for i in range(n_bands):
            start = band_edges[i]
            end = band_edges[i + 1]
            band_levels[i] = np.mean(noise_spectrum[start:end])

        profile = {
            'spectral_profile': noise_spectrum.astype(np.float32),
            'noise_floor_db': float(noise_floor_db),
            'band_levels': band_levels.astype(np.float32),
            'freq_axis': f,
        }

        self._noise_profile = profile
        return profile

    def denoise(self, audio: np.ndarray, mode: str = 'gentle',
                strength: float = 0.3) -> np.ndarray:
        """综合去噪入口

        多阶段降噪处理，根据模式选择不同的降噪强度组合：
        - gentle: 轻度降噪，几乎不影响音质
        - moderate: 中度降噪，平衡降噪量和音质
        - strong: 强降噪，最大限度去除噪声

        Args:
            audio: 输入音频 (n_samples,)
            mode: 降噪模式 ('gentle', 'moderate', 'strong')
            strength: 额外的强度调节因子 (0~1)

        Returns:
            去噪后的音频
        """
        if len(audio) == 0:
            return audio.copy()

        mode_configs = {
            'gentle': {
                'gate_threshold_db': -55,
                'gate_ratio': 1.5,
                'multiband_strength': 0.2,
                'breathiness': 0.8,
            },
            'moderate': {
                'gate_threshold_db': -50,
                'gate_ratio': 2.0,
                'multiband_strength': 0.4,
                'breathiness': 0.5,
            },
            'strong': {
                'gate_threshold_db': -45,
                'gate_ratio': 3.0,
                'multiband_strength': 0.7,
                'breathiness': 0.2,
            },
        }

        config = mode_configs.get(mode, mode_configs['gentle'])
        strength = np.clip(strength, 0, 1)

        if strength < 0.01:
            return audio.copy()

        result = audio.copy()

        try:
            self.detect_noise_profile(result)
        except Exception as e:
            logger.warning(f"噪声轮廓检测失败: {e}")

        try:
            result = self.multiband_denoise(
                result,
                strength=config['multiband_strength'] * strength
            )
        except Exception as e:
            logger.warning(f"多频段降噪失败: {e}")

        try:
            result = self.adaptive_noise_gate(
                result,
                threshold_db=config['gate_threshold_db'],
                ratio=config['gate_ratio'],
            )
        except Exception as e:
            logger.warning(f"自适应噪声门失败: {e}")

        # 暂时禁用噪声重整，避免引入高频噪声
        # try:
        #     harmonic, noise = self._separate_harmonic_noise(result)
        #     reshaped_noise = self.noise_reshaping(
        #         harmonic, noise,
        #         breathiness=config['breathiness']
        #     )
        #     result = harmonic + reshaped_noise

        #     peak_orig = np.max(np.abs(audio))
        #     peak_result = np.max(np.abs(result))
        #     if peak_result > peak_orig * 1.1:
        #         result = result * (peak_orig * 1.1 / peak_result)

        #     logger.info(f"噪声重整完成: 气息感={config['breathiness']:.2f}")
        # except Exception as e:
        #     logger.warning(f"噪声重整失败: {e}")

        return result.astype(np.float32)

    def denoise_mel(self, mel: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Mel频谱层面去噪

        直接在 Mel 频谱上进行降噪处理，适用于声码器前的特征处理。
        使用谱减法结合 Mel 频段的自适应阈值。

        Args:
            mel: Mel 频谱 (n_mels, n_frames)
            strength: 降噪强度 (0~1)

        Returns:
            去噪后的 Mel 频谱
        """
        from scipy.ndimage import gaussian_filter1d

        if mel.size == 0 or strength <= 0:
            return mel.copy()

        mel = mel.astype(np.float64)
        n_mels, n_frames = mel.shape

        if n_frames < 5:
            return mel.astype(np.float32)

        frame_energy = np.sum(mel, axis=0)
        sorted_energy = np.sort(frame_energy)
        low_idx = int(len(sorted_energy) * 0.2)
        low_energy_thresh = sorted_energy[max(0, low_idx)]
        noise_frames = frame_energy <= low_energy_thresh

        if np.any(noise_frames):
            noise_profile = np.mean(mel[:, noise_frames], axis=1)
        else:
            noise_profile = np.median(mel, axis=1)

        noise_profile = gaussian_filter1d(noise_profile, sigma=1)

        mel_smooth = gaussian_filter1d(mel, sigma=2, axis=1)

        snr = mel_smooth / (noise_profile.reshape(-1, 1) + 1e-10)

        oversubtraction = 1.0 + strength * 1.5
        flooring = 0.01 + strength * 0.02

        gain = np.ones_like(mel)
        high_snr = snr > oversubtraction
        gain[high_snr] = 1.0

        low_snr = snr < 1.0
        gain[low_snr] = flooring

        mid_mask = ~high_snr & ~low_snr
        if np.any(mid_mask):
            mid_snr = snr[mid_mask]
            gain[mid_mask] = (
                flooring +
                (1.0 - flooring) *
                (mid_snr - 1.0) / (oversubtraction - 1.0)
            )

        gain = gaussian_filter1d(gain, sigma=1, axis=0)
        gain = gaussian_filter1d(gain, sigma=1, axis=1)

        result = mel * gain

        return result.astype(np.float32)

    def _separate_harmonic_noise(self, audio: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        """谐波/噪声分离（内部方法）

        使用简化的谱减法进行谐波/噪声分离，
        用于噪声重整阶段。

        Args:
            audio: 输入音频 (n_samples,)

        Returns:
            (harmonic, noise) 两个时域信号
        """
        from scipy import signal
        from scipy.ndimage import gaussian_filter1d

        if len(audio) < self.n_fft:
            return audio.copy(), np.zeros_like(audio)

        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        noise_est = np.median(mag, axis=1, keepdims=True)
        noise_est = gaussian_filter1d(noise_est.squeeze(), sigma=3).reshape(-1, 1)

        alpha = 1.5
        beta = 0.05

        harmonic_mask = mag > alpha * noise_est
        harmonic_mag = np.where(harmonic_mask, mag - beta * noise_est, 0.0)
        noise_mag = mag - harmonic_mag
        noise_mag = np.maximum(noise_mag, 0.001 * mag)

        harmonic_mag = gaussian_filter1d(harmonic_mag, sigma=0.5, axis=1)
        noise_mag = gaussian_filter1d(noise_mag, sigma=1, axis=1)

        _, harmonic = signal.istft(
            harmonic_mag * np.exp(1j * phase),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        _, noise = signal.istft(
            noise_mag * np.exp(1j * phase),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )

        min_len = min(len(audio), len(harmonic), len(noise))
        harmonic = harmonic[:min_len]
        noise = noise[:min_len]

        return harmonic.astype(np.float32), noise.astype(np.float32)


def denoise_audio(audio: np.ndarray, sample_rate: int = 44100,
                  mode: str = 'gentle', strength: float = 0.3) -> np.ndarray:
    """便捷函数：智能去噪

    Args:
        audio: 输入音频
        sample_rate: 采样率
        mode: 降噪模式
        strength: 降噪强度

    Returns:
        去噪后的音频
    """
    denoiser = SmartDenoiser(sample_rate=sample_rate)
    return denoiser.denoise(audio, mode=mode, strength=strength)


def denoise_mel_spectrum(mel: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """便捷函数：Mel频谱去噪

    Args:
        mel: Mel 频谱 (n_mels, n_frames)
        strength: 降噪强度

    Returns:
        去噪后的 Mel 频谱
    """
    denoiser = SmartDenoiser()
    return denoiser.denoise_mel(mel, strength=strength)
