"""相位声码器 - 音高调整和时间拉伸

实现高质量的音高偏移和时间拉伸，基于 STFT/iSTFT 的相位声码器。
支持：
- 时间拉伸（保持音高不变）
- 音高偏移（保持时长不变）
- 同时时间拉伸和音高偏移
"""

import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class PhaseVocoder:
    """相位声码器

    提供音高调整和时间拉伸功能。
    """

    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048,
                 hop_length: int = 512, win_length: int = 2048):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = np.hanning(win_length).astype(np.float64)

    def stft(self, x: np.ndarray) -> np.ndarray:
        """短时傅里叶变换

        Args:
            x: 输入音频 (n_samples,)

        Returns:
            复数频谱 (n_fft//2+1, n_frames)
        """
        f, t, Zxx = signal.stft(
            x,
            fs=self.sample_rate,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            window=self.window,
            return_onesided=True,
            boundary='zeros',
            padded=True,
        )
        return Zxx

    def istft(self, Zxx: np.ndarray, length: int = None) -> np.ndarray:
        """逆短时傅里叶变换

        Args:
            Zxx: 复数频谱 (n_fft//2+1, n_frames)
            length: 输出长度（样本数）

        Returns:
            时域信号 (n_samples,)
        """
        t, x = signal.istft(
            Zxx,
            fs=self.sample_rate,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            window=self.window,
            input_onesided=True,
            boundary=True,
            time_axis=-1,
            freq_axis=-2,
        )
        if length is not None:
            if len(x) > length:
                x = x[:length]
            elif len(x) < length:
                x = np.pad(x, (0, length - len(x)))
        return x

    def pitch_shift(self, x: np.ndarray, n_semitones: float) -> np.ndarray:
        """音高偏移（半音数）

        保持时长不变的音高偏移。

        Args:
            x: 输入音频 (n_samples,)
            n_semitones: 半音数（正为升高，负为降低）

        Returns:
            音高偏移后的音频
        """
        if n_semitones == 0:
            return x.copy()

        # 音高比例因子
        pitch_factor = 2 ** (n_semitones / 12.0)
        # 先时间拉伸 1/pitch_factor，然后重采样到原始长度
        stretched = self.time_stretch(x, 1.0 / pitch_factor)
        # 重采样到原始长度
        from scipy.signal import resample
        result = resample(stretched, len(x))
        return result.astype(x.dtype)

    def time_stretch(self, x: np.ndarray, rate: float) -> np.ndarray:
        """时间拉伸

        保持音高不变的时间拉伸。

        Args:
            x: 输入音频 (n_samples,)
            rate: 拉伸比率（>1 变长变慢，<1 变快变短）

        Returns:
            时间拉伸后的音频
        """
        if rate == 1.0:
            return x.copy()

        # STFT
        Zxx = self.stft(x)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        n_freq, n_frames = Zxx.shape
        # 输出帧数
        n_frames_out = int(np.round(n_frames * rate))

        # 相位声码器相位推进
        # 预期相位增量
        phasor = 2 * np.pi * self.hop_length * np.arange(n_freq) / self.n_fft

        # 初始化
        mag_out = np.zeros((n_freq, n_frames_out), dtype=np.float32)
        phase_out = np.zeros((n_freq, n_frames_out), dtype=np.float32)

        # 相位累加
        phase_acc = phase[:, 0]

        # 使用线性插值的时间索引
        time_indices = np.arange(n_frames_out) / rate

        for i in range(n_frames_out):
            t = time_indices[i]
            t_floor = int(np.floor(t))
            t_ceil = min(t_floor + 1, n_frames - 1)
            t_frac = t - t_floor

            if t_floor < n_frames and t_ceil < n_frames:
                # 幅度插值
                mag_out[:, i] = (1 - t_frac) * mag[:, t_floor] + t_frac * mag[:, t_ceil]

                # 相位推进
                if i > 0:
                    # 计算瞬时频率
                    delta_phase = phasor + np.angle(
                        np.exp(1j * (phase[:, t_ceil] - phase[:, t_floor] - phasor))
                    )
                    # 按比率缩放相位增量
                    phase_acc += delta_phase / rate

                phase_out[:, i] = phase_acc

        # 重建复数频谱
        Zxx_out = mag_out * np.exp(1j * phase_out)

        # iSTFT
        y = self.istft(Zxx_out)

        return y.astype(x.dtype)

    def pitch_shift_time_stretch(self, x: np.ndarray, n_semitones: float,
                                  rate: float) -> np.ndarray:
        """同时进行音高偏移和时间拉伸

        Args:
            x: 输入音频
            n_semitones: 半音数
            rate: 时间拉伸比率

        Returns:
            处理后的音频
        """
        # 先时间拉伸，再音高偏移（音高偏移内部会做重采样）
        stretched = self.time_stretch(x, rate)
        shifted = self.pitch_shift(stretched, n_semitones)
        return shifted


def pitch_shift_simple(x: np.ndarray, sample_rate: int, n_semitones: float,
                       n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """简易音高偏移函数

    Args:
        x: 输入音频
        sample_rate: 采样率
        n_semitones: 半音数
        n_fft: FFT 大小
        hop_length: 帧移

    Returns:
        音高偏移后的音频
    """
    pv = PhaseVocoder(sample_rate=sample_rate, n_fft=n_fft,
                      hop_length=hop_length, win_length=n_fft)
    return pv.pitch_shift(x, n_semitones)


def time_stretch_simple(x: np.ndarray, sample_rate: int, rate: float,
                        n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """简易时间拉伸函数

    Args:
        x: 输入音频
        sample_rate: 采样率
        rate: 拉伸比率
        n_fft: FFT 大小
        hop_length: 帧移

    Returns:
        时间拉伸后的音频
    """
    pv = PhaseVocoder(sample_rate=sample_rate, n_fft=n_fft,
                      hop_length=hop_length, win_length=n_fft)
    return pv.time_stretch(x, rate)
