"""PitchAdjustableMelSpectrogram

标准 mel 前端，参数与声码器对齐。

支持 key_shift（按半音调整频谱）和 speed（调整帧移）。
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

try:
    from librosa.filters import mel as librosa_mel_fn
    _has_librosa = True
except ImportError:
    _has_librosa = False


def _mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """创建 Mel 滤波器组（优先使用 torchaudio，回退 librosa）"""
    if _has_librosa:
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    # 用 torchaudio 生成
    try:
        import torchaudio
        mel_spec_transform = torchaudio.transforms.MelScale(
            n_mels=n_mels, sample_rate=sr, f_min=fmin, f_max=fmax,
            n_stft=n_fft // 2 + 1,
        )
        # 提取滤波器矩阵 (n_mels, n_freqs)
        fb = mel_spec_transform.fb.numpy().T  # torchaudio 是 (n_freqs, n_mels)
        return fb
    except Exception:
        raise ImportError("需要 librosa 或 torchaudio 来生成 Mel 滤波器")


class PitchAdjustableMelSpectrogram:
    """可调音高的 Mel 频谱分析器

    特性：
    - key_shift: 按半音重新计算窗长度和频谱尺寸，实现音高移位后的频谱提取
    - speed: 调整帧移（hop_length），实现速度变化后的频谱提取
    """

    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        f_min=40.0,
        f_max=16000.0,
        n_mels=128,
        center=False,
        device="cpu",
    ):
        self.sample_rate = sample_rate
        self.device = device
        self.n_fft = n_fft
        self.win_size = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.center = center
        self.mel_basis = {}
        self.hann_window = {}

    def __call__(self, y, key_shift=0, speed=1.0):
        """提取 Mel 频谱

        Args:
            y: 输入波形 (batch, samples) 或 (samples,)
            key_shift: 音高移位（半音数，可小数）
            speed: 速度倍率（1.0=原速）

        Returns:
            mel: Mel 频谱 (batch, n_mels, T)
        """
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float().to(self.device)

        if y.dim() == 1:
            y = y.unsqueeze(0)

        factor = 2 ** (key_shift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_size_new = int(np.round(self.win_size * factor))
        hop_length = int(np.round(self.hop_length / speed))

        # mel_basis 始终使用原始 n_fft，这样频率轴保持一致
        # key_shift 通过调整 STFT 的窗长度和裁剪 spec 来实现频率轴缩放
        mel_basis_key = f"{self.f_max}_{y.device}"
        if mel_basis_key not in self.mel_basis:
            mel = _mel_filterbank(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max,
            )
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)

        hann_window_key = f"{key_shift}_{y.device}"
        if hann_window_key not in self.hann_window:
            self.hann_window[hann_window_key] = torch.hann_window(
                win_size_new, device=y.device
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((win_size_new - hop_length) // 2),
                int((win_size_new - hop_length + 1) // 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            n_fft_new,
            hop_length=hop_length,
            win_length=win_size_new,
            window=self.hann_window[hann_window_key],
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()

        # key_shift 处理：将 spec 裁剪/填充到原始 n_fft 对应的频率数
        # 并进行幅度补偿
        if key_shift != 0:
            size = self.n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))
            spec = spec[:, :size, :] * self.win_size / win_size_new

        spec = torch.matmul(self.mel_basis[mel_basis_key], spec)

        return self.dynamic_range_compression_torch(spec)

    def dynamic_range_compression_torch(self, x, C=1, clip_val=1e-9):
        return torch.log(torch.clamp(x, min=clip_val) * C)
