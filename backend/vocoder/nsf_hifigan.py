"""pc-nsf-hifigan Vocoder 封装

复用自参考实现 (util/nsf_hifigan.py)。

将 Mel 频谱 + F0 转换为高质量波形，支持音高控制。
"""

import os
import json
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AttrDict(dict):
    """字典属性访问"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class NsfHifiGAN:
    """pc-nsf-hifigan vocoder

    PyTorch 后端，支持 F0 输入的音高可控波形合成。
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "cpu",
    ):
        import torch

        self.device = device
        self.model_path = model_path
        self.config_path = config_path

        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.h = AttrDict(json.load(f))

        # 构建模型
        from .nsf_hifigan_model import Generator

        self.model = Generator(self.h)
        checkpoint = torch.load(model_path, map_location=device)
        if "generator" in checkpoint:
            state_dict = checkpoint["generator"]
        else:
            state_dict = checkpoint
        try:
            self.model.load_state_dict(state_dict)
            self.model.remove_weight_norm()
        except RuntimeError:
            self.model.remove_weight_norm()
            self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)

        logger.info(f"pc-nsf-hifigan 加载完成: {model_path}")

    def spec2wav_torch(self, mel: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """Mel + F0 → 波形

        Args:
            mel: Mel 频谱 (n_mels, T) 或 (T, n_mels)
            f0: 基频 (T,)

        Returns:
            waveform: 合成的波形 (samples,)
        """
        import torch

        # 转换为 torch tensor
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel).float().to(self.device)
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(self.device)

        # 确保维度: [1, n_mels, T]
        n_mel_channels = getattr(self.h, 'n_mel_channels', getattr(self.h, 'num_mels', 128))
        
        if mel.dim() == 2:
            if mel.size(0) == n_mel_channels:
                mel = mel.unsqueeze(0)  # (n_mels, T) -> (1, n_mels, T)
            else:
                mel = mel.t().unsqueeze(0)  # (T, n_mels) -> (n_mels, T) -> (1, n_mels, T)
        elif mel.dim() == 3:
            if mel.size(1) == n_mel_channels:
                pass  # already (1, n_mels, T)
            elif mel.size(2) == n_mel_channels:
                mel = mel.transpose(1, 2)  # (1, T, n_mels) -> (1, n_mels, T)
            else:
                raise ValueError(f"Mel维度错误: {mel.shape}, 期望 [1, {n_mel_channels}, T]")

        if f0.dim() == 1:
            f0 = f0.unsqueeze(0)  # (1, T)

        # 确保 mel 和 f0 长度一致
        if mel.size(2) != f0.size(1):
            min_len = min(mel.size(2), f0.size(1))
            mel = mel[:, :, :min_len]
            f0 = f0[:, :min_len]

        # 推理
        with torch.no_grad():
            waveform = self.model(mel, f0)

        waveform = waveform.squeeze().cpu().numpy()
        
        mel_np = mel.squeeze().cpu().numpy() if isinstance(mel, torch.Tensor) else np.array(mel)
        f0_np = f0.squeeze().cpu().numpy() if isinstance(f0, torch.Tensor) else np.array(f0)
        
        silent_frames = (np.mean(mel_np, axis=0) < -20.0) & (f0_np < 60.0)
        
        if np.any(silent_frames):
            hop_size = 512
            n_frames = len(silent_frames)
            
            for i in range(n_frames):
                if silent_frames[i]:
                    start_idx = i * hop_size
                    end_idx = min((i + 1) * hop_size, len(waveform))
                    
                    transition_frames = 1
                    left_edge = max(0, i - transition_frames)
                    right_edge = min(n_frames - 1, i + transition_frames)
                    
                    for j in range(left_edge, right_edge + 1):
                        j_start = j * hop_size
                        j_end = min((j + 1) * hop_size, len(waveform))
                        
                        if j == i:
                            waveform[j_start:j_end] = 0.0
                        else:
                            dist = abs(j - i)
                            frame_gain = 1.0 - dist / (transition_frames + 1)
                            waveform[j_start:j_end] *= frame_gain
        
        return waveform
