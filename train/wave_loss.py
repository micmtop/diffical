"""音域感知损失：冻结声码器可微，多分辨率 STFT 损失。"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MRSTFT(nn.Module):
    def __init__(self, ffts=(512, 1024, 2048), hop_ratio=4):
        super().__init__()
        self.ffts = ffts
        self.hop_ratio = hop_ratio
        self.register_buffer("win2048", torch.hann_window(2048))

    def _stft(self, x, n_fft):
        hop = max(1, n_fft // self.hop_ratio)
        win = self.win2048[:n_fft] if n_fft <= 2048 else torch.hann_window(n_fft, device=x.device)
        return torch.stft(x, n_fft, hop, window=win, center=True, return_complex=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (B, N) 或 (B, 1, N)
        if x.dim() == 3:
            x = x.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        n = min(x.shape[-1], y.shape[-1])
        x, y = x[..., :n], y[..., :n]
        loss = 0.0
        for n_fft in self.ffts:
            X = self._stft(x, n_fft); Y = self._stft(y, n_fft)
            mx = X.abs() + 1e-7; my = Y.abs() + 1e-7
            sc = (mx - my).norm(dim=(1, 2)) / my.norm(dim=(1, 2)).clamp_min(1e-7)
            lm = (torch.log(mx) - torch.log(my)).abs().mean(dim=(1, 2))
            loss = loss + (sc + lm).mean()
        return loss / len(self.ffts)


def load_vocoder(model_dir: str, device: torch.device):
    import json, os
    from backend_vocoder.nsf_hifigan_model import Generator
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        h = json.load(f)
    class A(dict):
        __getattr__ = dict.__getitem__
    gen = Generator(A(h))
    ck = torch.load(os.path.join(model_dir, "model.ckpt"), map_location="cpu")
    sd = ck["generator"] if "generator" in ck else ck
    try:
        gen.load_state_dict(sd); gen.remove_weight_norm()
    except RuntimeError:
        gen.remove_weight_norm(); gen.load_state_dict(sd)
    gen.eval().to(device)
    for p in gen.parameters():
        p.requires_grad_(False)
    return gen
