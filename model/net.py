"""整句 mel 生成器（绝对生成 + 全局瓶颈），目标 1.5–2.5M 参数。

输入通道（每帧）：
    source_mel 128 + f0 2(log2 + voiced) + phoneme emb 32 + boundary 1 + 位置 3
    = 166
输出：128 bins 的整句 log-mel（与规范前端同一尺度）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MELS = 128


def _gn(c: int) -> nn.GroupNorm:
    g = 8 if c % 8 == 0 else 1
    return nn.GroupNorm(g, c)


class ResBlock(nn.Module):
    def __init__(self, c: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 3, padding=dilation, dilation=dilation)
        self.norm = _gn(c)

    def forward(self, x):
        return x + F.silu(self.norm(self.conv(x)))


class Gen(nn.Module):
    def __init__(self, c: int = 192, global_c: int = 256, gru_hidden: int = 128,
                 use_f0: bool = True):
        super().__init__()
        # 无 phone embedding：音素身份完全由 source mel 携带（wav2mel 的产物）
        self.use_f0 = use_f0
        in_ch = N_MELS + (2 if use_f0 else 0) + 1 + 3
        self.inp = nn.Sequential(nn.Conv1d(in_ch, c, 5, padding=2), _gn(c), nn.SiLU(),
                                 nn.Conv1d(c, c, 5, padding=2))
        self.blocks = nn.Sequential(*[ResBlock(c, d) for d in (1, 2, 4, 8, 16, 32)])
        self.down = nn.Conv1d(c, global_c, 3, stride=4, padding=1)
        self.gru = nn.GRU(global_c, gru_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.fuse = nn.Conv1d(c + 2 * gru_hidden, c, 3, padding=1)
        self.out = nn.Conv1d(c, N_MELS, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, source, f0, boundary, progress, dist_bd, dur):
        """
        source: (B,128,T) f0: (B,T) Hz  phone: (B,T) int64
        boundary/progress/dist_bd/dur: (B,T)
        """
        parts = [source]
        if self.use_f0:
            voiced = (f0 > 0).float()
            f0n = torch.log2((f0.clamp_min(1.0)) / 440.0) * voiced
            parts += [f0n.unsqueeze(1), voiced.unsqueeze(1)]
        parts += [boundary.unsqueeze(1), progress.unsqueeze(1),
                  dist_bd.unsqueeze(1), dur.unsqueeze(1)]
        x = torch.cat(parts, dim=1)
        h = self.inp(x)
        h = self.blocks(h)
        g = self.down(h).transpose(1, 2)
        g, _ = self.gru(g)
        g = g.transpose(1, 2)
        g = F.interpolate(g, size=h.shape[-1], mode="nearest")
        h = self.fuse(torch.cat([h, g], dim=1))
        return source + self.out(h)   # 残差：零初始化头 => 初始输出 == source


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
