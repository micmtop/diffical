"""SetGen：模型接管拼接的生成器。

与旧 Gen 的根本区别：
  旧：输入是"程序摆好的画布"(128×T)，模型只能逐帧改值
  新：输入是**片段集合** {(128, L_i)} + 每帧的段归属/段内进度/边界/F0，
      模型用 cross-attention 自己决定每帧从哪个片段的哪个时刻取内容

输出：整句 mel (128, T)，无程序拼接参与。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MELS = 128


class SegEncoder(nn.Module):
    """把变长片段编码成 K 个 token。"""

    def __init__(self, c: int, k: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(N_MELS, c, 5, padding=2), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv1d(c, c, 5, padding=2), nn.GroupNorm(8, c), nn.SiLU(),
        )
        self.k = k

    def forward(self, segs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # segs: (B, S, 128, L), mask: (B, S, L)
        B, S, _, L = segs.shape
        x = segs.reshape(B * S, N_MELS, L)
        m = mask.reshape(B * S, 1, L).float()
        h = self.net(x) * m
        # 不做池化：保留片段的所有帧，供帧级 query 精确取内容
        return h.reshape(B, S, h.shape[1], h.shape[-1])


class SetGen(nn.Module):
    def __init__(self, c: int = 192, d: int = 192, k: int = 8, heads: int = 4,
                 layers: int = 2, n_frame_cond: int = 4):
        super().__init__()
        self.seg = SegEncoder(c, k)
        self.seg_proj = nn.Linear(c, d)
        # 帧侧条件：段归属 embedding + 段内进度 + 边界 + F0n + voiced
        self.seg_emb = nn.Embedding(256, 32, padding_idx=0)
        in_q = 32 + n_frame_cond
        self.q_proj = nn.Sequential(nn.Conv1d(in_q, d, 5, padding=2), nn.GroupNorm(8, d), nn.SiLU(),
                                    nn.Conv1d(d, d, 5, padding=2), nn.GroupNorm(8, d), nn.SiLU())
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.out = nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.SiLU(), nn.Conv1d(d, N_MELS, 1))

    def forward(self, segs, seg_mask, seg_idx, progress, boundary, f0):
        # segs (B,S,128,L), seg_mask (B,S,L), seg_idx (B,T) 0..S (0=pad)
        # progress/boundary/f0 (B,T)
        B, S, _, L = segs.shape
        T = seg_idx.shape[1]
        enc = self.seg(segs, seg_mask)                    # (B,S,c,L)
        tok = self.seg_proj(enc.permute(0, 1, 3, 2))       # (B,S,L,d)
        tok = tok.reshape(B, S * enc.shape[-1], -1)         # (B,S*L,d)
        key_mask = seg_mask.reshape(B, S * enc.shape[-1])   # (B,S*L)
        voiced = (f0 > 0).float()
        f0n = torch.log2(f0.clamp_min(1.0) / 440.0) * voiced
        q = torch.cat([self.seg_emb(seg_idx).transpose(1, 2),
                       progress.unsqueeze(1), boundary.unsqueeze(1),
                       f0n.unsqueeze(1), voiced.unsqueeze(1)], dim=1)   # (B,36,T)
        h = self.q_proj(q).transpose(1, 2)                                 # (B,T,d)
        for i, att in enumerate(self.attn):
            a, _ = att(h, tok, tok, key_padding_mask=~key_mask, need_weights=False)
            h = self.norm[i](h + a)
        return self.out(h.transpose(1, 2)), tok

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
