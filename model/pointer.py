"""PointerGen：软指针 + 残差。

每帧 t 的段归属 seg_idx[t] 与段内进度 progress[t] 是已知条件。
模型只学两件事：
  1. 采样位置 τ(t)（段内帧坐标，初始化为 progress*(L-1)）
  2. 残差修正
mel[t] = bilinear(seg[seg_idx[t]], τ(t)) + residual(t)
插值是标准可微算子；拼接的"放哪/取哪"由模型决定。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MELS = 128


def gather_seg(segs: torch.Tensor, seg_len: torch.Tensor, seg_idx: torch.Tensor, pos: torch.Tensor):
    """segs (B,S,128,L), seg_len (B,S), seg_idx (B,T) 1-based, pos (B,T) in [0,1].

    返回 (B,128,T)：在每帧所属片段的 pos 位置做线性插值。
    """
    B, S, _, L = segs.shape
    T = pos.shape[1]
    i = (seg_idx - 1).clamp(0, S - 1)                          # (B,T)
    ln = seg_len.gather(1, i).float()                           # (B,T)
    ptr = (pos * (ln - 1).clamp_min(0)).clamp(0, L - 1)         # (B,T)
    i0 = ptr.floor().long().clamp(0, L - 1)
    i1 = (i0 + 1).clamp(0, L - 1)
    w = (ptr - i0.float()).unsqueeze(1)                         # (B,1,T)
    bi = torch.arange(B, device=segs.device)[:, None].expand(B, T)
    g0 = segs[bi, i, :, i0].permute(0, 2, 1)                    # (B,128,T)
    g1 = segs[bi, i, :, i1].permute(0, 2, 1)
    return (1 - w) * g0 + w * g1


class PointerGen(nn.Module):
    def __init__(self, c: int = 192, n_ctx: int = 3):
        super().__init__()
        self.seg_emb = nn.Embedding(256, 32, padding_idx=0)
        self.cond = nn.Sequential(
            nn.Conv1d(32 + 4, c, 5, padding=2), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv1d(c, c, 5, padding=2), nn.GroupNorm(8, c), nn.SiLU(),
        )
        self.tau = nn.Conv1d(c, 1, 3, padding=1)
        # 残差分支吃 [画布内容, 条件]，receptive field 覆盖边界两侧，才能生成过渡
        # n_ctx=1 只用当前段；3 加左右邻居
        self.n_ctx = n_ctx
        self.res = nn.Sequential(
            nn.Conv1d(N_MELS * n_ctx + c, c, 5, padding=2), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv1d(c, c, 5, padding=2), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv1d(c, N_MELS, 1),
        )
        # 零初始化：tau 与残差初始都为 0 → 模型起点 = "按进度取段内内容"，即恒等
        nn.init.zeros_(self.tau.weight); nn.init.zeros_(self.tau.bias)
        nn.init.zeros_(self.res[-1].weight); nn.init.zeros_(self.res[-1].bias)

    def forward(self, segs, seg_len, seg_idx, progress, boundary, f0):
        B = segs.shape[0]
        voiced = (f0 > 0).float()
        f0n = torch.log2(f0.clamp_min(1.0) / 440.0) * voiced
        x = torch.cat([self.seg_emb(seg_idx).transpose(1, 2), progress.unsqueeze(1),
                       boundary.unsqueeze(1), f0n.unsqueeze(1), voiced.unsqueeze(1)], dim=1)
        h = self.cond(x)
        dtau = torch.tanh(self.tau(h).squeeze(1)) * 0.5           # ±0.5 段内偏移
        pos = (progress + dtau).clamp(0, 1)
        base = gather_seg(segs, seg_len, seg_idx, pos)
        if self.n_ctx == 1:
            ctx = base
        else:
            S = segs.shape[1]
            pre = gather_seg(segs, seg_len, (seg_idx - 1).clamp(1, S), pos)
            nex = gather_seg(segs, seg_len, (seg_idx + 1).clamp(1, S), pos)
            ctx = torch.cat([base, pre, nex], dim=1)
        res = self.res(torch.cat([ctx, h], dim=1))
        return base + torch.tanh(res)


def load_pointer(path, device="cpu"):
    """按检查点形状自动推断 c 与 n_ctx，避免架构错配静默加载。"""
    ck = torch.load(str(path), map_location=device)
    sd = ck["state_dict"]
    c = sd["cond.0.weight"].shape[0]
    in_ch = sd["res.0.weight"].shape[1]
    n_ctx = (in_ch - c) // N_MELS
    m = PointerGen(c=c, n_ctx=n_ctx)
    m.load_state_dict(sd)
    m.eval()
    return m


def count_params(m):
    return sum(p.numel() for p in m.parameters())
