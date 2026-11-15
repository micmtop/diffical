"""Mel 域 Patch 判别器：判"这段谱像不像真实演唱"，用于对抗过平滑。"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout, k=5, s=2):
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, s, padding=k // 2), nn.LeakyReLU(0.2, True),
    )


class PatchDisc(nn.Module):
    """输入 (B,128,T) log-mel，输出 patch 分数 (B,1,T')。"""

    def __init__(self, c=64):
        super().__init__()
        self.net = nn.Sequential(
            _block(128, c), _block(c, c * 2), _block(c * 2, c * 4),
            _block(c * 4, c * 4),
            nn.Conv1d(c * 4, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def hinge_d_loss(real_logits, fake_logits):
    return torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits):
    return -fake_logits.mean()
