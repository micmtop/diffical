"""用 PointerGen 做整句装配（替代 MelOverlapAccumulator）。

输入：phone_features（每个含已拉伸好的 mel 与帧起点）+ 整句 f0
输出：整句 log-mel (128, T)

环境变量：
  DIFFICAL_ENGINE=model        启用模型装配
  DIFFICAL_POINTER_CKPT=<path> 模型权重（默认 train/ckpt_gan/last.pt）
  DIFFICAL_SEAM_RAMP=<N>       接缝安全过渡帧数（0=关闭）
  DIFFICAL_SEAM_RAMP_GATE=<x>  只对输入接缝跳变>x 的边界启用（默认 1.5）
"""
from __future__ import annotations

import os
import pathlib
import threading
from typing import List

import numpy as np
import torch

from .pointer_model import load_pointer

_LOCK = threading.Lock()
_CACHE = {}

BOUNDARY_W = 4
# 接缝安全过渡（帧数，0=关闭）：把单帧台阶摊到 2k+1 帧，只改接缝附近。
# 生产默认关闭，用 DIFFICAL_SEAM_RAMP 显式开启做 A/B。
def _model():
    path = os.environ.get("DIFFICAL_POINTER_CKPT")
    if not path:
        path = str(pathlib.Path(__file__).resolve().parents[1] / "train" / "ckpt_gan" / "last.pt")
    with _LOCK:
        m = _CACHE.get(path)
        if m is None:
            m = load_pointer(path)
            m.eval()
            _CACHE[path] = m
    return m


def assemble(phone_features: List[dict], total_frames: int, f0: np.ndarray,
             n_mels: int = 128) -> np.ndarray:
    feats = sorted(phone_features, key=lambda f: f["start_frame"])
    feats = [f for f in feats if f.get("mel") is not None and f["mel"].shape[1] > 0]
    if not feats:
        return np.full((n_mels, total_frames), -30.0, dtype=np.float32)

    norm = os.environ.get("DIFFICAL_SEG_NORM", "1").lower() not in ("0", "false", "no")
    feats_n = []
    for f in feats:
        mel = np.asarray(f["mel"], dtype=np.float32)
        a = int(round(float(f.get("start_frame_exact", f["start_frame"]))))
        a = max(0, min(total_frames - 1, a))
        b = min(total_frames, a + mel.shape[1])
        if b - a < 2:
            continue
        feats_n.append((a, b, mel))
    segs, lens, spans = [], [], []
    for i, (a, b, mel) in enumerate(feats_n):
        if norm:
            # 生产段是重叠的：只保留会被 seg_idx 用到的帧，并让 progress 走满 0..1，
            # 与训练时的段语义一致。
            used = (feats_n[i + 1][0] - a) if i + 1 < len(feats_n) else (b - a)
            used = max(2, min(used, mel.shape[1]))
            seg = mel[:, :used]
            b_used = a + used
        else:
            seg = mel[:, : b - a]
            b_used = b
        segs.append(seg)
        lens.append(seg.shape[1])
        spans.append((a, b_used))
    if not segs:
        return np.full((n_mels, total_frames), -30.0, dtype=np.float32)

    S = len(segs)
    Lmax = max(lens)
    arr = np.full((S, n_mels, Lmax), -30.0, dtype=np.float32)
    for i, s in enumerate(segs):
        arr[i, :, : s.shape[1]] = s
    seg_idx = np.zeros(total_frames, dtype=np.int64)
    prog = np.zeros(total_frames, dtype=np.float32)
    for i, (a, b) in enumerate(spans):
        seg_idx[a:b] = i + 1
        prog[a:b] = np.linspace(0, 1, b - a)
    boundary = np.zeros(total_frames, dtype=np.float32)
    for a, _ in spans[1:]:
        boundary[max(0, a - BOUNDARY_W): min(total_frames, a + BOUNDARY_W + 1)] = 1.0

    f0v = np.asarray(f0, dtype=np.float32)[:total_frames]
    if f0v.shape[0] < total_frames:
        f0v = np.pad(f0v, (0, total_frames - f0v.shape[0]))
    f0v = np.where((f0v > 0) & (f0v < 55.0), 0.0, f0v)

    m = _model()
    with torch.no_grad():
        out = m(
            torch.from_numpy(arr).unsqueeze(0),
            torch.from_numpy(np.asarray(lens, dtype=np.int64)).unsqueeze(0),
            torch.from_numpy(seg_idx).unsqueeze(0),
            torch.from_numpy(prog).unsqueeze(0),
            torch.from_numpy(boundary).unsqueeze(0),
            torch.from_numpy(f0v).unsqueeze(0),
        )[0].numpy()
    return out.astype(np.float32)
