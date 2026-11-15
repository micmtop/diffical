"""P0.5 数据集（向量化版）：读 train/data/pairs1000/{split}/*.npz。

v2 变化：把逐帧 Python 循环改成一次性向量化 + 整条记录缓存。
旧版在 batch=2 时只有 6–8 it/s，瓶颈是 CPU 不是 GPU。
"""
from __future__ import annotations

import json
import pathlib
import random

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]


def build_vocab(pairs_dir: pathlib.Path) -> dict[str, int]:
    phones = set()
    for p in sorted((pairs_dir / "train").glob("*.npz")):
        phones.update([str(x) for x in np.load(p, allow_pickle=True)["phones"]])
    vocab = {p: i + 1 for i, p in enumerate(sorted(phones))}
    vocab["<pad>"] = 0
    (pairs_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    return vocab


class PairDataset(Dataset):
    def __init__(self, pairs_dir: pathlib.Path, split: str, frames: int, vocab: dict[str, int],
                 seed: int = 0, cache_all: bool = True):
        self.files = sorted((pairs_dir / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(pairs_dir / split)
        self.frames = frames
        self.vocab = vocab
        self.rng = random.Random(seed)
        self.cache_all = cache_all
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, i: int) -> dict:
        if i in self._cache:
            return self._cache[i]
        z = np.load(self.files[i], allow_pickle=True)
        tgt = z["target_mel"].astype(np.float32)
        src = z["source_mel"].astype(np.float32)
        utt = str(z["utt"])
        side = ROOT / "train/data/f0_rmvpe" / f"{utt}.npy"
        if side.exists():
            f0 = np.load(side).astype(np.float32)
            if f0.shape[0] != z["target_mel"].shape[1]:
                f0 = np.interp(np.linspace(0, 1, z["target_mel"].shape[1]),
                               np.linspace(0, 1, f0.shape[0]), f0).astype(np.float32)
        else:
            f0 = z["f0"].astype(np.float32)
        pid = z["phone_ids"].astype(np.int64)
        bd = (z["boundary"] > 0.5).astype(np.float32)
        phones = [str(x) for x in z["phones"]]
        T = tgt.shape[1]

        lut = np.array([self.vocab.get(p, 0) for p in phones] or [0], dtype=np.int64)
        valid = pid >= 0
        ph = np.zeros(T, dtype=np.int64)
        if valid.any():
            ph[valid] = lut[np.clip(pid[valid], 0, len(lut) - 1)]

        # 每个连续 run = 一个音素：进度与时长
        change = np.where(np.diff(pid) != 0)[0] + 1 if T > 1 else np.array([], dtype=np.int64)
        starts = np.r_[0, change]
        ends = np.r_[change, T]
        prog = np.zeros(T, dtype=np.float32)
        dur = np.zeros(T, dtype=np.float32)
        for s, e in zip(starts, ends):
            n = int(e - s)
            if n <= 0:
                continue
            prog[s:e] = np.linspace(0.0, 1.0, n, dtype=np.float32) if n > 1 else 0.0
            dur[s:e] = n / 100.0

        # 到最近边界的帧距（searchsorted，向量化）
        bidx = np.where(bd > 0.5)[0]
        if bidx.size:
            pos = np.arange(T)
            j = np.searchsorted(bidx, pos)
            left = np.where(j > 0, pos - bidx[np.clip(j - 1, 0, bidx.size - 1)], 99)
            right = np.where(j < bidx.size, bidx[np.clip(j, 0, bidx.size - 1)] - pos, 99)
            dist = np.minimum(np.minimum(np.abs(left), np.abs(right)), 99).astype(np.float32) / 10.0
        else:
            dist = np.zeros(T, dtype=np.float32)

        rec = {"tgt": tgt, "src": src, "f0": f0, "ph": ph, "bd": bd,
               "prog": prog, "dist": dist, "dur": dur, "T": T}
        if self.cache_all:
            self._cache[i] = rec
        return rec

    def __getitem__(self, i: int) -> dict:
        r = self._load(i)
        T = r["T"]
        L = min(self.frames, T)
        start = self.rng.randrange(0, max(1, T - L + 1))
        sl = slice(start, start + L)
        return {
            "source": torch.from_numpy(np.ascontiguousarray(r["src"][:, sl])),
            "target": torch.from_numpy(np.ascontiguousarray(r["tgt"][:, sl])),
            "f0": torch.from_numpy(r["f0"][sl].copy()),
            "phone": torch.from_numpy(r["ph"][sl].copy()),
            "boundary": torch.from_numpy(r["bd"][sl].copy()),
            "progress": torch.from_numpy(r["prog"][sl].copy()),
            "dist_bd": torch.from_numpy(r["dist"][sl].copy()),
            "dur": torch.from_numpy(r["dur"][sl].copy()),
        }


def collate(batch: list[dict]) -> dict:
    L = min(b["source"].shape[-1] for b in batch)
    return {k: torch.stack([b[k][..., :L] for b in batch]) for k in batch[0]}
