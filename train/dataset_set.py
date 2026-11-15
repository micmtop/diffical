"""SetGen 数据集：片段集合 + 帧条件。"""
from __future__ import annotations
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset


class SetDataset(Dataset):
    def __init__(self, root: pathlib.Path, split: str):
        self.files = sorted((pathlib.Path(root) / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(root / split)
        self._cache: dict[int, dict] = {}

    def __len__(self):
        return len(self.files)

    def _load(self, i: int) -> dict:
        if i in self._cache:
            return self._cache[i]
        z = np.load(self.files[i], allow_pickle=True)
        S = len(z["seg_len"])
        Lmax = int(z["seg_len"].max())
        segs = z["segs"][:, :, :Lmax].astype(np.float32)
        mask = np.zeros((S, Lmax), dtype=np.bool_)
        for k, L in enumerate(z["seg_len"]):
            mask[k, : int(L)] = True
        seg_len = z["seg_len"].astype(np.int64)
        rec = dict(segs=segs, mask=mask, seg_len=torch.from_numpy(seg_len),
                   seg_idx=z["seg_idx"].astype(np.int64),
                   progress=z["progress"].astype(np.float32),
                   boundary=z["boundary"].astype(np.float32),
                   f0=z["f0"].astype(np.float32),
                   target=z["target"].astype(np.float32), utt=str(z["utt"]))
        self._cache[i] = rec
        return rec

    def __getitem__(self, i: int) -> dict:
        r = self._load(i)
        return {k: (v if isinstance(v, torch.Tensor) else torch.from_numpy(v))
                for k, v in r.items() if k != "utt"}


def collate(batch: list[dict]) -> dict:
    S = max(b["segs"].shape[0] for b in batch)
    L = max(b["segs"].shape[2] for b in batch)
    T = max(b["target"].shape[1] for b in batch)
    B = len(batch)
    segs = torch.zeros(B, S, 128, L)
    mask = torch.zeros(B, S, L, dtype=torch.bool)
    seg_idx = torch.zeros(B, T, dtype=torch.long)
    progress = torch.zeros(B, T)
    boundary = torch.zeros(B, T)
    f0 = torch.zeros(B, T)
    target = torch.zeros(B, 128, T)
    seg_len = torch.zeros(B, S, dtype=torch.long)
    for j, b in enumerate(batch):
        s, l, t = b["segs"].shape[0], b["segs"].shape[2], b["target"].shape[1]
        segs[j, :s, :, :l] = b["segs"]
        mask[j, :s, :l] = b["mask"]
        seg_idx[j, :t] = b["seg_idx"]
        progress[j, :t] = b["progress"]
        boundary[j, :t] = b["boundary"]
        f0[j, :t] = b["f0"]
        target[j, :, :t] = b["target"]
        seg_len[j, :s] = b["seg_len"]
    valid = (seg_idx > 0).float()
    return dict(segs=segs, seg_mask=mask, seg_len=seg_len, seg_idx=seg_idx, progress=progress,
                boundary=boundary, f0=f0, target=target, valid=valid)
