"""谱细节指标：高频能量占比 + 谱平坦度，用来判断 GAN 是否真的补了细节。"""
from __future__ import annotations
import sys, pathlib
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.dataset_set import SetDataset, collate
from model.pointer import load_pointer, gather_seg

def hf_ratio(m):        # mel 80-127 段（约 8kHz 以上）能量占比
    p = np.exp(m.astype(np.float64))
    return float(p[80:].sum() / max(p.sum(), 1e-9))

def flatness(m):        # 谱平坦度（越大越像噪声）
    p = np.exp(m.astype(np.float64)) + 1e-12
    return float(np.exp(np.log(p).mean(0)).mean() / max(p.mean(0).mean(), 1e-9))

def run(ckpt):
    ds = SetDataset(ROOT / "train/data/all_v3", "eval")
    b = collate([ds[i] for i in range(24)])
    m = load_pointer(ckpt)
    with torch.no_grad():
        pred = m(b["segs"], b["seg_len"], b["seg_idx"], b["progress"], b["boundary"], b["f0"])
        base = gather_seg(b["segs"], b["seg_len"], b["seg_idx"], b["progress"])
    out = {}
    for name, x in [("canvas", base), ("target", b["target"]), ("model", pred)]:
        arr = x.numpy(); v = b["valid"].numpy().astype(bool)
        hf = np.mean([hf_ratio(arr[j][:, v[j]]) for j in range(arr.shape[0])])
        fl = np.mean([flatness(arr[j][:, v[j]]) for j in range(arr.shape[0])])
        out[name] = (hf, fl)
    return out

if __name__ == "__main__":
    for ck in sys.argv[1:]:
        r = run(ROOT / ck)
        print(ck)
        for k, (hf, fl) in r.items():
            print(f"  {k:7s} 高频占比={hf:.4f} 平坦度={fl:.4f}")
