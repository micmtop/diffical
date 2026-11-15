"""留出集真实片段的三段对比：原录音 | 画布(声码器) | 模型(声码器)。"""
from __future__ import annotations
import pathlib, sys
import numpy as np, soundfile as sf, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "diffical_new"))
from model.pointer import load_pointer, gather_seg
from train.dataset_set import SetDataset, collate
from backend.vocoder.nsf_hifigan import NsfHifiGAN
SR, HOP = 44100, 512
MODEL_DIR = ROOT.parent / "diffical_new/models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"

def main():
    ds = SetDataset(ROOT / "train/data/all_v3", "eval")
    voc = NsfHifiGAN(str(MODEL_DIR / "model.ckpt"), str(MODEL_DIR / "config.json"), device="cpu")
    model = load_pointer(ROOT / "train/ckpt_gan/last.pt")
    picks = [0, 1]
    b = collate([ds[i] for i in picks])
    with torch.no_grad():
        pred = model(b["segs"], b["seg_len"], b["seg_idx"], b["progress"], b["boundary"], b["f0"])
        base = gather_seg(b["segs"], b["seg_len"], b["seg_idx"], b["progress"])
    gap = np.zeros(int(0.4 * SR), np.float32)
    parts, names = [], []
    for j in range(len(picks)):
        T = int(b["valid"][j].sum())
        f0 = b["f0"][j, :T].numpy()
        yb = np.asarray(voc.spec2wav_torch(base[j, :, :T].numpy(), f0), np.float32).reshape(-1)
        yp = np.asarray(voc.spec2wav_torch(pred[j, :, :T].numpy(), f0), np.float32).reshape(-1)
        ref = np.sqrt(np.mean(yb.astype(np.float64) ** 2)) + 1e-12
        n = min(len(yb), len(yp))
        yb = yb[:n]; yp = (yp[:n] * (ref / (np.sqrt(np.mean(yp[:n].astype(np.float64) ** 2)) + 1e-12))).astype(np.float32)
        parts += [yb, gap, yp, gap]
        names.append(ds.files[picks[j]].stem)
    out = ROOT / "listen_test" / ("26_真实片段对比_" + "_".join(names) + ".wav")
    sf.write(out, np.concatenate(parts), SR)
    print("顺序：画布(声码器) — 模型(声码器) — 画布 — 模型", "\n", out)

if __name__ == "__main__":
    main()
