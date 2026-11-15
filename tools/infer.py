"""统一推理入口：片段集合 + 条件 → 整句 mel → 声码器 → wav。

用法：
  python3 tools/infer.py --data train/data/all_v3 --split eval --idx 0 3 5 --ckpt train/ckpt_gan/last.pt --out listen_test/27_demo.wav [--compare]
"""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np, soundfile as sf, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "diffical_new"))
from model.pointer import load_pointer, gather_seg
from train.dataset_set import SetDataset, collate
from backend.vocoder.nsf_hifigan import NsfHifiGAN
SR = 44100
MODEL_DIR = ROOT.parent / "diffical_new/models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train/data/all_v3")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--idx", type=int, nargs="+", default=[0])
    ap.add_argument("--ckpt", default="train/ckpt_gan/last.pt")
    ap.add_argument("--out", default="listen_test/27_demo.wav")
    ap.add_argument("--compare", action="store_true", help="同时输出画布版本做对照")
    args = ap.parse_args()
    ds = SetDataset(ROOT / args.data, args.split)
    b = collate([ds[i] for i in args.idx])
    model = load_pointer(ROOT / args.ckpt)
    voc = NsfHifiGAN(str(MODEL_DIR / "model.ckpt"), str(MODEL_DIR / "config.json"), device="cpu")
    with torch.no_grad():
        pred = model(b["segs"], b["seg_len"], b["seg_idx"], b["progress"], b["boundary"], b["f0"])
        base = gather_seg(b["segs"], b["seg_len"], b["seg_idx"], b["progress"])
    gap = np.zeros(int(0.35 * SR), np.float32)
    parts = []
    for j in range(len(args.idx)):
        T = int(b["valid"][j].sum())
        f0 = b["f0"][j, :T].numpy()
        y = np.asarray(voc.spec2wav_torch(pred[j, :, :T].numpy(), f0), np.float32).reshape(-1)
        parts.append(y); parts.append(gap)
        if args.compare:
            yb = np.asarray(voc.spec2wav_torch(base[j, :, :T].numpy(), f0), np.float32).reshape(-1)
            parts.append(yb); parts.append(gap)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, np.concatenate(parts), SR)
    print("写出的顺序：" + ("模型/画布 交替" if args.compare else "仅模型") + f"\n{out}")


if __name__ == "__main__":
    main()
