"""谐噪比(HNR)：identity vs model 渲染后的有声帧 HNR 中位数。不奖励抹平，奖励谐波清晰。"""
from __future__ import annotations
import pathlib, sys
import numpy as np, soundfile as sf, torch, importlib.util
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "diffical_new"))
from model.pointer import PointerGen, gather_seg  # noqa: E402
from train.dataset_set import SetDataset, collate  # noqa: E402
from backend.vocoder.nsf_hifigan import NsfHifiGAN  # noqa: E402
SR, HOP, N_FFT, N_MELS = 44100, 512, 2048, 128
MODEL_DIR = ROOT.parent / "diffical_new/models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"


def hnr(y, f0, frame=1024):
    T = min(len(y) // HOP, len(f0))
    vals = []
    for t in range(T):
        f = f0[t]
        if f <= 50: continue
        c = int(t * HOP)
        if c + frame > len(y): break
        x = y[c:c + frame].astype(np.float64)
        x = x - x.mean()
        e = np.dot(x, x)
        if e < 1e-8: continue
        lag = int(round(SR / f))
        if lag < 2 or lag >= frame - 2: continue
        r = np.dot(x[:-lag], x[lag:]) / e * (frame / (frame - lag))
        r = min(max(r, 1e-6), 0.9999)
        vals.append(10 * np.log10(r / (1 - r)))
    return float(np.median(vals)) if vals else float("nan")


def main():
    data = ROOT/"train/data/mm_v2"; ckpt = ROOT/"train/ckpt_mm2/last.pt"
    ds = SetDataset(data, "eval"); n = min(6, len(ds))
    ck = torch.load(ckpt, map_location="cpu"); model = PointerGen(); model.load_state_dict(ck["state_dict"]); model.eval()
    voc = NsfHifiGAN(str(MODEL_DIR/"model.ckpt"), str(MODEL_DIR/"config.json"), device="cpu")
    rows = []
    for i in range(n):
        d = ds[i]; utt = ds.files[i].stem
        segs = d["segs"][None]; sl = d["seg_len"][None]; si = d["seg_idx"][None]
        pr = d["progress"][None]; bd = d["boundary"][None]; f0 = d["f0"][None]
        with torch.no_grad():
            pred = model(segs, sl, si, pr, bd, f0)[0].numpy()
            base = gather_seg(segs, sl, si, pr)[0].numpy()
        T = min(pred.shape[1], base.shape[1]); f0v = f0[0, :T].numpy()
        y_m = np.asarray(voc.spec2wav_torch(pred[:, :T], f0v), np.float32).reshape(-1)
        y_i = np.asarray(voc.spec2wav_torch(base[:, :T], f0v), np.float32).reshape(-1)
        hi, hm = hnr(y_i, f0v), hnr(y_m, f0v)
        rows.append((hi, hm))
        print(f"{utt}: HNR identity={hi:6.2f} model={hm:6.2f} {'BETTER' if hm > hi else 'worse'}", flush=True)
    a = np.array(rows)
    print(f"mean HNR: identity={np.nanmean(a[:,0]):.2f} model={np.nanmean(a[:,1]):.2f} delta={np.nanmean(a[:,1]-a[:,0]):+.2f}")


if __name__ == "__main__":
    main()
