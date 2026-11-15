"""在现有装配算法上加 83ms 交叉淡化，并对比三种输出的边界质量。

对比：hard（硬切，现状） / xfade（余弦交叉淡化） / model（指针模型输出）
指标：边界跳变 Δmel vs 段内 Δmel（边界应 ≤ 段内）
"""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np, soundfile as sf, torch, importlib.util
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "diffical_new"))
from model.pointer import PointerGen, gather_seg, load_pointer  # noqa: E402
from backend.vocoder.nsf_hifigan import NsfHifiGAN  # noqa: E402
SR, HOP, N_FFT, N_MELS = 44100, 512, 2048, 128
OV_MS = 83.3
MODEL_DIR = ROOT.parent / "diffical_new/models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
_sp = importlib.util.spec_from_file_location("neo_wav2mel", ROOT / "backend/wav2mel.py")
_M = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_M)
EX = _M.PitchAdjustableMelSpectrogram(sample_rate=SR, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
                                     n_mels=N_MELS, f_min=40.0, f_max=16000.0, center=False, device="cpu")
def mel_of(w):
    with torch.no_grad(): return EX(torch.from_numpy(w.astype(np.float32))).cpu().numpy()[0]


def assemble(segs, F, ov):
    """把每段放到 [i*F, (i+1)*F)，相邻段在 ov 帧内余弦交叉淡化（幅度域）。"""
    S = len(segs); T = S * F
    mag = np.zeros((128, T), np.float64)
    wsum = np.zeros(T, np.float64)
    w = np.linspace(0, 1, ov + 2)[1:-1]            # 余弦形
    w = 0.5 - 0.5 * np.cos(np.pi * w)
    for i, s in enumerate(segs):
        a = i * F
        # 段向后延长 ov 帧，形成真正的时间重叠（不是同位置混帧）
        L = min(F + ov, s.shape[1])
        seg = s[:, :L]
        wt = np.ones(seg.shape[1])
        if i > 0:
            k = min(ov, wt.size)
            wt[:k] = w[:k]                      # 前 ov 帧淡入
        if i < S - 1:
            k = min(ov, wt.size)
            wt[-k:] = 1.0 - w[:k]               # 后 ov 帧淡出
        hi = min(T, a + seg.shape[1])
        wt = wt[: hi - a]; seg = seg[:, : hi - a]
        mag[:, a:hi] += np.exp(seg) * wt[None, :]
        wsum[a:hi] += wt
    wsum = np.maximum(wsum, 1e-6)
    return np.log(np.maximum(mag / wsum[None, :], 1e-9)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="train/ckpt_fix/last.pt")
    ap.add_argument("--width", type=int, default=416)
    ap.add_argument("--f0", type=float, default=523.0)
    ap.add_argument("--frames", type=int, default=45)
    args = ap.parse_args()
    bank = pathlib.Path.home() / "Library/OpenUtau/Singers/KAFU Cevio/C5"
    segs = []
    for n in ["a.wav", "i.wav", "u.wav"]:
        w, sr = sf.read(str(bank / n), dtype="float32")
        if w.ndim > 1: w = w.mean(1)
        m = mel_of(w)
        if m.shape[1] >= 4: segs.append(m)
    S, F = len(segs), args.frames
    ov = int(round(OV_MS / (HOP / SR * 1000)))
    T = S * F
    Lmax = max(s.shape[1] for s in segs)
    arr = np.zeros((S, 128, Lmax), np.float32)
    for i, s in enumerate(segs): arr[i, :, :s.shape[1]] = s
    lens = np.array([s.shape[1] for s in segs])
    idx = np.zeros((1, T), np.int64); prog = np.zeros((1, T), np.float32)
    for i in range(S):
        idx[0, i*F:(i+1)*F] = i + 1; prog[0, i*F:(i+1)*F] = np.linspace(0, 1, F)
    bd = np.zeros((1, T), np.float32)
    for i in range(1, S): bd[0, i*F-2:i*F+3] = 1
    f0 = np.full((1, T), args.f0, np.float32)
    m = load_pointer(ROOT / args.ckpt)
    with torch.no_grad():
        pred = m(torch.from_numpy(arr).unsqueeze(0), torch.from_numpy(lens).unsqueeze(0),
                 torch.from_numpy(idx), torch.from_numpy(prog), torch.from_numpy(bd),
                 torch.from_numpy(f0))[0].numpy()
        hard = gather_seg(torch.from_numpy(arr).unsqueeze(0), torch.from_numpy(lens).unsqueeze(0),
                          torch.from_numpy(idx), torch.from_numpy(prog))[0].numpy()
    xf = assemble(segs, F, ov)
    def jumps(x):
        return [float(np.abs(x[:, i*F] - x[:, i*F-1]).mean()) for i in range(1, S)]
    def interior(x):
        return float(np.median([np.abs(np.diff(x[:, i*F+2:(i+1)*F-2], axis=1)).mean() for i in range(S)]))
    for name, x in [("hard", hard), ("xfade", xf), ("model", pred)]:
        print(f"{name:6s} 边界跳变={[round(v,3) for v in jumps(x)]} 段内Δmel={interior(x):.4f} 整句Δmel={np.abs(np.diff(x,axis=1)).mean():.4f}")
    voc = NsfHifiGAN(str(MODEL_DIR / "model.ckpt"), str(MODEL_DIR / "config.json"), device="cpu")
    gap = np.zeros(int(0.3 * SR), np.float32)
    parts, ref = [], None
    for name, x in [("xfade", xf), ("model", pred)]:
        y = np.asarray(voc.spec2wav_torch(x, f0[0]), np.float32).reshape(-1)
        r = np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12
        if ref is None: ref = r
        y = (y * (ref / r)).astype(np.float32)
        parts += [y, gap]
    out = ROOT / "listen_test" / "24_交叉淡化_vs_模型.wav"
    sf.write(out, np.concatenate(parts), SR)
    print(out)


if __name__ == "__main__":
    main()
