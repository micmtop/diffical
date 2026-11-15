"""用真实声库样本跑模型推理：声库 wav → 片段 mel → PointerGen → mel → 声码器。

不依赖用户 USTX：从声库里取几个元音样本，按目标音高/时长拼成一句，比较
  raw   : 样本直接拼接（无模型、无过渡）
  ident : 恒等指针（按进度取样本内容）
  model : 指针模型输出
"""
from __future__ import annotations
import argparse, pathlib, sys, glob
import numpy as np, soundfile as sf, torch, importlib.util
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "diffical_new"))
from model.pointer import PointerGen, gather_seg  # noqa: E402
from backend.vocoder.nsf_hifigan import NsfHifiGAN  # noqa: E402

SR, HOP, N_FFT, N_MELS = 44100, 512, 2048, 128
MODEL_DIR = ROOT.parent / "diffical_new/models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
_sp = importlib.util.spec_from_file_location("neo_wav2mel", ROOT / "backend/wav2mel.py")
_M = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_M)
EX = _M.PitchAdjustableMelSpectrogram(sample_rate=SR, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
                                     n_mels=N_MELS, f_min=40.0, f_max=16000.0, center=False, device="cpu")


def mel_of(w):
    with torch.no_grad():
        return EX(torch.from_numpy(w.astype(np.float32))).cpu().numpy()[0]


def parse_oto(path):
    out = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split("=", 1)
        if len(p) != 2: continue
        f = p[1].split(",")
        if len(f) < 6: continue
        try: v = [float(x) for x in f[1:6]]
        except Exception: continue
        out[p[0].strip()] = (f[0].strip(), v)  # wav -> (alias, params)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="")
    ap.add_argument("--ckpt", default="train/ckpt_mm2/last.pt")
    ap.add_argument("--aliases", default="a,i,u")
    ap.add_argument("--f0", type=float, default=300.0)
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--width", type=int, default=192)
    args = ap.parse_args()
    root = pathlib.Path.home() / "Library/OpenUtau/Singers"
    variants = {"a": ("a", "あ"), "i": ("i", "い"), "u": ("u", "う")}
    tags = [t.lower() for t in args.aliases.split(",")]

    def match(tag, wav, alias):
        vs = variants.get(tag, (tag,))
        stem = wav.rsplit(".", 1)[0].lower()
        return stem in vs or any(x in alias for x in vs)

    if args.bank:
        inis = sorted((root / args.bank).rglob("*.ini"))
    else:
        inis = sorted(root.rglob("*.ini"))
    bank = None; oto = None; picked = None
    for ini in inis:
        o = parse_oto(ini)
        if not o:
            continue
        got = []
        for tag in tags:
            for wav, (alias, params) in o.items():
                if match(tag, wav, alias):
                    got.append((tag, wav, params)); break
        if len(got) >= max(2, len(tags) - 1):
            bank, oto, picked = ini.parent, o, got
            print("oto:", ini)
            break
    if not bank:
        print("no bank with usable vowels"); return 1
    print("bank:", bank.name, "entries:", len(oto))
    segs, lens = [], []
    for a, wavname, (off, con, cut, pre, ov) in picked:
        wp = bank / wavname
        if not wp.exists():
            g = list(bank.glob(wavname))
            if not g: continue
            wp = g[0]
        w, sr = sf.read(str(wp), dtype="float32")
        if w.ndim > 1: w = w.mean(1)
        w = w[int(off * sr / 1000): int((len(w) / sr * 1000 + cut) * sr / 1000)]
        if len(w) < 4096: continue
        m = mel_of(w)                                # (128, L)
        if m.shape[1] < 4: continue
        segs.append(m); lens.append(m.shape[1])
    if not segs:
        print("no usable samples"); return 1
    S = len(segs); Lmax = max(lens)
    arr = np.zeros((S, 128, Lmax), np.float32)
    for i, s in enumerate(segs): arr[i, :, : s.shape[1]] = s
    F = args.frames
    T = F * S
    seg_idx = np.zeros((1, T), np.int64); prog = np.zeros((1, T), np.float32)
    for i in range(S):
        seg_idx[0, i*F:(i+1)*F] = i + 1
        prog[0, i*F:(i+1)*F] = np.linspace(0, 1, F)
    bd = np.zeros((1, T), np.float32)
    for i in range(1, S): bd[0, i*F-2:i*F+3] = 1
    f0 = np.full((1, T), args.f0, np.float32)
    bt = dict(segs=torch.from_numpy(arr).unsqueeze(0), seg_len=torch.from_numpy(np.array(lens, np.int64)).unsqueeze(0),
              seg_idx=torch.from_numpy(seg_idx), progress=torch.from_numpy(prog),
              boundary=torch.from_numpy(bd), f0=torch.from_numpy(f0))
    ck = torch.load(ROOT / args.ckpt, map_location="cpu")
    model = PointerGen(c=args.width); model.load_state_dict(ck["state_dict"]); model.eval()
    with torch.no_grad():
        pred = model(bt["segs"], bt["seg_len"], bt["seg_idx"], bt["progress"], bt["boundary"], bt["f0"])[0].numpy()
        base = gather_seg(bt["segs"], bt["seg_len"], bt["seg_idx"], bt["progress"])[0].numpy()
    voc = NsfHifiGAN(str(MODEL_DIR / "model.ckpt"), str(MODEL_DIR / "config.json"), device="cpu")
    f0v = f0[0]
    raw = np.zeros(T * 128, np.float32)
    for i, s in enumerate(segs):
        src = base[:, i*F:(i+1)*F]
        raw[i*F*128:(i+1)*F*128] = 0
    out = {}
    out["ident"] = np.asarray(voc.spec2wav_torch(base, f0v), np.float32).reshape(-1)
    out["model"] = np.asarray(voc.spec2wav_torch(pred, f0v), np.float32).reshape(-1)
    gap = np.zeros(int(0.35 * SR), np.float32)
    parts = []; ref = None
    for k in ["ident", "model"]:
        y = out[k]
        if ref is None: ref = np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12
        r = np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12
        y = (y * (ref / r)).astype(np.float32)
        parts += [y, gap]
        print(k, "peak", round(float(np.max(np.abs(y))), 3), "rms", round(float(r), 5))
    p = ROOT / "listen_test" / f"21_声库推理_{bank.name}_{'_'.join(args.aliases.split(','))}.wav"
    sf.write(p, np.concatenate(parts), SR)
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
