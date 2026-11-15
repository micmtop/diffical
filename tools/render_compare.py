"""感知验证：同一句，渲染 real / identity gather / 模型输出，全部过声码器。"""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np, soundfile as sf, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "diffical_new"))
from model.pointer import PointerGen, gather_seg  # noqa: E402
from train.dataset_set import SetDataset, collate  # noqa: E402
from backend.vocoder.nsf_hifigan import NsfHifiGAN  # noqa: E402
import importlib.util  # noqa: E402
_sp = importlib.util.spec_from_file_location("neo_wav2mel", ROOT / "backend/wav2mel.py")
_M = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_M)
SR, HOP, N_FFT, N_MELS = 44100, 512, 2048, 128
MODEL_DIR = ROOT.parent / "diffical_new/models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
EX = _M.PitchAdjustableMelSpectrogram(sample_rate=SR, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
                                     n_mels=N_MELS, f_min=40.0, f_max=16000.0, center=False, device="cpu")


def mel_of(w):
    with torch.no_grad(): return EX(torch.from_numpy(w.astype(np.float32))).cpu().numpy()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="train/data/mm_v2")
    ap.add_argument("--i", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    ds = SetDataset(ROOT / args.data, "eval")
    d = ds[args.i]
    b = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}
    dev = torch.device(args.device)
    b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
    ck = torch.load(ROOT / args.ckpt, map_location=dev)
    model = PointerGen().to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
    with torch.no_grad():
        pred = model(b["segs"], b["seg_len"], b["seg_idx"], b["progress"], b["boundary"], b["f0"])
        base = gather_seg(b["segs"], b["seg_len"], b["seg_idx"], b["progress"])
    utt = ds.files[args.i].stem
    wav, sr = sf.read(str(ROOT / "data/opencpop/wavs" / f"{utt}.wav"), dtype="float32")
    assert sr == SR
    voc = NsfHifiGAN(str(MODEL_DIR / "model.ckpt"), str(MODEL_DIR / "config.json"), device="cpu")
    mel_real = mel_of(wav)
    T = min(mel_real.shape[1], pred.shape[-1])
    f0 = b["f0"][0, :T].cpu().numpy()
    items = [("real", wav[: T * HOP + N_FFT]),
             ("identity", voc.spec2wav_torch(base[0, :, :T].cpu().numpy(), f0)),
             ("model", voc.spec2wav_torch(pred[0, :, :T].cpu().numpy(), f0))]
    gap = np.zeros(int(0.35 * SR), np.float32)
    parts = []; ref = None
    for name, y in items:
        y = np.asarray(y, np.float32).reshape(-1)
        if ref is None: ref = np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12
        r = np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12
        y = (y * (ref / r)).astype(np.float32)
        parts += [y, gap]
        print(name, "len", len(y), "peak", round(float(np.max(np.abs(y))), 2))
    out = ROOT / "listen_test" / f"20_感知验证_{utt}_real_identity_model.wav"
    sf.write(out, np.concatenate(parts), SR)
    print(out)


if __name__ == "__main__":
    main()
