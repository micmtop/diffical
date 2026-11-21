"""HN-SEP torch vs onnxruntime parity on a real 1-2 s wav.

The reference renderer runs the hnsep CascadedNet through torch. The Rust
engine runs the exported ONNX mask network through onnxruntime with its own
STFT/ISTFT. This script isolates the first half of that comparison:

  * torch:  CascadedNet.predict_fromaudio (full torch STFT + mask + ISTFT)
  * onnx:   torch STFT, ORT mask, torch ISTFT (identical preprocessing)

Both branches share the STFT/ISTFT code, so the reported difference is the
model/export difference only. ``--dump-dir`` also writes the reference
harmonic, noise and input wav as raw little-endian f32 for the Rust
``hnsep-parity`` check, which then covers the STFT/ISTFT port too.

Usage:
  python rust/tools/hnsep_parity.py \
      --model /path/to/hnsep/vr/model.pt \
      --onnx  ../export/hnsep.onnx \
      --wav   /path/to/1-2s.wav \
      [--seconds 1.6] [--dump-dir /tmp/hnsep_golden]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

NEO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NEO))

from backend.vocoder.hnsep_nets import CascadedNet  # noqa: E402

N_FFT = 2048
HOP = 512
SEG = 32 * HOP


def build_net(model_path: pathlib.Path) -> CascadedNet:
    net = CascadedNet(
        n_fft=N_FFT, hop_length=HOP, nout=32, nout_lstm=128,
        is_complex=True, is_mono=True, fixed_length=True,
    )
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(state)
    net.eval()
    return net


def prep(wav: np.ndarray):
    """Reference pre-pad + STFT: returns (spec complex (1,1,F,T), tl_pad, t)."""
    x = torch.from_numpy(wav.astype(np.float32)).reshape(1, 1, -1)
    t = x.shape[-1]
    t1 = t + HOP
    t_pad = SEG * ((t1 - 1) // SEG + 1) - t1
    nl_pad = t_pad // 2 // HOP
    tl_pad = nl_pad * HOP
    x = F.pad(x, (tl_pad, t_pad - tl_pad))
    spec = torch.stft(
        x.reshape(-1, x.shape[-1]),
        n_fft=N_FFT, hop_length=HOP, return_complex=True,
        window=torch.hann_window(N_FFT), pad_mode="constant",
    )
    spec = spec.reshape(1, 1, spec.shape[-2], spec.shape[-1])
    return spec, tl_pad, t


def finish(spec, mask_complex, tl_pad, t):
    pred = (spec * mask_complex).reshape(-1, spec.shape[-2], spec.shape[-1])
    out = torch.istft(pred, N_FFT, HOP, window=torch.hann_window(N_FFT))
    return out[:, tl_pad: tl_pad + t].reshape(-1).numpy()


def separate_torch(net: CascadedNet, wav: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return net.predict_fromaudio(
            torch.from_numpy(wav.astype(np.float32)).reshape(1, 1, -1)
        ).reshape(-1).numpy()


def separate_onnx(net: CascadedNet, onnx_path: pathlib.Path, wav: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    spec, tl_pad, t = prep(wav)
    inp = torch.cat([spec.real, spec.imag], dim=1).numpy()  # (1,2,F,T)
    mask = sess.run(["mask"], {"spec": inp})[0]
    mask_complex = torch.complex(
        torch.from_numpy(mask[:, 0]), torch.from_numpy(mask[:, 1])
    ).unsqueeze(1)
    return finish(spec, mask_complex, tl_pad, t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--seconds", type=float, default=1.6)
    ap.add_argument("--dump-dir", default=None,
                    help="write input/harmonic/noise f32 goldens for the Rust check")
    args = ap.parse_args()

    import soundfile as sf

    wav, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    if sr != 44100:
        raise SystemExit(f"wav must be 44.1 kHz, got {sr}")
    wav = wav.mean(axis=1)[: int(args.seconds * sr)]
    wav = np.ascontiguousarray(wav, dtype=np.float32)

    net = build_net(pathlib.Path(args.model))
    torch_harm = separate_torch(net, wav)
    onnx_harm = separate_onnx(net, pathlib.Path(args.onnx), wav)

    diff = np.abs(torch_harm - onnx_harm)
    scale = float(np.abs(torch_harm).max())
    print(f"wav: {wav.size} samples @ {sr} Hz ({wav.size / sr:.3f} s)")
    print(f"harmonic: rms_torch={np.sqrt(np.mean(torch_harm ** 2)):.6f} "
          f"max_torch={scale:.6f} rms_db={20 * np.log10(np.sqrt(np.mean(torch_harm ** 2)) / np.sqrt(np.mean(wav ** 2))):.3f}")
    print(f"torch vs onnx: max_abs_diff={diff.max():.3e} "
          f"mean_abs_diff={diff.mean():.3e} rel={diff.max() / max(scale, 1e-12):.3e}")

    if args.dump_dir:
        out = pathlib.Path(args.dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, arr in (("wav", wav), ("harmonic", torch_harm),
                          ("noise", wav - torch_harm), ("onnx_harmonic", onnx_harm)):
            np.asarray(arr, dtype=np.float32).tofile(out / f"{name}.f32")
        (out / "meta.json").write_text(
            f'{{"samples": {wav.size}, "sample_rate": {sr}, '
            f'"wav": "{pathlib.Path(args.wav).name}", "seconds": {args.seconds}}}'
        )
        print(f"dumped goldens to {out}")

    if diff.max() >= 1e-4:
        raise SystemExit(f"HN-SEP parity failed: max abs diff {diff.max():.3e} >= 1e-4")
    print("HN-SEP parity PASS (gate 1e-4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
