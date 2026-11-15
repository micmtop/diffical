"""用 RMVPE 重提 F0，写成 sidecar：train/data/f0_rmvpe/<utt>.npy

在算力机（Python 3.13 + rmvpe-onnx）上跑：
    py -3.13 train/refit_f0.py --pairs train/data/pairs1000 --wavs data/zh_corpus/wavs --model D:/diffical_neo/rmvpe.onnx

输出与 npz 里 target_mel 的帧数 T 严格对齐（86.1328 fps），float16。
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = pathlib.Path(__file__).resolve().parents[1]
SR_IN = 44100
SR_RMVPE = 16000
HOP = 512
FRAME_MS = HOP / SR_IN * 1000.0


def resample_to_16k(x: np.ndarray) -> np.ndarray:
    # 44100 -> 16000 = 441:160
    return resample_poly(x.astype(np.float64), 160, 441).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="train/data/pairs1000")
    ap.add_argument("--wavs", default="data/zh_corpus/wavs")
    ap.add_argument("--model", default="D:/diffical_neo/rmvpe.onnx")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from rmvpe_onnx import RMVPE

    model = RMVPE(model_path=args.model, device=args.device)
    pairs = ROOT / args.pairs
    wav_dir = ROOT / args.wavs
    out_dir = ROOT / "train/data/f0_rmvpe"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted((pairs / "train").glob("*.npz")) + sorted((pairs / "eval").glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    print(f"files={len(files)} model={args.model} device={args.device}", flush=True)

    done = 0
    for k, p in enumerate(files, 1):
        try:
            z = np.load(p, allow_pickle=True)
            utt = str(z["utt"])
            T = int(z["target_mel"].shape[1])
            wav_path = wav_dir / f"{utt}.wav"
            if not wav_path.exists():
                print(f"MISSING {wav_path}", flush=True)
                continue
            wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != SR_IN:
                raise ValueError(f"{utt}: sr={sr}")
            x16 = resample_to_16k(wav)
            f0_100, *_ = model.predict(x16, SR_RMVPE)   # out[1] 是 F0(Hz)，100fps
            f0 = np.asarray(f0_100, dtype=np.float32)
            dur_ms = len(wav) / SR_IN * 1000.0
            # 目标帧中心（与 mel 的 hop512 网格一致）
            tgt_t = (np.arange(T) + 0.5) * FRAME_MS
            src_t = (np.arange(len(f0)) + 0.5) * (1000.0 / 100.0)
            out = np.interp(tgt_t, src_t, f0, left=0.0, right=0.0).astype(np.float16)
            np.save(out_dir / f"{utt}.npy", out)
            done += 1
            if k % 50 == 0 or k == len(files):
                print(f"[{k}/{len(files)}] {utt} T={T} voiced={float(np.mean(out>0)):.2f} dur={dur_ms/1000:.2f}s", flush=True)
        except Exception as e:
            print(f"ERR {p.name}: {e}", flush=True)
    print(f"done={done} -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
