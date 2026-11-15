"""用 RMVPE 批量提取 F0（在算力机跑），输出 {相对路径: f0(float16)} 的 npz。

f0 帧率与 hop512/44.1k 对齐（11.6ms/帧），并在提取后做 <55Hz→0、>1100Hz→0 清理。
"""
from __future__ import annotations
import argparse, pathlib
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

SR, SR16, HOP = 44100, 16000, 512
FRAME_MS = HOP / SR * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="D:/diffical_neo/rmvpe.onnx")
    ap.add_argument("--out", default="gtsinger_f0.npz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from rmvpe_onnx import RMVPE
    model = RMVPE(model_path=args.model, device=args.device)
    root = pathlib.Path(args.root)
    wavs = sorted(root.rglob("*.wav"))
    print(f"wavs={len(wavs)} device={args.device}", flush=True)
    out = {}
    for k, p in enumerate(wavs, 1):
        try:
            w, sr = sf.read(str(p), dtype="float32")
            if w.ndim > 1: w = w.mean(1)
            if sr != SR:
                w = resample_poly(w.astype(np.float64), SR, sr).astype(np.float32)
            w16 = resample_poly(w.astype(np.float64), 160, 441).astype(np.float32)
            times, f0_raw = model.predict(w16, SR16)[:2]   # RMVPE: (times, f0, ...)
            T = int(len(w) * 1000 / FRAME_MS)
            xt = np.arange(T) * (FRAME_MS / 1000.0)        # 目标帧时间（秒）
            if len(times) > 1:
                f0 = np.interp(xt, np.asarray(times, np.float64), np.asarray(f0_raw, np.float64)).astype(np.float32)
            else:
                f0 = np.zeros(T, np.float32)
            f0[(f0 > 0) & (f0 < 55.0)] = 0.0
            f0[f0 > 1100.0] = 0.0
            out[str(p.relative_to(root))] = f0.astype(np.float16)
        except Exception as e:
            print("SKIP", p.name, str(e)[:60], flush=True)
        if k % 200 == 0:
            print(f"{k}/{len(wavs)}", flush=True)
    np.savez_compressed(args.out, **out)
    print(f"saved {len(out)} entries -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
