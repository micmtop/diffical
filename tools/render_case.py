"""用真实案例 JSON 渲染（可切 legacy / model 引擎），用于对比试听。"""
from __future__ import annotations
import argparse, json, os, sys, pathlib
import numpy as np, soundfile as sf
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DIFFICAL_CONFIG", str(ROOT / "docs/config.yaml"))
from backend.server import _parse_request
from backend.renderer import render_phrase

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="legacy", choices=["legacy", "model"])
    args = ap.parse_args()
    os.environ["DIFFICAL_ENGINE"] = "" if args.engine == "legacy" else "model"
    data = json.load(open(args.case, encoding="utf-8"))
    req = _parse_request(data)
    res = render_phrase(req)
    wav = np.asarray(res.samples, dtype=np.float32)
    sr = int(getattr(res, "sample_rate", 44100) or 44100)
    sf.write(args.out, wav, sr)
    print(f"{args.engine}: {len(wav)} samples ({len(wav)/sr:.2f}s) -> {args.out}")

if __name__ == "__main__":
    main()
