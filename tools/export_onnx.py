"""把 PointerGen 导出为 ONNX，并做数值一致性/延迟校验。

用法：
  python tools/export_onnx.py --ckpt train/ckpt_fix/best.pt --out export/pointer.onnx

产出：
  - ONNX 模型（动态 S/L/T，opset 18）
  - JSON manifest：ckpt 路径、sha256、输入/输出名、动态轴、mel 契约、导出时 torch 版本
校验：
  - 三组不同 S/T/L 的随机输入，ONNX vs torch 最大绝对误差 < 1e-4
  - CPU 延迟（B=1, T=500）供 Rust 侧参考
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.pointer_model import load_pointer  # noqa: E402

DYNAMIC = {"segs": {1: "S", 3: "L"}, "seg_len": {1: "S"}, "seg_idx": {1: "T"},
           "progress": {1: "T"}, "boundary": {1: "T"}, "f0": {1: "T"}, "mel": {2: "T"}}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def make_case(B=1, S=4, T=32, L=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        segs=torch.randn(B, S, 128, L, generator=g),
        seg_len=torch.randint(L // 2, L + 1, (B, S), generator=g),
        seg_idx=torch.randint(1, S + 1, (B, T), generator=g),
        progress=torch.rand(B, T, generator=g),
        boundary=(torch.rand(B, T, generator=g) > 0.8).float(),
        f0=torch.rand(B, T, generator=g) * 300,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="train/ckpt_fix/best.pt")
    ap.add_argument("--out", default="export/pointer.onnx")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    model = load_pointer(str(ROOT / args.ckpt)); model.eval()
    case = make_case()
    with torch.no_grad():
        ref = model(**case)
    torch.onnx.export(
        model, tuple(case.values()), str(out), opset_version=args.opset,
        input_names=list(case), output_names=["mel"], dynamic_axes=DYNAMIC,
        do_constant_folding=True,
    )

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    worst = 0.0
    for S, T, L in ((3, 17, 8), (6, 64, 32), (10, 200, 48)):
        c = make_case(S=S, T=T, L=L, seed=S + T)
        with torch.no_grad():
            r = model(**c).numpy()
        o = sess.run(None, {k: v.numpy() for k, v in c.items()})[0]
        worst = max(worst, float(np.abs(o - r).max()))
    assert worst < 1e-4, f"ONNX parity failed: {worst}"

    lat = make_case(S=8, T=500, L=64, seed=99)
    feeds = {k: v.numpy() for k, v in lat.items()}
    t0 = __import__("time").perf_counter()
    for _ in range(5):
        sess.run(None, feeds)
    latency_ms = (__import__("time").perf_counter() - t0) / 5 * 1000

    manifest = {
        "model": "PointerGen", "ckpt": args.ckpt, "ckpt_sha256": sha256(ROOT / args.ckpt),
        "onnx_sha256": sha256(out), "opset": args.opset,
        "inputs": list(case), "output": "mel",
        "dynamic_axes": DYNAMIC,
        "mel": {"n_mels": 128, "hop_size": 512, "sample_rate": 44100,
                "fmin": 40.0, "fmax": 16000.0, "torch_version": torch.__version__},
        "parity_max_abs_diff": worst,
        "cpu_latency_ms_B1_T500": round(latency_ms, 2),
    }
    (out.with_suffix(".json")).write_text(json.dumps(manifest, indent=2))
    print(f"exported {out} ({out.stat().st_size/1e6:.1f} MB)")
    print(f"parity_max_abs_diff={worst:.2e}  cpu_latency_B1_T500={latency_ms:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
