"""SetGen/PointerGen 评估：模型 vs 恒等指针基线 vs 常数基线（全部按有效帧掩码）。"""
from __future__ import annotations
import argparse, pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.pointer import PointerGen, gather_seg, count_params  # noqa: E402
from train.dataset_set import SetDataset, collate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--width", type=int, default=192)
    args = ap.parse_args()
    ds = SetDataset(pathlib.Path(args.data), args.split)
    n = min(args.n, len(ds))
    b = {k: v for k, v in collate([ds[i] for i in range(n)]).items()}
    dev = torch.device(args.device)
    b = {k: v.to(dev) for k, v in b.items()}
    ck = torch.load(args.ckpt, map_location=dev)
    model = PointerGen(c=args.width).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    with torch.no_grad():
        pred = model(b["segs"], b["seg_len"], b["seg_idx"], b["progress"], b["boundary"], b["f0"])
        base = gather_seg(b["segs"], b["seg_len"], b["seg_idx"], b["progress"])
    vm = b["valid"].unsqueeze(1)
    den = vm.sum().clamp_min(1.0) * b["target"].shape[1]
    def l1(x): return float((((x - b["target"]).abs() * vm).sum() / den))
    tgt = b["target"]
    mu = (tgt * vm).sum(dim=2, keepdim=True) / vm.sum(dim=2, keepdim=True).clamp_min(1.0)
    const = l1(mu.expand_as(tgt))
    print(f"params={count_params(model)/1e6:.3f}M n={n} device={args.device}")
    print(f"  constant      = {const:.4f}")
    print(f"  identity_gather = {l1(base):.4f}")
    print(f"  model           = {l1(pred):.4f}")
    print(f"  model/base      = {l1(pred)/max(1e-9, l1(base)):.3f}")
    def dmel(x):
        v = b["valid"].unsqueeze(1)[:, :, 1:] * b["valid"].unsqueeze(1)[:, :, :-1]
        d = (x[:, :, 1:] - x[:, :, :-1]).abs()
        return float((d * v).sum() / v.sum().clamp_min(1.0))
    dp, db, dt = dmel(pred), dmel(base), dmel(b["target"])
    print(f"  dmel pred={dp:.4f} base={db:.4f} target={dt:.4f}")
    print(f"  content_keep    = {dp/max(1e-9, db):.3f}  (输出Δmel/输入Δmel，目标≥0.8)")


if __name__ == "__main__":
    main()
