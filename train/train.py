"""P0.5 训练器：绝对生成整句 mel。

用法（本机 smoke）：
    python3 train/train.py --steps 50 --batch 2 --frames 256 --device mps

Colab：
    python3 train/train.py --steps 30000 --batch 8 --frames 256 --device cuda --out /content/drive/MyDrive/diffical_neo/ckpt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard 缺失时静默降级
    SummaryWriter = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.net import Gen, count_params  # noqa: E402
from train.dataset import PairDataset, build_vocab, collate  # noqa: E402

FRAME_MS = 512 / 44100 * 1000.0
BOUNDARY_WIN = max(1, int(round(100.0 / FRAME_MS)))


def interior_flux(x, boundary):
    d = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=1)      # (B,T-1)
    m = boundary[:, 1:] < 0.5
    return (d * m).sum() / m.sum().clamp_min(1.0)


def loss_fn(pred, target, boundary, lam_d=2.0, lam_b=0.5, lam_flux=1.0):
    l1 = F.l1_loss(pred, target)
    d_pred = pred[:, :, 1:] - pred[:, :, :-1]
    d_tgt = target[:, :, 1:] - target[:, :, :-1]
    ld = F.l1_loss(d_pred, d_tgt)
    lf = (interior_flux(pred, boundary) - interior_flux(target, boundary)) ** 2
    if boundary is not None and boundary.any():
        w = 1.0 + lam_b * boundary.unsqueeze(1)
        lb = (torch.abs(pred - target) * w).mean()
    else:
        lb = l1
    return (l1 + lam_d * ld + lam_b * lb + lam_flux * lf,
            {"l1": l1.item(), "delta": ld.item(), "bound": lb.item(), "flux": float(lf)})


def interior_delta(mel: torch.Tensor, boundary: torch.Tensor) -> float:
    """内部 |Δmel| 中位数（与 P0.3 同口径，但在这里用于训练监控）。"""
    d = torch.abs(mel[:, :, 1:] - mel[:, :, :-1]).mean(dim=1)  # (B,T-1)
    m = boundary[:, 1:] < 0.5
    v = d[m]
    return float(v.median()) if v.numel() else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--pairs", default="train/data/pairs")
    ap.add_argument("--out", default="train/ckpt")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-f0", action="store_true")
    ap.add_argument("--no-tb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    pairs = ROOT / args.pairs
    vocab = build_vocab(pairs)
    tr = PairDataset(pairs, "train", args.frames, vocab, seed=args.seed)
    ev = PairDataset(pairs, "eval", 10_000, vocab, seed=args.seed)  # eval 用整句
    print(f"vocab={len(vocab)} train={len(tr)} eval={len(ev)} device={dev}")

    model = Gen(use_f0=not args.no_f0).to(dev)
    n_params = count_params(model)
    print(f"params={n_params/1e6:.3f}M")
    assert 1.5e6 <= n_params <= 2.5e6, f"param count {n_params} outside 1.5–2.5M"

    cfg = vars(args).copy()
    cfg["use_f0"] = not args.no_f0
    cfg["n_params"] = n_params
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    print(f"cfg_hash={cfg_hash}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps), eta_min=args.lr * 0.1)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(out / "tb")) if (SummaryWriter and not args.no_tb) else None
    if tb is not None:
        print(f"tensorboard -> {out / 'tb'}", flush=True)

    # 固定 eval batch（喂整句 -> 取前 512 帧）
    ev_batch = collate([ev[i] for i in range(min(2, len(ev)))])
    ev_batch = {k: v[..., : min(512, v.shape[-1])].to(dev) for k, v in ev_batch.items()}

    t0 = time.time()
    model.train()
    running = []
    for step in range(1, args.steps + 1):
        idx = np.random.randint(0, len(tr), size=args.batch)
        batch = collate([tr[int(i)] for i in idx])
        batch = {k: v.to(dev) for k, v in batch.items()}
        pred = model(batch["source"], batch["f0"],
                     batch["boundary"], batch["progress"], batch["dist_bd"], batch["dur"])
        loss, parts = loss_fn(pred, batch["target"], batch["boundary"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(parts["l1"])
        if tb is not None:
            tb.add_scalar("train/l1", parts["l1"], step)
            tb.add_scalar("train/delta", parts["delta"], step)
            tb.add_scalar("train/bound", parts["bound"], step)
            tb.add_scalar("train/flux", parts["flux"], step)
            tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
        if step % args.log_every == 0:
            dt = time.time() - t0
            model.eval()
            with torch.no_grad():
                ep = model(ev_batch["source"], ev_batch["f0"],
                           ev_batch["boundary"], ev_batch["progress"], ev_batch["dist_bd"], ev_batch["dur"])
                ev_i = interior_delta(ep, ev_batch["boundary"])
                tg_i = interior_delta(ev_batch["target"], ev_batch["boundary"])
                src_i = interior_delta(ev_batch["source"], ev_batch["boundary"])
                if tb is not None:
                    tb.add_scalar("eval/interior", ev_i, step)
                    tb.add_scalar("eval/source_interior", src_i, step)
                    tb.add_scalar("eval/target_interior", tg_i, step)
                    tb.add_scalar("eval/interior_ratio", ev_i / tg_i if tg_i else 0.0, step)
            model.train()
            print(f"step {step:5d} l1={np.mean(running):8.4f} (last {parts['l1']:.4f}) flux={parts['flux']:.5f} "
                  f"eval_interior={ev_i:.4f} (src {src_i:.4f} / tgt {tg_i:.4f}) "
                  f"{args.log_every/(time.time()-t0):.2f} it/s", flush=True)
            running = []
            t0 = time.time()
            ckpt = {"state_dict": model.state_dict(), "cfg": cfg, "cfg_hash": cfg_hash,
                    "step": step, "opt": opt.state_dict()}
            torch.save(ckpt, out / "last.pt")
    if tb is not None:
        tb.flush()
        tb.close()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
