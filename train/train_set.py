"""SetGen 训练：模型从片段集合直接生成整句 mel（无程序拼接）。"""
from __future__ import annotations
import argparse, pathlib, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.setgen import SetGen, count_params  # noqa: E402
from model.pointer import PointerGen  # noqa: E402
from train.dataset_set import SetDataset, collate  # noqa: E402


def _wm(x, valid):
    return x * valid


def interior_flux(x, valid):
    d = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(1)
    return (d * valid[:, 1:]).sum() / valid[:, 1:].sum().clamp_min(1.0)


def loss_fn(pred, target, boundary, valid, lambda_b=8.0, args_ls=0.0, lambda_tv=0.0):
    v = valid.unsqueeze(1)
    den = v.sum().clamp_min(1.0) * pred.shape[1]
    l1 = ((pred - target).abs() * v).sum() / den
    dp = pred[:, :, 1:] - pred[:, :, :-1]
    dt = target[:, :, 1:] - target[:, :, :-1]
    vm = valid[:, 1:].unsqueeze(1)
    ld = ((dp - dt).abs() * vm).sum() / (vm.sum().clamp_min(1.0) * pred.shape[1])
    # 只在边界帧计权，内部帧不重复计入（修复双重计数）
    lb = ((pred - target).abs() * boundary.unsqueeze(1) * v).sum() / den
    # 接缝跳变损失：直接惩罚边界帧的一阶差分误差
    _bm = boundary[:, 1:].unsqueeze(1) * (v[:, :, 1:] * v[:, :, :-1])
    _dp = pred[:, :, 1:] - pred[:, :, :-1]
    _dt = target[:, :, 1:] - target[:, :, :-1]
    lseam = ((_dp - _dt).abs() * _bm).sum() / _bm.sum().clamp_min(1.0)
    lf = (interior_flux(pred, valid) - interior_flux(target, valid)) ** 2
    # 边界「超额跳变」损失：输出在接缝处的帧间跳变不得超过目标跳变的 1.15 倍
    # （+0.05 容差）。直接压制接缝台阶，不惩罚目标本身的快速过渡。
    _excess = torch.relu(_dp.abs() - 1.15 * _dt.abs() - 0.05)
    ltv = (_excess * _bm).sum() / _bm.sum().clamp_min(1.0)
    return (l1 + 2.0 * ld + lambda_b * lb + args_ls * lseam + 1.0 * lf + lambda_tv * ltv,
            {"l1": l1.item(), "ld": ld.item(), "lf": lf.item(), "lseam": lseam.item(),
             "ltv": float(ltv)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="train/data/set_v1")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out", default="ckpt_set")
    ap.add_argument("--arch", default="attn", choices=["attn", "pointer"])
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--wave-lambda", type=float, default=0.0)
    ap.add_argument("--wave-frames", type=int, default=32)
    ap.add_argument("--wave-every", type=int, default=4)
    ap.add_argument("--wave-warmup", type=int, default=500)
    ap.add_argument("--voc-dir", default="voc_model")
    ap.add_argument("--init", default="")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--lambda-b", type=float, default=8.0)
    ap.add_argument("--lambda-seam", type=float, default=0.0)
    ap.add_argument("--lambda-tv", type=float, default=0.0)
    ap.add_argument("--adv-lambda", type=float, default=0.0)
    ap.add_argument("--r1", type=float, default=10.0)
    ap.add_argument("--r1-every", type=int, default=8)
    args = ap.parse_args()
    dev = torch.device(args.device)
    tr = SetDataset(ROOT / args.pairs, "train")
    ev = SetDataset(ROOT / args.pairs, "eval")
    if args.arch == "pointer":
        model = PointerGen(c=args.width).to(dev)
    else:
        model = SetGen().to(dev)
    print(f"params={count_params(model)/1e6:.3f}M arch={args.arch} device={args.device} train={len(tr)} eval={len(ev)}", flush=True)
    voc = mr = None
    if args.wave_lambda > 0:
        from train.wave_loss import MRSTFT, load_vocoder
        voc = load_vocoder(str(ROOT / args.voc_dir), dev)
        mr = MRSTFT().to(dev)
        print(f"wave loss ON lambda={args.wave_lambda} frames={args.wave_frames}", flush=True)
    if args.init:
        ck = torch.load(str(ROOT / args.init), map_location=dev)
        model.load_state_dict(ck["state_dict"])
        print(f"init from {args.init}", flush=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and dev.type == "cuda")
    disc = disc_opt = None
    if args.adv_lambda > 0:
        from model.discriminator import PatchDisc, hinge_d_loss, hinge_g_loss
        disc = PatchDisc().to(dev)
        disc_opt = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.5, 0.9))
        print(f"adversarial ON lambda={args.adv_lambda} r1={args.r1}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out / "tb"))
    except Exception as e:
        print(f"tb disabled: {e}", flush=True)
        writer = None
    it = 0
    t0 = time.time()
    while it < args.steps:
        perm = torch.randperm(len(tr)).tolist()
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            b = collate([tr[j] for j in perm[i:i + args.batch]])
            b = {k: v.to(dev) for k, v in b.items()}
            if args.arch == "pointer":
                with torch.cuda.amp.autocast(enabled=args.amp and dev.type == "cuda"):
                    pred = model(b["segs"], b["seg_len"], b["seg_idx"], b["progress"], b["boundary"], b["f0"])
            else:
                with torch.cuda.amp.autocast(enabled=args.amp and dev.type == "cuda"):
                    pred, _ = model(b["segs"], b["seg_mask"], b["seg_idx"], b["progress"], b["boundary"], b["f0"])
            loss, parts = loss_fn(pred, b["target"], b["boundary"], b["valid"], args.lambda_b, args.lambda_seam, args.lambda_tv)
            if disc is not None:
                real = b["target"].detach()
                fake = pred.detach()
                d_real = disc(real); d_fake = disc(fake)
                dloss = hinge_d_loss(d_real, d_fake)
                if (it % args.r1_every) == 0:
                    real_r = real.detach().requires_grad_(True)
                    r_out = disc(real_r)
                    g = torch.autograd.grad(r_out.sum(), real_r, create_graph=True)[0]
                    dloss = dloss + 0.5 * args.r1 * (g.flatten(1).pow(2).sum(1).mean())
                disc_opt.zero_grad(); dloss.backward(); disc_opt.step()
                with torch.no_grad():
                    adv = float(hinge_g_loss(disc(pred)))
                loss = loss + args.adv_lambda * hinge_g_loss(disc(pred))
                parts["d"] = float(dloss); parts["adv"] = adv
            # 内容保真：输出动态不得低于输入内容的 80%（防止抹平成鸣笛）
            from model.pointer import gather_seg as _gather
            with torch.no_grad():
                _base = _gather(b["segs"], b["seg_len"], b["seg_idx"], b["progress"])
            _vm = (b["valid"].unsqueeze(1)[:, :, 1:] * b["valid"].unsqueeze(1)[:, :, :-1])
            _dp = ((pred[:, :, 1:] - pred[:, :, :-1]).abs() * _vm).sum() / _vm.sum().clamp_min(1.0)
            _db = ((_base[:, :, 1:] - _base[:, :, :-1]).abs() * _vm).sum() / _vm.sum().clamp_min(1.0)
            _cp = torch.relu(0.8 * _db - _dp) ** 2
            loss = loss + 5.0 * _cp
            parts["cp"] = float(_cp)
            if voc is not None and it > args.wave_warmup and (it % args.wave_every == 0):
                L = min(args.wave_frames, pred.shape[2] // 2)
                wl = torch.zeros((), device=dev)
                nb = 0
                for j in range(pred.shape[0]):
                    Tj = int(b["valid"][j].sum())
                    if Tj < L + 1:
                        continue
                    s0 = int(torch.randint(0, Tj - L, (1,)).item())
                    mp = pred[j:j+1, :, s0:s0+L]
                    mt = b["target"][j:j+1, :, s0:s0+L]
                    f0c = b["f0"][j:j+1, s0:s0+L]
                    wp = voc(mp, f0c)
                    with torch.no_grad():
                        wt = voc(mt, f0c)
                    wl = wl + mr(wp, wt)
                    nb += 1
                if nb:
                    wl = wl / nb
                    loss = loss + args.wave_lambda * wl
                    parts["wave"] = float(wl)
                else:
                    parts["wave"] = 0.0
            opt.zero_grad()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            it += 1
            if it % args.log_every == 0 or it == 1:
                rate = it / max(1e-9, time.time() - t0)
                print(f"step {it:5d} l1={parts['l1']:.4f} ld={parts['ld']:.4f} lf={parts['lf']:.6f} {rate:.2f} it/s", flush=True)
                if writer is not None:
                    if "wave" in parts:
                        writer.add_scalar("loss/wave", parts["wave"], it)
                    writer.add_scalar("loss/l1", parts["l1"], it)
                    writer.add_scalar("loss/ld", parts["ld"], it)
                    writer.add_scalar("loss/lf", parts["lf"], it)
                    writer.add_scalar("loss/total", float(loss), it)
                    writer.add_scalar("lr", sched.get_last_lr()[0], it)
            if it >= args.steps:
                break
    # eval
    model.eval()
    with torch.no_grad():
        b = collate([ev[j] for j in range(len(ev))]) if len(ev) <= 4 else collate([ev[j] for j in range(4)])
        b = {k: v.to(dev) for k, v in b.items()}
        if args.arch == "pointer":
            pred = model(b["segs"], b["seg_len"], b["seg_idx"], b["progress"], b["boundary"], b["f0"])
        else:
            pred, _ = model(b["segs"], b["seg_mask"], b["seg_idx"], b["progress"], b["boundary"], b["f0"])
        vv = b["valid"].unsqueeze(1)
        l1 = (((pred - b["target"]).abs() * vv).sum() / (vv.sum().clamp_min(1.0) * pred.shape[1])).item()
        # 平凡基线：直接输出目标均值（衡量模型是否比常数更差）
        # 常数基线：只用有效帧求均值，再算偏差
        tgt = b["target"]
        vmask = b["valid"].unsqueeze(1)
        mu = (tgt * vmask).sum(dim=2, keepdim=True) / vmask.sum(dim=2, keepdim=True).clamp_min(1.0)
        const = (((tgt - mu).abs() * vmask).sum() / (vmask.sum().clamp_min(1.0) * tgt.shape[1])).item()
        print(f"eval l1={l1:.4f}  constant_baseline={const:.4f}", flush=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, out / "last.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
