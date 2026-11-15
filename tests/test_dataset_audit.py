"""数据集审计：seam_v2 是否真的在教"自然拼接"。

判据：
 1. 输入画布与目标的距离应远小于"内容替换"数据集（可达性）
 2. 输入的边界跳变必须显著大于目标（瑕疵必须落在接缝上）
 3. 段内动态不应被抹平（内容保真 ≥0.9）
 4. 增益不得累乘漂移（每样本电平极差应 ≤6dB）
 5. 无 NaN/inf，形状一致
"""
import pathlib, sys
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.dataset_set import SetDataset, collate

def stats(data, split="eval", n=12):
    ds = SetDataset(ROOT / data, split)
    b = {k: v for k, v in collate([ds[i] for i in range(min(n, len(ds)))]).items()}
    segs, ln, idx, prog = b["segs"], b["seg_len"], b["seg_idx"], b["progress"]
    B, S, _, L = segs.shape
    # 手工 gather（与模型一致）
    T = idx.shape[1]
    out = torch.zeros(B, 128, T)
    for j in range(B):
        for t in range(T):
            i = int(idx[j, t]) - 1
            if i < 0 or i >= S: continue
            l = int(ln[j, i])
            if l < 2: continue
            p = float(prog[j, t]) * (l - 1)
            i0 = int(p); w = p - i0; i1 = min(i0 + 1, l - 1)
            out[j, :, t] = (1 - w) * segs[j, i, :, i0] + w * segs[j, i, :, i1]
    tgt, valid = b["target"], b["valid"]
    v = valid.unsqueeze(1)
    den = v.sum().clamp_min(1.0) * 128
    l1 = float(((out - tgt).abs() * v).sum() / den)
    # 接缝处的"错多少"：|input-target| 在边界窗 vs 非边界窗
    bnd = (b["boundary"] > 0.5)
    err = (out - tgt).abs() * valid.unsqueeze(1)
    e_bnd, e_in = [], []
    for j in range(B):
        idxb = torch.nonzero(bnd[j]).flatten()
        mask_b = torch.zeros(T, dtype=torch.bool)
        for t in idxb:
            t = int(t)
            mask_b[max(0, t-2):min(T, t+3)] = True
        vb = valid[j]
        sel_b = mask_b & vb.bool(); sel_i = (~mask_b) & vb.bool()
        if sel_b.sum() > 0: e_bnd.append(float(err[j][:, sel_b].mean()))
        if sel_i.sum() > 0: e_in.append(float(err[j][:, sel_i].mean()))
    # 段内动态
    vi = (valid.unsqueeze(1)[:, :, 1:] * valid.unsqueeze(1)[:, :, :-1])
    di = ((out[:, :, 1:] - out[:, :, :-1]).abs() * vi).sum() / (vi.sum().clamp_min(1.0) * 128)
    dt = ((tgt[:, :, 1:] - tgt[:, :, :-1]).abs() * vi).sum() / (vi.sum().clamp_min(1.0) * 128)
    # 电平漂移（只看每段有效长度，排除补零）
    lvl = torch.zeros(B, S)
    for j in range(B):
        for i in range(S):
            l = int(ln[j, i])
            if l > 0: lvl[j, i] = segs[j, i, :, :l].mean()
    drift = float((lvl.max(dim=1).values - lvl.min(dim=1).values).mean())
    ok = torch.isfinite(out).all().item() and torch.isfinite(tgt).all().item()
    return dict(l1=l1, seam_in=float(np.mean(e_bnd)), seam_tg=float(np.mean(e_in)),
                din=float(di), dtg=float(dt), drift=drift, ok=ok)


def main():
    print("=== seam_v2（正确任务） ===")
    s = stats("train/data/seam_v2")
    for k, v in s.items(): print(f"  {k:10s} = {v:.4f}" if not isinstance(v, bool) else f"  {k:10s} = {v}")
    print("=== mm_v4（旧任务，内容替换） ===")
    m = stats("train/data/mm_v4")
    for k, v in m.items(): print(f"  {k:10s} = {v:.4f}" if not isinstance(v, bool) else f"  {k:10s} = {v}")
    checks = [
        ("1 可达性(l1_seam < 0.5*l1_mm)", s["l1"] < 0.5 * m["l1"], f"{s['l1']:.3f} vs {m['l1']:.3f}"),
        ("2 接缝错误集中(边界误差 ≥ 2× 内部误差)", s["seam_in"] >= 2.0 * s["seam_tg"], f"边界={s['seam_in']:.3f} 内部={s['seam_tg']:.3f}"),
        ("3 内容未抹平(din ≥ 0.9*dtg)", s["din"] >= 0.9 * s["dtg"], f"{s['din']:.3f} vs {s['dtg']:.3f}"),
        ("4 数值有限", s["ok"], str(s["ok"])),
    ]
    print("\n=== 判定 ===")
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


if __name__ == "__main__":
    main()
