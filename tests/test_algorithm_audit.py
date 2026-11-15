"""算法不变量审计：逐条判定"自然拼接"算法是否自洽。

每条都输出 PASS/FAIL 与数值，全部用同一批固定输入，确定性可复现。
"""
import pathlib, sys, random
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from model.pointer import PointerGen, gather_seg, load_pointer
from train.dataset_set import SetDataset, collate
from canvas import mel_of, spans_frames
import soundfile as sf

RESULTS = []
def check(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def masked_l1(x, t, valid):
    v = valid.unsqueeze(1)
    return float(((x - t).abs() * v).sum() / (v.sum().clamp_min(1.0) * t.shape[1]))


def audit_identity():
    """A. 零瑕疵时，装配必须等于目标（内容不可被破坏）。"""
    ds = SetDataset(ROOT / "train/data/set_v4", "eval")
    b = collate([ds[i] for i in range(4)])
    g = gather_seg(b["segs"], b["seg_len"], b["seg_idx"], b["progress"])
    l1 = masked_l1(g, b["target"], b["valid"])
    check("A 零瑕疵装配=目标", l1 < 2e-3, f"l1={l1:.6f}")


def audit_reachability():
    """B. 目标可达性：输入画布到目标的距离应远小于"内容替换"数据集。"""
    ds = SetDataset(ROOT / "train/data/seam_v1", "eval") if (ROOT/"train/data/seam_v1/eval").exists() else None
    if ds is None:
        check("B 目标可达性", False, "seam_v1 尚未构建，无法判定")
        return
    b = collate([ds[i] for i in range(4)])
    g = gather_seg(b["segs"], b["seg_len"], b["seg_idx"], b["progress"])
    l1_seam = masked_l1(g, b["target"], b["valid"])
    ds2 = SetDataset(ROOT / "train/data/mm_v4", "eval")
    b2 = collate([ds2[i] for i in range(4)])
    g2 = gather_seg(b2["segs"], b2["seg_len"], b2["seg_idx"], b2["progress"])
    l1_mm = masked_l1(g2, b2["target"], b2["valid"])
    check("B 目标可达性", l1_seam < 0.5 * l1_mm, f"seam={l1_seam:.3f} vs mismatch={l1_mm:.3f}")


def audit_content():
    """C. 内容保真：模型输出动态不得低于输入 80%。"""
    ck = ROOT / "train/ckpt_cp/last.pt"
    if not ck.exists():
        check("C 内容保真", False, "ckpt 缺失"); return
    ds = SetDataset(ROOT / "train/data/mm_v4", "eval")
    b = collate([ds[i] for i in range(8)])
    m = load_pointer(ck)
    with torch.no_grad():
        pred = m(b["segs"], b["seg_len"], b["seg_idx"], b["progress"], b["boundary"], b["f0"])
        base = gather_seg(b["segs"], b["seg_len"], b["seg_idx"], b["progress"])
    v = b["valid"].unsqueeze(1)[:, :, 1:] * b["valid"].unsqueeze(1)[:, :, :-1]
    dp = ((pred[:, :, 1:] - pred[:, :, :-1]).abs() * v).sum() / (v.sum().clamp_min(1.0) * pred.shape[1])
    db = ((base[:, :, 1:] - base[:, :, :-1]).abs() * v).sum() / (v.sum().clamp_min(1.0) * base.shape[1])
    keep = float(dp / db)
    check("C 内容保真≥0.8", keep >= 0.8, f"keep={keep:.3f}")


def audit_seam():
    """D. 接缝：边界跳变必须 ≤ 硬切的 50%（目标 ≤ 交叉淡化 1.2 倍）。"""
    ck = ROOT / "train/ckpt_cp/last.pt"
    if not ck.exists():
        check("D 接缝质量", False, "ckpt 缺失"); return
    bank = pathlib.Path.home() / "Library/OpenUtau/Singers/KAFU Cevio/C5"
    segs = []
    for n in ["a.wav", "i.wav", "u.wav"]:
        w, sr = sf.read(str(bank / n), dtype="float32")
        if w.ndim > 1: w = w.mean(1)
        segs.append(mel_of(w))
    F = 45; S = len(segs); T = S * F
    Lmax = max(s.shape[1] for s in segs)
    arr = np.zeros((S, 128, Lmax), np.float32)
    for i, s in enumerate(segs): arr[i, :, :s.shape[1]] = s
    lens = np.array([s.shape[1] for s in segs])
    idx = np.zeros((1, T), np.int64); prog = np.zeros((1, T), np.float32)
    for i in range(S):
        idx[0, i*F:(i+1)*F] = i + 1; prog[0, i*F:(i+1)*F] = np.linspace(0, 1, F)
    bd = np.zeros((1, T), np.float32)
    for i in range(1, S): bd[0, i*F-2:i*F+3] = 1
    f0 = np.full((1, T), 523.0, np.float32)
    m = load_pointer(ck)
    with torch.no_grad():
        pred = m(torch.from_numpy(arr).unsqueeze(0), torch.from_numpy(lens).unsqueeze(0),
                 torch.from_numpy(idx), torch.from_numpy(prog), torch.from_numpy(bd),
                 torch.from_numpy(f0))[0].numpy()
        hard = gather_seg(torch.from_numpy(arr).unsqueeze(0), torch.from_numpy(lens).unsqueeze(0),
                          torch.from_numpy(idx), torch.from_numpy(prog))[0].numpy()
    j = lambda x: float(np.mean([np.abs(x[:, i*F] - x[:, i*F-1]).mean() for i in range(1, S)]))
    ratio = j(pred) / j(hard)
    check("D 接缝≤硬切50%", ratio <= 0.5, f"ratio={ratio:.3f} (pred={j(pred):.3f} hard={j(hard):.3f})")


def audit_alignment():
    """E. 数据对齐：帧归属与音素跨度一致（phone_seg 精确校验）。"""
    root = ROOT / "train/data/mm_v4/train"
    bad = 0; checked = 0
    for p in sorted(root.glob("*.npz"))[:60]:
        z = np.load(p, allow_pickle=True)
        if not {"phone_seg", "durations", "seg_idx"}.issubset(z.files):
            bad += 1; continue
        si = z["seg_idx"].astype(np.int64); ps = z["phone_seg"].astype(np.int64)
        d = z["durations"].astype(np.int64)
        t = 0; ok = True
        for i, di in enumerate(d):
            if di <= 0: continue
            span = si[t:t+di]; want = int(ps[i])
            if (want == 0 and (span != 0).any()) or (want > 0 and not (span == want).all()):
                ok = False; break
            t += int(di)
        checked += 1
        if not ok: bad += 1
    check("E 帧归属对齐", bad == 0, f"checked={checked} bad={bad}")


def audit_assembly_norm():
    """F. 交叉淡化归一：自身内容重叠淡化后必须还原（否则淡化本身引入失真）。"""
    rng = random.Random(0)
    T = 200; seg = np.log(np.abs(np.random.RandomState(0).randn(128, T)) + 1.0).astype(np.float32)
    F = 50; ov = 7; S = 4
    pieces = [seg[:, i*F:(i+1)*F] for i in range(S)]
    mag = np.zeros((128, S*F)); wsum = np.zeros(S*F)
    w = 0.5 - 0.5*np.cos(np.pi*np.linspace(0, 1, ov+2)[1:-1])
    for i, s in enumerate(pieces):
        a = i*F; L = min(F+ov, s.shape[1]); x = s[:, :L]
        wt = np.ones(x.shape[1])
        if i > 0: k = min(ov, wt.size); wt[:k] = w[:k]
        if i < S-1: k = min(ov, wt.size); wt[-k:] = 1 - w[:k]
        hi = min(S*F, a+x.shape[1]); wt = wt[:hi-a]; x = x[:, :hi-a]
        mag[:, a:hi] += np.exp(x)*wt[None, :]; wsum[a:hi] += wt
    out = np.log(np.maximum(mag/np.maximum(wsum, 1e-6)[None, :], 1e-9))
    err = float(np.abs(out - np.concatenate(pieces, axis=1)).max())
    check("F 淡化归一还原", err < 1e-4, f"max_err={err:.2e}, wsum范围=[{wsum.min():.3f},{wsum.max():.3f}]")


if __name__ == "__main__":
    audit_identity(); audit_reachability(); audit_content(); audit_seam(); audit_alignment(); audit_assembly_norm()
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n=== {n_ok}/{len(RESULTS)} PASS ===")
    sys.exit(0 if n_ok == len(RESULTS) else 1)
