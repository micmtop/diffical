"""训练管线恒等性测试：数据映射 + 模型前向。

self 数据（segs=target 的切片）下：
  gather_seg(progress) 必须 == target
  零初始化 PointerGen 必须 == target
mismatch 数据下：
  seg_idx 必须是连续 1..S，且每个片段的 seg_len 与其有效宽度一致
"""
import pathlib, sys
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.pointer import PointerGen, gather_seg  # noqa: E402
from train.dataset_set import SetDataset, collate  # noqa: E402


def _batch(data, split, n=2):
    ds = SetDataset(ROOT / data, split)
    return ds, {k: v for k, v in collate([ds[i] for i in range(n)]).items()}


def _masked_l1(x, t, valid):
    v = valid.unsqueeze(1)
    den = v.sum().clamp_min(1.0) * t.shape[1]
    return float(((x - t).abs() * v).sum() / den)


def test_gather_equals_target_self():
    ds, b = _batch("train/data/set_v4", "eval")
    g = gather_seg(b["segs"], b["seg_len"], b["seg_idx"], b["progress"])
    l1 = _masked_l1(g, b["target"], b["valid"])
    print(f"[gather self] l1={l1:.6f}")
    assert l1 < 1e-3, f"self 数据下按进度取样应等于目标，实际 {l1:.6f}"


def test_pointer_identity_self():
    ds, b = _batch("train/data/set_v4", "eval")
    m = PointerGen(); m.eval()
    with torch.no_grad():
        out = m(b["segs"], b["seg_len"], b["seg_idx"], b["progress"], b["boundary"], b["f0"])
    l1 = _masked_l1(out, b["target"], b["valid"])
    print(f"[pointer init self] l1={l1:.6f}")
    assert l1 < 1e-3, f"零初始化 PointerGen 应等于目标，实际 {l1:.6f}"


def test_mismatch_alignment():
    """精确校验：对每个跨度非空的音素，seg_idx 在其帧跨度内必须等于 phone_seg。"""
    root = ROOT / "train/data/mm_v4/train"
    bad = []
    checked = 0
    for p in sorted(root.glob("*.npz"))[:80]:
        z = np.load(p, allow_pickle=True)
        need = {"phone_seg", "durations", "seg_idx"}
        if not need.issubset(z.files):
            bad.append((p.stem, f"missing {need - set(z.files)}")); continue
        seg_idx = z["seg_idx"].astype(np.int64)
        ps = z["phone_seg"].astype(np.int64)
        durs = z["durations"].astype(np.int64)
        t = 0
        for i, d in enumerate(durs):
            if d <= 0:
                continue
            span = seg_idx[t:t + d]
            want = int(ps[i])
            if want == 0:
                if (span != 0).any():
                    bad.append((p.stem, f"phone{i} has seg but phone_seg=0")); break
            elif not (span == want).all():
                bad.append((p.stem, f"phone{i} span mismatch", int(span[0]), want)); break
            t += int(d)
        else:
            checked += 1
        # 跨度总和不得超过 T
        if t > seg_idx.shape[0]:
            bad.append((p.stem, "durations exceed T"))
    print(f"[mismatch align] checked={checked} bad={len(bad)}")
    for b in bad[:5]: print("   ", b)
    assert not bad, f"{len(bad)} 个样本对齐异常"


if __name__ == "__main__":
    test_mismatch_alignment()
    test_gather_equals_target_self()
    test_pointer_identity_self()
    print("ALL OK")
