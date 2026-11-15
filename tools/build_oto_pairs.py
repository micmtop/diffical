"""oto 式 overlap 缺陷训练对：模拟坏 oto 造成的片段内容错位/截断。

缺陷类型（目标仍是原录音、片段仍取自目标自身）：
  shift      源窗起点 ±2..6 帧（oto offset 偏早/偏晚）
  trunc_end  尾部被裁 10–25%（cutoff 太短，元音尾巴缺失）——叠接处缺内容
  trunc_start 头部被裁 1–3 帧（offset 偏晚，辅音被切）
  long_con   起点提前 2–6 帧且尾部略裁（长辅音占窗）

构造与生产 segnorm 一致：先拉伸到 span+tail，再裁到实际使用的 span。
"""
from __future__ import annotations
import argparse, pathlib, random, sys
import numpy as np, soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from canvas import mel_of, spans_frames
from opencpop import parse_file
import pyworld as pw

SR, HOP = 44100, 512
FRAME_MS = HOP / SR * 1000.0
KW = 4


def f0_of(wav, T):
    f0, _ = pw.harvest(wav.astype(np.float64), SR, f0_floor=60.0, f0_ceil=1200.0,
                       frame_period=FRAME_MS)
    xs = np.arange(len(f0), dtype=np.float64)
    xt = np.linspace(0, len(f0) - 1, T) if len(f0) > 1 else np.zeros(T)
    out = np.interp(xt, xs, f0).astype(np.float32)
    out[(out > 0) & (out < 55.0)] = 0.0
    out[out > 1100.0] = 0.0
    return out


def _resample(seg, n):
    if seg.shape[1] == n:
        return seg
    xs = np.linspace(0, seg.shape[1] - 1, seg.shape[1])
    xt = np.linspace(0, seg.shape[1] - 1, n)
    return np.stack([np.interp(xt, xs, seg[b]) for b in range(seg.shape[0])]).astype(np.float32)


def build_one(u, rng, defect_frac=0.4):
    wav, sr = sf.read(f"data/opencpop/wavs/{u.utt_id}.wav", dtype="float32")
    assert sr == SR
    target = mel_of(wav); T = target.shape[1]
    spans = spans_frames(u.ph_dur)
    segs, lens, phone_seg = [], [], {}
    use_defect = rng.random() < defect_frac
    tilt = (np.arange(128, dtype=np.float32) - 63.5) / 63.5
    for i, (a, b) in enumerate(spans):
        a, b = max(0, min(T, a)), max(0, min(T, b))
        span = b - a
        if span < 2:
            continue
        if use_defect:
            kind = rng.choices(["shift", "trunc_end", "trunc_start", "long_con"],
                               [0.35, 0.30, 0.15, 0.20])[0]
            ratio = rng.uniform(0.75, 1.30)
            d = rng.randint(-6, 6) if kind == "shift" else rng.randint(-2, 2)
        else:
            kind = "clean"
            ratio = rng.uniform(0.85, 1.15)
            d = rng.randint(-2, 2)
        tail = rng.randint(0, 12) if use_defect else rng.randint(0, 6)
        nat = max(8, int(round(span * ratio)))
        s = max(0, min(T - 8, a + d)); e = max(s + 8, min(T, s + nat))
        seg = target[:, s:e].copy()
        if kind == "trunc_end" and seg.shape[1] > 12:
            c = int(round(seg.shape[1] * rng.uniform(0.10, 0.25)))
            seg = seg[:, : seg.shape[1] - c]
        elif kind == "trunc_start" and seg.shape[1] > 10:
            c = rng.randint(1, 3)
            seg = seg[:, c:]
        elif kind == "long_con" and seg.shape[1] > 12:
            s2 = max(0, s - rng.randint(2, 6))
            seg = target[:, s2: s2 + nat].copy()
            cut = rng.randint(1, 3)
            seg = seg[:, : max(8, seg.shape[1] - cut)]
        seg = _resample(seg, span + tail)[:, :span]
        if use_defect:
            seg = seg + np.float32(rng.uniform(-3.0, 3.0) / 8.686)
            seg = seg + tilt[:, None] * np.float32(rng.uniform(-0.25, 0.25))
        else:
            seg = seg + np.float32(rng.uniform(-2.0, 2.0) / 8.686)
        phone_seg[i] = len(segs)
        segs.append(seg.astype(np.float16)); lens.append(span)
    S = len(segs); Lmax = max(lens)
    arr = np.zeros((S, 128, Lmax), np.float16)
    for i, s in enumerate(segs):
        arr[i, :, : s.shape[1]] = s
    seg_idx = np.zeros(T, np.int16); prog = np.zeros(T, np.float16); bd = np.zeros(T, np.uint8)
    for i, (a, b) in enumerate(spans):
        a, b = max(0, min(T, a)), max(0, min(T, b))
        if b - a < 2 or i not in phone_seg:
            continue
        seg_idx[a:b] = phone_seg[i] + 1
        prog[a:b] = np.linspace(0, 1, b - a)
    for a, _ in spans[1:]:
        bd[max(0, a - KW): min(T, a + KW + 1)] = 1
    return dict(segs=arr, seg_len=np.array(lens, np.int16), seg_idx=seg_idx, progress=prog,
                boundary=bd, f0=f0_of(wav, T), target=target.astype(np.float16),
                utt=np.array(u.utt_id),
                phone_seg=np.array([phone_seg.get(i, -1) + 1 for i in range(len(spans))], np.int16),
                durations=np.array([max(0, min(T, b) - max(0, min(T, a))) for a, b in spans], np.int16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utts", required=True)
    ap.add_argument("--out", default="train/data/oto_v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--defect-frac", type=float, default=0.4)
    args = ap.parse_args()
    out = pathlib.Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(pathlib.Path(args.utts).glob("*.npz"))[args.shard:: args.shards]
    utts = {u.utt_id: u for u in parse_file("data/opencpop/transcriptions.txt")}
    n = 0
    for f in files:
        uid = f.stem
        if uid not in utts:
            continue
        rng = random.Random(args.seed * 100003 + hash(uid) % 100000)
        try:
            s = build_one(utts[uid], rng, args.defect_frac)
        except Exception as e:
            print("SKIP", uid, str(e)[:80]); continue
        np.savez_compressed(out / f"{uid}.npz", **s); n += 1
    print(f"{args.split}: {n} files -> {out}")


if __name__ == "__main__":
    main()
