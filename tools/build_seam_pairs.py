"""拼接任务的训练对：内容=目标自己的切片，瑕疵只注入在接缝。

每个样本：
  segs      片段 = 目标 mel 在音素跨度内的切片（内容保真）
  target    目标 mel = 原录音
  seg_idx / progress / boundary / f0 / durations 作为条件
接缝瑕疵（模型要修的对象）：
  - 段边界处随机 ±1~2 帧的时间偏移（内容重采样到跨度）
  - 相邻段随机增益跳变 ±2 dB
  - 可选轻微谱倾斜（模拟 take 差异）
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys
import numpy as np, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from canvas import mel_of, spans_frames
from opencpop import parse_file
import pyworld as pw

SR, HOP = 44100, 512
FRAME_MS = HOP / SR * 1000.0
KW = 4


def f0_of(wav, T):
    f0, _ = pw.harvest(wav.astype(np.float64), SR, f0_floor=60.0, f0_ceil=1200.0, frame_period=FRAME_MS)
    xs = np.arange(len(f0), dtype=np.float64)
    xt = np.linspace(0, len(f0) - 1, T) if len(f0) > 1 else np.zeros(T)
    out = np.interp(xt, xs, f0).astype(np.float32)
    # F0_CLEAN: 低于 55Hz 或高于 1100Hz 视作未发声（去掉提取噪声）
    out[(out > 0) & (out < 55.0)] = 0.0
    out[out > 1100.0] = 0.0
    return out


def _resample(seg, n):
    if seg.shape[1] == n:
        return seg
    xs = np.linspace(0, seg.shape[1] - 1, seg.shape[1])
    xt = np.linspace(0, seg.shape[1] - 1, n)
    return np.stack([np.interp(xt, xs, seg[b]) for b in range(seg.shape[0])]).astype(np.float32)


def build_one(u, rng):
    wav, sr = sf.read(f"data/opencpop/wavs/{u.utt_id}.wav", dtype="float32")
    assert sr == SR
    target = mel_of(wav); T = target.shape[1]
    spans = spans_frames(u.ph_dur)
    segs, lens, phone_seg = [], [], {}
    for i, (a, b) in enumerate(spans):
        a, b = max(0, min(T, a)), max(0, min(T, b))
        if b - a < 2: continue
        # 生产瑕疵 1：样本自然长度 != 音符长度 → 被拉伸（0.85~1.15）
        # 生产瑕疵 2：oto 偏移误差 ±2 帧
        # 生产瑕疵 3：电平跳变 ±2 dB（每段独立，不累乘）
        ratio = rng.uniform(0.85, 1.15)
        d = rng.choice([-2, -1, 0, 0, 1, 2])
        nat = max(4, int(round((b - a) * ratio)))     # 样本"自然"长度
        s = max(0, min(T - 2, a + d)); e = max(s + 4, min(T, s + nat))
        seg = target[:, s:e]
        seg = _resample(seg, b - a)                  # 拉伸到音符长度
        seg = seg + np.float32(rng.uniform(-2.0, 2.0) / 8.686)  # ±2 dB，独立
        phone_seg[i] = len(segs)
        segs.append(seg.astype(np.float16)); lens.append(b - a)
    S = len(segs); Lmax = max(lens)
    arr = np.zeros((S, 128, Lmax), np.float16)
    for i, s in enumerate(segs): arr[i, :, : s.shape[1]] = s
    seg_idx = np.zeros(T, np.int16); prog = np.zeros(T, np.float16); bd = np.zeros(T, np.uint8)
    for i, (a, b) in enumerate(spans):
        a, b = max(0, min(T, a)), max(0, min(T, b))
        if b - a < 2 or i not in phone_seg: continue
        seg_idx[a:b] = phone_seg[i] + 1
        prog[a:b] = np.linspace(0, 1, b - a)
    for a, _ in spans[1:]:
        bd[max(0, a - KW):min(T, a + KW + 1)] = 1
    return dict(segs=arr, seg_len=np.array(lens, np.int16), seg_idx=seg_idx, progress=prog,
                boundary=bd, f0=f0_of(wav, T), target=target.astype(np.float16), utt=np.array(u.utt_id),
                phone_seg=np.array([phone_seg.get(i, -1) + 1 for i in range(len(spans))], np.int16),
                durations=np.array([max(0, min(T, b) - max(0, min(T, a))) for a, b in spans], np.int16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utts", required=True)
    ap.add_argument("--out", default="train/data/seam_v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = pathlib.Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(pathlib.Path(args.utts).glob("*.npz"))[args.shard::args.shards]
    utts = {u.utt_id: u for u in parse_file("data/opencpop/transcriptions.txt")}
    n = 0
    for f in files:
        uid = f.stem
        if uid not in utts: continue
        rng = random.Random(args.seed * 100003 + hash(uid) % 100000)
        try:
            s = build_one(utts[uid], rng)
        except Exception as e:
            print("SKIP", uid, str(e)[:80]); continue
        name = f"{uid}.npz" if args.seed == 0 else f"{uid}_s{args.seed}.npz"
        np.savez_compressed(out / name, **s); n += 1
    print(f"{args.split}: {n} files -> {out}")


if __name__ == "__main__":
    main()
