"""生产式训练对：模拟推理时的片段几何与更狠的接缝瑕疵。

与 build_seam_pairs.py 的差异（针对 overlap 呛水）：
  1) 片段先拉伸到「跨度 + 尾巴」（生产里 mel 覆盖 preutterance/80ms tail），
     再裁到实际会被 seg_idx 使用的跨度（与推理 DIFFICAL_SEG_NORM 语义一致）；
  2) 偏移 ±4 帧（原 ±2）、拉伸 0.70–1.30（原 0.85–1.15）、电平 ±3dB（原 ±2）；
  3) 额外加轻微谱倾斜（模拟换 take/设备），幅度 ±0.25。

目标仍是原录音 mel；片段仍取自目标自己的切片（可达性铁律不破）。
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


def build_one(u, rng):
    wav, sr = sf.read(f"data/opencpop/wavs/{u.utt_id}.wav", dtype="float32")
    assert sr == SR
    target = mel_of(wav); T = target.shape[1]
    spans = spans_frames(u.ph_dur)
    segs, lens, phone_seg = [], [], {}
    tilt = (np.arange(128, dtype=np.float32) - 63.5) / 63.5
    for i, (a, b) in enumerate(spans):
        a, b = max(0, min(T, a)), max(0, min(T, b))
        span = b - a
        if span < 2:
            continue
        ratio = rng.uniform(0.70, 1.30)
        d = rng.choice([-4, -3, -2, -1, 0, 0, 1, 2, 3, 4])
        tail = rng.randint(0, 12)                       # 生产里 mel 超出跨度的尾巴
        nat = max(4, int(round(span * ratio)))
        s = max(0, min(T - 2, a + d)); e = max(s + 4, min(T, s + nat))
        seg = _resample(target[:, s:e], span + tail)     # 拉伸到跨度+尾巴
        seg = seg[:, :span]                              # 裁到实际使用跨度（推理语义）
        seg = seg + np.float32(rng.uniform(-3.0, 3.0) / 8.686)
        seg = seg + tilt[:, None] * np.float32(rng.uniform(-0.25, 0.25))
        phone_seg[i] = len(segs)
        segs.append(seg.astype(np.float16)); lens.append(span)
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
                boundary=bd, f0=f0_of(wav, T), target=target.astype(np.float16),
                utt=np.array(u.utt_id),
                phone_seg=np.array([phone_seg.get(i, -1) + 1 for i in range(len(spans))], np.int16),
                durations=np.array([max(0, min(T, b) - max(0, min(T, a))) for a, b in spans], np.int16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utts", required=True)
    ap.add_argument("--out", default="train/data/prod_v1")
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
