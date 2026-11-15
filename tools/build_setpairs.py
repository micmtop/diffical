"""构造"片段集合"数据集（B 架构：模型接管拼接）。

每个样本：
  segs     (S, 128, Lmax) float16  每个音素片段（目标录音在 [a,b) 的切片），右侧补零
  seg_len  (S,)          int16     每个片段的真实长度
  seg_idx  (T,)          int16     每帧所属片段（1 基，0=无）
  progress (T,)          float16   帧在该片段内的归一化位置
  boundary (T,)          uint8     边界 ±4 帧
  f0       (T,)          float32   目标 F0
  target   (128, T)      float16   真实 mel
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys
import numpy as np, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from canvas import mel_of, spans_frames  # noqa: E402
from opencpop import parse_file  # noqa: E402
import pyworld as pw  # noqa: E402

SR, HOP = 44100, 512
FRAME_MS = HOP / SR * 1000.0
KW = 4  # boundary 半宽


def f0_of(wav, T):
    f0, _ = pw.harvest(wav.astype(np.float64), SR, f0_floor=60.0, f0_ceil=1200.0, frame_period=FRAME_MS)
    xs = np.arange(len(f0), dtype=np.float64)
    xt = np.linspace(0, len(f0) - 1, T) if len(f0) > 1 else np.zeros(T)
    return np.interp(xt, xs, f0).astype(np.float32)


def build_one(u):
    wav, sr = sf.read(f"data/opencpop/wavs/{u.utt_id}.wav", dtype="float32")
    assert sr == SR
    target = mel_of(wav); T = target.shape[1]
    spans = spans_frames(u.ph_dur)
    S = len(spans); Lmax = max(b - a for a, b in spans)
    segs = np.zeros((S, 128, Lmax), np.float16)
    seg_len = np.zeros(S, np.int16)
    seg_idx = np.zeros(T, np.int16)
    progress = np.zeros(T, np.float16)
    boundary = np.zeros(T, np.uint8)
    phone_to_seg = {}
    for i, (a, b) in enumerate(spans):
        a, b = max(0, min(T, a)), max(0, min(T, b))
        if b <= a: continue
        phone_to_seg[i] = i
        segs[i, :, : b - a] = target[:, a:b].astype(np.float16)
        seg_len[i] = b - a
        seg_idx[a:b] = phone_to_seg[i] + 1
        progress[a:b] = np.linspace(0, 1, b - a)
    for a, _ in spans[1:]:
        boundary[max(0, a - KW):min(T, a + KW + 1)] = 1
    return dict(segs=segs, seg_len=seg_len, seg_idx=seg_idx, progress=progress,
                boundary=boundary, f0=f0_of(wav, T), target=target.astype(np.float16),
                utt=np.array(u.utt_id))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="train/data/set_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()
    utts = {u.utt_id: u for u in parse_file("data/opencpop/transcriptions.txt")}
    pool = [u for u in utts.values()
            if 3.0 <= u.total_s <= 8.0 and (pathlib.Path("data/opencpop/wavs") / f"{u.utt_id}.wav").exists()]
    rng = random.Random(args.seed); rng.shuffle(pool)
    picked = pool[: args.n][args.shard::args.shards]
    out = pathlib.Path(args.out)
    for split, part in (("train", picked[: int(len(picked)*0.9)]), ("eval", picked[int(len(picked)*0.9):])):
        d = out / split; d.mkdir(parents=True, exist_ok=True)
        for u in part:
            try:
                s = build_one(u)
            except Exception as e:
                print("SKIP", u.utt_id, e); continue
            np.savez_compressed(d / f"{u.utt_id}.npz", **s)
    print("built", len(picked), "->", out)


if __name__ == "__main__":
    main()
