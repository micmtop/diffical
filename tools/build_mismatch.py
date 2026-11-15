"""构造"内容不匹配"的片段集合：每个音素的片段取自**同一歌手另一首歌**的同一音素。

目标仍是原句的真实 mel——用来检验模型能否把外来片段装配成目标。
"""
from __future__ import annotations
import argparse, pathlib, random, sys
import collections
import numpy as np, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from canvas import mel_of, spans_frames  # noqa: E402
from opencpop import parse_file  # noqa: E402
import pyworld as pw  # noqa: E402

SR, HOP = 44100, 512
FRAME_MS = HOP / SR * 1000.0
KW = 4


def f0_of(wav, T):
    f0, _ = pw.harvest(wav.astype(np.float64), SR, f0_floor=60.0, f0_ceil=1200.0, frame_period=FRAME_MS)
    xs = np.arange(len(f0), dtype=np.float64)
    xt = np.linspace(0, len(f0) - 1, T) if len(f0) > 1 else np.zeros(T)
    return np.interp(xt, xs, f0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utts", default="train/data/set_v1/eval")
    ap.add_argument("--split", default="")
    ap.add_argument("--out", default="train/data/mismatch_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()
    utts = {u.utt_id: u for u in parse_file("data/opencpop/transcriptions.txt")}
    # 音素→(utt, a, b) 索引
    bank: dict[str, list] = {}
    for uid, u in utts.items():
        if not (pathlib.Path("data/opencpop/wavs") / f"{uid}.wav").exists():
            continue
        sp = spans_frames(u.ph_dur)
        for ph, (a, b) in zip(u.phones, sp):
            bank.setdefault(ph, []).append((uid, a, b))
    rng = random.Random(args.seed)
    # donor mel LRU 缓存：避免每个音素都重新读 wav
    cache: collections.OrderedDict = collections.OrderedDict()
    CACHE_MAX = 2000

    def donor_mel(uid):
        if uid in cache:
            cache.move_to_end(uid)
            return cache[uid]
        w, _ = sf.read(f"data/opencpop/wavs/{uid}.wav", dtype="float32")
        m = mel_of(w)
        cache[uid] = m
        if len(cache) > CACHE_MAX:
            cache.popitem(last=False)
        return m
    out = pathlib.Path(args.out)
    if args.split:
        out = out / args.split
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(pathlib.Path(args.utts).glob("*.npz"))[args.shard::args.shards]
    for npz in files:
        uid = npz.stem
        u = utts[uid]
        wav, sr = sf.read(f"data/opencpop/wavs/{uid}.wav", dtype="float32")
        target = mel_of(wav); T = target.shape[1]
        sp = spans_frames(u.ph_dur)
        segs, lens = [], []
        phone_to_seg = {}
        n_fallback = 0
        for pi, (ph, (a, b)) in enumerate(zip(u.phones, sp)):
            cands = [c for c in bank.get(ph, []) if c[0] != uid]
            if not cands:
                n_fallback += 1
                continue
            d_uid, da, db = cands[rng.randrange(len(cands))]
            dm = donor_mel(d_uid)
            da, db = max(0, min(dm.shape[1], da)), max(0, min(dm.shape[1], db))
            if db - da < 2:
                continue
            phone_to_seg[pi] = len(segs)
            segs.append(dm[:, da:db]); lens.append(db - da)
        S = len(segs); Lmax = max(lens)
        arr = np.zeros((S, 128, Lmax), np.float16)
        for i, s in enumerate(segs):
            arr[i, :, : s.shape[1]] = s.astype(np.float16)
        seg_idx = np.zeros(T, np.int16); prog = np.zeros(T, np.float16); bd = np.zeros(T, np.uint8)
        for i, (a, b) in enumerate(sp):
            a, b = max(0, min(T, a)), max(0, min(T, b))
            if b <= a or i not in phone_to_seg: continue
            seg_idx[a:b] = phone_to_seg[i] + 1
            prog[a:b] = np.linspace(0, 1, b - a)
        for a, _ in sp[1:]:
            bd[max(0, a - KW):min(T, a + KW + 1)] = 1
        np.savez_compressed(out / f"{uid}.npz", segs=arr,
                            seg_len=np.array(lens, np.int16), seg_idx=seg_idx, progress=prog,
                            boundary=bd, f0=f0_of(wav, T), target=target.astype(np.float16),
                            utt=np.array(uid), n_fallback=np.array(n_fallback),
                            phone_seg=np.array([phone_to_seg.get(i, -1) + 1 for i in range(len(sp))], np.int16),
                            durations=np.array([max(0, min(T, b) - max(0, min(T, a))) for a, b in sp], np.int16))
        print(f"{uid}: segs={S} fallback={n_fallback} Lmax={Lmax}", flush=True)
    print("->", out)


if __name__ == "__main__":
    main()
