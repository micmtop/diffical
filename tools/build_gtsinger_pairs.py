"""把 多语语料 片段转成我们的接缝数据集格式。

多语语料：48kHz mono，每 clip 8s，JSON 含 word/ph（音素）/ph_start/ph_end（秒）/note(MIDI)。
输出与 build_seam_pairs 相同：segs/target/seg_idx/progress/boundary/f0/durations/phone_seg
按"歌曲目录"划分 train/eval（同歌不跨集）。
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys
import numpy as np, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from canvas import mel_of
import pyworld as pw

SR_IN, SR, HOP = 48000, 44100, 512
FRAME_MS = HOP / SR * 1000.0
KW = 4


def resample(x, sr_in, sr_out):
    if sr_in == sr_out: return x
    n = int(round(len(x) * sr_out / sr_in))
    xs = np.arange(len(x)); xt = np.linspace(0, len(x) - 1, n)
    return np.interp(xt, xs, x).astype(np.float32)


def f0_of(wav, T):
    """从音频提取真实 F0（含颤音/滑音），与生产一致：F0 是连续曲线，不是曲谱台阶。"""
    f0, _ = pw.harvest(wav.astype(np.float64), SR, f0_floor=60.0, f0_ceil=1200.0,
                       frame_period=FRAME_MS)
    if len(f0) == 0:
        return np.zeros(T, np.float32)
    xs = np.arange(len(f0), dtype=np.float64)
    xt = np.linspace(0, len(f0) - 1, T) if len(f0) > 1 else np.zeros(T)
    out = np.interp(xt, xs, f0).astype(np.float32)
    # F0_CLEAN: 低于 55Hz 或高于 1100Hz 视作未发声（去掉提取噪声）
    out[(out > 0) & (out < 55.0)] = 0.0
    out[out > 1100.0] = 0.0
    return out


def _resample_seg(seg, n):
    if seg.shape[1] == n: return seg
    xs = np.linspace(0, seg.shape[1] - 1, seg.shape[1]); xt = np.linspace(0, seg.shape[1] - 1, n)
    return np.stack([np.interp(xt, xs, seg[b]) for b in range(seg.shape[0])]).astype(np.float32)


def build_one(wav_path, json_path, rng):
    w, sr = sf.read(str(wav_path), dtype="float32")
    if w.ndim > 1: w = w.mean(1)
    w = resample(w, sr, SR)
    target = mel_of(w); T = target.shape[1]
    ann = json.load(open(json_path, encoding="utf-8"))
    spans, phones = [], []
    for e in ann:
        for ph, a, b in zip(e["ph"], e["ph_start"], e["ph_end"]):
            fa = int(round(a * 1000 / FRAME_MS)); fb = int(round(b * 1000 / FRAME_MS))
            if fb - fa >= 2: spans.append((fa, fb)); phones.append(ph)
    if len(spans) < 3: return None
    segs, lens, phone_seg = [], [], {}
    for i, (a, b) in enumerate(spans):
        a, b = max(0, min(T, a)), max(0, min(T, b))
        if b - a < 2: continue
        ratio = rng.uniform(0.85, 1.15)
        d = rng.choice([-2, -1, 0, 0, 1, 2])
        nat = max(4, int(round((b - a) * ratio)))
        s = max(0, min(T - 2, a + d)); e = max(s + 4, min(T, s + nat))
        seg = _resample_seg(target[:, s:e], b - a)
        seg = seg + np.float32(rng.uniform(-2.0, 2.0) / 8.686)
        phone_seg[i] = len(segs); segs.append(seg.astype(np.float16)); lens.append(b - a)
    S = len(segs)
    if S < 3: return None
    Lmax = max(lens)
    arr = np.zeros((S, 128, Lmax), np.float16)
    for i, s in enumerate(segs): arr[i, :, :s.shape[1]] = s
    seg_idx = np.zeros(T, np.int16); prog = np.zeros(T, np.float16); bd = np.zeros(T, np.uint8)
    for i, (a, b) in enumerate(spans):
        a, b = max(0, min(T, a)), max(0, min(T, b))
        if b - a < 2 or i not in phone_seg: continue
        seg_idx[a:b] = phone_seg[i] + 1; prog[a:b] = np.linspace(0, 1, b - a)
    for a, _ in spans[1:]:
        bd[max(0, a - KW):min(T, a + KW + 1)] = 1
    return dict(segs=arr, seg_len=np.array(lens, np.int16), seg_idx=seg_idx, progress=prog,
                boundary=bd, f0=f0_of(w, T), target=target.astype(np.float16),
                phone_seg=np.array([phone_seg.get(i, -1) + 1 for i in range(len(spans))], np.int16),
                durations=np.array([max(0, min(T, b) - max(0, min(T, a))) for a, b in spans], np.int16),
                utt=np.array(wav_path.stem))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/multilingual/Chinese")
    ap.add_argument("--out", default="train/data/gt_seam")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    clips = sorted(pathlib.Path(args.root).rglob("Control_Group/*.wav"))
    songs = sorted({str(p.parent.parent) for p in clips})
    rng0 = random.Random(args.seed); rng0.shuffle(songs)
    n_eval = max(1, len(songs) // 10)
    eval_songs = set(songs[:n_eval])
    out_train = pathlib.Path(args.out) / "train"; out_eval = pathlib.Path(args.out) / "eval"
    out_train.mkdir(parents=True, exist_ok=True); out_eval.mkdir(parents=True, exist_ok=True)
    n_tr = n_ev = 0
    for idx, p in enumerate(clips[args.shard::args.shards]):
        npz_name = f"{p.parent.parent.name}_{p.stem}.npz"
        split = "eval" if str(p.parent.parent) in eval_songs else "train"
        dst = (out_eval if split == "eval" else out_train) / npz_name
        if dst.exists(): continue
        rng = random.Random(args.seed * 7 + hash(npz_name) % 100000)
        try:
            s = build_one(p, p.with_suffix(".json"), rng)
        except Exception as e:
            print("SKIP", p.name, str(e)[:70]); continue
        if s is None: continue
        np.savez_compressed(dst, **s)
        if split == "eval": n_ev += 1
        else: n_tr += 1
    print(f"shard {args.shard}: train={n_tr} eval={n_ev} (songs={len(songs)} eval_songs={len(eval_songs)})")


if __name__ == "__main__":
    main()
