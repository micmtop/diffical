"""P0.5 训练对生成器（本机跑，Colab 只训练）。

每个样本：
    target_mel   (128,T) float16   真实录音的整句 mel（规范前端）
    source_mel   (128,T) float16   用捐赠句片段 + 真实中性装配算子拼出的 naive mel（模型输入）
    f0           (T,)    float16   pyworld.harvest 实测 F0（Hz，0=清音）
    phone_ids    (T,)    int16     每帧所属音素下标
    boundary     (T,)    uint8     边界 ±4 帧 = 1
    durations    (n,)    int16     每音素帧长
    phones       list[str]         音素名
    utt          str               句 id

用法：
    python3 train/pairs.py --n 100 --out train/data/pairs
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np
import pyworld as pw
import soundfile as sf
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from backend.overlap import MelOverlapAccumulator  # noqa: E402
from backend.wav2mel import PitchAdjustableMelSpectrogram  # noqa: E402
from opencpop import parse_file  # noqa: E402

SR, HOP, N_FFT, WIN, N_MELS = 44100, 512, 2048, 2048, 128
FMIN, FMAX = 40.0, 16000.0
FRAME_MS = HOP / SR * 1000.0
OVERLAP_MS = 83.0          # §3.3 实测 oto overlap 中位
BOUNDARY_W = 4             # ±50ms
F0_FLOOR, F0_CEIL = 60.0, 1200.0

ORIGIN_HOP = 128                 # hifisampler 的 origin_hop_size
ORIGIN_MS = ORIGIN_HOP / SR * 1000.0
HOP_RATIO = HOP // ORIGIN_HOP    # hop512 / hop128 = 4

_extractor = PitchAdjustableMelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
    n_mels=N_MELS, f_min=FMIN, f_max=FMAX, center=False, device="cpu",
)
_extractor128 = PitchAdjustableMelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, win_length=WIN, hop_length=ORIGIN_HOP,
    n_mels=N_MELS, f_min=FMIN, f_max=FMAX, center=False, device="cpu",
)


def mel_of(wav: np.ndarray) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    with torch.no_grad():
        return _extractor(torch.from_numpy(wav.astype(np.float32))).cpu().numpy()[0]


def mel128_of(wav: np.ndarray) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    with torch.no_grad():
        return _extractor128(torch.from_numpy(wav.astype(np.float32))).cpu().numpy()[0]


def load_wav(path: pathlib.Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path.name}: sr={sr}")
    return wav


def f0_of(wav: np.ndarray, T: int) -> np.ndarray:
    f0, _t = pw.harvest(wav.astype(np.float64), SR, f0_floor=F0_FLOOR,
                        f0_ceil=F0_CEIL, frame_period=FRAME_MS)
    if len(f0) == 0:
        return np.zeros(T, dtype=np.float32)
    xs = np.arange(len(f0), dtype=np.float64)
    xt = np.linspace(0, len(f0) - 1, T) if len(f0) > 1 else np.zeros(T)
    return np.interp(xt, xs, f0).astype(np.float32)


def spans_frames(u) -> list[tuple[float, float]]:
    out, t = [], 0.0
    for d in u.ph_dur:
        a = t * 1000.0 / FRAME_MS
        t += d
        out.append((a, t * 1000.0 / FRAME_MS))
    return out


def stretch(frag: np.ndarray, n: int) -> np.ndarray:
    """旧路径（hop512 上均匀插值），仅 fallback 使用。"""
    if frag.shape[1] == n or frag.shape[1] == 0:
        return frag
    xs = np.linspace(0, frag.shape[1] - 1, frag.shape[1])
    xt = np.linspace(0, frag.shape[1] - 1, n)
    return np.stack([np.interp(xt, xs, frag[b]) for b in range(frag.shape[0])])


def warp_origin(seg128: np.ndarray, n_out: int, con_frac: float = 0.2) -> np.ndarray:
    """hifisampler 式 warp：hop128 域分段时域映射，直接采到 hop512 网格。

    stretch(t) = t / vel            (t < con)
               = con + (t-con) / r  (t >= con)
    这里 vel 取 1（无速度变化），r 由目标时长决定；辅音段不拉伸。
    """
    n_in = seg128.shape[1]
    if n_in < 2:
        return np.repeat(seg128, n_out, axis=1)[:, :n_out].astype(np.float32)
    src_ms = n_in * ORIGIN_MS
    tgt_ms = n_out * FRAME_MS
    con = max(1e-6, con_frac * src_ms)
    ratio = max(1e-6, (tgt_ms - con) / max(1e-6, src_ms - con))
    T = (np.arange(n_out) + 0.5) * FRAME_MS
    t_src = np.where(T < con, T, con + (T - con) / ratio)
    f_src = np.clip(t_src / ORIGIN_MS, 0, n_in - 1)
    # 线性插值矩阵
    W = np.zeros((n_out, n_in), dtype=np.float32)
    p = f_src.copy()
    i0 = np.floor(p).astype(int)
    i1 = np.minimum(i0 + 1, n_in - 1)
    fr = (p - i0).astype(np.float32)
    W[np.arange(n_out), i0] += 1 - fr
    W[np.arange(n_out), i1] += fr
    return (W @ seg128.T.astype(np.float32)).T.astype(np.float32)


class Bank:
    """捐赠句音素索引。"""

    def __init__(self, donors: list[str], utts: dict):
        self.utts = utts
        self.index: dict[str, list[tuple[str, int, int]]] = {}
        for uid in donors:
            u = utts[uid]
            for ph, (a, b) in zip(u.phones, spans_frames(u)):
                ia = int(round(a))
                ib = max(ia + 1, int(round(b)))
                self.index.setdefault(ph, []).append((uid, ia, ib))
        self._mel: dict[str, np.ndarray] = {}
        self._mel128: dict[str, np.ndarray] = {}

    def mel(self, uid: str) -> np.ndarray:
        if uid not in self._mel:
            self._mel[uid] = mel_of(load_wav(ROOT / "data/opencpop/wavs" / f"{uid}.wav"))
        return self._mel[uid]

    def mel128(self, uid: str) -> np.ndarray:
        if uid not in self._mel128:
            self._mel128[uid] = mel128_of(load_wav(ROOT / "data/opencpop/wavs" / f"{uid}.wav"))
        return self._mel128[uid]


def neutral_env(core_a: float, core_b: float, a0: float, a1: float) -> list[tuple[float, float]]:
    """5 点结构包络（两端为 0）——accumulator 的相干交接路径要求这个形状。

    传 2 点平坦包络会让 add() 走 blocked 分支（返回前不记录 neutral），
    导致重叠区不再做相干交接，听感就是边界突变/杂音。
    """
    half = OVERLAP_MS / 2.0
    p0 = a0 - core_a
    p1 = max(p0 + 5.0, p0 + max(5.0, OVERLAP_MS))
    p2 = max(0.0, p1)
    p4 = a1 - core_a
    p3 = max(p2, p4 - 35.0)
    return [(p0, 0.0), (p1, 1.0), (p2, 1.0), (p3, 1.0), (p4, 0.0)]


def build_one(u, bank: Bank, rng: random.Random) -> dict:
    wav = load_wav(ROOT / "data/opencpop/wavs" / f"{u.utt_id}.wav")
    target = mel_of(wav)
    T = target.shape[1]
    total_ms = u.total_s * 1000.0
    spans = spans_frames(u)
    half = OVERLAP_MS / 2.0

    phone_ids = np.full(T, -1, dtype=np.int16)
    boundary = np.zeros(T, dtype=np.uint8)
    for i, (a, b) in enumerate(spans):
        phone_ids[max(0, int(round(a))):min(T, int(round(b)))] = i
    for a, _b in spans[1:]:
        f = int(round(a))
        boundary[max(0, f - BOUNDARY_W):min(T, f + BOUNDARY_W + 1)] = 1

    acc = MelOverlapAccumulator(N_MELS, T, FRAME_MS)
    fallbacks = 0
    for i, ph in enumerate(u.phones):
        core_a = spans[i][0] * FRAME_MS
        core_b = spans[i][1] * FRAME_MS
        a0 = max(0.0, core_a - half)
        a1 = min(total_ms, core_b + half)
        n = max(1, int(round((a1 - a0) / FRAME_MS)))
        # 同一条录音的片段：取"含上下文的重叠窗口"(a0..a1)，再模拟裁切不准
        #   —— 关键是窗口本身要 ~n 帧，不能再 stretch 放大（否则短音素被拉坏）
        w0 = int(round(a0 / FRAME_MS))
        w1 = int(round(a1 / FRAME_MS))
        a_j = max(0, w0 + rng.randint(-2, 2))
        b_j = min(T, max(a_j + 2, w1 + rng.randint(-2, 2)))
        seg = target[:, a_j:b_j]
        cut = rng.randint(0, max(1, int(0.1 * seg.shape[1])))
        if cut:
            seg = seg[:, : seg.shape[1] - cut]
        frag = stretch(seg, n) if seg.shape[1] != n else seg
        acc.add(frag, a0 / FRAME_MS, neutral_env(core_a, core_b, a0, a1), core_a)
    source = acc.compose()

    return {
        "target_mel": target.astype(np.float16),
        "source_mel": source.astype(np.float16),
        "f0": f0_of(wav, T),
        "phone_ids": phone_ids,
        "boundary": boundary,
        "durations": np.array([int(round(b - a)) for a, b in spans], dtype=np.int16),
        "phones": np.array(u.phones, dtype=object),
        "utt": np.array(u.utt_id),
        "fallback_phones": np.array(fallbacks, dtype=np.int16),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="train/data/pairs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    utts = {u.utt_id: u for u in parse_file(ROOT / "data/opencpop/transcriptions.txt")}
    donors = json.loads((ROOT / "data/opencpop/donors.json").read_text())["ids"]
    donor_songs = {int(x) // 10**6 for x in donors}

    pool = [u for u in utts.values()
            if u.utt_id not in donors
            and all(int(u.utt_id) // 10**6 != s for s in donor_songs)
            and 3.0 <= u.total_s <= 8.0]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    picked = pool[: args.n]
    # 留 10% 做 eval，且 eval 的句子不进 Bank 的 target 侧（Bank 只从 donors 取）
    n_eval = max(5, len(picked) // 10)
    splits = {"eval": picked[:n_eval], "train": picked[n_eval:]}

    bank = Bank(donors, utts)
    out = ROOT / args.out
    manifest = {"seed": args.seed, "overlap_ms": OVERLAP_MS, "frame_ms": FRAME_MS,
                "n_mel": N_MELS, "splits": {}, "samples": []}
    for split, us in splits.items():
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        names = []
        for u in us:
            sample = build_one(u, bank, random.Random(args.seed * 100003 + int(u.utt_id)))
            p = d / f"{u.utt_id}.npz"
            np.savez_compressed(p, **sample)
            names.append(p.name)
            manifest["samples"].append({"split": split, "utt": u.utt_id,
                                        "frames": int(sample["target_mel"].shape[1]),
                                        "fallback_phones": int(sample["fallback_phones"]),
                                        "bytes": p.stat().st_size})
        manifest["splits"][split] = names
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(s["bytes"] for s in manifest["samples"])
    print(f"picked={len(picked)} train={len(splits['train'])} eval={len(splits['eval'])}")
    print(f"bytes={total/1e6:.1f}MB  fallback_total={sum(s['fallback_phones'] for s in manifest['samples'])}")
    print(f"manifest -> {out/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
