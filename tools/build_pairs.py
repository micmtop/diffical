"""唯一的数据集构造器（画布契约）。

输入（模型看到）：
  source_mel (128,T)  每个音素片段按目标跨度摆放，不做交叉淡化；施加 jitter=±1 帧、截尾≤5%
  phone_ids  (T,)     每帧所属音素
  boundary   (T,)     边界 ±4 帧
  durations  (n,)     每音素帧长
  phones     (n,)     音素名
  f0         (T,)     目标 F0（来自 harvest，推理时来自 OU 曲线）
输出（目标）：
  target_mel (128,T)  真实录音的 mel

恒等性：jitter=0, cut=0 时 source_mel 必须逐帧等于 target_mel（tests/test_canvas.py 校验）。
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, random, sys
import numpy as np, soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from canvas import mel_of, spans_frames, note_groups, canvas_from_overlap  # noqa: E402
from opencpop import parse_file  # noqa: E402
import pyworld as pw  # noqa: E402

SR, HOP, N_MELS = 44100, 512, 128
FRAME_MS = HOP / SR * 1000.0
OVERLAP_MS, BOUNDARY_W = 83.3, 4
F0_FLOOR, F0_CEIL = 60.0, 1200.0


def f0_of(wav: np.ndarray, T: int) -> np.ndarray:
    f0, _t = pw.harvest(wav.astype(np.float64), SR, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL,
                        frame_period=FRAME_MS)
    if len(f0) == 0:
        return np.zeros(T, dtype=np.float32)
    xs = np.arange(len(f0), dtype=np.float64)
    xt = np.linspace(0, len(f0) - 1, T) if len(f0) > 1 else np.zeros(T)
    return np.interp(xt, xs, f0).astype(np.float32)


def build_one(u, seed: int) -> dict:
    wav, sr = sf.read(str(ROOT / "data/opencpop" / "wavs" / f"{u.utt_id}.wav"), dtype="float32")
    assert sr == SR, (u.utt_id, sr)
    target = mel_of(wav)
    T = target.shape[1]
    spans = spans_frames(u.ph_dur)
    groups = note_groups(u)
    note_spans = [(spans[a][0], spans[b - 1][1]) for a, b in groups]
    ov_f = int(round(OVERLAP_MS / FRAME_MS))
    canvas, covered = canvas_from_overlap(target, note_spans, ov_f, taper=False)

    phone_ids = np.full(T, -1, dtype=np.int16)
    boundary = np.zeros(T, dtype=np.uint8)
    for i, (a, b) in enumerate(spans):
        phone_ids[max(0, min(T, a)):max(0, min(T, b))] = i
    for a, _b in spans[1:]:
        boundary[max(0, a - BOUNDARY_W):min(T, a + BOUNDARY_W + 1)] = 1

    return {
        "target_mel": target.astype(np.float16),
        "source_mel": canvas.astype(np.float16),
        "f0": f0_of(wav, T),
        "phone_ids": phone_ids,
        "boundary": boundary,
        "durations": np.array([b - a for a, b in spans], dtype=np.int16),
        "phones": np.array(u.phones, dtype=object),
        "utt": np.array(u.utt_id),
        "covered": np.array(float(covered.mean()), dtype=np.float32),
        "degrade": np.array([OVERLAP_MS, ov_f], dtype=np.float32),
        "note_count": np.array(len(groups), dtype=np.int16),
    }


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="train/data/pairs_v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    utts = {u.utt_id: u for u in parse_file(ROOT / "data/opencpop" / "transcriptions.txt")}
    def readable(uid: str) -> bool:
        try:
            sf.info(str(ROOT / "data/opencpop" / "wavs" / f"{uid}.wav"))
            return True
        except Exception:
            return False

    pool = [u for u in utts.values() if 3.0 <= u.total_s <= 8.0 and readable(u.utt_id)]
    print(f"pool={len(pool)} (readable wavs only)", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    picked = pool[: args.n]
    out = ROOT / args.out
    manifest = {"version": "v2-overlap", "overlap_ms": OVERLAP_MS,
                "boundary_w": BOUNDARY_W, "frame_ms": FRAME_MS, "n_mels": N_MELS,
                "identity_required": True, "samples": []}
    for split, part in (("train", picked[: max(1, int(len(picked) * 0.9))]), ("eval", picked[max(1, int(len(picked) * 0.9)):])):
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        for u in part:
            try:
                s = build_one(u, args.seed * 100003 + int(u.utt_id))
            except Exception as e:
                manifest.setdefault("skipped", []).append({"utt": u.utt_id, "error": str(e)[:120]})
                print(f"SKIP {u.utt_id}: {str(e)[:90]}", flush=True)
                continue
            p = d / f"{u.utt_id}.npz"
            np.savez_compressed(p, **s)
            manifest["samples"].append({"split": split, "utt": u.utt_id,
                                        "frames": int(s["target_mel"].shape[1]),
                                        "covered": float(s["covered"]),
                                        "sha256": sha256_file(p)})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"built {len(manifest['samples'])} pairs -> {out}")
    print(f"covered mean = {np.mean([s['covered'] for s in manifest['samples']]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
