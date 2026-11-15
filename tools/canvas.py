"""把整句 mel 按音素切成片段，再放回画布 —— 模型输入的构造核心。

设计契约：
  - 片段 = target[:, a:b]，a/b 是标注给的音素帧跨度（hop512）
  - 画布 = 片段按原位置放回去，**不做交叉淡化**
  - 无退化时，画布必须逐帧等于 target（恒等性）
  - 退化（裁切错位/截断/拉伸）是唯一允许的偏差来源

用法：python3 tools/canvas.py <utt>
"""
from __future__ import annotations
import importlib.util, pathlib, sys
import numpy as np, soundfile as sf, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from zh_corpus import parse_file  # noqa: E402

SR, HOP, N_FFT, N_MELS = 44100, 512, 2048, 128
FRAME_MS = HOP / SR * 1000.0

_spec = importlib.util.spec_from_file_location("neo_wav2mel", ROOT / "backend/wav2mel.py")
_M = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_M)
EX = _M.PitchAdjustableMelSpectrogram(sample_rate=SR, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
                                     n_mels=N_MELS, f_min=40.0, f_max=16000.0, center=False, device="cpu")


def mel_of(w: np.ndarray) -> np.ndarray:
    if w.ndim > 1:
        w = w.mean(axis=1)
    with torch.no_grad():
        return EX(torch.from_numpy(w.astype(np.float32))).cpu().numpy()[0]


def spans_frames(ph_dur) -> list[tuple[int, int]]:
    out, t = [], 0.0
    for d in ph_dur:
        a = t * 1000.0 / FRAME_MS
        t += d
        out.append((int(round(a)), int(round(t * 1000.0 / FRAME_MS))))
    return out


def _resample(seg: np.ndarray, n: int) -> np.ndarray:
    if seg.shape[1] == n:
        return seg
    xs = np.linspace(0, seg.shape[1] - 1, seg.shape[1])
    xt = np.linspace(0, seg.shape[1] - 1, n)
    return np.stack([np.interp(xt, xs, seg[b]) for b in range(seg.shape[0])]).astype(np.float32)


def canvas_from_segments(target: np.ndarray, spans, *, jitter=0, cut_ratio=0.0, rng=None):
    """片段按目标跨度摆放（不交叉淡化），偏移施加在**内容裁取**上。

    - 摆放位置始终是标注跨度 [a,b) → 覆盖率恒为 1（不会出现空洞）
    - jitter: 内容裁取窗口在 [a+δ1, b+δ2] 上偏移，模拟 oto 偏移误差
    - cut_ratio: 内容末尾截掉 0~cut_ratio，再重采样回跨度长度
    - jitter=0 且 cut_ratio=0 时，内容与跨度完全一致 → 恒等于 target
    """
    T = target.shape[1]
    out = np.zeros_like(target)
    covered = np.zeros(T, dtype=bool)
    for i, (a, b) in enumerate(spans):
        a, b = max(0, min(T, a)), max(0, min(T, b))
        if b <= a:
            continue
        lo, hi = a, b
        n = hi - lo
        d1 = rng.randint(-jitter, jitter) if (jitter and rng) else 0
        d2 = rng.randint(-jitter, jitter) if (jitter and rng) else 0
        ca = max(0, min(T - 1, lo + d1))
        cb = max(ca + 2, min(T, hi + d2))
        seg = target[:, ca:cb]
        if cut_ratio and rng is not None:
            cut = int(round(seg.shape[1] * rng.uniform(0, cut_ratio)))
            if cut and seg.shape[1] - cut >= 2:
                seg = seg[:, : seg.shape[1] - cut]
        out[:, lo:hi] = _resample(seg, n)
        covered[lo:hi] = True
    return out, covered


def main() -> int:
    utt = sys.argv[1] if len(sys.argv) > 1 else "2001000033"
    u = {x.utt_id: x for x in parse_file(ROOT / "data/zh_corpus" / "transcriptions.txt")}[utt]
    wav, sr = sf.read(str(ROOT / "data/zh_corpus" / "wavs" / f"{utt}.wav"), dtype="float32")
    assert sr == SR
    target = mel_of(wav)
    T = target.shape[1]
    spans = spans_frames(u.ph_dur)
    print(f"{utt}: T={T} phones={len(spans)} sum_span={sum(b-a for a,b in spans)}")

    ident, covered = canvas_from_segments(target, spans)
    diff = np.abs(ident - target)
    print(f"[恒等] mean|diff|={diff.mean():.3e}  max|diff|={diff.max():.3e}  covered={covered.mean():.3f}")

    import random
    rng = random.Random(7)
    jit, _ = canvas_from_segments(target, spans, jitter=2, rng=rng)
    rng = random.Random(7)
    jc, _ = canvas_from_segments(target, spans, jitter=2, cut_ratio=0.1, rng=rng)
    for name, c in [("jitter±2", jit), ("jitter±2+cut10%", jc)]:
        d = np.abs(c - target)
        print(f"[{name}] mean|diff|={d.mean():.4f} max={d.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------- oto 参数化的"音节样本"画布 ----------------

def note_groups(u) -> list[tuple[int, int]]:
    """把连续音素按 (midi_dur, pitch) 聚成音符/音节组，返回 [start,end) 音素下标区间。"""
    md = [float(x) for x in u.midi_dur]
    raw_pt = getattr(u, "pitches", [0.0] * len(md))
    try:
        pt = [float(x) for x in raw_pt]
    except (TypeError, ValueError):
        pt = [0.0] * len(md)
    groups, start = [], 0
    for i in range(1, len(md)):
        if abs(md[i] - md[i - 1]) > 1e-9 or abs(pt[i] - pt[i - 1]) > 1e-9:
            groups.append((start, i))
            start = i
    groups.append((start, len(md)))
    return groups


def canvas_from_oto(target: np.ndarray, groups, note_spans, oto, *, pre_f=0, cut_f=0, rng=None):
    """按 oto 语义摆放音节样本：内容从音符前 pre 帧开始、尾部 cut 帧截掉，拉伸到覆盖到音符末。

    pre_f=cut_f=0 时必须是恒等（便于测试）。
    重叠区按幅度相加（naive assembler，由模型学会正确的交接）。
    返回画布（log-mel）与覆盖率。
    """
    T = target.shape[1]
    mag = np.zeros_like(target, dtype=np.float64)
    covered = np.zeros(T, dtype=bool)
    for (g0, g1) in groups:
        n0, n1 = note_spans[g0][0], note_spans[g1 - 1][1]
        n0, n1 = max(0, min(T, n0)), max(0, min(T, n1))
        if n1 <= n0:
            continue
        p = rng.randint(0, pre_f) if (pre_f and rng) else 0
        c = rng.randint(cut_f, 0) if (cut_f and rng) else 0
        s = max(0, n0 - p)
        e = min(T, n1 + c)
        if e - s < 2:
            continue
        seg = target[:, s:e]
        n_fill = n1 - s                      # 内容要覆盖到音符末
        if n_fill < 2:
            continue
        stretched = _resample(seg, n_fill)
        lo, hi = s, min(T, s + n_fill)
        stretched = stretched[:, : hi - lo]
        mag[:, lo:hi] += np.exp(stretched.astype(np.float64))
        covered[lo:hi] = True
    canvas = np.log(np.maximum(mag, 1e-9)).astype(np.float32)
    return canvas, covered


def canvas_from_overlap(target: np.ndarray, note_spans, ov_f: int, taper: bool = False):
    """C 改进版：音节片段原样摆放，相邻片段向右侧延长 ov_f 帧，重叠区按幅度相加。

    - 不做任何拉伸/重采样（避免毁音素）
    - ov_f=0 时必须恒等于 target
    - 重叠区是"朴素相加"，幅度会凸起——这正是 accumulator 的相干交接要修、
      而模型要接管的东西
    """
    T = target.shape[1]
    mag = np.zeros_like(target, dtype=np.float64)
    covered = np.zeros(T, dtype=bool)
    spans = [(max(0, min(T, a)), max(0, min(T, b))) for a, b in note_spans]
    for i, (n0, n1) in enumerate(spans):
        if n1 <= n0:
            continue
        s = n0
        e = min(T, n1 + (ov_f if i < len(spans) - 1 else 0))
        seg = target[:, s:e].astype(np.float64)
        if taper and ov_f > 0:
            w = np.ones(seg.shape[1], dtype=np.float64)
            if i > 0:
                k = min(ov_f, w.size); w[:k] = np.linspace(0, 1, k)
            if i < len(spans) - 1:
                k = min(ov_f, w.size); w[-k:] = np.minimum(w[-k:], np.linspace(1, 0, k))
            seg = seg + np.log(np.maximum(w, 1e-6))[None, :]
        mag[:, s:e] += np.exp(seg)
        covered[s:e] = True
    return np.log(np.maximum(mag, 1e-9)).astype(np.float32), covered
