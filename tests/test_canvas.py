"""画布契约的回归测试：恒等性、覆盖率、退化量级。"""
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from canvas import canvas_from_segments, mel_of, spans_frames  # noqa: E402
from opencpop import parse_file  # noqa: E402
import soundfile as sf  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
UTT = "2001000033"


def _load():
    u = {x.utt_id: x for x in parse_file(ROOT / "data/opencpop" / "transcriptions.txt")}[UTT]
    wav, _ = sf.read(str(ROOT / "data/opencpop" / "wavs" / f"{UTT}.wav"), dtype="float32")
    return mel_of(wav), spans_frames(u.ph_dur)


def test_identity():
    target, spans = _load()
    canvas, covered = canvas_from_segments(target, spans)
    assert np.abs(canvas - target).max() < 1e-6, "无退化时画布必须恒等于目标"
    assert covered.all(), "覆盖率必须为 1"


def test_degradation_magnitude():
    """退化必须：覆盖率=1、偏差有限且随退化强度单调增大。具体数值记录而非猜测。"""
    import random
    target, spans = _load()
    ds = {}
    for jitter, cut in [(0, 0.0), (1, 0.05), (2, 0.0)]:
        rng = random.Random(7)
        c, covered = canvas_from_segments(target, spans, jitter=jitter, cut_ratio=cut, rng=rng)
        assert covered.all(), "退化后覆盖率仍必须为 1"
        ds[(jitter, cut)] = float(np.abs(c - target).mean())
    assert ds[(0, 0.0)] < 1e-6
    assert 0 < ds[(1, 0.05)] < ds[(2, 0.0)], f"退化强度未单调: {ds}"
    print("degradation magnitudes:", {f"j{k}+c{c}": round(v, 4) for (k, c), v in ds.items()})


if __name__ == "__main__":
    test_identity(); test_degradation_magnitude(); print("canvas tests OK")
