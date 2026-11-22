"""检验 HUM（Humanization）是否真的生效。

对同一个渲染请求渲染两次（HUM=0 与 HUM=N），比较帧能量包络：
  - 峰峰偏差 > 1 dB 说明音量通道在动
  - 再比较响帧/弱帧的高频占比，说明音色通道也在动

用法：
  python tools/hum_check.py --case <case.json> [--hum 40] [--url http://127.0.0.1:8573]
产物：
  artifacts/hum_check/hum_off.wav / hum_on_<N>.wav + 指标打印
"""
from __future__ import annotations

import argparse
import json
import pathlib
import urllib.request

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
SR = 44100
HOP = 512


def render(url: str, payload: dict) -> np.ndarray:
    req = urllib.request.Request(
        url.rstrip("/") + "/render",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return np.asarray(json.load(resp)["samples"], dtype=np.float32)


def frame_energy_db(wav: np.ndarray) -> np.ndarray:
    n = (len(wav) - HOP) // HOP
    return 20 * np.log10(np.array([
        np.sqrt((wav[i * HOP:(i + 1) * HOP] ** 2).mean() + 1e-12) for i in range(n)
    ]))


def hf_share(wav: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav)))) ** 2
    freq = np.fft.rfftfreq(len(wav), 1 / SR)
    return float(spec[freq > 4000].sum() / max(spec.sum(), 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=str(ROOT / "../diffical_new/artifacts/audio-v3-real-project/cases/phrase-111.857-fc5ff5c8effc47d9.json"))
    ap.add_argument("--hum", type=float, default=40.0)
    ap.add_argument("--url", default="http://127.0.0.1:8573")
    args = ap.parse_args()

    base = json.load(open(args.case, encoding="utf-8"))
    ticks = len(base.get("pitches") or [])
    out_dir = ROOT / "artifacts" / "hum_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, hum in (("off", 0.0), (f"on{int(args.hum)}", args.hum)):
        payload = json.loads(json.dumps(base))
        payload["curves"] = [{"abbr": "HUM", "values": [hum] * ticks}]
        payload["hash"] = int(base.get("hash", 0)) + 7000 + int(hum)   # 绕开前端缓存键
        results[label] = render(args.url, payload)
        sf.write(str(out_dir / f"hum_{label}.wav"), results[label], SR, subtype="PCM_16")

    off, on = results["off"], next(v for k, v in results.items() if k != "off")
    m = min(len(off), len(on))
    e_off, e_on = frame_energy_db(off[:m]), frame_energy_db(on[:m])
    delta = e_on - e_off
    p2p, std = float(np.ptp(delta)), float(np.std(delta))

    # 音色通道：取"响"与"弱"各 20% 帧，比较两者高频占比差
    order = np.argsort(e_off)
    quiet = off[:m][order[: len(order) // 5]]
    loud = off[:m][order[-len(order) // 5:]]
    tilt_off = hf_share(loud) - hf_share(quiet)
    on_q = on[:m][order[: len(order) // 5]]
    on_l = on[:m][order[-len(order) // 5:]]
    tilt_on = hf_share(on_l) - hf_share(on_q)

    print(f"HUM={args.hum:.0f} vs HUM=0")
    print(f"  帧能量偏差: 峰峰 {p2p:.2f} dB, 标准差 {std:.2f} dB, 相关性 {np.corrcoef(e_off, e_on)[0,1]:.5f}")
    print(f"  响/弱帧高频占比差: off {tilt_off:+.4f} -> on {tilt_on:+.4f} (差 {tilt_on - tilt_off:+.4f})")
    verdict = "生效" if p2p > 1.0 else "未生效"
    print(f"  结论: {verdict}（阈值 峰峰 > 1 dB）")
    print(f"  试听: {out_dir}/hum_off.wav 与 hum_on{int(args.hum)}.wav")
    return 0 if p2p > 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
