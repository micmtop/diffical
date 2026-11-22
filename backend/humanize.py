"""HUM（Humanization）：乐句级、音符驱动的慢速人性化起伏。

不是随机噪声，也不动音高和拼接几何。它读音符结构算出一条慢速包络，再按 HUM
曲线的强度缩放，在**装配完成、进声码器之前**施加：

  音量通道：在 DYN 之上叠加低速增益波动（mel 加常数 dB）
  音色通道：同一包络驱动一条谱倾斜——响的部分偏亮/偏紧，弱的部分偏暗/偏气

两个后端（Python / Rust）用同一套整数哈希与公式，结果逐位一致；HUM=0 时是空操作。

强度档位（与前端表达式一致，0..100）：
  0        关闭
  10..25   柔和自然
  30..45   起伏明显
  >50      幅度过大，慎用
"""
from __future__ import annotations

import os

import numpy as np

ABBR = "HUM"
# 未绘制 HUM 时的默认强度：0 = 不改变现有工程
DEFAULT_HUM = float(os.environ.get("DIFFICAL_HUM_DEFAULT", "0"))
MAX_GAIN_DB = 6.0     # HUM=100 时的音量起伏峰值
MAX_TILT_DB = 3.0     # HUM=100 时的高低频倾斜幅度
FRAME_MS = 512.0 / 44100.0 * 1000.0


def hum_per_frame(curves, total_frames: int, default: float = DEFAULT_HUM) -> np.ndarray:
    """取 HUM 曲线并重采样到逐帧；缺失时返回默认值。"""
    T = max(1, int(total_frames))
    for entry in curves or ():
        if isinstance(entry, dict):
            abbr, values = entry.get("abbr"), entry.get("values")
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            abbr, values = entry[0], entry[1]
        else:
            continue
        if str(abbr).upper() != ABBR or not values:
            continue
        v = np.asarray(values, dtype=np.float32)
        if v.size == T:
            return np.clip(v, 0.0, 100.0)
        src = np.linspace(0.0, 1.0, v.size)
        dst = np.linspace(0.0, 1.0, T)
        return np.clip(np.interp(dst, src, v).astype(np.float32), 0.0, 100.0)
    return np.full(T, float(default), dtype=np.float32)


def phrase_envelope(notes, total_frames: int, frame_ms: float = FRAME_MS,
                    lead_ms: float = 0.0) -> np.ndarray:
    """由音符结构生成慢速起伏包络，取值约在 [-1, 1]。

    每个音符贡献一个"起-落"的呼吸形状：
      - 音符越长，起伏越深越慢（短音符压低幅度）
      - 旋律上行时尾部微微抬升，下行时自然收弱
    """
    T = max(1, int(total_frames))
    env = np.zeros(T, dtype=np.float64)
    if not notes:
        return env.astype(np.float32)
    # 音符位置是工程绝对毫秒；乐句起点 = 首个音符位置 - 前导静音
    positions = [float(getattr(n, "position_ms", 0.0) or 0.0) for n in notes]
    base = min(positions) - max(0.0, float(lead_ms or 0.0))
    prev_tone = None
    for note in notes:
        dur = float(getattr(note, "duration_ms", 0.0) or 0.0)
        start = float(getattr(note, "position_ms", 0.0) or 0.0) - base
        tone = float(getattr(note, "tone", 0) or 0)
        if dur <= 0.0:
            continue
        depth = float(np.clip(dur / 900.0, 0.3, 1.0))
        slope = 0.0 if prev_tone is None else float(np.clip((tone - prev_tone) / 2.0, -1.0, 1.0))
        prev_tone = tone
        i0 = int(np.floor(start / frame_ms))
        i1 = int(np.ceil((start + dur) / frame_ms))
        i0, i1 = max(0, i0), min(T, i1)
        if i1 <= i0:
            continue
        idx = np.arange(i0, i1, dtype=np.float64)
        u = np.clip((idx * frame_ms - start) / dur, 0.0, 1.0)
        # 起-落形状：0.35 处到峰，之后按音符长度缓慢衰减
        rise = np.sin(np.pi * np.clip(u / 0.7, 0.0, 1.0))
        decay = np.exp(-3.0 * u)
        shape = rise * (0.55 + 0.45 * decay)
        # 旋律方向：上行抬尾，下行收弱
        shape = shape * (1.0 + 0.35 * slope * u)
        env[i0:i1] += depth * shape
    # 慢速平滑（约 250 ms），保证是乐句级起伏而不是抖动
    win = max(1, int(round(250.0 / frame_ms)))
    if win > 1 and T > win:
        kernel = np.ones(win, dtype=np.float64) / win
        env = np.convolve(env, kernel, mode="same")
    # 去掉直流：调制围绕原有 DYN，不整体抬高响度
    env -= float(env.mean())
    peak = float(np.max(np.abs(env))) if env.size else 0.0
    if peak > 1e-9:
        env /= peak
    return env.astype(np.float32)


def apply(mel: np.ndarray, hum: np.ndarray, env: np.ndarray) -> np.ndarray:
    """把 HUM 施加到整句 mel 上：音量增益 + 谱倾斜。

    mel 是 log 幅度（自然对数），dB 与它的换算系数是 8.686。
    """
    if mel.size == 0:
        return mel
    n_mels, T = mel.shape
    h = np.clip(np.asarray(hum, dtype=np.float64), 0.0, 100.0)[:T] / 100.0
    e = np.asarray(env, dtype=np.float64)[:T]
    if h.size < T:
        h = np.pad(h, (0, T - h.size))
    if e.size < T:
        e = np.pad(e, (0, T - e.size))
    drive = h * e
    if not np.any(np.abs(drive) > 1e-9):
        return mel
    gain_db = drive * MAX_GAIN_DB
    tilt_db = drive * MAX_TILT_DB
    # 低端 -1，高端 +1 的线性倾斜
    weight = np.linspace(-1.0, 1.0, n_mels)[:, None]
    delta = (gain_db[None, :] + tilt_db[None, :] * weight) / 8.686
    out = np.asarray(mel, dtype=np.float32) + delta.astype(np.float32)
    return out
