"""UTAU 几何：把音素窗口放到时间轴上，并给出每帧的包络增益。

训练和推理必须共用这一份实现，否则语义会再次漂移。

模型看到的东西（由本模块产出的 compose_inputs）：
  cur[t]  当前帧"主音素"在自己窗口里的内容（直读，不生成）
  prev[t] 前一音素在自己窗口里的内容（重叠区才有意义）
  nex[t]  后一音素在自己窗口里的内容
  g_cur/g_prev/g_nex  对应包络增益（线性，0..1）
  ov[t]   重叠位置：0=不在重叠区，1=重叠区末尾

时间轴：每个音素的播放位置由它自己的 envelope p0 决定。相邻音素的
重叠量 = 当前音素的 (p1 - p0)，与 UTAU 的 envelope 语义一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

FRAME_MS = 512.0 / 44100.0 * 1000.0
N_MELS = 128


@dataclass
class PhoneWindow:
    """一个音素在时间轴上的窗口。"""

    phoneme: str
    mel: np.ndarray            # (N_MELS, L) 窗口内容（hop512，已按 UTAU 语义拉伸）
    start_frame: int           # 窗口起点（短语内帧号）
    anchor_ms: float           # 音符起点在短语内的毫秒位置（envelope x=0 处）
    env: list[tuple[float, float]] = field(default_factory=list)
    preutter_ms: float = 0.0
    duration_ms: float = 0.0
    note_index: int = 0


def env_from_phone(phone) -> list[tuple[float, float]]:
    """从 RenderPhone 取 envelope 点；允许 dict / tuple / EnvelopePoint。"""
    src = getattr(phone, "envelope", None) or []
    out = []
    for p in src:
        if isinstance(p, dict):
            x, y = p.get("x", 0.0), p.get("y", 0.0)
        elif isinstance(p, (tuple, list)):
            x, y = p[0], p[1]
        else:
            x, y = getattr(p, "x", 0.0), getattr(p, "y", 0.0)
        out.append((float(x), float(y)))
    return out


def standard_envelope(preutter_ms: float, overlap_ms: float,
                      duration_ms: float, tail_ms: float) -> list[tuple[float, float]]:
    """按 UTAU 的 p0..p4 造一条标准包络（训练数据构造用）。

    p0 = -(preutter)                       起点（前一个音素的尾巴在这里）
    p1 = -(preutter) + overlap             重叠区结束，本音素增益到 1
    p2 = 0                                 音符起点
    p3 = duration                          音符终点
    p4 = duration + tail                   本音素尾巴结束
    """
    p0 = -float(preutter_ms)
    p1 = p0 + max(0.0, float(overlap_ms))
    p2 = 0.0
    p3 = max(p2 + 1.0, float(duration_ms))
    p4 = p3 + max(0.0, float(tail_ms))
    # UTAU 的 y 单位是 %：p0/p4 可以不是 0；这里用标准结构淡入淡出
    return [(p0, 0.0), (p1, 1.0), (p2, 1.0), (p3, 1.0), (p4, 0.0)]


def envelope_gain(frames_abs: np.ndarray, start_frame: float, frame_ms: float,
                  anchor_ms: float, env: Sequence[tuple[float, float]]) -> np.ndarray:
    """每个绝对帧上的包络增益（线性）。

    frames_abs 是短语内帧号；时间 = frames_abs*frame_ms - anchor_ms，
    与 envelope 的 x 坐标（相对音符起点）对齐。
    """
    if len(env) < 2:
        return np.ones(len(frames_abs), dtype=np.float32)
    xs = np.asarray([p[0] for p in env], dtype=np.float64)
    ys = np.asarray([p[1] for p in env], dtype=np.float64)
    # OpenUtau 的 envelope y 可能是 0..100（百分比）也可能是 0..1，按最大值判断
    if ys.max() > 1.5:
        ys = ys / 100.0
    t = np.asarray(frames_abs, dtype=np.float64) * frame_ms - float(anchor_ms)
    return np.interp(t, xs, ys, left=ys[0], right=ys[-1]).astype(np.float32)


def window_gain(win: PhoneWindow, frame_ms: float = FRAME_MS) -> np.ndarray:
    """该窗口每个 mel 帧的增益。"""
    frames = win.start_frame + np.arange(win.mel.shape[1])
    return envelope_gain(frames, win.start_frame, frame_ms, win.anchor_ms, win.env)


def compose_inputs(windows: list[PhoneWindow], total_frames: int,
                   frame_ms: float = FRAME_MS, n_mels: int = N_MELS) -> dict:
    """把窗口铺到时间轴上（向量化），产出模型输入。

    cur/prev/nex 是三个音素各自在自己窗口里的内容；g_* 是对应包络增益；
    ov 是主音素在 [p0,p1] 重叠区内的相对位置。
    """
    T = int(total_frames)
    S = len(windows)
    quiet = np.float32(-30.0)
    if S == 0:
        z = np.full((n_mels, T), quiet, dtype=np.float32)
        zz = np.zeros((1, T), dtype=np.float32)
        return dict(cur=z, prev=z.copy(), nex=z.copy(), g_cur=zz, g_prev=zz.copy(),
                    g_nex=zz.copy(), ov=zz.copy(), primary=np.full(T, -1, np.int64),
                    smooth=z.copy())

    starts = np.array([w.start_frame for w in windows], dtype=np.int64)
    lens = np.array([w.mel.shape[1] for w in windows], dtype=np.int64)
    Lmax = int(lens.max())
    mel = np.full((S, n_mels, Lmax), quiet, dtype=np.float32)
    gains = np.zeros((S, T), dtype=np.float32)
    for i, w in enumerate(windows):
        mel[i, :, :lens[i]] = w.mel[:, :Lmax]
        g = window_gain(w, frame_ms)
        a = max(0, starts[i]); b = min(T, starts[i] + lens[i])
        if b > a:
            gains[i, a:b] = g[a - starts[i]: b - starts[i]]

    active = gains > 1e-4
    primary = np.where(active.any(0), np.argmax(gains, axis=0), -1).astype(np.int64)
    t_idx = np.arange(T)
    cols = t_idx - starts[primary.clip(0)]
    rows = primary.clip(0)
    valid = (primary >= 0) & (cols >= 0) & (cols < Lmax)
    cols_clip = np.clip(cols, 0, Lmax - 1)
    cur = np.where(valid[None, :], mel[rows, :, cols_clip].T, quiet).astype(np.float32)
    g_cur = np.where(valid, gains[rows, np.clip(t_idx, 0, T - 1)], 0.0)[None, :].astype(np.float32)

    idx = np.arange(S)[:, None]
    before = active & (idx < primary[None, :])
    after = active & (idx > primary[None, :])
    has_prev = before.any(0); has_nex = after.any(0)
    prev_i = np.where(has_prev, np.argmax(np.where(before, gains, -1.0), axis=0), 0)
    nex_i = np.where(has_nex, np.argmax(np.where(after, gains, -1.0), axis=0), 0)
    for name, sel, has, out, gout in (("prev", prev_i, has_prev, "prev", "g_prev"),
                                      ("nex", nex_i, has_nex, "nex", "g_nex")):
        k = t_idx - starts[sel]
        ok = has & (k >= 0) & (k < Lmax)
        vals = np.where(ok[None, :], mel[sel, :, np.clip(k, 0, Lmax - 1)].T, quiet).astype(np.float32)
        gv = np.where(ok, gains[sel, np.clip(t_idx, 0, T - 1)], 0.0)[None, :].astype(np.float32)
        if out == "prev":
            prev, g_prev = vals, gv
        else:
            nex, g_nex = vals, gv

    # 重叠位置：主音素 envelope 的 [p0,p1] 区间
    p0 = np.array([w.env[0][0] if w.env else 0.0 for w in windows], np.float32)
    p1 = np.array([w.env[1][0] if len(w.env) > 1 else (w.env[0][0] + 1.0 if w.env else 1.0)
                   for w in windows], np.float32)
    anchor = np.array([w.anchor_ms for w in windows], np.float32)
    x = t_idx * frame_ms - anchor[rows]
    ov = np.where(has_prev, np.clip((x - p0[rows]) / np.maximum(p1[rows] - p0[rows], 1e-6), 0, 1), 0)
    ov = ov[None, :].astype(np.float32)

    power = np.zeros((n_mels, T), dtype=np.float64)
    for i, w in enumerate(windows):
        a = max(0, starts[i]); b = min(T, starts[i] + lens[i])
        if b > a:
            seg = mel[i][:, a - starts[i]: b - starts[i]]
            power[:, a:b] += np.exp(2.0 * np.clip(seg, -30.0, 2.0)) * gains[i, a:b][None, :]
    smooth = (np.log(np.maximum(power, 1e-24)) / 2.0).astype(np.float32)
    return dict(cur=cur, prev=prev, nex=nex, g_cur=g_cur, g_prev=g_prev, g_nex=g_nex,
                ov=ov, primary=primary, smooth=smooth)


def windows_from_features(phone_features: Iterable[dict],
                          frame_ms: float = FRAME_MS) -> list[PhoneWindow]:
    """把 renderer 的 phone_features 转成 PhoneWindow 列表（推理用）。"""
    out = []
    for f in phone_features:
        mel = f.get("mel")
        if mel is None or mel.shape[1] < 1:
            continue
        ph = f.get("phone")
        out.append(PhoneWindow(
            phoneme=getattr(ph, "phoneme", "?"),
            mel=np.asarray(mel, dtype=np.float32),
            start_frame=int(round(float(f.get("start_frame_exact", f.get("start_frame", 0))))),
            # 窗口起点 = 音符起点 - preutter，所以音符起点可以反推
            anchor_ms=float(round(float(f.get("start_frame_exact", f.get("start_frame", 0))))) * frame_ms
                     + float(getattr(ph, "preutter_ms", 0.0) or 0.0),
            env=env_from_phone(ph),
            preutter_ms=float(getattr(ph, "preutter_ms", 0.0) or 0.0),
            duration_ms=float(getattr(ph, "duration_ms", 0.0) or 0.0),
            note_index=int(getattr(ph, "note_index", 0) or 0),
        ))
    return out


# ---------------------------------------------------------------- 窗口与时间映射
def plan_oto_window(offset_ms: float, consonant_ms: float, cutoff_ms: float,
                    total_time_ms: float, preutter_ms: float, velocity: float,
                    envelope_span_ms: float, *, frame_ms: float = FRAME_MS,
                    origin_frame_ms: float = 128.0 / 44100.0 * 1000.0,
                    oto_preutter_ms: float = 0.0) -> dict:
    """UTAU 的 oto 窗口 + 时间映射（renderer.process 与数据构造共用这一份）。

    语义与 OpenUtau/UTAU 一致：
      - start = oto.offset；con = offset + consonant；end 由 cutoff 决定
      - 辅音段按 VEL 缩放（不随音符时长）；元音段按比例拉伸
      - 长音（拉伸超过 2.5 倍）走"起音 + 保持漂移"轨迹
    返回: start_ms / con_ms / end_ms / con_vel / target_n_frames /
          t_mapped_ms（每个输出帧对应的源时间，毫秒）/ source_present
    """
    start_ms = max(float(offset_ms), 0.0)
    con_ms = float(offset_ms) + float(consonant_ms)
    if cutoff_ms < 0:
        end_ms = float(offset_ms) + abs(float(cutoff_ms))
    elif cutoff_ms > 0:
        end_ms = float(total_time_ms) - float(cutoff_ms)
    else:
        end_ms = float(total_time_ms)
    end_ms = min(end_ms, float(total_time_ms))
    end_ms = max(end_ms, con_ms)

    vel = 2.0 ** (1.0 - float(velocity))
    pre_vel_ms = max(0.0, float(preutter_ms))
    source_anchor_skip_ms = float(oto_preutter_ms) * vel - pre_vel_ms
    con_vel = float(consonant_ms) * vel
    con_output_ms = con_vel - source_anchor_skip_ms

    total_span_ms = max(frame_ms, float(envelope_span_ms), con_output_ms + frame_ms)
    stretch_length_ms = max(0.0, end_ms - con_ms)
    vowel_output_ms = max(frame_ms, total_span_ms - con_output_ms)
    scaling_ratio = max(1.0, vowel_output_ms / max(stretch_length_ms, origin_frame_ms))
    target_n_frames = max(1, int(np.ceil(total_span_ms / frame_ms)))

    t_target_ms = np.arange(target_n_frames) * frame_ms + frame_ms / 2
    source_clock_ms = t_target_ms + source_anchor_skip_ms
    is_consonant = source_clock_ms < con_vel
    consonant_mapped = start_ms + source_clock_ms / vel
    vowel_mapped = con_ms + (source_clock_ms - con_vel) / scaling_ratio

    _src_vowel = max(0.0, stretch_length_ms)
    _tgt_vowel = vowel_output_ms
    _max_stretch = 2.5
    if _src_vowel > 1.0 and _tgt_vowel > _src_vowel * _max_stretch:
        _onset_src = _src_vowel * 0.45
        _onset_tgt = _onset_src * _max_stretch
        _rel = source_clock_ms - con_vel
        _in_onset = _rel < _onset_tgt
        _onset_map = con_ms + (_rel / max(_onset_tgt, 1e-9)) * _onset_src
        _hold_tgt = max(_tgt_vowel - _onset_tgt, 1e-9)
        _hold_rel = np.maximum(_rel - _onset_tgt, 0.0)
        _drift = np.minimum(_hold_rel / _hold_tgt, 1.0)
        _hold_map = con_ms + _src_vowel * (0.45 + _drift * 0.52)
        vowel_mapped = np.where(_in_onset, _onset_map, _hold_map)

    t_mapped_ms = np.where(is_consonant, consonant_mapped, vowel_mapped)
    source_present = source_clock_ms >= 0.0
    # 长音时源时钟会被推到窗口末端；硬 clip 会让源位置“冻结”在最后一帧，
    # 听感就是卡一下。改成在最后一段内来回折返（ping-pong），位置连续不停顿。
    _limit = max(start_ms, end_ms - origin_frame_ms)
    _over = t_mapped_ms > _limit
    if np.any(_over):
        _span = max(1.0, _limit - con_ms)
        _x = t_mapped_ms[_over] - _limit
        t_mapped_ms = np.array(t_mapped_ms, copy=True)
        t_mapped_ms[_over] = _limit - np.abs(np.mod(_x, 2.0 * _span) - _span)
    t_mapped_ms = np.clip(t_mapped_ms, start_ms, _limit)
    return dict(start_ms=start_ms, con_ms=con_ms, end_ms=end_ms, vel=vel,
                con_vel=con_vel, con_output_ms=con_output_ms,
                total_span_ms=total_span_ms, target_n_frames=target_n_frames,
                t_mapped_ms=t_mapped_ms, source_present=source_present,
                scaling_ratio=scaling_ratio)
