"""音色温暖度处理器

实现精细的多频段 EQ 调整：
- 低频提升：增加温暖感
- 中频塑形：调整音色特性
- 高频衰减：减少尖锐感
- 谐波压缩：控制动态范围
"""

import numpy as np


def apply_warmth(audio: np.ndarray, warmth_level: float,
                 sample_rate: int) -> np.ndarray:
    """应用温暖度 EQ

    Args:
        audio: 输入音频（通常是谐波分量）(n_samples,)
        warmth_level: 温暖度值 (0-100)
        sample_rate: 采样率

    Returns:
        处理后的音频
    """
    from scipy.signal import butter, filtfilt

    if len(audio) == 0:
        return audio.copy()

    warmth_factor = np.clip(warmth_level / 100, 0, 1)
    if warmth_factor < 0.01:
        return audio.copy()

    nyquist = sample_rate / 2

    low_center = 600
    mid_center = 1500
    high_center = 4000

    low_gain = 1 + warmth_factor * 0.6
    mid_gain = 1 + warmth_factor * 0.1
    high_gain = 1 - warmth_factor * 0.4

    low_q = 1.0
    mid_q = 1.2
    high_q = 0.8

    b_low, a_low = _create_peaking_filter(low_center, low_q, low_gain, nyquist)
    b_mid, a_mid = _create_peaking_filter(mid_center, mid_q, mid_gain, nyquist)
    b_high, a_high = _create_peaking_filter(high_center, high_q, high_gain, nyquist)

    processed = filtfilt(b_low, a_low, audio)
    processed = filtfilt(b_mid, a_mid, processed)
    processed = filtfilt(b_high, a_high, processed)

    blend = warmth_factor
    result = audio * (1 - blend) + processed * blend

    return result.astype(np.float32)


def apply_harmonic_compressor(audio: np.ndarray, compression_level: float,
                              sample_rate: int) -> np.ndarray:
    """应用谐波压缩（动态范围压缩）

    修复（原为 bug）：旧实现用固定绝对阈值 0.35 压缩 |audio|，而 hnsep 分离后的
    谐波分量峰值通常只有 ~0.1（远低于 0.35），导致压缩永远不触发 → HCMP 曲线无效。
    参考同类实现：阈值取相对信号峰值（peak - 22dB），压缩比随 HCMP 线性
    变化（1:1 → 20:1），并用 RMS 包络 + 软膝 + attack/release 平滑，输出按输入
    RMS 归一化（只压动态、不改变音量）。

    Args:
        audio: 输入音频（通常是谐波分量）(n_samples,)
        compression_level: 压缩值 (0-100)，0=无效果
        sample_rate: 采样率

    Returns:
        处理后的音频
    """
    if len(audio) == 0:
        return audio.copy()

    comp_factor = np.clip(compression_level / 100, 0, 1)
    if comp_factor < 0.01:
        return audio.copy()

    rms_in = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-12:
        return audio.copy()

    # 峰值相对阈值：peak - 22dB（对齐 同类引擎），ratio 1:1 → 20:1
    threshold_db = 20.0 * np.log10(peak + 1e-12) - 22.0
    ratio = 1.0 + comp_factor * 19.0

    x = _soft_knee_compressor(audio, threshold_db=threshold_db, ratio=ratio,
                              attack_ms=1.0, release_ms=60.0, sr=sample_rate)

    # 能量归一化：输出 RMS 回到输入 RMS（只压动态，音量不变）
    rms_out = np.sqrt(np.mean(x ** 2))
    if rms_out > 1e-12 and rms_in > 1e-12:
        gain = np.clip(rms_in / rms_out, 0.01, 100.0)
        x = x * gain

    # 软限幅
    out_peak = float(np.max(np.abs(x)))
    if out_peak > 0.99:
        x = x * (0.99 / out_peak)

    return x.astype(np.float32)


def _soft_knee_compressor(signal: np.ndarray, threshold_db: float, ratio: float,
                          knee_db: float = 6.0, attack_ms: float = 5.0,
                          release_ms: float = 60.0, sr: int = 44100) -> np.ndarray:
    """纯 numpy 软膝压缩器（RMS 包络 + attack/release 平滑）。

    参考同类实现 的 _numba_ops.soft_knee_compressor：
    - 10ms 滑动窗口 RMS 包络
    - 6dB 软膝：低于阈值无增益，高于阈值按 1/ratio 压缩，膝内 smoothstep 过渡
    - 增益用一阶 IIR 平滑（attack 快、release 慢），避免抽动
    """
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=np.float32)

    # ── RMS 包络（10ms 滑动窗口，反射边界） ──
    rms_win = max(1, int(sr * 0.01))
    half = rms_win // 2
    x2 = x * x
    padded = np.empty(n + 2 * half, dtype=np.float64)
    padded[:half] = x2[half - 1::-1]
    padded[half:half + n] = x2
    padded[half + n:] = x2[n - 1:n - 1 - half:-1]
    cum = np.concatenate(([0.0], np.cumsum(padded)))
    sum_sq = cum[rms_win:] - cum[:len(cum) - rms_win]
    rms = np.sqrt(np.maximum(sum_sq, 1e-12) / rms_win)

    # ── 软膝增益（dB） ──
    in_db = 20.0 * np.log10(rms + 1e-12)
    lo = threshold_db - knee_db / 2
    hi = threshold_db + knee_db / 2
    gain_db = np.zeros(n, dtype=np.float64)

    above = in_db > hi
    knee = (in_db >= lo) & (in_db <= hi)
    # 高于阈值部分: y = th + (in - th)/ratio → 增益 = (th - in) * (1 - 1/ratio)
    gain_db[above] = (threshold_db - in_db[above]) * (1.0 - 1.0 / ratio)
    # 膝内: smoothstep 从 0 到 hi 处增益
    k_pos = (in_db[knee] - lo) / knee_db
    g_hi = (threshold_db - hi) * (1.0 - 1.0 / ratio)
    gain_db[knee] = g_hi * (k_pos * k_pos * (3.0 - 2.0 * k_pos))

    # ── attack/release 平滑（一阶 IIR；attack 快、release 慢） ──
    attack = 1.0 - np.exp(-1.0 / (max(0.5, attack_ms) / 1000.0 * sr))
    release = 1.0 - np.exp(-1.0 / (max(1.0, release_ms) / 1000.0 * sr))
    smoothed = np.empty(n, dtype=np.float64)
    g = 0.0
    for i in range(n):
        tau = attack if gain_db[i] < g else release
        g += (gain_db[i] - g) * tau
        smoothed[i] = g

    return (x * np.power(10.0, smoothed / 20.0)).astype(np.float32)


def _create_peaking_filter(freq: float, q: float, gain: float, nyquist: float):
    """创建峰值滤波器（二阶 Butterworth 变体）"""
    from scipy.signal import butter

    w0 = freq / nyquist
    alpha = np.sin(w0) / (2 * q)

    if gain == 1.0:
        return butter(1, w0, btype='low')

    A = np.sqrt(gain)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = [b0, b1, b2]
    a = [a0, a1, a2]

    return b, a