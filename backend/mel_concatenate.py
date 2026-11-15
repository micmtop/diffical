"""Mel 频谱拼接

实现音素之间的 Mel 频谱平滑拼接，包括：
- 交叉淡入淡出
- 频谱包络匹配（改进版）
- 共振峰约束与平滑过渡
- 频率依赖的 crossfade
- 能量平滑过渡
- 相位连续性优化
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)


def crossfade(mel_a: np.ndarray, mel_b: np.ndarray,
              fade_len: int, mode: str = 'cosine') -> np.ndarray:
    """两个 Mel 频谱片段的交叉淡入淡出（在线性域操作，避免气泡音）

    Args:
        mel_a: 前一个片段 (n_mels, T1) - 自然对数尺度
        mel_b: 后一个片段 (n_mels, T2) - 自然对数尺度
        fade_len: 淡入淡出长度（帧数）
        mode: 淡入淡出模式 ('linear', 'cosine', 'gaussian')

    Returns:
        拼接后的 Mel (n_mels, T1 + T2 - fade_len) - 自然对数尺度
    """
    n_mels = mel_a.shape[0]
    len_a = mel_a.shape[1]
    len_b = mel_b.shape[1]

    if fade_len <= 0:
        return np.concatenate([mel_a, mel_b], axis=1)

    fade_len = min(fade_len, len_a, len_b)

    if mode == 'cosine':
        fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_len)))
        fade_in = 1 - fade_out
    elif mode == 'gaussian':
        t = np.linspace(0, 3, fade_len)
        fade_out = np.exp(-0.5 * t**2)
        fade_in = np.flip(fade_out)
    else:
        fade_out = np.linspace(1, 0, fade_len)
        fade_in = np.linspace(0, 1, fade_len)

    fade_out = fade_out.reshape(1, -1)
    fade_in = fade_in.reshape(1, -1)

    result = np.zeros((n_mels, len_a + len_b - fade_len), dtype=mel_a.dtype)

    result[:, :len_a - fade_len] = mel_a[:, :len_a - fade_len]

    tail_linear = np.exp(mel_a[:, len_a - fade_len:])
    head_linear = np.exp(mel_b[:, :fade_len])
    crossfade_linear = tail_linear * fade_out + head_linear * fade_in
    result[:, len_a - fade_len:len_a] = np.log(np.clip(crossfade_linear, 1e-10, None))

    result[:, len_a:] = mel_b[:, fade_len:]

    return result


def frequency_dependent_crossfade(mel_a: np.ndarray, mel_b: np.ndarray,
                                  fade_len: int, n_mels: int) -> np.ndarray:
    """频率依赖的交叉淡入淡出

    低频段使用较长的 crossfade，高频段使用较短的 crossfade，
    这样可以保留高频瞬态同时确保低频连续性。

    Args:
        mel_a: 前一个片段 (n_mels, T1) - 自然对数尺度
        mel_b: 后一个片段 (n_mels, T2) - 自然对数尺度
        fade_len: 基础淡入淡出长度（帧数）
        n_mels: Mel 频段数

    Returns:
        拼接后的 Mel (n_mels, T1 + T2 - fade_len) - 自然对数尺度
    """
    len_a = mel_a.shape[1]
    len_b = mel_b.shape[1]

    if fade_len <= 0:
        return np.concatenate([mel_a, mel_b], axis=1)

    fade_len = min(fade_len, len_a, len_b)

    freq_ratio = np.linspace(0, 1, n_mels).reshape(-1, 1)
    
    low_freq_fade = fade_len
    high_freq_fade = max(2, int(fade_len * 0.4))
    
    freq_fade_len = low_freq_fade - (low_freq_fade - high_freq_fade) * freq_ratio
    freq_fade_len = np.maximum(2, freq_fade_len).astype(int)

    result = np.zeros((n_mels, len_a + len_b - fade_len), dtype=mel_a.dtype)
    result[:, :len_a - fade_len] = mel_a[:, :len_a - fade_len]
    result[:, len_a:] = mel_b[:, fade_len:]

    for m in range(n_mels):
        fl = freq_fade_len[m, 0]
        if fl > fade_len:
            fl = fade_len
        
        if fl < 2:
            result[m, len_a - fade_len:len_a] = np.log(
                np.clip(np.exp(mel_a[m, len_a - fade_len:]) + np.exp(mel_b[m, :fade_len]), 1e-10, None)
            )
        else:
            fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fl)))
            fade_in = 1 - fade_out
            
            tail_linear = np.exp(mel_a[m, len_a - fl:len_a])
            head_linear = np.exp(mel_b[m, :fl])
            
            crossfade_linear = tail_linear * fade_out + head_linear * fade_in
            
            pad_start = fade_len - fl
            if pad_start > 0:
                result[m, len_a - fade_len:len_a - fl] = mel_a[m, len_a - fade_len:len_a - fl]
            
            result[m, len_a - fl:len_a] = np.log(np.clip(crossfade_linear, 1e-10, None))

    return result


def spectral_envelope_match(source_mel: np.ndarray, target_mel: np.ndarray,
                            n_formants: int = 3, sr: int = 44100,
                            n_fft: int = 2048, strength: float = 0.8) -> np.ndarray:
    """改进的共振峰匹配 - 将 source 的频谱包络对齐到 target

    使用多层平滑和渐进匹配，确保自然过渡。

    Args:
        source_mel: 源 Mel (n_mels, T)
        target_mel: 目标 Mel (n_mels, T)
        n_formants: 匹配的共振峰数量
        sr: 采样率
        n_fft: FFT 大小
        strength: 匹配强度 0~1

    Returns:
        匹配后的 Mel (n_mels, T)
    """
    if strength <= 0:
        return source_mel.copy()

    n_mels, T = source_mel.shape
    result = np.zeros_like(source_mel)

    for t in range(T):
        src_frame = source_mel[:, t]
        tgt_frame = target_mel[:, t]

        try:
            src_env = _smooth_spectrum(src_frame, n_mels, kernel_size=7)
            tgt_env = _smooth_spectrum(tgt_frame, n_mels, kernel_size=7)

            src_detail = src_frame - src_env
            tgt_detail = tgt_frame - tgt_env

            gain = np.where(src_env > 1e-6, tgt_env / src_env, 1.0)
            gain = np.clip(gain, 0.5, 2.0)

            smoothed_gain = _smooth_spectrum(gain, n_mels, kernel_size=5)

            matched_env = src_env * smoothed_gain
            matched_detail = src_detail + (tgt_detail - src_detail) * 0.3

            result[:, t] = matched_env + matched_detail
        except Exception:
            result[:, t] = src_frame

    return result


def formant_smooth_transition(mel_a: np.ndarray, mel_b: np.ndarray,
                              overlap_len: int) -> tuple:
    """共振峰平滑过渡（passthrough）

    原实现将 tail/head 都替换为同一个 transition mel（env+detail 的 crossfade），
    导致下游 frequency_dependent_crossfade 收到两个相同输入，per-band 交叉淡入淡出
    完全失效（高频瞬态保留逻辑形同虚设），边界谱通量偏高。
    参考 hifisampler：不做额外共振峰整形，直接交由 crossfade 处理。

    Args:
        mel_a: 前一个音素的 Mel (n_mels, T1)
        mel_b: 后一个音素的 Mel (n_mels, T2)
        overlap_len: overlap 长度（帧数）

    Returns:
        (mel_a 尾部 overlap_len 帧, mel_b 头部 overlap_len 帧) 原样返回
    """
    tail_a = mel_a[:, -overlap_len:] if mel_a.shape[1] >= overlap_len else mel_a
    head_b = mel_b[:, :overlap_len] if mel_b.shape[1] >= overlap_len else mel_b

    actual_len = min(tail_a.shape[1], head_b.shape[1])
    return tail_a[:, :actual_len], head_b[:, :actual_len]


def phase_lock(mel_a: np.ndarray, mel_b: np.ndarray,
               overlap_len: int) -> np.ndarray:
    """相位锁定优化

    通过频谱包络的相位一致性调整，减少"电子味"。
    在 overlap 区域确保两个音素的频谱包络相位连续。

    Args:
        mel_a: 前一个音素的 Mel (n_mels, T1)
        mel_b: 后一个音素的 Mel (n_mels, T2)
        overlap_len: overlap 长度（帧数）

    Returns:
        相位调整后的 mel_b 头部
    """
    n_mels = mel_a.shape[0]

    tail_a = mel_a[:, -overlap_len:] if mel_a.shape[1] >= overlap_len else mel_a
    head_b = mel_b[:, :overlap_len] if mel_b.shape[1] >= overlap_len else mel_b

    actual_len = min(tail_a.shape[1], head_b.shape[1])
    if actual_len < 2:
        return head_b

    tail_a = tail_a[:, :actual_len]
    head_b = head_b[:, :actual_len]

    tail_linear = np.exp(tail_a)
    head_linear = np.exp(head_b)

    tail_phase = np.angle(np.fft.rfft(tail_linear, axis=1))
    head_phase = np.angle(np.fft.rfft(head_linear, axis=1))

    phase_diff = head_phase[:, -1:] - tail_phase[:, -1:]
    phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi

    phase_correction = phase_diff * np.linspace(0, 1, actual_len).reshape(1, -1)

    head_corrected = np.fft.irfft(
        np.abs(np.fft.rfft(head_linear, axis=1)) * np.exp(1j * (head_phase - phase_correction)),
        n=actual_len,
        axis=1
    )

    return np.log(np.clip(head_corrected, 1e-10, None))


def compute_dynamic_overlap(phone_a: dict, phone_b: dict,
                            language: str, vb_type: str) -> int:
    """动态计算 overlap 长度

    根据前后音素类型、语言和声库类型自适应计算 overlap 长度。

    Args:
        phone_a: 前一个音素信息（包含 phoneme, is_vowel 等）
        phone_b: 后一个音素信息
        language: 语言 ('ZH', 'EN', 'JA', 'KO')
        vb_type: 声库类型 ('mono' 或 'vowel_connected')

    Returns:
        overlap 长度（帧数）
    """
    lang_params = {
        'ZH': {'min_ms': 60, 'max_ms': 180, 'vowel_vowel': 80, 'consonant_vowel': 100},
        'EN': {'min_ms': 80, 'max_ms': 200, 'vowel_vowel': 100, 'consonant_vowel': 120},
        'JA': {'min_ms': 50, 'max_ms': 150, 'vowel_vowel': 60, 'consonant_vowel': 80},
        'KO': {'min_ms': 60, 'max_ms': 160, 'vowel_vowel': 70, 'consonant_vowel': 90},
    }

    params = lang_params.get(language.upper(), lang_params['JA'])

    is_a_vowel = phone_a.get('is_vowel', False)
    is_b_vowel = phone_b.get('is_vowel', False)

    if is_a_vowel and is_b_vowel:
        base_ms = params['vowel_vowel']
    elif not is_a_vowel and is_b_vowel:
        base_ms = params['consonant_vowel']
    else:
        base_ms = (params['min_ms'] + params['max_ms']) // 2

    if vb_type == 'vowel_connected':
        base_ms = int(base_ms * 0.8)

    overlap_ms = base_ms

    overlap_ms = max(params['min_ms'], min(params['max_ms'], overlap_ms))

    return int(overlap_ms / 10)


def spectral_flux(mel: np.ndarray) -> np.ndarray:
    """计算频谱通量（用于评估频谱不连续性）

    Args:
        mel: Mel 频谱 (n_mels, T)

    Returns:
        频谱通量数组 (T-1,)
    """
    mel_linear = np.exp(mel)
    flux = np.sqrt(np.sum((mel_linear[:, 1:] - mel_linear[:, :-1])**2, axis=0))
    return flux


def _smooth_spectrum(spec: np.ndarray, n_bins: int,
                     kernel_size: int = 5) -> np.ndarray:
    """平滑频谱包络"""
    kernel = np.hanning(kernel_size * 2 + 1)
    kernel = kernel / kernel.sum()
    return np.convolve(spec, kernel, mode='same')


def _smooth_spectrum_2d(spec: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """对 2D 频谱的频率轴做平滑"""
    result = np.zeros_like(spec)
    kernel = np.hanning(kernel_size * 2 + 1)
    kernel = kernel / kernel.sum()

    for t in range(spec.shape[1]):
        result[:, t] = np.convolve(spec[:, t], kernel, mode='same')

    return result


def formant_constrain(mel: np.ndarray, ref_mel: np.ndarray,
                      strength: float = 0.5) -> np.ndarray:
    """共振峰约束

    确保生成的 Mel 频谱的共振峰结构不偏离参考（原始音色）太多，
    防止音色泄漏。

    Args:
        mel: 待约束的 Mel (n_mels, T)
        ref_mel: 参考 Mel (n_mels, T)
        strength: 约束强度 0~1

    Returns:
        约束后的 Mel
    """
    if strength <= 0:
        return mel

    mel_env = _smooth_spectrum_2d(mel, kernel_size=5)
    ref_env = _smooth_spectrum_2d(ref_mel, kernel_size=5)

    diff = mel_env - ref_env

    constrained_env = mel_env - diff * strength

    detail = mel - mel_env
    result = constrained_env + detail

    return result.astype(mel.dtype)


def energy_smooth(mel: np.ndarray, smooth_win: int = 5) -> np.ndarray:
    """能量包络平滑

    沿时间轴平滑 Mel 频谱的总能量，减少能量突变。
    在线性域计算能量，避免对数域能量计算错误。

    Args:
        mel: Mel 频谱 (n_mels, T) - 自然对数尺度
        smooth_win: 平滑窗口大小（帧数）

    Returns:
        平滑后的 Mel - 自然对数尺度
    """
    mel_linear = np.exp(mel)

    energy = np.sqrt(np.mean(mel_linear ** 2, axis=0))

    kernel = np.hanning(smooth_win * 2 + 1)
    kernel = kernel / kernel.sum()

    # 防御：np.convolve(mode='same') 当输入比 kernel 短时返回 max(len(energy), len(kernel))，
    # 导致后续 np.where 广播失败（如 energy 长度 9 < kernel 长度 21 → 返回 21）。
    # 输入过短时跳过平滑，直接返回原始 mel。
    if energy.shape[0] < kernel.shape[0]:
        return mel.astype(mel.dtype)

    energy_smooth = np.convolve(energy, kernel, mode='same')

    energy_smooth = np.where(energy > 1e-10, energy_smooth, energy)

    gain = np.where(energy > 1e-10, energy_smooth / energy, 1.0)
    mel_linear_smooth = mel_linear * gain.reshape(1, -1)

    return np.log(np.clip(mel_linear_smooth, 1e-10, None)).astype(mel.dtype)


def pre_echo_suppression(mel: np.ndarray, threshold_db: float = -40.0,
                         attack_frames: int = 3) -> np.ndarray:
    """时域预回声消除

    检测能量突然上升的位置，在上升前的几帧内逐渐衰减能量，
    减少快速能量变化产生的预回声伪影。

    Args:
        mel: Mel 频谱 (n_mels, T) - 自然对数尺度
        threshold_db: 能量上升阈值（dB）
        attack_frames: 攻击前衰减帧数

    Returns:
        处理后的 Mel - 自然对数尺度
    """
    mel_linear = np.exp(mel)
    n_mels, n_frames = mel_linear.shape

    energy = np.sqrt(np.mean(mel_linear ** 2, axis=0))
    energy_db = 20 * np.log10(np.clip(energy, 1e-10, None))

    energy_diff = np.diff(energy_db, prepend=energy_db[0])

    attack_points = np.where(energy_diff > threshold_db)[0]

    result = mel_linear.copy()

    for attack in attack_points:
        start = max(0, attack - attack_frames)
        if start >= attack:
            continue

        for j in range(start, attack):
            fade_pos = (j - start) / max(1, (attack - start))
            gain = fade_pos
            result[:, j] *= gain

    return np.log(np.clip(result, 1e-10, None)).astype(mel.dtype)


def concatenate_mels(mel_list: list, crossfade_frames: int = 6,
                     match_formants: bool = True,
                     smooth_energy: bool = True,
                     overlap_frames_list: list = None,
                     phone_info_list: list = None,
                     language: str = 'JA',
                     vb_type: str = 'mono') -> np.ndarray:
    """拼接多个 Mel 频谱片段（改进版）

    Args:
        mel_list: Mel 列表，每个 (n_mels, T_i)
        crossfade_frames: 默认交叉淡入淡出帧数
        match_formants: 是否进行共振峰匹配
        smooth_energy: 是否平滑能量
        overlap_frames_list: 每个边界的 overlap 帧数列表
        phone_info_list: 音素信息列表（用于动态 overlap 计算）
        language: 目标语言
        vb_type: 声库类型

    Returns:
        拼接后的 Mel (n_mels, total_T)
    """
    if not mel_list:
        raise ValueError("mel_list 为空")

    if len(mel_list) == 1:
        return mel_list[0].copy()

    result = mel_list[0].copy()

    for i in range(1, len(mel_list)):
        next_mel = mel_list[i].copy()

        if overlap_frames_list and i - 1 < len(overlap_frames_list):
            cf = overlap_frames_list[i - 1]
            if cf is not None and cf > 0:
                cf = min(cf, result.shape[1] // 3, next_mel.shape[1] // 3, crossfade_frames * 3)
            else:
                cf = crossfade_frames
        else:
            if phone_info_list and i - 1 < len(phone_info_list) and i < len(phone_info_list):
                cf = compute_dynamic_overlap(
                    phone_info_list[i-1], phone_info_list[i], language, vb_type
                )
            else:
                cf = crossfade_frames

        cf = max(3, min(cf, result.shape[1] - 3, next_mel.shape[1] - 3))

        if result.shape[1] >= cf and next_mel.shape[1] >= cf:
            tail_linear = np.exp(result[:, -cf:])
            head_linear = np.exp(next_mel[:, :cf])
            tail_energy = np.sqrt(np.mean(tail_linear ** 2))
            head_energy = np.sqrt(np.mean(head_linear ** 2))
            if head_energy > 1e-10:
                energy_ratio = tail_energy / head_energy
                energy_ratio = np.clip(energy_ratio, 0.5, 2.0)
                next_mel[:, :cf] = next_mel[:, :cf] + np.log(energy_ratio)

        if match_formants:
            processed_tail, processed_head = formant_smooth_transition(result, next_mel, cf)
            result[:, -cf:] = processed_tail
            next_mel[:, :cf] = processed_head

        result = frequency_dependent_crossfade(result, next_mel, cf, result.shape[0])

    if smooth_energy:
        result = energy_smooth(result)

    return result