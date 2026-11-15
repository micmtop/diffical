"""多音色混用处理器

实现不同音色（normal/roar/whisper/breath/power/soft）之间的平滑混合，包括：
- 子声库识别
- 跨音色 overlap 自适应调整
- 频谱包络渐变过渡
- 全局音色均衡
- 呼吸音特殊处理

与现有 Mel 拼接流程（mel_concatenate.py）兼容，Mel 频谱使用自然对数尺度。
"""

import os
import re
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)


TIMBRE_NORMAL = 'normal'
TIMBRE_ROAR = 'roar'
TIMBRE_WHISPER = 'whisper'
TIMBRE_BREATH = 'breath'
TIMBRE_POWER = 'power'
TIMBRE_SOFT = 'soft'

TIMBRE_TYPES = {
    TIMBRE_NORMAL: '正常音色（基础）',
    TIMBRE_ROAR: '咆哮/强力（高频丰富）',
    TIMBRE_WHISPER: '耳语/气声（噪声为主）',
    TIMBRE_BREATH: '呼吸音（纯噪声）',
    TIMBRE_POWER: '强力（与 roar 类似）',
    TIMBRE_SOFT: '柔软（高频减弱）',
}

_TIMBRE_KEYWORDS = {
    TIMBRE_ROAR: ['roar', 'growl', 'shout', 'scream', 'belt', 'distort'],
    TIMBRE_WHISPER: ['whisper', 'whispered', 'murmur', 'murmured', 'breathy'],
    TIMBRE_BREATH: ['breath', 'breathing', 'inhale', 'exhale', 'breathe'],
    TIMBRE_POWER: ['power', 'strong', 'loud', 'forte', 'ff', 'f'],
    TIMBRE_SOFT: ['soft', 'quiet', 'piano', 'pp', 'p', 'gentle', 'mild'],
}


class TimbreMixer:
    """多音色混用处理器

    提供跨音色拼接、过渡处理和全局均衡功能，
    确保不同子声库之间的音色平滑过渡。
    """

    def __init__(self, sample_rate: int = 44100, hop_size: int = 512):
        """初始化音色混合器

        Args:
            sample_rate: 采样率
            hop_size: 帧移（用于时间-帧数转换）
        """
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frame_ms = hop_size / sample_rate * 1000

    def detect_timbre_type(self, oto_file_path: str) -> str:
        """检测音色类型

        从路径或文件名中识别子声库类型，支持多种命名模式。

        Args:
            oto_file_path: oto 文件路径或文件名

        Returns:
            音色类型字符串（normal/roar/whisper/breath/power/soft）
        """
        if not oto_file_path:
            return TIMBRE_NORMAL

        path_lower = oto_file_path.lower()
        basename = os.path.basename(path_lower)
        dirname = os.path.dirname(path_lower)

        candidates = [basename, dirname, path_lower]

        for timbre_type, keywords in _TIMBRE_KEYWORDS.items():
            for candidate in candidates:
                for keyword in keywords:
                    pattern = r'(?:^|[_\\/\-\s])' + re.escape(keyword) + r'(?:$|[_\\/\-\s\.])'
                    if re.search(pattern, candidate):
                        logger.debug(f"检测到音色类型: {timbre_type} (路径: {oto_file_path}, 关键词: {keyword})")
                        return timbre_type

        return TIMBRE_NORMAL

    def compute_cross_timbre_overlap(self, prev_type: str, curr_type: str,
                                     base_overlap: int) -> int:
        """计算跨音色 overlap 长度

        根据前后音色类型调整 overlap 长度，差异越大的音色需要更长的过渡。

        系数规则：
        - 正常→正常：1.0x
        - 正常↔气声：1.5-2.0x
        - 正常↔咆哮：1.3-1.5x
        - 正常↔耳语：1.5x

        Args:
            prev_type: 前一个音素的音色类型
            curr_type: 当前音素的音色类型
            base_overlap: 基础 overlap 长度（帧数）

        Returns:
            调整后的 overlap 长度（帧数）
        """
        if prev_type == curr_type:
            return base_overlap

        prev_norm = self._normalize_timbre(prev_type)
        curr_norm = self._normalize_timbre(curr_type)

        scale = self._get_timbre_overlap_scale(prev_norm, curr_norm)

        adjusted = int(base_overlap * scale)
        adjusted = max(2, adjusted)

        logger.debug(
            f"跨音色 overlap 调整: {prev_type}→{curr_type}, "
            f"{base_overlap}帧 → {adjusted}帧 (x{scale:.2f})"
        )

        return adjusted

    def _normalize_timbre(self, timbre_type: str) -> str:
        """归一化音色类型，将相似音色归为一类"""
        if timbre_type in (TIMBRE_ROAR, TIMBRE_POWER):
            return TIMBRE_ROAR
        if timbre_type in (TIMBRE_WHISPER, TIMBRE_BREATH):
            return TIMBRE_BREATH
        return timbre_type

    def _get_timbre_overlap_scale(self, prev_norm: str, curr_norm: str) -> float:
        """获取两个归一化音色之间的 overlap 缩放系数"""
        pair = tuple(sorted([prev_norm, curr_norm]))

        scale_map = {
            (TIMBRE_BREATH, TIMBRE_BREATH): 1.0,
            (TIMBRE_BREATH, TIMBRE_NORMAL): 1.75,
            (TIMBRE_BREATH, TIMBRE_ROAR): 2.0,
            (TIMBRE_BREATH, TIMBRE_SOFT): 1.5,
            (TIMBRE_NORMAL, TIMBRE_NORMAL): 1.0,
            (TIMBRE_NORMAL, TIMBRE_ROAR): 1.4,
            (TIMBRE_NORMAL, TIMBRE_SOFT): 1.2,
            (TIMBRE_ROAR, TIMBRE_ROAR): 1.0,
            (TIMBRE_ROAR, TIMBRE_SOFT): 1.6,
            (TIMBRE_SOFT, TIMBRE_SOFT): 1.0,
        }

        return scale_map.get(pair, 1.5)

    def timbre_envelope_transition(self, mel_prev: np.ndarray, mel_curr: np.ndarray,
                                   overlap_frames: int, prev_type: str,
                                   curr_type: str) -> tuple:
        """音色包络过渡

        在 overlap 区域对两个音色的频谱包络进行渐变过渡，
        使用频率依赖的过渡曲线，确保音色平滑过渡。

        Args:
            mel_prev: 前一个音素的 Mel 频谱 (n_mels, T1) - 自然对数尺度
            mel_curr: 当前音素的 Mel 频谱 (n_mels, T2) - 自然对数尺度
            overlap_frames: overlap 长度（帧数）
            prev_type: 前一个音色类型
            curr_type: 当前音色类型

        Returns:
            (处理后的 mel_prev 尾部, 处理后的 mel_curr 头部)
            每个形状为 (n_mels, overlap_frames)
        """
        n_mels = mel_prev.shape[0]

        tail_prev = mel_prev[:, -overlap_frames:] if mel_prev.shape[1] >= overlap_frames else mel_prev
        head_curr = mel_curr[:, :overlap_frames] if mel_curr.shape[1] >= overlap_frames else mel_curr

        actual_len = min(tail_prev.shape[1], head_curr.shape[1])
        if actual_len < 2:
            return tail_prev, head_curr

        tail_prev = tail_prev[:, :actual_len]
        head_curr = head_curr[:, :actual_len]

        prev_env = self._smooth_spectrum_2d(tail_prev, kernel_size=7)
        curr_env = self._smooth_spectrum_2d(head_curr, kernel_size=7)

        prev_detail = tail_prev - prev_env
        curr_detail = head_curr - curr_env

        freq_curve = self._get_frequency_dependent_curve(n_mels, prev_type, curr_type)

        fade_base = np.linspace(0, 1, actual_len)
        fade_cosine = 0.5 * (1 - np.cos(fade_base * np.pi))

        transition_env = np.zeros_like(prev_env)
        transition_detail = np.zeros_like(prev_detail)

        for m in range(n_mels):
            freq_weight = freq_curve[m]
            fade = fade_cosine * freq_weight + fade_base * (1 - freq_weight)

            transition_env[m, :] = prev_env[m, :] * (1 - fade) + curr_env[m, :] * fade
            transition_detail[m, :] = prev_detail[m, :] * (1 - fade) + curr_detail[m, :] * fade

        processed_tail = transition_env + transition_detail
        processed_head = transition_env + transition_detail

        return processed_tail, processed_head

    def _get_frequency_dependent_curve(self, n_mels: int,
                                       prev_type: str, curr_type: str) -> np.ndarray:
        """获取频率依赖的过渡曲线权重

        不同音色之间的过渡在不同频段有不同的过渡速率：
        - 咆哮↔正常：高频过渡更慢（保留咆哮高频特征的渐变）
        - 气声↔正常：全频带噪声成分渐变
        - 柔软↔正常：高频渐进过渡
        """
        freq_ratio = np.linspace(0, 1, n_mels)

        prev_norm = self._normalize_timbre(prev_type)
        curr_norm = self._normalize_timbre(curr_type)

        has_roar = prev_norm == TIMBRE_ROAR or curr_norm == TIMBRE_ROAR
        has_breath = prev_norm == TIMBRE_BREATH or curr_norm == TIMBRE_BREATH
        has_soft = prev_type == TIMBRE_SOFT or curr_type == TIMBRE_SOFT

        if has_roar and not has_breath:
            curve = 0.6 + 0.4 * freq_ratio
        elif has_breath:
            curve = 0.8 + 0.2 * (1 - np.abs(freq_ratio - 0.5) * 2)
        elif has_soft:
            curve = 0.7 + 0.3 * freq_ratio
        else:
            curve = 0.5 + 0.5 * freq_ratio

        return curve

    def global_timbre_equalization(self, mel_list: list, phone_timbre_list: list) -> list:
        """全局音色均衡

        对整句应用全局音色均衡，确保不同音色片段的整体听感和谐。
        基于出现最多的音色作为参考，对其他音色进行轻微的均衡调整。

        Args:
            mel_list: Mel 频谱列表，每个 (n_mels, T_i) - 自然对数尺度
            phone_timbre_list: 每个音素对应的音色类型列表

        Returns:
            均衡后的 Mel 列表
        """
        if not mel_list or len(mel_list) <= 1:
            return mel_list

        ref_type = self._get_reference_timbre(phone_timbre_list)

        ref_mel_idx = None
        for i, t in enumerate(phone_timbre_list):
            if t == ref_type and mel_list[i].shape[1] > 5:
                ref_mel_idx = i
                break

        if ref_mel_idx is None:
            ref_mel_idx = 0

        ref_mel = mel_list[ref_mel_idx]
        ref_env_avg = self._compute_avg_envelope(ref_mel)

        result_list = []
        for i, (mel, timbre_type) in enumerate(zip(mel_list, phone_timbre_list)):
            if timbre_type == ref_type:
                result_list.append(mel.copy())
                continue

            strength = self._get_equalization_strength(ref_type, timbre_type)
            if strength <= 0:
                result_list.append(mel.copy())
                continue

            curr_env_avg = self._compute_avg_envelope(mel)
            eq_curve = self._compute_equalization_curve(ref_env_avg, curr_env_avg, strength)

            mel_eq = mel + eq_curve.reshape(-1, 1)
            result_list.append(mel_eq.astype(mel.dtype))

            logger.debug(
                f"全局均衡: 音素{i} ({timbre_type}→{ref_type}), 强度={strength:.2f}"
            )

        return result_list

    def _get_reference_timbre(self, timbre_list: list) -> str:
        """获取参考音色类型（出现次数最多的音色）"""
        from collections import Counter
        counter = Counter(timbre_list)
        return counter.most_common(1)[0][0]

    def _compute_avg_envelope(self, mel: np.ndarray) -> np.ndarray:
        """计算平均频谱包络"""
        mel_linear = np.exp(mel)
        avg_linear = np.mean(mel_linear, axis=1)
        avg_smoothed = self._smooth_spectrum(avg_linear, kernel_size=7)
        return np.log(np.clip(avg_smoothed, 1e-10, None))

    def _compute_equalization_curve(self, ref_env: np.ndarray, curr_env: np.ndarray,
                                    strength: float) -> np.ndarray:
        """计算均衡曲线"""
        diff = ref_env - curr_env

        diff_smoothed = self._smooth_spectrum(diff, kernel_size=9)

        diff_clipped = np.clip(diff_smoothed, -3.0, 3.0)

        curve = diff_clipped * strength

        return curve

    def _get_equalization_strength(self, ref_type: str, target_type: str) -> float:
        """获取均衡强度（不同音色对之间的均衡强度不同）"""
        ref_norm = self._normalize_timbre(ref_type)
        target_norm = self._normalize_timbre(target_type)

        if ref_norm == target_norm:
            return 0.0

        pair = tuple(sorted([ref_norm, target_norm]))

        strength_map = {
            (TIMBRE_BREATH, TIMBRE_NORMAL): 0.1,
            (TIMBRE_BREATH, TIMBRE_ROAR): 0.05,
            (TIMBRE_BREATH, TIMBRE_SOFT): 0.08,
            (TIMBRE_NORMAL, TIMBRE_ROAR): 0.15,
            (TIMBRE_NORMAL, TIMBRE_SOFT): 0.2,
            (TIMBRE_ROAR, TIMBRE_SOFT): 0.1,
        }

        return strength_map.get(pair, 0.1)

    def breath_sound_processing(self, wave: np.ndarray, sr: int,
                                intensity: float = 0.5) -> np.ndarray:
        """呼吸音特殊处理

        对呼吸音进行特殊处理，包括：
        - 噪声匹配（使呼吸音噪声更自然）
        - 更长的淡入淡出（避免呼吸音突兀）
        - 保持呼吸音的自然感

        Args:
            wave: 输入波形
            sr: 采样率
            intensity: 处理强度 0~1

        Returns:
            处理后的波形
        """
        if intensity <= 0 or len(wave) < 100:
            return wave

        result = wave.copy()

        fade_ms = int(0.05 * sr)
        fade_len = min(fade_ms, len(result) // 4)
        if fade_len > 0:
            fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len)))
            fade_out = np.flip(fade_in)
            result[:fade_len] *= fade_in
            result[-fade_len:] *= fade_out

        noise = np.random.randn(len(result)) * 0.01 * intensity
        noise_env = self._compute_noise_envelope(result, sr)
        noise = noise * noise_env

        result = result + noise.astype(result.dtype)

        max_val = np.max(np.abs(result))
        if max_val > 0.98:
            result = result * (0.98 / max_val)

        return result

    def _compute_noise_envelope(self, wave: np.ndarray, sr: int) -> np.ndarray:
        """计算噪声包络（基于信号能量的平滑包络）"""
        wave_abs = np.abs(wave)

        win_len = max(1, int(0.02 * sr))
        kernel = np.hanning(win_len * 2 + 1)
        kernel = kernel / kernel.sum()

        envelope = np.convolve(wave_abs, kernel, mode='same')

        envelope = envelope / (np.max(envelope) + 1e-10)

        return envelope

    def mix_timbres(self, phone_features_list: list,
                    phone_timbre_list: list,
                    base_overlap_frames: int = 6) -> np.ndarray:
        """多音色混合主入口

        对多个不同音色的音素进行跨音色拼接和过渡处理。

        Args:
            phone_features_list: 音素特征列表，每个是 Mel 频谱 (n_mels, T_i) - 自然对数尺度
            phone_timbre_list: 每个音素对应的音色类型列表
            base_overlap_frames: 基础 overlap 帧数

        Returns:
            拼接后的完整 Mel 频谱 (n_mels, total_T) - 自然对数尺度
        """
        if not phone_features_list:
            raise ValueError("phone_features_list 为空")

        if len(phone_features_list) != len(phone_timbre_list):
            raise ValueError("phone_features_list 和 phone_timbre_list 长度不一致")

        if len(phone_features_list) == 1:
            return phone_features_list[0].copy()

        logger.info(f"开始多音色混合，共 {len(phone_features_list)} 个音素")

        mel_list = [m.copy() for m in phone_features_list]

        mel_list = self.global_timbre_equalization(mel_list, phone_timbre_list)

        result = mel_list[0]

        for i in range(1, len(mel_list)):
            prev_type = phone_timbre_list[i - 1]
            curr_type = phone_timbre_list[i]
            next_mel = mel_list[i]

            overlap = self.compute_cross_timbre_overlap(
                prev_type, curr_type, base_overlap_frames
            )
            overlap = max(2, min(overlap, result.shape[1] // 3, next_mel.shape[1] // 3))

            if prev_type != curr_type:
                processed_tail, processed_head = self.timbre_envelope_transition(
                    result, next_mel, overlap, prev_type, curr_type
                )
                result[:, -overlap:] = processed_tail
                next_mel[:, :overlap] = processed_head

            result = self._crossfade_mels(result, next_mel, overlap)

            logger.debug(
                f"拼接音素 {i-1}→{i}: {prev_type}→{curr_type}, "
                f"overlap={overlap}帧"
            )

        logger.info(f"多音色混合完成，总时长 {result.shape[1]} 帧")

        return result

    def _crossfade_mels(self, mel_a: np.ndarray, mel_b: np.ndarray,
                        fade_len: int) -> np.ndarray:
        """两个 Mel 频谱的交叉淡入淡出（使用余弦曲线，在线性域操作）

        与 mel_concatenate.py 中的 crossfade 保持兼容。

        Args:
            mel_a: 前一个片段 (n_mels, T1) - 自然对数尺度
            mel_b: 后一个片段 (n_mels, T2) - 自然对数尺度
            fade_len: 淡入淡出长度（帧数）

        Returns:
            拼接后的 Mel (n_mels, T1 + T2 - fade_len) - 自然对数尺度
        """
        n_mels = mel_a.shape[0]
        len_a = mel_a.shape[1]
        len_b = mel_b.shape[1]

        if fade_len <= 0:
            return np.concatenate([mel_a, mel_b], axis=1)

        fade_len = min(fade_len, len_a, len_b)

        fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_len)))
        fade_in = 1 - fade_out

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

    def _smooth_spectrum(self, spec: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """平滑频谱包络（1D）"""
        kernel = np.hanning(kernel_size * 2 + 1)
        kernel = kernel / kernel.sum()
        return np.convolve(spec, kernel, mode='same')

    def _smooth_spectrum_2d(self, spec: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """对 2D 频谱的频率轴做平滑"""
        result = np.zeros_like(spec)
        kernel = np.hanning(kernel_size * 2 + 1)
        kernel = kernel / kernel.sum()

        for t in range(spec.shape[1]):
            result[:, t] = np.convolve(spec[:, t], kernel, mode='same')

        return result
