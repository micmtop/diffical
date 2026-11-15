"""音素时长与过渡优化模块

实现音素级别的时长调整和过渡优化，使合成语音更自然流畅。

主要功能：
1. 元音自然延长 - 根据上下文和节奏自动调整元音时长
2. 辅音过渡优化 - 根据前后音素调整辅音强度和时长
3. 音节级节奏优化 - 使整体节奏更自然
4. 用户手动调整 - 支持按音素调整时长比例

技术实现：
- 相位声码器 (Phase Vocoder) 进行时间拉伸
- SOLA (Synchronous Overlap-Add) 算法备选
- 与 Mel 频谱拼接流程兼容
- 调整范围限制在 0.8-1.2 倍避免不自然
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class PhonemeTimingOptimizer:
    """音素时长与过渡优化器

    对音素序列进行时长优化和过渡平滑，提升合成语音的自然度。

    Attributes:
        sample_rate: 采样率，默认 44100Hz
        n_fft: FFT 大小
        hop_length: 帧移（样本数）
        min_duration_ratio: 最小时长比例
        max_duration_ratio: 最大时长比例
    """

    VOWEL_SET = {
        'a', 'i', 'u', 'e', 'o',
        'あ', 'い', 'う', 'え', 'お',
        'AA', 'AE', 'AH', 'AO', 'EH', 'ER',
        'IH', 'IY', 'UH', 'UW', 'OW', 'OY',
        'AW', 'AY', 'EY',
    }

    VOICELESS_CONSONANTS = {
        'p', 't', 'k', 's', 'sh', 'f', 'h', 'ch', 'c', 'x', 'q', 'j',
        'か', 'さ', 'し', 'は', 'ひ', 'ふ', 'ぱ', 'ぴ', 'ぺ', 'ぽ',
        'P', 'T', 'K', 'S', 'SH', 'F', 'HH', 'CH', 'TH',
    }

    VOICED_CONSONANTS = {
        'b', 'd', 'g', 'z', 'zh', 'm', 'n', 'ng', 'l', 'r', 'w', 'y',
        'が', 'ざ', 'じ', 'ば', 'び', 'べ', 'ぼ', 'ま', 'み', 'む',
        'め', 'も', 'な', 'に', 'ぬ', 'ね', 'の', 'ら', 'り', 'る',
        'れ', 'ろ', 'や', 'ゆ', 'よ', 'わ',
        'B', 'D', 'G', 'Z', 'ZH', 'V', 'DH',
        'M', 'N', 'NG', 'L', 'R', 'W', 'Y', 'JH',
    }

    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048,
                 hop_length: int = 512, min_ratio: float = 0.8,
                 max_ratio: float = 1.2):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.min_duration_ratio = min_ratio
        self.max_duration_ratio = max_ratio
        self.window = np.hanning(self.win_length).astype(np.float64)

    def detect_vowel(self, phoneme: str) -> bool:
        """检测音素是否为元音

        Args:
            phoneme: 音素名称

        Returns:
            True 如果是元音，否则 False
        """
        p = phoneme.strip().upper()
        if p in self.VOWEL_SET:
            return True

        p_lower = phoneme.strip().lower()
        if p_lower in self.VOWEL_SET:
            return True

        for char in phoneme.strip():
            if char in self.VOWEL_SET or char.upper() in self.VOWEL_SET:
                return True

        return False

    def detect_consonant(self, phoneme: str) -> bool:
        """检测音素是否为辅音

        Args:
            phoneme: 音素名称

        Returns:
            True 如果是辅音，否则 False
        """
        p = phoneme.strip().upper()
        if p in self.VOICELESS_CONSONANTS or p in self.VOICED_CONSONANTS:
            return True

        p_lower = phoneme.strip().lower()
        if p_lower in self.VOICELESS_CONSONANTS or p_lower in self.VOICED_CONSONANTS:
            return True

        for char in phoneme.strip():
            if char in self.VOICELESS_CONSONANTS or char in self.VOICED_CONSONANTS:
                return True
            if char.upper() in self.VOICELESS_CONSONANTS or char.upper() in self.VOICED_CONSONANTS:
                return True

        return not self.detect_vowel(phoneme)

    def detect_syllable_boundary(self, prev_phone: str, curr_phone: str) -> bool:
        """检测音节边界

        音节边界通常出现在：
        - 辅音 + 元音 的过渡处
        - 词边界处

        Args:
            prev_phone: 前一个音素
            curr_phone: 当前音素

        Returns:
            True 如果是音节边界，否则 False
        """
        if not prev_phone or not curr_phone:
            return False

        prev_is_vowel = self.detect_vowel(prev_phone)
        curr_is_vowel = self.detect_vowel(curr_phone)
        prev_is_consonant = self.detect_consonant(prev_phone)
        curr_is_consonant = self.detect_consonant(curr_phone)

        if prev_is_consonant and curr_is_vowel:
            return True

        if prev_is_vowel and curr_is_consonant and self._is_plosive(curr_phone):
            return True

        return False

    def detect_word_boundary(self, prev_phone: str, curr_phone: str) -> bool:
        """检测词边界

        通过音素标记或间隔检测词边界。
        实际应用中通常由外部传入的标记决定，这里提供启发式判断。

        Args:
            prev_phone: 前一个音素
            curr_phone: 当前音素

        Returns:
            True 如果是词边界，否则 False
        """
        if not prev_phone or not curr_phone:
            return False

        if prev_phone in ('', ' ', 'pau', 'sil', 'br'):
            return True
        if curr_phone in ('', ' ', 'pau', 'sil', 'br'):
            return True

        return False

    def _is_plosive(self, phoneme: str) -> bool:
        """检测是否为爆破音"""
        plosives = {'p', 't', 'k', 'b', 'd', 'g',
                    'P', 'T', 'K', 'B', 'D', 'G',
                    'か', 'た', 'ぱ', 'が', 'だ', 'ば'}
        p = phoneme.strip()
        return p in plosives or p.upper() in plosives or p.lower() in plosives

    def _is_fricative(self, phoneme: str) -> bool:
        """检测是否为擦音"""
        fricatives = {'s', 'sh', 'f', 'h', 'z', 'zh', 'v',
                      'S', 'SH', 'F', 'HH', 'Z', 'ZH', 'V', 'TH', 'DH',
                      'さ', 'し', 'は', 'ひ', 'ふ', 'ざ', 'じ'}
        p = phoneme.strip()
        return p in fricatives or p.upper() in fricatives or p.lower() in fricatives

    def _is_voiceless(self, phoneme: str) -> bool:
        """检测是否为清辅音"""
        p = phoneme.strip()
        return (p in self.VOICELESS_CONSONANTS or
                p.upper() in self.VOICELESS_CONSONANTS or
                p.lower() in self.VOICELESS_CONSONANTS)

    def optimize_vowel_duration(self, phoneme_list: List[Dict],
                                tempo: float = 120.0) -> List[Dict]:
        """元音时长优化

        根据上下文位置自动调整元音时长：
        - 句末元音自然延长 1.1-1.3 倍
        - 重读音节延长 1.05-1.15 倍
        - 过渡元音轻微缩短 0.9-0.95 倍

        Args:
            phoneme_list: 音素列表，每个元素为包含 'phoneme', 'duration',
                          'is_stressed', 'is_final' 等字段的字典
            tempo: 语速（BPM），影响延长幅度

        Returns:
            更新后的音素列表，包含 'duration_ratio' 字段
        """
        if not phoneme_list:
            return phoneme_list

        result = []
        n = len(phoneme_list)

        tempo_factor = 120.0 / max(tempo, 60.0)
        tempo_factor = np.clip(tempo_factor, 0.8, 1.2)

        for i, phone in enumerate(phoneme_list):
            phone_copy = phone.copy()
            phoneme_name = phone.get('phoneme', '')
            is_vowel = self.detect_vowel(phoneme_name)

            if not is_vowel:
                phone_copy['duration_ratio'] = 1.0
                result.append(phone_copy)
                continue

            ratio = 1.0

            is_final = phone.get('is_final', False) or i == n - 1
            if is_final and is_vowel:
                final_extend = 1.15 + 0.15 * tempo_factor
                final_extend = np.clip(final_extend, 1.1, 1.3)
                ratio *= final_extend
                logger.debug(f"句末元音延长: {phoneme_name} x{final_extend:.2f}")

            is_stressed = phone.get('is_stressed', False)
            if is_stressed and is_vowel:
                stress_extend = 1.05 + 0.10 * tempo_factor
                stress_extend = np.clip(stress_extend, 1.05, 1.15)
                ratio *= stress_extend
                logger.debug(f"重读音节延长: {phoneme_name} x{stress_extend:.2f}")

            if i > 0 and i < n - 1:
                prev_vowel = self.detect_vowel(phoneme_list[i - 1].get('phoneme', ''))
                next_vowel = self.detect_vowel(phoneme_list[i + 1].get('phoneme', ''))
                if prev_vowel and next_vowel and not is_stressed and not is_final:
                    transition_shorten = 0.92 + 0.03 * (1 - tempo_factor)
                    transition_shorten = np.clip(transition_shorten, 0.9, 0.95)
                    ratio *= transition_shorten
                    logger.debug(f"过渡元音缩短: {phoneme_name} x{transition_shorten:.2f}")

            ratio = np.clip(ratio, self.min_duration_ratio, self.max_duration_ratio)
            phone_copy['duration_ratio'] = ratio
            result.append(phone_copy)

        return result

    def optimize_consonant_transition(self, phoneme_list: List[Dict]) -> List[Dict]:
        """辅音过渡优化

        根据前后音素类型调整辅音的时长和强度包络：
        - 清辅音+元音：辅音尾部渐强
        - 元音+清辅音：元音尾部渐弱
        - 连续辅音：调整重叠程度

        Args:
            phoneme_list: 音素列表

        Returns:
            更新后的音素列表，包含过渡相关参数
        """
        if not phoneme_list:
            return phoneme_list

        result = [p.copy() for p in phoneme_list]
        n = len(result)

        for i in range(n):
            curr = result[i]
            curr_phone = curr.get('phoneme', '')
            curr_is_consonant = self.detect_consonant(curr_phone)
            curr_is_vowel = self.detect_vowel(curr_phone)

            if i < n - 1:
                next_phone = result[i + 1].get('phoneme', '')
                next_is_vowel = self.detect_vowel(next_phone)
                next_is_consonant = self.detect_consonant(next_phone)

                if curr_is_consonant and next_is_vowel and self._is_voiceless(curr_phone):
                    curr['tail_envelope'] = 'ramp_up'
                    curr['overlap_ratio'] = 0.3
                    logger.debug(f"清辅音+元音过渡: {curr_phone} -> {next_phone}")

                if curr_is_vowel and next_is_consonant and self._is_voiceless(next_phone):
                    curr['tail_envelope'] = 'ramp_down'
                    curr['ramp_down_ratio'] = 0.2
                    logger.debug(f"元音+清辅音过渡: {curr_phone} -> {next_phone}")

                if curr_is_consonant and next_is_consonant:
                    if self._is_fricative(curr_phone) and self._is_plosive(next_phone):
                        curr['overlap_ratio'] = 0.5
                    elif self._is_plosive(curr_phone) and self._is_fricative(next_phone):
                        curr['overlap_ratio'] = 0.4
                    else:
                        curr['overlap_ratio'] = 0.3
                    logger.debug(f"连续辅音过渡: {curr_phone} -> {next_phone}")

            if i > 0:
                prev_phone = result[i - 1].get('phoneme', '')
                prev_is_consonant = self.detect_consonant(prev_phone)

                if curr_is_vowel and prev_is_consonant and self._is_voiceless(prev_phone):
                    curr['head_envelope'] = 'ramp_up'
                    curr['ramp_up_ratio'] = 0.15

        for phone in result:
            if 'duration_ratio' not in phone:
                phone['duration_ratio'] = 1.0

        return result

    def optimize_syllable_rhythm(self, phoneme_list: List[Dict],
                                 tempo: float = 120.0) -> List[Dict]:
        """音节级节奏优化

        对音节序列进行节奏微调，使整体更自然：
        - 弹性节奏：轻微的时长扰动，避免机械感
        - 音节间呼吸间隙：适当的微小停顿
        - 语速自适应：根据全局语速调整局部节奏

        Args:
            phoneme_list: 音素列表
            tempo: 语速（BPM）

        Returns:
            更新后的音素列表
        """
        if not phoneme_list:
            return phoneme_list

        result = [p.copy() for p in phoneme_list]
        n = len(result)

        tempo_factor = tempo / 120.0
        tempo_factor = np.clip(tempo_factor, 0.7, 1.4)

        jitter_strength = 0.02 * (1.0 / tempo_factor)
        jitter_strength = np.clip(jitter_strength, 0.005, 0.03)

        np.random.seed(42)
        jitter = np.random.normal(0, jitter_strength, n)
        jitter = np.clip(jitter, -jitter_strength * 2, jitter_strength * 2)

        syllable_boundaries = []
        for i in range(1, n):
            prev_phone = result[i - 1].get('phoneme', '')
            curr_phone = result[i].get('phoneme', '')
            if self.detect_syllable_boundary(prev_phone, curr_phone):
                syllable_boundaries.append(i)

        syllable_count = len(syllable_boundaries) + 1
        if syllable_count > 1:
            accel_ratio = 0.005 * (1.0 / tempo_factor)
            for idx in range(n):
                position = idx / max(n - 1, 1)
                phrase_jitter = jitter[idx]
                phrase_effect = 0.0
                if position < 0.5:
                    phrase_effect = -accel_ratio * (0.5 - position) * 2
                else:
                    phrase_effect = accel_ratio * (position - 0.5) * 2

                ratio = 1.0 + phrase_jitter + phrase_effect
                ratio = np.clip(ratio, self.min_duration_ratio, self.max_duration_ratio)

                if 'duration_ratio' in result[idx]:
                    result[idx]['duration_ratio'] *= ratio
                else:
                    result[idx]['duration_ratio'] = ratio

        word_boundary_indices = []
        for i in range(1, n):
            prev_phone = result[i - 1].get('phoneme', '')
            curr_phone = result[i].get('phoneme', '')
            if self.detect_word_boundary(prev_phone, curr_phone):
                word_boundary_indices.append(i)

        for idx in word_boundary_indices:
            if idx < n:
                phone = result[idx]
                phoneme_name = phone.get('phoneme', '')
                if self.detect_consonant(phoneme_name):
                    breath_gap = 0.02 * (1.0 / tempo_factor)
                    breath_gap = np.clip(breath_gap, 0.005, 0.04)
                    phone.setdefault('duration_ratio', 1.0)
                    phone['duration_ratio'] += breath_gap
                    phone['duration_ratio'] = np.clip(
                        phone['duration_ratio'],
                        self.min_duration_ratio,
                        self.max_duration_ratio
                    )
                    logger.debug(f"词边界呼吸间隙: {phoneme_name} +{breath_gap:.3f}")

        return result

    def apply_timing_adjustment(self, audio: np.ndarray,
                                adjustment_ratio: float,
                                method: str = 'phase_vocoder') -> np.ndarray:
        """应用时长调整

        使用时间拉伸算法调整音频时长，保持音高和音色不变。
        支持相位声码器和 SOLA 两种算法。

        Args:
            audio: 输入音频 (n_samples,)
            adjustment_ratio: 时长调整比例 (>1 变长, <1 变短)
            method: 算法选择 ('phase_vocoder' 或 'sola')

        Returns:
            调整后的音频

        Raises:
            ValueError: 如果 method 参数无效
        """
        if len(audio) == 0:
            return audio

        adjustment_ratio = np.clip(
            adjustment_ratio,
            self.min_duration_ratio,
            self.max_duration_ratio
        )

        if abs(adjustment_ratio - 1.0) < 1e-4:
            return audio.copy()

        if method == 'phase_vocoder':
            return self._phase_vocoder_stretch(audio, adjustment_ratio)
        elif method == 'sola':
            return self._sola_stretch(audio, adjustment_ratio)
        else:
            raise ValueError(f"未知的拉伸方法: {method}")

    def _phase_vocoder_stretch(self, audio: np.ndarray, ratio: float) -> np.ndarray:
        """相位声码器时间拉伸

        基于 STFT/iSTFT 的相位声码器算法，保持音高不变进行时间拉伸。

        Args:
            audio: 输入音频
            ratio: 拉伸比率

        Returns:
            拉伸后的音频
        """
        audio_f64 = audio.astype(np.float64)

        f, t, Zxx = signal.stft(
            audio_f64,
            fs=self.sample_rate,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            window=self.window,
            return_onesided=True,
            boundary='zeros',
            padded=True,
        )

        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        n_freq, n_frames = Zxx.shape
        n_frames_out = int(np.round(n_frames * ratio))

        if n_frames_out <= 0:
            return np.zeros(0, dtype=audio.dtype)

        phasor = 2 * np.pi * self.hop_length * np.arange(n_freq) / self.n_fft

        mag_out = np.zeros((n_freq, n_frames_out), dtype=np.float64)
        phase_out = np.zeros((n_freq, n_frames_out), dtype=np.float64)

        phase_acc = phase[:, 0].copy()

        time_indices = np.arange(n_frames_out) / ratio

        for i in range(n_frames_out):
            t_idx = time_indices[i]
            t_floor = int(np.floor(t_idx))
            t_ceil = min(t_floor + 1, n_frames - 1)
            t_frac = t_idx - t_floor

            t_floor = max(0, min(t_floor, n_frames - 1))
            t_ceil = max(0, min(t_ceil, n_frames - 1))

            mag_out[:, i] = (1 - t_frac) * mag[:, t_floor] + t_frac * mag[:, t_ceil]

            if i > 0:
                delta_phase = phase[:, t_ceil] - phase[:, t_floor] - phasor
                delta_phase = np.mod(delta_phase + np.pi, 2 * np.pi) - np.pi
                inst_freq = phasor + delta_phase
                phase_acc += inst_freq / ratio

            phase_out[:, i] = phase_acc

        Zxx_out = mag_out * np.exp(1j * phase_out)

        _, result = signal.istft(
            Zxx_out,
            fs=self.sample_rate,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            window=self.window,
            input_onesided=True,
            boundary=True,
        )

        target_len = int(len(audio) * ratio)
        if len(result) > target_len:
            result = result[:target_len]
        elif len(result) < target_len:
            result = np.pad(result, (0, target_len - len(result)))

        return result.astype(audio.dtype)

    def _sola_stretch(self, audio: np.ndarray, ratio: float) -> np.ndarray:
        """SOLA (Synchronous Overlap-Add) 时间拉伸

        基于时域重叠相加的算法，计算速度快，适合实时应用。

        Args:
            audio: 输入音频
            ratio: 拉伸比率

        Returns:
            拉伸后的音频
        """
        audio_f64 = audio.astype(np.float64)
        src_len = len(audio_f64)
        dst_len = int(src_len * ratio)

        if dst_len == 0:
            return np.zeros(0, dtype=audio.dtype)

        frame_len = self.win_length
        synthesis_hop = self.hop_length
        analysis_hop = int(synthesis_hop / ratio)

        if analysis_hop < 1:
            analysis_hop = 1

        output = np.zeros(dst_len + frame_len, dtype=np.float64)
        window = self.window.copy()
        window_sum = np.zeros(dst_len + frame_len, dtype=np.float64)

        output_frame = 0
        input_pos = 0

        while input_pos + frame_len <= src_len and output_frame + frame_len <= dst_len:
            frame = audio_f64[input_pos:input_pos + frame_len] * window

            if output_frame > 0:
                search_start = max(0, output_frame - synthesis_hop // 2)
                search_end = min(dst_len - frame_len, output_frame + synthesis_hop // 2)

                best_offset = output_frame
                best_corr = -1.0

                for offset in range(search_start, search_end + 1, 1):
                    if offset + frame_len > dst_len:
                        break
                    overlap_len = frame_len - (offset - (output_frame - synthesis_hop))
                    if overlap_len <= 0:
                        continue

                    ref = output[offset:offset + overlap_len]
                    src = frame[:overlap_len]

                    denom = np.sqrt(np.sum(ref ** 2) * np.sum(src ** 2))
                    if denom < 1e-10:
                        continue

                    corr = np.sum(ref * src) / denom
                    if corr > best_corr:
                        best_corr = corr
                        best_offset = offset

                output_frame = best_offset

            output[output_frame:output_frame + frame_len] += frame
            window_sum[output_frame:output_frame + frame_len] += window

            output_frame += synthesis_hop
            input_pos += analysis_hop

        result = np.zeros(dst_len, dtype=np.float64)
        output_len = min(dst_len, len(output))
        window_len = min(dst_len, len(window_sum))
        common_len = min(output_len, window_len)

        result[:common_len] = output[:common_len]

        valid = window_sum[:common_len] > 1e-10
        result[:common_len][valid] /= window_sum[:common_len][valid]

        return result.astype(audio.dtype)

    def apply_timing_to_mel(self, mel: np.ndarray,
                            adjustment_ratio: float) -> np.ndarray:
        """在 Mel 频谱层面应用时长调整

        通过时间轴插值实现 Mel 频谱的时长调整，与声码器流程兼容。
        这种方式比波形层面的调整更高效，因为声码器直接使用 Mel。

        Args:
            mel: Mel 频谱 (n_mels, n_frames)，自然对数尺度
            adjustment_ratio: 时长调整比例

        Returns:
            调整后的 Mel 频谱
        """
        if mel.ndim != 2 or mel.shape[0] == 0 or mel.shape[1] == 0:
            return mel

        adjustment_ratio = np.clip(
            adjustment_ratio,
            self.min_duration_ratio,
            self.max_duration_ratio
        )

        if abs(adjustment_ratio - 1.0) < 1e-4:
            return mel.copy()

        n_mels, n_frames = mel.shape
        n_frames_out = int(np.round(n_frames * adjustment_ratio))

        if n_frames_out <= 0:
            return np.zeros((n_mels, 0), dtype=mel.dtype)

        result = np.zeros((n_mels, n_frames_out), dtype=mel.dtype)

        time_in = np.arange(n_frames)
        time_out = np.linspace(0, n_frames - 1, n_frames_out)

        for m in range(n_mels):
            result[m, :] = np.interp(time_out, time_in, mel[m, :])

        return result

    def optimize_all(self, phoneme_list: List[Dict],
                     tempo: float = 120.0) -> List[Dict]:
        """执行所有优化步骤

        依次执行元音时长优化、辅音过渡优化和音节节奏优化。

        Args:
            phoneme_list: 音素列表
            tempo: 语速（BPM）

        Returns:
            完整优化后的音素列表
        """
        result = self.optimize_vowel_duration(phoneme_list, tempo)
        result = self.optimize_consonant_transition(result)
        result = self.optimize_syllable_rhythm(result, tempo)
        return result


def optimize_phoneme_timing(phoneme_list: List[Dict],
                            sample_rate: int = 44100,
                            tempo: float = 120.0) -> List[Dict]:
    """便捷函数：音素时长优化

    Args:
        phoneme_list: 音素列表
        sample_rate: 采样率
        tempo: 语速（BPM）

    Returns:
        优化后的音素列表
    """
    optimizer = PhonemeTimingOptimizer(sample_rate=sample_rate)
    return optimizer.optimize_all(phoneme_list, tempo)


def stretch_audio(audio: np.ndarray, ratio: float,
                  sample_rate: int = 44100,
                  method: str = 'phase_vocoder') -> np.ndarray:
    """便捷函数：音频时间拉伸

    Args:
        audio: 输入音频
        ratio: 拉伸比率
        sample_rate: 采样率
        method: 算法 ('phase_vocoder' 或 'sola')

    Returns:
        拉伸后的音频
    """
    optimizer = PhonemeTimingOptimizer(sample_rate=sample_rate)
    return optimizer.apply_timing_adjustment(audio, ratio, method)


def stretch_mel(mel: np.ndarray, ratio: float,
                sample_rate: int = 44100) -> np.ndarray:
    """便捷函数：Mel 频谱时间拉伸

    Args:
        mel: Mel 频谱 (n_mels, n_frames)
        ratio: 拉伸比率
        sample_rate: 采样率

    Returns:
        拉伸后的 Mel
    """
    optimizer = PhonemeTimingOptimizer(sample_rate=sample_rate)
    return optimizer.apply_timing_to_mel(mel, ratio)
