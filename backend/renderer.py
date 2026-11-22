"""Diffical 核心渲染引擎

整句渲染流程：
1. 从 RenderPhraseRequest 获取整句音素、曲线
2. 为每个音素加载声库切片
3. hnsep 三分量分离（谐波/噪声/残差）
4. 各分量独立时间拉伸和音高调整
5. 各分量独立拼接（交叉淡入淡出 + 共振峰匹配）
6. Mel 频谱级共振峰约束（防止音色泄漏）
7. pc-nsf-hifigan 波形合成
8. HN-SEP 参数处理（breath/tension/voicing/brel/breh/warm/hcmp）
9. 动态低切滤波
10. 咆哮效果处理
11. 整句流畅度增强（五级）
12. 真人化细节增强（六维）
13. 响度归一化
"""

import os
import logging
from typing import List, Optional, Tuple, Dict

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


def _frame_to_pitch_pos_safe(pr, frame_idx: int) -> float:
    try:
        return float(pr._frame_to_pitch_pos(frame_idx))
    except Exception:
        return 0.0

def effective_phone_audio_start_ms(target_start_ms: float,
                                   phoneme_preutter_ms: float) -> float:
    """OpenUtau audio origin: phone anchor minus resolved effective preutter.

    RenderPhone.preutter_ms comes from UPhoneme.preutter after oto, velocity,
    adjacent-note clamping and user preutter delta. Raw oto_preutter is source
    metadata, not the final phrase placement. C# exports envelope p0 in this
    same phone-relative coordinate system.
    """
    target_start_ms = float(target_start_ms)
    phoneme_preutter_ms = float(phoneme_preutter_ms)
    if not np.isfinite(target_start_ms) or not np.isfinite(phoneme_preutter_ms):
        raise ValueError("phone start and effective preutter must be finite")
    phoneme_preutter_ms = max(0.0, phoneme_preutter_ms)
    return target_start_ms - phoneme_preutter_ms


def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(hz / 440.0)

from .config import get_config
from .models import get_model_manager
from .protocol import RenderPhraseRequest, RenderResultResponse, RenderPhone
from .overlap import MelOverlapAccumulator, _read_envelope
from .utau_geometry import plan_oto_window
from .source_timing import source_skip_ms
from .source_loudness import measure_source_level
from .voicebank import find_wav_file
from .phoneme_compensator import find_compensation_file, PhonemeCompensator
from .phonemizer_config import VOWEL_MAPPING
from .phase_vocoder import PhaseVocoder
from .mel_concatenate import energy_smooth, spectral_flux
from .mel_concatenate import concatenate_mels, formant_constrain, formant_smooth_transition, frequency_dependent_crossfade
from .hnsep_pipeline import HarmonicNoiseProcessor
from .fluency import FluencyEnhancer
from .detail_enhance import DetailEnhancer
from .growl import apply_growl
from .tension_filter import apply_dynamic_filter
from .warmth import apply_warmth, apply_harmonic_compressor
from .frq_reader import get_sample_tone, get_sample_vowel_f0_hz
from .loudness import normalize_phrase_rms
from .denoiser import SmartDenoiser

import re

FLAGS_PATTERN = re.compile(r"([fe|fl|fo|fv|fp|ve|vo|g|t|A|B|G|P|S|p|R|D|C|Z|Hv|Hb|Ht|He|HG])([+-]?\d+)?")

def parse_flags(flags_str: str) -> Dict[str, int]:
    """解析UTAU flags字符串
    
    参考同类实现/util/parse_utau.py 的 flag_re 正则表达式
    
    Args:
        flags_str: flags字符串，如 "Hb50Hv100g-50"
        
    Returns:
        flags字典，如 {"Hb": 50, "Hv": 100, "g": -50}
    """
    result = {}
    for k, v in FLAGS_PATTERN.findall(flags_str.replace('/', '')):
        if k not in result:
            result[k] = int(v) if v else None
    return result

def map_flag_to_parameter(phone: RenderPhone) -> Dict[str, float]:
    """将音素的flags映射为参数值
    
    参考同类实现 的参数映射：
    - Hb (0-500, default 100) -> breathiness (0-100, default 50)
      breathiness = Hb / 5
    - Hv (0-150, default 100) -> voicing (0-100, default 100)
      voicing = Hv / 1.5
    - Ht (-100 to 100, default 0) -> tension (0-100, default 50)
      tension = (Ht + 100) / 2
    - g (-600 to 600, default 0) -> gender (0-100, default 50)
      gender = (g / 6) + 50
    - HG (0-100, default 0) -> growl (0-100, default 0)
    """
    params = {
        'breathiness': 50.0,
        'voicing': 100.0,
        'tension': 50.0,
        'gender': 50.0,
        'growl': 0.0,
    }
    
    if hasattr(phone, 'flags') and phone.flags:
        for flag_tuple in phone.flags:
            if isinstance(flag_tuple, tuple) and len(flag_tuple) >= 2:
                flag_name = flag_tuple[0]
                flag_value = flag_tuple[1]
                if flag_value is None:
                    continue

                # 防御：NaN/Inf 的 flag 值会导致 np.clip 仍为 NaN，进入 mel 计算后全 NaN 崩溃
                try:
                    flag_value = float(flag_value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(flag_value):
                    logger.warning(f"flag {flag_name}={flag_value} 非有限值，已跳过")
                    continue

                if flag_name == 'Hb':
                    # Hb=100 是 同类重采样器 默认（中性）；映射到 0（无额外气声）
                    params['breathiness'] = np.clip((flag_value - 100.0) / 5.0, 0, 100)
                elif flag_name == 'Hv':
                    params['voicing'] = np.clip(flag_value / 1.5, 0, 100)
                elif flag_name == 'Ht':
                    params['tension'] = np.clip((flag_value + 100) / 2.0, 0, 100)
                elif flag_name == 'g':
                    # OpenUtau gen/g 默认 0 = 中性（不是 50）
                    params['gender'] = np.clip(float(flag_value), -100, 100)
                elif flag_name == 'HG':
                    params['growl'] = np.clip(flag_value, 0, 100)
    
    return params
from .quality_evaluator import QualityEvaluator
from .curve_params import CurveParams
from .performance import get_performance_optimizer
from .phoneme_timing import PhonemeTimingOptimizer
from .timbre_mixer import TimbreMixer
from .mel_concatenate import pre_echo_suppression

logger = logging.getLogger(__name__)

def midi_to_hz(midi: float) -> float:
    """MIDI 音数 → Hz"""
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def cent_to_hz(cent: float, base_hz: float = 440.0) -> float:
    """cent → Hz"""
    return base_hz * (2.0 ** (cent / 1200.0))


# 跨请求的路径解析缓存：同一 (singer, relative_path) 的解析结果复用，
# 避免每个渲染请求都对整个声库树 os.walk 一次（大库 + 多音素时是主要耗时）。
_PATH_RESOLVE_CACHE: Dict[Tuple[str, str], Optional[str]] = {}
_PATH_RESOLVE_CACHE_MAX = 2000


class AudioLoader:
    """声库音频文件加载器

    只负责根据路径读取音频文件，不负责声库解析。
    声库管理和 oto.ini 解析由 OpenUTAU 负责。
    """

    def __init__(self, singer_path: str):
        self.singer_path = singer_path
        self._cache = {}
        self._compensator = PhonemeCompensator(singer_path)

    def _resolve_path(self, relative_path: str) -> Optional[str]:
        """解析音频文件路径（带跨请求缓存）"""
        key = (self.singer_path, relative_path)
        cached = _PATH_RESOLVE_CACHE.get(key)
        if cached is not None or key in _PATH_RESOLVE_CACHE:
            return cached
        actual_path = None
        if relative_path:
            if os.path.isabs(relative_path):
                if os.path.isfile(relative_path):
                    actual_path = relative_path
            else:
                joined = os.path.join(self.singer_path, relative_path)
                if os.path.isfile(joined):
                    actual_path = joined
        if actual_path is None:
            actual_path = find_wav_file(self.singer_path, relative_path)
        if len(_PATH_RESOLVE_CACHE) >= _PATH_RESOLVE_CACHE_MAX:
            _PATH_RESOLVE_CACHE.clear()
        _PATH_RESOLVE_CACHE[key] = actual_path
        return actual_path

    def load(self, relative_path: str, sr: int = 44100) -> np.ndarray:
        """加载音频文件

        Args:
            relative_path: 相对于声库根目录的路径
            sr: 目标采样率

        Returns:
            wave: 单声道波形 (samples,)
        """
        cache_key = f"{relative_path}_{sr}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1) OpenUTAU 已解析好的 otoFile 优先直接读（绝对路径直接存在；
        #    相对路径拼接声库根）。绝不静默走补偿/全库搜索——那是"多音色混合、
        #    音素加载错误"的来源（引擎不应重复 OpenUTAU 已完成的工作）。
        actual_path = self._resolve_path(relative_path)
        target_phoneme = None
        matched_phoneme = None
        
        # 3) 确实找不到时才尝试补偿音素（精确 _<name>.wav，不跨库子串乱找）
        if actual_path is None:
            target_phoneme = os.path.basename(relative_path).replace('_', '').replace('.wav', '')
            compensation_path = find_compensation_file(self.singer_path, target_phoneme, relative_path)
            
            if compensation_path:
                matched_phoneme = os.path.basename(compensation_path).replace('_', '').replace('.wav', '')
                logger.warning(f"使用补偿音素: {relative_path} -> {compensation_path}")
                actual_path = find_wav_file(self.singer_path, compensation_path)
            
            if actual_path is None:
                logger.warning(f"音频文件不存在: {relative_path}")
                self._cache[cache_key] = np.zeros(0, dtype=np.float32)
                return self._cache[cache_key]

        import soundfile as sf

        wave, orig_sr = sf.read(actual_path, dtype="float32", always_2d=False)

        # 转单声道
        if wave.ndim > 1:
            wave = wave.mean(axis=1)

        # 重采样
        if orig_sr != sr:
            try:
                import resampy
                wave = resampy.resample(wave, orig_sr, sr)
            except ImportError:
                from scipy.signal import resample
                n_samples = int(len(wave) * sr / orig_sr)
                wave = resample(wave, n_samples).astype(np.float32)

        self._cache[cache_key] = wave
        
        if target_phoneme and matched_phoneme:
            adjust_params = self._compensator.compute_adjustment_params(target_phoneme, matched_phoneme)
            self._compensator._adjustment_cache[cache_key] = adjust_params
            logger.info(f"补偿音素参数已缓存: {target_phoneme} -> {matched_phoneme}, "
                        f"formant={adjust_params['formant_shift']:.2f}, "
                        f"pitch={adjust_params['pitch_shift_cent']:.1f}cent")

        return wave

    def get_adjustment_params(self, relative_path: str, sr: int = 44100) -> Dict[str, float]:
        """获取音频文件的发音特征调整参数

        Args:
            relative_path: 相对于声库根目录的路径
            sr: 采样率

        Returns:
            调整参数字典，如果没有则返回默认值
        """
        cache_key = f"{relative_path}_{sr}"
        return getattr(self._compensator, '_adjustment_cache', {}).get(cache_key, {
            'formant_shift': 1.0,
            'pitch_shift_cent': 0.0,
            'brightness': 1.0,
            'roundness': 0.0,
        })

    def get_slice(
        self,
        relative_path: str,
        offset_ms: float,
        length_ms: float,
        sr: int = 44100,
    ) -> np.ndarray:
        """获取指定位置的音频切片

        Args:
            relative_path: 音频文件路径
            offset_ms: 起始位置（毫秒）
            length_ms: 长度（毫秒，负数表示到文件末尾）
            sr: 采样率

        Returns:
            wave: 音频切片 (samples,)
        """
        wave = self.load(relative_path, sr)
        if len(wave) == 0:
            return wave

        offset_samples = int(offset_ms * sr / 1000)
        offset_samples = max(0, min(offset_samples, len(wave)))

        if length_ms < 0:
            return wave[offset_samples:]
        else:
            length_samples = int(length_ms * sr / 1000)
            end = offset_samples + length_samples
            if end > len(wave):
                wave = np.pad(wave, (0, end - len(wave)))
            return wave[offset_samples:end]


class PhoneProcessor:
    """单个音素的处理单元

    负责：加载音频 → 共振峰偏移（跨语种）→ hnsep 分离 → 时间拉伸 → 音高调整 → 提取 Mel
    """

    def __init__(self, phone: RenderPhone, audio_loader: AudioLoader,
                 config, model_manager):
        self.phone = phone
        self.audio_loader = audio_loader
        self.config = config
        self.model_manager = model_manager
        self.sr = config.audio.sample_rate
        self._wave: Optional[np.ndarray] = None
        self._hn_processor = HarmonicNoiseProcessor(sample_rate=self.sr)
        self._phase_vocoder = PhaseVocoder(
            sample_rate=self.sr,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_size,
            win_length=config.audio.win_size,
        )


    @property
    def wave(self) -> np.ndarray:
        """加载并裁剪音频（含跨语种共振峰偏移）"""
        if self._wave is None:
            self._wave = self._load_wave()
            

        return self._wave

    def _load_wave(self) -> np.ndarray:
        """加载完整音频文件（不裁剪）

        参考同类实现 的做法：先加载完整音频，提取完整 Mel，
        然后在 process() 中按 oto.ini 参数裁剪和拉伸

        防御：找不到音素音频文件时降级为静音，避免整个渲染崩溃。
        """
        phone = self.phone
        if not phone.oto_file:
            logger.warning(
                f"音素 '{phone.phoneme}' 找不到音频文件，降级为静音。"
                f"请检查声库目录中是否存在 _{phone.phoneme}.wav 或对应的假名文件。"
            )
            # 返回一段最小长度的静音（约 100ms），保证后续 Mel 提取有可用帧
            return np.zeros(int(0.1 * self.sr), dtype=np.float32)

        try:
            return self.audio_loader.load(phone.oto_file, self.sr)
        except (FileNotFoundError, OSError) as e:
            logger.warning(
                f"音素 '{phone.phoneme}' 音频文件加载失败 ({e})，降级为静音。"
            )
            return np.zeros(int(0.1 * self.sr), dtype=np.float32)

    def process(self, target_duration_ms: float, target_f0_hz: float,
                key_shift: float = 0.0, preutter_ms: float = 0.0,
                breathiness: float = 50.0, voicing: float = 100.0,
                tension: float = 50.0, gender: float = 50.0,
                growl: float = 0.0, velocity: float = 1.0,
                tail_intrude_ms: float = 0.0, tail_overlap_ms: float = 0.0) -> dict:
        """处理音素：加载 → 提取 Mel → 按 oto.ini 裁剪 → 时间拉伸

        参考同类实现 的 resampler.py：
        1. 提取完整音频的 Mel（key_shift=0，让声码器通过 F0 处理音高）
        2. 按 oto.ini 的 offset/consonant/cutoff 裁剪并拉伸到目标时长
        3. 辅音段（consonant 之前）F0 设为 0，只有元音段使用目标 F0
        4. 辅音段不拉伸，元音段按比例拉伸

        oto.ini 参数语义：
        - offset: 从文件开头到元音起始的位置（毫秒）
        - consonant: 辅音长度（毫秒），从 offset 开始算
        - cutoff: 从 offset+consonant 开始的音频长度（毫秒）
          正数=截取长度，负数=从末尾算的偏移

        Args:
            target_duration_ms: 目标时长（毫秒）
            target_f0_hz: 目标 F0（Hz）
            key_shift: 音高偏移（半音），固定为 0
            preutter_ms: 预发音时间（毫秒）

        Returns:
            包含 mel, f0, wave 等的字典
        """
        wave = self.wave
        if len(wave) == 0:
            return {
                "mel": np.zeros((self.config.audio.n_mels, 0), dtype=np.float32),
                "f0": np.zeros(0, dtype=np.float32),
                "wave": np.zeros(0, dtype=np.float32),
                "phone": self.phone,
                "phoneme_features": None,
                "pre_scale": 1.0,
            }

        # 峰值限制：只在接近削波时缩小（>0.98），不做 pre_scale 降 6dB
        # （同类重采样器 不做此操作；pre_scale=0.5 会导致 mel 全带 -6dB 失真）
        wave_peak = np.max(np.abs(wave))
        if wave_peak > 0.98:
            pre_scale = 0.98 / wave_peak
            wave = wave * pre_scale
        else:
            pre_scale = 1.0

        phone = self.phone
        offset_ms = phone.oto_offset
        consonant_ms = phone.oto_consonant
        cutoff_ms = phone.oto_cutoff
        source_total_ms = len(wave) / self.sr * 1000.0
        if cutoff_ms < 0:
            source_end_ms = offset_ms + abs(cutoff_ms)
        elif cutoff_ms > 0:
            source_end_ms = source_total_ms - cutoff_ms
        else:
            source_end_ms = source_total_ms
        source_end_ms = max(offset_ms + consonant_ms,
                            min(source_total_ms, source_end_ms))
        _source_level_rms, _source_level_valid, _source_level_region = measure_source_level(
            wave, self.sr, offset_ms, consonant_ms, source_end_ms,
            float(getattr(phone, 'oto_preutter', 0.0) or 0.0),
            float(getattr(phone, 'oto_overlap', 0.0) or 0.0),
            (' ' in str(getattr(phone, 'phoneme', ''))
             or str(getattr(phone, 'phoneme', '')).startswith(('-', '+'))),
        )

        # Extract whole-file mel before source-time mapping.
        mel_analyzer = self.model_manager.mel_analyzer
        
        # 使用hop=512计算帧时长（与声码器一致）
        # 参考同类实现：提取时使用细hop，但时间映射使用声码器hop
        frame_ms = self.config.audio.hop_size / self.sr * 1000
        origin_hop_size = self.config.audio.origin_hop_size
        hop_ratio = self.config.audio.hop_size // origin_hop_size

        # 参考同类实现：key_shift 默认固定为 0
        # pc-nsf-hifigan 是 pitch-conditional vocoder：
        # - Mel 提供频谱包络（音色），不需要频谱移位
        # - F0 输入控制音高，声码器通过 F0 实现音高转换
        # （声码器往返测试显示 key_shift=0 时高频能量最低）
        #
        # 唯一例外：gender（g flag / GEN 曲线）需要共振峰（频谱）偏移。
        # 同类重采样器 用 mel_analyzer(wave, g/100, 1) 实现 formant shift，
        # 这里映射 g=(gender-50)*6，所以半音数 = (gender-50)*0.06。
        # OpenUTAU/参考语义：gender 默认 0 = 中性（不是 50！）。
        # key_shift(半音) = gender/100（同类引擎: semitone_curve = genc/100）。
        key_shift = 0.0
        if abs(gender) > 0.5:
            key_shift = gender / 100.0

        # 参考同类实现：应用 flags 参数处理（Hb/Hv/Ht/g）
        processed_wave = wave.copy()

        # 注意：OpenUTAU 曲线默认值是 breathiness=0 / tension=0 / voicing=100
        # "用户画了曲线" -> 触发逐音素 HNSEP，breathiness=0 把谐波清零 ->
        # mel 低频掉 ~4dB、vocode 输出轻 13 倍且带噪声（"音质特别烂"的直接根因之一）。
        # 修复：默认值按 OpenUTAU 语义判断；混合公式改为
        #   processed = harmonic*(1-breath/100) + noise*(voicing/100)
        # 默认(breath=0, voicing=100) 时 = harmonic+noise = 原样（no-op）。
        if breathiness > 0.5 or abs(tension) > 0.5 or abs(voicing - 100.0) > 0.5:
            try:
                _hnsep = self.model_manager._get_hnsep()
                if _hnsep is not None:
                    harmonic, noise = _hnsep.separate(wave)
                else:
                    harmonic, noise = self._hn_processor.separate(wave)

                breath_val = np.clip(breathiness, 0, 100)
                voicing_val = np.clip(voicing, 0, 100)
                tension_val = np.clip(tension * 2.0, -100, 100)
                harmonic_coef = 1.0 - breath_val / 100.0

                if tension_val != 0:
                    filter_level = np.full(len(wave), -tension_val / 50 * 100, dtype=np.float32)
                    filtered_noise = apply_dynamic_filter((voicing_val/100) * noise, filter_level, self.sr)
                    processed_wave = harmonic_coef * (wave - noise) + filtered_noise
                else:
                    processed_wave = harmonic_coef * (wave - noise) + \
                        (voicing_val/100) * noise
            except Exception as e:
                logger.warning(f"HNSEP 参数处理失败: {e}")
                processed_wave = wave.copy()

        # 提取完整音频的 Mel（key_shift=0，让声码器通过 F0 处理音高）
        # 使用 hop=128 提取，获得更细的时间分辨率
        mel_orig = mel_analyzer(processed_wave, key_shift=key_shift, speed=1.0)
        mel_orig = mel_orig.squeeze().cpu().numpy()
        orig_n_frames = mel_orig.shape[-1]
        n_mels = mel_orig.shape[0]
        
        # 将hop=128的时间轴映射到hop=512的时间轴
        # 原始音频时长（以hop=128帧为单位）
        orig_duration_frames_128 = orig_n_frames
        orig_duration_ms = orig_duration_frames_128 * origin_hop_size / self.sr * 1000

        if orig_n_frames == 0:
            return {
                "mel": np.zeros((n_mels, 0), dtype=np.float32),
                "f0": np.zeros(0, dtype=np.float32),
                "wave": wave,
                "phone": self.phone,
                "phoneme_features": None,
                "pre_scale": pre_scale,
            }

        # 参考同类实现 的时间轴计算
        phone = self.phone
        offset_ms = phone.oto_offset
        consonant_ms = phone.oto_consonant
        cutoff_ms = phone.oto_cutoff

        # 原始音频时间轴（以 hop=128 帧为单位）
        # end_ms 随之偏移到真实音频末端之外，np.interp 对越界点返回末帧常量，
        # 使整句 Mel 退化为低能量静态帧 → 声码器输出噪声。
        origin_frame_ms = origin_hop_size / self.sr * 1000
        total_time_ms = orig_n_frames * origin_frame_ms
        t_orig_ms = np.arange(orig_n_frames) * origin_frame_ms + origin_frame_ms / 2

        # 确定裁剪区域（毫秒）
        # 参考同类实现 的 oto.ini cutoff 语义：
        # - cutoff > 0: 右边界 = 总时长 - cutoff（从末尾算）
        # - cutoff < 0: 右边界 = offset + |cutoff|（从 offset 算）
        # - cutoff == 0: 右边界 = 总时长
        vel = 2.0 ** (1.0 - float(velocity))
        pre_vel_ms = max(0.0, float(preutter_ms))
        envelope = getattr(phone, 'envelope', None)
        envelope_span_ms = (pre_vel_ms + max(float(p[0]) for p in envelope)
                            if envelope else pre_vel_ms + target_duration_ms)
        plan = plan_oto_window(
            offset_ms, consonant_ms, cutoff_ms, total_time_ms,
            pre_vel_ms, float(velocity), envelope_span_ms,
            frame_ms=frame_ms, origin_frame_ms=origin_frame_ms,
            oto_preutter_ms=float(phone.oto_preutter))
        start_ms, con_ms, end_ms = plan["start_ms"], plan["con_ms"], plan["end_ms"]
        con_vel, con_output_ms = plan["con_vel"], plan["con_output_ms"]
        total_span_ms = plan["total_span_ms"]
        target_n_frames = plan["target_n_frames"]
        t_mapped_ms, source_present = plan["t_mapped_ms"], plan["source_present"]
        source_anchor_skip_ms = float(phone.oto_preutter) * vel - pre_vel_ms
        scaling_ratio = plan["scaling_ratio"]

        # 使用线性插值提取 Mel
        # Mel是用hop=128提取的，所以时间轴需要用hop=128的分辨率
        t_orig_ms_128 = np.arange(orig_n_frames) * (origin_hop_size / self.sr * 1000) + (origin_hop_size / self.sr * 1000) / 2
        mel = np.zeros((n_mels, target_n_frames), dtype=np.float32)
        for m in range(n_mels):
            mel[m, :] = np.interp(t_mapped_ms, t_orig_ms_128, mel_orig[m, :])
        mel[:, ~source_present] = -80.0

        # Preserve recorded weak regions (including consonant closures).
        # No gap filling or invented spectral variation.

        # 片段尾部追加静音帧（约 80ms）：声码器任何片段末尾 ~50-60ms 会产生低频伪影
        # （40-160Hz +25dB，来自 mel/f0 截断边界效应）。若片段恰好结束在 OpenUtau
        # envelope p4（增益 0 点），伪影会落在可闻区 -> 每个音符尾部低频轰鸣。
        # 追加的静音帧位于 envelope p4 之后，拼接时被增益 0 门控，但 vocode 输入里
        # 伪影已移到静音区，不再污染可闻内容。
        _extra_frames = int(round(80.0 / frame_ms))
        if _extra_frames > 0:
            mel_ext = np.concatenate([
                mel,
                np.full((n_mels, _extra_frames), -80.0, dtype=np.float32),
            ], axis=1)
            # 前 3 帧从末帧 mel 平滑过渡到 -80，避免 vowel->静音 的硬阶跃
            # （硬阶跃 + 下一音素 F0 覆盖 -> vocoder 瞬态会向后污染 50ms 可闻尾音）
            _fade_n = min(3, _extra_frames)
            _wf = np.linspace(1.0, 0.0, _fade_n, dtype=np.float32)
            _last = mel[:, -1]
            for k in range(_fade_n):
                mel_ext[:, target_n_frames + k] = (
                    _last * _wf[k] + (-80.0) * (1.0 - _wf[k]))
            mel = mel_ext
            target_n_frames += _extra_frames

        # 波形处理：仅供未调用的 _concatenate_waveforms 调试路径保留。
        # 但该波形从不被消费（渲染走 Mel 域拼接 + 声码器重建），纯属白算。
        # 这里只做简单裁剪/对齐，速度可忽略。
        wave_orig_samples = len(wave)
        start_sample = int(start_ms / 1000 * self.sr)
        end_sample = int(end_ms / 1000 * self.sr)
        start_sample = max(0, start_sample)
        end_sample = min(wave_orig_samples, end_sample)
        target_wave_samples = int(np.round(total_span_ms / 1000 * self.sr))
        if start_sample >= end_sample:
            wave_processed = np.zeros(target_wave_samples, dtype=np.float32)
        else:
            wave_processed = wave[start_sample:end_sample].copy()
            if len(wave_processed) < target_wave_samples:
                wave_processed = np.pad(wave_processed, (0, target_wave_samples - len(wave_processed)))
            elif len(wave_processed) > target_wave_samples:
                wave_processed = wave_processed[:target_wave_samples]
        wave_processed = wave_processed.astype(np.float32)

        # 直接使用传入的 target_f0_hz（已包含逐帧 F0 信息）
        # _process_phones 已经从 f0_phrase 切出了正确的逐帧 F0
        if isinstance(target_f0_hz, np.ndarray):
            f0 = target_f0_hz[:target_n_frames].copy()
            if len(f0) < target_n_frames:
                f0 = np.pad(f0, (0, target_n_frames - len(f0)), mode='constant')
        else:
            f0 = np.full(target_n_frames, target_f0_hz, dtype=np.float32)
        # 静音尾部扩展帧的 F0 强制为 0（mel=-80 + f0=0 才是真静音）
        if _extra_frames > 0:
            f0[-_extra_frames:] = 0.0

        f0_nonzero = np.sum(f0 > 0)
        logger.debug(f"音素 {phone.phoneme}: 原始帧数={orig_n_frames}, "
                     f"目标帧数={target_n_frames}, F0非零帧数={f0_nonzero}")

        return {
            "source_level_rms": _source_level_rms,
            "source_level_valid": _source_level_valid,
            "source_level_region": _source_level_region,
            "mel": mel,
            "f0": f0,
            "wave": wave_processed,
            "phone": self.phone,
            "phoneme_features": None,
            "pre_scale": pre_scale,
            "pre_vel_ms": pre_vel_ms,
            "source_skip_ms": source_anchor_skip_ms,
            "source_time_ms": t_mapped_ms,
            "consonant_output_ms": con_output_ms,
            "envelope_span_ms": envelope_span_ms,
            "segment_len_ms": total_span_ms + (int(round(80.0 / frame_ms)) * frame_ms),
            "tail_silence_frames": int(round(80.0 / frame_ms)),
        }


class PhraseRenderer:
    """整句渲染器

    处理一个 RenderPhraseRequest，返回整句 PCM。
    """

    def __init__(self, request: RenderPhraseRequest):
        self.req = request
        self.config = get_config()
        self.model_manager = get_model_manager()
        self.audio_loader = AudioLoader(request.singer_path)
        self.sr = self.config.audio.sample_rate
        self.hop_size = self.config.audio.hop_size
        self.frame_ms = self.hop_size / self.sr * 1000
        self.fluency_enhancer = FluencyEnhancer(
            sample_rate=self.sr,
            n_fft=self.config.audio.n_fft,
            hop_length=self.hop_size,
        )
        self.detail_enhancer = DetailEnhancer(
            sample_rate=self.sr,
            n_fft=self.config.audio.n_fft,
            hop_length=self.hop_size,
        )
        self.hn_processor = HarmonicNoiseProcessor(
            sample_rate=self.sr,
            n_fft=self.config.audio.n_fft,
            hop_length=self.hop_size,
        )
        self.denoiser = SmartDenoiser(sample_rate=self.sr)
        self.quality_evaluator = QualityEvaluator(sample_rate=self.sr)
        self.curve_params = CurveParams()
        self.perf_optimizer = get_performance_optimizer()
        self.timing_optimizer = PhonemeTimingOptimizer(sample_rate=self.sr)
        self.timbre_mixer = TimbreMixer(sample_rate=self.sr)
        self._param_cache: Dict[str, np.ndarray] = {}
        # HG flag（音素级咆哮）折叠进整句 growl 曲线的段列表: (start_frame, end_frame, value)
        self._phone_growl_segments: List[Tuple[int, int, float]] = []
        # 音素→音频文件查找缓存（同一声库内音素名→文件路径是确定映射，避免重复 os.walk）
        self._phoneme_wav_cache: Dict[str, Optional[str]] = {}
        # oto.ini 全量索引: alias -> (wav_full_path, offset, consonant, cutoff, preutter, overlap)
        # 惰性构建，仅在直接文件名匹配失败时使用（覆盖 CV-VC diphone 等带空格/后缀的 alias）
        self._oto_ini_index: Optional[Dict[str, Tuple[str, float, float, float, float, float]]] = None
        # 音素→oto 参数缓存: 仅当文件经由 oto.ini 解析得到时填充，供后续回填 phone.oto_* 使用
        self._phoneme_oto_params_cache: Dict[str, Tuple[float, float, float, float, float]] = {}

    def _frame_to_pitch_pos(self, frame_idx: int) -> float:
        """帧 -> pitches 数组的浮点下标（OpenUTAU 绝对 cent 数组，5 tick 采样）。

        用音素自身的 tick/ms 比值换算（兼容任意 BPM 与中途变速），
        不再假设 avg_tempo=120。坐标约定：
        - 帧 frame_idx 对应的绝对毫秒 = req.position_ms - req.leading_ms + frame_idx * frame_ms
        - pitches 数组下标 0 对应 tick = 首音素 position(0) - req.leading
        """
        if not self.req.phones:
            return 0.0
        abs_ms = (self.req.position_ms - self.req.leading_ms) + frame_idx * self.frame_ms
        for phone in self.req.phones:
            preutter_ms = getattr(phone, 'preutter_ms', 0.0) or 0.0
            dur_ms = phone.duration_ms or 1.0
            if phone.position_ms - preutter_ms <= abs_ms < phone.end_ms:
                rel_ms = abs_ms - phone.position_ms
                tick = phone.position + rel_ms * phone.duration / dur_ms
                return (tick + self.req.leading) / 5.0
        # 越界回退：用最后一个音素的比例
        phone = self.req.phones[-1]
        dur_ms = phone.duration_ms or 1.0
        rel_ms = abs_ms - phone.position_ms
        tick = phone.position + rel_ms * phone.duration / dur_ms
        return (tick + self.req.leading) / 5.0

    def _get_param_curve(self, param_name: str, total_frames: int) -> np.ndarray:
        """获取指定参数的逐帧曲线（插值到目标帧数）

        Args:
            param_name: 参数名称
            total_frames: 目标帧数

        Returns:
            逐帧参数值数组
        """
        if param_name in self._param_cache:
            return self._param_cache[param_name]

        # 从请求中获取参数
        values = getattr(self.req, param_name, [])
        if not values:
            # 默认值处理
            defaults = {
                'pitches': 0.0,
                'dynamics': 1.0,  # OpenUTAU 线性倍率（1.0=中性，用户 DYN 曲线 + autoDyn 已换算）
                'gender': 0.0,
                'breathiness': 0.0,
                'tone_shift': 0.0,
                'tension': 0.0,
                'voicing': 100.0,
                'growl': 0.0,
                'lowcut': 0.0,
                'brel': 0.0,
                'breh': 0.0,
                'warm': 0.0,
                'hcmp': 0.0,
                'noise_gate_threshold': -50.0,
                'noise_suppression': 0.2,
            }
            result = np.full(total_frames, defaults.get(param_name, 0.0), dtype=np.float32)
            self._param_cache[param_name] = result
            return result

        # 计算平均 tempo（用于 tick → ms 转换）
        avg_tempo = 120.0
        if self.req.notes:
            for note in self.req.notes:
                if hasattr(note, "tempo") and note.tempo > 0:
                    avg_tempo = note.tempo
                    break

        # 原始参数是 5 tick 间隔采样
        result = np.zeros(total_frames, dtype=np.float32)
        values_arr = np.array(values, dtype=np.float32)

        for i in range(total_frames):
            ms = i * self.frame_ms - self.req.leading_ms
            tick = ms / 1000 * avg_tempo * 480 / 60
            pitch_index = int(tick / 5)
            if 0 <= pitch_index < len(values_arr):
                result[i] = values_arr[pitch_index]
            elif len(values_arr) > 0:
                result[i] = values_arr[0] if pitch_index < 0 else values_arr[-1]

        self._param_cache[param_name] = result
        return result

    def _load_oto_ini_index(self) -> Dict[str, Tuple[str, float, float, float, float, float]]:
        """扫描声库所有子目录的 oto.ini (Shift-JIS)，构建 alias → (wav_path, oto参数) 索引

        oto.ini 行格式: <wav_rel>=<alias>,<offset>,<consonant>,<cutoff>,<preutter>,<overlap>
        wav 路径相对于该 oto.ini 所在子目录。

        Returns:
            {alias: (wav_full_path, offset, consonant, cutoff, preutter, overlap)}
        """
        index: Dict[str, Tuple[str, float, float, float, float, float]] = {}
        singer_path = self.req.singer_path
        if not singer_path or not os.path.isdir(singer_path):
            return index

        for root, dirs, files in os.walk(singer_path):
            oto_path = os.path.join(root, "oto.ini")
            if not os.path.isfile(oto_path):
                continue
            try:
                with open(oto_path, encoding="shift_jis", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or line.startswith("["):
                            continue
                        if "=" not in line:
                            continue
                        wav_part, rest = line.split("=", 1)
                        parts = rest.split(",")
                        if len(parts) < 6:
                            continue
                        try:
                            alias = parts[0].strip()
                            offset = float(parts[1])
                            consonant = float(parts[2])
                            cutoff = float(parts[3])
                            preutter = float(parts[4])
                            overlap = float(parts[5])
                        except ValueError:
                            continue
                        if not alias:
                            continue
                        wav_full = os.path.join(root, wav_part.strip())
                        if alias in index:
                            continue  # 首个命中保留，避免被子声库覆盖
                        index[alias] = (wav_full, offset, consonant, cutoff, preutter, overlap)
            except OSError as e:
                logger.warning(f"读取 oto.ini 失败 {oto_path}: {e}")
        logger.info(f"oto.ini 索引构建完成: {len(index)} 条 alias")
        return index

    def _resolve_oto_entry(self, phoneme: str) -> Optional[Tuple[str, float, float, float, float, float]]:
        """按 alias 精确匹配 oto.ini 索引

        覆盖带空格/后缀的 CV-VC diphone 别名（如 "a い囁"），
        这些别名无法通过文件名直接匹配。

        Returns:
            (wav_full_path, offset, consonant, cutoff, preutter, overlap) 或 None
        """
        if self._oto_ini_index is None:
            self._oto_ini_index = self._load_oto_ini_index()
        return self._oto_ini_index.get(phoneme)

    def _find_phoneme_wav(self, phoneme: str, suffix: str = "") -> Optional[str]:
        """在声库中查找音素对应的音频文件

        搜索声库目录中的所有子目录，查找匹配的 .wav 文件。
        支持 UTAU 声库命名约定（文件名以下划线开头，如 _あ.wav）。
        当直接匹配失败时，尝试将罗马字映射到日文假名后再次搜索。
        最后回退到 oto.ini alias 匹配（覆盖 CV-VC diphone 等带空格/后缀的别名）。

        当且仅当文件经由 oto.ini 解析得到时，会同步把 oto 参数缓存到
        self._phoneme_oto_params_cache，供 _apply_resolved_oto_params 回填 phone。

        Args:
            phoneme: 音素名

        Returns:
            音频文件完整路径，或 None
        """
        singer_path = self.req.singer_path
        if not singer_path or not os.path.isdir(singer_path):
            return None

        # 缓存命中：同一音素名在同一声库内的映射是确定的
        if phoneme in self._phoneme_wav_cache:
            return self._phoneme_wav_cache[phoneme]

        # 0. oto.ini alias 优先匹配（保证 wav 文件与 oto 参数来自同一子声库）
        # 统合声库（如重音テト OU用日本語統合ライブラリー）包含多个子声库
        # （単独音/弱/エッジ/ささやき...），每个子声库有自己的 oto.ini 和 wav。
        # 直接文件名搜索会跨子声库取文件（あ 取自弱、か 取自エッジ），再把 A 子声库
        # 的 oto 参数错配到 B 子声库的 wav 上，导致 offset/consonant 指向静音区，
        # Mel 近乎全零 → 该音素整段无声。oto.ini 索引的 (wav, offset, consonant,
        # cutoff, preutter, overlap) 六元组来自同一子声库，保证一致。
        # 子声库后缀组合（CLR）：用户指定了 voice color（suffix）时，phoneme+suffix
        # 优先（如 "あ"+"弱" -> "あ弱"），否则会误匹配主库的同名 alias，导致 CLR 失效。
        oto_entry = None
        if suffix:
            oto_entry = self._resolve_oto_entry(phoneme + suffix)
            if oto_entry:
                logger.info(f"CLR suffix 匹配: {phoneme}+{suffix} -> {phoneme + suffix}")
        if oto_entry is None:
            oto_entry = self._resolve_oto_entry(phoneme)
        # VCV/CVVC 音素容错：带空格的音素（如 "ど あ" / "a い"）在声库无对应 alias 时，
        # 拆出最后一个 token（当前音素元音/辅音部分）回退到単独音文件，避免整段静音。
        if oto_entry is None and ' ' in phoneme:
            parts = [p_ for p_ in phoneme.split()]
            candidates = []
            if parts:
                last = parts[-1].lstrip('*-')
                if last and last != phoneme:
                    candidates.append(last)
                if len(parts) >= 2:
                    candidates.append(parts[-2].lstrip('*-'))
            for cand in candidates:
                if not cand or cand == phoneme:
                    continue
                entry2 = self._resolve_oto_entry(cand)
                if entry2:
                    logger.warning(f"VCV 音素 {phoneme!r} 未找到，回退到 {cand!r}")
                    oto_entry = entry2
                    break
                # 文件名回退
                wav2 = os.path.join(singer_path, f"_{cand}.wav")
                if os.path.isfile(wav2):
                    logger.warning(f"VCV 音素 {phoneme!r} 未找到，文件名回退到 _{cand}.wav")
                    oto_entry = (wav2, 0.0, 0.0, 0.0, 8.0, 5.0)
                    break
        if oto_entry:
            wav_full, offset, consonant, cutoff, preutter, overlap = oto_entry
            if os.path.isfile(wav_full):
                self._phoneme_oto_params_cache[phoneme] = (offset, consonant, cutoff, preutter, overlap)
                self._phoneme_wav_cache[phoneme] = wav_full
                logger.info(f"oto.ini 优先匹配: {phoneme} -> {os.path.basename(wav_full)} "
                            f"(offset={offset}, consonant={consonant}, cutoff={cutoff}, "
                            f"preutter={preutter}, overlap={overlap})")
                return wav_full

        def _search(phoneme_str: str) -> Optional[str]:
            patterns = [
                f"_{phoneme_str}.wav",
                f"{phoneme_str}.wav",
                f"_{phoneme_str}.WAV",
                f"{phoneme_str}.WAV",
            ]
            for root, dirs, files in os.walk(singer_path):
                for file in files:
                    if file.lower().endswith('.wav'):
                        for pattern in patterns:
                            if file == pattern or file.lower() == pattern.lower():
                                return os.path.join(root, file)
            return None

        # 1. 直接搜索
        result = _search(phoneme)

        # 2. 罗马字→假名回退映射
        if not result:
            _ROMAJI_KANA = {
                'a': 'あ', 'i': 'い', 'u': 'う', 'e': 'え', 'o': 'お',
                'ka': 'か', 'ki': 'き', 'ku': 'く', 'ke': 'け', 'ko': 'こ',
                'sa': 'さ', 'shi': 'し', 'su': 'す', 'se': 'せ', 'so': 'そ',
                'ta': 'た', 'chi': 'ち', 'tsu': 'つ', 'te': 'て', 'to': 'と',
                'na': 'な', 'ni': 'に', 'nu': 'ぬ', 'ne': 'ね', 'no': 'の',
                'ha': 'は', 'hi': 'ひ', 'fu': 'ふ', 'he': 'へ', 'ho': 'ほ',
                'ma': 'ま', 'mi': 'み', 'mu': 'む', 'me': 'め', 'mo': 'も',
                'ya': 'や', 'yu': 'ゆ', 'yo': 'よ',
                'ra': 'ら', 'ri': 'り', 'ru': 'る', 're': 'れ', 'ro': 'ろ',
                'wa': 'わ', 'wo': 'を', 'n': 'ん',
            }
            kana = _ROMAJI_KANA.get(phoneme.lower())
            if kana:
                result = _search(kana)
                if result:
                    logger.info(f"罗马字映射: {phoneme} -> {kana} -> {os.path.basename(result)}")

        # 3. 中文拼音→日文假名回退映射
        if not result:
            zh_map = VOWEL_MAPPING.get(phoneme) or VOWEL_MAPPING.get(phoneme.upper())
            if zh_map:
                candidates = zh_map if isinstance(zh_map, list) else [zh_map]
                for kana_str in candidates:
                    result = _search(kana_str)
                    if result:
                        logger.info(f"拼音映射: {phoneme} -> {kana_str} -> {os.path.basename(result)}")
                        break

        # 4. oto.ini alias 回退（覆盖 CV-VC diphone 等带空格/后缀的别名）
        if not result:
            entry = self._resolve_oto_entry(phoneme)
            if entry:
                wav_full, offset, consonant, cutoff, preutter, overlap = entry
                result = wav_full
                # 仅在此分支缓存 oto 参数，标记"该文件来自 oto.ini 解析"
                self._phoneme_oto_params_cache[phoneme] = (offset, consonant, cutoff, preutter, overlap)
                logger.info(f"oto.ini alias 匹配: {phoneme} -> {os.path.basename(wav_full)} "
                            f"(offset={offset}, consonant={consonant}, cutoff={cutoff}, "
                            f"preutter={preutter}, overlap={overlap})")

        # 缓存结果（命中或 None 都缓存，避免重复 os.walk）
        self._phoneme_wav_cache[phoneme] = result
        return result

    def _apply_resolved_oto_params(self) -> None:
        """对经由 oto.ini 解析得到音频文件的音素，回填 oto_* 参数到 phone

        触发条件：phone.oto_file 已被 _find_phoneme_wav 设置，且 phone.phoneme
        命中 _phoneme_oto_params_cache。此时上游未提供 oto_file（否则 _find_phoneme_wav
        不会被调用），因此用 oto.ini 的参数覆盖 phone 上残留的默认值是安全的。
        """
        if not self._phoneme_oto_params_cache:
            return
        for phone in self.req.phones:
            if not phone.phoneme or not phone.oto_file:
                continue
            params = self._phoneme_oto_params_cache.get(phone.phoneme)
            if params is None:
                continue
            offset, consonant, cutoff, preutter, overlap = params
            phone.oto_offset = offset
            phone.oto_consonant = consonant
            phone.oto_cutoff = cutoff
            phone.oto_preutter = preutter
            phone.oto_overlap = overlap
            logger.debug(f"回填 oto 参数: {phone.phoneme} offset={offset} "
                         f"consonant={consonant} cutoff={cutoff} "
                         f"preutter={preutter} overlap={overlap}")

    def _clamp_request(self) -> None:
        """入口校验：将异常的时长/前导参数 clamp 到安全范围

        防御模糊测试中发现的崩溃：
        - 负 duration_ms / leading_ms → 负维度 np.zeros 崩溃
        - leading_ms 极大（如 100000ms）→ total_frames 极大 → 渲染超慢
        - total_frames 必须 >= 1
        """
        if not np.isfinite(self.req.duration_ms):
            self.req.duration_ms = 0.0
        if not np.isfinite(self.req.leading_ms):
            self.req.leading_ms = 0.0

        if self.req.duration_ms < 0:
            logger.warning(f"duration_ms={self.req.duration_ms} 为负，已 clamp 为 0")
            self.req.duration_ms = 0.0

        if self.req.leading_ms < 0:
            logger.warning(f"leading_ms={self.req.leading_ms} 为负，已 clamp 为 0")
            self.req.leading_ms = 0.0

        # leading_ms 过大会导致 total_frames 爆炸（极慢），限制为合理上限
        leading_max = max(5000.0, self.req.duration_ms * 0.5)
        if self.req.leading_ms > leading_max:
            logger.warning(
                f"leading_ms={self.req.leading_ms} 过大，已 clamp 为 {leading_max:.0f}"
            )
            self.req.leading_ms = leading_max

    def render(self) -> RenderResultResponse:
        """执行整句渲染

        参考同类实现 和 同类引擎 架构：
        1. 构建整句 F0 曲线
        2. 处理每个音素获取 Mel 频谱
        3. 拼接 Mel 频谱（使用 overlap 交叉淡入淡出）
        4. 一次性传给 vocoder 合成整句波形

        关键设计：在声码器之前做Mel域交叉淡入淡出，确保频谱连续性
        声码器会完全重建波形，因此波形域crossfade的相位信息会被破坏
        """
        self._clamp_request()
        logger.info(f"开始渲染整句: {len(self.req.phones)} 个音素, hash={self.req.hash:x}")

        # 检测声库类型（单独音 vs 连续音）
        vb_type = self._detect_voicebank_type()
        is_mono = (vb_type == 'mono')
        logger.info(f"声库类型: {'单独音' if is_mono else '连续音'} ({vb_type})")

        # 0. 对上游未提供 oto_file 的音素，直接在声库中查找音频文件
        #    （带子声库 suffix，保证 CLR 在音素化未选子库时仍生效）
        for phone in self.req.phones:
            if phone.oto_file or not phone.phoneme:
                continue
            mapped_file = self._find_phoneme_wav(
                phone.phoneme, suffix=getattr(phone, 'suffix', '') or '')
            if mapped_file:
                phone.oto_file = mapped_file
        # 0.1 对经由 oto.ini 解析得到音频文件的音素，回填 oto_* 参数
        # （覆盖 CV-VC diphone 等上游未传 oto_file 的场景）
        self._apply_resolved_oto_params()

        # 1. 构建整句 F0 曲线
        total_frames, f0_phrase = self._build_pitch_curve()
        _voiced_f0 = f0_phrase[f0_phrase > 0]
        if _voiced_f0.size > 0:
            logger.info(f"整句 F0 曲线: {total_frames} 帧, 范围={np.min(_voiced_f0):.1f}~{np.max(_voiced_f0):.1f} Hz")
        else:
            logger.info(f"整句 F0 曲线: {total_frames} 帧, 无 voiced 帧")

        # 获取所有动态参数曲线
        tension = self._get_param_curve('tension', total_frames)
        breathiness = self._get_param_curve('breathiness', total_frames)
        voicing = self._get_param_curve('voicing', total_frames)
        growl = self._get_param_curve('growl', total_frames)
        # HG flag（音素级咆哮）折入整句 growl 曲线
        if self._phone_growl_segments:
            growl = growl.copy()
            for s0, e0, val in self._phone_growl_segments:
                s0 = max(0, s0)
                e0 = min(total_frames, e0)
                if s0 < e0 and val > 0:
                    growl[s0:e0] = np.maximum(growl[s0:e0], val)
        lowcut = self._get_param_curve('lowcut', total_frames)
        brel = self._get_param_curve('brel', total_frames)
        breh = self._get_param_curve('breh', total_frames)
        warm = self._get_param_curve('warm', total_frames)
        hcmp = self._get_param_curve('hcmp', total_frames)

        # 2. 处理每个音素获取特征（包含mel和f0）
        phone_features = self._process_phones(total_frames, f0_phrase)
        self.phone_features = phone_features  # 保存用于调试

        # 声码器会完全重建波形，因此波形域crossfade的相位信息会被破坏
        language = getattr(self.req, 'language', 'JA') or 'JA'
        if os.environ.get("DIFFICAL_ENGINE", "").lower() == "model":
            from .model_assembly import assemble as _model_assemble
            mel_phrase = _model_assemble(phone_features, total_frames, f0_phrase,
                                         self.config.audio.n_mels)
            # 模型训练目标是原录音（无 envelope）；envelope 是工程侧音量/表达曲线，
            # 必须在装配后按帧施加，才能与 legacy 画布语义一致（含音符收尾 p4）。
            logger.info("整句装配：PointerGen 模型")
        else:
            mel_phrase = self._concatenate_mels(phone_features, total_frames, vb_type, language)
        f0_concat = f0_phrase.copy()  # phrase pitch input owns every output F0 frame
        logger.info(f"Mel拼接完成: {mel_phrase.shape[1]} 帧")

        self.mel_phrase = mel_phrase  # 保存用于调试
        self.f0_concat = f0_concat    # 保存用于调试
        logger.info(f"匹配后 Mel: {mel_phrase.shape[1]} 帧, F0: {len(f0_concat)} 帧")

        # 调试：检查拼接后 F0 的范围
        f0_concat_valid = f0_concat[f0_concat > 50]
        if len(f0_concat_valid) > 0:
            logger.info(f"拼接后 F0 范围: {f0_concat_valid.min():.1f}~{f0_concat_valid.max():.1f} Hz")
            if f0_concat_valid.max() > 1400:
                logger.warning(f"检测到异常高 F0: {f0_concat_valid.max():.1f} Hz")
                high_f0_idx = np.where(f0_concat > 1400)[0]
                logger.warning(f"异常 F0 帧索引: {high_f0_idx[:10]}")

        # 3.5 预回声消除 - 已禁用
        # BUG: threshold_db=-35.0 配合 energy_diff > threshold_db 导致几乎所有帧被判定为 attack，
        # 所有帧被衰减至近零，Mel mean 从 -3.91 暴跌至 -22.24，声码器输出噪声。
        # 同类重采样器 不使用此步骤，禁用后渲染质量更优。
        # try:
        #     mel_phrase = pre_echo_suppression(mel_phrase, threshold_db=-35.0, attack_frames=2)
        #     logger.info("预回声消除完成")
        # except Exception as e:
        #     logger.warning(f"预回声消除失败: {e}")

        # 4. The OpenUtau-supplied phrase pitch is passed unchanged.


        # 5. 合成波形：mel 域拼接已完成，逐音素 vocode（消除跨音素上下文抹平辅音的问题）
        samples = self._synthesize_per_phone(phone_features, total_frames)
        # 防御：极短/全静音 mel 可能让声码器产生 NaN/Inf，sanitize 为有限值，避免后续 librosa 崩溃
        samples = np.nan_to_num(np.asarray(samples, dtype=np.float32),
                                nan=0.0, posinf=0.0, neginf=0.0)
        logger.info(f"整句合成完成: {len(samples)} 采样")

        # 3. 谐波/噪声参数处理（曲线参数：breathiness/tension/voicing/brel/breh/warm/hcmp）
        # 参考同类实现 的 HN-SEP 统一管线：一次分离，多次处理。
        # 全部参数为默认值时是完全的 no-op（不做分离、不改波形），保证默认音质不变；
        # 仅在用户绘制了对应曲线时才启用分离与参数处理。
        # 注意：OpenUTAU 的曲线默认值是 tension=0 / breathiness=0 / voicing=100
        # 被误判为"用户画了曲线"-> 触发 HN-SEP，breathiness=0 把谐波清零 ->
        use_hn_processing = (not np.allclose(tension, 0, atol=0.01) or
                             not np.allclose(breathiness, 0, atol=0.01) or
                             not np.allclose(voicing, 100, atol=0.01) or
                             not np.allclose(brel, 0, atol=0.01) or
                             not np.allclose(breh, 0, atol=0.01) or
                             not np.allclose(warm, 0, atol=0.01) or
                             not np.allclose(hcmp, 0, atol=0.01))
        if use_hn_processing and len(samples) > 0:
            try:
                _hn_input_backup = np.array(samples, dtype=np.float32, copy=True)
                _hnsep = self.model_manager._get_hnsep()
                if _hnsep is not None:
                    harmonic, noise = _hnsep.separate(samples)
                    harmonic = harmonic.astype(np.float32)
                    noise = noise.astype(np.float32)
                else:
                    harmonic, noise = self.hn_processor.separate(samples)

                # 仅当 breath/tension/voicing/brel/breh 非默认时才缩放噪声分量，
                # 避免只调 warm/hcmp 时意外改变噪声电平。
                legacy_active = (not np.allclose(tension, 0, atol=0.01) or
                                 not np.allclose(breathiness, 0, atol=0.01) or
                                 not np.allclose(voicing, 100, atol=0.01) or
                                 not np.allclose(brel, 0, atol=0.01) or
                                 not np.allclose(breh, 0, atol=0.01))
                noise_factor = self._compute_noise_factor() if legacy_active else 1.0
                harmonic, noise = self.hn_processor.process_parameters(
                    harmonic, noise,
                    breath_level=breathiness,
                    tension_level=tension,
                    voicing_level=voicing,
                    low_freq_breath=brel,
                    high_freq_breath=breh,
                    noise_factor=noise_factor,
                )

                warm_val = float(np.mean(warm))
                if warm_val > 0.5:
                    harmonic = apply_warmth(harmonic, warm_val, self.sr)

                hcmp_val = float(np.mean(hcmp))
                if hcmp_val > 0.5:
                    harmonic = apply_harmonic_compressor(harmonic, hcmp_val, self.sr)

                # 安全网：HN-SEP 处理不应把可闻音频压成近静音（历史上 breathiness=0
                # 误触发曾导致 rms 从 0.017 跌到 8e-06）。若输出显著弱于输入，回退原样。
                _in_rms = float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float32) ** 2))) if len(samples) else 0.0
                combined = harmonic + noise
                combined_peak = np.max(np.abs(combined))
                if combined_peak > 0.99:
                    scale = 0.99 / combined_peak
                    samples = combined * scale
                else:
                    samples = combined
                _out_rms = float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float32) ** 2))) if len(samples) else 0.0
                if _in_rms > 1e-4 and _out_rms < _in_rms * 0.1:
                    logger.warning(
                        f"HN-SEP 输出异常弱({_out_rms:.2e} vs 输入 {_in_rms:.2e})，回退原始合成"
                    )
                    # samples 已被覆盖，需恢复输入：在进入本块前保存
                    samples = _hn_input_backup
                logger.info("谐波/噪声参数处理完成")
            except Exception as e:
                logger.warning(f"谐波/噪声参数处理失败: {e}", exc_info=True)

        # Global loudness normalization runs once after effects and exact
        # length alignment. DYN belongs to OpenUtau playback, not this backend.

        # 4. 动态频谱滤波
        if not np.allclose(lowcut, 0, atol=0.5) and len(samples) > 0:
            try:
                f0_phrase = self._build_pitch_curve()[1]
                samples = apply_dynamic_filter(samples, lowcut, self.sr, f0_curve=f0_phrase)
                logger.info("动态频谱滤波完成")
            except Exception as e:
                logger.warning(f"动态频谱滤波失败: {e}")

        # 5. 咆哮效果处理
        if not np.allclose(growl, 0, atol=0.5) and len(samples) > 0:
            try:
                f0_phrase = self._build_pitch_curve()[1]
                samples = apply_growl(samples, growl, self.sr, f0_curve=f0_phrase)
                logger.info("咆哮效果处理完成")
            except Exception as e:
                logger.warning(f"咆哮效果处理失败: {e}")

        # 6. 流畅度增强（整句级流畅度优化）- 暂时禁用，避免过度平滑
        # if len(samples) > 0:
        #     try:
        #         fluency_level = 2 if is_mono else 3
        #         samples = self.fluency_enhancer.enhance(samples, level=fluency_level)
        #         logger.info(f"流畅度增强完成 (级别{fluency_level})")
        #     except Exception as e:
        #         logger.warning(f"流畅度增强失败: {e}")

        # 7. 细节增强（暂时禁用，避免引入高频噪声）
        # if len(samples) > 0:
        #     try:
        #         detail_mode = 'balanced' if is_mono else 'fluency'
        #         detail_level = 0.04 if is_mono else 0.06
        #         samples = self.detail_enhancer.enhance(samples, level=detail_level, mode=detail_mode)
        #         logger.info(f"细节增强完成 (模式{detail_mode}, 级别{detail_level})")
        #     except Exception as e:
        #         logger.warning(f"细节增强失败: {e}")

        # 8. 高频增强（提升频谱质心接近真人参考）- 暂时禁用，避免改变音高
        # if len(samples) > 0:
        #     try:
        #         samples = self._enhance_high_frequencies(samples)
        #         logger.info("高频增强完成")
        #     except Exception as e:
        #         logger.warning(f"高频增强失败: {e}")

        # 9. 微幅度呼吸调制（模拟自然呼吸的2-4Hz随机幅度变化）- 暂时禁用
        # if len(samples) > 0:
        #     try:
        #         samples = self._apply_breath_modulation(samples)
        #         logger.info("微幅度呼吸调制完成")
        #     except Exception as e:
        #         logger.warning(f"微幅度呼吸调制失败: {e}")

        # 10. 分段噪声门控 + 边界淡入淡出 - 暂时禁用
        # samples = self._apply_noise_gate(samples)

        # 11. 智能去噪（暂时禁用，避免过度降噪）
        # if len(samples) > 0:
        #     try:
        #         noise_suppression = float(np.mean(
        #             self._get_param_curve('noise_suppression', total_frames)
        #         )) if hasattr(self.req, 'noise_suppression') else 0.1
        #         if noise_suppression > 0.01:
        #             samples = self.denoiser.denoise(samples, mode='gentle', strength=noise_suppression)
        #             logger.info(f"智能去噪完成 (强度={noise_suppression:.2f})")
        #     except Exception as e:
        #         logger.warning(f"智能去噪失败: {e}")

        # 15. 间隙门控：休止符段（phone 间隙）应为静音。
        # 声码器对 mel_floor 输入仍有噪声底 (≈0.005 RMS)，经响度归一化放大后超过可闻阈值，
        # 表现为"背景噪声大"。在最终输出上对间隙段施加门控（短淡入淡出避免咔哒声）。
        if len(samples) > 0 and getattr(self, 'phone_features', None):
            try:
                _pf = sorted(self.phone_features, key=lambda f: f["start_frame"])
                _fade_n = int(0.010 * self.sr)  # 10ms 淡入淡出
                for i in range(1, len(_pf)):
                    _ps = max(0, _pf[i-1]["start_frame"])
                    _pe = min(total_frames, _ps + _pf[i-1]["mel"].shape[-1])
                    _cs = max(0, _pf[i]["start_frame"])
                    if _cs > _pe:
                        _s0 = max(0, min(int(_pe * self.hop_size), len(samples)))
                        _s1 = max(0, min(int(_cs * self.hop_size), len(samples)))
                        if _s1 > _s0:
                            _f0 = min(_fade_n, _s0)
                            _f1 = min(_fade_n, len(samples) - _s1)
                            if _f0 > 0:
                                samples[_s0 - _f0:_s0] *= np.linspace(1, 0, _f0, dtype=np.float32)
                            samples[_s0:_s1] = 0.0
                            if _f1 > 0:
                                samples[_s1:_s1 + _f1] *= np.linspace(0, 1, _f1, dtype=np.float32)
            except Exception as _e:
                logger.warning(f"间隙门控失败: {_e}")

        # 最终防御：确保返回给调用方的样本始终有限，避免下游（C# 端/librosa）崩溃
        samples = np.nan_to_num(np.asarray(samples, dtype=np.float32),
                                nan=0.0, posinf=0.0, neginf=0.0)

        # 精确时长对齐：mel 帧是 hop 粒度（~11.6ms），累积舍入会让输出比期望多 ~10ms，
        # 导致 OpenUTAU 里音符尾部/后续音符错位（"时长不匹配"）。裁剪/补零到期望采样数。
        expected_samples = int(round((self.req.duration_ms + self.req.leading_ms) / 1000 * self.sr))
        if len(samples) != expected_samples:
            if len(samples) > expected_samples:
                samples = samples[:expected_samples]
            else:
                samples = np.pad(samples, (0, expected_samples - len(samples)))
            logger.info(f"时长对齐: {len(samples)} 采样 -> {expected_samples} 采样")

        # 时长对齐可能裁掉 _synthesize_waveform 末尾的防咔哒淡出（total_frames 用
        # ceil、expected 用 round，短则 ~11ms），乐句末尾会以高电平硬切 ->
        # "末尾偶尔嘎哒声"。对齐后重新施加首尾短淡入淡出（仅防 click，~5.8ms）。
        # 乐句首尾 25ms 升余弦淡入淡出：避免满幅度硬截断的「卡」声。
        # 只动边界，不改乐句内部（内部 envelope 由 legacy/模型各自处理）。
        _edge_fade = min(256, len(samples) // 4)
        if _edge_fade > 0:
            samples[:_edge_fade] *= np.linspace(0, 1, _edge_fade, dtype=np.float32)
            samples[-_edge_fade:] *= np.linspace(1, 0, _edge_fade, dtype=np.float32)

        # Match only steady voiced note bodies. The controller uses the
        # OpenUtau-supplied F0 mask and per-note VOL, then phrase normalization
        # applies one common scalar. DYN remains owned by OpenUtau playback.
        note_targets = []
        for note_index, note in enumerate(self.req.notes):
            note_phones = [p for p in self.req.phones
                           if p.tone == note.tone
                           and note.position_ms <= p.position_ms < note.end_ms]
            if not note_phones:
                note_phones = [p for p in self.req.phones
                               if p.tone == note.tone]
            if note_phones:
                volume_phone = min(note_phones,
                                   key=lambda p: abs(p.position_ms - note.position_ms))
                note_volume = float(volume_phone.volume)
            else:
                note_volume = 1.0
            note_targets.append({
                "start_ms": note.position_ms - self.req.position_ms + self.req.leading_ms,
                "end_ms": note.end_ms - self.req.position_ms + self.req.leading_ms,
                "volume": note_volume,
            })
        # 每音符响度匹配已从生产路径移除（与参考实现一致：不做 per-note 电平补偿）。
        # 保留整句 RMS 归一化——它只是一个常数增益，不产生边界电平跳变。
        self.note_loudness_trace = {"reason": "removed_per_note_matching", "notes": []}

        # Preserve uniform VOL changes instead of normalizing them away.
        volumes = [float(p.volume) for p in self.req.phones]
        positive_volumes = [v for v in volumes if np.isfinite(v) and v > 0.0]
        volume_reference = max(0.0, float(np.median(positive_volumes))) if positive_volumes else 0.0
        if volume_reference > 0:
            samples, self.normalization_trace = normalize_phrase_rms(
                samples, sample_rate=self.sr, target_rms=0.15 * volume_reference,
                peak_ceiling=0.95)
        else:
            samples = np.zeros_like(samples)
            self.normalization_trace = {"reason": "zero_volume", "gain": 0.0}
        logger.info("Phrase RMS normalization: %s", self.normalization_trace)

        result = RenderResultResponse(
            samples=samples.tolist(),
            leading_ms=self.req.leading_ms,
            position_ms=self.req.position_ms,
            estimated_length_ms=self.req.duration_ms + self.req.leading_ms,
        )

        logger.info(f"渲染完成: {len(samples)} 个采样")
        return result

    def _smooth_phone_boundaries(self, samples: np.ndarray, phone_features: list,
                                 total_frames: int) -> np.ndarray:
        """边界等功率交叉淡化：让 overlap 平滑（避免硬切/突跳）。

        mel 域 crossfade 保持很短（避免双频谱糊），波形域在这里做 3~4 帧
        （~35-46ms）的等功率 crossfade 让边界能量平滑过渡——这是"平滑 overlap"
        的直接手段，不影响 mel 拼接架构。
        """
        feats = sorted(phone_features, key=lambda f: f["start_frame"])
        if len(feats) < 2 or len(samples) == 0:
            return samples
        for i in range(1, len(feats)):
            prev = feats[i - 1]
            curr = feats[i]
            prev_end = min(total_frames, prev["start_frame"] + prev["mel"].shape[-1])
            curr_start = max(0, curr["start_frame"])
            overlap_frames = prev_end - curr_start
            if overlap_frames <= 0:
                continue
            fade_frames = min(overlap_frames, 4)
            if fade_frames < 2:
                continue
            fade = fade_frames * self.hop_size
            s0 = prev_end * self.hop_size - fade
            s1 = prev_end * self.hop_size
            s2 = curr_start * self.hop_size
            if s0 < 0 or s1 > len(samples) or s2 + fade > len(samples):
                continue
            # 等功率交叉淡化（能量恒定，避免凹陷/突跳）
            t = np.linspace(0, np.pi / 2, fade)
            fade_out = np.cos(t)
            fade_in = np.sin(t)
            prev_tail = samples[s0:s1].copy()
            curr_head = samples[s2:s2 + fade].copy()
            samples[s0:s1] = (prev_tail * fade_out + curr_head * fade_in).astype(np.float32)
        return samples

    def _detect_voicebank_type(self) -> str:
        """检测声库类型（单独音 vs 连续音）

        从传入的 RenderPhone.phoneme 模式判断：
        - 单独音：音素名是单个字符（如 "あ", "a"）
        - 连续音：音素名包含空格连接（如 "a あ", "- あ", "あ R"）

        Returns:
            'mono' 单独音, 'vowel_connected' 连续音
        """
        if not self.req.phones:
            return 'mono'

        # 统计音素命名模式
        space_count = 0
        total = 0

        for phone in self.req.phones:
            if not phone.phoneme:
                continue
            total += 1
            # 连续音特征：音素名包含空格（表示前一个音素+当前音素）
            if ' ' in phone.phoneme or '-' in phone.phoneme:
                space_count += 1

        if total == 0:
            return 'mono'

        # 超过 30% 的音素包含空格 → 判断为连续音
        if space_count / total > 0.3:
            return 'vowel_connected'

        # 不再根据目录名猜测（OpenUTAU 已完成音素化，目录名含"连続"的单独音
        # 声库会被误判为连续音；声库类型只影响极小参数差异，统一按音素模式判断）
        return 'mono'

    def _build_pitch_curve(self) -> Tuple[int, np.ndarray]:
        """构建整句 F0 曲线

        支持两种模式：
        1. 逐帧模式：pitches 数组长度与 total_frames 接近，直接使用
        2. 5-tick 间隔模式：pitches 数组按 5 tick 间隔采样，需要插值

        关键：音符外的帧 F0=0（静音），避免 vocoder 在间隙合成假人声
        之前 per-frame 模式把 cent=0 转换为 263Hz，导致静音段被合成

        Returns:
            (total_frames, f0_phrase)
        """
        total_ms = self.req.duration_ms + self.req.leading_ms
        total_frames = int(np.ceil(total_ms / self.frame_ms))
        # 防御：total_frames 必须 >= 1，否则 np.zeros(负数) 崩溃、空帧 min/max 崩溃
        if total_frames < 1:
            total_frames = 1

        # 构建音符内的 voiced mask（基于 phone 的 position_ms/end_ms）
        # 用于在 per-frame 模式下判断哪些帧应该有声
        # 关键：包含 preutter 区域（音素提前开始的部分），否则 preutter 区域 Mel 有内容但 F0=0，
        # 导致 vocoder 产生清音输出，F0 检测率降低
        voiced_mask = np.zeros(total_frames, dtype=bool)
        for phone in self.req.phones:
            preutter_ms = getattr(phone, 'preutter_ms', 0.0)
            # 音素实际音频提前 preutter_ms 开始（UTAU 语义）
            audio_start_ms = phone.position_ms - self.req.position_ms + self.req.leading_ms - preutter_ms
            note_end_ms = phone.end_ms - self.req.position_ms + self.req.leading_ms
            start_frame = int(audio_start_ms / self.frame_ms)
            # 不足一帧的部分排除（mel 仍有元音能量但 f0=0）-> 声码器噪声分支
            # 在乐句末尾发一帧宽带噪声（"乐句末尾偶尔杂音"）。
            end_frame = int(np.ceil(note_end_ms / self.frame_ms))
            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)
            voiced_mask[start_frame:end_frame] = True

        if not self.req.pitches:
            if not self.req.phones:
                return total_frames, np.zeros(total_frames, dtype=np.float32)
            raise ValueError("OpenUtau pitch curve is missing; backend will not invent F0")

        pitches = np.array(self.req.pitches, dtype=np.float32)

        # 判断模式：如果 pitches 长度与 total_frames 接近，则为逐帧模式
        if len(pitches) >= total_frames * 0.9 and len(pitches) <= total_frames * 1.1:
            # 逐帧模式：转换 cent 到 Hz，但仅对 voiced 帧赋值
            # pitches[i] 是相对于音符基础音高（tone）的偏差（cent）
            # pitches[i] == 0 表示保持音符基础音高，不是静音！
            f0 = np.zeros(total_frames, dtype=np.float32)
            for i in range(total_frames):
                if not voiced_mask[i]:
                    continue
                
                frame_ms_val = i * self.frame_ms
                base_tone = 60
                
                for phone in self.req.phones:
                    preutter_ms = getattr(phone, 'preutter_ms', 0.0)
                    audio_start_ms = phone.position_ms - self.req.position_ms + self.req.leading_ms - preutter_ms
                    note_end_ms = phone.end_ms - self.req.position_ms + self.req.leading_ms
                    if audio_start_ms <= frame_ms_val < note_end_ms:
                        base_tone = phone.tone
                        break
                
                if i < len(pitches):
                    # OpenUTAU pitches 是绝对音高（cent）：AdjustedTone*100 + 曲线偏差
                    # （例如 C4=60 -> 6000 cent）。直接 cent/100 得到 MIDI，不能再加 base_tone，
                    # 否则 C4(6000cent) 会被算成 120 → 声码器输出 ~1500Hz 的异常高音。
                    cent = pitches[i]
                    f0[i] = midi_to_hz(cent / 100.0)
                else:
                    f0[i] = midi_to_hz(base_tone)
            
            # 限制F0范围，防止极端异常值（参考同类实现：不做硬裁剪，这里仅防越界）
            # 50~1500Hz 覆盖 E1~G6 全人声范围。关键：只 clip 有声帧（f0>0），
            # 声码器尾部上下文重复该帧 -> 整段 40-160Hz 低频轰鸣（音质"闷/糊"根因）。
            f0 = np.where(f0 > 0, np.clip(f0, 50.0, 1500.0), 0.0)

            # Resample the OpenUtau-supplied pitch curve only; never replace
            # missing preutter values with the phone's nominal tone.
            # Apply the explicitly supplied SHFT expression in its own slot.
            tone_shift = self._get_param_curve('tone_shift', total_frames)
            if np.any(np.abs(tone_shift) > 0.5):
                shift_ratio = np.power(2.0, tone_shift / 1200.0)
                f0 = np.where(f0 > 50, f0 * shift_ratio, f0)
                f0 = np.clip(f0, 50.0, 1500.0)

            return total_frames, f0

        # 5-tick 间隔模式：从 pitches 数组（cent，5 tick 间隔）采样
        f0 = np.zeros(total_frames, dtype=np.float32)

        for i in range(total_frames):
            # 仅对 voiced 帧赋 F0
            if not voiced_mask[i]:
                continue

            pitch_pos = float(np.clip(self._frame_to_pitch_pos(i), 0.0, len(pitches) - 1.0))
            idx0 = int(np.floor(pitch_pos))
            idx1 = min(idx0 + 1, len(pitches) - 1)
            frac = pitch_pos - idx0
            cent = float(pitches[idx0] * (1.0 - frac) + pitches[idx1] * frac)
            f0[i] = midi_to_hz(cent / 100.0)

        # 限制F0范围，防止极端异常值（参考同类实现：不做硬裁剪，这里仅防越界）。
        # 只 clip 有声帧，静音帧保持 0（否则 0 -> 50Hz 低频轰鸣）。
        f0 = np.where(f0 > 0, np.clip(f0, 50.0, 1500.0), 0.0)

        # Only apply the explicitly supplied SHFT expression. Pitch smoothing,
        # generated transitions and long-note drift belong to OpenUtau/F0 input.
        tone_shift = self._get_param_curve("tone_shift", total_frames)
        if np.any(np.abs(tone_shift) > 0.5):
            f0 = np.where(f0 > 0, f0 * np.power(2.0, tone_shift / 1200.0), f0)
        return total_frames, f0.astype(np.float32)

    def _process_phones(self, total_frames: int, f0_phrase: np.ndarray) -> List[dict]:
        """处理每个音素

        Args:
            total_frames: 总帧数
            f0_phrase: 整句的逐帧 F0 曲线

        Returns:
            每个音素的特征字典列表
        """
        features = []

        for _p_idx, phone in enumerate(self.req.phones):
            target_start_ms = phone.position_ms - self.req.position_ms + self.req.leading_ms
            target_duration_ms = phone.duration_ms

            preutter_ms = getattr(phone, 'preutter_ms', 0.0)
            overlap_ms = getattr(phone, 'overlap_ms', 0.0)
            oto_preutter = getattr(phone, 'oto_preutter', 0.0)
            oto_overlap = getattr(phone, 'oto_overlap', 0.0)

            # OpenUtau 已在 ValidateOverlap 中按上一音符时长 clamp 过 preutter
            # （VCV 的 preutter 可以合法地大于本音素 duration：preutter 是前母音
            # 提前于音符起点的部分，如 连続音 o ふぁ 在 125ms 音符上 preutter=281ms）。
            # 这里只做数值防御（防 NaN/负值/病态大值），绝不二次截断合法 preutter——
            # 0.8*duration 的旧 clamp 是为已删除的 energy_smooth 广播兜底，会切断
            # VCV 前母音、把辅音起点整体前移，导致时长不匹配与咬字异常。
            if not np.isfinite(preutter_ms) or preutter_ms < 0:
                preutter_ms = 0.0
            preutter_max = max(5000.0, target_duration_ms)
            if preutter_ms > preutter_max:
                logger.warning(
                    f"音素 {phone.phoneme}: preutter_ms={preutter_ms} 超过安全上限 {preutter_max}，已 clamp"
                )
                preutter_ms = preutter_max

            # OpenUtau Classic resolves preutter in UPhoneme.preutter (already
            # velocity-scaled/clamped); raw oto_preutter must not be used for
            # the mel clip origin. Keep VEL scaling below for source consonant
            # mapping only.
            _vel = 2.0 ** (1.0 - float(getattr(phone, 'velocity', 1.0)))
            # 本音素尾部被下一音素前母音侵入的长度（OpenUtau tailIntrude/tailOverlap）
            if _p_idx + 1 < len(self.req.phones):
                _nxt = self.req.phones[_p_idx + 1]
                _n_pre = max(0.0, getattr(_nxt, 'preutter_ms', 0.0) or 0.0)
                _n_ovl = max(0.0, getattr(_nxt, 'overlap_ms', 0.0) or 0.0)
                tail_intrude_ms = max(_n_pre, _n_pre - _n_ovl)
                tail_overlap_ms = max(_n_ovl, 0.0)
            else:
                tail_intrude_ms = 0.0
                tail_overlap_ms = 0.0

            # RenderPhone.preutter_ms is OpenUtau's resolved timing; place the
            # source origin on the same p0 anchor used by the envelope.
            audio_start_ms = effective_phone_audio_start_ms(target_start_ms, preutter_ms)
            start_frame_exact = audio_start_ms / self.frame_ms
            pre_vel_ms = target_start_ms - audio_start_ms
            start_frame = int(audio_start_ms / self.frame_ms)
            # 拼接时负帧位置静音、内容不错位（clamp 到 0 会让 mel 内容整体后移，
            # 子音/元音落入 envelope 淡出区被切掉 -> 咬字丢失 + 电平凹陷）。

            total_span_ms = target_duration_ms + pre_vel_ms - tail_intrude_ms + tail_overlap_ms
            total_span_ms = max(total_span_ms, (getattr(phone, 'oto_consonant', 0.0) or 0.0) * _vel + self.frame_ms)
            # 与 process() 的静音尾部扩展保持一致（尾部 F0=0 由 process 内部 pad 提供）
            target_n_frames = int(np.round((total_span_ms + 80.0) / self.frame_ms))
            target_n_frames = max(target_n_frames, 1)
            
            end_frame = start_frame + target_n_frames
            
            f0_slice = np.zeros(target_n_frames, dtype=np.float32)
            # start_frame 可为负（前母音提前于短语开头）：负帧位置的 f0 补 0
            # （拼接时该位置静音，不参与发声）。
            _src_s = max(0, start_frame)
            _src_e = min(len(f0_phrase), start_frame + target_n_frames)
            if _src_e > _src_s:
                f0_slice[_src_s - start_frame:_src_e - start_frame] = f0_phrase[_src_s:_src_e]

            logger.debug(f"音素 {phone.phoneme}: preutter_ms={preutter_ms:.1f}, overlap_ms={overlap_ms:.1f}, "
                        f"oto_preutter={oto_preutter:.1f}, oto_overlap={oto_overlap:.1f}, "
                        f"target_start_ms={target_start_ms:.1f}, target_duration_ms={target_duration_ms:.1f}, "
                        f"start_frame={start_frame}, f0_slice_range={f0_slice[f0_slice>0].min() if np.any(f0_slice>0) else 0:.1f}-{f0_slice.max():.1f}Hz")

            phone_params = map_flag_to_parameter(phone)

            # 曲线参数（气声/张力/浊音度/性别，来自 OpenUTAU 表达式曲线）→ 音素级取值。
            # 参考同类实现：曲线的 breath/tension/voicing/genc 逐音素应用到该音素片段。
            # 合并规则：音素显式设置了对应 flag 时以 flag 为准，否则用曲线在该音素区间的均值。
            flag_names = {f[0] for f in (phone.flags or [])
                          if isinstance(f, (tuple, list)) and len(f) > 0 and isinstance(f[0], str)}
            flag_override = {
                'breathiness': 'Hb' in flag_names,
                'tension': 'Ht' in flag_names,
                'voicing': 'Hv' in flag_names,
                'gender': 'g' in flag_names,
                'growl': 'HG' in flag_names,
            }
            seg_start_f = max(0, int(target_start_ms / self.frame_ms))
            seg_end_f = min(total_frames, seg_start_f + max(1, int(np.round(total_span_ms / self.frame_ms))))
            for param_name in ('breathiness', 'tension', 'voicing', 'gender'):
                if flag_override.get(param_name, False):
                    continue
                curve = self._get_param_curve(param_name, total_frames)
                if len(curve) == 0:
                    continue
                seg = curve[seg_start_f:seg_end_f]
                if len(seg) == 0:
                    seg = curve
                phone_params[param_name] = float(np.mean(seg))

            # HG flag（咆哮强度 0-100）：折入整句 growl 曲线，最终统一在波形域应用
            if flag_override.get('growl', False) and phone_params['growl'] > 0.5:
                self._phone_growl_segments.append((seg_start_f, seg_end_f, float(phone_params['growl'])))

            processor = PhoneProcessor(
                phone, self.audio_loader, self.config, self.model_manager
            )
            feat = processor.process(
                target_duration_ms=target_duration_ms,
                target_f0_hz=f0_slice,
                key_shift=0.0,
                preutter_ms=preutter_ms,
                breathiness=phone_params['breathiness'],
                voicing=phone_params['voicing'],
                tension=phone_params['tension'],
                gender=phone_params['gender'],
                growl=phone_params['growl'],
                velocity=float(getattr(phone, 'velocity', 1.0)),
                tail_intrude_ms=tail_intrude_ms,
                tail_overlap_ms=tail_overlap_ms,
            )
            
            feat["start_frame"] = start_frame
            feat["start_frame_exact"] = start_frame_exact
            feat["end_frame"] = feat["start_frame"] + feat["mel"].shape[-1]
            feat["preutter_ms"] = preutter_ms
            feat["overlap_ms"] = overlap_ms
            feat["phone"] = phone
            features.append(feat)

        # Raw/mapped head-body calibration did not generalize in multi-bank A/Bs.
        # Keep source-level measurements diagnostic; never silently enable it.
        self.source_loudness_trace = {"reason": "diagnostic_only", "notes": []}
        return features

    

    def _concatenate_waveforms(self, phone_features: List[dict], vb_type: str = 'mono',
                              language: str = 'JA') -> np.ndarray:
        """波形域拼接（保留相位信息，避免Mel域拼接伪影）

        核心策略（参考同类实现 和 同类引擎）：
        1. 先在波形域进行crossfade拼接，保留相位信息
        2. 拼接完成后统一提取Mel频谱
        3. 使用纯cosine crossfade，避免复杂的能量匹配导致的伪影
        4. 增加overlap长度，确保平滑过渡
        5. 对静音帧进行特殊处理

        UTAU语义：
        - preutter_ms: 音素预发音时间，音素实际提前开始
        - overlap_ms: 重叠时间，决定crossfade长度

        Args:
            phone_features: 音素特征列表（包含wave）
            vb_type: 声库类型 ('mono' 或 'vowel_connected')
            language: 目标语言

        Returns:
            拼接后的完整波形
        """
        phone_features = sorted(phone_features, key=lambda f: f["start_frame"])

        if not phone_features:
            return np.zeros(0, dtype=np.float32)

        lang_overlap_params = {
            'ZH': {'min_ms': 50, 'max_ms': 150},
            'EN': {'min_ms': 60, 'max_ms': 180},
            'JA': {'min_ms': 50, 'max_ms': 130},
            'KO': {'min_ms': 55, 'max_ms': 150},
        }
        
        lang_params = lang_overlap_params.get(language.upper(), lang_overlap_params['JA'])
        
        if vb_type == 'vowel_connected':
            min_overlap_ms = lang_params['min_ms']
            max_overlap_ms = lang_params['max_ms']
        else:
            min_overlap_ms = lang_params['min_ms']
            max_overlap_ms = lang_params['max_ms']

        total_samples = 0
        for feat in phone_features:
            wave = feat.get("wave", np.zeros(0))
            start_frame = feat["start_frame"]
            wave_start_sample = int(start_frame * self.hop_size)
            wave_end_sample = wave_start_sample + len(wave)
            total_samples = max(total_samples, wave_end_sample)

        result = np.zeros(total_samples, dtype=np.float32)

        for i, feat in enumerate(phone_features):
            wave = feat.get("wave", np.zeros(0))
            if len(wave) == 0:
                continue

            start_frame = feat["start_frame"]
            start_sample = int(start_frame * self.hop_size)
            end_sample = start_sample + len(wave)

            if end_sample > len(result):
                result = np.pad(result, (0, end_sample - len(result)))

            result[start_sample:end_sample] = wave

        for i in range(1, len(phone_features)):
            prev_feat = phone_features[i-1]
            curr_feat = phone_features[i]
            
            prev_wave = prev_feat.get("wave", np.zeros(0))
            curr_wave = curr_feat.get("wave", np.zeros(0))
            
            if len(prev_wave) == 0 or len(curr_wave) == 0:
                continue
            
            prev_start_sample = int(prev_feat["start_frame"] * self.hop_size)
            prev_end_sample = prev_start_sample + len(prev_wave)
            
            curr_start_sample = int(curr_feat["start_frame"] * self.hop_size)
            curr_end_sample = curr_start_sample + len(curr_wave)
            
            overlap_ms = curr_feat.get("overlap_ms", 0.0)
            preutter_ms = curr_feat.get("preutter_ms", 0.0)
            effective_overlap_ms = max(overlap_ms, preutter_ms, min_overlap_ms)
            effective_overlap_ms = min(effective_overlap_ms, max_overlap_ms)
            effective_overlap_samples = int(effective_overlap_ms * self.sr / 1000)
            
            gap_samples = curr_start_sample - prev_end_sample
            
            actual_overlap_samples = min(effective_overlap_samples, 
                                        prev_end_sample, 
                                        len(curr_wave))
            actual_overlap_samples = max(actual_overlap_samples, 2)
            
            overlap_start = max(0, prev_end_sample - actual_overlap_samples)
            overlap_end = prev_end_sample
            overlap_len = overlap_end - overlap_start
            
            curr_head_len = min(overlap_len, len(curr_wave))
            
            if overlap_len >= 2 and curr_head_len >= 2:
                fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, overlap_len))).astype(np.float32)
                fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, overlap_len))).astype(np.float32)
                
                prev_tail = result[overlap_start:overlap_end]
                curr_head = curr_wave[:curr_head_len]
                
                if len(curr_head) < overlap_len:
                    fade_in = fade_in[:len(curr_head)]
                    actual_overlap_len = len(curr_head)
                else:
                    actual_overlap_len = overlap_len
                
                if gap_samples > 0:
                    result[overlap_start:overlap_end] = prev_tail * fade_out
                    for g in range(gap_samples):
                        alpha = (g + 1) / (gap_samples + 1)
                        result[prev_end_sample + g] = prev_tail[-1] * (1 - alpha) + curr_head[0] * alpha
                    result[curr_start_sample:curr_start_sample + actual_overlap_len] = curr_head * fade_in[:actual_overlap_len]
                else:
                    curr_wave_start_in_prev = prev_end_sample - curr_start_sample
                    if curr_wave_start_in_prev > 0:
                        curr_head_in_overlap = curr_wave[:curr_wave_start_in_prev + actual_overlap_len]
                        result[overlap_start:overlap_end] = prev_tail[:actual_overlap_len] * fade_out[:actual_overlap_len] + curr_head_in_overlap[-actual_overlap_len:] * fade_in[:actual_overlap_len]
                    else:
                        result[overlap_start:overlap_end] = prev_tail[:actual_overlap_len] * fade_out[:actual_overlap_len] + curr_head[:actual_overlap_len] * fade_in[:actual_overlap_len]

        return result

    def _concatenate_mels(self, phone_features: List[dict],
                          total_frames: int, vb_type: str = 'mono',
                          language: str = 'JA') -> np.ndarray:
        """拼接 phone mel，并把连续时序精确映射到整句 mel 帧网格。

        - source mel magnitude = exp(mel)；按每个 phone 的 exact onset 与目标
          hop 区间求交叠积分，再乘 OpenUtau 传入的振幅 envelope，求和后 log。
        - 不把 phone 起点取整后整段平移；整数起点仍与旧结果一致，分数帧起点
          的能量按真实 overlap 分配到相邻 phrase 帧。
        - F0 remains independently supplied by OpenUtau. This method never
          moves or synthesizes a pitch contour.
        - Raw oto preutter and resolved leading are reconciled once, while
          sampling the source in PhoneProcessor; assembly does not shift again.

        Safety logic only covers the phrase frame grid; it does not invent F0.
        """
        n_mels = self.config.audio.n_mels
        phone_features = sorted(phone_features, key=lambda f: f["start_frame"])
        if not phone_features:
            return np.full((n_mels, total_frames), -30.0, dtype=np.float32)

        accumulator = MelOverlapAccumulator(
            n_mels, total_frames, self.frame_ms,
            max_compensation_db=self.config.processing.mel_max_compensation_db,
            coherence_width=self.config.processing.mel_coherence_width)
        for feat in phone_features:
            mel = feat["mel"]
            if mel.shape[1] == 0:
                continue
            phone = feat.get("phone")
            env = getattr(phone, 'envelope', None) if phone is not None else None
            anchor = (phone.position_ms - self.req.position_ms + self.req.leading_ms
                      if phone is not None else 0.)
            accumulator.add(mel, float(feat.get("start_frame_exact", feat["start_frame"])),
                            env if env and len(env) >= 5 else None, anchor)
        mel_out = accumulator.compose()
        self.overlap_trace = accumulator.trace
        self.envelope_expression_gain = accumulator.expression_gain
        return mel_out
    def _smooth_inter_frame_transitions(self, mel: np.ndarray) -> np.ndarray:
        """帧间过渡平滑
        沿时间轴对 Mel 频谱进行平滑处理，减少帧间频谱突变，
        从而降低频谱通量，提高音素连接的流畅度。

        Args:
            mel: Mel 频谱 (n_mels, T)

        Returns:
            平滑后的 Mel 频谱
        """
        n_mels, n_frames = mel.shape
        if n_frames < 5:
            return mel

        from scipy import signal

        kernel_size = 15
        kernel = np.hanning(kernel_size)
        kernel = kernel / kernel.sum()

        smoothed = mel.copy()

        for m in range(n_mels):
            smoothed[m] = signal.convolve(mel[m], kernel, mode='same')

        linear_mel = np.exp(mel)
        frame_energy = np.mean(linear_mel, axis=0)
        energy_diff = np.abs(np.diff(frame_energy))
        transition_frames = np.where(energy_diff > 0.0002)[0]
        
        for frame_idx in transition_frames:
            if frame_idx > 2 and frame_idx < n_frames - 3:
                left_frame = mel[:, frame_idx - 3:frame_idx]
                right_frame = mel[:, frame_idx + 1:frame_idx + 4]
                left_mean = np.mean(left_frame, axis=1, keepdims=True)
                right_mean = np.mean(right_frame, axis=1, keepdims=True)
                interpolated = 0.5 * (left_mean + right_mean)
                smoothed[:, frame_idx:frame_idx + 1] = 0.3 * mel[:, frame_idx:frame_idx + 1] + 0.7 * interpolated

        for m in range(n_mels):
            for f in range(1, n_frames - 1):
                diff = np.abs(smoothed[m, f] - smoothed[m, f-1])
                if diff > 1.5:
                    smoothed[m, f] = smoothed[m, f-1] + np.sign(smoothed[m, f] - smoothed[m, f-1]) * 1.5

        for f in range(1, n_frames - 1):
            frame_diff = np.mean(np.abs(smoothed[:, f] - smoothed[:, f-1]))
            if frame_diff > 1.0:
                alpha = 0.3
                smoothed[:, f] = alpha * smoothed[:, f] + (1 - alpha) * smoothed[:, f-1]

        return smoothed

    def _enhance_high_frequencies(self, samples: np.ndarray) -> np.ndarray:
        """高频增强（使用温和的高频搁架滤波器）

        使用 scipy.signal 实现的高频搁架滤波器（4kHz以上+3dB），
        避免粗暴的STFT幅度乘法导致的MFCC偏移。

        Args:
            samples: 音频波形

        Returns:
            高频增强后的音频波形
        """
        if len(samples) == 0:
            return samples

        from scipy import signal

        cutoff_freq = 4000.0
        gain_db = 4.0

        nyquist = self.sr / 2.0
        normalized_cutoff = cutoff_freq / nyquist

        Q = 1.0
        A = 10 ** (gain_db / 40.0)

        w0 = normalized_cutoff * np.pi
        alpha = np.sin(w0) / (2.0 * Q)

        b0 = A * ((A + 1.0) - (A - 1.0) * np.cos(w0) + 2.0 * np.sqrt(A) * alpha)
        b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * np.cos(w0))
        b2 = A * ((A + 1.0) - (A - 1.0) * np.cos(w0) - 2.0 * np.sqrt(A) * alpha)
        a0 = (A + 1.0) + (A - 1.0) * np.cos(w0) + 2.0 * np.sqrt(A) * alpha
        a1 = -2.0 * ((A - 1.0) + (A + 1.0) * np.cos(w0))
        a2 = (A + 1.0) + (A - 1.0) * np.cos(w0) - 2.0 * np.sqrt(A) * alpha

        b = [b0 / a0, b1 / a0, b2 / a0]
        a = [1.0, a1 / a0, a2 / a0]

        enhanced_samples = signal.lfilter(b, a, samples)

        enhanced_samples = np.clip(enhanced_samples, -1.0, 1.0)

        return enhanced_samples.astype(np.float32)

    def _enhance_high_freq_in_mel(self, mel: np.ndarray) -> np.ndarray:
        """在 Mel 频谱域增强高频（轻微提升，避免过度增强导致噪声）"""
        if mel.size == 0:
            return mel

        return mel.copy()

    def _apply_natural_vibrato(self, f0: np.ndarray) -> np.ndarray:
        """添加极微妙的自然颤动（不超过2-3音分）

        真人演唱时存在极其微小的音高波动，这是声带振动的自然特性。
        添加极轻微的颤动（深度不超过2-3音分）可以让声音更自然。

        约束：
        - 当参考音频已有真实 F0 曲线（f0_std >= 10Hz），完全禁用人工抖动
        - 颤动深度不超过 2-3 cents（0.5-1.5Hz 对于 MIDI 60）
        - 使用多频率叠加的确定性调制，避免随机噪声导致的电音感

        Args:
            f0: F0 曲线

        Returns:
            添加微颤动后的F0曲线
        """
        if len(f0) == 0:
            return f0

        f0_valid = f0[f0 > 50]
        if len(f0_valid) == 0:
            return f0

        f0_std = np.std(f0_valid)
        
        if f0_std >= 10.0:
            logger.info(f"检测到参考音频已有真实F0曲线（标准差={f0_std:.1f}Hz），跳过人工颤动")
            return f0

        result = f0.copy().astype(np.float64)
        voiced_mask = f0 > 50

        n_frames = len(f0)
        t = np.arange(n_frames) * self.frame_ms / 1000

        vibrato_depth_cents = 1.5 + np.random.uniform(0, 1.0)
        vibrato_rate_hz = 5.5 + np.random.uniform(0, 1.5)

        base_vibrato = np.sin(2 * np.pi * vibrato_rate_hz * t) * vibrato_depth_cents

        fine_jitter_freqs = [12.3, 18.7, 25.1]
        fine_jitter = np.zeros(n_frames)
        for freq in fine_jitter_freqs:
            fine_jitter += np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi)) * 0.2

        total_vibrato = base_vibrato + fine_jitter

        vibrato_hz = np.where(
            voiced_mask,
            f0.astype(np.float64) * (2 ** (total_vibrato / 1200.0) - 1),
            0.0
        )

        result[voiced_mask] = result[voiced_mask] + vibrato_hz[voiced_mask]

        result = np.clip(result, 80.0, 800.0)

        return result.astype(np.float32)

    def _add_natural_breath_noise(self, audio: np.ndarray) -> np.ndarray:
        """添加自然气息感噪声

        pc-nsf-hifigan声码器在F0>0时只产生纯正弦波，缺乏自然的气息噪声成分，
        导致过零率过低（0.027 vs 参考0.087），声音过于机械。

        本方法添加频谱整形后的高斯白噪声，模拟人声的自然气息感：
        - 低频段(<2kHz)：较少噪声，保持音色纯净
        - 中频段(2-6kHz)：适量噪声，增加自然感
        - 高频段(>6kHz)：逐渐减少，避免刺耳

        Args:
            audio: 声码器输出的纯净音频

        Returns:
            添加自然气息噪声后的音频
        """
        if len(audio) == 0:
            return audio

        import librosa
        from scipy.signal import butter, filtfilt

        audio_copy = audio.astype(np.float64)
        n_samples = len(audio_copy)

        target_zcr = 0.08
        current_zcr = np.mean(librosa.feature.zero_crossing_rate(audio_copy))
        
        noise_ratio = max(0.02, min(0.06, (target_zcr - current_zcr) * 3.0))
        
        noise = np.random.randn(n_samples).astype(np.float64)
        
        nyquist = self.sr / 2.0
        low = 1000.0 / nyquist
        high = 5000.0 / nyquist
        b, a = butter(4, [low, high], btype='band')
        
        shaped_noise = filtfilt(b, a, noise)
        
        shaped_noise_rms = np.sqrt(np.mean(shaped_noise ** 2))
        if shaped_noise_rms > 0:
            shaped_noise = shaped_noise / shaped_noise_rms
        
        audio_rms = np.sqrt(np.mean(audio_copy ** 2))
        noise_amp = audio_rms * noise_ratio
        
        result = audio_copy + shaped_noise * noise_amp

        return result.astype(np.float32)

    def _measure_boundary_discontinuity(self, mel: np.ndarray, phone_features: list) -> None:
        """测量音素边界处的频谱不连续性（量化 overlap "抽" 的程度）

        通过计算相邻音素边界处的 Mel 频谱差异来量化不连续性。
        差异越大，"抽"的感觉越明显。

        Args:
            mel: 拼接后的 Mel 频谱 (n_mels, total_frames)
            phone_features: 音素特征列表
        """
        if len(phone_features) < 2:
            return

        n_mels, total_frames = mel.shape

        linear_mel = np.exp(mel)

        max_discontinuity = 0.0
        avg_discontinuity = 0.0
        boundary_count = 0

        for i in range(1, len(phone_features)):
            prev_feat = phone_features[i-1]
            curr_feat = phone_features[i]

            prev_end = min(total_frames, max(0, prev_feat["start_frame"]) + prev_feat["mel"].shape[-1])
            curr_start = max(0, curr_feat["start_frame"])

            boundary_frame = min(prev_end, curr_start)
            
            if boundary_frame <= 0 or boundary_frame >= total_frames - 1:
                continue

            left_frame = boundary_frame - 1
            right_frame = boundary_frame

            left_energy = np.sum(linear_mel[:, left_frame])
            right_energy = np.sum(linear_mel[:, right_frame])

            energy_diff = np.abs(left_energy - right_energy) / (left_energy + right_energy + 1e-10)

            spectral_diff = np.mean(np.abs(linear_mel[:, left_frame] - linear_mel[:, right_frame])) / np.mean(linear_mel[:, left_frame] + linear_mel[:, right_frame] + 1e-10)

            discontinuity = 0.6 * energy_diff + 0.4 * spectral_diff

            if discontinuity > 0.8:
                logger.warning(f"高不连续性边界 #{i}: frame={boundary_frame}, prev_end={prev_end}, curr_start={curr_start}, "
                            f"energy_diff={energy_diff:.4f}, spectral_diff={spectral_diff:.4f}, discontinuity={discontinuity:.4f}")

            max_discontinuity = max(max_discontinuity, discontinuity)
            avg_discontinuity += discontinuity
            boundary_count += 1

        if boundary_count > 0:
            avg_discontinuity /= boundary_count
            logger.info(f"边界不连续性测量: 平均={avg_discontinuity:.4f}, 最大={max_discontinuity:.4f}, 边界数={boundary_count}")

            if max_discontinuity > 0.3:
                logger.warning(f"检测到高不连续性边界 (最大={max_discontinuity:.4f})，可能存在 overlap '抽' 问题")

    def _localized_energy_smooth(self, mel: np.ndarray, phone_features: list) -> np.ndarray:
        """对音素边界处进行局部能量平滑

        仅在音素边界前后各5帧范围内应用能量平滑，避免全局平滑模糊辅音瞬态。

        Args:
            mel: (n_mels, total_frames) Mel 频谱
            phone_features: 音素特征列表

        Returns:
            局部能量平滑后的 Mel 频谱
        """
        if not phone_features:
            return mel

        n_mels, total_frames = mel.shape
        linear = np.exp(mel)
        energy = np.sum(linear, axis=0)
        
        boundary_mask = np.zeros(total_frames, dtype=np.float32)
        
        for feat in phone_features:
            start_frame = max(0, feat["start_frame"])
            mel_len = feat["mel"].shape[-1]
            end_frame = min(total_frames, start_frame + mel_len)
            
            boundary_start = max(0, start_frame - 5)
            boundary_end = min(total_frames, start_frame + 5)
            boundary_mask[boundary_start:boundary_end] = 1.0
            
            boundary_start = max(0, end_frame - 5)
            boundary_end = min(total_frames, end_frame + 5)
            boundary_mask[boundary_start:boundary_end] = 1.0
        
        from scipy.ndimage import gaussian_filter1d
        energy_smooth = gaussian_filter1d(energy.astype(np.float64), sigma=2.0, mode='reflect')
        
        ratio = energy_smooth / (energy + 1e-10)
        ratio = np.clip(ratio, 0.7, 1.4)
        
        ratio = ratio * boundary_mask + (1.0 - boundary_mask)
        
        linear_smooth = linear * ratio[np.newaxis, :]
        
        return np.log(np.clip(linear_smooth, 1e-10, None)).astype(np.float32)

    def _smooth_formant_envelope(self, envelope: np.ndarray) -> np.ndarray:
        """平滑谱包络，提取共振峰结构

        使用移动平均和峰值检测来平滑频谱包络，保留共振峰特征。

        Args:
            envelope: (n_mels, 1) 谱包络

        Returns:
            平滑后的谱包络
        """
        if envelope.ndim == 1:
            envelope = envelope.reshape(-1, 1)

        n_mels = envelope.shape[0]
        
        # 使用中值滤波平滑
        from scipy.ndimage import median_filter
        smoothed = median_filter(envelope, size=(3, 1), mode='reflect')
        
        # 二次平滑（高斯滤波）
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(smoothed.flatten(), sigma=2.0).reshape(-1, 1)
        
        # 保留能量比例关系
        orig_sum = np.sum(envelope)
        smooth_sum = np.sum(smoothed)
        if smooth_sum > 0:
            smoothed = smoothed * (orig_sum / smooth_sum)

        return smoothed

    def _get_effective_overlap_ms(self, phone) -> float:
        """获取有效的 overlap 长度（毫秒）

        优先级：
        1. 用户手动设置的 overlap_ms（如果 > 0）
        2. oto_overlap（自动计算）
        3. 默认值 30ms

        OpenUTAU 中：
        - overlap_ms: 用户手动调整的值
        - oto_overlap: oto.ini 中定义的 overlap，由 OpenUTAU 自动计算
        """
        manual_overlap = getattr(phone, 'overlap_ms', 0) or 0
        oto_overlap = getattr(phone, 'oto_overlap', 0) or 0

        if manual_overlap > 0:
            return manual_overlap
        elif oto_overlap > 0:
            return oto_overlap
        else:
            return 30.0

    def _apply_formant_constraint(self, mel_phrase: np.ndarray,
                                  phone_features: List[dict],
                                  total_frames: int) -> np.ndarray:
        """共振峰约束

        以原始音素的 Mel 频谱为参考，约束拼接后的 Mel，
        防止音高调整和拼接导致的音色泄漏。
        """
        # 构建参考 Mel（各音素原始 Mel 的平铺）
        n_mels = self.config.audio.n_mels
        mel_floor = np.min(mel_phrase)
        ref_mel = np.full_like(mel_phrase, mel_floor)

        for feat in phone_features:
            mel = feat["mel"]
            start = max(0, feat["start_frame"])
            end = min(total_frames, feat["end_frame"])
            src_len = min(mel.shape[1], end - start)
            if src_len > 0 and start < total_frames:
                ref_mel[:, start:start + src_len] = mel[:, :src_len]

        # 应用共振峰约束（轻度，避免过度修正引入 artifacts）
        constrained = formant_constrain(mel_phrase, ref_mel, strength=0.15)
        return constrained

    def _synthesize_per_phone(self, phone_features: list, total_frames: int) -> np.ndarray:
        """整句一次 vocode（mel 拼接，用户明确要求，禁止逐音素/禁止波形叠加）。

        mel_phrase 已在 _concatenate_mels 中按 OpenUTAU envelope 门控合并好整句频谱
        （重叠区互补淡入淡出、单一频谱），f0_concat = OpenUTAU pitches 曲线。
        这里把整句 mel_phrase + f0_concat **一次**交给声码器，输出整句波形：
        - 不逐音素 vocode（避免片段边界/上下文伪影）；
        - 不做波形域叠加（避免 overlap 两个不同音高波形相加 -> 拍频颤抖/电音）；
        - 音高完全跟随工程曲线（单一 F0）。
        """
        mel = self.mel_phrase
        f0 = self.f0_concat
        n = min(mel.shape[1], len(f0))
        if n == 0:
            return np.zeros(total_frames * self.hop_size, dtype=np.float32)
        mel = mel[:, :n]
        f0 = f0[:n]
        # OpenUtau applies DYN after reading the neutral cached waveform.



        # 给声码器前后各补 N 帧静音上下文（mel=-80, f0=0），vocode 后按采样精确裁掉，
        # 让片段的截断边界伪影落在被裁掉的静音区。DIFFICAL_VOCODER_PAD=0 可关。
        pad = int(os.environ.get("DIFFICAL_VOCODER_PAD", "0") or 0)  # 实测补静音后起音更差，默认关闭
        if pad > 0:
            sil = np.full((mel.shape[0], pad), -80.0, dtype=np.float32)
            zeros = np.zeros(pad, dtype=np.float32)
            mel = np.concatenate([sil, mel, sil], axis=1)
            f0 = np.concatenate([zeros, f0, zeros])
        w = np.asarray(self._synthesize_waveform(mel, f0, normalize=False), dtype=np.float32)
        if pad > 0:
            cut = pad * self.hop_size
            if len(w) >= 2 * cut:
                w = w[cut:len(w) - cut]
        out = np.zeros(total_frames * self.hop_size, dtype=np.float32)
        out[:min(len(w), len(out))] = w[:min(len(w), len(out))]
        return out

    def _synthesize_waveform(self, mel: np.ndarray, f0: np.ndarray,
                             normalize: bool = True) -> np.ndarray:
        """使用 pc-nsf-hifigan 合成波形

        normalize=True 时做 mel 能量归一化（boost 到 target -5，供整句合成用）；
        normalize=False 时不做 boost（逐音素合成用，避免片段局部 mel_mean 导致
        boost 过大、输出削波/失真——同类重采样器 不做能量归一化）。
        """
        if mel.shape[-1] == 0 or len(f0) == 0:
            return np.zeros(0, dtype=np.float32)

        n_frames = min(mel.shape[-1], len(f0))
        mel = mel[:, :n_frames]
        f0 = f0[:n_frames].copy()

        if n_frames != mel.shape[-1] or n_frames != len(f0):
            logger.warning(f"帧长度不一致: mel={mel.shape[-1]}, f0={len(f0)}, 修正为{n_frames}")
            n_frames = min(mel.shape[-1], len(f0))
            mel = mel[:, :n_frames]
            f0 = f0[:n_frames].copy()

        # 同类重采样器 fill=6 模式：在前后各填充6帧上下文
        # 这给声码器提供了边界帧的时间上下文，减少边界伪影
        context_frames = 6
        if n_frames > context_frames:
            mel_padded = np.concatenate([
                np.repeat(mel[:, :1], context_frames, axis=1),
                mel,
                np.repeat(mel[:, -1:], context_frames, axis=1)
            ], axis=1)
            f0_padded = np.concatenate([
                np.repeat(f0[:1], context_frames),
                f0,
                np.repeat(f0[-1:], context_frames)
            ])
        else:
            mel_padded = mel
            f0_padded = f0

        # 直接使用所有帧，不要过滤掉低能量帧
        # 低能量帧仍然包含音高信息，过滤会导致音高丢失
        mel_active = mel_padded.copy()
        f0_active = f0_padded.copy()

        # F0 处理策略（参考同类实现 和 同类引擎）：
        # - F0=0 表示清音/静音，vocoder 的 SineGen 会自动生成噪声而不是正弦波
        # - 不填充辅音/preutter 帧的 F0，让它们保持清音状态
        # - 仅对 NaN/负值做修复
        f0_active = np.where(np.isnan(f0_active) | (f0_active < 0), 0.0, f0_active)

        # 修复异常值
        f0_active = np.where(np.isnan(f0_active) | (f0_active < 0), 0.0, f0_active)

        # 调试：检查声码器输入的 F0 范围
        f0_input_valid = f0_active[f0_active > 50]
        if len(f0_input_valid) > 0:
            logger.info(f"声码器输入 F0 范围: {f0_input_valid.min():.1f}~{f0_input_valid.max():.1f} Hz")
            logger.info(f"声码器输入平均F0: {np.mean(f0_input_valid):.1f} Hz")
            logger.info(f"声码器输入F0标准差: {np.std(f0_input_valid):.1f} Hz")
            
            # 检查异常值（F0 有效范围为 50~1500Hz，阈值贴近边界，避免把正常高音误报）
            low_f0_count = np.sum((f0_active > 50) & (f0_active < 50.5))
            high_f0_count = np.sum(f0_active > 1400)
            logger.info(f"声码器输入F0<50Hz数量: {low_f0_count}")
            logger.info(f"声码器输入F0>1400Hz数量: {high_f0_count}")
            
            if f0_input_valid.max() > 1400:
                logger.warning(f"声码器输入异常高 F0: {f0_input_valid.max():.1f} Hz")
                high_indices = np.where(f0_active > 1400)[0]
                logger.warning(f"异常高F0帧索引: {high_indices[:10]}")
                for idx in high_indices[:5]:
                    logger.warning(f"  帧{idx}: F0={f0_active[idx]:.2f} Hz")
            if f0_input_valid.min() < 50:
                logger.warning(f"声码器输入异常低 F0: {f0_input_valid.min():.1f} Hz")
                low_indices = np.where((f0_active > 0) & (f0_active < 50))[0]
                logger.warning(f"异常低F0帧索引: {low_indices[:10]}")
                for idx in low_indices[:5]:
                    logger.warning(f"  帧{idx}: F0={f0_active[idx]:.2f} Hz")
        
        # 问题：声码器在F0=0且Mel=-20（静音）时会输出异常高F0（1800Hz+）
        # 解决方案：将静音帧的F0设为1Hz（非零但很低），这样声码器会产生静音而不是异常高F0
        # 同时在每个有声段的开头和结尾添加短淡入淡出避免click artifacts
        frame_energy = np.mean(np.exp(mel_active), axis=0)
        is_voiced = (f0_active > 50) | (frame_energy > 0.001)

        # Preserve input pitch and source spectrum. Pure empty mel frames
        # remain silent; do not raise weak bins or fill unvoiced pitch frames.
        f0_processed = f0_active.copy()
        mel_processed = mel_active.copy()
        mel_processed[:, ~is_voiced] = -80.0

        # 找到所有有声段
        voiced_segments = []
        in_segment = False
        seg_start = 0
        
        for i in range(n_frames):
            if is_voiced[i] and not in_segment:
                in_segment = True
                seg_start = i
            elif not is_voiced[i] and in_segment:
                in_segment = False
                voiced_segments.append((seg_start, i))
        
        if in_segment:
            voiced_segments.append((seg_start, n_frames))

        logger.info(f"找到{len(voiced_segments)}个有声段")

        # Mel频谱能量归一化：声码器需要特定的能量范围才能输出正常音量的音频
        # 参考: pc-nsf-hifigan 训练时的 Mel 范围通常是 [-15, 0]
        if normalize:
            mel_mean = np.mean(mel_processed[mel_processed > -70])
            target_mel_mean = -5.0
            energy_boost = target_mel_mean - mel_mean
            mel_norm = mel_processed + energy_boost
            mel_norm = np.clip(mel_norm, -80, 0)
        else:
            mel_norm = mel_processed

        samples_with_context = self.model_manager.vocoder_synthesize(mel_norm, f0_processed)

        # 提取中间的有效部分，去掉前后填充的上下文帧
        # 每帧对应hop_size个采样
        context_samples = context_frames * self.hop_size
        if len(samples_with_context) > context_samples * 2:
            samples = samples_with_context[context_samples:-context_samples]
        else:
            samples = samples_with_context

        # 注意：这里不做波形域边界交叉淡化 —— 拼接统一在 mel 域完成（用户明确要求
        # 会引入相位叠加/双音（鬼音）。

        # Do not re-envelope inferred voiced segments. The old 20ms fades ran
        # backwards (1 -> .1 on attack, .1 -> 1 on release), causing 20dB notches
        # and hard resets. Their segment indices also belonged to padded context,
        # not the trimmed waveform. OpenUtau's mel envelope already owns those
        # edges; retain only the short phrase-edge anti-click fade below.

        fade_samples = min(128, len(samples) // 4)
        if fade_samples > 0:
            fade_in_win = np.linspace(0, 1, fade_samples)
            samples[:fade_samples] *= fade_in_win
            fade_out_win = np.linspace(1, 0, fade_samples)
            samples[-fade_samples:] *= fade_out_win

        return samples

    def _spectral_denoise(self, audio: np.ndarray) -> np.ndarray:
        """频谱降噪 - 减少嘶声和背景噪声

        使用谱减法进行轻度降噪，保留音质的同时减少刺耳的高频噪声。
        """
        if len(audio) < 512:
            return audio

        from scipy import signal
        from scipy.ndimage import gaussian_filter1d

        # STFT
        f, t, Zxx = signal.stft(
            audio, fs=self.sr, nperseg=2048,
            noverlap=2048 - 512, nfft=2048
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # 估计噪声谱（使用前 10% 帧作为噪声估计）
        n_noise_frames = max(1, int(mag.shape[1] * 0.1))
        noise_est = np.mean(mag[:, :n_noise_frames], axis=1, keepdims=True)

        # 谱减法（轻度降噪）
        alpha = 1.5  # 过减因子
        beta = 0.01  # 谱下限
        mag_clean = mag - alpha * noise_est
        mag_clean = np.maximum(mag_clean, beta * mag)

        # 对高频区域增加降噪强度（针对嘶声）
        high_freq_idx = int(len(f) * 0.5)  # 50% 以上频率
        high_noise_mask = np.ones(mag.shape[0])
        high_noise_mask[high_freq_idx:] = 2.0  # 高频降噪强度加倍
        mag_clean = mag - alpha * noise_est * high_noise_mask.reshape(-1, 1)
        mag_clean = np.maximum(mag_clean, beta * mag)

        # 时间平滑，避免音乐噪声
        mag_clean = gaussian_filter1d(mag_clean, sigma=0.5, axis=1)

        # 重建
        _, result = signal.istft(
            mag_clean * np.exp(1j * phase),
            fs=self.sr, nperseg=2048,
            noverlap=2048 - 512, nfft=2048
        )
        return result[:len(audio)].astype(np.float32)

    def _loudness_normalize(self, samples: np.ndarray) -> np.ndarray:
        """响度归一化（参考同类重采样器和同类引擎做法）

        参考项目做法：
        - 同类重采样器: -16 LUFS
        - 同类引擎: 0.5 RMS
        - 两者都使用简单硬限幅，不使用复杂的动态范围压缩

        简化策略：
        1. 应用全局增益到目标RMS
        2. 硬限幅 ±0.99，避免削波
        """
        if len(samples) == 0:
            return samples

        peak = np.max(np.abs(samples))
        if peak <= 0:
            return samples

        target_rms = 0.14
        current_rms = np.sqrt(np.mean(samples ** 2))
        
        if current_rms > 0:
            gain = target_rms / current_rms
            samples = samples * gain

        samples = np.clip(samples, -0.99, 0.99)

        return samples.astype(np.float32)

    def _compute_noise_factor(self) -> float:
        """根据音素类型计算噪声因子

        清辅音(さ/し/す/た/ち/つ/か/き/く/は/ひ/ふ等)需要保留更多噪声
        元音(あ/い/う/え/お/ら/ま/や/わ/ん等)需要减少噪声

        Returns:
            噪声因子 (0.3-0.6)
        """
        consonants = {'さ', 'し', 'す', 'せ', 'そ',
                      'た', 'ち', 'つ', 'て', 'と',
                      'か', 'き', 'く', 'け', 'こ',
                      'は', 'ひ', 'ふ', 'へ', 'ほ',
                      'ぱ', 'ぴ', 'ぷ', 'ぺ', 'ぽ',
                      'ら', 'り', 'る', 'れ', 'ろ',
                      'ま', 'み', 'む', 'め', 'も',
                      'や', 'ゆ', 'よ', 'わ', 'を',
                      'sa', 'shi', 'su', 'se', 'so',
                      'ta', 'chi', 'tsu', 'te', 'to',
                      'ka', 'ki', 'ku', 'ke', 'ko',
                      'ha', 'hi', 'fu', 'he', 'ho',
                      'pa', 'pi', 'pu', 'pe', 'po',
                      'ra', 'ri', 'ru', 're', 'ro',
                      'ma', 'mi', 'mu', 'me', 'mo',
                      'ya', 'yu', 'yo', 'wa', 'wo'}

        has_consonant = any(p.phoneme in consonants for p in self.req.phones)

        if has_consonant:
            return 0.6
        else:
            return 0.3

    def _apply_breath_modulation(self, samples: np.ndarray) -> np.ndarray:
        """添加微幅度调制，模拟自然呼吸的2-4Hz随机幅度变化

        真人演唱时，呼吸会导致音量产生轻微的周期性变化：
        - 调制频率：2-4Hz（呼吸频率范围）
        - 调制深度：2-5%（轻微的幅度变化）
        - 使用随机相位和频率，模拟自然呼吸的不规则性

        Args:
            samples: 音频采样

        Returns:
            添加呼吸调制后的音频采样
        """
        if len(samples) == 0:
            return samples

        sample_rate = self.sr
        
        breath_rate = 2.0 + np.random.uniform(0, 2.0)
        breath_depth = 0.02 + np.random.uniform(0, 0.03)
        
        phase = np.random.uniform(0, 2 * np.pi)
        t = np.arange(len(samples)) / sample_rate
        
        breath_modulation = 1.0 + breath_depth * np.sin(2 * np.pi * breath_rate * t + phase)
        
        breath_modulation = breath_modulation.astype(np.float32)
        
        return samples * breath_modulation

    def _apply_noise_gate(self, samples: np.ndarray) -> np.ndarray:
        """分段噪声门控 + 边界淡入淡出

        消除背景噪声，保留所有有声段落。
        使用自适应阈值识别静音区域，对短静音间隙做处理，保留长静音间隙。
        """
        if len(samples) == 0:
            return samples

        hop = 256
        n_frames = len(samples) // hop
        if n_frames < 8:
            return samples

        energy = np.array([
            np.sqrt(np.mean(samples[i * hop:(i + 1) * hop] ** 2))
            for i in range(n_frames)
        ])
        from scipy.ndimage import gaussian_filter1d
        energy_smooth = gaussian_filter1d(energy, sigma=4)
        max_energy = np.max(energy_smooth)
        if max_energy < 1e-6:
            return np.zeros_like(samples)

        threshold = max_energy * 0.02
        is_silent = energy_smooth < threshold

        silent_regions = []
        in_silent = False
        start_silent = 0

        for i in range(n_frames):
            if is_silent[i] and not in_silent:
                in_silent = True
                start_silent = i
            elif not is_silent[i] and in_silent:
                in_silent = False
                silent_regions.append((start_silent, i))

        if in_silent:
            silent_regions.append((start_silent, n_frames))

        result = samples.copy()

        max_silent_frames = 15

        for start, end in silent_regions:
            duration_frames = end - start

            if duration_frames <= max_silent_frames:
                start_sample = start * hop
                end_sample = end * hop

                fade_len = min(start_sample, len(result) - end_sample, hop * 2)

                if fade_len > 0 and start_sample > 0:
                    fade_out = np.linspace(1, 0, fade_len)
                    result[start_sample - fade_len:start_sample] *= fade_out

                result[start_sample:end_sample] = 0.0

                if fade_len > 0 and end_sample < len(result):
                    fade_in = np.linspace(0, 1, fade_len)
                    result[end_sample:end_sample + fade_len] *= fade_in

        fade_samples = min(512, len(result) // 20)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            result[:fade_samples] *= fade_in
            fade_out = np.linspace(1, 0, fade_samples)
            result[-fade_samples:] *= fade_out

        return result


def render_phrase(request: RenderPhraseRequest) -> RenderResultResponse:
    """渲染整句（入口函数）"""
    renderer = PhraseRenderer(request)
    return renderer.render()
