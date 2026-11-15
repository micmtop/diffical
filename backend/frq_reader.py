"""FRQ 文件读取器

UTAU .frq 文件格式解析：
- 头部："FREQ0003" (8字节 ASCII)
- hopSize：int32
- averageF0：double
- 空白：16字节
- length：int32
- f0数组：length个double
- amp数组：length个double

参考：OpenUtau/OpenUtau.Core/Classic/Frq.cs
"""

import struct
import os
import math
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def freq_to_midi(freq_hz: float) -> float:
    """频率 (Hz) → MIDI 音高"""
    if freq_hz <= 0:
        return 0.0
    return 69 + 12 * math.log2(freq_hz / 440.0)


def midi_to_hz(midi: float) -> float:
    """MIDI 音高 → 频率 (Hz)"""
    return 440.0 * (2 ** ((midi - 69) / 12))


def read_frq_file(wav_path: str, return_arrays: bool = False) -> Optional[Tuple[float, int]]:
    """读取 .frq 文件，获取平均录制音高

    Args:
        wav_path: WAV 文件路径（.frq 文件应与 WAV 文件同名）
        return_arrays: 是否返回完整的 F0 和振幅数组

    Returns:
        (average_f0_hz, hop_size) 或 None
        如果 return_arrays=True，返回 (average_f0_hz, hop_size, f0_array, amp_array)
    """
    frq_path = wav_path.replace('.wav', '_wav.frq').replace('.WAV', '_WAV.frq')
    if not os.path.exists(frq_path):
        base_name = os.path.splitext(wav_path)[0]
        frq_path = base_name + '.frq'
        if not os.path.exists(frq_path):
            return None

    try:
        with open(frq_path, 'rb') as f:
            header = f.read(8)
            if header != b'FREQ0003':
                logger.warning(f"Invalid FRQ header: {frq_path}")
                return None

            hop_size = struct.unpack('<i', f.read(4))[0]
            average_f0 = struct.unpack('<d', f.read(8))[0]

            _ = f.read(16)

            length = struct.unpack('<i', f.read(4))[0]

            if return_arrays:
                f0_array = np.zeros(length, dtype=np.float64)
                amp_array = np.zeros(length, dtype=np.float64)
                for i in range(length):
                    f0_array[i] = struct.unpack('<d', f.read(8))[0]
                for i in range(length):
                    amp_array[i] = struct.unpack('<d', f.read(8))[0]
                return average_f0, hop_size, f0_array, amp_array

            return average_f0, hop_size

    except Exception as e:
        logger.warning(f"读取 FRQ 文件失败: {frq_path}, {e}")
        return None


def get_sample_vowel_f0_hz(wav_path: str, singer_path: str) -> float:
    """获取样本的真实元音 F0（Hz）

    从 .frq 文件中提取高振幅语音帧的中值 F0，排除辅音和静音部分。
    这比平均 F0 更准确，因为平均 F0 包含了辅音帧（F0 较低）。

    Args:
        wav_path: WAV 文件相对路径
        singer_path: 声库根目录

    Returns:
        元音 F0（Hz），失败时返回 261.6 (C4)
    """
    full_wav_path = os.path.join(singer_path, wav_path)
    frq_data = read_frq_file(full_wav_path, return_arrays=True)

    if frq_data is not None:
        avg_f0, hop_size, f0_array, amp_array = frq_data
        
        if len(f0_array) > 0:
            # 过滤：F0 在合理范围内（人声范围）
            voiced_mask = (f0_array > 150) & (f0_array < 600)
            
            if np.any(voiced_mask):
                voiced_f0 = f0_array[voiced_mask]
                
                if len(voiced_f0) > 3:
                    # 计算中值作为元音 F0
                    vowel_f0 = float(np.median(voiced_f0))
                    
                    # 验证：如果与平均 F0 差异太大，使用平均 F0（防止错误的低 F0）
                    if avg_f0 > 0 and vowel_f0 > 0:
                        if vowel_f0 < avg_f0 * 0.7 or vowel_f0 > avg_f0 * 1.3:
                            logger.debug(f"元音 F0 ({vowel_f0:.1f}Hz) 与平均 F0 ({avg_f0:.1f}Hz) 差异过大，使用平均 F0")
                            vowel_f0 = avg_f0
                    
                    logger.debug(f"从 FRQ 获取元音 F0: {wav_path} -> {vowel_f0:.1f}Hz (avg={avg_f0:.1f}Hz)")
                    return vowel_f0

    return 261.6


def get_sample_tone(wav_path: str, singer_path: str) -> float:
    """获取样本录制的基准音高（MIDI）

    优先从 .frq 文件读取，回退到默认值 60 (C4)。

    Args:
        wav_path: WAV 文件相对路径
        singer_path: 声库根目录

    Returns:
        MIDI 音高（如 60 = C4）
    """
    full_wav_path = os.path.join(singer_path, wav_path)
    frq_data = read_frq_file(full_wav_path)

    if frq_data is not None:
        avg_f0, _ = frq_data
        if avg_f0 > 0:
            midi = freq_to_midi(avg_f0)
            logger.debug(f"从 FRQ 获取基准音高: {wav_path} -> {midi:.1f} (F0={avg_f0:.1f}Hz)")
            return midi

    return 60.0