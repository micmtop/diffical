"""测试/诊断脚本共用：构造 OpenUTAU 语义的绝对音高数组（cent）。

OpenUTAU 的 RenderPhrase.pitches 是绝对音高（AdjustedTone*100 + 颤音/曲线偏差），
Diffical 后端按绝对 cent 解释（midi = cent/100）。旧的测试脚本传 `[0]*frames`
（相对偏差约定）在修复绝对语义后会产生 50Hz 低音，统一改用本工具构造真实语义。
"""
import numpy as np


def abs_pitch_per_frame(phones, total_frames, frame_ms):
    """按音素列表生成 per-frame 绝对音高数组（cent）。

    phones: 含 position_ms / end_ms / tone 属性的对象列表（如 RenderPhone）
    frame_ms: 单帧毫秒数（hop/sr*1000）
    """
    out = np.zeros(total_frames, dtype=np.float32)
    for ph in phones:
        s = max(0, int(ph.position_ms / frame_ms))
        e = min(total_frames, int(np.ceil(ph.end_ms / frame_ms)))
        if s < e:
            out[s:e] = float(ph.tone) * 100.0
    return out.tolist()
