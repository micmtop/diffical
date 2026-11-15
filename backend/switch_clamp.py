"""临时切换 renderer.py 的 start_frame clamp 行为: mode=base|exact
用法: python backend/switch_clamp.py base|exact
"""
import sys, re

def apply(mode):
    p = "backend/renderer.py"
    src = open(p).read()
    if mode == "exact":
        # 恢复精确修复 (负 start_frame)
        src = src.replace(
            """            audio_start_ms = target_start_ms - pre_vel_ms
            start_frame = int(audio_start_ms / self.frame_ms)
            # 允许负 start_frame：片段起点可在短语开头之前（VCV 前母音提前），
            # 拼接时负帧位置静音、内容不错位（clamp 到 0 会让 mel 内容整体后移，
            # 子音/元音落入 envelope 淡出区被切掉 -> 咬字丢失 + 电平凹陷）。""",
            """            audio_start_ms = target_start_ms - pre_vel_ms
            start_frame = int(audio_start_ms / self.frame_ms)
            start_frame = max(0, start_frame)""")
        src = src.replace('_jj = fr - _s["st_raw"]', '_jj = fr - _s["st"]')
        src = src.replace('_jj2 = fr - _s2["st_raw"]', '_jj2 = fr - _s2["st"]')
        src = src.replace('_pre_end = min(_note_s, _s["en"]) - _s["st_raw"]',
                          '_pre_end = min(_note_s, _s["en"]) - _s["st"]')
        src = src.replace('"st_raw": st_raw, ', '')
        src = src.replace('t_ms = (st_raw + np.arange(T, dtype=np.float64)) * frame_ms - note_abs_ms',
                          't_ms = (st + np.arange(T, dtype=np.float64)) * frame_ms - note_abs_ms')
        src = src.replace('st_raw = feat["start_frame"]  # 可为负（前母音提前于短语开头）\n            st = max(0, st_raw)',
                          'st = max(0, feat["start_frame"])')
        src = src.replace('_T2 = _ds2["mel"].shape[1]\n                _in_pre2 = (_ds2["st_raw"] + _dj2) < _ds2.get("note_start_fr", _ds2["en"])',
                          '_T2 = _ds2["en"] - _ds2["st"]\n                _in_pre2 = (_ds2["st"] + _dj2) < _ds2.get("note_start_fr", _ds2["en"])')
        src = src.replace('_src_e = min(len(f0_phrase), start_frame + target_n_frames)\n            if _src_e > _src_s:\n                f0_slice[_src_s - start_frame:_src_e - start_frame] = f0_phrase[_src_s:_src_e]',
                          'end_frame = start_frame + target_n_frames\n            if end_frame <= len(f0_phrase):\n                f0_slice = f0_phrase[start_frame:end_frame].copy()\n            elif start_frame < len(f0_phrase):\n                f0_slice[:len(f0_phrase) - start_frame] = f0_phrase[start_frame:].copy()')
    else:
        raise ValueError("only supports base (revert) via exact arg")
    open(p, "w").write(src)
    print(f"switched to {mode}")

if __name__ == "__main__":
    apply("exact")
