"""OpenCpop 标注解析与 P0.3 边界选例。

字段（7 段 | 分隔）：
    name | lyrics | ph_seq | note_pitch | midi_dur | ph_dur | is_slur

- ph_dur 单位是**秒**，且其求和等于该句 wav 时长（已实测：4.078s vs 359718B/88200）。
- midi_dur 不是计时量（求和与 wav 时长差 ~2×），不得用于边界定位。
- 边界时刻 = ph_dur 的累积和。
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, asdict
from typing import Iterable

# 普通话声母里的塞音（含送气写法）
PLOSIVES = {"b", "p", "d", "t", "g", "k", "p_h", "t_h", "k_h"}
# 韵母：以 a/o/e/i/u/v 开头且可能带韵尾的 token 都视作元音性
VOWEL_INITIALS = set("aoeiuv")
RESTS = {"SP", "AP"}


@dataclass
class Utterance:
    utt_id: str
    lyrics: str
    phones: list[str]
    pitches: list[str]
    midi_dur: list[float]
    ph_dur: list[float]
    slur: list[int]

    @property
    def total_s(self) -> float:
        return sum(self.ph_dur)

    def boundaries_s(self) -> list[float]:
        """每个音素边界（不含 0 与末尾）的绝对时刻，单位秒。"""
        out, t = [], 0.0
        for d in self.ph_dur[:-1]:
            t += d
            out.append(t)
        return out


def parse_line(line: str) -> Utterance:
    f = line.split("|")
    if len(f) != 7:
        raise ValueError(f"expected 7 fields, got {len(f)}")
    phones = f[2].split()
    ph_dur = [float(x) for x in f[5].split()]
    if len(phones) != len(ph_dur):
        raise ValueError(f"{f[0]}: len(ph)={len(phones)} != len(ph_dur)={len(ph_dur)}")
    return Utterance(
        utt_id=f[0],
        lyrics=f[1],
        phones=phones,
        pitches=f[3].split(),
        midi_dur=[float(x) for x in f[4].split()],
        ph_dur=ph_dur,
        slur=[int(float(x)) for x in f[6].split()],
    )


def parse_file(path: str | pathlib.Path) -> list[Utterance]:
    out = []
    for line in pathlib.Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            out.append(parse_line(line))
    return out


def is_vowel(ph: str) -> bool:
    return bool(ph) and ph[0] in VOWEL_INITIALS


def is_nasal_final(ph: str) -> bool:
    return is_vowel(ph) and (ph.endswith("n") or ph.endswith("ng"))


def boundary_types(u: Utterance) -> dict[str, list[int]]:
    """返回 {类型: [边界下标]}；下标 i 表示 phones[i] 的起点边界。"""
    res: dict[str, list[int]] = {"plosive_onset": [], "nasal_coda": [], "long_to_breath": [], "same_vowel": []}
    for i in range(1, len(u.phones)):
        prev, cur = u.phones[i - 1], u.phones[i]
        if cur in PLOSIVES:
            res["plosive_onset"].append(i)
        if is_nasal_final(prev) and is_vowel(cur) and cur != prev:
            res["nasal_coda"].append(i)
        if cur in RESTS and is_vowel(prev) and u.ph_dur[i - 1] >= 0.5:
            res["long_to_breath"].append(i)
        if prev == cur and is_vowel(cur):
            res["same_vowel"].append(i)
    return res


def select(utts: Iterable[Utterance], per_type: int = 3) -> list[dict]:
    """每种类型挑 per_type 条，优先：边界多、句子 3–8s、AP 少、id 不重复。"""
    picked: list[dict] = []
    used: set[str] = set()
    for btype in ("plosive_onset", "nasal_coda", "long_to_breath", "same_vowel"):
        scored = []
        for u in utts:
            idxs = boundary_types(u)[btype]
            if not idxs:
                continue
            if not (3.0 <= u.total_s <= 8.0):
                continue
            ap = sum(1 for p in u.phones if p == "AP")
            scored.append(((-len(idxs), ap, u.total_s), u, idxs))
        scored.sort(key=lambda x: x[0])
        n = 0
        for _, u, idxs in scored:
            if u.utt_id in used:
                continue
            used.add(u.utt_id)
            i = idxs[0]
            b = u.boundaries_s()
            picked.append({
                "type": btype,
                "utt_id": u.utt_id,
                "boundary_index": i,
                "boundary_s": b[i - 1],
                "prev_phone": u.phones[i - 1],
                "next_phone": u.phones[i],
                "prev_dur_s": u.ph_dur[i - 1],
                "total_s": u.total_s,
            })
            n += 1
            if n >= per_type:
                break
    return picked


if __name__ == "__main__":
    import collections
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    utts = parse_file(root / "data/opencpop/transcriptions.txt")
    print(f"utterances: {len(utts)}")
    inv = collections.Counter(p for u in utts for p in u.phones)
    print(f"phoneme inventory ({len(inv)}):", " ".join(sorted(inv)))
    sel = select(utts)
    out = root / "data/opencpop/selected12.json"
    out.write_text(json.dumps(sel, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"selected: {len(sel)} -> {out}")
    for s in sel:
        print(f"  {s['type']:16s} {s['utt_id']} @ {s['boundary_s']:.3f}s  {s['prev_phone']} -> {s['next_phone']}")
    sys.exit(0 if len(sel) == 12 else 1)
