"""为 P0.3 第三臂选捐赠句：同歌手（OpenCpop 全库同一女声）不同歌的同音素片段。
用法：python3 tools/make_donors.py [n]
输出：data/opencpop/donors.json（含要解压的 id 列表）
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from opencpop import parse_file  # noqa: E402


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    utts = parse_file(ROOT / "data/opencpop/transcriptions.txt")
    targets = json.loads((ROOT / "data/opencpop/selected12.json").read_text(encoding="utf-8"))
    target_ids = {t["utt_id"] for t in targets}
    target_songs = {int(t["utt_id"]) // 10**6 for t in targets}

    pool = [u for u in utts
            if u.utt_id not in target_ids
            and int(u.utt_id) // 10**6 not in target_songs
            and 3.0 <= u.total_s <= 8.0]
    rng = random.Random(0)
    rng.shuffle(pool)

    # 贪心扩大音素覆盖：优先覆盖尚未被覆盖的音素
    covered: set[str] = set()
    picked = []
    remaining = pool[:]
    while remaining and len(picked) < n:
        best, best_gain = None, -1
        for u in remaining[:400]:
            gain = len(set(u.phones) - covered)
            if gain > best_gain:
                best, best_gain = u, gain
        if best is None or best_gain <= 0:
            break
        picked.append(best)
        covered |= set(best.phones)
        remaining.remove(best)
    for u in remaining[: n - len(picked)]:
        picked.append(u)

    out = {
        "n": len(picked),
        "covered_phonemes": sorted(covered),
        "ids": [u.utt_id for u in picked],
    }
    path = ROOT / "data/opencpop/donors.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"donors={len(picked)} covered={len(covered)} -> {path}")
    print(" ".join(out["ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
