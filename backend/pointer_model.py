"""兼容层：模型定义只在 model/pointer.py 维护，这里只做转出。

历史上接入层带过一份复制品；现在单一代码源是 `diffical/model/pointer.py`，
避免两份副本漂移。
"""
from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.pointer import (  # noqa: E402,F401
    N_MELS,
    PointerGen,
    count_params,
    gather_seg,
    load_pointer,
)
