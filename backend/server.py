"""Diffical HTTP 服务器

接收 C# DifficalRenderer 的整句渲染请求，返回渲染结果。
"""

import os
import sys
import json
import logging
import asyncio
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# 启动方式兼容：`python -m backend.server`（模块）与 `python backend/server.py`（脚本）
# 都可用。脚本方式下把项目根目录加入 sys.path，保证 backend 包内的相对导入可解析。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from aiohttp import web
except ModuleNotFoundError:
    print("错误：当前 Python 缺少 aiohttp 模块：" + sys.executable)
    print("请用带 aiohttp 的 Python 启动（例如 ./start.sh 或：")
    print("  /opt/anaconda3/bin/python -m backend.server")
    print("或先安装依赖：pip install -r requirements.txt")
    sys.exit(1)

from backend.config import get_config
from backend.models import get_model_manager
from backend.renderer import render_phrase
from backend.protocol import RenderPhraseRequest, RenderResultResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# 全局线程池（推理用）
_executor: Optional[ThreadPoolExecutor] = None
_RUNTIME_KEY = web.AppKey('runtime_identity', dict)
AUDIO_CACHE_VERSION = '20261018handover2'


def _runtime_identity() -> dict:
    """Snapshot once at app creation, not from disk on each status request.

    These are backend source/config fingerprints, not a hash of Python, native
    dependencies or model weights. Model assets are verified separately.
    """
    sources = {str(p.relative_to(_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((Path(_ROOT) / 'backend').rglob('*.py'))}
    config = get_config()
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(',', ':')).encode()
    return {
        'algorithm': 'mel-handover-v2', 'audio_cache_version': AUDIO_CACHE_VERSION,
        'pid': os.getpid(), 'app_created_at': datetime.now(timezone.utc).isoformat(),
        'backend_source_sha256': sources,
        'backend_source_digest': hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        'config_sha256': hashlib.sha256(encoded).hexdigest(),
        'mel_coherence_width': config.processing.mel_coherence_width,
        'mel_max_compensation_db': config.processing.mel_max_compensation_db,
        'refine_enabled': config.model.refine_enabled,
    }


def get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        config = get_config()
        workers = config.server.max_workers
        # 并发渲染请求（OpenUtau 播放/导出同时进行）时避免互相挤占；
        # torch CPU 推理单请求已多线程，worker 数不宜超过核数。
        import os
        workers = max(2, min(workers, os.cpu_count() or 4))
        _executor = ThreadPoolExecutor(max_workers=workers)
    return _executor


async def handle_health(request: web.Request) -> web.Response:
    """健康检查"""
    return web.Response(text="ok", status=200)


async def handle_render(request: web.Request) -> web.Response:
    """整句渲染接口

    POST / with JSON body (RenderPhraseRequest)
    """
    try:
        body = await request.read()
        data = json.loads(body)

        # 反序列化请求
        phrase_request = _parse_request(data)

        # 真实请求落盘（诊断用）：把 C# 发送的原始 JSON 保存到 /tmp/diffical_requests/
        # 便于精确定位"引擎看到的请求"与手写测试请求的差异。
        try:
            _dump_dir = "/tmp/diffical_requests"
            os.makedirs(_dump_dir, exist_ok=True)
            _hash = 0
            try:
                _hash = int(data.get("hash") or 0)
            except Exception:
                pass
            with open(os.path.join(_dump_dir, f"req_{_hash:x}.json"), "w",
                      encoding="utf-8") as _f:
                json.dump(data, _f, ensure_ascii=False, indent=1)
        except Exception as _e:
            logger.warning(f"请求落盘失败: {_e}")

        # 真实请求日志（诊断用）：记录声库/音素/曲线等关键信息
        try:
            def _safe_num(x):
                try:
                    return round(float(x), 1) if x is not None else 0.0
                except Exception:
                    return 0.0
            _phones = phrase_request.phones or []
            _samples = []
            for p in _phones[:6]:
                _samples.append({
                    "phoneme": getattr(p, 'phoneme', ''),
                    "tone": getattr(p, 'tone', 0),
                    "dur_ms": _safe_num(getattr(p, 'duration_ms', 0)),
                    "pos_ms": _safe_num(getattr(p, 'position_ms', 0)),
                    "end_ms": _safe_num(getattr(p, 'end_ms', 0)),
                    "leading_ms": _safe_num(getattr(p, 'leading_ms', 0)),
                    "preutter_ms": _safe_num(getattr(p, 'preutter_ms', 0)),
                    "overlap_ms": _safe_num(getattr(p, 'overlap_ms', 0)),
                    "oto_file": (getattr(p, 'oto_file', '') or '')[-80:],
                    "oto_offset": _safe_num(getattr(p, 'oto_offset', 0)),
                    "oto_consonant": _safe_num(getattr(p, 'oto_consonant', 0)),
                    "oto_cutoff": _safe_num(getattr(p, 'oto_cutoff', 0)),
                    "oto_preutter": _safe_num(getattr(p, 'oto_preutter', 0)),
                    "oto_overlap": _safe_num(getattr(p, 'oto_overlap', 0)),
                    "suffix": getattr(p, 'suffix', '') or '',
                    "flags": [(f[0], f[1]) for f in (getattr(p, 'flags', None) or [])
                              if isinstance(f, (tuple, list)) and len(f) > 0],
                    "envelope": [(round(float(e[0]), 1), round(float(e[1]), 3))
                                 for e in (getattr(p, 'envelope', None) or [])],
                    "velocity": _safe_num(getattr(p, 'velocity', 1.0)),
                })
            logger.info(
                "[REQLOG] singer=%s phones=%d total_ms=%.0f hash=%d curves_pitch=%d dyn=%d gen=%d bre=%d tns=%d voi=%d "
                "first=%s",
                (phrase_request.singer_path or '')[-80:], len(_phones),
                _safe_num(phrase_request.duration_ms), getattr(phrase_request, 'hash', 0) or 0,
                len(phrase_request.pitches or []), len(phrase_request.dynamics or []),
                len(phrase_request.gender or []), len(phrase_request.breathiness or []),
                len(phrase_request.tension or []), len(phrase_request.voicing or []),
                str(_samples),
            )
        except Exception as _e:
            logger.warning(f"请求日志失败: {_e}")

        # 在线程池中执行渲染（避免阻塞事件循环）
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            get_executor(),
            render_phrase,
            phrase_request,
        )

        return web.json_response(text=result.to_json())

    except Exception as e:
        logger.exception("渲染失败")
        return web.json_response(
            {"error": str(e), "type": type(e).__name__},
            status=500,
        )


def _camel_to_snake(name: str) -> str:
    """camelCase → snake_case"""
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _convert_keys(data):
    """递归将字典的 camelCase 键转为 snake_case"""
    if isinstance(data, dict):
        return {_camel_to_snake(k): _convert_keys(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_keys(item) for item in data]
    else:
        return data


def _parse_request(data: dict) -> RenderPhraseRequest:
    """解析渲染请求（支持 camelCase 输入）"""
    from .protocol import RenderNote, RenderPhone

    # 转换 camelCase → snake_case
    data = _convert_keys(data)

    notes = [RenderNote(**n) for n in data.pop("notes", [])]

    phones = []
    for p_data in data.pop("phones", []):
        if "flags" in p_data:
            p_data["flags"] = [tuple(f) if isinstance(f, list) else f for f in p_data["flags"]]
        if "envelope" in p_data:
            # C# 发送 DifficalEnvelopePoint {x,y} 对象 -> 转 (x, y) 元组
            p_data["envelope"] = [
                (float(e["x"]), float(e["y"])) if isinstance(e, dict) else tuple(e)
                for e in p_data["envelope"]]
        phones.append(RenderPhone(**p_data))

    curves = [tuple(c) if isinstance(c, list) else c for c in data.pop("curves", [])]

    return RenderPhraseRequest(notes=notes, phones=phones, curves=curves, **data)


async def handle_status(request: web.Request) -> web.Response:
    """服务器状态"""
    mm = get_model_manager()
    config = get_config()

    status = {
        "version": "0.1.0",
        "runtime": request.app[_RUNTIME_KEY],
        "models_loaded": {
            "vocoder": mm.ort_session is not None or mm.vocoder is not None,
            "hnsep": mm.hnsep_model is not None,
            "mel_analyzer": mm.mel_analyzer is not None,
        },
        "config": {
            "sample_rate": config.audio.sample_rate,
            "hop_size": config.audio.hop_size,
            "n_mels": config.audio.n_mels,
            "device": config.model.device,
        },
    }
    return web.json_response(status)


def create_app() -> web.Application:
    """创建 aiohttp 应用"""
    app = web.Application()
    app[_RUNTIME_KEY] = _runtime_identity()
    app.router.add_get("/", handle_health)
    app.router.add_get("/status", handle_status)
    app.router.add_post("/render", handle_render)
    return app


def main():
    """启动服务器"""
    config = get_config()

    logger.info("=" * 60)
    logger.info("Diffical UTAU Engine Backend")
    logger.info(f"  Host: {config.server.host}")
    logger.info(f"  Port: {config.server.port}")
    logger.info(f"  Device: {config.model.device}")
    logger.info(f"  Sample Rate: {config.audio.sample_rate}")
    logger.info(f"  Workers: {config.server.max_workers}")
    logger.info("=" * 60)

    # 初始化模型
    logger.info("正在初始化模型...")
    mm = get_model_manager()
    mm.initialize()

    # 启动服务器
    app = create_app()
    identity = app[_RUNTIME_KEY]
    logger.info('Runtime: cache=%s source=%s config=%s', identity['audio_cache_version'],
                identity['backend_source_digest'], identity['config_sha256'])
    try:
        web.run_app(
            app,
            host=config.server.host,
            port=config.server.port,
            access_log=None,  # 关闭访问日志
        )
    except OSError as e:
        if e.errno in (48, 98) or "address already in use" in str(e).lower():
            print(f"端口 {config.server.port} 已被占用（可能已有 Diffical 后端在运行）。")
            print("如果 OpenUtau 已自动启动后端，请直接使用，无需重复启动。")
            print("如需重启：先关闭占用 8573 端口的进程（或退出 OpenUtau）。")
            sys.exit(2)
        raise


if __name__ == "__main__":
    main()
