"""Diffical 性能优化模块

提供音素特征缓存、Mel 频谱缓存、并行处理、JIT 编译、性能监控等功能，
与现有渲染流程无缝集成，提升整体渲染性能。

主要功能：
- 音素特征缓存（ContentVec）
- Mel 频谱缓存
- 音素级并行处理
- 热点函数 JIT 编译（numba 可选）
- LRU 缓存策略与内存管理
- 性能监控与统计
"""

import os
import time
import logging
import threading
import functools
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

try:
    from numba import jit as _numba_jit
    NUMBA_AVAILABLE = True
    logger.info("numba 可用，热点函数将启用 JIT 编译")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.info("numba 不可用，热点函数将使用纯 Python 实现")


def optional_jit(*args, **kwargs) -> Callable[[F], F]:
    """可选 JIT 编译装饰器

    如果 numba 可用，则使用 numba.jit 编译函数；
    否则直接返回原函数，不影响功能。

    Args:
        *args: 传递给 numba.jit 的位置参数
        **kwargs: 传递给 numba.jit 的关键字参数

    Returns:
        装饰后的函数
    """
    def decorator(func: F) -> F:
        if NUMBA_AVAILABLE:
            try:
                return cast(F, _numba_jit(*args, **kwargs)(func))
            except Exception as e:
                logger.warning(f"JIT 编译失败 {func.__name__}: {e}，回退到纯 Python")
                return func
        return func
    return decorator


class LRUDict:
    """线程安全的 LRU 缓存字典

    使用 OrderedDict 实现 LRU 策略，支持最大容量限制和自动过期清理。
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[float] = None):
        """初始化 LRU 缓存

        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存存活时间（秒），None 表示永不过期
        """
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[Any, float] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: Any) -> Optional[Any]:
        """获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值，如果不存在或已过期则返回 None
        """
        with self._lock:
            if key in self._cache:
                if self._ttl is not None:
                    age = time.time() - self._timestamps.get(key, 0)
                    if age > self._ttl:
                        del self._cache[key]
                        self._timestamps.pop(key, None)
                        self._evictions += 1
                        self._misses += 1
                        return None
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def set(self, key: Any, value: Any) -> None:
        """设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    self._timestamps.pop(oldest_key, None)
                    self._evictions += 1
            self._cache[key] = value
            self._timestamps[key] = time.time()

    def delete(self, key: Any) -> bool:
        """删除缓存条目

        Args:
            key: 缓存键

        Returns:
            是否删除成功
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._timestamps.pop(key, None)
                return True
            return False

    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        with self._lock:
            return key in self._cache

    @property
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate': hit_rate,
            }

    def estimate_memory_bytes(self, sample_size: int = 10) -> int:
        """估算缓存占用的内存大小（字节）

        通过采样部分条目估算整体内存使用。

        Args:
            sample_size: 采样数量

        Returns:
            估算的内存大小（字节）
        """
        with self._lock:
            if not self._cache:
                return 0
            items = list(self._cache.values())[:sample_size]
            total_size = 0
            for item in items:
                if isinstance(item, np.ndarray):
                    total_size += item.nbytes
                elif isinstance(item, (list, tuple)):
                    total_size += len(item) * 8
                else:
                    total_size += 64
            avg_size = total_size / len(items) if items else 0
            return int(avg_size * len(self._cache))


class timer:
    """性能计时上下文管理器

    用法：
        with timer("operation_name") as t:
            do_something()
        print(f"耗时: {t.elapsed:.3f}s")
    """

    def __init__(self, name: str = "", log_level: int = logging.DEBUG):
        """初始化计时器

        Args:
            name: 计时操作名称
            log_level: 日志级别
        """
        self.name = name
        self.log_level = log_level
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> 'timer':
        self._start = time.perf_counter()
        if self.name:
            logger.log(self.log_level, f"[{self.name}] 开始执行")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.name:
            logger.log(
                self.log_level,
                f"[{self.name}] 执行完成，耗时 {self.elapsed:.4f}s"
            )


def profile_function(func: F) -> F:
    """函数性能分析装饰器

    记录函数的调用次数、总耗时、平均耗时等统计信息。

    Args:
        func: 要分析的函数

    Returns:
        装饰后的函数
    """
    stats = {
        'calls': 0,
        'total_time': 0.0,
        'min_time': float('inf'),
        'max_time': 0.0,
    }
    lock = threading.Lock()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            with lock:
                stats['calls'] += 1
                stats['total_time'] += elapsed
                stats['min_time'] = min(stats['min_time'], elapsed)
                stats['max_time'] = max(stats['max_time'], elapsed)

    wrapper._profile_stats = stats
    return cast(F, wrapper)


class PerformanceOptimizer:
    """性能优化器

    提供音素特征缓存、Mel 频谱缓存、并行处理、内存管理等功能，
    用于优化 Diffical 渲染流程的性能。
    """

    _instance: Optional['PerformanceOptimizer'] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        max_contentvec_cache: int = 500,
        max_mel_cache: int = 200,
        contentvec_ttl: Optional[float] = None,
        mel_ttl: Optional[float] = 3600.0,
        memory_limit_mb: float = 512.0,
    ):
        """初始化性能优化器

        Args:
            max_contentvec_cache: ContentVec 特征缓存最大条目数
            max_mel_cache: Mel 频谱缓存最大条目数
            contentvec_ttl: ContentVec 缓存存活时间（秒），None 表示永不过期
            mel_ttl: Mel 缓存存活时间（秒），默认 1 小时
            memory_limit_mb: 内存限制（MB），超过时自动清理
        """
        self._contentvec_cache = LRUDict(
            max_size=max_contentvec_cache,
            ttl_seconds=contentvec_ttl,
        )
        self._mel_cache = LRUDict(
            max_size=max_mel_cache,
            ttl_seconds=mel_ttl,
        )
        self._memory_limit = memory_limit_mb * 1024 * 1024
        self._lock = threading.RLock()
        self._timers: Dict[str, List[float]] = {}
        self._function_stats: Dict[str, Dict[str, Any]] = {}
        self._cpu_count = multiprocessing.cpu_count()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()

    @classmethod
    def get_instance(cls) -> 'PerformanceOptimizer':
        """获取单例实例

        Returns:
            PerformanceOptimizer 单例
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_cached_contentvec_feature(self, phoneme_key: str) -> Optional[np.ndarray]:
        """获取缓存的 ContentVec 特征

        Args:
            phoneme_key: 音素键（通常是音素名或文件路径）

        Returns:
            缓存的特征数组，如果不存在则返回 None
        """
        result = self._contentvec_cache.get(phoneme_key)
        if result is not None:
            logger.debug(f"ContentVec 缓存命中: {phoneme_key}")
        return result

    def cache_contentvec_feature(self, phoneme_key: str, feature: np.ndarray) -> None:
        """缓存 ContentVec 特征

        Args:
            phoneme_key: 音素键
            feature: 特征数组
        """
        self._contentvec_cache.set(phoneme_key, feature.copy())
        self._check_memory_limit()

    def get_cached_mel(
        self,
        phoneme_key: str,
        pitch: float,
        key_shift: float = 0.0,
    ) -> Optional[np.ndarray]:
        """获取缓存的 Mel 频谱

        Args:
            phoneme_key: 音素键
            pitch: 音高（Hz 或 MIDI）
            key_shift: 音高偏移（半音）

        Returns:
            缓存的 Mel 频谱，如果不存在则返回 None
        """
        cache_key = f"{phoneme_key}|{pitch:.2f}|{key_shift:.2f}"
        result = self._mel_cache.get(cache_key)
        if result is not None:
            logger.debug(f"Mel 缓存命中: {cache_key}")
        return result

    def cache_mel(
        self,
        phoneme_key: str,
        pitch: float,
        key_shift: float,
        mel: np.ndarray,
    ) -> None:
        """缓存 Mel 频谱

        Args:
            phoneme_key: 音素键
            pitch: 音高
            key_shift: 音高偏移
            mel: Mel 频谱数组
        """
        cache_key = f"{phoneme_key}|{pitch:.2f}|{key_shift:.2f}"
        self._mel_cache.set(cache_key, mel.copy())
        self._check_memory_limit()

    def clear_cache(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._contentvec_cache.clear()
            self._mel_cache.clear()
            logger.info("所有缓存已清空")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息

        Returns:
            包含各缓存统计信息的字典
        """
        contentvec_stats = self._contentvec_cache.stats
        mel_stats = self._mel_cache.stats
        contentvec_memory = self._contentvec_cache.estimate_memory_bytes()
        mel_memory = self._mel_cache.estimate_memory_bytes()
        total_memory = contentvec_memory + mel_memory

        return {
            'contentvec': {
                **contentvec_stats,
                'memory_bytes': contentvec_memory,
                'memory_mb': contentvec_memory / (1024 * 1024),
            },
            'mel': {
                **mel_stats,
                'memory_bytes': mel_memory,
                'memory_mb': mel_memory / (1024 * 1024),
            },
            'total': {
                'memory_bytes': total_memory,
                'memory_mb': total_memory / (1024 * 1024),
                'memory_limit_mb': self._memory_limit / (1024 * 1024),
            },
            'numba_available': NUMBA_AVAILABLE,
            'cpu_count': self._cpu_count,
        }

    def process_phonemes_parallel(
        self,
        phoneme_list: List[Any],
        process_func: Callable[[Any], Any],
        n_jobs: Optional[int] = None,
        use_processes: bool = False,
    ) -> List[Any]:
        """音素级并行处理

        使用多进程或多线程并行处理音素列表。
        对于 CPU 密集型任务建议使用多进程，
        对于 I/O 密集型任务建议使用多线程。

        Args:
            phoneme_list: 音素列表
            process_func: 处理单个音素的函数
            n_jobs: 并行工作数，None 表示自动检测
            use_processes: 是否使用多进程（默认多线程）

        Returns:
            处理结果列表，顺序与输入一致
        """
        if not phoneme_list:
            return []

        if n_jobs is None:
            n_jobs = min(self._cpu_count, len(phoneme_list))
        n_jobs = max(1, min(n_jobs, len(phoneme_list)))

        if n_jobs == 1:
            logger.debug(f"单线程处理 {len(phoneme_list)} 个音素")
            return [process_func(p) for p in phoneme_list]

        executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        mode = "多进程" if use_processes else "多线程"
        logger.debug(f"使用 {mode} 并行处理 {len(phoneme_list)} 个音素，工作数={n_jobs}")

        results = [None] * len(phoneme_list)
        with executor_cls(max_workers=n_jobs) as executor:
            future_to_idx = {
                executor.submit(process_func, phoneme): idx
                for idx, phoneme in enumerate(phoneme_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"音素处理失败 (索引 {idx}): {e}")
                    results[idx] = None

        return results

    def _check_memory_limit(self) -> None:
        """检查内存使用，超过限制时自动清理"""
        total_memory = (
            self._contentvec_cache.estimate_memory_bytes()
            + self._mel_cache.estimate_memory_bytes()
        )
        if total_memory > self._memory_limit:
            logger.warning(
                f"内存使用 {total_memory / (1024*1024):.1f}MB "
                f"超过限制 {self._memory_limit / (1024*1024):.1f}MB，触发清理"
            )
            self._evict_memory(total_memory - self._memory_limit)

    def _evict_memory(self, bytes_to_free: int) -> None:
        """驱逐缓存以释放内存

        优先驱逐 Mel 缓存（占用更大），然后是 ContentVec 缓存。

        Args:
            bytes_to_free: 需要释放的字节数
        """
        freed = 0
        mel_stats = self._mel_cache.stats
        if mel_stats['size'] > 0:
            mel_memory = self._mel_cache.estimate_memory_bytes()
            if mel_memory > 0:
                items_per_mb = mel_stats['size'] / (mel_memory / (1024 * 1024))
                items_to_remove = max(1, int(bytes_to_free / (mel_memory / mel_stats['size'])))
                items_to_remove = min(items_to_remove, mel_stats['size'])
                for _ in range(items_to_remove):
                    oldest_key = next(iter(self._mel_cache._cache))
                    self._mel_cache.delete(oldest_key)
                freed = int(items_to_remove * (mel_memory / mel_stats['size']))

        if freed < bytes_to_free:
            remaining = bytes_to_free - freed
            contentvec_stats = self._contentvec_cache.stats
            if contentvec_stats['size'] > 0:
                cv_memory = self._contentvec_cache.estimate_memory_bytes()
                if cv_memory > 0:
                    items_to_remove = max(
                        1, int(remaining / (cv_memory / contentvec_stats['size']))
                    )
                    items_to_remove = min(items_to_remove, contentvec_stats['size'])
                    for _ in range(items_to_remove):
                        oldest_key = next(iter(self._contentvec_cache._cache))
                        self._contentvec_cache.delete(oldest_key)

    def start_background_cleanup(self, interval_seconds: float = 60.0) -> None:
        """启动后台清理线程

        定期检查并清理过期缓存。

        Args:
            interval_seconds: 清理间隔（秒）
        """
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return

        self._stop_cleanup.clear()

        def cleanup_loop():
            while not self._stop_cleanup.is_set():
                try:
                    self._check_memory_limit()
                    self._purge_expired()
                except Exception as e:
                    logger.warning(f"后台清理出错: {e}")
                self._stop_cleanup.wait(interval_seconds)

        self._cleanup_thread = threading.Thread(
            target=cleanup_loop,
            daemon=True,
            name="perf-optimizer-cleanup",
        )
        self._cleanup_thread.start()
        logger.info(f"后台清理线程已启动，间隔 {interval_seconds}s")

    def stop_background_cleanup(self) -> None:
        """停止后台清理线程"""
        if self._cleanup_thread:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5.0)
            self._cleanup_thread = None
            logger.info("后台清理线程已停止")

    def _purge_expired(self) -> None:
        """清理所有过期缓存"""
        now = time.time()
        for cache in [self._contentvec_cache, self._mel_cache]:
            if cache._ttl is None:
                continue
            with cache._lock:
                expired_keys = []
                for key, timestamp in cache._timestamps.items():
                    if now - timestamp > cache._ttl:
                        expired_keys.append(key)
                for key in expired_keys:
                    del cache._cache[key]
                    del cache._timestamps[key]
                    cache._evictions += 1
                if expired_keys:
                    logger.debug(f"清理了 {len(expired_keys)} 个过期缓存条目")

    def record_timer(self, name: str, elapsed: float) -> None:
        """记录计时数据

        Args:
            name: 操作名称
            elapsed: 耗时（秒）
        """
        with self._lock:
            if name not in self._timers:
                self._timers[name] = []
            self._timers[name].append(elapsed)

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息

        Returns:
            包含性能统计的字典
        """
        with self._lock:
            timer_stats = {}
            for name, times in self._timers.items():
                if times:
                    timer_stats[name] = {
                        'count': len(times),
                        'total': sum(times),
                        'avg': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times),
                    }

            return {
                'timers': timer_stats,
                'cache': self.get_cache_stats(),
            }


_optimizer_instance: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """获取全局性能优化器实例

    Returns:
        PerformanceOptimizer 单例
    """
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = PerformanceOptimizer.get_instance()
    return _optimizer_instance


def contentvec_cached(phoneme_key_func: Callable[..., str]) -> Callable[[F], F]:
    """ContentVec 特征缓存装饰器

    自动缓存函数返回的 ContentVec 特征。

    Args:
        phoneme_key_func: 从参数中提取音素键的函数

    Returns:
        装饰器
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            key = phoneme_key_func(*args, **kwargs)
            cached = optimizer.get_cached_contentvec_feature(key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            if result is not None and isinstance(result, np.ndarray):
                optimizer.cache_contentvec_feature(key, result)
            return result
        return cast(F, wrapper)
    return decorator


def mel_cached(
    phoneme_key_func: Callable[..., str],
    pitch_func: Callable[..., float],
    key_shift_func: Callable[..., float] = lambda *a, **kw: 0.0,
) -> Callable[[F], F]:
    """Mel 频谱缓存装饰器

    自动缓存函数返回的 Mel 频谱。

    Args:
        phoneme_key_func: 从参数中提取音素键的函数
        pitch_func: 从参数中提取音高的函数
        key_shift_func: 从参数中提取音高偏移的函数

    Returns:
        装饰器
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            key = phoneme_key_func(*args, **kwargs)
            pitch = pitch_func(*args, **kwargs)
            key_shift = key_shift_func(*args, **kwargs)
            cached = optimizer.get_cached_mel(key, pitch, key_shift)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            if result is not None and isinstance(result, np.ndarray):
                optimizer.cache_mel(key, pitch, key_shift, result)
            return result
        return cast(F, wrapper)
    return decorator
