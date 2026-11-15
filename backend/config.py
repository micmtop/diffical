"""Diffical 配置管理

管理模型路径、音频参数、处理选项等配置。
支持从 YAML 文件加载配置。
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """模型相关配置"""
    # pc-nsf-hifigan vocoder
    vocoder_path: str = "./models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx"
    vocoder_config_path: str = "./models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/config.json"
    model_type: str = "onnx"  # 官方 model.onnx；权重文件只读，不在代码中替换

    # hnsep CascadedNet
    hnsep_model_path: str = "./models/hnsep/vr/model.pt"
    hnsep_config_path: str = "./models/hnsep/vr/config.yaml"

    # AI mel 精修暂时硬关闭；模型文件/训练数据保留，未来重新立项时再接回。
    refine_model_path: str = ""
    refine_enabled: bool = False

    # ContentVec（跨语种音素映射和特征提取）
    contentvec_model_path: str = "./models/contentvec/"

    # 设备
    device: str = "cpu"  # "cpu", "cuda", "mps"
    use_onnx: bool = True


@dataclass
class AudioConfig:
    """音频参数配置"""
    sample_rate: int = 44100
    win_size: int = 2048
    hop_size: int = 512           # vocoder 帧移（必须与模型训练一致）
    origin_hop_size: int = 128    # mel 提取用的更细 hop（用于时间拉伸）
    n_fft: int = 2048
    n_mels: int = 128
    mel_fmin: float = 40.0
    mel_fmax: float = 16000.0


@dataclass
class ProcessingConfig:
    """处理选项配置"""
    # 响度归一化
    wave_norm: bool = True
    trim_silence: bool = True
    silence_threshold: float = -52.0  # dB

    # 拼接
    crossfade_ms: float = 30.0  # legacy/debug 路径；生产 overlap 时长由 OpenUtau envelope 决定
    mel_coherence_width: float = .08  # [0+,1]：越小越早使用等功率谱交接
    mel_max_compensation_db: float = 3.01029995664  # 中性结构淡化的能量补偿上限；不是波形限幅
    phase_align: bool = True     # 相位对齐

    # 缓存
    use_cache: bool = True
    cache_dir: str = "./cache/"

    # 极致音质模式
    ultra_quality: bool = True


@dataclass
class ServerConfig:
    """HTTP 服务器配置"""
    host: str = "127.0.0.1"
    port: int = 8573  # 与 默认端口
    max_workers: int = 2  # 推理线程池大小


@dataclass
class DifficalConfig:
    """Diffical 总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "DifficalConfig":
        """从 YAML 文件加载配置"""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        config = cls()
        if "model" in data:
            for k, v in data["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        if "audio" in data:
            for k, v in data["audio"].items():
                if hasattr(config.audio, k):
                    setattr(config.audio, k, v)
        if "processing" in data:
            for k, v in data["processing"].items():
                if hasattr(config.processing, k):
                    setattr(config.processing, k, v)
        if "server" in data:
            for k, v in data["server"].items():
                if hasattr(config.server, k):
                    setattr(config.server, k, v)
        return config

    @classmethod
    def default(cls) -> "DifficalConfig":
        """返回默认配置"""
        return cls()


# 全局配置实例
_config: Optional[DifficalConfig] = None


def get_config() -> DifficalConfig:
    """获取全局配置"""
    global _config
    if _config is None:
        config_path = os.environ.get("DIFFICAL_CONFIG", "config.yaml")
        if not os.path.exists(config_path):
            # 仓库约定：模板在 docs/config.example.yaml，用户复制成 docs/config.yaml。
            # 从仓库根目录启动时没传 DIFFICAL_CONFIG 也能找到它。
            repo_docs = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "docs", "config.yaml")
            if os.path.exists(repo_docs):
                config_path = repo_docs
        if os.path.exists(config_path):
            _config = DifficalConfig.from_yaml(config_path)
        else:
            _config = DifficalConfig.default()
    return _config


def set_config(config: DifficalConfig):
    """设置全局配置"""
    global _config
    _config = config
