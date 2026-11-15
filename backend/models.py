"""模型管理器

管理：
- pc-nsf-hifigan vocoder
- hnsep CascadedNet
- Mel 频谱分析器
- ContentVec 音素特征提取器

注意：必须先下载模型文件，不提供回退方案。
"""

import os
import logging
from typing import Optional

import numpy as np

from .config import get_config

logger = logging.getLogger(__name__)


class ModelManager:
    """全局模型管理器（单例）"""

    _instance: Optional["ModelManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.vocoder = None
        self.ort_session = None
        self.hnsep_model = None
        self.mel_analyzer = None
        self._initialized = True

    def initialize(self):
        """初始化所有模型（必需）

        hnsep 改为懒加载：只有渲染带 flag/曲线的音频时才需要（_get_hnsep），
        默认渲染路径用内置频谱分离即可，启动/首渲染可省 ~1.5s。
        """
        config = get_config()
        logger.info("开始初始化模型...")

        try:
            self.init_mel_analyzer()
            self._init_vocoder(config)
        except Exception as e:
            # MPS/GPU 加载失败（torch MPS 对小张量有已知 bug）时回退 CPU，
            # 保证 device=mps 配置不会导致渲染整体崩溃。
            if config.model.device != "cpu":
                logger.warning(f"{config.model.device} 模型加载失败({e})，回退 CPU")
                config.model.device = "cpu"
                self.vocoder = None
                self.ort_session = None
                self.mel_analyzer = None
                self.init_mel_analyzer()
                self._init_vocoder(config)
            else:
                raise

        logger.info("模型初始化完成。")

    def _get_hnsep(self):
        """按需初始化 hnsep CascadedNet（首次使用 flag/曲线时）"""
        if self.hnsep_model is None:
            config = get_config()
            self._init_hnsep(config)
        return self.hnsep_model

    def init_mel_analyzer(self):
        """初始化 PitchAdjustableMelSpectrogram
        
        使用 origin_hop_size=128 提取 Mel（更细的时间分辨率），
        然后插值到 hop_size=512 给声码器。
        参考: hifisampler 使用 hop=128 提取，插值到 hop=512
        """
        from .vocoder.wav2mel import PitchAdjustableMelSpectrogram
        config = get_config()

        self.mel_analyzer = PitchAdjustableMelSpectrogram(
            sample_rate=config.audio.sample_rate,
            win_length=config.audio.win_size,
            hop_length=config.audio.origin_hop_size,
            n_fft=config.audio.n_fft,
            n_mels=config.audio.n_mels,
            f_min=config.audio.mel_fmin,
            f_max=config.audio.mel_fmax,
            device=config.model.device,
        )
        self.mel_hop_ratio = config.audio.hop_size // config.audio.origin_hop_size
        logger.info(f"Mel 分析器初始化完成 (提取hop={config.audio.origin_hop_size}, 声码器hop={config.audio.hop_size}, 插值比例={self.mel_hop_ratio})")

    def _init_vocoder(self, config):
        """初始化 pc-nsf-hifigan vocoder（必需）"""
        # ONNX 只支持 CPU（onnxruntime 无 MPS 后端）；device=mps 时强制 ckpt
        if config.model.model_type == "onnx" and config.model.device != "cpu":
            logger.warning(f"ONNX vocoder 仅支持 CPU，device={config.model.device} 强制回退 ckpt")
            config.model.model_type = "ckpt"
            # MPS 用官方权重（与 ONNX 一致），不用 model_fixed_v2（与官方差异大）
            config.model.vocoder_path = config.model.vocoder_path.replace(
                "model.onnx", "model.ckpt")
        if config.model.model_type == "onnx":
            self._init_vocoder_onnx(config)
        else:
            self._init_vocoder_ckpt(config)

        if self.ort_session is None and self.vocoder is None:
            raise RuntimeError(
                "Vocoder 模型初始化失败！请确保已下载模型文件。\n"
                f"期望路径: {config.model.vocoder_path}\n"
                "查看 MODELS.md 获取模型下载清单。"
            )

    def _init_vocoder_onnx(self, config):
        """ONNX 模式初始化 vocoder"""
        import onnxruntime as ort

        vocoder_path = config.model.vocoder_path
        if not os.path.exists(vocoder_path):
            raise FileNotFoundError(
                f"Vocoder 模型文件不存在: {vocoder_path}\n"
                "请从 https://huggingface.co/openvpi/pc-nsf-hifigan 下载。"
            )

        providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 2

        self.ort_session = ort.InferenceSession(
            vocoder_path,
            sess_options=opts,
            providers=providers,
        )
        logger.info(f"Vocoder (ONNX) 加载完成: {vocoder_path}")

    def _init_vocoder_ckpt(self, config):
        """PyTorch checkpoint 模式初始化 vocoder"""
        from .vocoder.nsf_hifigan import NsfHifiGAN

        vocoder_path = config.model.vocoder_path
        config_path = config.model.vocoder_config_path

        if not os.path.exists(vocoder_path):
            raise FileNotFoundError(
                f"Vocoder 模型文件不存在: {vocoder_path}\n"
                "请从 https://huggingface.co/openvpi/pc-nsf-hifigan 下载。"
            )

        self.vocoder = NsfHifiGAN(
            model_path=vocoder_path,
            config_path=config_path,
            device=config.model.device,
        )
        logger.info(f"Vocoder (ckpt) 加载完成: {vocoder_path}")

    def _init_hnsep(self, config):
        """初始化 hnsep CascadedNet（可选）"""
        try:
            from .vocoder.hnsep import HnsepModel
        except Exception as e:
            logger.warning(f"无法导入 HnsepModel: {e}")
            return

        hnsep_path = config.model.hnsep_model_path
        if not os.path.exists(hnsep_path):
            logger.warning(f"hnsep 模型文件不存在: {hnsep_path}")
            logger.warning("hnsep 参数处理将使用简化版本")
            return

        import yaml
        config_path = config.model.hnsep_config_path
        hnsep_config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                hnsep_config = yaml.safe_load(f) or {}

        self.hnsep_model = HnsepModel(
            model_path=hnsep_path,
            n_fft=hnsep_config.get("n_fft", 2048),
            hop_length=hnsep_config.get("hop_length", 512),
            n_out=hnsep_config.get("n_out", 32),
            n_out_lstm=hnsep_config.get("n_out_lstm", 128),
            is_mono=hnsep_config.get("is_mono", True),
            fixed_length=hnsep_config.get("fixed_length", True),
            device=config.model.device,
            sr=hnsep_config.get("sr", 44100),
        )
        logger.info(f"hnsep 模型加载完成: {hnsep_path}")

    def vocoder_synthesize(self, mel: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """使用 vocoder 合成波形（必需）"""
        config = get_config()

        if config.model.model_type == "onnx" and self.ort_session is not None:
            mel_input = mel.astype(np.float32)
            if mel_input.ndim == 2:
                mel_input = mel_input[np.newaxis, :]
            mel_input = np.transpose(mel_input, (0, 2, 1))
            f0_input = f0.astype(np.float32)
            if f0_input.ndim == 1:
                f0_input = f0_input[np.newaxis, :]

            outputs = self.ort_session.run(
                ["waveform"],
                {"mel": mel_input, "f0": f0_input},
            )
            return outputs[0].squeeze()

        elif self.vocoder is not None:
            return self.vocoder.spec2wav_torch(mel, f0)

        else:
            raise RuntimeError(
                "Vocoder 未初始化！请确保已下载并正确配置模型。\n"
                "查看 MODELS.md 获取模型下载清单。"
            )

    def hnsep_separate(self, wave: np.ndarray) -> tuple:
        """使用 hnsep 分离谐波分量"""
        if self.hnsep_model is None:
            raise RuntimeError("hnsep 模型未初始化")

        harmonic = self.hnsep_model.predict_fromaudio(wave)
        noise = wave - harmonic
        return harmonic, noise, None


_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """获取全局模型管理器"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
        _model_manager.initialize()
    return _model_manager