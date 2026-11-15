"""hnsep CascadedNet 封装

复用自参考实现 (util/hnsep.py + hnsep/nets.py)。

将音频分离为谐波分量（voicing）和噪声分量。
用于：
- 音色融合的三分量独立处理
- 呼吸音自然过渡
- AI 边界增强
"""

import os
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class HnsepModel:
    """hnsep 谐波/噪声分离模型

    支持 ONNX 和 PyTorch 两种后端。
    """

    def __init__(
        self,
        model_path: str,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_out: int = 32,
        n_out_lstm: int = 128,
        is_mono: bool = True,
        fixed_length: Optional[int] = None,
        device: str = "cpu",
        sr: int = 44100,
    ):
        self.model_path = model_path
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_out = n_out
        self.n_out_lstm = n_out_lstm
        self.is_mono = is_mono
        self.fixed_length = fixed_length if fixed_length is not None else True
        self.device = device

        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".onnx":
            self._backend = "onnx"
            self._init_onnx()
        elif ext in (".pt", ".pth", ".ckpt"):
            self._backend = "torch"
            self._init_torch()
        else:
            raise ValueError(f"不支持的模型文件格式: {ext}")

    def _init_onnx(self):
        """初始化 ONNX 后端"""
        import onnxruntime as ort

        providers = []
        if self.device == "cuda":
            providers.append("CUDAExecutionProvider")
        if os.name == "nt":
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=providers,
        )
        logger.info(f"hnsep (ONNX) 加载完成: {self.model_path}")

    def _init_torch(self):
        """初始化 PyTorch 后端"""
        import torch
        from .hnsep_nets import CascadedNet

        self.torch_model = CascadedNet(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            nout=self.n_out,
            nout_lstm=self.n_out_lstm,
            is_complex=True,
            is_mono=self.is_mono,
            fixed_length=self.fixed_length,
        )
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if "model" in checkpoint:
            self.torch_model.load_state_dict(checkpoint["model"])
        else:
            self.torch_model.load_state_dict(checkpoint)
        self.torch_model.eval()
        self.torch_model.to(self.device)
        logger.info(f"hnsep (PyTorch) 加载完成: {self.model_path}")

    def predict_fromaudio(self, x: np.ndarray) -> np.ndarray:
        """从音频预测谐波分量

        Args:
            x: 输入波形 (samples,)

        Returns:
            harmonic: 谐波分量 (samples,)
        """
        import torch

        if isinstance(x, np.ndarray):
            x_torch = torch.from_numpy(x).float().to(self.device)
        else:
            x_torch = x

        if x_torch.dim() == 1:
            x_torch = x_torch.unsqueeze(0).unsqueeze(0)
        elif x_torch.dim() == 2:
            x_torch = x_torch.unsqueeze(0)

        with torch.no_grad():
            harmonic = self.torch_model.predict_fromaudio(x_torch)

        return harmonic.squeeze().cpu().numpy()

    def _forward_onnx(self, spec: "torch.Tensor") -> "torch.Tensor":
        """ONNX 前向传播"""
        import torch

        spec_real = spec.real.unsqueeze(0)
        spec_imag = spec.imag.unsqueeze(0)
        spec_input = torch.cat([spec_real, spec_imag], dim=1)

        mask = self.session.run(
            ["mask"],
            {"input": spec_input.cpu().numpy()},
        )[0]

        mask = torch.from_numpy(mask).to(spec.device)
        if mask.shape[1] == 2:
            mask_complex = torch.complex(mask[:, 0], mask[:, 1])
        else:
            mask_complex = mask.squeeze(0)

        return mask_complex

    def _forward_torch(self, spec: "torch.Tensor") -> "torch.Tensor":
        """PyTorch 前向传播"""
        with torch.no_grad():
            mask = self.torch_model.forward(spec.unsqueeze(0))
        return mask.squeeze(0)

    def separate(self, x: np.ndarray) -> tuple:
        """分离谐波和噪声

        Args:
            x: 输入波形 (samples,)

        Returns:
            harmonic: 谐波分量
            noise: 噪声分量
        """
        harmonic = self.predict_fromaudio(x)
        noise = x - harmonic
        return harmonic, noise
