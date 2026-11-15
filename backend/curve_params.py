"""曲线参数系统（逐帧控制）

管理所有渲染参数的逐帧曲线，支持帧级插值、平滑滤波和自定义参数透传。

主要功能：
- 关键帧插值（线性/三次样条）
- 曲线平滑滤波
- 参数值范围验证和裁剪
- 动态参数数量支持
- 默认值机制
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


class CurveParams:
    """曲线参数管理器

    管理所有渲染参数的逐帧曲线，支持关键帧插值、平滑滤波等功能。

    支持的默认参数：
        - pitch_shift: 逐帧音高偏移（cent），默认 0.0
        - gender: 性别参数（0-100），默认 50.0
        - breathiness: 气声强度（0-100），默认 50.0
        - tension: 张力（0-100），默认 50.0
        - voicing: 浊音度（0-100），默认 100.0
        - growl: 咆哮强度（0-100），默认 0.0
        - lowcut: 动态低切（0-100），默认 0.0
        - brel: 低频气声（0-100），默认 0.0
        - breh: 高频气声（0-100），默认 0.0
        - warm: 温暖度（0-100），默认 0.0
        - hcmp: 谐波压缩（0-100），默认 0.0
        - noise_gate_threshold: 噪声门阈值，默认 0.0
        - noise_suppression: 噪声抑制强度（0-100），默认 0.0
    """

    DEFAULT_PARAMS: Dict[str, Dict] = {
        'pitch_shift': {'default': 0.0, 'min': -1200.0, 'max': 1200.0},
        'gender': {'default': 50.0, 'min': 0.0, 'max': 100.0},
        'breathiness': {'default': 50.0, 'min': 0.0, 'max': 100.0},
        'tension': {'default': 50.0, 'min': 0.0, 'max': 100.0},
        'voicing': {'default': 100.0, 'min': 0.0, 'max': 100.0},
        'growl': {'default': 0.0, 'min': 0.0, 'max': 100.0},
        'lowcut': {'default': 0.0, 'min': 0.0, 'max': 100.0},
        'brel': {'default': 0.0, 'min': 0.0, 'max': 100.0},
        'breh': {'default': 0.0, 'min': 0.0, 'max': 100.0},
        'warm': {'default': 0.0, 'min': 0.0, 'max': 100.0},
        'hcmp': {'default': 0.0, 'min': 0.0, 'max': 100.0},
        'noise_gate_threshold': {'default': 0.0, 'min': 0.0, 'max': 1.0},
        'noise_suppression': {'default': 0.0, 'min': 0.0, 'max': 100.0},
    }

    def __init__(self):
        """初始化曲线参数管理器"""
        self._param_curves: Dict[str, np.ndarray] = {}
        self._param_keyframes: Dict[str, List[Tuple[int, float]]] = {}
        self._param_interp_methods: Dict[str, str] = {}
        self._custom_params: Dict[str, Dict] = {}
        self._cache: Dict[str, np.ndarray] = {}

    def register_param(self, param_name: str, default: float = 0.0,
                       min_val: float = -float('inf'),
                       max_val: float = float('inf')) -> None:
        """注册自定义参数

        Args:
            param_name: 参数名称
            default: 默认值
            min_val: 最小值
            max_val: 最大值
        """
        self._custom_params[param_name] = {
            'default': default,
            'min': min_val,
            'max': max_val,
        }

    def _get_param_info(self, param_name: str) -> Dict:
        """获取参数信息（默认参数或自定义参数）

        Args:
            param_name: 参数名称

        Returns:
            参数字典，包含 default、min、max

        Raises:
            KeyError: 参数不存在
        """
        if param_name in self.DEFAULT_PARAMS:
            return self.DEFAULT_PARAMS[param_name]
        if param_name in self._custom_params:
            return self._custom_params[param_name]
        raise KeyError(f"未知参数: {param_name}")

    def _validate_param_value(self, param_name: str, value: float) -> float:
        """验证并裁剪参数值

        Args:
            param_name: 参数名称
            value: 参数值

        Returns:
            裁剪后的参数值
        """
        info = self._get_param_info(param_name)
        return float(np.clip(value, info['min'], info['max']))

    def set_param(self, param_name: str, value: float,
                  frame_idx: Optional[int] = None) -> None:
        """设置单帧参数值

        Args:
            param_name: 参数名称
            value: 参数值
            frame_idx: 帧索引，为 None 时设置第 0 帧
        """
        value = self._validate_param_value(param_name, value)

        if param_name not in self._param_curves:
            self._param_curves[param_name] = np.array([], dtype=np.float32)

        curve = self._param_curves[param_name]
        if frame_idx is None:
            frame_idx = 0

        if frame_idx >= len(curve):
            new_curve = np.full(frame_idx + 1,
                                self._get_param_info(param_name)['default'],
                                dtype=np.float32)
            new_curve[:len(curve)] = curve
            curve = new_curve
            self._param_curves[param_name] = curve

        curve[frame_idx] = value
        self._cache.pop(param_name, None)

    def set_param_curve(self, param_name: str,
                        keyframes: List[Tuple[int, float]],
                        interpolation: str = 'linear') -> None:
        """设置关键帧曲线

        Args:
            param_name: 参数名称
            keyframes: 关键帧列表，格式为 [(frame_idx, value), ...]
            interpolation: 插值方式，'linear' 或 'cubic'

        Raises:
            ValueError: 插值方式不支持
        """
        if interpolation not in ('linear', 'cubic'):
            raise ValueError(f"不支持的插值方式: {interpolation}，"
                             f"支持 'linear' 和 'cubic'")

        validated_keyframes = []
        for frame_idx, value in keyframes:
            validated_value = self._validate_param_value(param_name, value)
            validated_keyframes.append((int(frame_idx), validated_value))

        validated_keyframes.sort(key=lambda x: x[0])

        self._param_keyframes[param_name] = validated_keyframes
        self._param_interp_methods[param_name] = interpolation
        self._param_curves.pop(param_name, None)
        self._cache.pop(param_name, None)

    def get_param_frame(self, param_name: str, frame_idx: int) -> float:
        """获取某帧的参数值

        Args:
            param_name: 参数名称
            frame_idx: 帧索引

        Returns:
            该帧的参数值
        """
        if frame_idx < 0:
            frame_idx = 0

        if param_name in self._param_curves:
            curve = self._param_curves[param_name]
            if frame_idx < len(curve):
                return float(curve[frame_idx])

        if param_name in self._param_keyframes:
            n_frames = max(frame_idx + 1,
                           max(kf[0] for kf in self._param_keyframes[param_name]) + 1)
            curve = self.get_param_curve(param_name, n_frames)
            if frame_idx < len(curve):
                return float(curve[frame_idx])

        return self._get_param_info(param_name)['default']

    def get_param_curve(self, param_name: str, n_frames: int) -> np.ndarray:
        """获取完整参数曲线

        Args:
            param_name: 参数名称
            n_frames: 目标帧数

        Returns:
            逐帧参数值数组，形状为 (n_frames,)
        """
        cache_key = f"{param_name}_{n_frames}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        info = self._get_param_info(param_name)
        default_value = info['default']

        if param_name in self._param_keyframes and self._param_keyframes[param_name]:
            keyframes = self._param_keyframes[param_name]
            method = self._param_interp_methods.get(param_name, 'linear')
            curve = self.interpolate_keyframes(keyframes, n_frames, method)
        elif param_name in self._param_curves and len(self._param_curves[param_name]) > 0:
            curve = self._param_curves[param_name].copy()
            if len(curve) < n_frames:
                last_val = curve[-1] if len(curve) > 0 else default_value
                curve = np.pad(curve, (0, n_frames - len(curve)),
                               constant_values=last_val)
            elif len(curve) > n_frames:
                curve = curve[:n_frames]
        else:
            curve = np.full(n_frames, default_value, dtype=np.float32)

        curve = np.clip(curve, info['min'], info['max']).astype(np.float32)

        self._cache[cache_key] = curve.copy()
        return curve

    def smooth_curve(self, curve: np.ndarray,
                     smooth_size: int = 5,
                     method: str = 'gaussian') -> np.ndarray:
        """平滑曲线

        Args:
            curve: 输入曲线
            smooth_size: 平滑窗口大小
            method: 平滑方法，'gaussian' 或 'moving_average'

        Returns:
            平滑后的曲线

        Raises:
            ValueError: 平滑方法不支持
        """
        if smooth_size <= 1 or len(curve) <= 1:
            return curve.copy()

        if method == 'gaussian':
            sigma = smooth_size / 3.0
            smoothed = gaussian_filter1d(curve.astype(np.float64), sigma)
            return smoothed.astype(np.float32)
        elif method == 'moving_average':
            kernel_size = min(smooth_size, len(curve))
            kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
            smoothed = np.convolve(curve.astype(np.float64), kernel, mode='same')
            return smoothed.astype(np.float32)
        else:
            raise ValueError(f"不支持的平滑方法: {method}，"
                             f"支持 'gaussian' 和 'moving_average'")

    def interpolate_keyframes(self,
                              keyframes: List[Tuple[int, float]],
                              n_frames: int,
                              method: str = 'linear') -> np.ndarray:
        """关键帧插值

        Args:
            keyframes: 关键帧列表，格式为 [(frame_idx, value), ...]
            n_frames: 目标帧数
            method: 插值方式，'linear' 或 'cubic'

        Returns:
            插值后的曲线，形状为 (n_frames,)

        Raises:
            ValueError: 插值方式不支持或关键帧不足
        """
        if n_frames <= 0:
            return np.array([], dtype=np.float32)

        if not keyframes:
            raise ValueError("关键帧列表不能为空")

        sorted_kfs = sorted(keyframes, key=lambda x: x[0])

        if len(sorted_kfs) == 1:
            return np.full(n_frames, sorted_kfs[0][1], dtype=np.float32)

        frame_indices = np.array([kf[0] for kf in sorted_kfs], dtype=np.float64)
        values = np.array([kf[1] for kf in sorted_kfs], dtype=np.float64)

        target_frames = np.arange(n_frames, dtype=np.float64)

        if method == 'linear':
            f = interp1d(frame_indices, values, kind='linear',
                         fill_value='extrapolate', bounds_error=False)
        elif method == 'cubic':
            if len(sorted_kfs) < 4:
                f = interp1d(frame_indices, values, kind='linear',
                             fill_value='extrapolate', bounds_error=False)
            else:
                f = interp1d(frame_indices, values, kind='cubic',
                             fill_value='extrapolate', bounds_error=False)
        else:
            raise ValueError(f"不支持的插值方式: {method}，"
                             f"支持 'linear' 和 'cubic'")

        result = f(target_frames).astype(np.float32)

        first_val = sorted_kfs[0][1]
        last_val = sorted_kfs[-1][1]
        first_frame = sorted_kfs[0][0]
        last_frame = sorted_kfs[-1][0]

        result = np.where(target_frames < first_frame, first_val, result)
        result = np.where(target_frames > last_frame, last_val, result)

        return result

    def apply_to_renderer(self, renderer, n_frames: int) -> Dict[str, np.ndarray]:
        """将参数应用到渲染器

        将所有已设置的参数曲线提取出来，准备应用到渲染流程中。

        Args:
            renderer: 渲染器对象（PhraseRenderer）
            n_frames: 总帧数

        Returns:
            参数字典，键为参数名，值为逐帧参数数组
        """
        params = {}

        all_param_names = set(self.DEFAULT_PARAMS.keys())
        all_param_names.update(self._custom_params.keys())
        all_param_names.update(self._param_curves.keys())
        all_param_names.update(self._param_keyframes.keys())

        for param_name in all_param_names:
            try:
                curve = self.get_param_curve(param_name, n_frames)
                params[param_name] = curve
            except KeyError:
                continue

        return params

    def get_all_params(self, n_frames: int) -> Dict[str, np.ndarray]:
        """获取所有参数的曲线

        Args:
            n_frames: 总帧数

        Returns:
            所有参数的曲线路径
        """
        return self.apply_to_renderer(None, n_frames)

    def clear_cache(self) -> None:
        """清除缓存"""
        self._cache.clear()

    def reset(self) -> None:
        """重置所有参数"""
        self._param_curves.clear()
        self._param_keyframes.clear()
        self._param_interp_methods.clear()
        self._custom_params.clear()
        self._cache.clear()
