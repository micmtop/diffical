"""自动质量评估系统

对合成语音进行多维度质量评估，包括：
- 频谱不连续性（音素边界处的频谱通量）
- 能量连续性（边界处的能量突变）
- F0 平滑度（基频曲线平滑度）
- 音色一致性（MFCC 余弦相似度）
- 综合质量分数（0-100）
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cosine
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QualityEvaluator:
    """语音合成质量评估器

    从多个维度评估合成语音的质量，并给出综合评分。
    """

    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048,
                 hop_length: int = 512, n_mels: int = 128,
                 n_mfcc: int = 20):
        """初始化质量评估器

        Args:
            sample_rate: 采样率，默认 44100 Hz
            n_fft: FFT 窗口大小
            hop_length: 帧移
            n_mels: Mel 频段数
            n_mfcc: MFCC 系数数量
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.win_length = n_fft
        self.window = np.hanning(self.win_length)

    def evaluate_spectral_continuity(self, mel: np.ndarray,
                                     boundaries: np.ndarray) -> Dict:
        """频谱连续性评估

        计算音素边界处的频谱通量，评估拼接处的频谱不连续性。

        Args:
            mel: Mel 频谱 (n_mels, T)，自然对数尺度
            boundaries: 边界帧索引数组 (N,)

        Returns:
            包含频谱连续性指标的字典：
            - mean_flux: 边界处平均频谱通量
            - max_flux: 边界处最大频谱通量
            - std_flux: 边界处频谱通量标准差
            - flux_values: 各边界处的频谱通量值
            - score: 频谱连续性分数 (0-100)
        """
        if mel.ndim != 2:
            raise ValueError(f"mel 应为 2 维数组，实际为 {mel.ndim} 维")

        n_mels, n_frames = mel.shape

        if len(boundaries) == 0 or n_frames < 2:
            return {
                "mean_flux": 0.0,
                "max_flux": 0.0,
                "std_flux": 0.0,
                "flux_values": np.array([]),
                "score": 100.0,
            }

        mel_linear = np.exp(mel)
        flux_full = np.sqrt(
            np.sum((mel_linear[:, 1:] - mel_linear[:, :-1]) ** 2, axis=0)
        )

        boundary_fluxes = []
        for b in boundaries:
            b = int(np.clip(b, 0, n_frames - 2))
            if b < len(flux_full):
                boundary_fluxes.append(flux_full[b])

        if len(boundary_fluxes) == 0:
            return {
                "mean_flux": 0.0,
                "max_flux": 0.0,
                "std_flux": 0.0,
                "flux_values": np.array([]),
                "score": 100.0,
            }

        boundary_fluxes = np.array(boundary_fluxes)

        mean_flux = float(np.mean(boundary_fluxes))
        max_flux = float(np.max(boundary_fluxes))
        std_flux = float(np.std(boundary_fluxes))

        reference_flux = np.median(flux_full) if len(flux_full) > 0 else 1.0
        if reference_flux < 1e-6:
            reference_flux = 1.0

        flux_ratio = mean_flux / reference_flux
        score = float(np.clip(100.0 * np.exp(-0.5 * (flux_ratio - 1.0)), 0.0, 100.0))

        return {
            "mean_flux": mean_flux,
            "max_flux": max_flux,
            "std_flux": std_flux,
            "flux_values": boundary_fluxes,
            "score": score,
        }

    def evaluate_energy_continuity(self, audio: np.ndarray, sr: int,
                                   boundaries: np.ndarray) -> Dict:
        """能量连续性评估

        计算音素边界处的能量变化，评估拼接处的能量突变程度。

        Args:
            audio: 音频波形 (samples,)
            sr: 采样率
            boundaries: 边界帧索引数组 (N,)（基于 hop_length 的帧）

        Returns:
            包含能量连续性指标的字典：
            - mean_energy_delta: 边界处平均能量变化（dB）
            - max_energy_delta: 边界处最大能量变化（dB）
            - energy_values: 各边界处的能量变化值
            - score: 能量连续性分数 (0-100)
        """
        if len(audio) == 0 or len(boundaries) == 0:
            return {
                "mean_energy_delta": 0.0,
                "max_energy_delta": 0.0,
                "energy_values": np.array([]),
                "score": 100.0,
            }

        hop_length = self.hop_length
        win_length = self.win_length

        n_frames = 1 + (len(audio) - win_length) // hop_length
        if n_frames < 2:
            return {
                "mean_energy_delta": 0.0,
                "max_energy_delta": 0.0,
                "energy_values": np.array([]),
                "score": 100.0,
            }

        energy = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            start = i * hop_length
            end = start + win_length
            frame = audio[start:end] * self.window
            energy[i] = np.sum(frame ** 2) / win_length

        energy_db = 10.0 * np.log10(np.maximum(energy, 1e-10))
        energy_delta = np.abs(np.diff(energy_db))

        boundary_deltas = []
        for b in boundaries:
            b = int(np.clip(b, 0, n_frames - 2))
            if b < len(energy_delta):
                boundary_deltas.append(energy_delta[b])

        if len(boundary_deltas) == 0:
            return {
                "mean_energy_delta": 0.0,
                "max_energy_delta": 0.0,
                "energy_values": np.array([]),
                "score": 100.0,
            }

        boundary_deltas = np.array(boundary_deltas)

        mean_delta = float(np.mean(boundary_deltas))
        max_delta = float(np.max(boundary_deltas))

        score = float(np.clip(100.0 - mean_delta * 5.0, 0.0, 100.0))

        return {
            "mean_energy_delta": mean_delta,
            "max_energy_delta": max_delta,
            "energy_values": boundary_deltas,
            "score": score,
        }

    def evaluate_f0_smoothness(self, f0: np.ndarray, sr: int) -> Dict:
        """F0 平滑度评估

        评估基频曲线的平滑程度，通过计算 F0 差分的标准差来衡量。

        Args:
            f0: F0 曲线 (T,)，单位 Hz，非 voiced 段为 0
            sr: 采样率

        Returns:
            包含 F0 平滑度指标的字典：
            - mean_delta: 平均 F0 变化量（Hz/帧）
            - std_delta: F0 变化量的标准差
            - max_delta: 最大单帧 F0 变化量
            - jitter: 微抖动（相对变化率）
            - score: F0 平滑度分数 (0-100)
        """
        if len(f0) < 3:
            return {
                "mean_delta": 0.0,
                "std_delta": 0.0,
                "max_delta": 0.0,
                "jitter": 0.0,
                "score": 100.0,
            }

        voiced_mask = f0 > 0
        if np.sum(voiced_mask) < 3:
            return {
                "mean_delta": 0.0,
                "std_delta": 0.0,
                "max_delta": 0.0,
                "jitter": 0.0,
                "score": 50.0,
            }

        f0_voiced = f0[voiced_mask]
        f0_smooth = gaussian_filter1d(f0_voiced.astype(float), sigma=1.0)
        f0_residual = f0_voiced - f0_smooth
        f0_diff = np.diff(f0_voiced)

        mean_delta = float(np.mean(np.abs(f0_diff)))
        std_delta = float(np.std(f0_diff))
        max_delta = float(np.max(np.abs(f0_diff)))

        mean_f0 = float(np.mean(f0_voiced))
        if mean_f0 > 0:
            jitter = float(np.std(f0_residual) / mean_f0 * 100.0)
        else:
            jitter = 0.0

        frame_ms = self.hop_length / self.sample_rate * 1000
        max_delta_per_sec = max_delta / frame_ms * 1000

        if max_delta_per_sec > 0:
            score = float(np.clip(100.0 * np.exp(-max_delta_per_sec / 500.0), 0.0, 100.0))
        else:
            score = 100.0

        jitter_penalty = min(jitter * 5.0, 30.0)
        score = float(np.clip(score - jitter_penalty, 0.0, 100.0))

        return {
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "max_delta": max_delta,
            "jitter": jitter,
            "score": score,
        }

    def evaluate_timbre_consistency(self, audio_ref: np.ndarray,
                                    audio_test: np.ndarray,
                                    sr: int) -> Dict:
        """音色一致性评估

        通过 MFCC 特征的余弦相似度来评估两段音频的音色一致性。

        Args:
            audio_ref: 参考音频 (samples,)
            audio_test: 测试音频 (samples,)
            sr: 采样率

        Returns:
            包含音色一致性指标的字典：
            - mean_cosine_similarity: 平均余弦相似度
            - mfcc_distortion: MFCC 失真度
            - score: 音色一致性分数 (0-100)
        """
        if len(audio_ref) == 0 or len(audio_test) == 0:
            return {
                "mean_cosine_similarity": 0.0,
                "mfcc_distortion": 0.0,
                "score": 0.0,
            }

        try:
            import librosa
            mfcc_ref = librosa.feature.mfcc(
                y=audio_ref, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.win_length
            )
            mfcc_test = librosa.feature.mfcc(
                y=audio_test, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.win_length
            )
        except ImportError:
            mfcc_ref = self._compute_mfcc_manual(audio_ref, sr)
            mfcc_test = self._compute_mfcc_manual(audio_test, sr)

        mfcc_ref = mfcc_ref[1:, :]
        mfcc_test = mfcc_test[1:, :]

        n_frames_ref = mfcc_ref.shape[1]
        n_frames_test = mfcc_test.shape[1]
        n_frames = min(n_frames_ref, n_frames_test)

        if n_frames < 1:
            return {
                "mean_cosine_similarity": 0.0,
                "mfcc_distortion": 0.0,
                "score": 0.0,
            }

        mfcc_ref_trim = mfcc_ref[:, :n_frames]
        mfcc_test_trim = mfcc_test[:, :n_frames]

        cosine_sims = []
        for i in range(n_frames):
            ref_vec = mfcc_ref_trim[:, i]
            test_vec = mfcc_test_trim[:, i]
            ref_norm = np.linalg.norm(ref_vec)
            test_norm = np.linalg.norm(test_vec)
            if ref_norm > 1e-10 and test_norm > 1e-10:
                sim = 1.0 - cosine(ref_vec, test_vec)
                cosine_sims.append(sim)

        if len(cosine_sims) == 0:
            return {
                "mean_cosine_similarity": 0.0,
                "mfcc_distortion": 0.0,
                "score": 0.0,
            }

        mean_sim = float(np.mean(cosine_sims))

        mfcc_diff = mfcc_ref_trim - mfcc_test_trim
        mfcc_distortion = float(np.sqrt(np.mean(mfcc_diff ** 2)))

        score = float(np.clip(mean_sim * 100.0, 0.0, 100.0))

        return {
            "mean_cosine_similarity": mean_sim,
            "mfcc_distortion": mfcc_distortion,
            "score": score,
        }

    def _compute_mfcc_manual(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """手动计算 MFCC（当 librosa 不可用时）

        Args:
            audio: 音频波形
            sr: 采样率

        Returns:
            MFCC 系数 (n_mfcc, T)
        """
        from scipy.fft import dct

        f, t, Zxx = signal.stft(
            audio, fs=sr, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)

        mel_filterbank = self._create_mel_filterbank(sr)
        mel_spec = np.dot(mel_filterbank, mag)
        log_mel = np.log(np.maximum(mel_spec, 1e-10))

        mfcc = dct(log_mel, type=2, axis=0, norm='ortho')[:self.n_mfcc, :]

        return mfcc

    def _create_mel_filterbank(self, sr: int) -> np.ndarray:
        """创建 Mel 滤波器组

        Args:
            sr: 采样率

        Returns:
            Mel 滤波器矩阵 (n_mels, n_fft//2 + 1)
        """
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        n_freq_bins = self.n_fft // 2 + 1
        mel_min = hz_to_mel(0)
        mel_max = hz_to_mel(sr / 2)

        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        bin_points = np.floor((self.n_fft + 1) * hz_points / sr).astype(int)
        bin_points = np.clip(bin_points, 0, n_freq_bins - 1)

        filterbank = np.zeros((self.n_mels, n_freq_bins))

        for m in range(1, self.n_mels + 1):
            left = bin_points[m - 1]
            center = bin_points[m]
            right = bin_points[m + 1]

            for k in range(left, center):
                if center > left:
                    filterbank[m - 1, k] = (k - left) / (center - left)
            for k in range(center, right):
                if right > center:
                    filterbank[m - 1, k] = (right - k) / (right - center)

        return filterbank

    def detect_boundaries(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """自动检测音素边界

        基于能量变化和频谱变化检测音素边界。

        Args:
            audio: 音频波形 (samples,)
            sr: 采样率

        Returns:
            边界帧索引数组
        """
        if len(audio) < self.win_length:
            return np.array([], dtype=int)

        f, t, Zxx = signal.stft(
            audio, fs=sr, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)

        energy = np.sqrt(np.mean(mag ** 2, axis=0))
        energy_diff = np.abs(np.diff(energy))

        spectral_flux = np.sqrt(
            np.sum((mag[:, 1:] - mag[:, :-1]) ** 2, axis=0)
        )

        combined = energy_diff * spectral_flux

        if len(combined) == 0:
            return np.array([], dtype=int)

        threshold = np.median(combined) + 0.5 * np.std(combined)

        peaks, _ = signal.find_peaks(
            combined,
            height=threshold,
            distance=max(3, int(50 * sr / 1000 / self.hop_length))
        )

        return peaks.astype(int)

    def detect_boundaries_from_mel(self, mel: np.ndarray) -> np.ndarray:
        """从 Mel 频谱检测边界

        基于频谱通量检测音素边界。

        Args:
            mel: Mel 频谱 (n_mels, T)，自然对数尺度

        Returns:
            边界帧索引数组
        """
        if mel.ndim != 2 or mel.shape[1] < 3:
            return np.array([], dtype=int)

        mel_linear = np.exp(mel)
        flux = np.sqrt(
            np.sum((mel_linear[:, 1:] - mel_linear[:, :-1]) ** 2, axis=0)
        )

        if len(flux) == 0:
            return np.array([], dtype=int)

        energy = np.sqrt(np.mean(mel_linear ** 2, axis=0))
        energy_diff = np.abs(np.diff(energy))

        flux_norm = flux / (np.median(flux) + 1e-10)
        energy_norm = energy_diff / (np.median(energy_diff) + 1e-10)

        combined = flux_norm * energy_norm

        threshold = np.median(combined) + 0.5 * np.std(combined)

        min_distance = max(3, int(50 * self.sample_rate / 1000 / self.hop_length))

        peaks, _ = signal.find_peaks(
            combined,
            height=threshold,
            distance=min_distance
        )

        return peaks.astype(int)

    def evaluate_all(self, audio: np.ndarray, sr: int,
                     mel: np.ndarray, f0: np.ndarray,
                     boundaries: Optional[np.ndarray] = None,
                     ref_audio: Optional[np.ndarray] = None) -> Dict:
        """综合质量评估

        对所有维度进行评估，返回完整的质量报告。

        Args:
            audio: 测试音频波形 (samples,)
            sr: 采样率
            mel: Mel 频谱 (n_mels, T)
            f0: F0 曲线 (T,)
            boundaries: 边界帧索引，为 None 时自动检测
            ref_audio: 参考音频，用于音色一致性评估

        Returns:
            包含所有评估指标的字典
        """
        if boundaries is None:
            if mel is not None and mel.ndim == 2:
                boundaries = self.detect_boundaries_from_mel(mel)
            else:
                boundaries = self.detect_boundaries(audio, sr)

        results = {
            "spectral_continuity": self.evaluate_spectral_continuity(
                mel, boundaries
            ),
            "energy_continuity": self.evaluate_energy_continuity(
                audio, sr, boundaries
            ),
            "f0_smoothness": self.evaluate_f0_smoothness(f0, sr),
            "boundaries": boundaries,
            "n_boundaries": len(boundaries),
        }

        if ref_audio is not None:
            results["timbre_consistency"] = self.evaluate_timbre_consistency(
                ref_audio, audio, sr
            )
        else:
            results["timbre_consistency"] = {
                "mean_cosine_similarity": None,
                "mfcc_distortion": None,
                "score": None,
            }

        results["overall_score"] = self.get_quality_score(results)

        return results

    def get_quality_score(self, metrics: Dict) -> float:
        """计算综合质量分数（0-100）

        各维度权重：
        - 频谱连续性: 30%
        - 能量连续性: 20%
        - F0 平滑度: 25%
        - 音色一致性: 25%（如果有参考音频）

        Args:
            metrics: 评估指标字典（来自 evaluate_all）

        Returns:
            综合质量分数（0-100）
        """
        weights = {
            "spectral_continuity": 0.30,
            "energy_continuity": 0.20,
            "f0_smoothness": 0.25,
            "timbre_consistency": 0.25,
        }

        total_score = 0.0
        total_weight = 0.0

        for key, weight in weights.items():
            if key in metrics and metrics[key] is not None:
                score = metrics[key].get("score")
                if score is not None:
                    total_score += score * weight
                    total_weight += weight

        if total_weight > 0:
            return float(round(total_score / total_weight, 2))

        return 0.0

    def generate_report(self, metrics: Dict) -> str:
        """生成可视化质量报告文本

        Args:
            metrics: 评估指标字典

        Returns:
            格式化的质量报告字符串
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  语音合成质量评估报告")
        lines.append("=" * 60)
        lines.append("")

        overall = metrics.get("overall_score", 0.0)
        lines.append(f"  综合质量分数: {overall:.1f} / 100")
        lines.append("")

        if overall >= 90:
            grade = "优秀"
        elif overall >= 80:
            grade = "良好"
        elif overall >= 70:
            grade = "中等"
        elif overall >= 60:
            grade = "及格"
        else:
            grade = "较差"
        lines.append(f"  质量等级: {grade}")
        lines.append("")

        n_boundaries = metrics.get("n_boundaries", 0)
        lines.append(f"  检测到边界数: {n_boundaries}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("  各维度详细评估")
        lines.append("-" * 60)
        lines.append("")

        spec = metrics.get("spectral_continuity", {})
        if spec:
            score = spec.get("score", 0)
            lines.append(f"  [频谱连续性]  分数: {score:.1f}/100")
            lines.append(f"    平均边界频谱通量: {spec.get('mean_flux', 0):.4f}")
            lines.append(f"    最大边界频谱通量: {spec.get('max_flux', 0):.4f}")
            lines.append(f"    频谱通量标准差:   {spec.get('std_flux', 0):.4f}")
            lines.append("")

        energy = metrics.get("energy_continuity", {})
        if energy:
            score = energy.get("score", 0)
            lines.append(f"  [能量连续性]  分数: {score:.1f}/100")
            lines.append(f"    平均能量变化: {energy.get('mean_energy_delta', 0):.2f} dB")
            lines.append(f"    最大能量变化: {energy.get('max_energy_delta', 0):.2f} dB")
            lines.append("")

        f0_metrics = metrics.get("f0_smoothness", {})
        if f0_metrics:
            score = f0_metrics.get("score", 0)
            lines.append(f"  [F0 平滑度]   分数: {score:.1f}/100")
            lines.append(f"    平均F0变化量:  {f0_metrics.get('mean_delta', 0):.2f} Hz/帧")
            lines.append(f"    F0变化标准差:  {f0_metrics.get('std_delta', 0):.2f} Hz/帧")
            lines.append(f"    最大单帧变化:  {f0_metrics.get('max_delta', 0):.2f} Hz")
            lines.append(f"    微抖动 (jitter): {f0_metrics.get('jitter', 0):.3f} %")
            lines.append("")

        timbre = metrics.get("timbre_consistency", {})
        if timbre and timbre.get("score") is not None:
            score = timbre.get("score", 0)
            lines.append(f"  [音色一致性]  分数: {score:.1f}/100")
            lines.append(f"    平均余弦相似度: {timbre.get('mean_cosine_similarity', 0):.4f}")
            lines.append(f"    MFCC 失真度:    {timbre.get('mfcc_distortion', 0):.4f}")
            lines.append("")

        lines.append("-" * 60)
        lines.append("")

        suggestions = self._generate_suggestions(metrics)
        if suggestions:
            lines.append("  改进建议:")
            for i, s in enumerate(suggestions, 1):
                lines.append(f"    {i}. {s}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def _generate_suggestions(self, metrics: Dict) -> List[str]:
        """根据评估结果生成改进建议

        Args:
            metrics: 评估指标字典

        Returns:
            改进建议列表
        """
        suggestions = []

        spec = metrics.get("spectral_continuity", {})
        if spec.get("score", 100) < 70:
            suggestions.append(
                "频谱连续性较差，建议增加音素拼接处的交叉淡入淡出长度，"
                "或使用频率依赖的 crossfade 算法"
            )

        energy = metrics.get("energy_continuity", {})
        if energy.get("score", 100) < 70:
            suggestions.append(
                "能量连续性较差，建议在拼接处进行能量包络匹配和平滑过渡，"
                "或使用 gain envelope 渐变"
            )

        f0_metrics = metrics.get("f0_smoothness", {})
        if f0_metrics.get("score", 100) < 70:
            if f0_metrics.get("jitter", 0) > 2:
                suggestions.append(
                    "F0 抖动过大，建议对 F0 曲线进行适度平滑处理，"
                    "或使用更精确的 F0 提取算法"
                )
            else:
                suggestions.append(
                    "F0 曲线平滑度不足，建议在音素边界处进行 F0 渐变过渡，"
                    "或使用 F0 contour smoothing"
                )

        timbre = metrics.get("timbre_consistency", {})
        if timbre.get("score") is not None and timbre["score"] < 70:
            suggestions.append(
                "音色一致性较差，建议使用频谱包络匹配（如 formant shifting），"
                "或使用音色迁移技术统一音色"
            )

        return suggestions
