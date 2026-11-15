"""真人化细节增强（八维）

对合成音频添加精细化的自然感增强：
1. 谐波细节 - 添加谐波精细结构
2. 噪声调制 - 自然噪声调制
3. 微动态 - 微小的能量波动
4. 瞬态增强 - 辅音瞬态增强
5. 空间感 - 轻微的空间效果
6. 声带振动模拟 - 声带振动的细微不规则性
7. 共振峰精细调整 - LPC 共振峰校正
8. 频谱细节恢复 - 高频细节补偿
9. 声门脉冲模拟 - 更真实的声带振动模型
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import toeplitz
import logging

logger = logging.getLogger(__name__)


class DetailEnhancer:
    """八维细节增强器"""

    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048,
                 hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft

    def enhance(self, audio: np.ndarray, level: float = 0.5,
                mode: str = 'balanced') -> np.ndarray:
        """应用细节增强

        Args:
            audio: 输入音频
            level: 增强强度 0~1
            mode: 增强模式 ('balanced', 'clarity', 'fluency')
                  - balanced: 平衡模式（默认）
                  - clarity: 清晰度优先（适合单独音）
                  - fluency: 流畅度优先（适合连续音）

        Returns:
            增强后的音频
        """
        if level <= 0:
            return audio

        result = audio.copy()

        if mode == 'clarity':
            result = self._formant_sharpen(result, strength=level * 0.008)
            result = self._spectral_detail_restore(result, strength=level * 0.008)
            result = self._harmonic_detail(result, strength=level * 0.01)
            result = self._micro_dynamics(result, strength=level * 0.01)
            result = self._vocal_fold_vibration(result, strength=level * 0.005)
            result = self._spectral_envelope_smoothing(result, strength=level * 0.08)
        elif mode == 'fluency':
            result = self._glottal_pulse_simulation(result, strength=level * 0.03)
            result = self._harmonic_detail(result, strength=level * 0.02)
            result = self._micro_dynamics(result, strength=level * 0.02)
            result = self._vocal_fold_vibration(result, strength=level * 0.01)
            result = self._breath_simulation(result, strength=level * 0.02)
            result = self._formant_drift(result, strength=level * 0.08)
            result = self._spectral_envelope_smoothing(result, strength=level * 0.12)
        else:
            result = self._formant_sharpen(result, strength=level * 0.015)
            result = self._spectral_detail_restore(result, strength=level * 0.015)
            result = self._harmonic_detail(result, strength=level * 0.02)
            result = self._micro_dynamics(result, strength=level * 0.02)
            result = self._vocal_fold_vibration(result, strength=level * 0.01)
            result = self._breath_simulation(result, strength=level * 0.02)
            result = self._formant_drift(result, strength=level * 0.06)
            result = self._spectral_envelope_smoothing(result, strength=level * 0.12)

        return result

    def _harmonic_detail(self, audio: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """谐波细节增强

        为谐波添加细微的非周期性变化，使其更自然。
        使用确定性低频调制代替随机噪声，避免电音感。
        """
        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        n_frames = mag.shape[1]
        n_freqs = mag.shape[0]
        t_mod = np.arange(n_frames) / (n_frames + 1e-6)

        phase_jitter = np.zeros_like(mag)
        for freq_hz in [3, 5, 7]:
            phase_offset = np.sin(2 * np.pi * freq_hz * t_mod) * 0.008 * strength
            phase_jitter += phase_offset.reshape(1, -1)

        mag_out = mag

        _, result = signal.istft(
            mag_out * np.exp(1j * (phase + phase_jitter)),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        return result[:len(audio)]

    def _noise_modulation(self, audio: np.ndarray, strength: float = 0.2) -> np.ndarray:
        """噪声调制

        添加有调制感的自然噪声，模拟气息变化。
        仅在有声区域添加，静音区保持干净。
        """
        hop = 512
        n_frames = len(audio) // hop
        if n_frames < 2:
            return audio

        energy = np.array([
            np.sqrt(np.mean(audio[i * hop:(i + 1) * hop] ** 2))
            for i in range(n_frames)
        ])
        max_energy = np.max(energy)
        if max_energy < 1e-6:
            return audio

        noise_floor = max_energy * 0.02
        voiced_mask = energy > noise_floor

        if not np.any(voiced_mask):
            return audio

        energy_smooth = gaussian_filter1d(energy, sigma=3)
        energy_norm = energy_smooth / (max_energy + 1e-6)

        from scipy.interpolate import interp1d
        t_orig = np.arange(len(energy_norm)) * hop + hop / 2
        t_new = np.arange(len(audio))
        env = interp1d(t_orig, energy_norm, kind='linear',
                       fill_value=0.0, bounds_error=False)(t_new)
        env = np.clip(env, 0.0, 1.0)

        noise = np.random.randn(len(audio)) * 0.0003 * strength
        noise = noise * env

        return audio + noise

    def _micro_dynamics(self, audio: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """微动态

        添加细微的能量波动，模拟自然演唱的不稳定感。
        """
        # 生成慢速波动
        n_samples = len(audio)
        t = np.arange(n_samples) / self.sample_rate

        # 多频率叠加的缓慢波动
        fluct = np.zeros_like(audio, dtype=np.float64)
        for freq in [4, 6.5, 9, 12]:  # Hz
            phase = np.random.rand() * 2 * np.pi
            fluct += np.sin(2 * np.pi * freq * t + phase) * (0.01 * strength)

        # 应用波动
        return audio * (1 + fluct)

    def _transient_enhance(self, audio: np.ndarray, strength: float = 0.4) -> np.ndarray:
        """瞬态增强

        增强辅音等瞬态信号的清晰度。
        """
        # 计算瞬态能量
        audio_diff = np.diff(audio, prepend=0)
        transient = np.abs(audio_diff)

        # 平滑瞬态包络
        transient_smooth = gaussian_filter1d(transient, sigma=2)

        # 检测瞬态区域
        threshold = np.mean(transient_smooth) * 2
        transient_mask = transient_smooth > threshold

        # 对瞬态区域做增强
        gain = 1 + transient_mask * strength
        # 平滑增益曲线
        gain = gaussian_filter1d(gain.astype(np.float64), sigma=3)

        return audio * gain

    def _spatial_effect(self, audio: np.ndarray, strength: float = 0.2) -> np.ndarray:
        """空间感

        添加轻微的反射/混响感，增加空间维度。
        简化版本：添加一个短延迟。
        """
        delay_ms = 15  # 15ms 早期反射
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        decay = 0.15 * strength

        if delay_samples >= len(audio):
            return audio

        # 添加早期反射
        early = np.zeros_like(audio)
        early[delay_samples:] = audio[:-delay_samples] * decay

        # 第二个反射
        delay2 = int(delay_samples * 1.6)
        if delay2 < len(audio):
            early2 = np.zeros_like(audio)
            early2[delay2:] = audio[:-delay2] * decay * 0.6
            early = early + early2

        return audio + early

    def _vocal_fold_vibration(self, audio: np.ndarray, strength: float = 0.15) -> np.ndarray:
        """声带振动模拟

        添加极细微的音高抖动（jitter）和振幅抖动（shimmer），
        模拟真实声带振动的不完全规则性。
        使用确定性调制避免随机噪声导致的电音感。

        关键改进：大幅降低高频频段的调制强度，避免引入高频噪声。
        """
        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # 检测有声区域
        energy = np.sum(mag, axis=0)
        voiced_mask = energy > np.max(energy) * 0.01
        n_frames = mag.shape[1]

        t_mod = np.arange(n_frames) / (n_frames + 1e-6)

        # 相位抖动（确定性低频调制）- 降低强度
        jitter_phase = np.zeros_like(phase)
        for freq_hz in [4.2, 6.7, 9.3]:
            jitter_phase += np.sin(2 * np.pi * freq_hz * t_mod) * 0.006 * strength
        jitter_phase = gaussian_filter1d(jitter_phase, sigma=2, axis=1)
        jitter_phase *= voiced_mask.reshape(1, -1)

        # 振幅抖动（shimmer，确定性）- 降低强度
        shimmer = np.ones_like(mag)
        for freq_hz in [3.1, 5.4]:
            shimmer += np.sin(2 * np.pi * freq_hz * t_mod) * 0.003 * strength
        shimmer = 1 + (shimmer - 1) * voiced_mask.reshape(1, -1)
        shimmer = np.clip(shimmer, 0.99, 1.01)

        # 对高频频段应用衰减，避免引入高频噪声
        n_freqs = mag.shape[0]
        freq_weight = np.ones(n_freqs)
        high_freq_start = int(n_freqs * 0.6)
        freq_weight[high_freq_start:] = np.linspace(1.0, 0.1, n_freqs - high_freq_start)
        jitter_phase *= freq_weight.reshape(-1, 1)
        shimmer = 1 + (shimmer - 1) * freq_weight.reshape(-1, 1)

        # 重建
        _, result = signal.istft(
            mag * shimmer * np.exp(1j * (phase + jitter_phase)),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        return result[:len(audio)]

    def _formant_sharpen(self, audio: np.ndarray, strength: float = 0.2) -> np.ndarray:
        """共振峰精细调整

        使用 LPC 分析提取共振峰位置，然后对共振峰区域进行轻微增强，
        使音色更加清晰和有辨识度。
        """
        if strength <= 0:
            return audio

        lpc_order = 16
        hop = self.hop_length
        n_frames = len(audio) // hop

        if n_frames < 2:
            return audio

        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        mag_out = mag.copy()

        for i in range(n_frames):
            frame = audio[i * hop:(i + 1) * hop]
            if len(frame) < lpc_order:
                continue

            r = np.correlate(frame, frame, mode='full')
            r = r[len(r) // 2:len(r) // 2 + lpc_order]

            try:
                a = self._levinson_durbin(r, lpc_order)
                roots = np.roots(np.concatenate([[1.0], a]))
                roots = roots[np.abs(roots) < 1.0]

                formant_freqs = np.angle(roots) * self.sample_rate / (2 * np.pi)
                formant_freqs = formant_freqs[formant_freqs > 50]
                formant_freqs = formant_freqs[formant_freqs < self.sample_rate / 2]
                formant_freqs = np.sort(formant_freqs)[:3]

                for freq in formant_freqs:
                    freq_idx = np.argmin(np.abs(f - freq))
                    bandwidth = 3
                    start = max(0, freq_idx - bandwidth)
                    end = min(len(f), freq_idx + bandwidth + 1)
                    
                    boost = 1 + strength * 0.05
                    mag_out[start:end, i] *= boost
                    
                    total_boost = np.sum(mag_out[start:end, i] - mag[start:end, i])
                    if total_boost > 0:
                        other_bandwidth = 10
                        other_start = max(0, freq_idx - other_bandwidth)
                        other_end = min(len(f), freq_idx + other_bandwidth + 1)
                        other_mask = np.ones(other_end - other_start, dtype=bool)
                        other_mask[start - other_start:end - other_start] = False
                        if np.any(other_mask):
                            mag_out[other_start:other_end, i][other_mask] *= (1 - total_boost / np.sum(mag[other_start:other_end, i][other_mask]) * strength * 0.05)

            except Exception:
                continue

        _, result = signal.istft(
            mag_out * np.exp(1j * phase),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        return result[:len(audio)]

    def _levinson_durbin(self, r, order):
        """Levinson-Durbin 递归算法"""
        a = np.zeros(order)
        e = r[0]

        for i in range(order):
            if i == 0:
                sum_val = r[1]
            else:
                sum_val = np.sum(a[:i] * r[i:0:-1])

            k = sum_val / e
            a[i] = k

            if i > 0:
                a[:i] = a[:i] - k * a[:i][::-1]

            e = e * (1 - k * k)

        return a

    def _spectral_detail_restore(self, audio: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """频谱细节恢复

        合成音频通常在高频区域缺乏细节，通过分析原始频谱的精细结构，
        添加高频谐波细节，提升清晰度和真实感。

        关键改进：只增强有声区域的谐波结构，避免放大噪声。
        降低高频增强强度，避免引入高频噪声。
        """
        if strength <= 0:
            return audio

        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        n_frames = mag.shape[1]
        
        energy = np.sum(mag, axis=0)
        max_energy = np.max(energy)
        noise_floor = max_energy * 0.02
        voiced_mask = energy > noise_floor

        high_freq_idx = int(len(f) * 0.5)

        mag_smooth = gaussian_filter1d(mag, sigma=12, axis=0)
        detail = mag - mag_smooth

        mag_out = mag.copy()
        
        mid_high_idx = int(len(f) * 0.75)
        for i in range(n_frames):
            if not voiced_mask[i]:
                continue
            mag_out[high_freq_idx:mid_high_idx, i] += detail[high_freq_idx:mid_high_idx, i] * strength * 0.08
            mag_out[mid_high_idx:, i] += detail[mid_high_idx:, i] * strength * 0.02

        max_mag = np.max(mag)
        if max_mag > 0:
            mag_out = np.clip(mag_out, 0, max_mag * 1.02)

        _, result = signal.istft(
            mag_out * np.exp(1j * phase),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        return result[:len(audio)]

    def _glottal_pulse_simulation(self, audio: np.ndarray, strength: float = 0.15) -> np.ndarray:
        """声门脉冲模拟

        模拟真实声门脉冲的形状（双指数衰减），使合成语音更接近真实人声。
        声门脉冲形状影响语音的音色和自然度。
        """
        if strength <= 0:
            return audio

        hop = 256
        n_frames = len(audio) // hop

        if n_frames < 2:
            return audio

        energy = np.array([
            np.sqrt(np.mean(audio[i * hop:(i + 1) * hop] ** 2))
            for i in range(n_frames)
        ])
        max_energy = np.max(energy)
        if max_energy < 1e-6:
            return audio

        voiced_mask = energy > max_energy * 0.02

        t_pulse = np.linspace(0, 1, hop)
        glottal_pulse = np.zeros(hop)
        tau = 0.3
        for i in range(hop):
            if t_pulse[i] < tau:
                glottal_pulse[i] = (t_pulse[i] / tau) ** 2 * np.exp(-2 * t_pulse[i] / tau)
            else:
                glottal_pulse[i] = ((1 - t_pulse[i]) / (1 - tau)) ** 2 * np.exp(-2 * (1 - t_pulse[i]) / (1 - tau))
        glottal_pulse = glottal_pulse / np.max(glottal_pulse)

        result = audio.copy()
        for i in range(n_frames):
            if voiced_mask[i]:
                frame_start = i * hop
                frame_end = min(frame_start + hop, len(audio))
                frame_len = frame_end - frame_start
                pulse = glottal_pulse[:frame_len]
                result[frame_start:frame_end] = result[frame_start:frame_end] * (1 + (pulse - 0.5) * strength * 0.3)

        return result

    def _micro_pitch_jitter(self, audio: np.ndarray, strength: float = 0.05) -> np.ndarray:
        """微音高抖动（细节颤音）

        添加真实人声的微小音高不规则性，模拟声带振动的自然波动：
        - 频率：8-15 Hz（比明显颤音更高频，不易察觉）
        - 深度：0.5-2 cents（极细微，不影响整体音高）
        - 不规则性：使用白噪声调制，模拟真实声带振动的随机性

        这是"细节"颤音，不是明显的演唱颤音，不会与用户通过音高曲线设置的颤音冲突。
        用户的音高曲线（pitches）控制的是宏观颤音，这里只添加微观的自然抖动。
        """
        if strength <= 0:
            return audio

        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        n_frames = mag.shape[1]

        np.random.seed(42)
        jitter_noise = np.random.randn(n_frames) * 0.005 * strength

        jitter_smooth = gaussian_filter1d(jitter_noise, sigma=2)

        freq_shift = np.exp(jitter_smooth)

        mag_out = mag.copy()
        for i in range(n_frames):
            shift_factor = freq_shift[i]
            if shift_factor > 1:
                mag_out[:, i] = np.interp(
                    np.linspace(0, len(f) - 1, len(f)),
                    np.linspace(0, len(f) - 1, int(len(f) / shift_factor)),
                    mag[:int(len(f) / shift_factor), i],
                    left=0, right=0
                )
            elif shift_factor < 1:
                mag_out[:, i] = np.interp(
                    np.linspace(0, len(f) - 1, int(len(f) / shift_factor)),
                    np.linspace(0, len(f) - 1, len(f)),
                    mag[:, i],
                    left=0, right=0
                )[:len(f)]

        _, result = signal.istft(
            mag_out * np.exp(1j * phase),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        return result[:len(audio)]

    def _breath_simulation(self, audio: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """气息模拟

        在音频中添加真实的气息声：
        - 吸气声：在音素边界处添加轻微的吸气声
        - 呼气声：在元音结尾处添加轻微的气息释放
        - 唇齿摩擦：在某些辅音前添加轻微的摩擦声

        使用白噪声和带通滤波模拟真实气息。
        """
        if strength <= 0:
            return audio

        hop = 512
        n_frames = len(audio) // hop

        if n_frames < 2:
            return audio

        energy = np.array([
            np.sqrt(np.mean(audio[i * hop:(i + 1) * hop] ** 2))
            for i in range(n_frames)
        ])
        max_energy = np.max(energy)
        if max_energy < 1e-6:
            return audio

        energy_smooth = gaussian_filter1d(energy, sigma=3)

        breath_noise = np.random.randn(len(audio)) * 0.0005 * strength

        b, a = signal.butter(4, [500, 8000], btype='band', fs=self.sample_rate)
        breath_noise = signal.filtfilt(b, a, breath_noise)

        edge_mask = np.zeros(len(audio))
        for i in range(1, n_frames - 1):
            prev_energy = energy_smooth[i - 1]
            curr_energy = energy_smooth[i]
            next_energy = energy_smooth[i + 1]

            if curr_energy < max_energy * 0.05:
                if prev_energy > max_energy * 0.1 or next_energy > max_energy * 0.1:
                    frame_start = i * hop
                    frame_end = min((i + 1) * hop, len(audio))
                    edge_mask[frame_start:frame_end] = 1.0

        edge_mask = gaussian_filter1d(edge_mask, sigma=5)

        breath_audio = breath_noise * edge_mask

        fade_samples = int(0.05 * self.sample_rate)
        breath_audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        breath_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        return audio + breath_audio

    def _formant_drift(self, audio: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """共振峰漂移模拟

        模拟真实人声中共振峰的缓慢漂移，使音色更自然：
        - 频率漂移：共振峰中心频率缓慢变化（1-3 Hz）
        - 幅度漂移：共振峰幅度轻微波动
        - 非线性：漂移不是完全周期性的

        这是微观的音色变化，不会影响整体音质。
        """
        if strength <= 0:
            return audio

        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        n_freqs = mag.shape[0]
        n_frames = mag.shape[1]

        np.random.seed(7)
        drift_phase = np.random.rand(n_freqs) * 2 * np.pi
        drift_freq = 1.0 + np.random.rand(n_freqs) * 2.0

        mag_out = mag.copy()

        for i in range(n_frames):
            t_sec = i * self.hop_length / self.sample_rate
            for j in range(n_freqs):
                freq_drift = np.sin(2 * np.pi * drift_freq[j] * t_sec + drift_phase[j]) * 0.005 * strength
                mag_out[j, i] = mag[j, i] * (1 + freq_drift)

        mag_out = gaussian_filter1d(mag_out, sigma=0.5, axis=1)

        _, result = signal.istft(
            mag_out * np.exp(1j * phase),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        return result[:len(audio)]

    def _spectral_envelope_smoothing(self, audio: np.ndarray, strength: float = 0.2) -> np.ndarray:
        """频谱包络平滑

        对频谱包络进行极轻度的时间平滑，消除帧间的频谱突变，使音色过渡更自然。
        """
        if strength <= 0:
            return audio

        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        mag_smooth = gaussian_filter1d(mag, sigma=3, axis=1)
        
        alpha = strength * 0.3
        mag_out = mag_smooth * alpha + mag * (1 - alpha)

        _, result = signal.istft(
            mag_out * np.exp(1j * phase),
            fs=self.sample_rate, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        return result[:len(audio)]


def enhance_detail(audio: np.ndarray, sample_rate: int = 44100,
                   level: float = 0.5, mode: str = 'balanced') -> np.ndarray:
    """便捷函数：细节增强"""
    enhancer = DetailEnhancer(sample_rate=sample_rate)
    return enhancer.enhance(audio, level=level, mode=mode)
