import sys
from pathlib import Path
import os
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import get_config

class AudioQualityInspector:
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
    
    def _calculate_high_freq_ratio(self, audio):
        n_fft = 2048
        hop_length = 512
        fft_result = np.fft.rfft(audio, n=n_fft)
        freq_bins = np.fft.rfftfreq(n_fft, 1/self.sr)
        high_freq_mask = freq_bins > 8000
        total_energy = np.sum(np.abs(fft_result)**2)
        high_freq_energy = np.sum(np.abs(fft_result[high_freq_mask])**2)
        return high_freq_energy / (total_energy + 1e-10)
    
    def _calculate_mel_diff_mean(self, generated, reference):
        n_fft = 2048
        hop_length = 512
        n_mels = 128
        mel1 = librosa.feature.melspectrogram(y=generated, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_db1 = librosa.power_to_db(mel1, ref=np.max)
        mel2 = librosa.feature.melspectrogram(y=reference, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_db2 = librosa.power_to_db(mel2, ref=np.max)
        min_h = min(mel_db1.shape[0], mel_db2.shape[0])
        min_w = min(mel_db1.shape[1], mel_db2.shape[1])
        diff = mel_db1[:min_h, :min_w] - mel_db2[:min_h, :min_w]
        return np.mean(diff)
    
    def _calculate_mel_diff_std(self, generated, reference):
        n_fft = 2048
        hop_length = 512
        n_mels = 128
        mel1 = librosa.feature.melspectrogram(y=generated, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_db1 = librosa.power_to_db(mel1, ref=np.max)
        mel2 = librosa.feature.melspectrogram(y=reference, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_db2 = librosa.power_to_db(mel2, ref=np.max)
        min_h = min(mel_db1.shape[0], mel_db2.shape[0])
        min_w = min(mel_db1.shape[1], mel_db2.shape[1])
        diff = mel_db1[:min_h, :min_w] - mel_db2[:min_h, :min_w]
        return np.std(diff)
    
    def _calculate_quality_score(self, audio, reference_audio=None):
        scores = []
        
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        zcr_score = max(0, 100 - abs(zcr - 0.1) * 500)
        scores.append(zcr_score)
        
        centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sr))
        centroid_score = max(0, 100 - abs(centroid - 4000) / 10)
        scores.append(centroid_score)
        
        rms = np.sqrt(np.mean(audio**2))
        rms_score = max(0, 100 - abs(rms - 0.05) * 1000)
        scores.append(rms_score)
        
        peak = np.max(np.abs(audio))
        peak_score = 100 if peak <= 0.99 else max(0, 100 - (peak - 0.99) * 10000)
        scores.append(peak_score)
        
        if reference_audio is not None:
            mel_diff_mean = self._calculate_mel_diff_mean(audio, reference_audio)
            mel_score = max(0, 100 - abs(mel_diff_mean) * 2)
            scores.append(mel_score)
        
        return np.mean(scores)
    
    def analyze_time_domain(self, audio):
        return {
            'rms': float(np.sqrt(np.mean(audio**2))),
            'peak_amplitude': float(np.max(np.abs(audio))),
            'zcr': float(np.mean(librosa.feature.zero_crossing_rate(audio))),
            'dynamic_range': float(20 * np.log10(np.max(np.abs(audio)) / (np.sqrt(np.mean(audio**2)) + 1e-10))),
        }
    
    def analyze_frequency_domain(self, audio):
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)
        return {
            'spectral_centroid': float(np.mean(centroid)),
            'spectral_bandwidth': float(np.mean(bandwidth)),
            'high_freq_ratio': float(self._calculate_high_freq_ratio(audio)),
        }
    
    def compare_with_reference(self, generated, reference):
        return {
            'mel_diff_mean': float(self._calculate_mel_diff_mean(generated, reference)),
            'mel_diff_std': float(self._calculate_mel_diff_std(generated, reference)),
            'centroid_diff': float(np.mean(librosa.feature.spectral_centroid(y=generated, sr=self.sr)) - 
                                    np.mean(librosa.feature.spectral_centroid(y=reference, sr=self.sr))),
            'zcr_ratio': float(np.mean(librosa.feature.zero_crossing_rate(generated)) / 
                               (np.mean(librosa.feature.zero_crossing_rate(reference)) + 1e-10)),
        }
    
    def evaluate_pitch(self, audio):
        f0, voiced_flag, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), 
                                           fmax=librosa.note_to_hz('C7'), sr=self.sr)
        valid_f0 = f0[voiced_flag & (f0 > 0)]
        return {
            'f0_mean': float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0,
            'f0_std': float(np.std(valid_f0)) if len(valid_f0) > 0 else 0,
            'f0_min': float(np.min(valid_f0)) if len(valid_f0) > 0 else 0,
            'f0_max': float(np.max(valid_f0)) if len(valid_f0) > 0 else 0,
            'jitter': float(np.mean(np.abs(np.diff(valid_f0)))) if len(valid_f0) > 1 else 0,
        }
    
    def generate_report(self, audio, reference_audio=None):
        report = {
            'time_domain': self.analyze_time_domain(audio),
            'frequency_domain': self.analyze_frequency_domain(audio),
            'pitch': self.evaluate_pitch(audio),
            'quality_score': self._calculate_quality_score(audio, reference_audio),
        }
        if reference_audio is not None:
            report['comparison'] = self.compare_with_reference(audio, reference_audio)
        return report
    
    def plot_waveform_comparison(self, generated, reference, output_path):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        axes[0, 0].plot(np.arange(len(generated))/self.sr, generated)
        axes[0, 0].set_title('生成音频 完整波形')
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylim([-1.1, 1.1])
        
        axes[0, 1].plot(np.arange(len(reference))/self.sr, reference)
        axes[0, 1].set_title('参考音频 完整波形')
        axes[0, 1].set_xlabel('时间 (s)')
        axes[0, 1].set_ylim([-1.1, 1.1])
        
        sample_len = min(len(generated), len(reference), 4410)
        axes[1, 0].plot(np.arange(sample_len)/self.sr, generated[:sample_len])
        axes[1, 0].set_title('生成音频 放大波形(0.1s)')
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylim([-1.1, 1.1])
        
        axes[1, 1].plot(np.arange(sample_len)/self.sr, reference[:sample_len])
        axes[1, 1].set_title('参考音频 放大波形(0.1s)')
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].set_ylim([-1.1, 1.1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"波形对比图已保存: {output_path}")
    
    def plot_spectrum_comparison(self, generated, reference, output_path):
        n_fft = 2048
        hop_length = 512
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        spec1 = librosa.amplitude_to_db(np.abs(librosa.stft(generated, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
        img1 = librosa.display.specshow(spec1, sr=self.sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=axes[0])
        axes[0].set_title('生成音频 频谱')
        plt.colorbar(img1, ax=axes[0])
        
        spec2 = librosa.amplitude_to_db(np.abs(librosa.stft(reference, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
        img2 = librosa.display.specshow(spec2, sr=self.sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=axes[1])
        axes[1].set_title('参考音频 频谱')
        plt.colorbar(img2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"频谱对比图已保存: {output_path}")
    
    def print_report(self, report, generated_path, reference_path=None):
        print("\n" + "=" * 70)
        print("音频质量检验报告")
        print("=" * 70)
        print(f"测试文件: {generated_path}")
        if reference_path:
            print(f"参考文件: {reference_path}")
        
        print("\n--- 时域特征 ---")
        td = report['time_domain']
        print(f"RMS: {td['rms']:.6f} (合格范围: 0.02-0.10)")
        print(f"峰值振幅: {td['peak_amplitude']:.6f} (合格范围: <=0.99)")
        print(f"过零率: {td['zcr']:.6f} (合格范围: 0.05-0.15)")
        print(f"动态范围: {td['dynamic_range']:.1f} dB")
        
        print("\n--- 频域特征 ---")
        fd = report['frequency_domain']
        print(f"频谱质心: {fd['spectral_centroid']:.1f} Hz (合格范围: 3000-5000Hz)")
        print(f"频谱带宽: {fd['spectral_bandwidth']:.1f} Hz (合格范围: 3500-5000Hz)")
        print(f"高频能量占比: {fd['high_freq_ratio']:.2%} (合格范围: <20%)")
        
        print("\n--- 音高特征 ---")
        pitch = report['pitch']
        print(f"F0均值: {pitch['f0_mean']:.1f} Hz")
        print(f"F0范围: {pitch['f0_min']:.1f} - {pitch['f0_max']:.1f} Hz")
        print(f"F0标准差: {pitch['f0_std']:.1f} Hz")
        print(f"F0抖动: {pitch['jitter']:.1f} Hz")
        
        if 'comparison' in report:
            comp = report['comparison']
            print("\n--- 参考对比 ---")
            print(f"Mel差异均值: {comp['mel_diff_mean']:.1f} dB")
            print(f"Mel差异标准差: {comp['mel_diff_std']:.1f} dB")
            print(f"频谱质心差异: {comp['centroid_diff']:.1f} Hz")
            print(f"过零率比值: {comp['zcr_ratio']:.2f}x")
        
        print("\n--- 质量评分 ---")
        score = report['quality_score']
        print(f"综合评分: {score:.1f}/100")
        
        if score >= 90:
            grade = 'S'
            desc = '优秀 - 接近专业级合成器水平'
        elif score >= 80:
            grade = 'A'
            desc = '良好 - 优秀质量'
        elif score >= 70:
            grade = 'B'
            desc = '合格 - 良好质量'
        elif score >= 60:
            grade = 'C'
            desc = '基本合格 - 需要优化'
        elif score >= 40:
            grade = 'D'
            desc = '较差 - 需要重大优化'
        else:
            grade = 'F'
            desc = '极差 - 严重问题'
        
        print(f"质量等级: {grade} - {desc}")
        
        print("\n--- 问题清单 ---")
        issues = []
        if td['zcr'] > 0.15:
            issues.append(f"[P1] 过零率过高({td['zcr']:.4f}) - 可能存在大量高频噪声")
        elif td['zcr'] < 0.03:
            issues.append(f"[P2] 过零率过低({td['zcr']:.4f}) - 可能过度压缩")
        
        if fd['spectral_centroid'] > 5000:
            issues.append(f"[P1] 频谱质心过高({fd['spectral_centroid']:.1f}Hz) - 高频能量过多")
        elif fd['spectral_centroid'] < 2500:
            issues.append(f"[P2] 频谱质心过低({fd['spectral_centroid']:.1f}Hz) - 缺乏高频细节")
        
        if fd['high_freq_ratio'] > 0.2:
            issues.append(f"[P1] 高频能量占比过高({fd['high_freq_ratio']:.2%}) - 尖锐噪声")
        
        if td['peak_amplitude'] > 0.99:
            issues.append(f"[P1] 峰值削波({td['peak_amplitude']:.6f}) - 可能存在失真")
        
        if td['rms'] < 0.02:
            issues.append(f"[P2] RMS过低({td['rms']:.6f}) - 音量不足")
        elif td['rms'] > 0.1:
            issues.append(f"[P2] RMS过高({td['rms']:.6f}) - 可能过载")
        
        if pitch['jitter'] > 10:
            issues.append(f"[P2] F0抖动过大({pitch['jitter']:.1f}Hz) - 音高不稳定")
        
        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            print("  未发现明显问题")
        
        print("=" * 70)

def main():
    config = get_config()
    inspector = AudioQualityInspector(sample_rate=config.audio.sample_rate)
    
    gen_path = './test/output/test_ustx_gen.wav'
    ref_path = './test/opencpop/wavs/2001000001.wav'
    
    gen_data, gen_sr = sf.read(gen_path)
    ref_data, ref_sr = sf.read(ref_path)
    
    if len(gen_data.shape) > 1:
        gen_data = gen_data[:, 0]
    if len(ref_data.shape) > 1:
        ref_data = ref_data[:, 0]
    
    report = inspector.generate_report(gen_data, ref_data)
    
    inspector.print_report(report, gen_path, ref_path)
    
    output_dir = './test/output'
    os.makedirs(output_dir, exist_ok=True)
    
    inspector.plot_waveform_comparison(gen_data, ref_data, 
                                       os.path.join(output_dir, 'waveform_comparison_detailed.png'))
    inspector.plot_spectrum_comparison(gen_data, ref_data,
                                       os.path.join(output_dir, 'spectrum_comparison.png'))

if __name__ == '__main__':
    main()
