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

def analyze_audio(filepath, name):
    data, sr = sf.read(filepath)
    print(f"\n=== {name} ===")
    print(f"文件: {filepath}")
    print(f"采样率: {sr}")
    print(f"时长: {len(data)/sr:.2f}s")
    print(f"最大振幅: {np.max(np.abs(data)):.6f}")
    print(f"均方根: {np.sqrt(np.mean(data**2)):.6f}")
    
    f0, voiced_flag, _ = librosa.pyin(data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    valid_f0 = f0[voiced_flag & (f0 > 0)]
    if len(valid_f0) > 0:
        print(f"F0范围: {valid_f0.min():.1f}~{valid_f0.max():.1f} Hz")
        print(f"平均F0: {np.mean(valid_f0):.1f} Hz")
        print(f"F0标准差: {np.std(valid_f0):.1f} Hz")
    else:
        print("F0: 无有效值")
    
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sr)
    print(f"频谱质心: {np.mean(spectral_centroid):.1f} Hz")
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sr)
    print(f"频谱带宽: {np.mean(spectral_bandwidth):.1f} Hz")
    
    rms = librosa.feature.rms(y=data)
    print(f"RMS均值: {np.mean(rms):.6f}")
    print(f"RMS标准差: {np.std(rms):.6f}")
    
    zcr = librosa.feature.zero_crossing_rate(data)
    print(f"过零率: {np.mean(zcr):.6f}")
    
    return data, sr

def compare_mel_spectrogram(data1, sr1, data2, sr2, output_path):
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    
    mel1 = librosa.feature.melspectrogram(y=data1, sr=sr1, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db1 = librosa.power_to_db(mel1, ref=np.max)
    
    mel2 = librosa.feature.melspectrogram(y=data2, sr=sr2, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db2 = librosa.power_to_db(mel2, ref=np.max)
    
    print(f"\n=== Mel频谱对比 ===")
    print(f"生成音频Mel范围: {mel_db1.min():.1f}~{mel_db1.max():.1f} dB")
    print(f"参考音频Mel范围: {mel_db2.min():.1f}~{mel_db2.max():.1f} dB")
    print(f"生成音频Mel均值: {np.mean(mel_db1):.1f} dB")
    print(f"参考音频Mel均值: {np.mean(mel_db2):.1f} dB")
    
    mel_diff = mel_db1[:min(mel_db1.shape[0], mel_db2.shape[0]), :min(mel_db1.shape[1], mel_db2.shape[1])] - \
                mel_db2[:min(mel_db1.shape[0], mel_db2.shape[0]), :min(mel_db1.shape[1], mel_db2.shape[1])]
    print(f"Mel差异均值: {np.mean(mel_diff):.1f} dB")
    print(f"Mel差异标准差: {np.std(mel_diff):.1f} dB")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    img1 = librosa.display.specshow(mel_db1, sr=sr1, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axes[0])
    axes[0].set_title('生成音频 Mel频谱')
    if img1:
        plt.colorbar(img1, ax=axes[0])
    
    img2 = librosa.display.specshow(mel_db2, sr=sr2, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axes[1])
    axes[1].set_title('参考音频 Mel频谱')
    if img2:
        plt.colorbar(img2, ax=axes[1])
    
    img3 = librosa.display.specshow(mel_diff, sr=sr1, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axes[2], vmin=-30, vmax=30)
    axes[2].set_title('Mel频谱差异')
    if img3:
        plt.colorbar(img3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n频谱对比图已保存: {output_path}")
    
    return mel_db1, mel_db2

def compare_waveform(data1, sr1, data2, sr2, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    axes[0].plot(np.arange(len(data1))/sr1, data1)
    axes[0].set_title('生成音频 波形')
    axes[0].set_xlabel('时间 (s)')
    
    axes[1].plot(np.arange(len(data2))/sr2, data2)
    axes[1].set_title('参考音频 波形')
    axes[1].set_xlabel('时间 (s)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"波形对比图已保存: {output_path}")

def main():
    config = get_config()
    
    gen_path = './test/output/test_ustx_gen.wav'
    ref_path = './test/opencpop/wavs/2001000001.wav'
    
    gen_data, gen_sr = analyze_audio(gen_path, '生成音频')
    ref_data, ref_sr = analyze_audio(ref_path, '参考音频')
    
    compare_mel_spectrogram(gen_data, gen_sr, ref_data, ref_sr, 
                           './test/output/mel_comparison.png')
    compare_waveform(gen_data, gen_sr, ref_data, ref_sr,
                     './test/output/waveform_comparison.png')
    
    print("\n=== 关键差异分析 ===")
    gen_zcr = np.mean(librosa.feature.zero_crossing_rate(gen_data))
    ref_zcr = np.mean(librosa.feature.zero_crossing_rate(ref_data))
    print(f"过零率对比: 生成={gen_zcr:.6f}, 参考={ref_zcr:.6f}, 比值={gen_zcr/ref_zcr:.2f}x")
    
    gen_centroid = np.mean(librosa.feature.spectral_centroid(y=gen_data, sr=gen_sr))
    ref_centroid = np.mean(librosa.feature.spectral_centroid(y=ref_data, sr=ref_sr))
    print(f"频谱质心对比: 生成={gen_centroid:.1f}Hz, 参考={ref_centroid:.1f}Hz, 差值={gen_centroid-ref_centroid:.1f}Hz")

if __name__ == '__main__':
    main()
