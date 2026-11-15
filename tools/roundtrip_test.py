"""决定性测试：真实录音 → mel → 声码器往返，对比原音频与模型输出。

如果 roundtrip 已经"电"，问题在声码器/mel 管线，不在模型。
"""
from __future__ import annotations
import pathlib, sys
import numpy as np, soundfile as sf, torch, importlib.util, pyworld as pw
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "diffical_new"))
from backend.vocoder.nsf_hifigan import NsfHifiGAN  # noqa: E402
SR, HOP, N_FFT, N_MELS = 44100, 512, 2048, 128
MODEL_DIR = ROOT.parent / "diffical_new/models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
spec = importlib.util.spec_from_file_location("neo_wav2mel", ROOT / "backend/wav2mel.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
EX = M.PitchAdjustableMelSpectrogram(sample_rate=SR, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
                                     n_mels=N_MELS, f_min=40.0, f_max=16000.0, center=False, device="cpu")

def mel_of(w):
    with torch.no_grad(): return EX(torch.from_numpy(w.astype(np.float32))).cpu().numpy()[0]

def hnr(y, f0):
    vals=[]
    for t in range(min(len(y)//HOP, len(f0))):
        f=f0[t]
        if f<=50: continue
        c=t*HOP
        if c+1024>len(y): break
        x=y[c:c+1024].astype(np.float64); x=x-x.mean()
        e=np.dot(x,x)
        if e<1e-8: continue
        lag=int(round(SR/f))
        if lag<2 or lag>=1022: continue
        r=np.dot(x[:-lag],x[lag:])/e*(1024/(1024-lag)); r=min(max(r,1e-6),0.9999)
        vals.append(10*np.log10(r/(1-r)))
    return float(np.median(vals)) if vals else float('nan')

def main():
    bank = pathlib.Path.home()/"Library/OpenUtau/Singers/KAFU Cevio/C5"
    voc = NsfHifiGAN(str(MODEL_DIR/"model.ckpt"), str(MODEL_DIR/"config.json"), device="cpu")
    out_items=[]
    for name in ["a.wav","i.wav"]:
        w, sr = sf.read(str(bank/name), dtype="float32")
        if w.ndim>1: w=w.mean(1)
        if sr != SR:
            xt=np.linspace(0,len(w)-1,int(len(w)*SR/sr)); w=np.interp(xt,np.arange(len(w)),w).astype(np.float32)
        m = mel_of(w)
        f0,_ = pw.harvest(w.astype(np.float64), SR, f0_floor=60, f0_ceil=1200, frame_period=HOP/SR*1000)
        T = min(m.shape[1], len(f0))
        y = np.asarray(voc.spec2wav_torch(m[:, :T], f0[:T]), np.float32).reshape(-1)
        n = min(len(w), len(y))
        print(f"{name}: real_hnr={hnr(w[:n], f0[:T]):.2f}dB  roundtrip_hnr={hnr(y[:n], f0[:T]):.2f}dB")
        out_items.append((name, w[:n], y[:n]))
    gap = np.zeros(int(0.3*SR), np.float32)
    parts=[]
    for name, real, rt in out_items:
        r = np.sqrt(np.mean(real.astype(np.float64)**2))+1e-12
        rt = (rt*(r/(np.sqrt(np.mean(rt.astype(np.float64)**2))+1e-12))).astype(np.float32)
        parts += [real, gap, rt, gap]
    p = ROOT/"listen_test"/"25_往返测试_真实与声码器.wav"
    sf.write(p, np.concatenate(parts), SR)
    print(p)

if __name__ == "__main__":
    main()
