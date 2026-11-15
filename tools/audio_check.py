"""音频域客观验证：identity vs model 渲染后，与真实录音的 MCD / 对数mel 距离。"""
from __future__ import annotations
import pathlib, sys
import numpy as np, soundfile as sf, torch, importlib.util
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "diffical_new"))
from model.pointer import PointerGen, gather_seg  # noqa: E402
from train.dataset_set import SetDataset, collate  # noqa: E402
from backend.vocoder.nsf_hifigan import NsfHifiGAN  # noqa: E402
SR, HOP, N_FFT, N_MELS = 44100, 512, 2048, 128
MODEL_DIR = ROOT.parent / "diffical_new/models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
_sp = importlib.util.spec_from_file_location("neo_wav2mel", ROOT / "backend/wav2mel.py")
_M = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_M)
EX = _M.PitchAdjustableMelSpectrogram(sample_rate=SR, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
                                     n_mels=N_MELS, f_min=40.0, f_max=16000.0, center=False, device="cpu")
def mel_of(w):
    with torch.no_grad(): return EX(torch.from_numpy(w.astype(np.float32))).cpu().numpy()[0]
def mcd(a, b, n=13):
    T = min(a.shape[1], b.shape[1])
    A = np.log(np.maximum(np.exp(a[:, :T]), 1e-9)); B = np.log(np.maximum(np.exp(b[:, :T]), 1e-9))
    from scipy.fftpack import dct
    CA = dct(A, type=2, axis=0, norm="ortho")[:n]; CB = dct(B, type=2, axis=0, norm="ortho")[:n]
    return float((10/np.log(10)) * np.mean(np.sqrt(2*np.sum((CA-CB)**2, axis=0))))
def main():
    data = ROOT/"train/data/mm_v2"; ckpt = ROOT/"train/ckpt_mm2/last.pt"
    ds = SetDataset(data, "eval"); n = min(6, len(ds))
    b = {k: v for k, v in collate([ds[i] for i in range(n)]).items()}
    ck = torch.load(ckpt, map_location="cpu"); model = PointerGen(); model.load_state_dict(ck["state_dict"]); model.eval()
    voc = NsfHifiGAN(str(MODEL_DIR/"model.ckpt"), str(MODEL_DIR/"config.json"), device="cpu")
    rows=[]
    for i in range(n):
        d = ds[i]; utt = ds.files[i].stem
        segs = d["segs"][None]; sl = d["seg_len"][None]; si = d["seg_idx"][None]
        pr = d["progress"][None]; bd = d["boundary"][None]; f0 = d["f0"][None]
        with torch.no_grad():
            pred = model(segs, sl, si, pr, bd, f0)[0].numpy()
            base = gather_seg(segs, sl, si, pr)[0].numpy()
        wav, _sr = sf.read(str(ROOT/"data/opencpop/wavs"/f"{utt}.wav"), dtype="float32")
        real = mel_of(wav); T = min(real.shape[1], pred.shape[1], base.shape[1])
        f0v = f0[0, :T].numpy()
        y_m = np.asarray(voc.spec2wav_torch(pred[:, :T], f0v), np.float32).reshape(-1)
        y_i = np.asarray(voc.spec2wav_torch(base[:, :T], f0v), np.float32).reshape(-1)
        mr, mi, mm = mel_of(real.T.copy() if False else wav), None, None
        # 渲染音频重新提 mel 后比
        m_id = mel_of(y_i); m_md = mel_of(y_m)
        T2 = min(real.shape[1], m_id.shape[1], m_md.shape[1])
        d_id = mcd(real[:, :T2], m_id[:, :T2]); d_md = mcd(real[:, :T2], m_md[:, :T2])
        rows.append((utt, d_id, d_md))
        print(f"{utt}: MCD identity={d_id:.2f} model={d_md:.2f} {'BETTER' if d_md < d_id else 'worse'}", flush=True)
    a = np.array([[r[1], r[2]] for r in rows])
    print(f"mean MCD: identity={a[:,0].mean():.2f} model={a[:,1].mean():.2f} delta={a[:,1].mean()-a[:,0].mean():+.2f}")
if __name__ == "__main__":
    main()
