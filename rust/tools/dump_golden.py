"""Export golden tensors for Rust parity tests (mel frontend stage).

Writes into rust/golden/:
  wav.f32       raw mono 44.1k samples
  mel.f32       128 x T log-mel (float32, row-major)
  basis.f32     128 x 1025 mel filterbank
  params.json   shape/parameters
"""
import json, pathlib, sys
import numpy as np, soundfile as sf, torch

NEO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NEO))
from backend.wav2mel import PitchAdjustableMelSpectrogram, _mel_filterbank  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "golden"
OUT.mkdir(parents=True, exist_ok=True)
wav_path = NEO / "data/opencpop/wavs/2001000002.wav"
y, sr = sf.read(wav_path, dtype="float32")
y = y.mean(1) if y.ndim > 1 else y
import scipy.signal as ss
if sr != 44100:
    y = ss.resample_poly(y, 44100, sr).astype(np.float32)
    sr = 44100
y = y[: 44100 * 3]  # 3 s is enough for parity

an = PitchAdjustableMelSpectrogram(sample_rate=44100, n_fft=2048, win_length=2048,
                                   hop_length=512, f_min=40.0, f_max=16000.0, n_mels=128)
mel = an(torch.from_numpy(y[None, :])).squeeze(0).numpy().astype(np.float32)
basis = _mel_filterbank(44100, 2048, 128, 40.0, 16000.0).astype(np.float32)

y.astype(np.float32).tofile(OUT / "wav.f32")
mel.tofile(OUT / "mel.f32")
basis.tofile(OUT / "basis.f32")
params = dict(sample_rate=44100, n_fft=2048, win_size=2048, hop=512,
              n_mels=128, fmin=40.0, fmax=16000.0,
              mel_shape=list(mel.shape), basis_shape=list(basis.shape),
              pad_left=(2048 - 512) // 2, pad_right=(2048 - 512 + 1) // 2)
(OUT / "params.json").write_text(json.dumps(params, indent=2))
print("golden:", mel.shape, "mel range", float(mel.min()), float(mel.max()))
