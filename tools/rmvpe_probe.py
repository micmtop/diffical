import numpy as np
from rmvpe_onnx import RMVPE

m = RMVPE(model_path=r"D:/diffical_neo/rmvpe.onnx", device="cpu")
sr = 16000
t = np.arange(sr * 2) / sr
x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
out = m.predict(x, sr)
print("n_out", len(out))
for i, o in enumerate(out):
    a = np.asarray(o)
    extra = ""
    if a.size and a.dtype.kind == "f":
        pos = a[a > 0]
        extra = f"median_pos={float(np.median(pos)) if pos.size else 0:.3f}"
    print(i, a.shape, a.dtype, extra)
