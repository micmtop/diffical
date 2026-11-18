"""Golden tensors for the Rust ONNX assembler parity test."""
import json, pathlib, sys
import numpy as np, torch
NEO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NEO))
from train.dataset_set import SetDataset, collate
from backend.pointer_model import load_pointer

out = pathlib.Path(__file__).resolve().parents[1] / "golden_onnx"
out.mkdir(parents=True, exist_ok=True)
ds = SetDataset(NEO / "train/data/oto_clean", "eval")
b = collate([ds[0]])
m = load_pointer(str(NEO / "train/ckpt_fix/best.pt")); m.eval()
with torch.no_grad():
    ref = m(b["segs"], b["seg_len"], b["seg_idx"], b["progress"], b["boundary"], b["f0"]).numpy()
for k in ("segs", "seg_len", "seg_idx", "progress", "boundary", "f0"):
    arr = b[k].numpy()
    arr.astype(np.float32).tofile(out / f"{k}.f32")
    (out / f"{k}.json").write_text(json.dumps({"shape": list(arr.shape)}))
ref.astype(np.float32).tofile(out / "mel.f32")
(out / "mel.json").write_text(json.dumps({"shape": list(ref.shape)}))
print("golden_onnx:", ref.shape)
