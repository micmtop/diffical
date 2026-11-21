"""Export the HN-SEP (CascadedNet) mask network to ONNX.

The Rust engine runs the mask network with ONNX Runtime and does its own
STFT/ISTFT (the reference wrapper keeps those in torch). The exported graph is
therefore a pure spectral mask:

    input  "spec": float32 (1, 2, 1025, T)  real/imag of a 2048-point STFT
    output "mask": float32 (1, 2, 1025, T)  real/imag of the bounded mask

T is dynamic, so a phone of any length can be processed in one call.

Usage:
  python rust/tools/export_hnsep.py \
      --model /path/to/hnsep/vr/model.pt \
      --out ../export/hnsep.onnx \
      [--opset 17] [--check-wav /path/to/1-2s.wav]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

NEO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NEO))

from backend.vocoder.hnsep_nets import CascadedNet  # noqa: E402

N_FFT = 2048
HOP = 512
N_OUT = 32
N_OUT_LSTM = 128


class MaskExport(nn.Module):
    """CascadedNet.forward for a real/imag spectrum pair, no complex tensors.

    Mirrors ``CascadedNet.forward(is_complex=True, is_mono=True)``: the 1025-bin
    input is truncated to 1024 bins, the mask is bounded with the same
    ``tanh(|m|) * m / (|m| + eps)`` rule and the last bin is replicated.
    """

    def __init__(self, net: CascadedNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        net = self.net
        x = spec[:, :, : net.max_bin]
        bandw = x.size()[2] // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        l1 = net.stg1_low_band_net(l1_in)
        h1 = net.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)

        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)
        l2 = net.stg2_low_band_net(l2_in)
        h2 = net.stg2_high_band_net(h2_in)
        aux2 = torch.cat([l2, h2], dim=2)

        f3_in = torch.cat([x, aux1, aux2], dim=1)
        f3 = net.stg3_full_band_net(f3_in)

        mask = net.out(f3)
        real, imag = mask[:, :1], mask[:, 1:]
        mag = torch.sqrt(real * real + imag * imag)
        scale = torch.tanh(mag) / (mag + 1e-8)
        mask = mask * scale
        return F.pad(mask, (0, 0, 0, 1), mode="replicate")


def build_net(model_path: pathlib.Path) -> CascadedNet:
    net = CascadedNet(
        n_fft=N_FFT,
        hop_length=HOP,
        nout=N_OUT,
        nout_lstm=N_OUT_LSTM,
        is_complex=True,
        is_mono=True,
        fixed_length=True,
    )
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(state)
    net.eval()
    return net


def spec_from_wav(wav: np.ndarray) -> np.ndarray:
    """(2, 1025, T) real/imag STFT with the reference wrapper's padding."""
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).reshape(1, 1, -1)
    net_hop = HOP
    seg_length = 32 * net_hop
    t = x.shape[-1]
    t1 = t + net_hop
    t_pad = seg_length * ((t1 - 1) // seg_length + 1) - t1
    nl_pad = t_pad // 2 // net_hop
    tl_pad = nl_pad * net_hop
    x = F.pad(x, (tl_pad, t_pad - tl_pad))
    spec = torch.stft(
        x.reshape(-1, x.shape[-1]),
        n_fft=N_FFT,
        hop_length=HOP,
        return_complex=True,
        window=torch.hann_window(N_FFT),
        pad_mode="constant",
    )
    spec = spec.reshape(1, 1, spec.shape[-2], spec.shape[-1])
    return torch.cat([spec.real, spec.imag], dim=1)[0].numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="CascadedNet checkpoint (model.pt)")
    ap.add_argument("--out", required=True, help="destination .onnx")
    ap.add_argument("--opset", type=int, default=17, choices=(17, 18))
    ap.add_argument("--check-wav", default=None,
                    help="optional real 1-2 s wav for a torch-vs-ort mask check")
    args = ap.parse_args()

    net = build_net(pathlib.Path(args.model))
    wrapper = MaskExport(net).eval()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seq = 160  # multiple of 32: the reference pads spec frames to 32k
    dummy = torch.zeros(1, 2, N_FFT // 2 + 1, seq, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out),
        input_names=["spec"],
        output_names=["mask"],
        dynamic_axes={"spec": {0: "batch", 3: "frames"},
                      "mask": {0: "batch", 3: "frames"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    import onnxruntime as ort

    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, opset {args.opset})")
    print("inputs:", [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
                      for i in model.graph.input])
    print("outputs:", [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
                       for o in model.graph.output])

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    checks = []
    if args.check_wav:
        import soundfile as sf

        wav, sr = sf.read(args.check_wav, dtype="float32", always_2d=True)
        if sr != 44100:
            raise SystemExit(f"check wav must be 44.1 kHz, got {sr}")
        wav = wav.mean(axis=1)
        wav = wav[: int(1.6 * sr)]
        checks.append(("real-wav-spec", spec_from_wav(wav)))
    for n in (seq, 32, 320, 1024):
        checks.append((f"random-{n}",
                       rng.standard_normal((2, N_FFT // 2 + 1, n)).astype(np.float32) * 0.1))

    worst = 0.0
    np.set_printoptions(precision=3, suppress=True)
    for label, spec in checks:
        with torch.no_grad():
            want = wrapper(torch.from_numpy(spec)[None]).numpy()
        got = sess.run(["mask"], {"spec": spec[None]})[0]
        diff = np.abs(got - want)
        worst = max(worst, float(diff.max()))
        print(f"  {label}: shape {got.shape} mask max_abs_diff={diff.max():.3e} "
              f"mean={diff.mean():.3e} mask_abs_max={np.abs(want).max():.3f}")
    if worst >= 1e-4:
        raise SystemExit(f"torch vs onnxruntime mask diff {worst:.3e} >= 1e-4")
    print(f"export OK: worst mask max_abs_diff={worst:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
