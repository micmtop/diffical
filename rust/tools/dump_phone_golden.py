"""Dump per-phone and phrase goldens from the Python reference renderer.

The Rust phone/assemble/vocoder parity tests consume raw little-endian f32
arrays plus small JSON shape files, so they never need numpy at runtime.

Usage:
  DIFFICAL_CONFIG=... python rust/tools/dump_phone_golden.py <request.json> <outdir> \
      [--strip-flags] [--render]

Outputs:
  meta.json          request/phrase/phone metadata
  f0_phrase.f32      phrase F0 (total_frames)
  mel_phrase.f32     assembled log-mel (128 x total_frames)
  phone_XX_mel.f32   per-phone stretched mel (128 x target_n_frames)
  phone_XX_f0.f32    per-phone F0 slice after the 80 ms tail pad
  phone_XX_src.f32   processed source wave fed to the hop=128 mel analyzer
  phone_XX_mel_orig.f32  hop=128 mel of the processed source wave
  samples.f32        rendered phrase samples (with --render)
  samples.wav        rendered phrase (with --render)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

NEO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NEO))
os.environ.setdefault("DIFFICAL_CONFIG", str(NEO / "docs/config.yaml"))
os.environ["DIFFICAL_ENGINE"] = "model"
os.environ.setdefault("DIFFICAL_POINTER_CKPT", str(NEO / "train/ckpt_fix/best.pt"))
os.environ.setdefault("DIFFICAL_SEG_NORM", "1")

from backend.server import _parse_request  # noqa: E402
from backend.renderer import PhraseRenderer  # noqa: E402
from backend.models import get_model_manager  # noqa: E402
from backend.model_assembly import assemble as model_assemble  # noqa: E402


def write_f32(path: pathlib.Path, arr) -> None:
    np.asarray(arr, dtype=np.float32).tofile(path)


def write_shape(path: pathlib.Path, arr) -> None:
    path.write_text(json.dumps({"shape": list(np.asarray(arr).shape)}))


def install_mel_recorder():
    """Record the wave/mel handed to the hop=128 analyzer without editing backend code."""
    analyzer = get_model_manager().mel_analyzer
    original = analyzer.__class__.__call__
    records: list[dict] = []

    def wrapper(self, y, key_shift=0, speed=1.0):
        import torch

        wave = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        out = original(self, y, key_shift=key_shift, speed=speed)
        records.append({
            "wave": np.asarray(wave, dtype=np.float32).reshape(-1),
            "mel": out.detach().cpu().numpy()[0].astype(np.float32),
            "key_shift": float(key_shift),
        })
        return out

    analyzer.__class__.__call__ = wrapper
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    ap.add_argument("outdir")
    ap.add_argument("--strip-flags", action="store_true",
                    help="clear phone flags (no-flag no-curve parity request)")
    ap.add_argument("--render", action="store_true",
                    help="also run the full renderer and dump reference samples")
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.request).read_text(encoding="utf-8"))
    if args.strip_flags:
        for phone in data.get("phones", []):
            phone["flags"] = []
        data["breathiness"] = []
        data["tension"] = []
        data["voicing"] = []
        data["gender"] = []
        data["toneShift"] = []
        data["growl"] = []
        data["curves"] = []

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    records = install_mel_recorder()
    req = _parse_request(data)
    renderer = PhraseRenderer(req)
    renderer._clamp_request()
    total_frames, f0_phrase = renderer._build_pitch_curve()
    phone_features = renderer._process_phones(total_frames, f0_phrase)

    if len(records) != len(phone_features):
        raise RuntimeError(f"mel recorder saw {len(records)} calls, {len(phone_features)} phones")

    write_f32(out / "f0_phrase.f32", f0_phrase)
    write_shape(out / "f0_phrase.json", f0_phrase)

    mel_phrase = model_assemble(phone_features, total_frames, f0_phrase,
                                renderer.config.audio.n_mels)
    write_f32(out / "mel_phrase.f32", mel_phrase)
    write_shape(out / "mel_phrase.json", mel_phrase)

    # Mirror backend/model_assembly.py's segment construction for diagnostics.
    # (Kept here rather than changing backend code, which is read-only.)
    norm = os.environ.get("DIFFICAL_SEG_NORM", "1").lower() not in ("0", "false", "no")
    feats_n = []
    for feat in sorted(phone_features, key=lambda f: f["start_frame"]):
        mel = np.asarray(feat["mel"], dtype=np.float32)
        if mel.shape[1] == 0:
            continue
        a = int(round(float(feat.get("start_frame_exact", feat["start_frame"]))))
        a = max(0, min(total_frames - 1, a))
        b = min(total_frames, a + mel.shape[1])
        if b - a < 2:
            continue
        feats_n.append((a, b, mel))
    segs, lens, spans = [], [], []
    for i, (a, b, mel) in enumerate(feats_n):
        if norm:
            used = (feats_n[i + 1][0] - a) if i + 1 < len(feats_n) else (b - a)
            used = max(2, min(used, mel.shape[1]))
            seg = mel[:, :used]
            b_used = a + used
        else:
            seg = mel[:, : b - a]
            b_used = b
        segs.append(seg)
        lens.append(seg.shape[1])
        spans.append((a, b_used))
    lmax = max(lens)
    arr = np.full((len(segs), renderer.config.audio.n_mels, lmax), -30.0, dtype=np.float32)
    for i, seg in enumerate(segs):
        arr[i, :, : seg.shape[1]] = seg
    seg_idx = np.zeros(total_frames, dtype=np.int64)
    prog = np.zeros(total_frames, dtype=np.float32)
    for i, (a, b) in enumerate(spans):
        seg_idx[a:b] = i + 1
        prog[a:b] = np.linspace(0, 1, b - a)
    boundary = np.zeros(total_frames, dtype=np.float32)
    for a, _ in spans[1:]:
        boundary[max(0, a - 4): min(total_frames, a + 4 + 1)] = 1.0
    f0v = np.asarray(f0_phrase, dtype=np.float32)[:total_frames]
    if f0v.shape[0] < total_frames:
        f0v = np.pad(f0v, (0, total_frames - f0v.shape[0]))
    f0v = np.where((f0v > 0) & (f0v < 55.0), 0.0, f0v)
    for name, value in (("segs", arr), ("seg_len", np.asarray(lens, dtype=np.int64)),
                        ("seg_idx", seg_idx), ("progress", prog),
                        ("boundary", boundary), ("assembler_f0", f0v)):
        write_f32(out / f"{name}.f32", value)
        write_shape(out / f"{name}.json", value)

    phones_meta = []
    for i, (feat, rec) in enumerate(zip(phone_features, records)):
        mel = np.asarray(feat["mel"], dtype=np.float32)
        f0 = np.asarray(feat["f0"], dtype=np.float32)
        wave = np.asarray(feat["wave"], dtype=np.float32)
        prefix = out / f"phone_{i:02d}"
        write_f32(prefix.with_name(prefix.name + "_mel.f32"), mel)
        write_shape(prefix.with_name(prefix.name + "_mel.json"), mel)
        write_f32(prefix.with_name(prefix.name + "_f0.f32"), f0)
        write_shape(prefix.with_name(prefix.name + "_f0.json"), f0)
        write_f32(prefix.with_name(prefix.name + "_wave.f32"), wave)
        write_shape(prefix.with_name(prefix.name + "_wave.json"), wave)
        write_f32(prefix.with_name(prefix.name + "_src.f32"), rec["wave"])
        write_shape(prefix.with_name(prefix.name + "_src.json"), rec["wave"])
        write_f32(prefix.with_name(prefix.name + "_mel_orig.f32"), rec["mel"])
        write_shape(prefix.with_name(prefix.name + "_mel_orig.json"), rec["mel"])
        phones_meta.append({
            "index": i,
            "phoneme": feat["phone"].phoneme,
            "start_frame": int(feat["start_frame"]),
            "start_frame_exact": float(feat["start_frame_exact"]),
            "end_frame": int(feat["end_frame"]),
            "preutter_ms": float(feat["preutter_ms"]),
            "overlap_ms": float(feat["overlap_ms"]),
            "pre_vel_ms": float(feat["pre_vel_ms"]),
            "source_skip_ms": float(feat["source_skip_ms"]),
            "source_time_ms": float(feat["consonant_output_ms"]),
            "envelope_span_ms": float(feat["envelope_span_ms"]),
            "segment_len_ms": float(feat["segment_len_ms"]),
            "tail_silence_frames": int(feat["tail_silence_frames"]),
            "pre_scale": float(feat["pre_scale"]),
            "mel_frames": int(mel.shape[1]),
            "src_samples": int(rec["wave"].size),
            "mel_orig_frames": int(rec["mel"].shape[1]),
            "key_shift": rec["key_shift"],
        })

    if len(phones_meta) != len(renderer.req.phones):
        raise RuntimeError("phone feature count mismatch")

    meta = {
        "request": os.path.abspath(args.request),
        "flags_stripped": bool(args.strip_flags),
        "sample_rate": renderer.sr,
        "hop_size": renderer.hop_size,
        "frame_ms": renderer.frame_ms,
        "origin_hop_size": renderer.config.audio.origin_hop_size,
        "n_mels": renderer.config.audio.n_mels,
        "n_fft": renderer.config.audio.n_fft,
        "win_size": renderer.config.audio.win_size,
        "fmin": renderer.config.audio.mel_fmin,
        "fmax": renderer.config.audio.mel_fmax,
        "request_position_ms": float(renderer.req.position_ms),
        "request_leading_ms": float(renderer.req.leading_ms),
        "request_duration_ms": float(renderer.req.duration_ms),
        "total_frames": int(total_frames),
        "f0_phrase_frames": int(f0_phrase.shape[0]),
        "mel_phrase_frames": int(mel_phrase.shape[1]),
        "seg_count": int(len(segs)),
        "seg_lmax": int(lmax),
        "phones": phones_meta,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    if args.render:
        records.clear()
        response = renderer.render()
        samples = np.asarray(response.samples, dtype=np.float32)
        write_f32(out / "samples.f32", samples)
        write_shape(out / "samples.json", samples)
        import soundfile as sf

        sf.write(str(out / "samples.wav"), samples, renderer.sr)
        meta["rendered_samples"] = int(samples.size)
        meta["estimated_length_ms"] = float(response.estimated_length_ms)
        meta["normalization_trace"] = getattr(renderer, "normalization_trace", None)
        (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(json.dumps({
        "outdir": str(out),
        "phones": len(phones_meta),
        "total_frames": int(total_frames),
        "mel_phrase_frames": int(mel_phrase.shape[1]),
        "rendered": bool(args.render),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
