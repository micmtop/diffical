"""Dump a Python reference render for flag parity.

Renders the request with the Python backend, optionally after replacing every
phone's flags with a flags string (the same string the Rust `flag-parity`
command takes via `--inject-flags`). Writes the reference wav and per-phone
mel so the Rust engine can be compared flag by flag.

Usage:
  python rust/tools/dump_flag_golden.py <request.json> <outdir> \
      [--inject-flags "Hb200Hv120Ht30g100Dt-50"] [--neutralize-curves] \
      [--no-render]

Outputs:
  meta.json          request, injected flags, per-phone geometry and mel shape
  phone_XX_mel.f32   stretched per-phone log-mel (128 x frames)
  phone_XX_f0.f32    per-phone F0 slice (after the 80 ms tail)
  samples.f32/.wav   rendered phrase (unless --no-render)
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

# Same order as the Rust parser: two-character names first, then single.
TWO_CHAR = ["Hv", "Hb", "Ht", "He", "HG", "fe", "fl", "fo", "fv", "fp", "ve", "vo"]
SINGLE_CHAR = list("gtABGPSpRDCZ")
STANDARD = set(TWO_CHAR) | set(SINGLE_CHAR)

# Curves that make the Python renderer apply phrase-level effects which the
# Rust flag pass does not implement. They must stay at their neutral defaults
# for a flag-only comparison.
NEUTRAL = {
    "breathiness": 0.0,
    "tension": 0.0,
    "voicing": 100.0,
    "gender": 0.0,
    "growl": 0.0,
    "toneShift": 0.0,
    "brel": 0.0,
    "breh": 0.0,
    "warm": 0.0,
    "hcmp": 0.0,
    "lowcut": 0.0,
}


def parse_flag_string(text: str) -> list[tuple[str, float | None]]:
    """Case-sensitive names, optional signed value, first occurrence wins."""
    out: list[tuple[str, float | None]] = []
    seen: set[str] = set()
    i = 0
    while i < len(text):
        name = next((n for n in TWO_CHAR + SINGLE_CHAR if text.startswith(n, i)), None)
        if name is None:
            i += 1
            continue
        i += len(name)
        j = i
        if j < len(text) and text[j] in "+-":
            j += 1
        digits = j
        while j < len(text) and text[j].isdigit():
            j += 1
        value = float(text[i:j]) if j > digits else None
        i = j
        if name in seen:
            continue
        seen.add(name)
        out.append((name, value))
    return out


def inject(data: dict, entries: list[tuple[str, float | None]]) -> None:
    flags = [[name, value, name] for name, value in entries]
    for phone in data.get("phones", []):
        phone["flags"] = [list(f) for f in flags]


def check_curves(data: dict, neutralize: bool) -> list[str]:
    warnings: list[str] = []
    for key, neutral in NEUTRAL.items():
        values = data.get(key) or []
        if len(values) == 0:
            continue
        arr = np.asarray(values, dtype=np.float64)
        if neutralize:
            data[key] = []
            continue
        if not np.allclose(arr, neutral, atol=1e-6):
            warnings.append(f"{key} curve is non-neutral (min {arr.min():.3f}, max {arr.max():.3f})")
    if neutralize:
        data["curves"] = []
    return warnings


def write_f32(path: pathlib.Path, arr) -> None:
    np.asarray(arr, dtype=np.float32).tofile(path)


def write_shape(path: pathlib.Path, arr) -> None:
    path.write_text(json.dumps({"shape": list(np.asarray(arr).shape)}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    ap.add_argument("outdir")
    ap.add_argument("--inject-flags", default=None,
                    help='flags string applied to every phone, e.g. "Hb200Hv120g100"')
    ap.add_argument("--neutralize-curves", action="store_true",
                    help="clear expression curves that would add phrase-level effects")
    ap.add_argument("--no-render", action="store_true",
                    help="skip the full render (mel/F0 only)")
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.request).read_text(encoding="utf-8"))
    entries: list[tuple[str, float | None]] = []
    if args.inject_flags:
        entries = parse_flag_string(args.inject_flags)
        if not entries:
            raise SystemExit(f"no standard flags found in {args.inject_flags!r}")
        inject(data, entries)

    warnings = check_curves(data, args.neutralize_curves)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if warnings and not args.neutralize_curves:
        print("WARNING: phrase-level curve effects are active; flag parity for this "
              "request will include them. Re-run with --neutralize-curves to isolate flags.",
              file=sys.stderr)

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    req = _parse_request(data)
    renderer = PhraseRenderer(req)
    renderer._clamp_request()
    total_frames, f0_phrase = renderer._build_pitch_curve()
    features = renderer._process_phones(total_frames, f0_phrase)

    phones_meta = []
    for i, feat in enumerate(features):
        mel = np.asarray(feat["mel"], dtype=np.float32)
        f0 = np.asarray(feat["f0"], dtype=np.float32)
        prefix = out / f"phone_{i:02d}"
        write_f32(prefix.with_name(prefix.name + "_mel.f32"), mel)
        write_shape(prefix.with_name(prefix.name + "_mel.json"), mel)
        write_f32(prefix.with_name(prefix.name + "_f0.f32"), f0)
        write_shape(prefix.with_name(prefix.name + "_f0.json"), f0)
        phones_meta.append({
            "index": i,
            "phoneme": feat["phone"].phoneme,
            "start_frame": int(feat["start_frame"]),
            "start_frame_exact": float(feat["start_frame_exact"]),
            "mel_frames": int(mel.shape[1]),
            "f0_frames": int(f0.shape[0]),
            "preutter_ms": float(feat["preutter_ms"]),
            "overlap_ms": float(feat["overlap_ms"]),
            "pre_vel_ms": float(feat["pre_vel_ms"]),
            "source_skip_ms": float(feat["source_skip_ms"]),
            "segment_len_ms": float(feat["segment_len_ms"]),
            "tail_silence_frames": int(feat["tail_silence_frames"]),
            "pre_scale": float(feat["pre_scale"]),
        })

    meta = {
        "request": os.path.abspath(args.request),
        "inject_flags": args.inject_flags,
        "flags": [[name, value] for name, value in entries],
        "neutralize_curves": bool(args.neutralize_curves),
        "curve_warnings": warnings,
        "sample_rate": renderer.sr,
        "hop_size": renderer.hop_size,
        "frame_ms": renderer.frame_ms,
        "total_frames": int(total_frames),
        "phones": phones_meta,
    }

    if not args.no_render:
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
        "inject_flags": args.inject_flags,
        "rendered": not args.no_render,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
