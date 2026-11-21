# Rust backend (Diffical model render path)

Replaces the Python render path with one Rust binary that runs both ONNX models
(PointerGen assembler + pc-nsf-hifigan vocoder) and the HN-SEP mask network.
The server needs no Python at runtime; Python is only used to generate parity
fixtures.

## Status

- [x] Mel frontend parity, hop 512: max 9.060e-5, mean 6.185e-7
- [x] Assembler ONNX parity, fixed input tensors: max 9.060e-6, mean 3.480e-7
- [x] `protocol.rs`: camelCase request parsing, inspection, clamping (`inspect`)
- [x] `phone.rs`/`stretch.rs`: source window, envelope span, velocity, stretch
      mapping, hold path, `He` loop, hop-128 mel -> hop-512 target frames, 80 ms
      tail (`phone-parity`)
- [x] `pitch.rs`: per-frame and 5-tick pitch curves -> phrase F0
- [x] `assemble.rs`: segs/seg_len/seg_idx/progress/boundary/f0 + SEG_NORM
      (`assemble-parity`)
- [x] `vocoder.rs`: 6-frame context pad, voiced mask, context trim, gap gate,
      exact duration alignment, 5.8 ms edge fade, phrase RMS normalization
      (`e2e-parity`)
- [x] `flags.rs`: the standard flag set with documented semantics (`FLAGS.md`);
      first-occurrence-wins parsing, optional values, no panic on NaN/Inf
- [x] `hnsep.rs`: CascadedNet mask network over ONNX Runtime with a reference
      STFT/ISTFT (`hnsep-parity`), plus a bounded LRU separation cache.
      Default is per-window separation (first flagged render ~4 s); set
      `DIFFICAL_HNSEP_WHOLE=1` for whole-file separation, which is what the
      Python reference does and the mode used by `flag-parity` (slower);
      loaded lazily, only for `Hb`/`Hv` renders
- [x] `effects.rs`: `Ht` tilt, `HG` growl, `A` pitch-following amplitude
- [x] `axum` server: `GET /health`, `POST /render` (`response-parity`), optional
      `--hnsep` for flagged requests
- [ ] expression curves (breathiness/tension/voicing/gender/growl/... as
      request-wide curves), mel refinement, resampling, config YAML. See
      `PROGRESS.md`.

## Layout

```
diffical-core/src/   protocol config mel phone source stretch pitch assemble
                     inference vocoder pipeline wav flags hnsep effects biquad
                     runtime
diffical-cli/src/    main parity inspect_cmd flags_cmd stats serve
tools/               dump_golden.py dump_onnx_golden.py dump_phone_golden.py
                     dump_flag_golden.py export_hnsep.py hnsep_parity.py
                     flag_parity.sh
```

`mel.rs` builds the Slaney mel filterbank itself; no dumped basis or Python call
is needed at runtime. The filterbank projection only visits each band's non-zero
span. `mel.rs` also implements the `g` key-shift analysis variant.

## Build

```bash
cd rust
cargo build --release
cargo test --workspace
```

## Models and fixtures

Model paths are CLI flags / environment variables; none are baked into the
binary.

```bash
cd rust
VOCODER=/Users/antony/Documents/diffical/models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx
ASSEMBLER=../export/pointer_fix_single.onnx
HNSEP=../export/hnsep.onnx
REQ=/tmp/diffical_requests/req_-725f2e9940438609.json   # 1.6 s, 12 phones
WAV=/path/to/1-2s-44.1k.wav                             # real voicebank sample
```

`REQ` is also copied to `golden/flag_request.json` (gitignored) in this
checkout. The fixture request used by earlier revisions had been removed from
`/tmp`, so the 2026-11 fixture set was regenerated from the request above; the
request-swap is the only reason the absolute numbers below differ from the
previous revision.

```bash
# per-phone goldens for a no-flag / no-curve request
python tools/dump_phone_golden.py golden/flag_request.json golden_phone_noflags \
    --strip-flags --render
# full-pipeline goldens: mel_phrase, F0, segment inputs, reference wav
python tools/dump_phone_golden.py golden/flag_request.json golden_e2e --render
# HN-SEP reference (torch harmonic/noise + input wav) for the Rust chain
python tools/hnsep_parity.py \
    --model /Users/antony/Documents/diffical/models/hnsep/vr/model.pt \
    --onnx ../export/hnsep.onnx \
    --wav "$WAV" --seconds 1.6 --dump-dir /tmp/hnsep_golden
```

`dump_phone_golden.py` forces `DIFFICAL_ENGINE=model` and
`DIFFICAL_POINTER_CKPT=train/ckpt_fix/best.pt` (the checkpoint the assembler was
exported from) unless overridden.

## ONNX export (HN-SEP)

`export/hnsep.onnx` is built from the CascadedNet checkpoint by
`tools/export_hnsep.py`. The graph is a pure spectral mask: input
`spec: float32 (1, 2, 1025, T)` (real/imag), output `mask` of the same shape,
`T` dynamic. Frames must be a multiple of 32, which the reference "pad to whole
segments" preprocessing guarantees.

```bash
python tools/export_hnsep.py \
    --model /Users/antony/Documents/diffical/models/hnsep/vr/model.pt \
    --out ../export/hnsep.onnx --opset 17
```

Export-time check (torch vs onnxruntime mask): worst case 4.3e-4 on random
inputs with `|mask| ~ 1`; real-audio mask differences are far smaller. The
deliverable gate is on the separated harmonic: torch vs ONNX Runtime on a real
1.6 s wav gives `max_abs_diff = 5.960e-08` (`tools/hnsep_parity.py`), and the
Rust STFT/ISTFT + ORT chain against the torch reference gives
`max_abs_diff = 9.313e-08` (`diffical-cli hnsep-parity`).

## Parity commands and measured results

All commands run from `rust/` with the release binary.

### 1. Request inspection

```bash
./target/release/diffical-cli inspect "$REQ"
```

Prints parse status, sizes, pitch-curve mode, curve names, notes, and for every
phone the active flags (`active=[g t Hb/Hv/Ht HG]`) and ignored names.

### 2. Mel frontend

```bash
./target/release/diffical-cli mel-parity golden
```

`values=33024 max_abs_diff=9.060e-5 mean_abs_diff=6.185e-7` (gate 1e-4).

### 3. Assembler ONNX

```bash
./target/release/diffical-cli onnx-parity golden_onnx "$ASSEMBLER"
```

`shape=[1,128,317] max_abs_diff=9.060e-6 mean_abs_diff=3.480e-7` (gate 1e-4).

### 4. Per-phone mel

```bash
./target/release/diffical-cli phone-parity "$REQ" golden_phone_noflags
```

`per-phone mel: max_abs_diff=3.147e-4 mean_abs_diff=1.373e-6` (gate 1e-3),
worst phone 10. All 12 mel lengths and every `start_frame`/`start_frame_exact`/
`pre_vel_ms`/`source_skip_ms`/`envelope_span_ms`/`segment_len_ms`/
`tail_silence_frames` matched the Python meta exactly. `phone-parity` accepts
`--hnsep <onnx>` for requests that carry `Hb`/`Hv` flags.

### 5. Assembled phrase mel

```bash
./target/release/diffical-cli assemble-parity "$REQ" golden_e2e "$ASSEMBLER"
```

`mel_phrase: 128x139 max_abs_diff=2.985e-4 mean_abs_diff=2.049e-6` (gate 1e-3);
`seg_len`/`seg_idx` exact. Add `--hnsep <onnx>` for flagged requests.

### 6. End-to-end waveform

```bash
./target/release/diffical-cli e2e-parity "$REQ" golden_e2e "$ASSEMBLER" "$VOCODER"
```

```
samples=71129 max_abs_diff=1.255e-4 ncc=0.999999997 rms_got=0.109969 rms_ref=0.109967 ratio=1.000018
normalization: reason=peak_limited gain=2.961169965 rms_before=0.037760 peak_before=0.320819 active_windows=156/161
python normalization gain: 2.961115501
e2e parity PASS
```

Gate: equal sample count, NCC > 0.99, RMS ratio within 5%; all met. `--hnsep`
is accepted for flagged requests.

### 7. HN-SEP chain

```bash
./target/release/diffical-cli hnsep-parity /tmp/hnsep_golden "$HNSEP"
```

`hnsep: 70560 samples, harmonic max_abs_diff=9.313e-8 mean_abs_diff=1.750e-9`
(gate 1e-4). This covers the Rust STFT/ISTFT port plus the ONNX session against
the torch reference harmonic.

### 8. Flags

```bash
tools/flag_parity.sh "$REQ" "$ASSEMBLER" "$VOCODER" "$HNSEP" \
    /tmp/diffical_requests/req_-292419ff6f6d320e.json
```

The script renders each case with `tools/dump_flag_golden.py --inject-flags`,
then runs `diffical-cli flag-parity` (per-phone mel gate 1e-3; wav NCC > 0.99 and
RMS ratio within 10%). Measured on the 1.6 s request:

| Case | mel max | wav NCC | RMS ratio | gate |
| --- | --- | --- | --- | --- |
| `g100` | 6.027e-4 | 0.999999997 | 1.000017 | mel |
| `t-50` | 3.147e-4 | 0.008428 (pitch moved) | 1.039107 | mel |
| `Hb200` | 2.775e-4 | 0.999999997 | 0.999993 | mel |
| `Hv120` | 5.226e-4 | 0.999999997 | 1.000018 | mel |
| `Ht30` | 5.535e-1 (deliberate tilt) | 0.998473 | 0.972818 | wav |
| `HG50` | 3.147e-4 | 0.997793 | 1.012626 | mel + wav |
| `A100` | 3.147e-4 | 0.981644 | 0.996770 | mel |
| `He` (short notes) | 3.147e-4 | 0.999999997 | 1.000018 | mel + wav |
| `He` (1.1 s note) | 1.990 (deliberate loop) | 0.994611 | 0.999996 | wav |
| `Hb250Hv90g100` | 6.533e-4 | 0.999999997 | 1.000003 | mel + wav |
| `t20HG40` | 3.147e-4 | -0.158090 (pitch moved) | 1.009074 | mel |

Single-flag CLI form:

```bash
./target/release/diffical-cli flag-parity "$REQ" flag_golden/Hb200 \
    "$ASSEMBLER" "$VOCODER" --inject-flags Hb200
```

`flag-parity` options: `--inject-flags "<flags>"` replaces every phone's flags
(the same string is passed to `dump_flag_golden.py`); `--allow-mel-diff` and
`--allow-wav-diff` switch off a gate for flags whose documented effect the
Python reference does not implement. See `FLAGS.md` for the full table and the
no-op reasons.

### 9. HTTP server

```bash
./target/release/diffical-cli serve --port 18574 \
    --assembler "$ASSEMBLER" --vocoder "$VOCODER" --hnsep "$HNSEP"
curl -s http://127.0.0.1:18574/health
curl -s -X POST -H 'content-type: application/json' \
     --data-binary @/tmp/req_combo.json http://127.0.0.1:18574/render \
     > /tmp/render_response.json
./target/release/diffical-cli response-parity /tmp/render_response.json /tmp/golden_combo
```

`/health` ->
`{"status":"ok","version":"0.1.0","model_sha":"b42a72911a55a3535e2eedee5918af3279380a32f81a9bcba8d60def2624eea9"}`
(`model_sha` is SHA-256 over assembler bytes followed by vocoder bytes).

A flagged POST (`Hb250Hv90...`) returned 200 with 71129 samples and
`response-parity`: `max_abs_diff=1.372e-4 ncc=0.999999997 ratio=1.000009`.

## Notes

- `ort` is used with `download-binaries`; no external dylib is required.
  Sessions are CPU-only (matching Python's `CPUExecutionProvider`), with
  optimization level `All`, 4 intra-op / 2 inter-op threads.
- The whole render is a single vocoder call over the assembled phrase mel,
  matching the Python model path.
- HN-SEP: `--hnsep <onnx>`, then `$DIFFICAL_HNSEP`, then `hnsep.onnx` next to the
  assembler. The session is created lazily on the first `Hb`/`Hv` render; the
  default flag path never loads it and is byte-identical with or without the
  option. A render that needs it without a model fails instead of silently
  changing the voice.
- HN-SEP cache: separations are keyed by canonical path + mtime + half-open
  sample window + sample rate and held in a mutex-protected LRU (64 entries /
  256 MiB). The default key is the whole file, which is what the reference
  separates, so parity is exact: first flagged render of the 12-source, 1.6 s
  phrase takes 22.4 s (phone stage 21.8 s incl. the 2.5 s lazy model load; 12
  misses), a repeat render 0.34 s (12 hits, phone 0.16 s, byte-identical wav).
  `DIFFICAL_HNSEP_WINDOW=1` caches per-phone windows (stretch window plus one
  STFT window of context each side, `DIFFICAL_HNSEP_GUARD` overrides): first
  render 4.2 s (load 2.54 s + 12 window separations 1.06 s + mixing/mel 0.17 s),
  repeat 0.33 s, cache 1.6 MB, RSS ~0.87 GB. Window mode is a documented
  speed/parity trade-off: wav NCC 0.999918 / RMS 0.998903 for Hb200, but the
  per-phone mel drifts (0.259) because the mask network's global context changes
  when it sees only a window. The default remains whole-file so the mel gate is
  met (2.775e-4).
- `A` is level-preserving: after shaping, a scalar gain restores the
  voiced-region RMS and the signal is peak-limited to the pre-`A` peak. On the
  `A100` parity case the final wav RMS ratio is 0.996770 (0.32 %) with
  NCC 0.981644 (the remaining difference is the documented modulation, which
  the reference does not implement).
- `Rendered` also returns `mel_phrase`, `f0` and per-phone features for debugging.
