# Rust backend (Diffical model render path)

Replaces the Python render path with one Rust binary that runs both ONNX models
(PointerGen assembler + pc-nsf-hifigan vocoder). The server needs no Python at
runtime; Python is only used to generate parity fixtures.

## Status

- [x] Mel frontend parity, hop 512 (existing): max 9.060e-5, mean 6.185e-7
- [x] Assembler ONNX parity, fixed input tensors (existing): max 9.060e-6, mean 3.480e-7
- [x] `protocol.rs`: camelCase request parsing, inspection, clamping (`inspect`)
- [x] `phone.rs`: source window, envelope span, velocity, stretch mapping,
      hop-128 mel -> hop-512 target frames, 80 ms tail (`phone-parity`)
- [x] `pitch.rs`: per-frame and 5-tick OpenUtau pitch curves -> phrase F0
- [x] `assemble.rs`: segs/seg_len/seg_idx/progress/boundary/f0 + SEG_NORM (`assemble-parity`)
- [x] `vocoder.rs`: 6-frame context pad, voiced mask, context trim, gap gate,
      exact duration alignment, 5.8 ms edge fade, phrase RMS normalization
      (`e2e-parity`)
- [x] `axum` server: `GET /health`, `POST /render` (`response-parity`)
- [ ] flags, expression curves, HN-SEP / mel refinement, resampling, config YAML.
      See `PROGRESS.md`.

## Layout

```
diffical-core/src/   protocol config mel phone pitch assemble inference vocoder pipeline wav
diffical-cli/src/    main parity serve
tools/               dump_golden.py dump_onnx_golden.py dump_phone_golden.py
```

`mel.rs` builds the Slaney mel filterbank itself (`librosa.filters.mel(...,
htk=False, norm="slaney")`); no dumped basis or Python call is needed at runtime.
The filterbank projection only visits each band's non-zero span.

## Build

```bash
cd rust
cargo build --release
cargo test --workspace
```

## Fixtures

The two parity golden directories are generated (and gitignored). Model paths
below come from `docs/config.yaml` and `export/`; none are baked into the binary.

```bash
cd rust
REQ=/tmp/diffical_requests/req_1f87cfc723804b5b.json
VOCODER=/Users/antony/Documents/diffical/models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx
ASSEMBLER=../export/pointer_fix_single.onnx

# per-phone goldens for a no-flag / no-curve request (the stage-2 gate)
python tools/dump_phone_golden.py "$REQ" golden_phone_noflags --strip-flags --render
# full-pipeline goldens: mel_phrase, f0_phrase, segment inputs, reference wav
python tools/dump_phone_golden.py "$REQ" golden_e2e --render
```

`dump_phone_golden.py` forces `DIFFICAL_ENGINE=model` and
`DIFFICAL_POINTER_CKPT=train/ckpt_fix/best.pt` (the checkpoint
`pointer_fix_single.onnx` was exported from) unless overridden.

## Parity commands and measured results

All commands run from `rust/` with the release binary.

### 1. Request inspection

```bash
./target/release/diffical-cli inspect "$REQ"
```

Prints `parsed OK`, sizes, pitch-curve mode, curve names, all phones and notes.
Test request: 10 phones, 10 notes, 1695.471 ms, 147 frames, 462-point 5-tick
pitch curve, no curves.

### 2. Mel frontend (unchanged)

```bash
./target/release/diffical-cli mel-parity golden
```

`values=33024 max_abs_diff=9.060e-5 mean_abs_diff=6.185e-7` (gate 1e-4).

### 3. Assembler ONNX (unchanged)

```bash
./target/release/diffical-cli onnx-parity golden_onnx "$ASSEMBLER"
```

`shape=[1,128,317] max_abs_diff=9.060e-6 mean_abs_diff=3.480e-7` (gate 1e-4).

### 4. Per-phone mel

```bash
./target/release/diffical-cli phone-parity "$REQ" golden_phone_noflags
```

`per-phone mel: max_abs_diff=2.365e-4 mean_abs_diff=1.506e-6` (gate 1e-3),
worst phone 3 (`"o の"`). Supplementary diagnostics: `f0_max=6.104e-5`,
`mel_orig_max=5.264e-4`, `src_max=0.000e0` (wav loading/peak guard exact).
All 10 mel lengths and every `start_frame`/`start_frame_exact`/`pre_vel_ms`/
`source_skip_ms`/`envelope_span_ms`/`segment_len_ms`/`tail_silence_frames`
matched the Python meta exactly.

### 5. Assembled phrase mel

```bash
./target/release/diffical-cli assemble-parity "$REQ" golden_e2e "$ASSEMBLER"
```

`f0_phrase max_abs_diff=6.104e-5`; `segs max_abs_diff=9.060e-5`;
`progress`, `boundary` exact; `seg_len`/`seg_idx` exact;
`mel_phrase: 128x147 max_abs_diff=7.820e-5 mean_abs_diff=1.320e-6` (gate 1e-3).

### 6. End-to-end waveform

```bash
./target/release/diffical-cli e2e-parity "$REQ" golden_e2e "$ASSEMBLER" "$VOCODER"
```

```
samples=74770 max_abs_diff=3.129e-5 ncc=1.000000000 rms_got=0.147458 rms_ref=0.147458 ratio=1.000000
normalization: reason=target_reached gain=0.871658491 rms_before=0.172086 peak_before=0.688151 active_windows=164/169
python normalization gain: 0.871658256
e2e parity PASS
```

Gate: equal sample count, NCC > 0.99, RMS ratio within 5%; all met.

### 7. HTTP server

```bash
./target/release/diffical-cli serve --port 18573 --assembler "$ASSEMBLER" --vocoder "$VOCODER"
curl -s http://127.0.0.1:18573/health
curl -s -X POST -H 'content-type: application/json' \
     --data-binary @"$REQ" http://127.0.0.1:18573/render > /tmp/render_response.json
./target/release/diffical-cli response-parity /tmp/render_response.json golden_e2e
```

`/health` ->
`{"status":"ok","version":"0.1.0","model_sha":"b42a72911a55a3535e2eedee5918af3279380a32f81a9bcba8d60def2624eea9"}`
(`model_sha` is SHA-256 over assembler bytes followed by vocoder bytes).

`response-parity` on the HTTP body:
`samples=74770 max_abs_diff=3.129e-5 ncc=1.000000000 ratio=1.000000` — same
metrics as the direct pipeline. Two identical POSTs returned byte-identical
JSON.

Steady-state render for this request on an idle machine (models loaded,
`DIFFICAL_TIMING=1`): phones ~93-97 ms, assemble ~15 ms, vocode ~240-256 ms,
post <1 ms (~0.36 s total). The first request after startup is slower
(~0.9-1.0 s, lazy ORT allocation). A separate Python backend process was
started on this machine during part of the session and consumed ~4 cores;
under that contention the same render measured 0.8-1.1 s wall, unchanged
output.

## Notes

- `ort` is used with `download-binaries`; no external dylib is required.
  Sessions are CPU-only (matching Python's `CPUExecutionProvider`), with
  optimization level `All`, 4 intra-op / 2 inter-op threads.
- The whole render is a single vocoder call over the assembled phrase mel,
  matching the Python model path.
- `Rendered` also returns `mel_phrase`, `f0` and per-phone features for debugging.
