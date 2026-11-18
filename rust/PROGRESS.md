# Rust rewrite progress

Last updated: 2026-11-18. Scope: `rust/` only; Python backend untouched.

## Done

### Request protocol

- `protocol.rs` mirrors `backend/protocol.py::RenderPhraseRequest`,
  `RenderNote`, `RenderPhone` and the response shape.
- `backend/server.py::_convert_keys` camelCase -> snake_case is reproduced
  recursively before serde deserialization, so mixed keys (`positionMs` plus
  `preutter_ms`) parse.
- `_clamp_request` is ported (non-finite/negative timing, `leading_ms` cap).
- Unknown extra fields are tolerated; missing numeric fields default to 0
  instead of failing the whole render.
- `diffical-cli inspect <request.json>` prints parse status, phrase geometry,
  pitch mode, curve names, notes and phones.

### Phone stage (`phone.rs`, `pitch.rs`)

Ported from `PhoneProcessor.process` and `_process_phones` for the model path:

- WAV loading: mono mean, PCM 8/16/24/32-bit and 32-bit float; 44.1 kHz only.
- Peak guard only above 0.98 (`pre_scale`), no unconditional attenuation.
- Hop-128 log-mel of the full source file (Slaney basis computed in Rust).
- oto source window: offset / consonant / cutoff semantics, `end >= con`,
  clamp to the mel timeline.
- Velocity (`2^(1 - velocity)`), source skip (`oto_preutter * vel - pre_vel`),
  envelope span from the request's p0..p4 points.
- Consonant vs vowel frame mapping, stretch ratio, the long-note hold path
  (`> 2.5x`) with deterministic drift.
- `source_present` frames set to `-80`, 80 ms tail silence with a 3-frame fade.
- F0 slice from `_process_phones` (negative start frames padded with 0).
- Phrase F0 curve (`_build_pitch_curve`): voiced mask, per-frame mode,
  5-tick interpolation mode, 50-1500 Hz clip.

Not used by the model path and therefore not ported: `measure_source_level`
(diagnostic), `wave_processed` / `_concatenate_waveforms` (debug-only), oto.ini
parsing / voicebank search (OpenUtau already supplies resolved `oto_*`), phase
vocoder.

### Assembler (`assemble.rs`)

- `model_assembly.assemble` with `DIFFICAL_SEG_NORM` default on:
  `start_frame_exact.round()` origin, clamp, `b-a < 2` skip, used-span
  trimming, `-30` padding, `seg_len`, `seg_idx` (1-based), inclusive
  `progress` linspace, boundary +/-4 frames, F0 `< 55 Hz` clearing.
- Reuses one process-wide `ort` environment for assembler + vocoder
  (`ort` rejects a second environment).
- Empty segment list returns a `-30` mel, like Python.

### Vocoder + phrase post-processing (`vocoder.rs`)

- pc-nsf-hifigan ONNX, mono, `normalize=False`.
- 6-frame first/last frame repeat context, trimmed after vocoding.
- Frame voiced mask `(f0 > 50) or (mean(exp(mel)) > 0.001)`; unvoiced frames
  forced to `-80`; NaN/negative F0 -> 0.
- Vocoder-internal 128-sample anti-click fades, gap gate with 10 ms ramps,
  NaN/Inf sanitation.
- Exact duration alignment to `round((duration_ms + leading_ms)/1000 * sr)`.
- Final 256-sample (5.8 ms) phrase edge fades.
- `loudness.normalize_phrase_rms`: 20 ms windows, 10 ms hop, -45 dB activity
  gate, target `0.15 * median positive phone volume`, peak ceiling 0.95.

### HTTP server (`serve.rs`)

- `axum` + `tokio`; `GET /health` and `GET /`:
  `{status, version, model_sha}`; `model_sha` = SHA-256 over assembler bytes
  then vocoder bytes.
- `POST /render`: same camelCase JSON as Python; returns
  `{samples, sample_rate, leading_ms, position_ms, estimated_length_ms,
  pitch_ticks, pitch_tones, real_curves}` (last three null).
- Parse errors -> 400; render/join errors -> 500; render runs on
  `spawn_blocking` behind a mutex.
- Loads models once, prints the listen URL and the model hash.

### Tooling / parity

- `tools/dump_phone_golden.py` dumps per-phone mel/F0/source/mel_orig,
  phrase F0/assembled mel, assembler inputs, reference wav and normalization
  trace. It records the hop-128 analyzer input by wrapping the analyzer call,
  so the backend code is not modified.
- Commands: `inspect`, `phone-parity`, `assemble-parity`, `e2e-parity`,
  `response-parity`, `render`, `serve`; `mel-parity`/`onnx-parity` unchanged.
- `cargo test --workspace`: 9 unit tests pass (JSON key conversion, clamping,
  request parsing, `np.interp` endpoints, PCM 16/24-bit wav loading and
  downmixing, sample-rate rejection).

## Measured parity (test request `req_1f87cfc723804b5b.json`)

| Gate | Metric | Result |
| --- | --- | --- |
| mel frontend (512 hop) | max abs | 9.060e-5 |
| assembler ONNX tensors | max abs | 9.060e-6 |
| per-phone mel | max abs | 2.365e-4 (mean 1.506e-6) |
| phrase F0 | max abs | 6.104e-5 |
| assembler input `segs` | max abs | 9.060e-5 |
| `mel_phrase` (128x147) | max abs | 7.820e-5 (mean 1.320e-6) |
| waveform (74770 samples) | max abs | 3.129e-5 |
| waveform | NCC | 1.000000000 |
| waveform | RMS ratio | 1.000000 |
| HTTP response body | NCC / RMS | 1.000000000 / 1.000000 |

Normalization gain: Rust 0.871658491 vs Python 0.871658256.

## Performance

Release binary, same Mac, single request, steady state (models loaded, idle
machine; CPU-only ORT, 4 intra-op / 2 inter-op threads):

| Stage | Rust | Python reference |
| --- | --- | --- |
| phones (hop-128 mel + stretch) | ~93-115 ms | ~300 ms |
| assembler | ~6-33 ms | ~20 ms |
| vocoder | ~240-310 ms | ~600 ms |
| post-processing | <1 ms | ~5 ms |
| total render | ~0.36-0.45 s | 0.80-0.92 s |

The first server request after startup is ~0.9-1.0 s (lazy ORT allocation).
The dense mel filterbank projection was initially ~1.9 s; using per-band
non-zero spans and frame-major magnitudes brought the phone stage to ~0.1 s.
During later reruns an unrelated `python -m backend.server` process (started
before this session) held ~4 cores; wall times rose to 0.8-1.1 s with identical
parity output. Latency numbers on this shared machine should be treated as
indicative, not a benchmark.

## Not done / remaining gaps

### Flags and curves

- `flags` are parsed and inspected but ignored. Python maps `g` (gender),
  `Hb`/`Hv`/`Ht`/`HG` and the curve override rule; that is not ported.
  For the test request all flags are neutral (`g 0`, `B 0`, `H 0`, `P 86`) and
  the no-flag golden is bit-identical, which is why the gates pass.
- Expression curves (breathiness, tension, voicing, gender, growl, lowcut,
  brel, breh, warm, hcmp, tone_shift) are ignored. `tone_shift` is a real
  audio-path gap: requests with a non-empty SHFT curve will be quieter/different
  than Python. `dynamics` is intentionally ignored because Python applies DYN
  in OpenUtau playback, not in this backend.
- Curves are still parsed, so the server accepts the request; it just renders
  the neutral voice instead of the drawn expression.

### HN-SEP and effects

- HN-SEP separation (`_get_hnsep`), per-phone HN processing, dynamic low-cut,
  growl, warmth and harmonic compression are not ported. All of them are
  no-ops for default/empty curves, which is the current parity target.
- AI mel refinement is disabled in Python (`AI_REFINE_DISABLED = True`) and is
  not ported.
- Per-note loudness matching is removed in Python and deliberately not ported.

### Audio input

- Sample-rate conversion is rejected: only 44.1 kHz voicebanks are supported.
  Python falls back to `resampy`/`scipy`; adding a matching resampler is future
  work.
- PCM 8/16/24/32-bit and 32-bit float WAV are supported; 64-bit float and
  compressed formats are rejected.
- Path resolution: absolute `oto_file` or `singer_path`-relative join only. The
  Python loader also searches the voicebank and falls back to compensation
  phonemes; Rust emits ~100 ms of silence with a stderr warning instead.

### Error handling / robustness

- Bad JSON and missing pitch curves return errors; the server maps them to
  400/500 instead of panicking.
- A missing/unreadable phone wav degrades to silence (Python behaviour), but
  there is no structured warning channel yet (stderr only).
- Allocation guards: phrase <= 600,000 ms, per-phone target <= 200,000 mel
  frames, source file <= 500,000 hop-128 frames. HTTP body size uses axum's
  default limit (2 MiB), which is above the observed request size (~42 KB) but
  not configurable yet.
- No request timeout, cancellation, or worker pool. One `RenderEngine` behind
  a mutex serializes concurrent renders; Python has `max_workers=2` but its
  per-render model use is not trivially parallel either.
- Server has no auth/CORS; it binds loopback by default.

### Config

- `AudioConfig` defaults mirror `docs/config.yaml` (44.1 kHz, hop 512/128,
  n_fft 2048, 128 mels, 40-16000 Hz). The YAML is not parsed; changing those
  values requires a code change. Model paths are CLI flags / env vars
  (`DIFFICAL_ASSEMBLER`, `DIFFICAL_VOCODER`).
- `DIFFICAL_SEG_NORM` is honored; other Python environment switches
  (`DIFFICAL_VOCODER_PAD`, `DIFFICAL_SEAM_RAMP`, stretch debug) are not.

### Parity coverage

- Gates cover one 10-phone, 1.7 s request. There is no batch harness over
  `listen_test/captured` yet.
- Some float rounding uses Rust `round()` (half away from zero) where Python
  uses banker's rounding (`np.round`, `round`). This only matters for exact
  `.5` frame/sample boundaries; no observed request hits it.
- Golden generation depends on Python; the Rust runtime does not.
