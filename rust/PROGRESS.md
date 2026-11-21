# Rust rewrite progress

Last updated: 2026-11-21. Scope: `rust/` only; Python backend untouched.

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
parsing / voicebank search (the host editor already supplies resolved `oto_*`), phase
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

### Flags (`flags.rs`, `FLAGS.md`)

- Parser: first occurrence wins, case-sensitive names, optional values, values
  clamped to the documented range, non-finite values fall back to the default
  (never an error/panic). Unknown names are ignored and listed by `inspect`.
- Active: `g` (key-shift mel), `t` (F0 cents), `Hb`/`Hv` (HN-SEP mix), `Ht`
  (documented spectral tilt), `HG` (growl), `A` (pitch-following amplitude),
  `He` (long-note source-vowel loop).
- No-op with one-line reasons: `fe fl fo fv fp ve vo B S p R D C Z G P`.
- `--inject-flags "Hb200g100"` replaces every phone's flags for tests; the same
  parser rules are implemented in `tools/dump_flag_golden.py`.
- `t` is applied to both the per-phone F0 slice and the phrase F0 curve (later
  phone wins when phones overlap); `g`/`Hb`/`Hv`/`Ht`/`He` never touch F0 and
  `HG`/`A` never touch the mel.

### HN-SEP (`hnsep.rs`, `export/hnsep.onnx`)

- `tools/export_hnsep.py` exports CascadedNet as a real/imag spectral mask
  network (opset 17, input `(1,2,1025,T)`, output `mask`, dynamic `T`). The
  model's decoder assumes 32-aligned frame counts, which the reference
  whole-segment padding guarantees.
- `hnsep.rs` reimplements `torch.stft`/`torch.istft` semantics (periodic Hann,
  center=True, constant pad, onesided, window-squared envelope normalization)
  and calls the ONNX mask. Frames are always 32-aligned.
- Loaded lazily from `--hnsep`, `$DIFFICAL_HNSEP`, or `hnsep.onnx` next to the
  assembler. Only `Hb`/`Hv` renders load it; `Ht`-only renders use the tilt
  without separation, and the flag-free path is byte-identical with or without
  the option (verified with `cmp` on two rendered wavs).
- Separations are cached in-process (`HnsepCache`): key = canonical path +
  mtime + half-open sample window `[start, end)` + sample rate, LRU bound
  64 entries / 256 MiB, mutex-protected, `harmonic`/`noise` shared as
  `Arc<[f32]>`. Per-phone mixing runs after the cached separation.
- Two window strategies share that cache and the same per-phone path:
  - **whole-file (default)**: `[0, len)` per source, exactly what the reference
    does, so flag parity is exact; 12 misses on the 1.6 s fixture, repeat
    render 12 hits with byte-identical HTTP output.
  - **window (`DIFFICAL_HNSEP_WINDOW=1`)**: separates only the phone's stretch
    window `[start_ms, end_ms]` plus one STFT window of context on each side
    (`DIFFICAL_HNSEP_GUARD`, default 2048 samples). This makes the first render
    fast (12 small separations, 1.06 s total) but the mask network's global
    context differs from the whole-file reference, so per-phone mel drifts even
    with a 1 s guard (measured 0.22-0.33 max). Wav parity stays inside the
    standing gate (Hb200 NCC 0.999918, RMS ratio 0.998903); the default stays
    whole-file because the mel gate is exact there.
- A render that needs HN-SEP without a model path fails with a clear error
  (HTTP 500 on `serve`) instead of silently rendering the neutral voice.

### Flag effects (`effects.rs`, `biquad.rs`)

- `HG` growl reimplemented end to end (400 Hz 4th-order high-pass, square LFO
  following 0.25*F0, +/-100 cent band modulation, drift high-pass, RMS restore).
- `Ht` tilt: 2nd-order Butterworth high-pass (positive) or low-pass (negative)
  at `150 + 250*|Ht|/100` Hz, crossfaded by `|Ht|/100`.
- `A`: frame gain `1 + (A/100)*0.5*clamp((d log2 f0/dt)/2, -1, 1)`, followed
  by an RMS-restoring scalar gain and a peak limiter at the pre-`A` peak, so the
  stage changes the envelope rather than the phrase level.
- `He`: ping-pong loop over 35-92 % of the source vowel, active only where the
  long-note hold applies (target vowel > 2.5x source, source vowel >= 200 ms).
- Butterworth biquads match scipy `butter(2|4)` responses; processing in f64.

### Tooling / parity

- `tools/dump_phone_golden.py` dumps per-phone mel/F0/source/mel_orig,
  phrase F0/assembled mel, assembler inputs, reference wav and normalization
  trace. It records the hop-128 analyzer input by wrapping the analyzer call,
  so the backend code is not modified.
- `tools/dump_flag_golden.py` renders a request with a flags string injected on
  every phone (or neutralizes curves) and dumps per-phone mel/F0 + wav.
- `tools/hnsep_parity.py` compares torch and ONNX Runtime HN-SEP harmonics on a
  real 1.6 s wav and dumps the reference for the Rust check.
- `tools/flag_parity.sh` drives the whole per-flag matrix.
- Commands: `inspect`, `phone-parity` (+`--hnsep`), `assemble-parity`
  (+`--hnsep`), `e2e-parity` (+`--hnsep`), `hnsep-parity`, `flag-parity`,
  `response-parity`, `render` (+`--hnsep`), `serve` (+`--hnsep`).
- `cargo test --workspace`: 33 unit tests pass (protocol/key conversion/clamping,
  flag parsing and ranges, `np.interp` endpoints, biquad responses, tension
  tilt, growl RMS, amplitude slope, `He` loop continuity, wav loading,
  sample-rate rejection).

## Measured parity (1.6 s, 12-phone request)

The fixture request that earlier revisions used had been deleted from `/tmp`;
the 2026-11-21 set was regenerated from
`/tmp/diffical_requests/req_-725f2e9940438609.json` (copied to
`golden/flag_request.json`). The request swap is the only reason these absolute
numbers differ from the previous revision.

| Gate | Metric | Result |
| --- | --- | --- |
| mel frontend (512 hop) | max abs | 9.060e-5 |
| assembler ONNX tensors | max abs | 9.060e-6 |
| per-phone mel | max abs | 3.147e-4 (mean 1.373e-6) |
| phrase F0 | max abs | 6.104e-5 |
| assembler input `segs` | max abs | 3.147e-4 |
| `mel_phrase` (128x139) | max abs | 2.985e-4 (mean 2.049e-6) |
| waveform (71129 samples) | max abs | 1.255e-4 |
| waveform | NCC | 0.999999997 |
| waveform | RMS ratio | 1.000018 |
| HTTP response body | NCC / RMS | 0.999999997 / 1.000009 |
| HN-SEP torch vs ONNX Runtime | max abs | 5.960e-8 |
| HN-SEP Rust chain vs torch | max abs | 9.313e-8 |

Normalization gain: Rust 2.961169965 vs Python 2.961115501.

Flag matrix (`tools/flag_parity.sh`; mel gate 1e-3, wav NCC > 0.99, RMS ratio
within 10 %):

| Case | mel max | wav NCC | RMS ratio |
| --- | --- | --- | --- |
| `g100` | 6.027e-4 | 0.999999997 | 1.000017 |
| `t-50` | 3.147e-4 | 0.008428 | 1.039107 |
| `Hb200` | 2.775e-4 | 0.999999997 | 0.999993 |
| `Hv120` | 5.226e-4 | 0.999999997 | 1.000018 |
| `Ht30` | 5.535e-1 | 0.998473 | 0.972818 |
| `HG50` | 3.147e-4 | 0.997793 | 1.012626 |
| `A100` | 3.147e-4 | 0.981644 | 0.996770 |
| `He` (short notes) | 3.147e-4 | 0.999999997 | 1.000018 |
| `He` (1.1 s note) | 1.990 | 0.994611 | 0.999996 |
| `Hb250Hv90g100` | 6.533e-4 | 0.999999997 | 1.000003 |
| `t20HG40` | 3.147e-4 | -0.158090 | 1.009074 |

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

Flagged renders: `Hb`/`Hv` add an HN-SEP pass on a cache miss. Measured on the
12-source, 1.6 s Hb200 phrase through `serve` (same process, same port):

| Mode | Render | wall | phone stage | assemble | vocode |
| --- | --- | --- | --- | --- | --- |
| whole-file (default) | first Hb200 (12 misses, incl. lazy model load) | 22.44 s | 21.77 s | 0.14 s | 0.52 s |
| whole-file (default) | second Hb200 (12 hits) | 0.34 s | 0.16 s | 0.004 s | 0.17 s |
| whole-file (default) | no-flag baseline | 0.32 s | 0.14 s | 0.004 s | 0.16 s |
| window (`DIFFICAL_HNSEP_WINDOW=1`) | first Hb200 (12 window misses + load) | 4.20 s | 3.77 s | 0.06 s | 0.36 s |
| window | second Hb200 (12 hits) | 0.33 s | 0.15 s | 0.004 s | 0.17 s |
| window | no-flag baseline | 0.32 s | 0.15 s | 0.003 s | 0.16 s |

Window-mode first-render split: HN-SEP model load 2.54 s, 12 window
separations 1.06 s total (39-105 ms each, 0.6-2.3 MB of audio each), mixing +
mel + stretch 0.17 s; wall 4.20 s includes HTTP/JSON. Whole-file cache
footprint after the fill: 21.2 MB for 12 files; window-mode cache: 1.6 MB.
Process RSS: 0.31 GB after models, 1.14 GB after the first whole-file render,
0.87-0.90 GB after window-mode renders (the mask-network ORT session/arena
dominates, not the cache). `Ht`-only renders do not need the HN-SEP model and
stay at the unflagged speed.

The first server request after startup is ~0.9-1.0 s (lazy ORT allocation).
The dense mel filterbank projection was initially ~1.9 s; using per-band
non-zero spans and frame-major magnitudes brought the phone stage to ~0.1 s.
During later reruns an unrelated `python -m backend.server` process (started
before this session) held ~4 cores; wall times rose to 0.8-1.1 s with identical
parity output. Latency numbers on this shared machine should be treated as
indicative, not a benchmark.

## Not done / remaining gaps

### Flags and curves

- Expression curves (breathiness, tension, voicing, gender, growl, lowcut,
  brel, breh, warm, hcmp, tone_shift) are still ignored. Flag parity requests
  must use neutral curves; `dump_flag_golden.py` warns and can neutralize them.
  `tone_shift` remains a real audio-path gap: a non-empty SHFT curve will be
  quieter/different than the Python render. `dynamics` is intentionally ignored
  because the Python backend applies DYN during playback, not in the render.
- `A`'s peak limiter can flatten the loudest samples when the shaped peak would
  exceed the pre-`A` peak; this is documented and measured (final RMS change
  0.32 % at A=100), but it is a deliberate level guarantee rather than a pure
  gain.
- `Ht` deliberately diverges: the Python reference's dynamic filter factor is
  `clip(-2 * (Ht + 100) / 100, 0, 1)` which is zero for every accepted `Ht`, so
  its `Ht` renders are unchanged. Rust applies the documented tilt instead
  (mel diff 5.535e-1, wav NCC 0.998473, RMS ratio 0.972818 vs the reference).
- `t`, `A` and `He` are Rust-only semantics (the reference ignores them); their
  per-phone mel matches the reference exactly and the wav differs on purpose.
- `g`/key-shift mel accuracy is the tightest numerical part: single `g` cases
  measure 5.2e-4..8.5e-4 against the 1e-3 gate, and a combined
  `Hb250Hv90g60` case measured 1.358e-3 (just over the per-flag gate) while the
  same combination with `g100` measured 6.533e-4. The residual is the f32 FFT
  difference on near-silent frames, amplified by the log; the waveform is
  unaffected (`Hb250Hv90g60` e2e NCC was still 1.0 in the e2e check).
- A value-less valued flag falls back to its default. The Python reference's
  accidental -6 dB result for a value-less `Hb` is not reproduced.
- Flags do not affect the request-wide curve arrays or the reference's
  phrase-level HN-SEP block; only the per-phone stages listed in `FLAGS.md`.

### HN-SEP

- The default whole-file mode still pays ~1.8 s per 5 s source on the first
  render: the cache only removes repeat work. Window mode
  (`DIFFICAL_HNSEP_WINDOW=1`) cuts that to ~1.1 s of separation but cannot match
  the reference's per-phone mel (<1e-3) because the reference separates the
  whole file; see the measured drift above. The default therefore keeps exact
  parity.
- On macOS the ORT runtime's automatic device policy registers the CoreML EP
  even when the session requests CPU; this writes a 33 MB compiled model to the
  system temp dir per session and can fail with `SaveModel ... I/O error` when
  the disk is nearly full (`hnsep load` 2.5 s is dominated by this). Setting
  `AutoDevicePolicy::PreferCPU` or cleaning the temp dir is future work; the
  build here is unchanged.
- A source set larger than 64 entries / 256 MiB evicts LRU entries.
- The mask-network session dominates peak memory (~1.14 GB RSS after the first
  flagged render, cache 21.2 MB); an arena-limit or provider tweak was not
  attempted because the parity target is the reference CPU path.
- Cache keys use path + mtime + samples + rate; a file rewritten with the same
  size and mtime (rare) would hit a stale entry.
- Only the mask network is exported; STFT/ISTFT stay in Rust. The graph assumes
  32-aligned frame counts and the fixed 2048/512 geometry (44.1 kHz only).
- The ONNX file is ~59 MB in `export/`; it is not committed and is regenerated by
  `tools/export_hnsep.py`.
- No fallback separation path: if the model is missing, HN-SEP renders fail
  (documented) rather than degrade.

### Parity coverage

- The gates cover one 1.6 s, 12-phone request plus one 1.6 s long-note request
  for `He`. There is no batch harness over `listen_test/captured` yet.
- Golden generation depends on Python; the Rust runtime does not.
- Some float rounding uses Rust `round()` (half away from zero) where Python
  uses banker's rounding (`np.round`, `round`). This only matters for exact
  `.5` frame/sample boundaries; no observed request hits it.

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
- `DIFFICAL_HNSEP` is honored (same as `--hnsep`); the default is `hnsep.onnx`
  next to the assembler.
- `DIFFICAL_SEG_NORM` is honored; other Python environment switches
  (`DIFFICAL_VOCODER_PAD`, `DIFFICAL_SEAM_RAMP`, stretch debug) are not.


## HN-SEP default changed to per-window

- Default: per-window separation, cache keyed by (path, mtime, start, end).
  12-source Hb200 fixture: first render 4.20 s (model load 2.54 s + 12 window
  separations 1.06 s), second render 0.33 s, RSS 868 MB.
- Whole-file mode kept behind `DIFFICAL_HNSEP_WHOLE=1`; it is the Python
  reference behaviour and is what `flag-parity` runs. First render 22.4 s,
  second 0.34 s, RSS 1.14 GB.
- Whole vs window output: NCC 0.99991761, RMS ratio 0.998910, short-time RMS
  correlation 0.999998839. Per-phone mel max 0.26 (edge frames), so mel parity
  is only asserted in whole-file mode.
- A=100 after RMS compensation: final RMS ratio 0.996770 vs no-A, peak 0.95
  unchanged, NCC 0.98164.
