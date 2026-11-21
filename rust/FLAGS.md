# Flag support

The Rust engine parses the standard resampler flag set carried by
`request.phones[*].flags` (entries are `[name, value?, abbr?]`). This document
is the contract: every name is either **active** with the effect described here,
or **no-op** with the reason it has no stage in this engine.

Parsing rules:

- Names are case-sensitive (`Hb` is not `hb`).
- The first occurrence of a name wins; later duplicates are ignored.
- A value is optional. A missing, `null` or non-finite (`NaN`/`Inf`) value falls
  back to that flag's default; it never errors or panics.
- Values are clamped to the documented range. Unknown names are ignored (and
  reported by `diffical-cli inspect` as `ignored=[...]`, e.g. the bare `H` some
  projects send).
- The same rules apply to `--inject-flags` strings such as
  `"Hb200Hv120Ht30g100Dt-50He"` (used by `flag-parity`).

## Table

| Flag | Range | Default | Stage | Effect | Status |
| --- | --- | --- | --- | --- | --- |
| `g` | -600..600 | 0 | source mel (hop 128), per phone | formant shift `key_shift = clip(g, -100, 100) / 100` semitones; 0 = neutral | active |
| `t` | -1200..1200 | 0 | phrase F0, per phone | adds `t` cents to the phone's voiced F0 frames (`f0 > 0`) | active |
| `Hb` | 0..500 | 100 | source waveform, per phone | breath mix: `breath = clip((Hb-100)/5, 0, 100)`, `(1-breath/100)*harmonic` | active |
| `Hv` | 0..150 | 100 | source waveform, per phone | voiced mix: `voicing = clip(Hv/1.5, 0, 100)`, `(voicing/100)*noise`; absent/valueless = neutral | active |
| `Ht` | -100..100 | 0 | source waveform, per phone | dynamic spectral tilt, strength `|Ht|/100` (see below) | active |
| `HG` | 0..100 | 0 | phrase waveform, after vocoding | growl: square-LFO pitch modulation of the >400 Hz band | active |
| `A` | -100..100 | 0 | phrase waveform, after growl | level-preserving amplitude modulation from the local F0 slope | active |
| `He` | switch | absent | source-time vowel mapping | ping-pong source-vowel loop instead of the long-note hold | active |
| `G` | switch | absent | - | no cache exists, so nothing to regenerate | no-op |
| `P` | 0..100 | - | - | no per-note output makeup; only one phrase RMS normalization | no-op |
| `fe` | value or switch | - | - | fade-in length: the request envelope + 80 ms tail + phrase edges own fades | no-op |
| `fl` | value or switch | - | - | source loop fade: sources are used once and stretched, there is no loop | no-op |
| `fo` | value or switch | - | - | fade-out length: the envelope owns the release | no-op |
| `fv` | value or switch | - | - | fade-in variant: the envelope owns the attack | no-op |
| `fp` | value or switch | - | - | pre-utterance fade: preutterance comes from the request's oto fields | no-op |
| `ve` | value or switch | - | - | volume envelope: level comes from `phone.volume` and one phrase RMS pass | no-op |
| `vo` | value or switch | - | - | voice colour: timbre is the voicebank/model output, no colour stage | no-op |
| `B` | value or switch | - | - | legacy breath amount; breath is `Hb` in this engine, applying both would double count | no-op |
| `S` | value or switch | - | - | consonant speed/stretch: timing already comes from oto fields + velocity | no-op |
| `p` | value or switch | - | - | phoneme/preutterance position: supplied per phone by the request | no-op |
| `R` | value or switch | - | - | legacy quality/mode switch; the engine has a single model path | no-op |
| `D` | value or switch | - | - | legacy tuning switch with no model-path counterpart | no-op |
| `C` | value or switch | - | - | legacy tuning switch with no model-path counterpart | no-op |
| `Z` | value or switch | - | - | legacy tuning switch with no model-path counterpart | no-op |

## Order of application

1. Source wav load + peak guard (`pre_scale`).
2. `Hb`/`Hv` HN-SEP mixing and `Ht` tilt on the source waveform.
3. `g` formant shift in the hop-128 mel analysis.
4. `He` (or the default hold) in the vowel source-time mapping.
5. `t` on the per-phone F0 slices and the phrase F0 curve.
6. Assembler + vocoder.
7. `HG` growl, then `A` amplitude, on the phrase waveform.
8. Exact duration alignment, 5.8 ms edge fades, one phrase RMS normalization.

`t` never touches the mel; `g`/`Hb`/`Hv`/`Ht`/`He` never touch F0; `HG`/`A`
never touch the mel. This separation is what makes per-stage parity meaningful.

## Active flags

### `g` - formant shift

`key_shift = clip(g, -100, 100) / 100` semitones. The source mel is recomputed
with `factor = 2^(key_shift/12)`: window/FFT size `round(2048 * factor)`, the
magnitude spectrum is cropped or zero-padded back to the 1025-bin basis, and
amplitudes are compensated by `2048 / round(2048 * factor)`. `g = 0` uses the
normal analysis unchanged. The accepted range is +/-600, but (matching the
Python reference's mapping) the effective shift saturates at +/-1 semitone.

### `t` - pitch offset

The phone's voiced F0 frames are multiplied by `2^(t/1200)`. The shifted phrase
F0 feeds both the assembler and the vocoder, so the effect is audible pitch.
Unvoiced frames stay 0. Where phones overlap, the later phone owns the overlap
(it is the one whose content is assembled there). The mel is unaffected, so
`t` renders have identical per-phone mel to the reference.

### `Hb` / `Hv` - breath and voiced mix

HN-SEP separates the peak-guarded source waveform into `harmonic` and
`noise = wave - harmonic`; the phone wave becomes

```
processed = (1 - breath/100) * harmonic + (voicing/100) * noise
```

with `breath = clip((Hb - 100)/5, 0, 100)` and
`voicing = clip(Hv / 1.5, 0, 100)`. An absent or valueless flag leaves its mix
factor at 1.0, so the output is `wave` and the stage is skipped entirely. Note
the reference mapping: an *explicit* `Hv = 150` also gives 1.0, while an
explicit `Hv = 100` gives 0.667 (it lowers the noise mix). Separation runs only
when `breath > 0.5` or `|voicing - 100| > 0.5`; it requires
the HN-SEP mask network (`--hnsep`, `$DIFFICAL_HNSEP`, or `hnsep.onnx` next to
the assembler). A render that needs it without a model fails with a clear error
(HTTP 500 from `serve`).

Separations are cached in-process: the key is the canonical path, mtime, the
half-open sample window and the sample rate; the bound is 64 entries /
256 MiB (LRU) and the cache is mutex-protected (`harmonic`/`noise` shared as
`f32` buffers). By default the cached window is the whole file, which is what
the reference separates, so flag parity is exact: on a 12-source, 1.6 s phrase
the first flagged render separates 12 files (12 misses) and a repeat render
reuses all of them (12 hits, byte-identical output).

`DIFFICAL_HNSEP_WINDOW=1` caches per-phone windows instead: the stretch window
`[start_ms, end_ms]` plus `DIFFICAL_HNSEP_GUARD` samples of context (default
2048, one STFT window) on each side. This is a documented speed trade-off: the
first render drops from ~22 s to ~4.2 s (12 small separations, ~1.1 s), but the
mask network sees only the window, so its global context differs from the
reference and per-phone mel drifts (measured 0.20-0.26 max at the guard sizes
tried, versus 2.8e-4 whole-file). Wav gates still pass (Hb200 NCC 0.999918,
RMS ratio 0.998903). The default stays whole-file because the mel gate is only
met there.

### `Ht` - tension / spectral tilt

`strength = |Ht| / 100`, crossover
`cutoff = clamp(150 + 250 * strength, 60, 0.45 * sample_rate)` Hz. Positive `Ht`
crossfades toward a 2nd-order Butterworth high-pass, negative `Ht` toward the
complementary low-pass: `out = dry * (1 - strength) + filtered * strength`.
`Ht = 0` is exactly neutral and does not load the HN-SEP model.

Deliberate reference divergence: in the Python reference the same filter is
wired with a level of `-2 * tension` which is <= 0 for every accepted `Ht`, so
its filter factor clips to zero and `Ht` renders are unchanged there. This
engine implements the documented tilt instead. `Ht` still participates in the
HN-SEP trigger in the reference; here the separation is skipped when it would
be an identity mix.

### `HG` - growl

Reimplemented from the reference signal chain: 4th-order Butterworth high-pass
at 400 Hz, an LFO frequency of `clip(0.25 * f0, 30, 200)` Hz (80 Hz where
`f0 <= 50`), a square LFO from `sign(sin(cumsum(2*pi*f)/sr))`, band pitch
modulation of up to +/-100 cents scaled by `HG/100`, a 20 Hz high-pass on the
accumulated drift, `np.interp`-style resampling of the high band, and an RMS
restore of the modulated band. The low band passes through unchanged. The
per-phone level is the maximum over the phone's frames; levels below 0.5 are
inactive. `HG` changes only the wav, not the mel.

### `A` - pitch-following amplitude modulation

For each voiced frame the slope of `log2(f0)` is measured in octaves/second
(central difference, one-sided at the phrase edges, 0 for unvoiced frames) and
normalized as `clamp(slope / 2, -1, 1)`. The frame gain is
`1 + (A/100) * 0.5 * normalized`, interpolated linearly to samples and applied
after `HG`, before duration alignment and normalization.

The stage is level-preserving. After shaping, one scalar gain restores the
voiced-region RMS to its pre-`A` value, and the signal is then peak-limited to
the pre-`A` peak (the limiter only touches the loudest samples whose gain is
above 1). Both matter because the phrase RMS normalization below is
peak-limited on loud material; keeping RMS and peak equal keeps the final
phrase level equal. Measured on the flag-parity request: `A = 100` changed the
final wav RMS by 0.32 % (ratio 0.996770) instead of the 16 % drop of the
uncompensated version. `A = 0` is a no-op and the stage is skipped.

### `He` - long-note loop

Active only where the engine would otherwise use the long-note hold, i.e. when
the target vowel is longer than `2.5x` the source vowel and the source vowel is
at least 200 ms; otherwise `He` is a no-op. Instead of freezing near the end of
the source (the default hold), the sustain traverses the stable part of the
source vowel (`35 %..92 %` of the vowel span) back and forth with a ping-pong
position curve, one full cycle per `2 * max(source vowel, 1500 ms)`. Position is
continuous (no spectral jumps), the phone length and F0 are unchanged, and the
phrase RMS normalization keeps the level. Limits: no effect on consonant frames,
no effect on short notes, and at most the documented stable region is looped, so
a vowel whose formants move through the sustain is re-coloured rather than
extended. The Python reference does not implement `He`; the parity harness
verifies the no-trigger no-op on normal notes and measures the deliberate
difference on a 1.1 s sustained note.

## No-op flags

| Flag | Reason |
| --- | --- |
| `G` | There is no render cache, so there is nothing to invalidate. |
| `P` | The engine intentionally applies one phrase-level RMS normalization and no per-note output makeup. |
| `fe`, `fo`, `fv` | Fade lengths are owned by the request envelope, the 80 ms tail and the 5.8 ms phrase edges; a second fade would double-shape the signal. |
| `fl` | No source looping exists; sources are used once and source-time stretched. |
| `fp` | Pre-utterance timing comes from the request's oto fields, not from a fade flag. |
| `ve` | Level is `phone.volume` plus the phrase RMS pass; the flag is not a second gain stage. |
| `vo` | There is no voice-colour stage; timbre is the voicebank and model output. |
| `B` | Breath control is `Hb` in this engine; supporting `B` as well would double count. |
| `S` | Consonant timing is already resolved by the oto fields and velocity. |
| `p` | Phoneme position/preutterance is supplied per phone by the request. |
| `R`, `D`, `C`, `Z` | Legacy resampler tuning switches with no counterpart in a single model-based render path; parsed and ignored. |

## Verification

`tools/flag_parity.sh` regenerates the Python reference per flag and runs
`diffical-cli flag-parity` for each case. Measured on a 1.6 s, 12-phone request
(`golden/flag_request.json`) unless noted; mel gate 1e-3, wav gate NCC > 0.99 and
RMS ratio within 10 %:

| Case | per-phone mel max | wav NCC | RMS ratio | gate |
| --- | --- | --- | --- | --- |
| `g100` | 6.027e-4 | 0.999999997 | 1.000017 | mel |
| `t-50` | 3.147e-4 | 0.008428 (expected: pitch moved) | 1.039107 | mel |
| `Hb200` | 2.775e-4 | 0.999999997 | 0.999993 | mel |
| `Hv120` | 5.226e-4 | 0.999999997 | 1.000018 | mel |
| `Ht30` | 5.535e-1 (deliberate tilt) | 0.998473 | 0.972818 | wav |
| `HG50` | 3.147e-4 (mel untouched) | 0.997793 | 1.012626 | mel + wav |
| `A100` | 3.147e-4 (mel untouched) | 0.981644 | 0.996770 | mel |
| `He` (short notes) | 3.147e-4 | 0.999999997 | 1.000018 | mel + wav |
| `He` (1.1 s note) | 1.990 (deliberate loop) | 0.994611 | 0.999996 | wav |
| `Hb250Hv90g100` | 6.533e-4 | 0.999999997 | 1.000003 | mel + wav |

The `t` pitch shift is exact: the diagnostic F0 difference peaks at 15.76 Hz,
which is `553.7 * (1 - 2^(-50/1200))` at the top of that phrase's F0 curve.

## HN-SEP separation mode

`Hb`/`Hv`/`Ht` run the mask network on the source waveform before mel analysis.

- Default: per-window. Only the window the phone reads (plus one STFT window of
  guard on each side) is separated and cached by `(path, mtime, start, end)`.
  First flagged render on the 12-source fixture is about 4 s, later renders are
  cache hits.
- `DIFFICAL_HNSEP_WHOLE=1`: whole-file separation, no window guard. This is the
  mode the Python reference uses, so `flag-parity` exports it for exact
  comparison. It costs about 22 s on first render for the same fixture.
- The two modes differ in mask context: output waveform NCC 0.99992, RMS 0.11%.
  Their per-phone mel differs more (max 0.26) at window edges, which is why
  mel-level parity is checked in whole-file mode only.
