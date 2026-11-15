"""Recording-level matching before OpenUtau expression/envelope application."""
from __future__ import annotations
import numpy as np


def match_source_levels(features: list[dict], max_gain_db: float = 9.0) -> list[dict]:
    """Apply one constant magnitude gain per recorded phone, never a time curve.

    PhoneProcessor measures a voiced source reference before time warping or
    envelope gating. Adjacent aliases often come from differently leveled takes.
    Use the median source RMS as reference, excluding absent/quiet/unvoiced
    references. VOL/ATK/DEC and DYN are not inputs to the measurement and thus
    remain authoritative downstream. The log-mel spectrum receives only a
    uniform offset; timing, spectral ratios, and pitch stay unchanged.
    """
    if not np.isfinite(max_gain_db) or max_gain_db < 0:
        raise ValueError('max_gain_db must be finite and nonnegative')
    eligible = [(i, float(f.get('source_level_rms', 0.0)))
                for i, f in enumerate(features)
                if f.get('source_level_valid', False)
                and np.isfinite(f.get('source_level_rms', 0.0))
                and float(f.get('source_level_rms', 0.0)) > 1e-5]
    if len(eligible) < 2:
        return []
    target = float(np.exp(np.median(np.log([value for _, value in eligible]))))
    rows = []
    for index, level in eligible:
        requested_db = 20.0 * np.log10(target / level)
        gain_db = float(np.clip(requested_db, -max_gain_db, max_gain_db))
        feature = features[index]
        gain = 10.0 ** (gain_db / 20.0)
        mel = np.asarray(feature['mel'], dtype=np.float32).copy()
        # Keep explicit padding silent, even for a loudness boost.
        content = np.any(mel > -60.0, axis=0)
        mel[:, content] += np.float32(np.log(gain))
        feature['mel'] = mel
        rows.append({'index': index, 'phoneme': feature['phone'].phoneme,
                     'source_rms': level, 'target_rms': target,
                     'gain': gain, 'gain_db': gain_db,
                     'gain_limited': abs(requested_db) > max_gain_db,
                     'reference_region': feature.get('source_level_region')})
    return rows


def measure_source_level(wave: np.ndarray, sr: int, offset_ms: float,
                         consonant_ms: float, end_ms: float,
                         raw_preutter_ms: float, raw_overlap_ms: float,
                         connected: bool) -> tuple[float, bool, list[float]]:
    """Measure recording level in an OTO-declared vowel, avoiding closures.

    Connected phones use their recorded pre-vowel; isolated/CV phones use the
    steady vowel after the fixed consonant. The upper half of 20ms window RMS
    rejects silence/closure windows without raising them in the output. There
    is no sample-wise gate, compressor, or frequency filter.
    """
    x = np.asarray(wave, dtype=np.float64)
    if x.ndim != 1 or sr <= 0:
        raise ValueError('invalid source waveform')
    if connected:
        # Connected/VC aliases carry the previous vowel before the oto
        # consonant boundary. Their measurable voiced material is the
        # preutter-to-consonant span, not the post-consonant CV body.
        vowel_end = offset_ms + min(max(0.0, raw_preutter_ms), consonant_ms)
        vowel_span = max(0.0, vowel_end - offset_ms)
        left = offset_ms + .20 * vowel_span
        right = offset_ms + .80 * vowel_span
    else:
        # CV/initial aliases carry their note vowel after the consonant mark.
        vowel_start = offset_ms + consonant_ms
        vowel_span = max(0.0, end_ms - vowel_start)
        left = vowel_start + .20 * vowel_span
        right = vowel_start + .70 * vowel_span
    a = max(0, int(np.ceil(left * sr / 1000)))
    b = min(len(x), int(np.floor(min(right, end_ms) * sr / 1000)))
    region = [float(left), float(min(right, end_ms))]
    if b - a < round(.050 * sr):
        return 0.0, False, region
    window = max(1, round(.020 * sr))
    hop = max(1, round(.010 * sr))
    if b - a < window:
        return 0.0, False, region
    rms = np.asarray([np.sqrt(np.mean(x[s:s+window]**2))
                      for s in range(a, b-window+1, hop)])
    active = rms[rms > max(1e-5, float(np.max(rms)) * .1)]
    if not len(active):
        return 0.0, False, region
    typical = active[active >= np.median(active)]
    return float(np.sqrt(np.mean(typical**2))), True, region
