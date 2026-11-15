import os
import numpy as np


def normalize_phrase_rms(audio: np.ndarray,
                         sample_rate: int = 44100,
                         target_rms: float = 0.18,
                         peak_ceiling: float = 0.95,
                         activity_floor_db: float = -45.0) -> tuple[np.ndarray, dict]:
    """Apply one constant gain to a phrase; preserve every internal level ratio."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("phrase audio must be mono")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if (not np.isfinite(target_rms) or target_rms <= 0
            or not np.isfinite(peak_ceiling) or peak_ceiling <= 0
            or not np.isfinite(activity_floor_db)):
        raise ValueError("normalization settings must be finite and positive")
    if not np.isfinite(x).all():
        raise ValueError("phrase audio must be finite")
    if x.size == 0:
        return x.copy(), {"reason": "empty", "gain": 1.0}
    peak = float(np.max(np.abs(x)))
    if peak <= 1e-12:
        return x.copy(), {"reason": "silent", "gain": 1.0, "peak_before": peak}
    window = max(1, int(round(sample_rate * 0.020)))
    hop = max(1, int(round(sample_rate * 0.010)))
    starts = (np.asarray([0], dtype=np.int64) if len(x) <= window else
              np.arange(0, len(x) - window + 1, hop, dtype=np.int64))
    if len(x) > window:
        final_start = len(x) - window
        if starts.size == 0 or starts[-1] != final_start:
            starts = np.append(starts, final_start)
    frame_rms = np.asarray([np.sqrt(np.mean(x[s:s + window].astype(np.float64) ** 2))
                            for s in starts])
    threshold = max(1e-5, peak * 10.0 ** (activity_floor_db / 20.0))
    active = frame_rms >= threshold
    if not np.any(active):
        return x.copy(), {"reason": "no_active_windows", "gain": 1.0,
                          "peak_before": peak, "activity_threshold": threshold}
    measured = float(np.sqrt(np.mean(frame_rms[active] ** 2)))
    requested_gain = float(target_rms / max(measured, 1e-12))
    peak_gain = float(peak_ceiling / peak)
    gain = min(requested_gain, peak_gain)
    normalized = (x.astype(np.float64) * gain).astype(np.float32)
    return normalized, {
        "reason": "peak_limited" if gain < requested_gain else "target_reached",
        "gain": gain, "gain_db": float(20.0 * np.log10(max(gain, 1e-12))),
        "active_rms_before": measured, "active_rms_after": float(measured * gain),
        "target_rms": float(target_rms), "activity_threshold": threshold,
        "peak_before": peak, "peak_after": float(np.max(np.abs(normalized))),
        "active_windows": int(np.count_nonzero(active)), "total_windows": int(len(frame_rms)),
    }


def _periodic_vowel_rms(x, sample_rate, frame_ms, f0, frame_ids, sample_start,
                        sample_end, expression_gain):
    """Measure sustained periodic windows, never raise/flatten their individual samples."""
    values = []
    for frame in frame_ids:
        lag = max(1, int(round(sample_rate / f0[frame])))
        length = max(int(.030 * sample_rate), 3 * lag)
        center = int(round((frame + .5) * frame_ms * sample_rate / 1000.))
        a = max(sample_start, center - length // 2)
        b = min(sample_end, center + length // 2)
        if b - a < max(128, 2 * lag):
            continue
        segment = x[a:b].astype(np.float64)
        if expression_gain is not None:
            # De-express ONLY the measurement. A centre-frame division would
            # misread a steep ATK/DEC ramp within this 30ms analysis window.
            gains = np.interp(np.arange(a, b) * 1000. / sample_rate,
                              (np.arange(len(expression_gain)) + .5) * frame_ms, expression_gain)
            if np.any(gains < .01):
                continue
            segment /= gains
        centered = segment - np.mean(segment)
        correlations = []
        for candidate in (max(1, lag-1), lag, lag+1):
            left, right = centered[:-candidate], centered[candidate:]
            norm = np.sqrt(np.dot(left, left) * np.dot(right, right))
            correlations.append(float(np.dot(left, right) / max(norm, 1e-20)))
        if max(correlations) < .5:
            continue  # closure/frication/noise is not a vowel loudness teacher
        rms = float(np.sqrt(np.mean(segment * segment)))
        if rms > 1e-5:
            values.append(rms)
    if len(values) < 2:
        return None, len(values)
    values = np.asarray(values)
    low, high = np.quantile(values, [.4, .9])
    stable = values[(values >= low) & (values <= high)]
    if not len(stable):
        stable = values
    return float(np.sqrt(np.mean(stable * stable))), len(values)


def match_note_loudness(audio: np.ndarray, notes: list[dict], f0: np.ndarray,
                        sample_rate: int, frame_ms: float,
                        *, max_gain_db: float = 15.0,
                        voiced_region: tuple[float, float] = (.35, .75),
                        periodic_vowels: bool = False,
                        expression_gain: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Match steady voiced note levels, then preserve OpenUtau VOL explicitly.

    The renderer opts into periodic vowel windows across 10-95% of each note;
    the legacy fixed-window path remains for isolated level-matching callers.
    De-express measurement with the structural mix gain, never the audio.
    Closures/noise and user attack/release are not normalization targets. The median
    intrinsic note RMS is the reference; the desired output level is that
    reference times this note's OpenUtau VOL. One bounded scalar is applied
    per note, with a short boundary-only ramp. This is not compression or a
    moving loudness envelope and never edits F0 or spectrum.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('audio must be finite mono')
    if sample_rate <= 0 or frame_ms <= 0:
        raise ValueError('invalid sample timing')
    if (not np.isfinite(max_gain_db) or max_gain_db < 0
            or len(voiced_region) != 2
            or not np.isfinite(voiced_region).all()
            or not 0.0 <= voiced_region[0] < voiced_region[1] <= 1.0):
        raise ValueError('invalid loudness matching bounds')
    if len(x) == 0:
        return x.copy(), {'reason': 'empty', 'notes': []}
    if float(np.max(np.abs(x))) <= 1e-12:
        return x.copy(), {'reason': 'silent', 'notes': []}
    if not notes or len(f0) == 0:
        return x.copy(), {'reason': 'no_notes_or_f0', 'notes': []}

    hop_samples = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    f0 = np.asarray(f0, dtype=np.float64)
    n_frames = int(np.ceil(len(x) / hop_samples))
    if len(f0) < n_frames:
        f0 = np.pad(f0, (0, n_frames - len(f0)))
    f0 = f0[:n_frames]
    max_gain = 10.0 ** (float(max_gain_db) / 20.0)
    min_gain = 1.0 / max_gain

    if expression_gain is not None:
        expression_gain = np.asarray(expression_gain, dtype=np.float64)
        if expression_gain.ndim != 1 or not len(expression_gain) or not np.isfinite(expression_gain).all() or np.any(expression_gain < 0):
            raise ValueError('expression_gain must be finite nonnegative frame gains')
    measured = []
    for index, note in enumerate(notes):
        start, end = float(note['start_ms']), float(note['end_ms'])
        if not np.isfinite(start + end) or end <= start:
            continue
        span = end - start
        lo = start + span * float(voiced_region[0])
        hi = start + span * float(voiced_region[1])
        f0_start = max(0, int(np.floor(lo / frame_ms)))
        f0_end = min(n_frames, int(np.ceil(hi / frame_ms)))
        frame_ids = np.arange(f0_start, f0_end)
        frame_ids = frame_ids[f0[frame_ids] > 50.0]
        if periodic_vowels:
            centers = (frame_ids + .5) * frame_ms
            frame_ids = frame_ids[(centers >= lo) & (centers < hi)]
        if len(frame_ids) == 0:
            continue
        sample_start = max(0, int(np.floor(lo / 1000 * sample_rate)))
        sample_end = min(len(x), int(np.ceil(hi / 1000 * sample_rate)))
        if sample_end <= sample_start + max(128, hop_samples):
            continue
        voiced = np.zeros(sample_end - sample_start, dtype=bool)
        for frame in frame_ids:
            a = max(sample_start, frame * hop_samples) - sample_start
            b = min(sample_end, (frame + 1) * hop_samples) - sample_start
            if b > a:
                voiced[a:b] = True
        values = x[sample_start:sample_end][voiced]
        if values.size == 0:
            continue
        rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
        volume = float(note.get('volume', 1.0))
        if not np.isfinite(volume):
            volume = 1.0
        volume = float(np.clip(volume, 0.0, 2.0))
        periodic_count = None
        if periodic_vowels:
            rms, periodic_count = _periodic_vowel_rms(
                x, sample_rate, frame_ms, f0, frame_ids, sample_start, sample_end, expression_gain)
            if rms is None:
                continue
            if expression_gain is not None:
                # Compare intrinsic levels, then retain VOL in the existing
                # target calculation. ATK/DEC are removed only from measurement,
                # not from audio, so matching cannot cancel a user's expression.
                rms *= volume
        if rms <= 1e-5:
            continue
        measured.append({'index': index, 'rms': rms, 'volume': volume,
                         'start_ms': start, 'end_ms': end,
                         'measurement_interval_ms': [float(lo), float(hi)],
                         'periodic_vowel_windows': periodic_count})

    if len(measured) < 2:
        return x.copy(), {'reason': 'insufficient_voiced_notes', 'notes': measured}

    # Only positive-VOL measurements teach the target. Legacy measured zero-VOL
    # entries use zero gain; periodic envelope-zero windows are skipped above.
    reference_notes = [m for m in measured if m['volume'] > 1e-6]
    if len(reference_notes) < 2:
        return x.copy(), {'reason': 'insufficient_nonzero_volume_notes', 'notes': measured}
    intrinsic = np.asarray([m['rms'] / m['volume'] for m in reference_notes])
    target = float(np.median(intrinsic))
    entries = []
    for m in measured:
        desired = target * m['volume']
        raw_gain = desired / max(m['rms'], 1e-12)
        gain = 0.0 if m['volume'] <= 1e-6 else float(np.clip(raw_gain, min_gain, max_gain))
        entries.append(dict(m, gain=gain,
                            reason=('volume_zero' if m['volume'] <= 1e-6 else
                                    'gain_limited' if gain != raw_gain else 'matched')))

    # 相邻音符增益差限幅 + 更长的升余弦过渡，避免 10ms 内大增益跳变造成抽气声。
    ramp_ms = float(os.environ.get("DIFFICAL_NOTE_RAMP_MS", "80") or 80)
    cap_db = float(os.environ.get("DIFFICAL_NOTE_MAX_STEP_DB", "2.0") or 0)
    ordered_pre = sorted(entries, key=lambda item: item['start_ms'])
    if cap_db > 0:
        step = 10.0 ** (cap_db / 20.0)
        for i in range(1, len(ordered_pre)):
            prev_g = float(ordered_pre[i - 1]['gain'])
            gain = float(ordered_pre[i]['gain'])
            if prev_g > 0:
                gain = min(max(gain, prev_g / step), prev_g * step)
            ordered_pre[i]['gain'] = gain

    # Apply only within note spans; rests stay untouched. Adjacent-note gain
    # changes are interpolated over a longer raised-cosine window.
    curve = np.ones(len(x), dtype=np.float64)
    ordered = sorted(entries, key=lambda item: item['start_ms'])
    bounds = []
    for entry in ordered:
        start = max(0, int(np.floor(entry['start_ms'] / 1000 * sample_rate)))
        end = min(len(x), int(np.ceil(entry['end_ms'] / 1000 * sample_rate)))
        if end > start:
            curve[start:end] = entry['gain']
        bounds.append((entry, start, end))
    for (prev, _ps, prev_end), (curr, curr_start, _ce) in zip(bounds[:-1], bounds[1:]):
        if prev_end <= 0 or curr_start >= len(x):
            continue
        if curr_start <= prev_end:
            boundary = curr_start
            left_span = max(1, prev_end - _ps)
            right_span = max(1, _ce - curr_start)
            half = min(int(ramp_ms / 1000 * sample_rate), left_span // 3, right_span // 3)
            if half > 0:
                left, right = max(0, boundary - half), min(len(x), boundary + half)
                if right > left:
                    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, right - left))
                    curve[left:right] = prev['gain'] + (curr['gain'] - prev['gain']) * w
        else:
            # Restore unity through an actual rest; do not interpolate note
            # correction across silence or into the next note.
            ramp = min(int(ramp_ms / 2000 * sample_rate), (prev_end - _ps) // 4)
            if ramp > 0:
                a, b = max(_ps, prev_end-ramp), prev_end
                curve[a:b] = np.linspace(prev['gain'], 1.0, b-a, endpoint=False)
            ramp = min(int(ramp_ms / 2000 * sample_rate), (_ce-curr_start) // 4)
            if ramp > 0:
                a, b = curr_start, min(_ce, curr_start+ramp)
                curve[a:b] = np.linspace(1.0, curr['gain'], b-a, endpoint=True)

    out = (x.astype(np.float64) * curve).astype(np.float32)
    return out, {'reason': 'matched', 'target_intrinsic_rms': target,
                 'max_gain_db': max_gain_db, 'notes': entries,
                 'f0_unchanged': True,
                 'measurement': 'periodic_vowels' if periodic_vowels else 'legacy_fixed_window'}
