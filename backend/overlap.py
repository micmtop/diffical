"""OpenUtau envelope integration and fractional-hop mel placement."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


_GAIN_MIN = 0.0
_GAIN_MAX = 2.0


def _read_envelope(points: Sequence, first_phone: bool,
                   last_phone: bool) -> tuple[np.ndarray, np.ndarray]:
    parsed = []
    for point in points:
        if isinstance(point, Mapping):
            x, y = point["x"], point["y"]
        else:
            x, y = point
        x, y = float(x), float(y)
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("envelope points must be finite")
        parsed.append((x, y))
    if len(parsed) < 2:
        raise ValueError("envelope requires at least two points")

    xs = np.asarray([p[0] for p in parsed], dtype=np.float64)
    ys = np.asarray([p[1] for p in parsed], dtype=np.float64)
    if first_phone:
        ys[0] = 1.0
    if last_phone:
        ys[-1] = 1.0
    return xs, ys


def _collapse_duplicate_x(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(xs) > 1:
        # At a duplicate envelope coordinate OpenUtau's sequential envelope
        # reaches the last point at that coordinate; use its rightmost value.
        keep = np.r_[xs[1:] != xs[:-1], True]
        xs, ys = xs[keep], ys[keep]
    return xs, ys


def _clipped_linear_curve(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Represent clip(linear(points), 0, 2) as an exact piecewise-linear curve."""
    xs, ys = _collapse_duplicate_x(xs, ys)
    if len(xs) == 1:
        return xs, np.clip(ys, _GAIN_MIN, _GAIN_MAX)

    out_x = [float(xs[0])]
    out_y = [float(np.clip(ys[0], _GAIN_MIN, _GAIN_MAX))]
    for x0, x1, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        cuts = [float(x0), float(x1)]
        if y1 != y0:
            for level in (_GAIN_MIN, _GAIN_MAX):
                cross = float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
                if x0 < cross < x1:
                    cuts.append(cross)
        cuts.sort()
        for left, right in zip(cuts[:-1], cuts[1:]):
            midpoint = (left + right) * 0.5
            raw_mid = y0 + (y1 - y0) * (midpoint - x0) / (x1 - x0)
            if raw_mid <= _GAIN_MIN:
                left_y = right_y = _GAIN_MIN
            elif raw_mid >= _GAIN_MAX:
                left_y = right_y = _GAIN_MAX
            else:
                left_y = float(np.clip(y0 + (y1 - y0) * (left - x0) / (x1 - x0),
                                       _GAIN_MIN, _GAIN_MAX))
                right_y = float(np.clip(y0 + (y1 - y0) * (right - x0) / (x1 - x0),
                                        _GAIN_MIN, _GAIN_MAX))
            if right <= out_x[-1]:
                continue
            out_x.append(right)
            out_y.append(right_y)
    return np.asarray(out_x), np.asarray(out_y)


def _integral_function(xs: np.ndarray, ys: np.ndarray):
    """Return an antiderivative for a clipped piecewise-linear curve."""
    xs, ys = _collapse_duplicate_x(xs, ys)
    if len(xs) == 1:
        value = float(np.clip(ys[0], _GAIN_MIN, _GAIN_MAX))

        def constant_integral(t):
            t = np.asarray(t, dtype=np.float64)
            return value * (t - xs[0])
        return constant_integral, value, value

    xs, ys = _clipped_linear_curve(xs, ys)
    dx = np.diff(xs)
    area = np.concatenate(([0.0], np.cumsum(0.5 * (ys[:-1] + ys[1:]) * dx)))

    def integrate(t):
        t = np.asarray(t, dtype=np.float64)
        idx = np.clip(np.searchsorted(xs, t, side="right") - 1, 0, len(xs) - 2)
        delta = t - xs[idx]
        slope = (ys[idx + 1] - ys[idx]) / (xs[idx + 1] - xs[idx])
        value = area[idx] + ys[idx] * delta + 0.5 * slope * delta * delta
        value = np.where(t < xs[0], ys[0] * (t - xs[0]), value)
        value = np.where(t >= xs[-1], area[-1] + ys[-1] * (t - xs[-1]), value)
        return value

    return integrate, float(ys[0]), float(ys[-1])


def envelope_gain_at_mel_frame_starts(
        points: Sequence,
        start_frame: int,
        frame_count: int,
        frame_ms: float,
        anchor_ms: float,
        *,
        first_phone: bool = False,
        last_phone: bool = False) -> np.ndarray:
    """Sample OpenUtau envelope at legacy frame starts (pitch ownership only)."""
    frame_ms, anchor_ms = float(frame_ms), float(anchor_ms)
    if not np.isfinite(frame_ms) or frame_ms <= 0 or not np.isfinite(anchor_ms):
        raise ValueError("frame_ms must be positive and anchor_ms finite")
    if int(start_frame) != start_frame or int(frame_count) != frame_count or frame_count < 0:
        raise ValueError("frame indices/count must be integers; count must be nonnegative")
    xs, ys = _read_envelope(points, first_phone, last_phone)
    t = (int(start_frame) + np.arange(int(frame_count), dtype=np.float64)) * frame_ms - anchor_ms
    # Keep np.interp's historical behavior for non-monotone OpenUtau envelopes.
    return np.interp(t, xs, ys, left=ys[0], right=ys[-1])


def envelope_gain_average_per_mel_frame(
        points: Sequence,
        start_frame: int,
        frame_count: int,
        frame_ms: float,
        anchor_ms: float,
        *,
        first_phone: bool = False,
        last_phone: bool = False) -> np.ndarray:
    """Exact time-average of a piecewise-linear envelope over each mel hop.

    If an OpenUtau envelope is non-monotone (possible for very short notes),
    retain the previous renderer's point-sampling behavior rather than rejecting
    an otherwise renderable request.
    """
    frame_ms, anchor_ms = float(frame_ms), float(anchor_ms)
    if not np.isfinite(frame_ms) or frame_ms <= 0 or not np.isfinite(anchor_ms):
        raise ValueError("frame_ms must be positive and anchor_ms finite")
    if int(start_frame) != start_frame or int(frame_count) != frame_count or frame_count < 0:
        raise ValueError("frame indices/count must be integers; count must be nonnegative")
    xs, ys = _read_envelope(points, first_phone, last_phone)
    if frame_count == 0:
        return np.empty(0, dtype=np.float64)
    if np.any(np.diff(xs) < 0):
        return np.clip(envelope_gain_at_mel_frame_starts(
            points, start_frame, frame_count, frame_ms, anchor_ms,
            first_phone=first_phone, last_phone=last_phone), _GAIN_MIN, _GAIN_MAX)

    xs, ys = _collapse_duplicate_x(xs, ys)
    integral, _left, _right = _integral_function(xs, ys)
    edges = ((int(start_frame) + np.arange(int(frame_count) + 1, dtype=np.float64))
             * frame_ms - anchor_ms)
    return np.clip(np.diff(integral(edges)) / frame_ms, _GAIN_MIN, _GAIN_MAX)


def _integrate_mel_to_phrase_frames(
        mel_log_magnitude: np.ndarray,
        source_start_frame: float,
        total_frames: int,
        envelope_points: Sequence | None,
        envelope_anchor_ms: float,
        frame_ms: float,
        *,
        first_phone: bool = False,
        last_phone: bool = False,
        return_components: bool = False,
        magnitude: bool = False):
    """Align and integrate one phone's linear mel amplitude or power on the phrase grid.

    ``mel_log_magnitude[:, i]`` represents source time interval
    ``[source_start_frame + i, source_start_frame + i + 1)``. Each destination
    phrase frame receives the exact time integral of source mel amplitude/power multiplied
    by the OpenUtau envelope over that frame. Fractional source starts therefore
    distribute energy across adjacent output frames instead of rounding the
    whole phone by up to one hop.

    Returns ``(first_phrase_frame, power_contribution)`` by default. With
    ``return_components=True`` it additionally returns the integrated envelope
    weight and envelope-weighted log-magnitude, used for spectral handover
    morphing. No F0 is accepted or produced here.
    """
    mel = np.asarray(mel_log_magnitude, dtype=np.float64)
    source_start_frame = float(source_start_frame)
    envelope_anchor_ms, frame_ms = float(envelope_anchor_ms), float(frame_ms)
    if mel.ndim != 2 or not np.isfinite(mel).all():
        raise ValueError("mel must be a finite (bins, frames) array")
    if not np.isfinite(source_start_frame) or not np.isfinite(envelope_anchor_ms):
        raise ValueError("source start and envelope anchor must be finite")
    if not np.isfinite(frame_ms) or frame_ms <= 0:
        raise ValueError("frame_ms must be finite and positive")
    if int(total_frames) != total_frames or total_frames < 0:
        raise ValueError("total_frames must be a nonnegative integer")

    # Avoid a phantom extra frame from floating-point representations of exact
    # integer starts/ends such as (n * frame_ms) / frame_ms.
    nearest = round(source_start_frame)
    if abs(source_start_frame - nearest) < 1e-10:
        source_start_frame = float(nearest)
    T = mel.shape[1]
    first_frame = int(np.floor(source_start_frame))
    end_frame = int(np.ceil(source_start_frame + T - 1e-12))
    first_clipped = max(0, first_frame)
    end_clipped = min(int(total_frames), end_frame)
    if T == 0 or end_clipped <= first_clipped:
        empty_power = np.zeros((mel.shape[0], 0), dtype=np.float64)
        if return_components:
            return first_clipped, empty_power, np.zeros(0), empty_power.copy()
        return first_clipped, empty_power

    use_envelope = envelope_points is not None and len(envelope_points) >= 2
    if use_envelope:
        xs_ms, ys_raw = _read_envelope(envelope_points, first_phone, last_phone)
        monotone = not np.any(np.diff(xs_ms) < 0)
        if monotone:
            xs = envelope_anchor_ms / frame_ms + xs_ms / frame_ms
            curve_x, curve_y = _clipped_linear_curve(xs, ys_raw)
            envelope_integral, env_left, env_right = _integral_function(curve_x, curve_y)
        else:
            # Match the old renderer's np.interp fallback on unusual envelope
            # ordering, while still applying fractional mel placement.
            fallback_gain = np.clip(np.interp(
                (np.arange(first_clipped, end_clipped, dtype=np.float64) * frame_ms
                 - envelope_anchor_ms), xs_ms, ys_raw,
                left=ys_raw[0], right=ys_raw[-1]), _GAIN_MIN, _GAIN_MAX)
            envelope_integral = None
            env_left = env_right = 1.0
    else:
        envelope_integral = None
        fallback_gain = np.ones(end_clipped - first_clipped, dtype=np.float64)
        env_left = env_right = 1.0

    # OpenUtau envelopes are amplitude gains. Production integrates magnitude;
    # the legacy power helper remains available for historical A/B diagnostics.
    source_power = np.exp(np.clip(mel if magnitude else 2.0 * mel, -80.0, 80.0))
    result = np.zeros((mel.shape[0], end_clipped - first_clipped), dtype=np.float64)
    weight_result = np.zeros(end_clipped - first_clipped, dtype=np.float64)
    log_magnitude_result = np.zeros_like(result)
    for frame in range(first_clipped, end_clipped):
        # Source frame i occupies [source_start+i, source_start+i+1). A
        # destination frame can intersect at most two source mel frames.
        src_left = frame - source_start_frame
        src_right = frame + 1.0 - source_start_frame
        i0 = max(0, int(np.floor(src_left)))
        i1 = min(T - 1, int(np.ceil(src_right) - 1))
        for i in range(i0, i1 + 1):
            left = max(float(frame), source_start_frame + i)
            right = min(float(frame + 1), source_start_frame + i + 1)
            if right <= left:
                continue
            if envelope_integral is not None:
                gain_area = envelope_integral(np.asarray([right, left], dtype=np.float64))
                gain_area = max(0.0, float(gain_area[0] - gain_area[1]))
            else:
                gain_area = (right - left) * fallback_gain[frame - first_clipped]
            frame_index = frame - first_clipped
            result[:, frame_index] += source_power[:, i] * gain_area
            weight_result[frame_index] += gain_area
            log_magnitude_result[:, frame_index] += mel[:, i] * gain_area

    if return_components:
        return first_clipped, result, weight_result, log_magnitude_result
    return first_clipped, result


def integrate_mel_power_to_phrase_frames(
        mel_log_magnitude, source_start_frame, total_frames, envelope_points,
        envelope_anchor_ms, frame_ms, *, first_phone=False, last_phone=False,
        return_components=False):
    """Legacy power-weighted placement, retained for diagnostics/regressions.

    Multiplying power by an amplitude envelope gives sqrt(gain) after conversion
    to log magnitude. Do not use this helper for production amplitude envelopes.
    """
    return _integrate_mel_to_phrase_frames(
        mel_log_magnitude, source_start_frame, total_frames, envelope_points,
        envelope_anchor_ms, frame_ms, first_phone=first_phone, last_phone=last_phone,
        return_components=return_components)


def integrate_mel_magnitude_to_phrase_frames(
        mel_log_magnitude, source_start_frame, total_frames, envelope_points,
        envelope_anchor_ms, frame_ms, *, return_components=False):
    """Integrate exp(mel) * OpenUtau amplitude envelope at exact sub-hop onset.

    Linear magnitude addition models a single shared-excitation spectral
    handover, without geometric/log averaging's source-specific band notches.
    The supplied envelopes (including VOL/ATK/DEC) are neither normalized nor
    widened. Fractional placement retains the exact envelope integral.
    """
    return _integrate_mel_to_phrase_frames(
        mel_log_magnitude, source_start_frame, total_frames, envelope_points,
        envelope_anchor_ms, frame_ms, return_components=return_components, magnitude=True)


def compose_phone_magnitude_sum(total_magnitude: np.ndarray, floor: float = -30.0) -> np.ndarray:
    """One log-magnitude output after summing amplitude-gated source spectra."""
    magnitude = np.asarray(total_magnitude, dtype=np.float64)
    if magnitude.ndim != 2 or not np.isfinite(magnitude).all() or np.any(magnitude < 0):
        raise ValueError('total mel magnitude must be a finite nonnegative matrix')
    result = np.log(np.maximum(magnitude, 1e-12)).astype(np.float32)
    result[magnitude <= 1e-12] = floor
    return result


class MelOverlapAccumulator:
    """Amplitude-faithful handover with bounded, expression-independent energy control.

    Neutral structural fades define the energy budget independently of user
    VOL/ATK/DEC. Highly coherent spectra use magnitude addition; different spectra
    approach an equal-power handover with expression gains squared. This avoids
    geometric averaging's missing bands and raw magnitude blending's energy dip.
    A neutral-only bound limits interpolation compensation to 3dB by default.
    Outside a real fade this is exact amplitude addition, not sqrt(envelope).
    """
    def __init__(self, bins: int, frames: int, frame_ms: float,
                 max_compensation_db: float = 3.01029995664,
                 coherence_width: float = .08):
        if (not np.isfinite(max_compensation_db) or max_compensation_db < 0
                or not np.isfinite(coherence_width) or not 0 < coherence_width <= 1):
            raise ValueError('invalid overlap controls')
        self.frame_ms = frame_ms
        self.frames = frames
        self.limit = 10. ** (max_compensation_db / 20.)
        self.actual = np.zeros((bins, frames), dtype=np.float64)
        self.neutral = np.zeros_like(self.actual)
        self.coherence_width = coherence_width
        self.neutral_power = np.zeros_like(self.actual)
        self.expression_power = np.zeros_like(self.actual)
        self.unit_sum = np.zeros_like(self.actual)
        self.blocked = np.zeros(frames, dtype=bool)
        self.weight = np.zeros(frames, dtype=np.float64)
        self.weight_sq = np.zeros(frames, dtype=np.float64)
        self.count = np.zeros(frames, dtype=np.int32)
        self.trace = {}

    def add(self, mel, start_frame, envelope, anchor_ms):
        start, actual, actual_weight, _logs = integrate_mel_magnitude_to_phrase_frames(
            mel, start_frame, self.frames, envelope, anchor_ms, self.frame_ms, return_components=True)
        end = start + actual.shape[1]
        self.actual[:, start:end] += actual
        if envelope is None or len(envelope) != 5 or actual.shape[1] == 0:
            self.blocked[start:end] = True
            return
        xs, ys = _read_envelope(envelope, False, False)
        if np.any(np.diff(xs) < 0) or ys[0] != 0 or ys[-1] != 0:
            self.blocked[start:end] = True
            return  # Cannot infer a standard structural fade from this curve.
        structural = list(zip(xs, (0., 1., 1., 1., 0.)))
        _s, neutral, weight, _logs = integrate_mel_magnitude_to_phrase_frames(
            mel, start_frame, self.frames, structural, anchor_ms, self.frame_ms, return_components=True)
        _s, neutral_power = integrate_mel_power_to_phrase_frames(
            mel, start_frame, self.frames, structural, anchor_ms, self.frame_ms)
        self.neutral[:, start:end] += neutral
        self.neutral_power[:, start:end] += neutral_power
        expression = actual_weight / np.maximum(weight, 1e-12)
        self.expression_power[:, start:end] += neutral_power * expression[None, :] ** 2
        norm = np.sqrt(np.sum(neutral * neutral, axis=0))
        self.unit_sum[:, start:end] += neutral * (weight / np.maximum(norm, 1e-12))[None, :]
        self.weight[start:end] += weight
        self.weight_sq[start:end] += weight * weight
        self.count[start:end] += weight > 1e-6

    def compose(self):
        observed = np.sum(self.neutral * self.neutral, axis=0)
        target = self.weight * np.sum(self.neutral_power, axis=0)
        pairs = self.weight ** 2 - self.weight_sq
        cosine = np.clip((np.sum(self.unit_sum ** 2, axis=0) - self.weight_sq)
                         / np.maximum(pairs, 1e-12), 0., 1.)
        fade = ((self.count >= 2) & ~self.blocked & (pairs > 1e-12)
                & (self.weight_sq < self.weight - 1e-6) & (observed > 1e-20))
        blend = np.where(fade, np.clip((1. - cosine) / self.coherence_width, 0., 1.), 0.)
        relative_energy = np.maximum(target / np.maximum(observed, 1e-24) - 1., 0.)
        blend = np.minimum(blend, (self.limit ** 2 - 1.) / np.maximum(relative_energy, 1e-12))
        neutral_energy = observed + blend * (target - observed)
        output_power = ((1. - blend)[None, :] * self.actual ** 2
                        + (blend * self.weight)[None, :] * self.expression_power)
        # Unrecognized envelope clips still use plain amplitude addition.
        actual_energy = np.sum(output_power, axis=0)
        self.expression_gain = np.sqrt(actual_energy / np.maximum(neutral_energy, 1e-24))
        self.expression_gain[observed <= 1e-20] = 1.
        self.expression_gain[self.blocked] = 0.  # Do not de-express an unknown curve for note matching.
        gain = np.sqrt(neutral_energy / np.maximum(observed, 1e-24))
        gain[observed <= 1e-20] = 1.
        self.trace = dict(mode='coherence_aware_amplitude_power_handover',
                          overlap_frames=int(np.count_nonzero(fade)),
                          max_compensation_db=float(20 * np.log10(max(1., float(np.max(gain))))) if gain.size else 0.,
                          coherence_width=self.coherence_width, expression_independent=True)
        return compose_phone_magnitude_sum(np.sqrt(np.maximum(output_power, 0.)))


def compose_phone_power_sum(total_power: np.ndarray, floor: float = -30.0) -> np.ndarray:
    """Sum envelope-weighted mel power without erasing source-specific bands.

    The input already contains fractional-time and OpenUtau-envelope
    integration. Retained for historical power-composition A/Bs, not production.
    """
    power = np.asarray(total_power, dtype=np.float64)
    if power.ndim != 2 or not np.isfinite(power).all():
        raise ValueError('total mel power must be a finite matrix')
    result = (0.5 * np.log(np.maximum(power, 1e-24))).astype(np.float32)
    result[power <= 1e-24] = floor
    return result


def compose_phone_mel(total_power: np.ndarray,
                      weighted_log_magnitude: np.ndarray,
                      envelope_weight: np.ndarray,
                      active_phone_count: np.ndarray,
                      floor: float = -30.0,
                      envelope_ceiling: np.ndarray | None = None) -> np.ndarray:
    """Diagnostic-only log-mel morph candidate; never used in production.

    Geometric log-mel averaging can erase frequency bands unique to either
    source; production uses :class:`MelOverlapAccumulator` instead.
    """
    power = np.asarray(total_power, dtype=np.float64)
    log_mag = np.asarray(weighted_log_magnitude, dtype=np.float64)
    weight = np.asarray(envelope_weight, dtype=np.float64)
    count = np.asarray(active_phone_count)
    if power.ndim != 2 or log_mag.shape != power.shape or weight.shape != (power.shape[1],) or count.shape != weight.shape:
        raise ValueError('incompatible phone composition arrays')
    if not all(np.isfinite(v).all() for v in (power, log_mag, weight, count)):
        raise ValueError('phone composition inputs must be finite')
    if envelope_ceiling is None:
        envelope_ceiling = np.ones(power.shape[1], dtype=np.float64)
    ceiling = np.asarray(envelope_ceiling, dtype=np.float64)
    if ceiling.shape != weight.shape or not np.isfinite(ceiling).all():
        raise ValueError('incompatible envelope ceiling')
    result = compose_phone_power_sum(power, floor)
    overlap = (count >= 2) & (weight > 1e-12)
    if np.any(overlap):
        average = log_mag[:, overlap] / weight[overlap][np.newaxis, :]
        retained_weight = np.minimum(weight[overlap], ceiling[overlap])
        result[:, overlap] = (average + 0.5 * np.log(
            np.maximum(retained_weight, 1e-12))[np.newaxis, :]).astype(np.float32)
    return result


__all__ = ["envelope_gain_average_per_mel_frame",
           "envelope_gain_at_mel_frame_starts",
           "integrate_mel_magnitude_to_phrase_frames",
           "compose_phone_magnitude_sum",
           "MelOverlapAccumulator",
           "integrate_mel_power_to_phrase_frames",
           "compose_phone_power_sum",
           "compose_phone_mel"]
