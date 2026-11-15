"""Translate OpenUtau's resolved preutter to source-time skip/pad coordinates."""
import numpy as np


def source_skip_ms(oto_preutter_ms: float, effective_preutter_ms: float,
                   velocity: float) -> float:
    """Classic ResamplerItem: skipOver = oto.Preutter * stretch - leading.

    OpenUtau owns effective preutter and velocity. A positive skip trims the
    rendered source prefix; a negative skip pads before the source. Neither
    operation changes where the recorded oto preutter anchor meets the note.
    """
    values = np.asarray([oto_preutter_ms, effective_preutter_ms, velocity], dtype=float)
    if not np.isfinite(values).all() or min(oto_preutter_ms, effective_preutter_ms) < 0:
        raise ValueError('source preutters must be finite and nonnegative')
    stretch = 2.0 ** (1.0 - float(velocity))
    return float(oto_preutter_ms) * stretch - float(effective_preutter_ms)


def shift_mel_for_source_skip(mel: np.ndarray, skip_ms: float,
                              frame_ms: float, floor: float = -80.0) -> np.ndarray:
    """Diagnostic Classic skipOver on an already-resampled phone mel.

    A positive skip advances the source content but leaves the phone's OpenUtau
    p0/envelope anchor fixed. A negative skip inserts leading silence. This
    helper is for controlled timing A/Bs; it does not alter phone positions,
    envelopes, F0 or output duration.
    """
    values = np.asarray(mel, dtype=np.float64)
    skip_ms, frame_ms, floor = float(skip_ms), float(frame_ms), float(floor)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('mel must be a finite matrix')
    if not np.isfinite([skip_ms, frame_ms, floor]).all() or frame_ms <= 0:
        raise ValueError('skip and floor must be finite; frame_ms must be positive')
    if values.shape[1] == 0 or abs(skip_ms) < 1e-12:
        return values.astype(np.float32, copy=True)
    source_positions = np.arange(values.shape[1], dtype=np.float64) + skip_ms / frame_ms
    target_positions = np.arange(values.shape[1], dtype=np.float64)
    shifted = np.empty(values.shape, dtype=np.float32)
    for band, row in enumerate(values):
        shifted[band] = np.interp(source_positions, target_positions, row,
                                  left=floor, right=floor)
    return shifted
