"""Liveness/vocoder-artifact features on the caller channel: pitch- and
noise-based cues that don't look at behavior or broad prosody at all, so this
signal stays useful even if a synthetic caller's turn-timing (features/turns.py)
gets good enough to blend in. ch0-only, numpy + scipy.signal only, and
deliberately self-contained -- no dependency on vad.py or the other feature
modules (see HANDOFF.md for the agreed design).

Wired into features/build.py's FEATURE_NAMES after comparing liveness-only
vs. turns-only vs. combined val metrics per the agreed validation plan:
liveness-only alone matched turns-only (val AUC 0.987 each), and combined
reached val AUC 1.000 with f0_jitter_mean/voiced_ratio carrying real
permutation importance -- not on faith.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "f0_jitter_mean",
    "shimmer_mean",
    "hnr_proxy_mean",
    "voiced_ratio",
    "noise_floor_cv",
]

FRAME_S = 0.032
HOP_S = 0.016
F0_MIN_HZ = 80
F0_MAX_HZ = 400
NOISE_SEGMENTS = 8
NOISE_QUANTILE = 0.2


def _frame_signal(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    n_frames = max((len(x) - frame) // hop + 1, 0)
    if n_frames == 0:
        return np.empty((0, frame))
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


def _pitch_track(
    x: np.ndarray, sr: int, frame: int, hop: int
) -> tuple[np.ndarray, np.ndarray]:
    """One autocorrelation pass over ch0 -> (f0_per_frame, periodicity_per_frame).

    f0 is 0.0 and periodicity is 0.0 for unvoiced/silent frames. Lag search is
    restricted to F0_MIN_HZ..F0_MAX_HZ so the "peak lag" is always a plausible
    pitch period rather than e.g. lag 0.
    """
    frames = _frame_signal(x, frame, hop)
    if frames.shape[0] == 0:
        return np.empty(0), np.empty(0)

    lag_min = max(int(sr / F0_MAX_HZ), 1)
    lag_max = min(int(sr / F0_MIN_HZ), frame - 1)
    if lag_max <= lag_min:
        return np.zeros(frames.shape[0]), np.zeros(frames.shape[0])

    windowed = frames * np.hanning(frame)
    f0 = np.zeros(frames.shape[0])
    periodicity = np.zeros(frames.shape[0])
    for i, seg in enumerate(windowed):
        energy = np.dot(seg, seg)
        if energy <= 1e-12:
            continue
        # Full autocorrelation via FFT, keep non-negative lags only.
        fft_size = 1 << (2 * frame - 1).bit_length()
        spec = np.fft.rfft(seg, n=fft_size)
        acf = np.fft.irfft(spec * np.conj(spec), n=fft_size)[:frame]
        acf /= acf[0] + 1e-12
        window = acf[lag_min : lag_max + 1]
        if window.size == 0:
            continue
        peak_offset = int(np.argmax(window))
        peak_val = float(window[peak_offset])
        if peak_val <= 0:
            continue
        lag = lag_min + peak_offset
        f0[i] = sr / lag
        periodicity[i] = peak_val

    return f0, periodicity


def liveness_features(ch0: np.ndarray, sr: int = 8000) -> dict[str, float]:
    """Flat dict, fixed key set (FEATURE_NAMES), scalar values."""
    if ch0.size == 0:
        return dict.fromkeys(FEATURE_NAMES, 0.0)

    x = ch0.astype(np.float64)
    frame = max(int(sr * FRAME_S), 1)
    hop = max(int(sr * HOP_S), 1)

    f0, periodicity = _pitch_track(x, sr, frame, hop)
    voiced = periodicity > 0.3

    feats = dict.fromkeys(FEATURE_NAMES, 0.0)
    if voiced.any():
        feats["voiced_ratio"] = float(voiced.mean())
        feats["hnr_proxy_mean"] = float(periodicity[voiced].mean())

        voiced_f0 = f0[voiced]
        if voiced_f0.size > 1 and voiced_f0.mean() > 0:
            jitter = np.abs(np.diff(voiced_f0))
            feats["f0_jitter_mean"] = float(jitter.mean() / voiced_f0.mean())

        frames = _frame_signal(x, frame, hop)
        voiced_frames = frames[voiced[: frames.shape[0]]]
        if voiced_frames.shape[0] > 1:
            rms = np.sqrt(np.mean(voiced_frames**2, axis=1))
            if rms.mean() > 0:
                shimmer = np.abs(np.diff(rms))
                feats["shimmer_mean"] = float(shimmer.mean() / rms.mean())

    frames = _frame_signal(x, frame, hop)
    if frames.shape[0] > 0:
        frame_rms = np.sqrt(np.mean(frames**2, axis=1))
        seg_bounds = np.array_split(np.arange(frames.shape[0]), NOISE_SEGMENTS)
        seg_means = []
        for seg_idx in seg_bounds:
            if seg_idx.size == 0:
                continue
            seg_rms = frame_rms[seg_idx]
            thresh = np.quantile(seg_rms, NOISE_QUANTILE)
            quiet = seg_rms[seg_rms <= thresh]
            if quiet.size:
                seg_means.append(float(quiet.mean()))
        if seg_means:
            arr = np.array(seg_means)
            mean = float(arr.mean())
            feats["noise_floor_cv"] = float(arr.std() / mean) if mean > 0 else 0.0

    return feats
