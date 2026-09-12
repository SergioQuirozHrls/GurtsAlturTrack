"""Secondary acoustic features on the caller channel: spectral/prosody
summary stats via scipy.signal only -- no librosa (CLAUDE.md section 3).

Only worth merging into the trained feature set if the turn-feature-only AUC
falls short of ~0.65 (CLAUDE.md section 9 fallback plan). Don't invest here
before that check has actually failed.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

FEATURE_NAMES = [
    "spectral_centroid_mean",
    "spectral_flatness_mean",
    "rms_mean",
    "rms_std",
    "zero_crossing_rate_mean",
]

FRAME_S = 0.02
HOP_S = 0.01


def acoustic_features(ch0: np.ndarray, sr: int = 8000) -> dict[str, float]:
    """Flat dict, fixed key set (FEATURE_NAMES), scalar values."""
    if ch0.size == 0:
        return dict.fromkeys(FEATURE_NAMES, 0.0)

    x = ch0.astype(np.float64)
    freqs, psd = welch(x, fs=sr, nperseg=min(512, len(x)))
    psd_sum = psd.sum() + 1e-12
    centroid = float((freqs * psd).sum() / psd_sum)
    flatness = float(np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd) + 1e-12))

    frame = max(int(sr * FRAME_S), 1)
    hop = max(int(sr * HOP_S), 1)
    rms_vals = []
    zcr_vals = []
    for start in range(0, max(len(x) - frame, 1), hop):
        seg = x[start : start + frame]
        if seg.size == 0:
            continue
        rms_vals.append(np.sqrt(np.mean(seg**2)))
        zcr_vals.append(np.mean(np.abs(np.diff(np.sign(seg))) > 0))

    return {
        "spectral_centroid_mean": centroid,
        "spectral_flatness_mean": flatness,
        "rms_mean": float(np.mean(rms_vals)) if rms_vals else 0.0,
        "rms_std": float(np.std(rms_vals)) if rms_vals else 0.0,
        "zero_crossing_rate_mean": float(np.mean(zcr_vals)) if zcr_vals else 0.0,
    }
