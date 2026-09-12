"""Voice-activity segmentation used at inference time, when only raw audio
(no turns/*.json) is available. numpy + scipy.signal only -- no librosa
(CLAUDE.md section 3).

Replaces the placeholder fixed -45 dB floor + median filter with a
noise-floor-relative adaptive threshold plus hysteresis:

- Adaptive ON threshold: recording level varies a lot across calls (line
  gain, mic distance), so a fixed dB floor is wrong on quiet calls. The ON
  threshold is set against the channel's own noise floor (low percentile)
  and speech level (high percentile); a channel whose dynamic range is too
  small is treated as silent.
- Hysteresis: a region stays voiced until energy drops several dB below the
  ON threshold, so trailing word endings aren't clipped.
- Min-gap merge: turns separated by a very short silence are merged so one
  utterance isn't fragmented into many tiny segments.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FRAME_MS = 20
HOP_MS = 10
MIN_TURN_S = 0.15
MIN_GAP_S = 0.20
NOISE_PCT = 20.0
SPEECH_PCT = 95.0
THRESH_FRACTION = 0.30
HYSTERESIS_DB = 6.0
HANGOVER_FRAMES = 3
MIN_RANGE_DB = 6.0
MIN_ENERGY_DB = -60.0


def _frame_energy_db(x: np.ndarray, sr: int) -> np.ndarray:
    frame = int(sr * FRAME_MS / 1000)
    hop = int(sr * HOP_MS / 1000)
    if len(x) < frame:
        return np.array([])
    frames = sliding_window_view(x.astype(np.float64), frame)[::hop]
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def _adaptive_on_db(energies: np.ndarray) -> float:
    """ON threshold (dB) from the channel's own noise floor and speech peak.

    Returns -inf when the channel has too little dynamic range to separate
    speech from noise -- an effectively silent or noise-only channel, which
    yields no turns.
    """
    if energies.size < 32:
        return -np.inf
    floor = float(np.percentile(energies, NOISE_PCT))
    peak = float(np.percentile(energies, SPEECH_PCT))
    if peak - floor < MIN_RANGE_DB:
        return -np.inf
    return floor + THRESH_FRACTION * (peak - floor)


def _energy_to_turns(
    energies: np.ndarray, hop_s: float, on_db: float
) -> list[tuple[float, float]]:
    """Voiced-region state machine with hysteresis.

    A frame is a candidate for speech when its energy is at/above the adaptive
    on_db. Once voiced, the region persists while energy stays above on_db -
    HYSTERESIS_DB; trailing word endings are kept via a short hangover.
    """
    if not np.isfinite(on_db) or energies.size == 0:
        return []
    off_db = on_db - HYSTERESIS_DB
    hangover = HANGOVER_FRAMES
    below = 0
    in_speech = False
    start = 0
    turns: list[tuple[float, float]] = []

    for i in range(len(energies)):
        if in_speech:
            if energies[i] >= off_db and energies[i] >= MIN_ENERGY_DB:
                below = 0
            else:
                below += 1
                if below >= hangover:
                    turns.append((start * hop_s, (i - hangover + 1) * hop_s))
                    in_speech = False
        elif energies[i] >= on_db and energies[i] >= MIN_ENERGY_DB:
            in_speech = True
            start = i
            below = 0

    if in_speech:
        turns.append((start * hop_s, len(energies) * hop_s))
    return turns


def _merge_and_filter(turns: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in turns:
        if merged and start - merged[-1][1] < MIN_GAP_S:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [(start, end) for start, end in merged if end - start >= MIN_TURN_S]


def detect_turns(ch0: np.ndarray, ch1: np.ndarray, sr: int = 8000) -> dict:
    """Segment each channel into voiced turns. Depends only on the raw channel
    arrays -- /detect only ever receives audio, never turns/*.json.

    Returns {"caller_turns": [(start_s, end_s), ...], "agent_turns": [...]}.
    """
    hop_s = HOP_MS / 1000
    ch0_db = _frame_energy_db(ch0, sr)
    ch1_db = _frame_energy_db(ch1, sr)
    caller_turns = _energy_to_turns(ch0_db, hop_s, _adaptive_on_db(ch0_db))
    agent_turns = _energy_to_turns(ch1_db, hop_s, _adaptive_on_db(ch1_db))
    return {
        "caller_turns": _merge_and_filter(caller_turns),
        "agent_turns": _merge_and_filter(agent_turns),
    }
