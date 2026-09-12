"""Voice-activity segmentation used at inference time, when only raw audio
(no turns/*.json) is available. Energy-threshold VAD on each channel,
scipy.signal only -- no librosa (CLAUDE.md section 3).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

FRAME_MS = 20
HOP_MS = 10
ENERGY_FLOOR_DB = -45.0
MIN_TURN_S = 0.15


def _frame_energy_db(x: np.ndarray, sr: int) -> np.ndarray:
    frame = int(sr * FRAME_MS / 1000)
    hop = int(sr * HOP_MS / 1000)
    if len(x) < frame:
        return np.array([])
    n_frames = 1 + (len(x) - frame) // hop
    energies = np.empty(n_frames)
    for i in range(n_frames):
        start = i * hop
        seg = x[start : start + frame]
        rms = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
        energies[i] = 20 * np.log10(rms + 1e-12)
    return energies


def _energy_to_turns(energies: np.ndarray, hop_s: float) -> list[tuple[float, float]]:
    if energies.size == 0:
        return []
    voiced = medfilt((energies > ENERGY_FLOOR_DB).astype(np.int32), kernel_size=5) > 0
    turns: list[tuple[float, float]] = []
    start = None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            turns.append((start * hop_s, i * hop_s))
            start = None
    if start is not None:
        turns.append((start * hop_s, len(voiced) * hop_s))
    return [(s, e) for s, e in turns if e - s >= MIN_TURN_S]


def detect_turns(ch0: np.ndarray, ch1: np.ndarray, sr: int = 8000) -> dict:
    """Segment each channel into voiced turns using an energy threshold.

    Returns {"caller_turns": [(start_s, end_s), ...], "agent_turns": [...]}.
    Depends only on the raw channel arrays -- /detect only ever receives
    audio, never turns/*.json.
    """
    hop_s = HOP_MS / 1000
    caller_turns = _energy_to_turns(_frame_energy_db(ch0, sr), hop_s)
    agent_turns = _energy_to_turns(_frame_energy_db(ch1, sr), hop_s)
    return {"caller_turns": caller_turns, "agent_turns": agent_turns}
