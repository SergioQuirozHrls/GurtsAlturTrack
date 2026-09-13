"""Shared feature path for train.py and app/api.py. CLAUDE.md section 4
requires this be the ONE place ch0/ch1 audio turns into a feature vector,
so training and serving can never silently diverge.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from features.acoustic import FEATURE_NAMES as ACOUSTIC_FEATURE_NAMES
from features.acoustic import acoustic_features
from features.liveness import FEATURE_NAMES as LIVENESS_FEATURE_NAMES
from features.liveness import liveness_features
from features.turns import FEATURE_NAMES as TURN_FEATURE_NAMES
from features.turns import turn_features
from features.vad import detect_turns

FEATURE_NAMES = [*TURN_FEATURE_NAMES, *ACOUSTIC_FEATURE_NAMES, *LIVENESS_FEATURE_NAMES]


def call_features(ch0: np.ndarray, ch1: np.ndarray, sr: int = 8000) -> dict[str, float]:
    """ch0, ch1 -> flat feature dict, keys == FEATURE_NAMES. The only feature
    entrypoint both train.py and app/api.py call -- never reimplement feature
    extraction anywhere else.
    """
    turns = detect_turns(ch0, ch1, sr)
    feats = turn_features(turns["caller_turns"], turns["agent_turns"])
    feats.update(acoustic_features(ch0, sr))
    feats.update(liveness_features(ch0, sr))
    return feats


def _provided_turns_to_lists(turns_dir: Path, anon_id: str) -> tuple[list, list] | None:
    """turns/<anon_id>.json -> (caller_turns, agent_turns), or None if absent.

    On-disk format is {"turns": [{"channel": 0|1, "start": s, "end": s}, ...]}
    -- channel 0 is the caller, channel 1 is the agent (dataset README).
    """
    path = turns_dir / f"{anon_id}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    caller_turns = [
        (t["start"], t["end"]) for t in payload["turns"] if t["channel"] == 0
    ]
    agent_turns = [
        (t["start"], t["end"]) for t in payload["turns"] if t["channel"] == 1
    ]
    return caller_turns, agent_turns


def build_dataset(
    manifest_path: Path, audio_dir: Path, turns_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """manifest.csv + turns/*.json + audio/*.wav -> (X, y, split).

    manifest.csv columns are anon_id, label ("human"/"synthetic"), split
    ("train"/"val"), duration_s -- see the dataset README. There is no
    separate caller id: train/val are already speaker-disjoint by
    construction, so `split` (not a re-derived group) is what val metrics
    must be scored against (CLAUDE.md sections 2 and 9).

    Prefers provided turns/*.json over VAD-derived turns when present
    (closer to ground truth for training).
    """
    manifest = pd.read_csv(manifest_path, encoding="utf-8")

    split_counts = manifest.groupby("anon_id")["split"].nunique()
    leaked = sorted(split_counts[split_counts > 1].index)
    if leaked:
        raise ValueError(
            "anon_id values appear in both train and val splits -- this is a "
            f"scoring-on-train fail condition per CLAUDE.md section 2: {leaked}"
        )

    rows: list[list[float]] = []
    labels: list[int] = []
    splits: list[str] = []

    for _, row in manifest.iterrows():
        anon_id = str(row["anon_id"])
        data, sr = sf.read(audio_dir / f"{anon_id}.wav", always_2d=True)
        ch0, ch1 = data[:, 0], data[:, 1]

        provided = _provided_turns_to_lists(turns_dir, anon_id)
        if provided is not None:
            caller_turns, agent_turns = provided
            feats = turn_features(caller_turns, agent_turns)
            feats.update(acoustic_features(ch0, sr))
            feats.update(liveness_features(ch0, sr))
        else:
            feats = call_features(ch0, ch1, sr)

        rows.append([feats[name] for name in FEATURE_NAMES])
        labels.append(1 if row["label"] == "synthetic" else 0)
        splits.append(str(row["split"]))

    return np.array(rows), np.array(labels), np.array(splits)
