"""Compare features.vad.detect_turns against the provided ground-truth
turns/*.json on held-out (val) calls -- the risk check from CLAUDE.md section
9: at /detect time only raw audio exists, so how closely the VAD reproduces
the provided turns decides whether the team can train on ground-truth turns
or must fall back to VAD-derived features.

Run with:  python scripts/vad_validate.py [--n-calls N] [--seed N]
Output goes verbatim into the PR description.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.vad import detect_turns

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "manifest.csv"
AUDIO_DIR = ROOT / "audio"
TURNS_DIR = ROOT / "turns"
GRID_S = 0.01
MATCH_S = 1.0


def _provided_turns(anon_id: str) -> tuple[list, list]:
    with (TURNS_DIR / f"{anon_id}.json").open(encoding="utf-8") as f:
        payload = json.load(f)
    caller = [(t["start"], t["end"]) for t in payload["turns"] if t["channel"] == 0]
    agent = [(t["start"], t["end"]) for t in payload["turns"] if t["channel"] == 1]
    return caller, agent


def _mask(turns: list[tuple[float, float]], n_frames: int) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for start, end in turns:
        mask[int(start / GRID_S) : int(end / GRID_S)] = True
    return mask


def _iou(detected: list, reference: list, n_frames: int) -> float:
    a = _mask(detected, n_frames)
    b = _mask(reference, n_frames)
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    return float((a & b).sum()) / union


def _boundary_delta(detected: list, reference: list, which: str) -> float:
    ref = sorted((s if which == "start" else e) for s, e in reference)
    det = sorted(s if which == "start" else e for s, e in detected)
    deltas = []
    for x in det:
        diff = min((abs(x - y) for y in ref), default=float("inf"))
        if diff <= MATCH_S:
            deltas.append(diff)
    return (float(np.mean(deltas)) * 1000.0) if deltas else float("nan")


def _row_stats(detected: list, reference: list, n_frames: int) -> dict:
    return {
        "iou": _iou(detected, reference, n_frames),
        "end_delta_ms": _boundary_delta(detected, reference, "end"),
        "start_delta_ms": _boundary_delta(detected, reference, "start"),
        "turn_mismatch": abs(len(detected) - len(reference)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(
            "manifest.csv not found -- extract the dataset first (CLAUDE.md section 5)."
        )
        return

    manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8")
    val = manifest[manifest["split"] == "val"]
    picks = val.sample(args.n_calls, random_state=args.seed)

    print(
        f"{'anon_id':<18}{'label':>10}{'ch':>3}{'iou':>8}"
        f"{'start_ms':>10}{'end_ms':>10}{'turns':>6}"
    )
    totals = {"iou": [], "start_delta_ms": [], "end_delta_ms": [], "turn_mismatch": []}
    for _, row in picks.iterrows():
        anon_id = str(row["anon_id"])
        data, sr = sf.read(AUDIO_DIR / f"{anon_id}.wav", always_2d=True)
        ch0, ch1 = data[:, 0], data[:, 1]
        detected = detect_turns(ch0, ch1, sr)
        ref_caller, ref_agent = _provided_turns(anon_id)
        n_frames = int(len(ch0) / sr / GRID_S)

        for ch_name, det, ref in (
            ("0", detected["caller_turns"], ref_caller),
            ("1", detected["agent_turns"], ref_agent),
        ):
            s = _row_stats(det, ref, n_frames)
            for k, value in s.items():
                totals[k].append(value)
            print(
                f"{anon_id:<18}{row['label']!s:>10}{ch_name:>3}"
                f"{s['iou']:>8.3f}{s['start_delta_ms']:>10.1f}"
                f"{s['end_delta_ms']:>10.1f}{len(det) - len(ref):>6}"
            )

    print(
        f"{'MEAN':<18}{'':>10}{'':>3}"
        f"{np.mean(totals['iou']):>8.3f}{np.mean(totals['start_delta_ms']):>10.1f}"
        f"{np.mean(totals['end_delta_ms']):>10.1f}{0:>6}"
    )


if __name__ == "__main__":
    main()
