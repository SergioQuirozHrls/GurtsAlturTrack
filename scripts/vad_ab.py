"""A/B: new adaptive vad.py vs the placeholder that shipped on `main`
(fixed -45 dB floor + median filter) -- same val calls, same metrics, same
helper functions imported from scripts/vad_validate.py so train/serve and
validation share one metric path (CLAUDE.md section 7).

The placeholder VAD is loaded from git (`git show main:features/vad.py`) into a
temp module, so the working tree never holds two copies and the `vs-orig`
numbers are provably the committed baseline.

Also runs the CLAUDE.md section 9 risk check: how closely the VAD reproduces
the provided turns/*.json turns (mean IoU + signed boundary bias + turn-count
mismatch), and the upstream-continuity checks -- pearson r between
turn_features() computed on VAD-derived turns vs on provided turns, and
between VAD-derived turn features and provided-derived ones. If those
correlate near 1.0, downstream .turns features are continuous at /detect time.

Run:  python scripts/vad_ab.py [--n-calls N] [--seed N]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.turns import FEATURE_NAMES as TURN_FEATURE_NAMES
from features.turns import turn_features
from features.vad import detect_turns as detect_new

GRID_S = 0.01
MATCH_S = 1.0


def _orig_detect():
    """Placeholder VAD from main, loaded into a throwaway temp module."""
    blob = subprocess.run(
        ["git", "show", "main:features/vad.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
    ).stdout
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blob)
        tmp = f.name
    spec = importlib.util.spec_from_file_location("orig_vad", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.detect_turns


def _provided_turns(anon_id: str) -> tuple[list, list]:
    payload = json.loads(
        (ROOT / "turns" / f"{anon_id}.json").read_text(encoding="utf-8")
    )
    caller = [(t["start"], t["end"]) for t in payload["turns"] if t["channel"] == 0]
    agent = [(t["start"], t["end"]) for t in payload["turns"] if t["channel"] == 1]
    return caller, agent


def _mask(turns: list, n_frames: int) -> np.ndarray:
    m = np.zeros(n_frames, dtype=bool)
    for s, e in turns:
        m[int(s / GRID_S) : int(e / GRID_S)] = True
    return m


def _iou(det: list, ref: list, n_frames: int) -> float:
    a, b = _mask(det, n_frames), _mask(ref, n_frames)
    u = int((a | b).sum())
    return 1.0 if u == 0 else float((a & b).sum()) / u


def _mean_boundary_ms(det: list, ref: list, which: str) -> float:
    refs = sorted(s if which == "s" else e for s, e in ref)
    dets = sorted(s if which == "s" else e for s, e in det)
    deltas = []
    for x in dets:
        best, best_y = float("inf"), None
        for y in refs:
            d = abs(x - y)
            if d < best:
                best, best_y = d, y
        if best_y is not None and best <= MATCH_S:
            deltas.append(best)
    return float(np.mean(deltas) * 1000) if deltas else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = pd.read_csv(ROOT / "manifest.csv", encoding="utf-8")
    picks = manifest[manifest["split"] == "val"].sample(
        args.n_calls, random_state=args.seed
    )

    orig_detect = _orig_detect()
    cols = ["iou", "start_ms", "end_ms", "turn_mismatch"]
    totals = {k: [] for k in cols}
    print(
        f"{'anon_id':<18}{'label':>10}{'module':>8}{'ch':>3} "
        + "".join(f"{k:>10}" for k in cols)
    )

    feat_ref_rows: dict[str, list] = {k: [] for k in TURN_FEATURE_NAMES}
    feat_vad_rows: dict[str, list] = {k: [] for k in TURN_FEATURE_NAMES}

    for _, row in picks.iterrows():
        anon_id = str(row["anon_id"])
        data, sr = sf.read(ROOT / "audio" / f"{anon_id}.wav", always_2d=True)
        ch0, ch1 = data[:, 0], data[:, 1]
        n_frames = int(len(ch0) / sr / GRID_S)
        ref_caller, ref_agent = _provided_turns(anon_id)

        for mod_name, detect in (("new", detect_new), ("orig", orig_detect)):
            det = detect(ch0, ch1, sr)
            for ch_name, d, ref in (
                ("0", det["caller_turns"], ref_caller),
                ("1", det["agent_turns"], ref_agent),
            ):
                s = {
                    "iou": _iou(d, ref, n_frames),
                    "start_ms": _mean_boundary_ms(d, ref, "s"),
                    "end_ms": _mean_boundary_ms(d, ref, "e"),
                    "turn_mismatch": abs(len(d) - len(ref)),
                }
                if mod_name == "new":
                    for k in cols:
                        totals[k].append(s[k])
                print(
                    f"{anon_id:<18}{row['label']!s:>10}{mod_name:>8}{ch_name:>3} "
                    + "".join(f"{s[k]:>10.1f}" for k in cols)
                )

        det_new = detect_new(ch0, ch1, sr)
        f_vad = turn_features(det_new["caller_turns"], det_new["agent_turns"])
        f_ref = turn_features(ref_caller, ref_agent)
        for k in TURN_FEATURE_NAMES:
            feat_vad_rows[k].append(f_vad[k])
            feat_ref_rows[k].append(f_ref[k])

    print(
        f"{'MEAN':<18}{'':>10}{'':>8}{'':>3} "
        + "".join(f"{np.mean(totals[k]):>10.1f}" for k in cols)
    )

    print("\nturn-feature continuity (VAD turns vs provided turns), pearson r:")
    for k in TURN_FEATURE_NAMES:
        r = pearsonr(feat_vad_rows[k], feat_ref_rows[k]).statistic
        print(f"  {k:<30}{r:>8.3f}")


if __name__ == "__main__":
    main()
