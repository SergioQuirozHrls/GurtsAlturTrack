"""Prints top contributing features for two held-out calls, one human and one
synthetic -- CLAUDE.md section 7, demo step 4. "Contributing" here means the
features furthest (in z-score) from the training distribution, since
HistGradientBoostingClassifier has no built-in feature_importances_.

Run with: python scripts/demo.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.build import FEATURE_NAMES, call_features

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "manifest.csv"
AUDIO_DIR = ROOT / "audio"
MODEL_PATH = ROOT / "model.pkl"
TOP_N = 5


def _pick_demo_calls() -> list[tuple[str, np.ndarray, np.ndarray, int, int]]:
    manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8")
    val_calls = manifest[manifest["split"] == "val"]
    picks = []
    for label_str, label in (("human", 0), ("synthetic", 1)):
        row = val_calls[val_calls["label"] == label_str].iloc[0]
        anon_id = str(row["anon_id"])
        data, sr = sf.read(AUDIO_DIR / f"{anon_id}.wav", always_2d=True)
        picks.append((anon_id, data[:, 0], data[:, 1], sr, label))
    return picks


def main() -> None:
    with MODEL_PATH.open("rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    mean = artifact["feature_mean"]
    std = artifact["feature_std"]

    n_expected = getattr(model, "n_features_in_", len(FEATURE_NAMES))
    if n_expected != len(FEATURE_NAMES):
        print(
            f"model.pkl was trained on {n_expected} features but current "
            f"features/build.py produces {len(FEATURE_NAMES)}. Retrain with "
            "the current main before running the demo."
        )
        sys.exit(1)

    if not MANIFEST_PATH.exists():
        print(
            "manifest.csv not found -- extract the dataset to see real "
            "held-out calls (CLAUDE.md section 5). Nothing to demo yet."
        )
        return

    for anon_id, ch0, ch1, sr, label in _pick_demo_calls():
        feats = call_features(ch0, ch1, sr)
        x = np.array([feats[name] for name in FEATURE_NAMES])
        confidence = float(model.predict_proba(x.reshape(1, -1))[0, 1])

        z_scores = (x - mean) / std
        top_idx = np.argsort(-np.abs(z_scores))[:TOP_N]

        print(
            f"\nanon_id={anon_id}  true_label={'synthetic' if label else 'human'}"
            f"  predicted_confidence={confidence:.3f}"
        )
        for i in top_idx:
            print(f"  {FEATURE_NAMES[i]:<28} value={x[i]:+.3f}  z={z_scores[i]:+.2f}")


if __name__ == "__main__":
    main()
