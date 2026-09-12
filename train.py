"""Speaker-disjoint fit + calibration -> model.pkl.

CLAUDE.md sections 2, 4, 9: never score on train, calibrate confidence, print
val AUC/EER + p95 latency. Train/val come straight from manifest.csv's
`split` column -- the dataset is already speaker-disjoint by construction
(no caller appears in both), so that's the split to honor, not one we
re-derive ourselves.

Falls back to a small synthetic placeholder dataset when manifest.csv hasn't
been extracted yet, so model.pkl always exists and app/api.py can be
exercised end to end (CLAUDE.md: "model.pkl is committed").
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from features.build import FEATURE_NAMES, build_dataset

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.csv"
AUDIO_DIR = ROOT / "audio"
TURNS_DIR = ROOT / "turns"
MODEL_PATH = ROOT / "model.pkl"

RANDOM_STATE = 0
AUC_FLOOR = 0.65


def _synthetic_dataset(n_calls: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    n_feats = len(FEATURE_NAMES)
    y = rng.integers(0, 2, size=n_calls)
    x = rng.normal(loc=y[:, None] * 0.5, scale=1.0, size=(n_calls, n_feats))
    split = np.where(rng.random(n_calls) < 0.75, "train", "val")
    return x, y, split


def _eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def main() -> None:
    if MANIFEST_PATH.exists():
        x, y, split = build_dataset(MANIFEST_PATH, AUDIO_DIR, TURNS_DIR)
    else:
        print(
            "manifest.csv not found -- training on a synthetic placeholder "
            "dataset so model.pkl exists. Re-run once the real dataset is "
            "extracted (see CLAUDE.md section 5)."
        )
        x, y, split = _synthetic_dataset()

    x_train, x_val = x[split == "train"], x[split == "val"]
    y_train, y_val = y[split == "train"], y[split == "val"]

    base = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(x_train, y_train)

    val_scores = model.predict_proba(x_val)[:, 1]
    auc = roc_auc_score(y_val, val_scores)
    eer = _eer(y_val, val_scores)

    latencies = []
    for i in range(len(x_val)):
        t0 = time.perf_counter()
        model.predict_proba(x_val[i : i + 1])
        latencies.append(time.perf_counter() - t0)
    p95_ms = float(np.percentile(latencies, 95) * 1000)

    print(f"val AUC: {auc:.3f}  val EER: {eer:.3f}  inference p95: {p95_ms:.2f} ms")
    if auc < AUC_FLOOR:
        print(
            f"AUC below {AUC_FLOOR} -- per CLAUDE.md section 9, keep the turn "
            "features and lean harder on features/acoustic.py rather than "
            "dropping the behavioral signal."
        )

    artifact = {
        "model": model,
        "feature_mean": x_train.mean(axis=0),
        "feature_std": x_train.std(axis=0) + 1e-9,
    }
    with MODEL_PATH.open("wb") as f:
        pickle.dump(artifact, f)


if __name__ == "__main__":
    main()
