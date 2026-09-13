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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from features.build import FEATURE_NAMES, build_dataset

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.csv"
AUDIO_DIR = ROOT / "audio"
TURNS_DIR = ROOT / "turns"
MODEL_PATH = ROOT / "model.pkl"

RANDOM_STATE = 0
AUC_FLOOR = 0.65

# Regularization grid for HistGradientBoostingClassifier, searched by CV on
# the train split only (val is never touched during this search -- tuning
# against val would just be overfitting the val metric by hand instead of
# fixing anything). At ~280 train rows and 19 features the default settings
# have enough capacity to carve out a near-perfect decision boundary on a
# small held-out set; these ranges bias the search toward shallower, less
# confident trees (fewer leaves, larger leaves, stronger L2) to see whether a
# more conservative model is still competitive on val, which is a better
# signal than one point estimate at default settings.
REG_PARAM_GRID = {
    "max_leaf_nodes": [7, 15, 31],
    "min_samples_leaf": [20, 40, 60],
    "l2_regularization": [0.0, 1.0, 10.0],
    "max_iter": [50, 100],
}


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


def _ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over fixed-width bins (a stricter,
    weighted-by-bin-mass complement to the Brier score / reliability curve
    above -- Brier can look fine while individual bins are still off)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_score, bin_edges[1:-1]), 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        confidence = y_score[mask].mean()
        accuracy = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(accuracy - confidence)
    return float(ece)


def _bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = RANDOM_STATE,
) -> tuple[float, float, int]:
    """95% percentile bootstrap CI for a val-set metric. With only 71 val
    rows, a single AUC/EER point estimate hides how much sampling noise is
    in it -- resampling (with replacement) from the same 71 rows shows the
    range a different held-out draw could plausibly have landed in."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b, s_b = y_true[idx], y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        stats.append(metric_fn(y_b, s_b))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi), len(stats)


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

    search = GridSearchCV(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        REG_PARAM_GRID,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        refit=False,
    )
    search.fit(x_train, y_train)
    best_params = search.best_params_
    default_params = {
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_iter": 100,
    }
    default_idx = search.cv_results_["params"].index(default_params)
    default_cv_auc = search.cv_results_["mean_test_score"][default_idx]
    print(
        f"regularization search (5-fold CV AUC on train only): best "
        f"{best_params} -> CV AUC {search.best_score_:.4f} "
        f"(sklearn defaults scored {default_cv_auc:.4f})"
    )

    # With only ~282 train rows, CalibratedClassifierCV(cv=3)'s internal
    # calibration slice is ~94 samples per fold. sklearn's own calibration
    # guide warns isotonic "is not advised" below ~1000 calibration samples
    # because it tends to overfit, and recommends sigmoid (Platt) instead:
    # https://scikit-learn.org/stable/modules/calibration.html#calibrating-a-classifier
    # We don't take that on faith -- both are fit and compared on our real,
    # speaker-disjoint val split (Brier score + reliability curve) and the
    # winner is what ships in model.pkl.
    candidates: dict[str, dict] = {}
    for method in ("isotonic", "sigmoid"):
        base = HistGradientBoostingClassifier(random_state=RANDOM_STATE, **best_params)
        candidate_model = CalibratedClassifierCV(base, method=method, cv=3)
        candidate_model.fit(x_train, y_train)

        scores = candidate_model.predict_proba(x_val)[:, 1]
        brier = brier_score_loss(y_val, scores)
        ece = _ece(y_val, scores)
        mean_pred, frac_pos = calibration_curve(
            y_val, scores, n_bins=5, strategy="quantile"
        )
        candidates[method] = {
            "model": candidate_model,
            "scores": scores,
            "brier": brier,
            "ece": ece,
            "curve": list(zip(mean_pred, frac_pos)),
        }
        curve_str = ", ".join(
            f"({p:.2f}->{o:.2f})" for p, o in candidates[method]["curve"]
        )
        print(
            f"[{method}] val AUC: {roc_auc_score(y_val, scores):.3f}  "
            f"val EER: {_eer(y_val, scores):.3f}  Brier: {brier:.4f}  "
            f"ECE(10-bin): {ece:.4f}  reliability (pred->observed): {curve_str}"
        )

    # ECE disagrees with Brier here (isotonic's ECE is lower), but at 71 val
    # rows split into 10 fixed-width bins that's ~7 samples/bin -- noisier
    # than the Brier score, which is a proper scoring rule over every point
    # rather than a bin-count-dependent statistic. We still defer to Brier
    # (matches the sklearn small-sample-calibration guidance cited above);
    # ECE is printed as a caveat, not a second vote.
    best_method = min(candidates, key=lambda m: candidates[m]["brier"])
    print(f"calibration choice: {best_method} (lower val Brier score wins)")
    model = candidates[best_method]["model"]
    val_scores = candidates[best_method]["scores"]

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

    # --- Diagnostics below: none of this feeds model.pkl, it's here so a
    # suspiciously high val AUC (0.998 on the real data) doesn't get taken at
    # face value before judging. Three questions: (1) how much sampling noise
    # is in that one AUC/EER estimate from only 71 val rows, (2) is the raw
    # signal stable across resplits of train, (3) is the score being carried
    # by one feature (an artifact/leak risk) or spread across the behavioral
    # signal as intended.
    auc_lo, auc_hi, n_boot_ok = _bootstrap_ci(y_val, val_scores, roc_auc_score)
    eer_lo, eer_hi, _ = _bootstrap_ci(y_val, val_scores, _eer)
    print(f"val AUC 95% bootstrap CI (n={n_boot_ok}): [{auc_lo:.3f}, {auc_hi:.3f}]")
    print(f"val EER 95% bootstrap CI (n={n_boot_ok}): [{eer_lo:.3f}, {eer_hi:.3f}]")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE, **best_params),
        x_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
    )
    print(
        f"train 5-fold CV AUC (uncalibrated, stability check only -- not a "
        f"substitute for the real val split): {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}"
    )

    perm = permutation_importance(
        model, x_val, y_val, scoring="roc_auc", n_repeats=30, random_state=RANDOM_STATE
    )
    order = np.argsort(perm.importances_mean)[::-1]
    print("permutation importance on val (AUC drop when shuffled), top 5:")
    for i in order[:5]:
        print(
            f"  {FEATURE_NAMES[i]}: {perm.importances_mean[i]:.4f} "
            f"+/- {perm.importances_std[i]:.4f}"
        )

    top_idx = int(order[0])
    keep_cols = np.ones(x.shape[1], dtype=bool)
    keep_cols[top_idx] = False
    ablated = HistGradientBoostingClassifier(random_state=RANDOM_STATE, **best_params)
    ablated.fit(x_train[:, keep_cols], y_train)
    ablated_auc = roc_auc_score(y_val, ablated.predict_proba(x_val[:, keep_cols])[:, 1])
    print(
        f"ablation -- drop top feature '{FEATURE_NAMES[top_idx]}': "
        f"val AUC {ablated_auc:.3f} (full model {auc:.3f})"
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
