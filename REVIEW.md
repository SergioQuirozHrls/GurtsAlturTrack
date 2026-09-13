# Project Review — key decisions

A retrospective for the team, now that `/detect` is finished. Not a spec (see
`CLAUDE.md` for that) — this is *why* the finished system looks the way it
does, reconstructed from every commit on `main`.

## How we decide human vs. synthetic

`/detect` never runs a sequence of checks. `features/build.py:call_features()`
computes four feature groups on one call and concatenates them into **one
24-value vector**, which a single `HistGradientBoostingClassifier` (wrapped in
`CalibratedClassifierCV`) scores jointly — the model decides which features
matter at each split, nothing is gated or short-circuited by module.

| Module | # feats | Measures | Why it should generalize to unseen speakers/engines |
|---|---|---|---|
| `features/turns.py` | 14 | Response-latency mean/CV, barge-in rate + overlap depth, post-silence re-entry, turn-duration CV, latency-to-agent-duration ratio | Behavioral pacing, not voice — a new TTS engine still has to react in *some* rhythm, and scale-free (CV/ratio) framing keeps it from just encoding accent or call length |
| `features/acoustic.py` | 5 | Spectral centroid/flatness, RMS mean/std, zero-crossing rate | Broad prosody/spectral shape, added as the CLAUDE.md §9 fallback in case timing alone underperformed |
| `features/liveness.py` | 5 | F0 jitter, amplitude shimmer, HNR proxy, voiced ratio, noise-floor CV — one autocorrelation pitch-tracker pass over ch0 | Vocal-fold micro-instability that real speech has and many vocoders smooth away; independent of behavior, so it survives even if an agent's timing gets human-like |
| `features/vad.py` | — (support) | Derives caller/agent turn boundaries from raw audio | `/detect` only ever gets audio, never `turns/*.json` — VAD *is* the production turn source, so it has to be trustworthy on its own |

`build_dataset()` prefers provided `turns/*.json` over VAD when training (closer
to ground truth), but `app/api.py` always calls VAD — this is the one place
train and serve *could* silently diverge, which is why the VAD rewrite below
was A/B-validated before being trusted.

## Key decisions

| Decision | Why | Commit |
|---|---|---|
| `HistGradientBoostingClassifier`, not lightgbm/xgboost | Those need `libomp.dylib`, absent on a team Mac with no Homebrew | `e40c287` (CLAUDE.md) |
| `numpy`/`scipy.signal`, not librosa | librosa's `soxr` dep has no `win_arm64` wheel | `e40c287` (CLAUDE.md) |
| Barge-in feature swapped from "gap to next turn" to "overlap depth" | Gap-based version hurt val (AUC 0.982→0.975); overlap depth measures how far into the agent's turn the caller talks and actually helped | `deee09b` |
| CV/ratio features replacing raw std (`response_latency_cv`, `latency_agent_dur_ratio_*`) | Scale-free so accent/device/call-length pace differences don't masquerade as "more variable"; removing the ratio feature alone dropped turn-only AUC 0.987→0.968 | `deee09b` |
| Adaptive noise-floor VAD replacing fixed -45 dB threshold | Recording level varies a lot across calls; fixed floor misclassified quiet calls | `cf89794`, validated by `77e3737`'s A/B harness against real `turns/*.json` before being trusted |
| Fail fast on `anon_id` leakage across train/val | CLAUDE.md treats scoring-on-train as a fail condition, not a preference — verified 0 leaks on the real 353-row manifest, but the check now runs every time, not just once | `9a70bec` |
| Sigmoid calibration over isotonic | Fit both, compared real val Brier (0.0257 vs 0.0262) rather than trusting sklearn's general small-sample advice blindly; later re-decided per-run on whichever wins (isotonic now wins post-liveness) | `e908a6f`, re-decided in `train.py`'s per-run comparison |
| Diagnostics added around the calibration choice: bootstrap CI, 5-fold train CV, permutation importance, ablation | A 0.998 val AUC on 71 rows needs scrutiny, not blind trust, before judging | `e1ab60c` |
| Send `max(p, 1-p)` as `confidence`, not raw `P(synthetic)` | Judge's real `check_endpoint.py` recovers `P(synthetic)` as `confidence if is_synthetic else 1-confidence` — sending raw `P(synthetic)` silently flipped the sign for every `is_synthetic=False` call, tanking AUC 0.515 with balanced_accuracy still showing 1.000 | `1475e3e` (vendored the real client), `b886054` (fix) |
| Liveness module added as a 4th, independent signal | Motivated in `HANDOFF.md`: hedges against a future agent nailing human turn-timing; liveness-only matched turns-only (AUC 0.987 each), combined reached AUC 1.000 with real (non-drowned-out) permutation importance on both | `71f459c` |
| Regularization chosen by 5-fold CV on train only, val never touched | Rules out "we tuned against val by hand"; even the most-regularized grid point still hit 0.999 train CV AUC, showing the near-perfect score is a dataset-separability property, not a capacity/overfitting artifact | `52968da` |
| Cloudflare **named** tunnel over the anonymous quick tunnel | Quick tunnel has no uptime guarantee and a URL that changes every restart — got a 502 mid-test; named tunnel verified stable across restarts | `448d258`, `f1f07cf` |
| `/detect` never raises (`except ValueError` / `except Exception` → fixed fallback) | A wrong answer scores; a 500 scores nothing (CLAUDE.md §7) | `0ce626c`, hardened in `50aee66`, `6906efd` |

## Validation discipline

Every `train.py` run prints, beyond AUC/EER: a 95% bootstrap CI on val AUC/EER
(quantifies sampling noise from only 71 val rows), 5-fold CV AUC on train alone
(checks the raw signal isn't a fluke of one split), permutation importance, and
single-feature ablation (checks the score isn't riding on one feature). The
README's "Is val AUC 1.000 trustworthy?" section adds one more pass: turns-only
alone already reaches ~0.965 AUC; acoustic + liveness add the rest and that
lift survives RMS-normalizing `ch0` (ruling out a raw-loudness recording
artifact). What none of this can rule out: `manifest.csv` has no TTS-engine
column, so we can't verify engine diversity in train/val — the real test of
whether 1.000 reflects genuine synthetic-ness detection vs. this dataset's
specific engines is the hidden judging set.

## Explicitly out of scope

No ASR/Whisper/transformers/torch — banned in `CLAUDE.md` §3/§9 to protect the
cross-platform wheel guarantee (same reasoning that killed librosa/lightgbm)
and to avoid touching caller speech content, which the dataset terms forbid
attempting to identify. No real-time streaming (`/detect` is batch clip in,
verdict out) and no speaker identification/diarization beyond the two given
channels — both explicitly excluded in `CLAUDE.md` §8.
