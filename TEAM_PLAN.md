# Build plan — skeleton first, then 4-way parallel split

Context: full architecture/stack rules live in `CLAUDE.md` (§3, §4). This doc
just turns that into a concrete file list, an owner per file, and frozen
function signatures at the seams — so all four of us can write code today in
parallel without stepping on each other's files or diverging on features.

## Real dataset schema (confirmed from the challenge repo, not guessed)

`manifest.csv` columns: **`anon_id, label, split, duration_s`** — not
`call_id`/`is_synthetic`/`groups`. `label` is the string `"human"` or
`"synthetic"`; `split` is `"train"` or `"val"`, **already speaker-disjoint by
the organizers** — there is no separate caller-id column, so we honor
`split` directly rather than re-deriving our own group split.

Files: `audio/<anon_id>.wav`, `turns/<anon_id>.json`. The turns JSON shape is
`{"turns": [{"channel": 0, "start": 12.4, "end": 15.1}, ...]}` (channel 0 =
caller, channel 1 = agent) — not `{"caller_turns": [...], "agent_turns": [...]}`.
`features/build.py` already converts between the two; if you write anything
else that reads `turns/*.json` directly, use this shape.

`train.py` splits on `manifest["split"]`, not `GroupShuffleSplit` — already
updated in the skeleton below.

## Why signatures are frozen first

`features/turns.py`, `features/acoustic.py`, and `features/build.py` are
imported by **both** `train.py` and `app/api.py`. If train and serve ever
compute features differently, we lose silently (wrong verdicts, no error).
So step 0 below creates every file with real signatures and a working (if
simple) body, proving the wiring end-to-end before anyone builds the "real"
feature logic. Step 1 is everyone filling in their own file only.

## Step 0 — Skeleton (done — merged to `main`)

```
app/
  api.py              # FastAPI app `api`, POST /detect
features/
  vad.py              # audio -> turn boundaries (server-side, no turns/*.json)
  turns.py            # ch0/ch1 turns -> timing feature dict (PRIMARY SIGNAL)
  acoustic.py         # ch0 audio -> spectral/prosody feature dict (SECONDARY)
  build.py            # manifest + turns/audio -> X, y, split (shared by train + api)
train.py              # speaker-disjoint fit + calibration -> model.pkl
scripts/
  smoke_detect.py      # contract check: keys, types, latency
  demo.py               # prints top contributing features for 2 held-out calls
model.pkl              # placeholder model committed pre-dataset
README.md              # short approach writeup
```

### Frozen interfaces — do not change without telling everyone

```python
# features/vad.py
def detect_turns(ch0: np.ndarray, ch1: np.ndarray, sr: int = 8000) -> dict:
    """{"caller_turns": [(start_s, end_s), ...], "agent_turns": [...]}."""


# features/turns.py
def turn_features(
    caller_turns: list[tuple[float, float]], agent_turns: list[tuple[float, float]]
) -> dict[str, float]:
    """Behavioral timing features. Flat dict, fixed key set = FEATURE_NAMES."""


# features/acoustic.py
def acoustic_features(ch0: np.ndarray, sr: int = 8000) -> dict[str, float]:
    """Spectral/prosody summary stats, scipy.signal only. Flat dict, fixed keys."""


# features/build.py
def call_features(ch0: np.ndarray, ch1: np.ndarray, sr: int = 8000) -> dict[str, float]:
    """THE shared entrypoint both train.py and app/api.py call. Never
    reimplement feature extraction anywhere else."""


def build_dataset(
    manifest_path, audio_dir, turns_dir
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, split). split = manifest's "train"/"val" column,
    already speaker-disjoint by construction -- don't re-derive it."""
```

```python
# app/api.py
@api.post("/detect")
def detect(payload: DetectRequest) -> DetectResponse:
    """base64 stereo 8kHz WAV -> {"is_synthetic": bool, "confidence": float}.
    Never raises: internal errors -> {"is_synthetic": false, "confidence": 0.5}."""
```

Request body key: `{"audio_base64": "<base64-encoded WAV bytes>"}`.

## Step 1 — Who owns what (branch per person, per CLAUDE.md §6 naming)

| Owner | Branch | Files (owns exclusively) | Task |
|---|---|---|---|
| **Gabriel** | `gabriel/vad` | `features/vad.py` | Replace the placeholder energy-threshold VAD with something better on ch0/ch1 (scipy.signal only). Validate against provided `turns/*.json` on a few real calls once the dataset is extracted locally, and note how closely they agree in the PR description (CLAUDE.md §9 risk). |
| **Sergio** | `sergio/turn-features` | `features/turns.py` | Improve/extend the primary behavioral features: response latency, barge-in rate, overlap, post-silence re-entry timing. This is our Originality/Robustness signal — document *why* each feature should transfer to unseen speakers/engines, we'll need that for judge Q&A. |
| **Johana** | `johana/build-and-train` | `features/build.py`, `train.py` | Run `build_dataset` against the real `manifest.csv`/`turns/`/`audio/` once extracted — train/val already come from manifest's `split` column, no re-splitting needed. Verify calibration, sanity-check that no `anon_id` value is duplicated across train and val (it shouldn't be, but confirm), and report real val AUC/EER + p95 latency in your PR. |
| **Angel** | `angel/detect-endpoint` | `app/api.py`, `scripts/smoke_detect.py`, `scripts/demo.py` | Harden the endpoint: confirm mono/short/malformed input never 500s, tune the `is_synthetic` threshold once real calibration exists, keep `smoke_detect.py` and `demo.py` in sync with any contract changes. |
| **Whoever lands first** | — | `features/acoustic.py` | Improve spectral/prosody features **only if** Johana's turn-feature-only AUC comes in under ~0.65 (CLAUDE.md §9). Don't invest here otherwise — it's the secondary signal. |

Each PR: `ruff format . && ruff check .` clean, `python scripts/smoke_detect.py`
passing locally, rebased on `main` before opening. Small scoped commits,
Conventional Commits (`feat:`, `fix:`, `refactor:`, `chore:`).

## Step 2 — Integration order

1. Skeleton is merged — everyone branches from `main` now.
2. Gabriel (`vad.py`) and Sergio (`turns.py`) can work immediately — pure
   functions, testable with synthetic arrays even before the dataset zip is
   extracted.
3. Angel (`api.py`) can already build/test against the current stub
   `call_features` — doesn't need real features to prove the contract.
4. Johana needs the real dataset extracted locally (`audio/`, `manifest.csv`,
   `turns/` from `altur-challenge-audio.zip` — never commit it) to get real
   numbers, but the split/calibration/logging logic is already wired.
5. Once Gabriel + Sergio's PRs land, Johana re-runs `train.py` for real
   metrics and commits the real `model.pkl`. Angel's `api.py` needs zero
   changes — it only ever calls `call_features()`.
6. Acoustic features only get pulled into the merged feature set if the
   turn-only AUC check fails. Don't build that in preemptively.

## Verification

- `python scripts/smoke_detect.py` must pass after every PR.
- `ruff format . && ruff check .` must be clean before every commit.
- Speaker-disjoint AUC/EER + p95 latency get printed by `train.py` and pasted
  into the PR description (no experiment tracking tool, per CLAUDE.md §3).
- Before judging: full CLAUDE.md §7 demo-path rehearsal on the chosen demo
  machine, including the `cloudflared` tunnel fallback (§9).
