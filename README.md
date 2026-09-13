# Defend the Bank Against Voice Deepfakes

CPU-only HTTP service that takes a recorded Mexican-Spanish bank support call
(stereo 8 kHz WAV, ch0 = caller, ch1 = AI agent) and decides whether the
**caller** is a real human or a synthetic voice.

## Approach

We classify primarily on **how the caller behaves in the conversation**, not
on voice timbre:

- **Turn-timing features** (`features/turns.py`) built from ch0↔ch1 turns:
  response-latency mean/variance, barge-in rate, overlap rate, and
  post-silence re-entry timing. Humans recover from interruptions and silence
  messily and inconsistently; scripted TTS+LLM pipelines recover on a
  tighter, more consistent schedule. This transfers to unseen speakers and
  TTS engines because it's not a voice-quality signal.
- **Acoustic features** (`features/acoustic.py`), a secondary spectral/prosody
  summary (scipy.signal only), added to the model only if the turn-feature-only
  validation AUC falls short.
- **Liveness features** (`features/liveness.py`), a pitch-tracker-based signal
  independent of behavior or broad prosody: F0 jitter, amplitude shimmer, an
  HNR proxy, voiced ratio, and noise-floor stability. Real vocal folds wobble
  cycle-to-cycle in ways many vocoders smooth over — this stays useful even if
  a synthetic caller's turn-timing gets good enough to blend in.
- **Speaker-disjoint validation always** — no caller appears in both train and
  val, so the reported AUC/EER reflects generalization, not memorization.
- A calibrated confidence score (train.py fits isotonic and sigmoid on held-out
  val and keeps whichever has the lower Brier score) so the returned
  `confidence` is meaningful, not just a raw model score.

At inference time, turn boundaries are derived from the raw audio itself
(`features/vad.py`) since `/detect` only ever receives audio, never the
offline `turns/*.json`.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate   # or py -3.14 on Windows
pip install -r requirements.txt

# optional: with the dataset extracted into the repo root (audio/, manifest.csv, turns/)
python train.py

python scripts/smoke_detect.py
uvicorn app.api:api --host 0.0.0.0 --port 8000
```

## API

`POST /detect`

Request (the judge also sends `call_id`, `sample_rate`, `channels`; we only read `audio_base64`):

```json
{"call_id": "...", "audio_base64": "<base64-encoded stereo 8kHz WAV bytes>", "sample_rate": 8000, "channels": 2}
```

Response (`confidence` is our confidence in the returned `is_synthetic` verdict, i.e.
`max(p_synthetic, 1 - p_synthetic)`, not raw `P(synthetic)` — this matches how the
judge's `check_endpoint.py` recovers `P(synthetic)` for AUC/calibration):

```json
{"is_synthetic": false, "confidence": 0.88}
```

Malformed, mono, or short input never causes a 500 — the handler falls back
to `{"is_synthetic": false, "confidence": 0.5}` and logs the error.

## Demo

```bash
python scripts/demo.py
```

Prints, for one human and one synthetic held-out call, the model's predicted
confidence and the features that deviate most from the training distribution
— i.e. what drove the verdict.

See `CLAUDE.md` for full architecture, constraints, and team workflow, and
`TEAM_PLAN.md` for the current task split.
