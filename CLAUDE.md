# CLAUDE.md — HackMTY26 / Altur: Defend the Bank Against Voice Deepfakes

Instructions for Claude Code sessions in this repo. Read fully before writing code.

## 1. Project summary

We build a CPU-only HTTP service that receives a recorded Mexican-Spanish bank
support call (stereo 8 kHz WAV, ch0 = caller, ch1 = AI agent) and decides whether the
**caller** is a real human or a synthetic voice. Users are bank contact centers whose
phone line is the only channel many LatAm customers have, and which currently has no
reliable way to know a voice belongs to a person. Our bet against the track's judging
criteria: classify on **how the caller behaves in the conversation** (response latency,
recovery from interruptions/overlap/silence) rather than on voice timbre alone —
behavior transfers to unseen speakers and TTS engines (Robustness), is a signal beyond
an off-the-shelf audio classifier (Originality), and is cheap enough to answer in
milliseconds on telephony audio a bank already has (Latency, Feasibility).

## 2. Track alignment

Judged criteria, in our words:

- **Robustness** — hidden test set uses callers, voices and engines in *neither* split.
  Anything that memorizes speakers or a specific TTS scores zero here.
- **Originality** — "does it use signals beyond an off-the-shelf audio classifier?"
- **Technical depth** — well executed, and *we can explain why it works* in the 15-min slot.
- **Feasibility** — deployable on real phone audio (8 kHz, noisy, no GPU assumptions).
- **Latency** — how fast we reach a verdict with reasonable confidence.

Track guidance we obey: *"depth beats breadth — one signal done well outscores three
that half-work"*, and *"you get both sides of the conversation for a reason."*

Three choices that score directly:

1. **Both-channel timing features** (ch1 agent turns → ch0 caller reaction). Barge-in
   recovery time, response-latency mean/variance, overlap rate, post-silence re-entry.
   Humans recover messily and inconsistently; TTS+LLM pipelines recover consistently.
   → Originality + Robustness.
2. **Speaker-disjoint validation, always.** Never score on `train`. Report val metrics
   grouped so no caller leaks. Prefer features with low variance across devices/accents.
   → Robustness.
3. **Calibrated `confidence`** (isotonic/Platt on val) + p50/p95 latency printed by the
   server on every request. → Latency + tie-breaks + Technical depth.

### Non-negotiable (fail conditions, not preferences)

- [ ] `POST /detect` accepts base64 stereo 8 kHz WAV, returns `{"is_synthetic": bool, "confidence": float}`.
      `is_synthetic` required. Exact key names. No auth, no extra required fields.
- [ ] Endpoint reachable for the **entire** 15-minute judging visit.
- [ ] Short `README.md` at repo root explaining the approach (separate from this file).
- [ ] Dataset terms: do **not** redistribute audio, do **not** attempt to identify any
      caller, do not commit `audio/` or any WAV to git.

## 3. Tech stack

**The team is mixed-OS: macOS + Windows.** Every pin in `requirements.txt` has a
verified wheel on `win_amd64`, `win_arm64`, macOS arm64 and macOS x86_64. Nothing gets
added without passing that check — the command is in `requirements.txt`.

| Piece | Choice | Why |
| --- | --- | --- |
| Language | Python 3.14 | Cross-platform wheels confirmed. On macOS use `python3` (python.org build) — **not** `/usr/bin/python3`, which is 3.9.6. On Windows install from python.org and use `py -3.14`. |
| Server | FastAPI + plain uvicorn | Fastest path to one JSON POST endpoint. Never `uvicorn[standard]` — uvloop is not available on Windows. |
| Audio I/O | `soundfile` | Reads stereo 8 kHz PCM per channel without resampling; bundles libsndfile on all four platforms. |
| Features | `numpy` + `scipy.signal` only | **No librosa** — its dep `soxr` has no `win_arm64` wheel, so it breaks ARM Windows laptops. `scipy.signal.stft`/`welch` covers every spectral summary we need. |
| Model | sklearn `HistGradientBoostingClassifier` | **No lightgbm/xgboost** — wheels need `libomp.dylib`, absent on the macOS machine (no Homebrew). This is the same algorithm class, zero extra deps. |
| Data | `pandas` | `manifest.csv` handling. |
| Lint/format | `ruff` (format + check) | One tool, identical output on both OSes. |

No torch, no transformers, no Whisper, no ASR in v1 — see §9.

**Cross-platform code rules (mandatory, not style):**

- Use `pathlib.Path`, never string concatenation or `/`-joined literals.
- Open text files with explicit `encoding="utf-8"` — Windows defaults to cp1252 and the
  data is Spanish, so accents will corrupt or crash without it.
- No shell-outs to `ls`, `unzip`, `sox`, `ffmpeg`, or `curl` from Python. Pure Python only.
- Don't rely on filename case-insensitivity; match `anon_id` exactly as `manifest.csv` gives it.

## 4. Architecture

Single process. No services, no queue, no DB.

```
manifest.csv + turns/*.json + audio/*.wav
        │
        ├── features/turns.py     ch0↔ch1 timing features  ← PRIMARY SIGNAL
        ├── features/acoustic.py  ch0 spectral/prosody summary stats, scipy.signal only (secondary)
        └── features/build.py     -> X (n_calls × n_feats), y, groups=caller
        │
   train.py  ── speaker-disjoint fit + calibration ──> model.pkl (committed)
        │
   app/api.py  POST /detect: b64 → wav → same feature fns → model.pkl → JSON
```

Rules:

- Feature extraction is **one code path** shared by training and serving. If `train.py`
  and `api.py` compute features differently, we lose silently. Import, never re-implement.
- The server derives turns from the audio itself (`features/vad.py`) because `/detect`
  gets audio only — no `turns/*.json` at inference. Validate offline that VAD-derived
  turns reproduce the provided-JSON features closely before trusting val scores.
- `model.pkl` is committed so a fresh clone can serve without training.
- No abstract base classes, no plugin registry, no config framework.

## 5. Setup & run commands

Same steps on both OSes; only venv activation and the LAN-IP lookup differ.

**macOS**

```bash
git clone <TEAM_GITHUB_REMOTE_URL>   # PLACEHOLDER: fill once remote exists
cd hackmty26-altur
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
git clone <TEAM_GITHUB_REMOTE_URL>   # PLACEHOLDER: fill once remote exists
cd hackmty26-altur
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1     # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
pip install -r requirements.txt
```

**Both, after activation**

```bash
# dataset (NOT in git): get altur-challenge-audio.zip from the challenge Releases page,
# put it in the repo root, then — works on macOS and Windows 10+ alike:
tar -xf altur-challenge-audio.zip          # → audio/, plus manifest.csv, turns/

python train.py     # speaker-disjoint val AUC/EER + p95 latency, writes model.pkl
python scripts/smoke_detect.py             # contract check: keys, types, latency
ruff format . && ruff check .              # before every commit

# serve — bind 0.0.0.0 so judges on the LAN can reach it
uvicorn app.api:api --host 0.0.0.0 --port 8000
```

No test suite beyond `scripts/smoke_detect.py`. Keep it passing.

`scripts/smoke_detect.py` and `scripts/demo.py` must run unchanged on both OSes —
no bash-isms, no hardcoded `/tmp`, use `tempfile` and `pathlib`.

## 6. Git workflow & code consistency

Team: **Sergio, Angel, Johana, Gabriel** — all four developers. Default branch `main`.

- **Never commit to `main` directly.** Branch names: `<firstname>/<short-feature>`
  → `sergio/turn-features`, `angel/detect-endpoint`, `johana/calibration`, `gabriel/vad`.
- **Conventional Commits**, present tense, short subject:
  `feat: add barge-in recovery features` / `fix: handle mono wav in /detect` /
  `chore: pin requirements` / `refactor: share feature path between train and api`.
- **Small, logically scoped commits.** One idea per commit. Big dumps are unrevertable
  at 3am when the demo breaks.
- **`git pull --rebase origin main` before starting work and again before opening a PR.**
  Catch conflicts early, not during judging.
- **Every merge to `main` goes through a PR.** Lightweight review — a teammate's glance
  or a deliberate self-review is enough, no formal approval gate — but nothing merges
  unreviewed, and nothing merges without `scripts/smoke_detect.py` passing locally.
- **Run `ruff format . && ruff check .` before every commit.** Style is not a discussion.
- **Line endings are handled by `.gitattributes` (`* text=auto eol=lf`).** Do not delete it
  and do not "fix" line endings by hand. Windows teammates run once:
  `git config --global core.autocrlf input`. Without this, a Windows commit shows every
  line of a file as changed and real edits become invisible in review.
- **A PR that only differs by line endings or formatting gets rebased, not merged.**
  Mixed-OS whitespace churn is how we lose a real change during a conflict resolution.
- **Merge conflicts:** talk to whoever touched the file last (`git log -1 <file>`).
  Never silently overwrite a teammate's logic.
- **Never force-push `main`. Never rewrite shared history.**
- Files that must not enter git: `audio/`, `*.wav`, `.venv/`, `__pycache__/`, any
  extracted transcript of caller speech. Keep `.gitignore` current.

## 7. Demo path (top priority — protect over any feature)

The flow that must never break during the 15-minute judge visit:

1. **Demo machine: PLACEHOLDER — pick ONE laptop (whose?) and rehearse on it.** It serves;
   the other three don't. On venue wifi with `uvicorn app.api:api --host 0.0.0.0 --port 8000`
   already running before judges arrive.
2. Read that machine's LAN IP and hand judges `http://<IP>:8000/detect`.
   - macOS: `ipconfig getifaddr en0`
   - Windows: `ipconfig` → IPv4 Address of the Wi-Fi adapter
   - Windows also needs the port opened once: Windows Defender Firewall will prompt on
     first bind — **click Allow on private networks**, or judges get a silent timeout.
     Rehearse this; it is the most common way this demo fails.
   → **PLACEHOLDER: confirm judges will be on the same network segment.**
3. Judges run their benchmark against `/detect` with the hidden set; server stays up,
   logs per-request latency, never 500s (malformed/mono/short input → valid JSON verdict).
4. We run `python scripts/demo.py` on 2 held-out val calls — one human, one synthetic —
   showing the top contributing features per verdict.
5. We explain the behavioral signal in one sentence each: response-latency variance,
   barge-in recovery, overlap handling — and why it survives unseen voices.
6. We show speaker-disjoint val metrics + the calibration curve.

Rules protecting it:

- **No merges to `main` in the 60 minutes before judging.** Freeze, then only hotfixes
  that a teammate has watched pass the smoke test.
- `/detect` never raises. Wrap the whole handler: on any internal error return
  `{"is_synthetic": false, "confidence": 0.5}` and log the traceback. A wrong answer
  scores; a 500 scores nothing.
- If a feature is not on this path and it is late, the answer is no.

## 8. Constraints & conventions

- Team: 4 developers (Sergio, Angel, Johana, Gabriel). Task split decided after the
  first end-to-end path exists — build the skeleton first, parallelize second.
- **Mixed OS: macOS + Windows.** Code and scripts must run on both (§3 rules). Anything
  that only works on the author's machine is broken, however well it demos locally.
- **Deadline: PLACEHOLDER — fill exact submission/judging date + time.**
- **Judging slot: PLACEHOLDER — fill assigned 15-min window.**
- Time budget: get `/detect` returning a real verdict from a trained model within the
  first third of the event. Everything after that is accuracy work on a working system.

Explicitly out of scope — do not build:

- Auth, rate limiting, HTTPS, Docker, CI, cloud deploy, DB, migrations.
- Frontend/dashboard. `scripts/demo.py` printing to a terminal is the UI.
- Real-time streaming detection. Batch clip in, verdict out.
- Speaker identification/diarization beyond the two given channels.
- Any attempt to identify callers (dataset terms).

Shortcuts that are fine here:

- Hardcode paths, thresholds, and the port. Config = module constants.
- Commit `model.pkl`. No experiment tracking; log metrics to stdout and paste in PRs.
- Global model loaded at import time.
- Bare `except Exception` in the request handler only (see §7).

## 9. Known risks / fallback plan

**Riskiest bet — the behavioral signal may not separate classes.** If the synthetic
callers were driven by a low-latency pipeline, response-timing features may overlap
with humans.
→ Check first: train on turn features alone, print speaker-disjoint val AUC. If AUC is
under ~0.65 after honest feature work, keep the behavioral features and add the acoustic
feature set (§4 `features/acoustic.py`) into the same model. Do not throw the behavioral
path away — it is our Originality score.

**Feature mismatch train vs serve.** `turns/*.json` exists offline but not at inference;
our VAD must reproduce it. → Validate VAD-derived features against provided turns early.
If they diverge badly, train on **VAD-derived features only**, ignoring `turns/*.json`,
so train and serve match by construction. Accept lower val numbers for real hidden-set
performance.

**Local-network-only hosting is our single point of failure** (chosen tradeoff). Venue
wifi, client isolation, or judges on another VLAN and we score nothing.
→ Prepare the fallback *before* judging day, do not improvise it: install a tunnel
(`cloudflared tunnel --url http://localhost:8000`) and confirm one end-to-end request
through the public URL once. Keep the URL written down. Also verify a phone hotspot
works as a backup network.

**Environment drift across 4 mixed-OS laptops.** Two dependency traps are already
confirmed, not hypothetical: `lightgbm` fails on the macOS machine (no `libomp`), and
`librosa` fails on ARM Windows (its `soxr` dep has no `win_arm64` wheel). Both are banned
in `requirements.txt`.
→ Rule: any new dependency must pass the two `pip download --platform win_amd64/win_arm64`
checks documented at the top of `requirements.txt` **before** it lands in a PR. If a
teammate's machine can't install something, we drop the dependency, not the deadline.
→ First 15 minutes of the hackathon: all four teammates clone, install, and run
`python scripts/smoke_detect.py`. Find environment breakage while it's cheap, not at hour 20.
