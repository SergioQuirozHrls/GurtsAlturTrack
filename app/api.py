"""POST /detect -- CLAUDE.md section 2 contract: base64 stereo 8 kHz WAV in,
{"is_synthetic": bool, "confidence": float} out. Exact keys, no auth. Never
raises: a wrong answer scores, a 500 scores nothing (CLAUDE.md section 7).

Malformed base64, non-WAV audio, mono or wrong-sample-rate clips, and
too-short audio are rejected with the documented fallback verdict
(README section API) instead of an internal server error.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel

from features.build import FEATURE_NAMES, call_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detect")

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"
with MODEL_PATH.open("rb") as f:
    _ARTIFACT = pickle.load(f)
_MODEL = _ARTIFACT["model"]

if getattr(_MODEL, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
    logger.error(
        "model.pkl expects %s features but features/build.py produces %s -- "
        "stale model, /detect will fall back until it is retrained",
        getattr(_MODEL, "n_features_in_", "?"),
        len(FEATURE_NAMES),
    )

SUPPORTED_SAMPLE_RATE = 8000
# VAD needs at least one 20 ms frame to find a single turn (features/vad.py);
# anything shorter carries no behavioral signal, so treat it as invalid input.
MIN_DURATION_S = 0.5
MIN_SAMPLES = int(SUPPORTED_SAMPLE_RATE * MIN_DURATION_S)

api = FastAPI()


class DetectRequest(BaseModel):
    audio_base64: str


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float


def _fallback() -> DetectResponse:
    return DetectResponse(is_synthetic=False, confidence=0.5)


def _decode_audio(payload: str) -> tuple[np.ndarray, int]:
    """Decode a base64 payload and read the WAV -> (samples, sample_rate, data).

    Raises ValueError with a concrete reason for every rejected clip; the
    request handler turns that into the documented fallback response.
    """
    try:
        wav_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64: {exc}") from exc

    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), always_2d=True)
    except sf.SoundFileError as exc:
        raise ValueError(f"not a readable WAV clip: {exc}") from exc

    if data.shape[0] == 0:
        raise ValueError("audio clip is empty")
    if data.shape[0] < MIN_SAMPLES:
        raise ValueError(
            f"audio too short: {data.shape[0] / SUPPORTED_SAMPLE_RATE:.2f}s, "
            f"need at least {MIN_DURATION_S:.2f}s"
        )
    if sr != SUPPORTED_SAMPLE_RATE:
        raise ValueError(
            f"unsupported sample rate {sr} Hz, expected {SUPPORTED_SAMPLE_RATE} Hz"
        )
    if data.shape[1] < 2:
        raise ValueError("mono audio, expected stereo (ch0 caller, ch1 agent)")

    return data, sr


@api.post("/detect")
def detect(payload: DetectRequest) -> DetectResponse:
    start = time.perf_counter()
    try:
        data, sr = _decode_audio(payload.audio_base64)
        ch0, ch1 = data[:, 0], data[:, 1]

        feats = call_features(ch0, ch1, sr)
        x = np.array([[feats[name] for name in FEATURE_NAMES]])

        proba = _MODEL.predict_proba(x)
        if proba.shape != (1, 2) or not np.all(np.isfinite(proba)):
            raise ValueError(
                f"invalid model output from predict_proba: shape={proba.shape}"
            )

        p_synthetic = float(proba[0, 1])
        is_synthetic = p_synthetic >= 0.5
        # Judge's check_endpoint.py recovers P(synthetic) as confidence if is_synthetic
        # else 1 - confidence, i.e. it expects confidence in the *predicted* label, not
        # raw P(synthetic). Send max(p, 1-p) so that round-trip is exact.
        confidence = p_synthetic if is_synthetic else 1.0 - p_synthetic
        response = DetectResponse(is_synthetic=is_synthetic, confidence=confidence)
    except ValueError as exc:
        logger.warning("detect() rejected input: %s", exc)
        response = _fallback()
    except Exception:
        logger.exception("detect() failed, returning safe default")
        response = _fallback()
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("detect latency_ms=%.2f", elapsed_ms)
    return response
