"""POST /detect -- CLAUDE.md section 2 contract: base64 stereo 8kHz WAV in,
{"is_synthetic": bool, "confidence": float} out. Exact keys, no auth. Never
raises: a wrong answer scores, a 500 scores nothing (CLAUDE.md section 7).
"""

from __future__ import annotations

import base64
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

api = FastAPI()


class DetectRequest(BaseModel):
    audio_base64: str


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float


@api.post("/detect")
def detect(payload: DetectRequest) -> DetectResponse:
    start = time.perf_counter()
    try:
        wav_bytes = base64.b64decode(payload.audio_base64)
        data, sr = sf.read(io.BytesIO(wav_bytes), always_2d=True)
        if data.shape[1] < 2:
            ch0 = ch1 = data[:, 0]
        else:
            ch0, ch1 = data[:, 0], data[:, 1]

        feats = call_features(ch0, ch1, sr)
        x = np.array([[feats[name] for name in FEATURE_NAMES]])
        confidence = float(_MODEL.predict_proba(x)[0, 1])
        response = DetectResponse(is_synthetic=confidence >= 0.5, confidence=confidence)
    except Exception:
        logger.exception("detect() failed, returning safe default")
        response = DetectResponse(is_synthetic=False, confidence=0.5)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("detect latency_ms=%.2f", elapsed_ms)
    return response
