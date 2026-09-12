"""Contract check for POST /detect -- exact keys, correct types, and no 500s
on malformed/adversarial input (CLAUDE.md sections 2 and 7; README section
API). Boots the real server as a subprocess and hits it with stdlib urllib,
so no test-only HTTP client dependency is needed. Run with:

    python scripts/smoke_detect.py
"""

from __future__ import annotations

import base64
import io
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 8000
STARTUP_TIMEOUT_S = 15
MIN_DURATION_S = 0.5

FALLBACK = {"is_synthetic": False, "confidence": 0.5}


def _wav_base64(channels: int, sr: int, duration_s: float) -> str:
    rng = np.random.default_rng(0)
    n = int(sr * duration_s)
    cols = [(rng.normal(size=n) * 0.05).astype(np.float32) for _ in range(channels)]
    stereo = np.stack(cols, axis=1)
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _random_wav_bytes() -> bytes:
    rng = np.random.default_rng(0)
    return rng.bytes(8192)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except urllib.error.URLError:
            time.sleep(0.2)
    raise TimeoutError(f"server did not come up at {url} within {timeout_s}s")


def _post(base_url: str, payload: dict | None) -> tuple[int, dict, float]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else b""
    req = urllib.request.Request(
        f"{base_url}/detect",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = json.loads(exc.read().decode("utf-8") or "{}")
    elapsed_ms = (time.perf_counter() - start) * 1000
    return status, body, elapsed_ms


def _check_ok(name: str, status: int, body: dict) -> bool:
    tries = []
    if status != 200:
        tries.append(f"expected 200, got {status}")
    if set(body.keys()) >= {"is_synthetic", "confidence"}:
        if not isinstance(body["is_synthetic"], bool):
            tries.append("is_synthetic is not bool")
        if not isinstance(body["confidence"], (int, float)):
            tries.append("confidence is not numeric")
        elif not 0.0 <= body["confidence"] <= 1.0:
            tries.append(f"confidence out of range: {body['confidence']}")
    else:
        tries.append(f"missing keys: {sorted(body.keys())}")
    return tries


def main() -> None:
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api:api",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
    )
    results = []
    try:
        base_url = f"http://127.0.0.1:{port}"
        _wait_for_server(f"{base_url}/docs", STARTUP_TIMEOUT_S)

        tests = [
            (
                "TEST 1 valid stereo 8kHz WAV",
                {"audio_base64": _wav_base64(2, SAMPLE_RATE, 3.0)},
                "verdict contract",
            ),
            (
                "TEST 2 invalid base64",
                {"audio_base64": "not valid base64!!"},
                FALLBACK,
            ),
            (
                "TEST 3 random bytes pretending to be audio",
                {"audio_base64": base64.b64encode(_random_wav_bytes()).decode("ascii")},
                FALLBACK,
            ),
            (
                "TEST 4 mono WAV",
                {"audio_base64": _wav_base64(1, SAMPLE_RATE, 3.0)},
                FALLBACK,
            ),
            (
                "TEST 5 wrong sample rate (44100 Hz)",
                {"audio_base64": _wav_base64(2, 44100, 3.0)},
                FALLBACK,
            ),
            (
                "TEST 6 very short WAV (0.2s)",
                {"audio_base64": _wav_base64(2, SAMPLE_RATE, 0.2)},
                FALLBACK,
            ),
            (
                "TEST 7 empty audio (empty base64)",
                {"audio_base64": ""},
                FALLBACK,
            ),
            (
                "TEST 8 missing audio_base64 field",
                {},
                422,
            ),
            (
                "TEST 9 non-string audio_base64",
                {"audio_base64": 12345},
                422,
            ),
        ]

        for name, payload, expectation in tests:
            status, body, elapsed = _post(base_url, payload)
            if expectation == "verdict contract":
                problems = _check_ok(name, status, body)
            elif isinstance(expectation, dict):
                problems = []
                if body != expectation:
                    problems.append(f"expected fallback {expectation}, got {body}")
            else:
                problems = []
                if status != expectation:
                    problems.append(f"expected HTTP {expectation}, got {status}")

            if problems:
                results.append(False)
                print(f"FAIL {name}: {'; '.join(problems)}  latency_ms={elapsed:.2f}")
            else:
                results.append(True)
                print(f"OK   {name}: {body}  latency_ms={elapsed:.2f}")
    finally:
        proc.terminate()
        proc.wait(timeout=5)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} smoke tests passed")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
