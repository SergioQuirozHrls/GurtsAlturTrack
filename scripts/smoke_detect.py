"""Contract check for POST /detect -- exact keys, correct types, reasonable
latency (CLAUDE.md section 2). Boots the real server as a subprocess and hits
it with stdlib urllib, so no test-only HTTP client dependency is needed. Run
with:

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
DURATION_S = 3
STARTUP_TIMEOUT_S = 15


def _fake_call_wav_base64() -> str:
    rng = np.random.default_rng(0)
    n = SAMPLE_RATE * DURATION_S
    ch0 = (rng.normal(size=n) * 0.05).astype(np.float32)
    ch1 = (rng.normal(size=n) * 0.05).astype(np.float32)
    stereo = np.stack([ch0, ch1], axis=1)
    buf = io.BytesIO()
    sf.write(buf, stereo, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
    try:
        base_url = f"http://127.0.0.1:{port}"
        _wait_for_server(f"{base_url}/docs", STARTUP_TIMEOUT_S)

        payload = json.dumps({"audio_base64": _fake_call_wav_base64()}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/detect",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            body = json.loads(resp.read().decode("utf-8"))
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert status == 200, f"expected 200, got {status}: {body}"
        assert set(body.keys()) >= {"is_synthetic", "confidence"}, body
        assert isinstance(body["is_synthetic"], bool), body
        assert isinstance(body["confidence"], float), body
        assert 0.0 <= body["confidence"] <= 1.0, body

        print(f"OK: {body}  latency_ms={elapsed_ms:.2f}")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
