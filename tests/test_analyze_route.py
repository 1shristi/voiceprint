"""End-to-end tests for the /analyze route."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import numpy as np
import parselmouth
from fastapi.testclient import TestClient

from app.main import app

SAMPLE_RATE = 16_000


def _sine_wav_b64(freq_hz: float, duration_s: float = 1.0) -> str:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    samples = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    sound = parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    try:
        sound.save(str(path), "WAV")
        return base64.b64encode(path.read_bytes()).decode("ascii")
    finally:
        path.unlink(missing_ok=True)


def test_analyze_returns_features_for_synthetic_audio() -> None:
    client = TestClient(app)
    res = client.post(
        "/analyze",
        json={"audio_base64": _sine_wav_b64(220.0, 1.0), "format": "wav"},
    )
    assert res.status_code == 200, res.text
    body = res.json()

    assert body["duration_s"] == round(body["duration_s"], 3)
    assert body["duration_s"] > 0.9

    assert body["f0"]["mean_hz"] is not None
    assert 215 <= body["f0"]["mean_hz"] <= 225

    assert "formants" in body
    assert body["vot_ms"] is None  # phase 2c
    assert isinstance(body["notes"], list)


def test_analyze_short_clip_warns_in_notes() -> None:
    client = TestClient(app)
    res = client.post(
        "/analyze",
        json={"audio_base64": _sine_wav_b64(220.0, 0.2), "format": "wav"},
    )
    assert res.status_code == 200, res.text
    notes = res.json()["notes"]
    assert any("short" in n.lower() for n in notes)
