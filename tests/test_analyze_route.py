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
    assert "vot" in body
    assert isinstance(body["vot"]["measurements"], list)
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


def test_analyze_response_advertises_phoneme_alphabet_version() -> None:
    """Loop 3a Phase 0: every response advertises the IPA-mapping version."""
    client = TestClient(app)
    res = client.post(
        "/analyze",
        json={"audio_base64": _sine_wav_b64(220.0, 1.0), "format": "wav"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["phoneme_alphabet_version"] == "1.0.0"


def test_analyze_without_transcript_omits_aligned_phonemes() -> None:
    """Loop 3a Phase 1: when transcript is absent, response is unchanged from
    Phase 0 — aligned_phonemes is null and no related notes appear."""
    client = TestClient(app)
    res = client.post(
        "/analyze",
        json={"audio_base64": _sine_wav_b64(220.0, 1.0), "format": "wav"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["aligned_phonemes"] is None


def test_analyze_with_transcript_but_model_disabled_surfaces_note() -> None:
    """When a transcript is supplied but the phoneme model is disabled (the
    default in tests via VOICEPRINT_DISABLE_PHONEMES=1), the route must not
    crash. Surface the situation as a note so the caller knows alignment
    didn't run."""
    client = TestClient(app)
    res = client.post(
        "/analyze",
        json={
            "audio_base64": _sine_wav_b64(220.0, 1.0),
            "format": "wav",
            "transcript": {
                "format": "ipa",
                "content": "θ s f",
                "expected_language": "en",
            },
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["aligned_phonemes"] is None
    assert any("phoneme model is disabled" in n for n in body["notes"])


def test_analyze_transcript_input_validation_rejects_missing_format() -> None:
    """Pydantic rejects a transcript that's missing the required `format` field."""
    client = TestClient(app)
    res = client.post(
        "/analyze",
        json={
            "audio_base64": _sine_wav_b64(220.0, 1.0),
            "format": "wav",
            "transcript": {"content": "θ s f"},  # missing format
        },
    )
    assert res.status_code == 422, res.text


def test_analyze_with_transcript_populates_aligned_phonemes_when_artifacts_available(
    monkeypatch,
) -> None:
    """Stub the phoneme model with synthetic alignment artifacts to verify the
    `/analyze` route surfaces the `aligned_phonemes` block end-to-end without
    needing the 370 MB model.
    """
    import torch
    from app.services import phonemes

    # Build a synthetic alignment artifact: 3 frames, vocab covering θ/s/blank.
    synthetic_vocab = {"<pad>": 0, "θ": 1, "s": 2}
    V = len(synthetic_vocab)
    logits = torch.zeros(3, V)
    logits[0, synthetic_vocab["θ"]] = 8.0  # frame 0 → θ
    logits[1, synthetic_vocab["<pad>"]] = 8.0  # frame 1 → blank
    logits[2, synthetic_vocab["s"]] = 8.0  # frame 2 → s
    log_probs = torch.log_softmax(logits, dim=-1)

    artifacts = phonemes.AlignmentArtifacts(
        log_probs=log_probs,
        blank_id=0,
        vocab=synthetic_vocab,
        audio_duration_s=0.06,
    )

    def fake_extract(sound, *, expected_language=None, include_alignment_artifacts=False):
        inv = phonemes.PhonemeInventory(counts={}, total_tokens=0, occurrences=[])
        if include_alignment_artifacts:
            inv.alignment_artifacts = artifacts
        return inv

    monkeypatch.setattr(phonemes, "extract_phonemes", fake_extract)

    client = TestClient(app)
    res = client.post(
        "/analyze",
        json={
            "audio_base64": _sine_wav_b64(220.0, 1.0),
            "format": "wav",
            "transcript": {"format": "ipa", "content": "θ s", "expected_language": "en"},
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["aligned_phonemes"] is not None
    assert body["aligned_phonemes"]["transcript_format"] == "ipa"
    assert len(body["aligned_phonemes"]["positions"]) == 2
    classifications = [
        p["match_classification"] for p in body["aligned_phonemes"]["positions"]
    ]
    assert classifications == ["produced", "produced"]
