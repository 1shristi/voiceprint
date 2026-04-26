"""Phoneme detection tests.

These hit the actual Wav2Vec2 model and download ~370 MB on first run.
Marked `slow` so the default test run skips them.

To run: `pytest -m slow tests/test_phonemes.py`
"""

from __future__ import annotations

import os

import numpy as np
import parselmouth
import pytest


SAMPLE_RATE = 16_000


@pytest.fixture(autouse=True)
def _enable_phonemes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOICEPRINT_DISABLE_PHONEMES", raising=False)


@pytest.mark.slow
def test_phoneme_extraction_on_synthetic_signal_runs() -> None:
    """A pure tone won't yield meaningful phonemes, but we verify the pipeline runs."""
    from app.services import phonemes

    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    samples = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    sound = parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)

    inv = phonemes.extract_phonemes(sound)
    assert isinstance(inv.counts, dict)
    assert inv.total_tokens >= 0


@pytest.mark.slow
def test_phoneme_disabled_via_env() -> None:
    """When VOICEPRINT_DISABLE_PHONEMES=1, extraction should short-circuit to empty."""
    os.environ["VOICEPRINT_DISABLE_PHONEMES"] = "1"
    try:
        from app.services import phonemes

        t = np.linspace(0, 0.5, SAMPLE_RATE // 2, endpoint=False)
        samples = 0.5 * np.sin(2 * np.pi * 220.0 * t)
        sound = parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)

        inv = phonemes.extract_phonemes(sound)
        assert inv.counts == {}
        assert inv.total_tokens == 0
    finally:
        del os.environ["VOICEPRINT_DISABLE_PHONEMES"]
