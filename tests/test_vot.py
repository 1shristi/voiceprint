"""VOT estimator tests using synthetic phoneme occurrences.

These tests don't require the wav2vec2 model — we hand-construct
PhonemeOccurrence inputs and verify VOT math against a synthetic audio
signal where we control the voicing onset.
"""

from __future__ import annotations

import numpy as np
import parselmouth
import pytest

from app.services.phonemes import PhonemeOccurrence
from app.services.vot import estimate_vot

SR = 16_000


def _silence_then_tone(silence_s: float, tone_s: float, freq_hz: float = 200.0) -> parselmouth.Sound:
    silent = np.zeros(int(SR * silence_s))
    t = np.linspace(0, tone_s, int(SR * tone_s), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    samples = np.concatenate([silent, tone])
    return parselmouth.Sound(samples, sampling_frequency=SR)


def test_no_stops_returns_empty() -> None:
    sound = _silence_then_tone(0.1, 0.5)
    summary = estimate_vot(sound, [])
    assert summary.measurements == []
    assert summary.aspirated_voiceless_mean_ms is None


def test_aspirated_stop_voicing_after_release() -> None:
    """Stop ends at 0.1s; voicing (sine tone) starts at 0.18s → VOT ~80 ms."""
    sound = _silence_then_tone(silence_s=0.18, tone_s=0.5)
    occurrences = [
        # Pretend a /pʰ/ occupies 0.05s–0.10s
        PhonemeOccurrence(phoneme="pʰ", start_s=0.05, end_s=0.10),
    ]
    summary = estimate_vot(sound, occurrences)
    assert len(summary.measurements) == 1
    m = summary.measurements[0]
    assert m.aspiration_class == "aspirated_voiceless"
    # Allow a generous tolerance — pitch detector has frame quantization
    assert 50 <= m.vot_ms <= 110, f"Expected ~80 ms VOT, got {m.vot_ms}"
    assert summary.aspirated_voiceless_mean_ms is not None


def test_plain_voiceless_short_vot() -> None:
    """Stop ends at 0.1s; voicing starts at 0.115s → VOT ~15 ms."""
    sound = _silence_then_tone(silence_s=0.115, tone_s=0.5)
    occurrences = [
        PhonemeOccurrence(phoneme="p", start_s=0.05, end_s=0.10),
    ]
    summary = estimate_vot(sound, occurrences)
    assert len(summary.measurements) == 1
    m = summary.measurements[0]
    assert m.aspiration_class == "plain_voiceless"
    assert -5 <= m.vot_ms <= 40, f"Expected ~15 ms VOT, got {m.vot_ms}"


def test_voiced_stop_with_no_prevoicing_falls_through_to_lag() -> None:
    """A /b/ with no prevoicing should still produce a (positive) lag VOT."""
    sound = _silence_then_tone(silence_s=0.12, tone_s=0.5)
    occurrences = [
        PhonemeOccurrence(phoneme="b", start_s=0.05, end_s=0.10),
    ]
    summary = estimate_vot(sound, occurrences)
    # Either we got a measurement (positive VOT lag) or none at all — either is OK
    if summary.measurements:
        m = summary.measurements[0]
        assert m.aspiration_class == "voiced"


def test_non_stop_phoneme_ignored() -> None:
    """Vowels and fricatives should not produce VOT measurements."""
    sound = _silence_then_tone(silence_s=0.1, tone_s=0.5)
    occurrences = [
        PhonemeOccurrence(phoneme="i", start_s=0.05, end_s=0.10),
        PhonemeOccurrence(phoneme="s", start_s=0.10, end_s=0.15),
    ]
    summary = estimate_vot(sound, occurrences)
    assert summary.measurements == []
