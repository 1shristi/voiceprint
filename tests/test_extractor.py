"""Tests for phonetic feature extraction.

Synthetic signals are used to verify each extractor against known ground truth.
"""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import parselmouth
import pytest

from app.services import extractor


SAMPLE_RATE = 16_000


# ─── helpers ────────────────────────────────────────────────────────────────


def _make_sine(freq_hz: float, duration_s: float = 1.0, amplitude: float = 0.5) -> parselmouth.Sound:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    samples = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)


def _make_pulsed_signal(freq_hz: float, num_pulses: int, gap_s: float = 0.05) -> parselmouth.Sound:
    """Pulses of sine separated by silence — used to test syllable rate detection."""
    pulse_dur = 0.15
    silence = np.zeros(int(SAMPLE_RATE * gap_s))
    pulses = []
    for _ in range(num_pulses):
        t = np.linspace(0, pulse_dur, int(SAMPLE_RATE * pulse_dur), endpoint=False)
        envelope = np.hanning(len(t))
        pulses.append(0.5 * envelope * np.sin(2 * np.pi * freq_hz * t))
        pulses.append(silence)
    samples = np.concatenate(pulses[:-1])
    return parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)


def _sound_to_wav_b64(sound: parselmouth.Sound) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    try:
        sound.save(str(path), "WAV")
        return base64.b64encode(path.read_bytes()).decode("ascii")
    finally:
        path.unlink(missing_ok=True)


# ─── decode_audio ────────────────────────────────────────────────────────────


def test_decode_wav_roundtrip() -> None:
    sound = _make_sine(220.0, duration_s=0.5)
    b64 = _sound_to_wav_b64(sound)
    decoded = extractor.decode_audio(b64, "wav")
    assert decoded.get_total_duration() == pytest.approx(0.5, abs=0.01)


def test_decode_invalid_base64_raises() -> None:
    with pytest.raises(extractor.AudioDecodeError):
        extractor.decode_audio("!!!not base64!!!", "wav")


def test_decode_empty_payload_raises() -> None:
    with pytest.raises(extractor.AudioDecodeError):
        extractor.decode_audio("", "wav")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decode_webm_via_ffmpeg() -> None:
    """End-to-end: synthesize wav → encode to webm via ffmpeg → decode through our pipeline."""
    sound = _make_sine(220.0, duration_s=0.5)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "in.wav"
        webm_path = Path(tmp) / "in.webm"
        sound.save(str(wav_path), "WAV")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
             "-c:a", "libopus", "-b:a", "32k", str(webm_path)],
            check=True,
        )
        b64 = base64.b64encode(webm_path.read_bytes()).decode("ascii")
        decoded = extractor.decode_audio(b64, "webm")
        assert decoded.get_total_duration() == pytest.approx(0.5, abs=0.05)


# ─── F0 extraction ──────────────────────────────────────────────────────────


def test_f0_pure_tone_recovers_frequency() -> None:
    """A 200 Hz sine should yield mean F0 ~200 Hz."""
    sound = _make_sine(200.0, duration_s=1.0)
    result = extractor.extract_f0(sound)
    assert result.mean_hz is not None
    assert result.mean_hz == pytest.approx(200.0, abs=5.0)
    assert result.voiced_fraction > 0.5


def test_f0_silent_clip_returns_none() -> None:
    silence = parselmouth.Sound(np.zeros(SAMPLE_RATE), sampling_frequency=SAMPLE_RATE)
    result = extractor.extract_f0(silence)
    assert result.mean_hz is None
    assert result.voiced_fraction < 0.05


def test_f0_higher_pitch_recovers_higher_frequency() -> None:
    sound = _make_sine(320.0, duration_s=1.0)
    result = extractor.extract_f0(sound)
    assert result.mean_hz is not None
    assert result.mean_hz == pytest.approx(320.0, abs=10.0)


# ─── Formants ────────────────────────────────────────────────────────────────


def test_formants_run_without_error_on_real_signal() -> None:
    """Formants on a pure sine are degenerate; just check the call shape works."""
    sound = _make_sine(150.0, duration_s=1.0)
    result = extractor.extract_formants(sound)
    # On a pure tone, formants are unreliable; we just verify the call returns the dataclass.
    assert isinstance(result, extractor.FormantResult)


# ─── Syllable rate ───────────────────────────────────────────────────────────


def test_syllable_rate_counts_pulses_approximately() -> None:
    """Five distinct pulses in ~1 second should yield ~5 Hz syllable rate."""
    sound = _make_pulsed_signal(220.0, num_pulses=5, gap_s=0.05)
    rate = extractor.estimate_syllable_rate(sound)
    assert rate is not None
    # Allow generous tolerance — peak detection isn't exact on synthetic Hanning pulses
    assert 3.0 <= rate <= 8.0, f"Expected ~5 Hz, got {rate}"


def test_syllable_rate_silence_returns_low() -> None:
    silence = parselmouth.Sound(np.zeros(SAMPLE_RATE), sampling_frequency=SAMPLE_RATE)
    rate = extractor.estimate_syllable_rate(silence)
    # Silence may legitimately return None or 0.0 depending on intensity contour
    assert rate is None or rate == 0.0


# ─── extract_all (orchestrator) ─────────────────────────────────────────────


def test_extract_all_on_synthetic_wav() -> None:
    sound = _make_sine(180.0, duration_s=1.0)
    b64 = _sound_to_wav_b64(sound)
    features = extractor.extract_all(b64, "wav")
    assert features.duration_s == pytest.approx(1.0, abs=0.01)
    assert features.f0.mean_hz is not None
    assert features.f0.mean_hz == pytest.approx(180.0, abs=10.0)
    assert features.vot_ms is None  # phase 2c
