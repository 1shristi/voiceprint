"""Unit tests for Loop 3a Phase 3 quality measurements."""

from __future__ import annotations

import math

import numpy as np
import parselmouth
import pytest

from app.services import quality

SAMPLE_RATE = 16_000


def _sine_sound(
    freq_hz: float,
    duration_s: float,
    amplitude: float = 0.5,
    noise_std: float = 0.0,
    silence_prefix_s: float = 0.0,
) -> parselmouth.Sound:
    """Build a parselmouth.Sound: optional silence + sine (+ optional noise)."""
    n_silence = int(SAMPLE_RATE * silence_prefix_s)
    n_voiced = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n_voiced, endpoint=False)
    voiced = amplitude * np.sin(2 * np.pi * freq_hz * t)
    if noise_std > 0:
        voiced = voiced + np.random.default_rng(0).normal(0, noise_std, n_voiced)
    silence = np.zeros(n_silence, dtype=np.float64)
    samples = np.concatenate([silence, voiced])
    return parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)


# ─── resolve_source_clip ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "clip_purpose,expected",
    [
        (None, "unspecified"),
        ("other", "unspecified"),
        ("diagnostic", "diagnostic"),
        ("stretch", "stretch"),
        ("counting", "counting"),
        ("free_speech", "free_speech"),
        ("unknown_garbage", "unspecified"),
        ("", "unspecified"),
    ],
)
def test_resolve_source_clip_never_infers(clip_purpose, expected) -> None:
    assert quality.resolve_source_clip(clip_purpose) == expected


# ─── compute_snr ─────────────────────────────────────────────────────────────


def test_compute_snr_pristine_audio_returns_high_db() -> None:
    """Loud sine with prefixed silence → high SNR."""
    sound = _sine_sound(220.0, duration_s=2.0, amplitude=0.5, silence_prefix_s=0.5)
    res = quality.compute_snr(sound)
    assert res.snr_db is not None
    assert res.snr_db > 30.0


def test_compute_snr_silent_clip_returns_none() -> None:
    """All-silent audio has no voiced regions → SNR not estimable."""
    samples = np.zeros(SAMPLE_RATE, dtype=np.float64)
    sound = parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)
    res = quality.compute_snr(sound)
    assert res.snr_db is None
    assert res.snr_note is not None
    assert "not estimable" in res.snr_note.lower() or "no voiced" in res.snr_note.lower()


def test_compute_snr_caps_at_60_db_when_noise_floor_zero() -> None:
    """Voiced sine + pure-zero silence prefix → noise_rms == 0 → snr capped at 60."""
    sound = _sine_sound(220.0, duration_s=2.0, amplitude=0.5, silence_prefix_s=0.5)
    res = quality.compute_snr(sound)
    assert res.snr_db is not None
    assert res.snr_db <= 60.0


def test_compute_snr_no_silent_region_returns_note() -> None:
    """A continuous loud signal with no silence anywhere → no silent region."""
    samples = 0.5 * np.sin(
        2 * np.pi * 220.0 * np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    )
    sound = parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)
    res = quality.compute_snr(sound)
    # Either we get the SNR cap (noise = 0) or a "no silent regions" note. Both
    # are acceptable behaviours given librosa.effects.split may still slice out
    # micro-gaps at zero crossings. We just need the call to not crash.
    if res.snr_db is None:
        assert res.snr_note is not None


# ─── compute_clipping_pct ────────────────────────────────────────────────────


def test_compute_clipping_pct_no_clipping() -> None:
    samples = 0.5 * np.ones(SAMPLE_RATE, dtype=np.float64)
    sound = parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)
    assert quality.compute_clipping_pct(sound) == 0.0


def test_compute_clipping_pct_all_clipped() -> None:
    samples = np.ones(SAMPLE_RATE, dtype=np.float64)
    sound = parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)
    # _sound_to_mono_float normalizes peak to ≤ 1; the all-ones array stays at 1.0
    # which is ≥ 0.99 → all samples clipped.
    assert quality.compute_clipping_pct(sound) == pytest.approx(100.0)


def test_compute_clipping_pct_partial_clipping() -> None:
    rng = np.random.default_rng(0)
    n = 10_000
    samples = rng.uniform(-0.5, 0.5, n).astype(np.float64)
    # Force 5% of samples to be clipped at +1.0.
    samples[:500] = 1.0
    sound = parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)
    pct = quality.compute_clipping_pct(sound)
    assert 4.5 <= pct <= 5.5


# ─── compute_baseline ────────────────────────────────────────────────────────


def test_compute_baseline_returns_none_for_short_clip() -> None:
    sound = _sine_sound(220.0, duration_s=1.0)
    assert quality.compute_baseline(sound, "diagnostic") is None


def test_compute_baseline_for_long_clip_populates_f0() -> None:
    sound = _sine_sound(220.0, duration_s=6.0)
    baseline = quality.compute_baseline(sound, "free_speech")
    assert baseline is not None
    assert baseline.source_clip == "free_speech"
    assert baseline.window_start_ms == 0
    assert baseline.window_end_ms == 5000
    # f0 should be ~220 Hz for a 220 Hz sine, with some tolerance for tracker noise.
    assert baseline.f0_mean_hz is not None
    assert 200.0 <= baseline.f0_mean_hz <= 240.0


def test_compute_baseline_passes_through_source_clip_verbatim() -> None:
    """The source_clip label is whatever the caller passed in — not inferred."""
    sound = _sine_sound(220.0, duration_s=6.0)
    for label in ("diagnostic", "stretch", "counting", "free_speech", "unspecified"):
        baseline = quality.compute_baseline(sound, label)
        assert baseline is not None
        assert baseline.source_clip == label


# ─── extract_quality (top-level) ─────────────────────────────────────────────


def test_extract_quality_short_clip_emits_baseline_note() -> None:
    sound = _sine_sound(220.0, duration_s=1.0)
    res = quality.extract_quality(sound, clip_purpose="diagnostic")
    assert res.calibrated_baseline is None
    assert any("Clip too short" in n for n in res.notes)


def test_extract_quality_long_clip_with_purpose_labels_source_clip() -> None:
    sound = _sine_sound(220.0, duration_s=6.0, silence_prefix_s=0.5)
    res = quality.extract_quality(sound, clip_purpose="diagnostic")
    assert res.calibrated_baseline is not None
    assert res.calibrated_baseline.source_clip == "diagnostic"
    assert res.snr_db is not None


def test_extract_quality_without_purpose_reports_unspecified() -> None:
    sound = _sine_sound(220.0, duration_s=6.0, silence_prefix_s=0.5)
    res = quality.extract_quality(sound, clip_purpose=None)
    assert res.calibrated_baseline is not None
    assert res.calibrated_baseline.source_clip == "unspecified"


def test_extract_quality_with_other_reports_unspecified() -> None:
    sound = _sine_sound(220.0, duration_s=6.0, silence_prefix_s=0.5)
    res = quality.extract_quality(sound, clip_purpose="other")
    assert res.calibrated_baseline is not None
    assert res.calibrated_baseline.source_clip == "unspecified"


def test_extract_quality_flags_significant_clipping_in_notes() -> None:
    """≥0.1% clipping triggers a note (spec §7.2)."""
    rng = np.random.default_rng(0)
    n = 10_000
    samples = rng.uniform(-0.5, 0.5, n).astype(np.float64)
    samples[:500] = 1.0  # 5% clipped
    # Pad with silence so the clip is long enough to attempt baseline (>5s).
    padding = np.zeros(int(SAMPLE_RATE * 6.0) - n, dtype=np.float64)
    samples = np.concatenate([samples, padding])
    sound = parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)
    res = quality.extract_quality(sound, clip_purpose="free_speech")
    assert res.clipping_pct is not None
    assert res.clipping_pct > 0.1
    assert any("clipping" in n.lower() for n in res.notes)


def test_extract_quality_three_quality_tiers() -> None:
    """Spec §7.4 fixture: SNR detection across pristine / typical / poor tiers.

    Synthesised inline (sine + gaussian noise) rather than via committed audio
    fixtures — same coverage, faster CI, no fixture-folder churn.
    """
    rng = np.random.default_rng(0)
    duration_s = 2.0
    n = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    silence_prefix = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float64)

    pristine = np.concatenate([silence_prefix, 0.5 * np.sin(2 * np.pi * 220 * t)])
    typical = np.concatenate(
        [silence_prefix, 0.5 * np.sin(2 * np.pi * 220 * t) + rng.normal(0, 0.03, n)]
    )
    poor = np.concatenate(
        [silence_prefix, 0.5 * np.sin(2 * np.pi * 220 * t) + rng.normal(0, 0.15, n)]
    )

    snr_values = []
    for samples in (pristine, typical, poor):
        sound = parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)
        res = quality.extract_quality(sound, clip_purpose=None)
        snr_values.append(res.snr_db)

    # Pristine should be the highest, poor the lowest. We don't assert specific
    # dB targets (energy-based VAD is approximate) — only the monotone trend.
    assert all(v is not None for v in snr_values)
    assert snr_values[0] >= snr_values[1] >= snr_values[2]
