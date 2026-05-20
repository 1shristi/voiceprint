"""Clip-quality measurements: SNR, clipping percentage, within-session baseline.

Loop 3a Phase 3. All measurements are additive — they augment the /analyze
response with a `quality` block but never modify existing fields.

Design notes:

- SNR uses librosa's energy-based VAD (`librosa.effects.split`) for noise-floor
  estimation. Spec §7.1 documents the trade-off: librosa is approximate but
  sufficient; lighter alternatives (silero-vad, webrtcvad) are deferred follow-ups.
- `source_clip` in the calibrated baseline echoes the caller-supplied
  `clip_purpose` verbatim or reports `"unspecified"`. We deliberately do NOT
  infer clip type from duration / energy / content — per spec §7.3, that's the
  exact failure mode this field exists to eliminate.
- Calibration ALWAYS runs (whenever clip ≥ 5s); `clip_purpose` only governs
  the `source_clip` label, not whether the measurements are computed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import parselmouth


_BASELINE_WINDOW_S = 5.0
_CLIPPING_THRESHOLD = 0.99  # |sample| ≥ this counts as clipped
_CLIPPING_NOTE_PCT = 0.1    # ≥0.1% triggers a notes-block warning
_SNR_VAD_TOP_DB = 30.0      # librosa.effects.split threshold; quieter than this is "silence"
_SNR_MAX_DB = 60.0          # cap when noise floor is at machine zero

# Valid clip_purpose values per spec §7.3. Anything outside this set (including
# None and "other") resolves to "unspecified".
_VALID_CLIP_PURPOSES = frozenset({"diagnostic", "stretch", "counting", "free_speech"})


@dataclass
class SnrResult:
    snr_db: float | None
    snr_note: str | None = None


@dataclass
class CalibratedBaselineResult:
    source_clip: str
    window_start_ms: int
    window_end_ms: int
    f0_mean_hz: float | None
    f0_std_semitones: float | None
    syllable_rate_hz: float | None
    notes: str


@dataclass
class QualityResult:
    snr_db: float | None
    clipping_pct: float | None
    calibrated_baseline: CalibratedBaselineResult | None
    snr_note: str | None = None
    notes: list[str] = field(default_factory=list)


def resolve_source_clip(clip_purpose: str | None) -> str:
    """Map caller-supplied clip_purpose to the response's `source_clip` value.

    Never infers from anything other than the caller's explicit label.
    """
    if clip_purpose is None:
        return "unspecified"
    if clip_purpose in _VALID_CLIP_PURPOSES:
        return clip_purpose
    return "unspecified"


def _sound_to_mono_float(sound: parselmouth.Sound) -> np.ndarray:
    samples = np.asarray(sound.values[0], dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak
    return samples


def compute_snr(sound: parselmouth.Sound) -> SnrResult:
    """Energy-based SNR estimate.

    Detect non-silent intervals via librosa's VAD; the complement is "silence"
    used as the noise floor. Returns None with a `snr_note` when no silent
    region is detectable (e.g. a clip that's voiced edge-to-edge).
    """
    samples = _sound_to_mono_float(sound)
    if samples.size == 0:
        return SnrResult(snr_db=None, snr_note="Empty audio; SNR not estimable.")

    # Import inside the function so cold imports of the module stay cheap and
    # so the librosa dependency doesn't gate test collection.
    import librosa

    sr = int(sound.sampling_frequency)
    intervals = librosa.effects.split(samples, top_db=_SNR_VAD_TOP_DB)

    if intervals.size == 0:
        # Whole clip below the energy threshold — treat as silence.
        return SnrResult(snr_db=None, snr_note="No voiced regions detected; SNR not estimable.")

    voiced_mask = np.zeros(samples.shape[0], dtype=bool)
    for start, end in intervals:
        voiced_mask[start:end] = True

    voiced = samples[voiced_mask]
    silent = samples[~voiced_mask]

    if silent.size == 0:
        return SnrResult(snr_db=None, snr_note="No silent regions detected; SNR not estimable.")
    if voiced.size == 0:
        return SnrResult(snr_db=None, snr_note="No voiced regions detected; SNR not estimable.")

    signal_rms = float(np.sqrt(np.mean(np.square(voiced))))
    noise_rms = float(np.sqrt(np.mean(np.square(silent))))

    if signal_rms == 0.0:
        return SnrResult(snr_db=None, snr_note="Signal energy is zero; SNR not estimable.")

    if noise_rms == 0.0:
        return SnrResult(snr_db=_SNR_MAX_DB)

    snr_db = 20.0 * math.log10(signal_rms / noise_rms)
    if not math.isfinite(snr_db):
        return SnrResult(snr_db=None, snr_note="SNR computation produced non-finite value.")
    return SnrResult(snr_db=min(snr_db, _SNR_MAX_DB))


def compute_clipping_pct(sound: parselmouth.Sound) -> float:
    """Percent of samples at amplitude ≥ 0.99 or ≤ -0.99.

    Assumes float audio normalized to [-1, 1] (which `_sound_to_mono_float`
    enforces).
    """
    samples = _sound_to_mono_float(sound)
    if samples.size == 0:
        return 0.0
    clipped = int(np.sum(np.abs(samples) >= _CLIPPING_THRESHOLD))
    return float(clipped) / float(samples.size) * 100.0


def compute_baseline(
    sound: parselmouth.Sound,
    source_clip: str,
) -> CalibratedBaselineResult | None:
    """Within-session baseline from the first ~5 seconds of the clip.

    Returns None if the clip is shorter than 5 seconds — caller surfaces that
    as a note. `source_clip` is the already-resolved label (caller decides;
    we don't second-guess it).
    """
    duration = sound.get_total_duration()
    if duration < _BASELINE_WINDOW_S:
        return None

    # Lazy import so the module remains usable in tests that monkeypatch the
    # extractor module before this file imports it.
    from app.services import extractor

    try:
        window = sound.extract_part(
            from_time=0.0,
            to_time=_BASELINE_WINDOW_S,
            preserve_times=False,
        )
    except Exception:
        # If parselmouth refuses the window for any reason, decline to
        # calibrate rather than crash.
        return None

    try:
        f0 = extractor.extract_f0(window)
    except extractor.FeatureExtractionError:
        f0 = None

    try:
        syllable_rate = extractor.estimate_syllable_rate(window)
    except extractor.FeatureExtractionError:
        syllable_rate = None

    return CalibratedBaselineResult(
        source_clip=source_clip,
        window_start_ms=0,
        window_end_ms=int(_BASELINE_WINDOW_S * 1000),
        f0_mean_hz=f0.mean_hz if f0 is not None else None,
        f0_std_semitones=f0.std_semitones if f0 is not None else None,
        syllable_rate_hz=syllable_rate,
        notes="First 5s of clip used as baseline; values for normalisation only.",
    )


def extract_quality(
    sound: parselmouth.Sound,
    clip_purpose: str | None,
) -> QualityResult:
    """Top-level entry point: compute SNR, clipping, and baseline.

    Every measurement is independently best-effort. A failure in any one of
    them leaves the corresponding field null and the rest intact — never
    crashes the response.
    """
    notes: list[str] = []

    try:
        snr = compute_snr(sound)
        snr_db = snr.snr_db
        snr_note = snr.snr_note
    except Exception as e:  # noqa: BLE001 — quality measurement failures are non-fatal
        snr_db = None
        snr_note = f"SNR estimation unavailable: {type(e).__name__}"

    try:
        clipping_pct = compute_clipping_pct(sound)
    except Exception as e:  # noqa: BLE001
        clipping_pct = None
        notes.append(f"clipping detection unavailable: {type(e).__name__}")

    if clipping_pct is not None and clipping_pct >= _CLIPPING_NOTE_PCT:
        notes.append(
            f"Audio significantly affected by clipping ({clipping_pct:.2f}%)."
        )

    source_clip = resolve_source_clip(clip_purpose)

    duration = sound.get_total_duration()
    if duration < _BASELINE_WINDOW_S:
        baseline = None
        notes.append("Clip too short (<5s) for baseline calibration.")
    else:
        try:
            baseline = compute_baseline(sound, source_clip)
        except Exception as e:  # noqa: BLE001
            baseline = None
            notes.append(f"baseline calibration unavailable: {type(e).__name__}")

    return QualityResult(
        snr_db=snr_db,
        clipping_pct=clipping_pct,
        calibrated_baseline=baseline,
        snr_note=snr_note,
        notes=notes,
    )
