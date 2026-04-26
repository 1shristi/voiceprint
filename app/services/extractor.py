"""Phonetic feature extraction using Praat (via parselmouth)."""

from __future__ import annotations

import base64
import binascii
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parselmouth


class AudioDecodeError(Exception):
    """Raised when the input audio can't be decoded."""


class FeatureExtractionError(Exception):
    """Raised when Praat extraction fails on a clip."""


@dataclass
class F0Result:
    mean_hz: float | None
    min_hz: float | None
    max_hz: float | None
    std_hz: float | None
    voiced_fraction: float


@dataclass
class FormantResult:
    f1_mean_hz: float | None
    f2_mean_hz: float | None
    f3_mean_hz: float | None


@dataclass
class FeatureSet:
    duration_s: float
    f0: F0Result
    formants: FormantResult
    syllable_rate_hz: float | None
    vot_ms: float | None
    phoneme_counts: dict[str, int]
    phoneme_total_tokens: int
    notes: list[str]


# ─── Audio decoding ──────────────────────────────────────────────────────────


def _ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise AudioDecodeError(
            "ffmpeg not found on PATH. Install via `brew install ffmpeg` (macOS) "
            "or `apt-get install ffmpeg` (Linux)."
        )
    return path


def decode_audio(audio_b64: str, fmt: str) -> parselmouth.Sound:
    """Decode base64 audio (webm/wav/etc.) into a parselmouth Sound."""
    try:
        audio_bytes = base64.b64decode(audio_b64, validate=False)
    except (binascii.Error, ValueError) as e:
        raise AudioDecodeError(f"Invalid base64 audio: {e}") from e

    if not audio_bytes:
        raise AudioDecodeError("Empty audio payload.")

    suffix = f".{fmt.lower().lstrip('.')}" if fmt else ".bin"
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / f"input{suffix}"
        in_path.write_bytes(audio_bytes)

        # If already wav, parselmouth can read directly
        if suffix == ".wav":
            try:
                return parselmouth.Sound(str(in_path))
            except Exception as e:
                raise AudioDecodeError(f"Failed to read WAV: {e}") from e

        # Otherwise transcode to 16 kHz mono PCM wav via ffmpeg
        out_path = Path(tmpdir) / "decoded.wav"
        try:
            subprocess.run(
                [
                    _ffmpeg_path(),
                    "-y",
                    "-loglevel", "error",
                    "-i", str(in_path),
                    "-ar", "16000",
                    "-ac", "1",
                    "-f", "wav",
                    str(out_path),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            raise AudioDecodeError(f"ffmpeg failed to decode audio: {stderr.strip()[:300]}") from e
        except subprocess.TimeoutExpired as e:
            raise AudioDecodeError("ffmpeg timed out decoding audio.") from e

        try:
            return parselmouth.Sound(str(out_path))
        except Exception as e:
            raise AudioDecodeError(f"parselmouth failed to load decoded WAV: {e}") from e


# ─── F0 (pitch) ──────────────────────────────────────────────────────────────


def extract_f0(sound: parselmouth.Sound, *, pitch_floor: float = 75.0, pitch_ceiling: float = 600.0) -> F0Result:
    """Extract F0 statistics. Defaults match Praat's recommended ranges for adult speech."""
    try:
        pitch = sound.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    except Exception as e:
        raise FeatureExtractionError(f"Pitch extraction failed: {e}") from e

    values = pitch.selected_array["frequency"]
    voiced = values[values > 0]
    voiced_fraction = float(len(voiced) / len(values)) if len(values) > 0 else 0.0

    if len(voiced) < 5:
        return F0Result(
            mean_hz=None,
            min_hz=None,
            max_hz=None,
            std_hz=None,
            voiced_fraction=voiced_fraction,
        )

    return F0Result(
        mean_hz=float(np.mean(voiced)),
        min_hz=float(np.min(voiced)),
        max_hz=float(np.max(voiced)),
        std_hz=float(np.std(voiced)),
        voiced_fraction=voiced_fraction,
    )


# ─── Formants ────────────────────────────────────────────────────────────────


def extract_formants(sound: parselmouth.Sound) -> FormantResult:
    """Mean F1/F2/F3 across the clip (during voiced regions)."""
    try:
        formant = sound.to_formant_burg()
    except Exception as e:
        raise FeatureExtractionError(f"Formant extraction failed: {e}") from e

    duration = sound.get_total_duration()
    times = np.arange(0.0, duration, 0.01)

    f1, f2, f3 = [], [], []
    for t in times:
        v1 = formant.get_value_at_time(1, t)
        v2 = formant.get_value_at_time(2, t)
        v3 = formant.get_value_at_time(3, t)
        # Praat returns NaN for unvoiced or unanalyzable points
        if not np.isnan(v1):
            f1.append(v1)
        if not np.isnan(v2):
            f2.append(v2)
        if not np.isnan(v3):
            f3.append(v3)

    return FormantResult(
        f1_mean_hz=float(np.mean(f1)) if f1 else None,
        f2_mean_hz=float(np.mean(f2)) if f2 else None,
        f3_mean_hz=float(np.mean(f3)) if f3 else None,
    )


# ─── Syllable rate ───────────────────────────────────────────────────────────


def estimate_syllable_rate(sound: parselmouth.Sound, *, min_dip_db: float = 2.0) -> float | None:
    """
    Estimate syllable rate from intensity peaks.

    Method (de Jong & Wempe 2009 simplified): find peaks in intensity contour,
    require a minimum drop between adjacent peaks to count as separate syllables.
    """
    try:
        intensity = sound.to_intensity(time_step=0.01)
    except Exception as e:
        raise FeatureExtractionError(f"Intensity extraction failed: {e}") from e

    values = intensity.values[0]  # shape (T,)
    if len(values) < 10:
        return None

    duration = sound.get_total_duration()
    if duration <= 0:
        return None

    # Find local maxima above (mean - 5 dB) threshold
    threshold = float(np.mean(values)) - 5.0
    peaks: list[int] = []
    for i in range(1, len(values) - 1):
        if values[i] > threshold and values[i] > values[i - 1] and values[i] > values[i + 1]:
            peaks.append(i)

    if not peaks:
        return 0.0

    # Filter peaks: require >= min_dip_db drop from previous peak's neighborhood
    accepted = [peaks[0]]
    for p in peaks[1:]:
        prev = accepted[-1]
        between = values[prev:p]
        if len(between) == 0:
            continue
        dip = float(values[prev] - np.min(between))
        if dip >= min_dip_db:
            accepted.append(p)

    return float(len(accepted) / duration)


# ─── VOT — placeholder ───────────────────────────────────────────────────────


def estimate_vot(_sound: parselmouth.Sound) -> float | None:
    """
    Voice Onset Time — true VOT requires per-stop segmentation, which needs
    forced alignment or a phoneme model. Returning None until phase 2c.
    """
    return None


# ─── Top-level extraction ────────────────────────────────────────────────────


def extract_all(audio_b64: str, fmt: str, *, include_phonemes: bool = True) -> FeatureSet:
    sound = decode_audio(audio_b64, fmt)
    notes: list[str] = []

    duration = sound.get_total_duration()
    if duration < 0.3:
        notes.append("clip is very short (<0.3s); results may be unreliable")

    f0 = extract_f0(sound)
    if f0.voiced_fraction < 0.05:
        notes.append("very little voiced audio detected; pitch may be unreliable")

    formants = extract_formants(sound)
    syllable_rate = estimate_syllable_rate(sound)
    vot = estimate_vot(sound)

    phoneme_counts: dict[str, int] = {}
    phoneme_total = 0
    if include_phonemes:
        try:
            from app.services import phonemes

            inv = phonemes.extract_phonemes(sound)
            phoneme_counts = inv.counts
            phoneme_total = inv.total_tokens
        except Exception as e:  # noqa: BLE001 — phoneme failures are non-fatal
            notes.append(f"phoneme detection unavailable: {type(e).__name__}")

    return FeatureSet(
        duration_s=float(duration),
        f0=f0,
        formants=formants,
        syllable_rate_hz=syllable_rate,
        vot_ms=vot,
        phoneme_counts=phoneme_counts,
        phoneme_total_tokens=phoneme_total,
        notes=notes,
    )
