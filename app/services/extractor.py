"""Phonetic feature extraction using Praat (via parselmouth)."""

from __future__ import annotations

import base64
import binascii
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import parselmouth

if TYPE_CHECKING:
    from app.services.alignment import TranscriptInput


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
    # Semitone metrics: anatomy-invariant (log scale), useful for cross-speaker
    # classification. Reference is arbitrary for std/range — both are translation
    # invariant under log shift.
    std_semitones: float | None
    range_semitones_p10_p90: float | None
    voiced_fraction: float


@dataclass
class FormantResult:
    f1_mean_hz: float | None
    f2_mean_hz: float | None
    f3_mean_hz: float | None


@dataclass
class VotMeasurementOut:
    phoneme: str
    time_s: float
    vot_ms: float
    aspiration_class: str


@dataclass
class StretchProbeOut:
    label: str
    ipa: str
    expected_count: int
    count: int
    approximate_count: int
    status: str


@dataclass
class StretchScoreOut:
    expected_language: str
    probes: list[StretchProbeOut]


@dataclass
class LanguageMatchOut:
    claimed_language: str
    verdict: str
    score: float | None
    positive_hits: int
    positive_total: int
    negative_clean: int
    negative_total: int
    notes: list[str]


@dataclass
class AlignedTop3Out:
    phoneme: str
    prob: float


@dataclass
class AlignedPositionOut:
    target_phoneme: str
    target_index_in_transcript: int
    start_ms: int
    end_ms: int
    avg_log_prob: float
    top1_predicted: str
    top3_alternatives: list[AlignedTop3Out]
    match_classification: str


@dataclass
class AlignedSummaryOut:
    expected_count: int
    produced_count: int
    near_miss_count: int
    absent_count: int
    evidence_strength: str


@dataclass
class AlignedPhonemesOut:
    transcript_format: str
    alignment_quality: str
    positions: list[AlignedPositionOut]
    summary_by_phoneme: dict[str, AlignedSummaryOut]
    alignment_warnings: list[str]


@dataclass
class ConfusionContributionOut:
    from_symbol: str
    raw_count: int
    weight: float
    contribution: float


@dataclass
class ConfusionEvidenceOut:
    raw_count: int
    smoothed_count: float
    evidence_from: list[ConfusionContributionOut]
    interpretation: str


@dataclass
class FeatureSet:
    duration_s: float
    f0: F0Result
    formants: FormantResult
    syllable_rate_hz: float | None
    vot_aspirated_voiceless_mean_ms: float | None
    vot_plain_voiceless_mean_ms: float | None
    vot_voiced_mean_ms: float | None
    vot_measurements: list[VotMeasurementOut]
    phoneme_counts: dict[str, int]
    phoneme_total_tokens: int
    # Loop 3a Phase 2 — confusion-smoothed counts and per-target audit trail.
    # Computed on every /analyze call where the phoneme model is enabled. Empty
    # when the phoneme model is disabled (tests) or the clip emitted no symbols.
    phoneme_smoothed_counts: dict[str, float]
    phoneme_confusion_evidence: dict[str, ConfusionEvidenceOut]
    stretch_score: StretchScoreOut | None
    language_match: LanguageMatchOut | None
    aligned_phonemes: AlignedPhonemesOut | None
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
            std_semitones=None,
            range_semitones_p10_p90=None,
            voiced_fraction=voiced_fraction,
        )

    voiced_st = 12.0 * np.log2(voiced / 100.0)
    p10, p90 = np.percentile(voiced_st, [10, 90])

    return F0Result(
        mean_hz=_finite_or_none(float(np.mean(voiced))),
        min_hz=_finite_or_none(float(np.min(voiced))),
        max_hz=_finite_or_none(float(np.max(voiced))),
        std_hz=_finite_or_none(float(np.std(voiced))),
        std_semitones=_finite_or_none(float(np.std(voiced_st))),
        range_semitones_p10_p90=_finite_or_none(float(p90 - p10)),
        voiced_fraction=voiced_fraction,
    )


def _finite_or_none(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


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


# ─── Top-level extraction ────────────────────────────────────────────────────


def extract_all(
    audio_b64: str,
    fmt: str,
    *,
    include_phonemes: bool = True,
    expected_language: str | None = None,
    claimed_language: str | None = None,
    transcript: "TranscriptInput | None" = None,
) -> FeatureSet:
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

    phoneme_counts: dict[str, int] = {}
    phoneme_total = 0
    phoneme_occurrences: list = []
    phoneme_smoothed_counts: dict[str, float] = {}
    phoneme_confusion_evidence: dict[str, ConfusionEvidenceOut] = {}
    stretch_out: StretchScoreOut | None = None
    language_match_out: LanguageMatchOut | None = None
    aligned_phonemes_out: AlignedPhonemesOut | None = None
    if include_phonemes:
        try:
            from app.services import phonemes

            inv = phonemes.extract_phonemes(
                sound,
                expected_language=expected_language,
                include_alignment_artifacts=transcript is not None,
            )
            phoneme_counts = inv.counts
            phoneme_total = inv.total_tokens
            phoneme_occurrences = inv.occurrences

            # Loop 3a Phase 2 — compute confusion-smoothed counts on every call
            # where free decoding produced something. Failures are non-fatal:
            # raw counts already populated above, so a smoothing error just
            # leaves smoothed_counts/confusion_evidence empty and a note.
            if phoneme_counts:
                try:
                    from app.services import smoothing

                    smoothing_result = smoothing.compute_smoothed_counts(phoneme_counts)
                    phoneme_smoothed_counts = smoothing_result.smoothed_counts
                    phoneme_confusion_evidence = {
                        target: ConfusionEvidenceOut(
                            raw_count=ev.raw_count,
                            smoothed_count=ev.smoothed_count,
                            evidence_from=[
                                ConfusionContributionOut(
                                    from_symbol=c.from_symbol,
                                    raw_count=c.raw_count,
                                    weight=c.weight,
                                    contribution=c.contribution,
                                )
                                for c in ev.evidence_from
                            ],
                            interpretation=ev.interpretation,
                        )
                        for target, ev in smoothing_result.confusion_evidence.items()
                    }
                except Exception as e:  # noqa: BLE001 — smoothing failures are non-fatal
                    notes.append(f"confusion smoothing unavailable: {type(e).__name__}")

            if inv.stretch_score is not None:
                stretch_out = StretchScoreOut(
                    expected_language=inv.stretch_score.expected_language,
                    probes=[
                        StretchProbeOut(
                            label=p.label,
                            ipa=p.ipa,
                            expected_count=p.expected_count,
                            count=p.count,
                            approximate_count=p.approximate_count,
                            status=p.status,
                        )
                        for p in inv.stretch_score.probes
                    ],
                )
            elif expected_language:
                notes.append(f"no stretch probe set for language={expected_language!r}")

            if claimed_language:
                try:
                    from app.services import language_check

                    match = language_check.score_language_match(inv, claimed_language)
                    if match is not None:
                        language_match_out = LanguageMatchOut(
                            claimed_language=match.claimed_language,
                            verdict=match.verdict,
                            score=match.score,
                            positive_hits=match.positive_hits,
                            positive_total=match.positive_total,
                            negative_clean=match.negative_clean,
                            negative_total=match.negative_total,
                            notes=list(match.notes),
                        )
                except Exception as e:  # noqa: BLE001 — language match failures are non-fatal
                    notes.append(f"language match unavailable: {type(e).__name__}")

            if transcript is not None and inv.alignment_artifacts is not None:
                try:
                    from app.services import alignment as alignment_service

                    result = alignment_service.align_against_transcript(
                        inv.alignment_artifacts.log_probs,
                        blank_id=inv.alignment_artifacts.blank_id,
                        vocab=inv.alignment_artifacts.vocab,
                        audio_duration_s=inv.alignment_artifacts.audio_duration_s,
                        transcript=transcript,
                    )
                    aligned_phonemes_out = AlignedPhonemesOut(
                        transcript_format=result.transcript_format,
                        alignment_quality=result.alignment_quality,
                        positions=[
                            AlignedPositionOut(
                                target_phoneme=p.target_phoneme,
                                target_index_in_transcript=p.target_index_in_transcript,
                                start_ms=p.start_ms,
                                end_ms=p.end_ms,
                                avg_log_prob=p.avg_log_prob,
                                top1_predicted=p.top1_predicted,
                                top3_alternatives=[
                                    AlignedTop3Out(phoneme=a.phoneme, prob=a.prob)
                                    for a in p.top3_alternatives
                                ],
                                match_classification=p.match_classification,
                            )
                            for p in result.positions
                        ],
                        summary_by_phoneme={
                            k: AlignedSummaryOut(
                                expected_count=v.expected_count,
                                produced_count=v.produced_count,
                                near_miss_count=v.near_miss_count,
                                absent_count=v.absent_count,
                                evidence_strength=v.evidence_strength,
                            )
                            for k, v in result.summary_by_phoneme.items()
                        },
                        alignment_warnings=list(result.alignment_warnings),
                    )
                except Exception as e:  # noqa: BLE001 — alignment failures are non-fatal
                    notes.append(f"forced alignment unavailable: {type(e).__name__}")
            elif transcript is not None and inv.alignment_artifacts is None:
                # Phoneme model disabled (e.g. VOICEPRINT_DISABLE_PHONEMES=1) but
                # a transcript was supplied. Surface a note so the caller knows
                # alignment didn't run; don't fail the request.
                notes.append("transcript supplied but phoneme model is disabled; alignment skipped")
        except Exception as e:  # noqa: BLE001 — phoneme failures are non-fatal
            notes.append(f"phoneme detection unavailable: {type(e).__name__}")

    # VOT depends on phoneme timing — only meaningful when occurrences are available.
    vot_aspirated_mean: float | None = None
    vot_plain_mean: float | None = None
    vot_voiced_mean: float | None = None
    vot_measurements_out: list[VotMeasurementOut] = []
    if phoneme_occurrences:
        try:
            from app.services import vot as vot_service

            summary = vot_service.estimate_vot(sound, phoneme_occurrences)
            vot_aspirated_mean = summary.aspirated_voiceless_mean_ms
            vot_plain_mean = summary.plain_voiceless_mean_ms
            vot_voiced_mean = summary.voiced_mean_ms
            vot_measurements_out = [
                VotMeasurementOut(
                    phoneme=m.phoneme,
                    time_s=m.time_s,
                    vot_ms=m.vot_ms,
                    aspiration_class=m.aspiration_class,
                )
                for m in summary.measurements
            ]
        except Exception as e:  # noqa: BLE001 — VOT failures are non-fatal
            notes.append(f"VOT estimation unavailable: {type(e).__name__}")

    return FeatureSet(
        duration_s=float(duration),
        f0=f0,
        formants=formants,
        syllable_rate_hz=syllable_rate,
        vot_aspirated_voiceless_mean_ms=vot_aspirated_mean,
        vot_plain_voiceless_mean_ms=vot_plain_mean,
        vot_voiced_mean_ms=vot_voiced_mean,
        vot_measurements=vot_measurements_out,
        phoneme_counts=phoneme_counts,
        phoneme_total_tokens=phoneme_total,
        phoneme_smoothed_counts=phoneme_smoothed_counts,
        phoneme_confusion_evidence=phoneme_confusion_evidence,
        stretch_score=stretch_out,
        language_match=language_match_out,
        aligned_phonemes=aligned_phonemes_out,
        notes=notes,
    )
