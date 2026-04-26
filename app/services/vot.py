"""Voice Onset Time (VOT) estimation.

For each stop consonant detected by Wav2Vec2-Phoneme, measure VOT using
Praat's pitch contour: VOT = (voicing_onset_time) - (stop_release_time).

Positive VOT = release precedes voicing (English /pʰ/, /tʰ/, /kʰ/).
Short positive VOT = unaspirated voiceless stops (Spanish /p/, French /p/).
Negative VOT = voicing leads release ("prevoicing", e.g. Spanish /b/, /d/, /ɡ/).

This is a best-effort estimate. Real phonetician-grade VOT requires manual
boundary verification on a spectrogram. Treat results as indicative, not exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import numpy as np
import parselmouth

from app.services.phonemes import PhonemeOccurrence


# eSpeak / IPA stop phonemes by aspiration class.
# Aspirated voiceless: long positive VOT (~50–100 ms).
ASPIRATED_VOICELESS = {"pʰ", "tʰ", "kʰ", "p_h", "t_h", "k_h"}
# Plain voiceless: short positive VOT (~10–30 ms).
PLAIN_VOICELESS = {"p", "t", "k"}
# Voiced stops: VOT can be positive (short) or negative (prevoiced).
VOICED_STOPS = {"b", "d", "ɡ", "g"}
# Affricates/ejectives we don't currently classify but may want to flag
OTHER_STOPS = {"t͡ʃ", "d͡ʒ", "ʈ", "ɖ", "q", "ɢ"}

ALL_STOPS = ASPIRATED_VOICELESS | PLAIN_VOICELESS | VOICED_STOPS | OTHER_STOPS


@dataclass
class VotMeasurement:
    phoneme: str
    time_s: float          # release time (== end of the stop occurrence)
    vot_ms: float          # voicing_onset - release; can be negative
    aspiration_class: str  # 'aspirated_voiceless' | 'plain_voiceless' | 'voiced' | 'other'


@dataclass
class VotSummary:
    measurements: list[VotMeasurement]
    aspirated_voiceless_mean_ms: float | None
    plain_voiceless_mean_ms: float | None
    voiced_mean_ms: float | None


def _classify(phoneme: str) -> str:
    if phoneme in ASPIRATED_VOICELESS:
        return "aspirated_voiceless"
    if phoneme in PLAIN_VOICELESS:
        return "plain_voiceless"
    if phoneme in VOICED_STOPS:
        return "voiced"
    return "other"


def _voicing_onset_after(
    pitch: parselmouth.Data, after_time_s: float, *, max_search_s: float = 0.25
) -> float | None:
    """First voiced frame in the pitch contour at or after `after_time_s`."""
    f0 = pitch.selected_array["frequency"]  # 0 = unvoiced
    times = pitch.xs()  # frame center times
    if len(f0) == 0:
        return None

    # Find first index >= after_time_s
    start_idx = int(np.searchsorted(times, after_time_s, side="left"))
    end_time = after_time_s + max_search_s
    end_idx = int(np.searchsorted(times, end_time, side="right"))
    end_idx = min(end_idx, len(f0))

    for i in range(start_idx, end_idx):
        if f0[i] > 0:
            return float(times[i])
    return None


def _voicing_onset_before(
    pitch: parselmouth.Data, before_time_s: float, *, max_search_s: float = 0.20
) -> float | None:
    """Most recent voiced frame ending before `before_time_s` (used to detect prevoicing)."""
    f0 = pitch.selected_array["frequency"]
    times = pitch.xs()
    if len(f0) == 0:
        return None

    start_time = max(0.0, before_time_s - max_search_s)
    start_idx = int(np.searchsorted(times, start_time, side="left"))
    end_idx = int(np.searchsorted(times, before_time_s, side="right"))
    end_idx = min(end_idx, len(f0))

    last_voiced: float | None = None
    for i in range(start_idx, end_idx):
        if f0[i] > 0:
            last_voiced = float(times[i])
    return last_voiced


def estimate_vot(
    sound: parselmouth.Sound,
    occurrences: list[PhonemeOccurrence],
) -> VotSummary:
    """Compute per-stop VOT and class summaries from phoneme timing + pitch contour."""
    measurements: list[VotMeasurement] = []
    if not occurrences:
        return VotSummary(measurements=[], aspirated_voiceless_mean_ms=None,
                          plain_voiceless_mean_ms=None, voiced_mean_ms=None)

    try:
        pitch = sound.to_pitch(time_step=0.005)  # 5 ms resolution for VOT precision
    except Exception:
        return VotSummary(measurements=[], aspirated_voiceless_mean_ms=None,
                          plain_voiceless_mean_ms=None, voiced_mean_ms=None)

    for occ in occurrences:
        if occ.phoneme not in ALL_STOPS:
            continue
        release_t = occ.end_s
        cls = _classify(occ.phoneme)

        if cls == "voiced":
            # Voiced stops can prevoice — voicing leads the release.
            prevoice = _voicing_onset_before(pitch, occ.start_s)
            if prevoice is not None:
                # Negative VOT = voicing lead duration before release
                vot_ms = (prevoice - release_t) * 1000.0
                measurements.append(VotMeasurement(
                    phoneme=occ.phoneme, time_s=release_t, vot_ms=vot_ms,
                    aspiration_class=cls,
                ))
                continue

        # Plain or aspirated: look for voicing AFTER release
        onset = _voicing_onset_after(pitch, release_t)
        if onset is None:
            continue
        vot_ms = (onset - release_t) * 1000.0
        # Sanity bound — anything wildly outside plausible VOT range is likely an artifact
        if -200 <= vot_ms <= 250:
            measurements.append(VotMeasurement(
                phoneme=occ.phoneme, time_s=release_t, vot_ms=vot_ms,
                aspiration_class=cls,
            ))

    by_class: dict[str, list[float]] = {
        "aspirated_voiceless": [],
        "plain_voiceless": [],
        "voiced": [],
    }
    for m in measurements:
        if m.aspiration_class in by_class:
            by_class[m.aspiration_class].append(m.vot_ms)

    return VotSummary(
        measurements=measurements,
        aspirated_voiceless_mean_ms=float(mean(by_class["aspirated_voiceless"])) if by_class["aspirated_voiceless"] else None,
        plain_voiceless_mean_ms=float(mean(by_class["plain_voiceless"])) if by_class["plain_voiceless"] else None,
        voiced_mean_ms=float(mean(by_class["voiced"])) if by_class["voiced"] else None,
    )
