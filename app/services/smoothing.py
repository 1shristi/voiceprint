"""Confusion-matrix smoothing for free-decoded phoneme counts.

Loop 3a Phase 2. Even without forced alignment, the raw counts emitted by
Wav2Vec2-Phoneme can be improved by accounting for known model confusions:
a `s` detected in the audio is *evidence* that the speaker may have produced
`θ` — Wav2Vec2 is documented to misclassify /θ/ as /s/ in many L2 contexts.

This module is pure-function and self-contained: given a `raw_counts` dict
(keyed by whatever the tokenizer emits — a mix of IPA and eSpeak X-SAMPA
holdouts per app/data/phoneme_alphabet.json) it returns:

  - `smoothed_counts`: a per-symbol dict where every confusion-matrix target
    has its raw count augmented with weighted contributions from documented
    near-miss symbols, and every non-target symbol's smoothed value equals
    its raw count (unchanged).
  - `confusion_evidence`: a per-target audit trail naming which near-miss
    symbols contributed how much, so downstream consumers (and post-hoc
    debugging) can see exactly why a smoothed value moved away from raw.

Critical: this never mutates the raw counts. Loop 3a's whole guarantee is that
existing consumers continue reading `phonemes.counts` exactly as before, while
new consumers opt into the smoothed view.

The smoothed_counts keys are *canonical IPA* (matching the `ipa` field in
phoneme_alphabet.json). Wav2Vec2 emits X-SAMPA holdouts like `S`, `tS`, `dZ`
alongside their direct-IPA equivalents (`ʃ`, `tʃ`, `dʒ`); the alphabet table
collapses these onto a single canonical IPA form before smoothing so a clip
where the model happened to emit `S` rather than `ʃ` still counts as
postalveolar-fricative evidence under the same key linguamatch reads from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CONFUSION_MATRIX_PATH = _DATA_DIR / "confusion_matrix.json"
_ALPHABET_PATH = _DATA_DIR / "phoneme_alphabet.json"


# ─── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class EvidenceContribution:
    """One row in confusion_evidence[target].evidence_from."""

    from_symbol: str
    raw_count: int
    weight: float
    contribution: float


@dataclass
class ConfusionEvidence:
    """Per-target audit trail surfaced in the /analyze response."""

    raw_count: int
    smoothed_count: float
    evidence_from: list[EvidenceContribution] = field(default_factory=list)
    interpretation: str = ""


@dataclass
class SmoothingResult:
    smoothed_counts: dict[str, float]
    confusion_evidence: dict[str, ConfusionEvidence]


# ─── Data loading ───────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_confusion_matrix() -> dict[str, Any]:
    return json.loads(_CONFUSION_MATRIX_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_alphabet() -> dict[str, Any]:
    return json.loads(_ALPHABET_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _espeak_to_ipa_map() -> dict[str, str]:
    """Map every eSpeak vocab symbol to its canonical IPA form.

    Symbols where eSpeak == IPA (most of the alphabet — `s`, `θ`, `ð`, etc.)
    map to themselves and are no-ops at lookup time. The X-SAMPA holdouts
    (`S` -> `ʃ`, `tS` -> `tʃ`, `dZ` -> `dʒ`, `t^` -> `ʈ`, etc.) collapse
    onto their canonical IPA so the smoother and downstream consumers
    don't need to know the eSpeak/IPA dichotomy exists.
    """
    alphabet = _load_alphabet()
    out: dict[str, str] = {}
    for entry in alphabet.get("alphabet", []):
        espeak = entry["espeak_symbol"]
        ipa = entry["ipa"]
        out[espeak] = ipa
    return out


def _normalise_counts_to_ipa(raw_counts: dict[str, int]) -> dict[str, int]:
    """Aggregate counts under canonical IPA keys.

    Wav2Vec2 emits some symbols as eSpeak X-SAMPA (e.g. `S` for postalveolar
    fricative) and others as direct IPA (`ʃ` for the same phoneme). For
    smoothing, we want both to count toward the same target. This collapses
    the dict onto IPA-keyed counts using the alphabet table; unknown symbols
    (vocab entries not in the alphabet) keep their original key.
    """
    lookup = _espeak_to_ipa_map()
    normalised: dict[str, int] = {}
    for sym, count in raw_counts.items():
        if not isinstance(count, (int, float)):
            continue
        ipa = lookup.get(sym, sym)
        normalised[ipa] = normalised.get(ipa, 0) + int(count)
    return normalised


# ─── Smoothing core ─────────────────────────────────────────────────────────


def _build_interpretation(
    target: str,
    raw_count: int,
    smoothed_count: float,
    contributions: list[EvidenceContribution],
) -> str:
    """Build the user-readable interpretation string per spec §6.3 example."""
    if not contributions and raw_count == 0:
        return (
            f"Direct detection: 0. No near-miss evidence for /{target}/ either. "
            "Speaker did not produce this phoneme in any clip."
        )
    if not contributions:
        # Target appeared directly but no near-miss contributors.
        return f"Direct detection: {raw_count}. No additional smoothed evidence."
    source_symbols = ", ".join(c.from_symbol for c in contributions)
    if raw_count == 0:
        return (
            f"Direct detection: 0. Smoothed evidence: {smoothed_count:.2f} "
            f"(from {source_symbols}). Speaker may produce /{target}/ that the "
            "model misclassifies."
        )
    return (
        f"Direct detection: {raw_count}. Smoothed evidence: {smoothed_count:.2f} "
        f"(direct + contributions from {source_symbols})."
    )


def compute_smoothed_counts(raw_counts: dict[str, int]) -> SmoothingResult:
    """Compute smoothed counts and per-target evidence from raw counts.

    The returned `smoothed_counts` is a full copy of the (IPA-normalised) raw
    counts: every non-target symbol's smoothed value equals its raw value, and
    every confusion-matrix target's smoothed value is `raw + sum(weight * raw_NM)`.
    Targets that the matrix lists but the audio never emitted directly *and*
    have no near-miss evidence are omitted from `smoothed_counts` (they would
    add zero-valued noise to the response). The evidence dict is populated only
    for targets where at least one near-miss contributed or the target had a
    non-zero raw count — pure-zero targets are skipped to keep the response lean.
    """
    matrix = _load_confusion_matrix()
    confusions = matrix.get("confusions", {})

    ipa_counts = _normalise_counts_to_ipa(raw_counts)
    smoothed: dict[str, float] = {sym: float(c) for sym, c in ipa_counts.items()}
    evidence: dict[str, ConfusionEvidence] = {}

    for target, target_data in confusions.items():
        raw_target = ipa_counts.get(target, 0)
        contributions: list[EvidenceContribution] = []
        smoothed_extra = 0.0

        for nm in target_data.get("near_misses", []):
            from_symbol = nm["from_symbol"]
            weight = float(nm["weight"])
            raw_nm = ipa_counts.get(from_symbol, 0)
            contribution = weight * raw_nm
            smoothed_extra += contribution
            # Only surface contributions where the near-miss actually fired —
            # otherwise evidence_from is dominated by zero rows for every
            # target whose near-misses didn't appear in the clip.
            if raw_nm > 0:
                contributions.append(
                    EvidenceContribution(
                        from_symbol=from_symbol,
                        raw_count=raw_nm,
                        weight=weight,
                        contribution=contribution,
                    )
                )

        new_smoothed = float(raw_target) + smoothed_extra

        # Always emit smoothed_counts[target] when the target had ANY signal
        # (direct or smoothed), so consumers can see the smoothed view for
        # every matrix target the clip touched. Targets the clip didn't touch
        # at all are skipped to keep the response from ballooning.
        if new_smoothed > 0:
            smoothed[target] = new_smoothed
        if contributions or raw_target > 0:
            evidence[target] = ConfusionEvidence(
                raw_count=raw_target,
                smoothed_count=new_smoothed,
                evidence_from=contributions,
                interpretation=_build_interpretation(
                    target, raw_target, new_smoothed, contributions
                ),
            )

    return SmoothingResult(smoothed_counts=smoothed, confusion_evidence=evidence)
