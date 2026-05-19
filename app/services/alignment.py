"""CTC forced-alignment against a known transcript.

Loop 3a Phase 1. Free-decoded phoneme detection asks "did /θ/ appear anywhere
in 20 seconds of audio?". Forced alignment asks "did the speaker produce /θ/
in the four positions the diagnostic sentence required it?". The second
question is far more sensitive — a speaker who produces three /θ/s and one
near-miss /s/ is producing /θ/, not absent for it.

This module operates on logits that have already been computed by the
Wav2Vec2-Phoneme forward pass (see `phonemes.extract_phonemes` with
`include_alignment_artifacts=True`), so the model runs exactly once per clip
even when both free-decoded counts and forced alignment are needed.

Design:

- `tokenize_transcript` converts a transcript (IPA / eSpeak / orthography)
  into an ordered list of phoneme symbols matching the model's tokenizer
  vocabulary. Phonemes that have no vocab equivalent emit a warning and are
  skipped; alignment continues around them.
- `align_against_transcript` runs `torchaudio.functional.forced_align` over
  the log-probability tensor and the tokenized transcript. For each target
  position it extracts the span (start/end ms), avg log-prob, top-1
  prediction and top-3 alternatives.
- `classify_match` and `summarise_by_phoneme` apply the thresholds from
  `app/data/alignment_thresholds.json` to produce per-position labels and a
  rolled-up summary keyed by target phoneme.

The function call into `torchaudio.functional.forced_align` is the only
non-trivial dependency; everything else operates on tensors and dicts and
is straightforward to unit-test with synthetic logits.
"""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_ALPHABET_PATH = _DATA_DIR / "phoneme_alphabet.json"
_THRESHOLDS_PATH = _DATA_DIR / "alignment_thresholds.json"

_NON_PHONEME_TOKENS = {"<pad>", "<s>", "</s>", "<unk>", "|", "[PAD]", "[UNK]", " ", ""}


# ─── Public dataclasses ─────────────────────────────────────────────────────


@dataclass
class TranscriptInput:
    """User-supplied transcript that voiceprint should align audio against."""

    format: str  # "ipa" | "espeak" | "orthography_with_dictionary"
    content: str
    expected_language: str | None = None  # only consulted for orthography→IPA


@dataclass
class AlignedTop3Alternative:
    phoneme: str
    prob: float


@dataclass
class AlignedPosition:
    target_phoneme: str
    target_index_in_transcript: int
    start_ms: int
    end_ms: int
    avg_log_prob: float
    top1_predicted: str
    top3_alternatives: list[AlignedTop3Alternative]
    match_classification: str  # "produced" | "near_miss" | "absent"


@dataclass
class AlignedPhonemeSummary:
    expected_count: int
    produced_count: int
    near_miss_count: int
    absent_count: int
    evidence_strength: str  # "strong" | "moderate" | "weak"


@dataclass
class AlignmentResult:
    transcript_format: str
    alignment_quality: str  # "high" | "medium" | "low"
    positions: list[AlignedPosition]
    summary_by_phoneme: dict[str, AlignedPhonemeSummary]
    alignment_warnings: list[str] = field(default_factory=list)


# ─── Alphabet + thresholds loading ──────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_alphabet() -> dict[str, Any]:
    """Lazy-load `phoneme_alphabet.json`. Cached at process scope."""
    return json.loads(_ALPHABET_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_thresholds() -> dict[str, Any]:
    """Lazy-load `alignment_thresholds.json`. Cached at process scope."""
    return json.loads(_THRESHOLDS_PATH.read_text(encoding="utf-8"))


def _alphabet_index(alphabet: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Return (ipa→espeak, espeak→ipa) mappings derived from alphabet.json.

    Both lookups are normalised to NFC because Unicode equivalence classes
    (especially combining diacritics) bite forced-alignment hard otherwise.
    """
    ipa_to_espeak: dict[str, str] = {}
    espeak_to_ipa: dict[str, str] = {}
    for entry in alphabet.get("alphabet", []):
        espeak = unicodedata.normalize("NFC", entry["espeak_symbol"])
        ipa = unicodedata.normalize("NFC", entry["ipa"])
        # eSpeak symbols are unique (they are the vocab); IPA may collide
        # (multiple eSpeak symbols can map to the same canonical IPA). Don't
        # overwrite the first-seen mapping for IPA→eSpeak — it's the most
        # canonical match.
        ipa_to_espeak.setdefault(ipa, espeak)
        for alt in entry.get("ipa_alternates", []):
            ipa_to_espeak.setdefault(unicodedata.normalize("NFC", alt.strip("/")), espeak)
        espeak_to_ipa.setdefault(espeak, ipa)
    return ipa_to_espeak, espeak_to_ipa


# ─── Transcript tokenisation ────────────────────────────────────────────────


_PHONEMIZER_LANG_MAP = {
    "en": "en-us",
    "english": "en-us",
    "zh": "cmn",
    "mandarin": "cmn",
    "chinese": "cmn",
    "ar": "ar",
    "arabic": "arabic",
    "msa": "arabic",
}


def _normalise_transcript_text(content: str) -> str:
    """NFC-normalise + strip wrapping slashes that human-written IPA often carries."""
    return unicodedata.normalize("NFC", content).strip()


def _split_ipa(content: str) -> list[str]:
    """Split a flat IPA string into atomic glyphs (base + combining marks).

    Forced alignment needs each transcript symbol to correspond to one vocab
    entry. Pre-composed sequences like /tʃ/, /tɕʰ/, /aɪ/ stay together — the
    model's vocabulary has them as single tokens. Plain whitespace, slashes,
    word boundary markers, and stress marks are dropped (the vocab has no
    entries for them; they'd produce dead positions).
    """
    cleaned = (
        _normalise_transcript_text(content)
        .replace("/", " ")
        .replace(".", " ")  # syllable boundary in IPA
    )
    # Strip primary/secondary stress and other suprasegmental marks; they're
    # not in the vocab and add noise to alignment.
    cleaned = cleaned.replace("ˈ", "").replace("ˌ", "")
    tokens: list[str] = []
    for word in cleaned.split():
        # Within a word, walk character-by-character but absorb combining
        # marks, tie bars and length markers into the previous base symbol so
        # tokens like /tʃ/, /t͡ʃ/, /aː/ stay intact.
        cur = ""
        for ch in word:
            cat = unicodedata.category(ch)
            if cat.startswith("M") or ch in {"ː", "ˑ", "ʰ", "ʷ", "ʲ", "ˤ", "ˠ", "̥", "̪", "̩", "͡"}:
                cur += ch
            elif ch in {"ʃ", "ʒ", "ɕ", "ʑ", "ɲ", "ŋ", "ɭ", "ʈ", "ɖ", "ʂ", "ʐ"}:
                # Postalveolar/retroflex glyphs are atomic but may follow a
                # /t/ or /d/ to form an affricate — only merge if the
                # previous char is t/d/ʈ/ɖ AND the buffer currently holds it.
                if cur in {"t", "d", "ʈ", "ɖ"}:
                    cur += ch
                else:
                    if cur:
                        tokens.append(cur)
                    cur = ch
            else:
                if cur:
                    tokens.append(cur)
                cur = ch
        if cur:
            tokens.append(cur)
    return tokens


def _orthography_to_ipa(content: str, expected_language: str | None) -> tuple[str, list[str]]:
    """Convert orthography to IPA via phonemizer. Returns (ipa, warnings)."""
    warnings: list[str] = []
    if not expected_language:
        warnings.append(
            "transcript.format=orthography_with_dictionary requires "
            "transcript.expected_language; defaulting to en-us."
        )
        locale = "en-us"
    else:
        locale = _PHONEMIZER_LANG_MAP.get(expected_language.strip().lower(), expected_language.strip().lower())
    try:
        from phonemizer import phonemize

        ipa = phonemize(
            content,
            language=locale,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            with_stress=True,
        )
    except Exception as e:  # noqa: BLE001 — graceful degradation
        warnings.append(
            f"phonemizer failed ({type(e).__name__}); orthography transcript "
            "could not be converted to IPA. Alignment skipped."
        )
        return "", warnings
    return ipa, warnings


def tokenize_transcript(
    transcript: TranscriptInput,
    vocab: dict[str, int],
) -> tuple[list[tuple[str, int]], list[str]]:
    """Tokenise a transcript into (target_phoneme_ipa, vocab_index) pairs.

    Returns a list of (ipa, vocab_idx) tuples preserving transcript order, plus
    warnings for any positions that couldn't be mapped to a vocab entry. The
    IPA glyph is preserved on the output so the response surface stays
    consumer-friendly (linguamatch reads `target_phoneme` as IPA).
    """
    warnings: list[str] = []
    alphabet = _load_alphabet()
    ipa_to_espeak, espeak_to_ipa = _alphabet_index(alphabet)
    vocab_norm = {unicodedata.normalize("NFC", k): v for k, v in vocab.items()}

    fmt = transcript.format.lower().strip()
    if fmt == "orthography_with_dictionary":
        ipa_text, ortho_warnings = _orthography_to_ipa(
            transcript.content, transcript.expected_language
        )
        warnings.extend(ortho_warnings)
        if not ipa_text:
            return [], warnings
        symbols = _split_ipa(ipa_text)
    elif fmt == "ipa":
        symbols = _split_ipa(transcript.content)
    elif fmt == "espeak":
        # eSpeak symbols are already vocab keys; split on whitespace and drop
        # empty entries. Don't NFC here — eSpeak X-SAMPA holdouts like "tS"
        # are ASCII and shouldn't be touched.
        symbols = [s for s in transcript.content.split() if s]
    else:
        warnings.append(
            f"Unknown transcript format {transcript.format!r}; expected one of "
            "ipa, espeak, orthography_with_dictionary. Alignment skipped."
        )
        return [], warnings

    tokens: list[tuple[str, int]] = []
    for position, sym in enumerate(symbols):
        if sym in _NON_PHONEME_TOKENS:
            continue
        nfc = unicodedata.normalize("NFC", sym)
        # Try direct vocab hit first (the symbol may already be in the
        # tokenizer vocabulary, e.g. "tʃ" is a single vocab entry).
        if nfc in vocab_norm:
            tokens.append((nfc, vocab_norm[nfc]))
            continue
        # Otherwise translate IPA → eSpeak via the alphabet table.
        espeak = ipa_to_espeak.get(nfc)
        if espeak is not None and espeak in vocab_norm:
            ipa_canonical = espeak_to_ipa.get(espeak, nfc)
            tokens.append((ipa_canonical, vocab_norm[espeak]))
            continue
        warnings.append(
            f"Phoneme /{sym}/ at position {position} has no model vocabulary "
            "mapping; alignment skipped for this position."
        )
    return tokens, warnings


# ─── Match classification ───────────────────────────────────────────────────


def _classify_match(
    target_vocab_idx: int,
    span_log_probs: Any,
    span_top1: int,
    top3_indices: list[int],
    top3_probs: list[float],
    thresholds: dict[str, Any],
) -> tuple[str, float]:
    """Classify a single aligned position. Returns (label, avg_log_prob).

    span_log_probs is a 1-D tensor of per-frame log_probs for the *target*
    token across the assigned frames. We average them in linear-prob space
    via the standard logsumexp trick to avoid skewing toward outlier frames.
    """
    import torch

    if hasattr(span_log_probs, "numel") and span_log_probs.numel() == 0:
        # Should not happen in practice; the caller only ever invokes this with
        # a non-empty span. Return a very-negative finite sentinel rather than
        # -inf so the response stays JSON-serialisable.
        return "absent", -1e9
    avg_log_prob = float(torch.mean(span_log_probs).item())
    if not math.isfinite(avg_log_prob):
        avg_log_prob = -1e9

    t = thresholds["thresholds"]
    if span_top1 == target_vocab_idx and avg_log_prob >= t["produced_min_avg_log_prob"]:
        return "produced", avg_log_prob

    # near-miss: target shows up in top-3 with non-trivial probability
    if target_vocab_idx in top3_indices:
        idx = top3_indices.index(target_vocab_idx)
        target_prob = top3_probs[idx]
        if target_prob >= t["near_miss_min_top3_prob"]:
            return "near_miss", avg_log_prob

    # absent: target either not in top-3 at all, or top-3 prob is trivially
    # low. The else branch catches the "near top-3 but very low" tail.
    return "absent", avg_log_prob


def _classify_alignment_quality(
    avg_log_probs: list[float], thresholds: dict[str, Any]
) -> str:
    """Roll per-position avg_log_probs into a single high/medium/low label."""
    if not avg_log_probs:
        return "low"
    finite = [v for v in avg_log_probs if math.isfinite(v)]
    if not finite:
        return "low"
    grand_mean = sum(finite) / len(finite)
    q = thresholds["alignment_quality_thresholds"]
    if grand_mean >= q["high_min_avg_log_prob"]:
        return "high"
    if grand_mean >= q["medium_min_avg_log_prob"]:
        return "medium"
    return "low"


def _summarise_by_phoneme(
    positions: list[AlignedPosition], thresholds: dict[str, Any]
) -> dict[str, AlignedPhonemeSummary]:
    """Roll per-position classifications into a per-target-phoneme summary."""
    buckets: dict[str, dict[str, int]] = {}
    for pos in positions:
        b = buckets.setdefault(
            pos.target_phoneme, {"expected": 0, "produced": 0, "near_miss": 0, "absent": 0}
        )
        b["expected"] += 1
        b[pos.match_classification] += 1

    r = thresholds["evidence_strength_ratios"]
    summary: dict[str, AlignedPhonemeSummary] = {}
    for phoneme, b in buckets.items():
        expected = b["expected"]
        produced = b["produced"]
        near_miss = b["near_miss"]
        absent = b["absent"]
        strong_floor = math.ceil(expected * r["strong_min_produced_ratio"])
        moderate_floor = math.ceil(expected * r["moderate_min_produced_plus_near_miss_ratio"])
        if produced >= strong_floor:
            strength = "strong"
        elif produced + near_miss >= moderate_floor:
            strength = "moderate"
        else:
            strength = "weak"
        summary[phoneme] = AlignedPhonemeSummary(
            expected_count=expected,
            produced_count=produced,
            near_miss_count=near_miss,
            absent_count=absent,
            evidence_strength=strength,
        )
    return summary


# ─── Forced-alignment driver ────────────────────────────────────────────────


def _top3_at_frame(
    log_probs_frame: Any, vocab_inverse: dict[int, str]
) -> tuple[list[int], list[float], list[str]]:
    """Top-3 vocab indices, their probabilities, and IPA labels at a single frame."""
    import torch

    k = min(3, log_probs_frame.shape[-1])
    top_log_probs, top_indices = torch.topk(log_probs_frame, k=k)
    indices = top_indices.tolist()
    probs = [float(math.exp(lp)) for lp in top_log_probs.tolist()]
    labels = [vocab_inverse.get(i, f"<vocab:{i}>") for i in indices]
    return indices, probs, labels


def align_against_transcript(
    log_probs: Any,
    *,
    blank_id: int,
    vocab: dict[str, int],
    audio_duration_s: float,
    transcript: TranscriptInput,
) -> AlignmentResult:
    """Run CTC forced alignment of `log_probs` against `transcript`.

    `log_probs` is a (T, V) tensor of log-probabilities (already passed
    through log_softmax). `vocab` maps tokenizer symbol → vocab index.
    `audio_duration_s` lets us convert frame indices back to millisecond
    positions for the response.
    """
    import torch
    from torchaudio.functional import forced_align

    thresholds = _load_thresholds()
    _, espeak_to_ipa = _alphabet_index(_load_alphabet())
    vocab_inverse_espeak: dict[int, str] = {v: k for k, v in vocab.items()}
    vocab_inverse_ipa: dict[int, str] = {
        idx: espeak_to_ipa.get(sym, sym) for idx, sym in vocab_inverse_espeak.items()
    }

    tokens, warnings = tokenize_transcript(transcript, vocab)

    # Empty transcript → no positions but the response is still well-formed.
    if not tokens:
        warnings.append("Transcript produced no alignable tokens; alignment skipped.")
        return AlignmentResult(
            transcript_format=transcript.format,
            alignment_quality="low",
            positions=[],
            summary_by_phoneme={},
            alignment_warnings=warnings,
        )

    target_ids = [vocab_idx for _, vocab_idx in tokens]
    target_ipa = [ipa for ipa, _ in tokens]

    num_frames = int(log_probs.shape[0])
    num_targets = len(target_ids)

    if num_targets > num_frames:
        warnings.append(
            f"Transcript ({num_targets} phonemes) is longer than the audio "
            f"({num_frames} frames); alignment skipped."
        )
        return AlignmentResult(
            transcript_format=transcript.format,
            alignment_quality="low",
            positions=[],
            summary_by_phoneme={},
            alignment_warnings=warnings,
        )

    log_probs_batched = log_probs.unsqueeze(0) if log_probs.dim() == 2 else log_probs
    targets_tensor = torch.tensor([target_ids], dtype=torch.long)

    try:
        alignment, scores = forced_align(
            log_probs=log_probs_batched.contiguous(),
            targets=targets_tensor,
            blank=blank_id,
        )
    except Exception as e:  # noqa: BLE001 — surface as warning, never crash the response
        warnings.append(
            f"torchaudio.functional.forced_align raised "
            f"{type(e).__name__}: {e}. Alignment skipped."
        )
        return AlignmentResult(
            transcript_format=transcript.format,
            alignment_quality="low",
            positions=[],
            summary_by_phoneme={},
            alignment_warnings=warnings,
        )

    alignment_seq = alignment[0].tolist()
    frame_duration_ms = (audio_duration_s * 1000.0) / num_frames if num_frames else 0.0

    # forced_align returns per-frame vocab IDs (with blanks interleaved). The
    # target-position index is implicit in the order non-blank tokens appear.
    # Walk the alignment, group consecutive frames that share the same
    # vocab id, and emit one position per target token in order.
    positions: list[AlignedPosition] = []
    avg_log_probs: list[float] = []

    target_position = 0
    frame_idx = 0
    while frame_idx < num_frames and target_position < num_targets:
        token_id = alignment_seq[frame_idx]
        if token_id == blank_id:
            frame_idx += 1
            continue
        # Find the contiguous run of this token id.
        run_start = frame_idx
        while frame_idx < num_frames and alignment_seq[frame_idx] == token_id:
            frame_idx += 1
        run_end = frame_idx  # exclusive

        target_vocab_id = target_ids[target_position]
        target_phoneme_ipa = target_ipa[target_position]

        # Average log-prob over the assigned span at the target's vocab index.
        span_log_probs_at_target = log_probs[run_start:run_end, target_vocab_id]

        # top1 over the span = argmax of mean log-probs across the span vocab.
        span_logits_mean = log_probs[run_start:run_end].mean(dim=0)
        span_top1_idx = int(torch.argmax(span_logits_mean).item())

        # top-3 over the same span-averaged distribution.
        top3_indices, top3_probs, _top3_labels = _top3_at_frame(
            span_logits_mean, vocab_inverse_ipa
        )

        classification, avg_log_prob = _classify_match(
            target_vocab_idx=target_vocab_id,
            span_log_probs=span_log_probs_at_target,
            span_top1=span_top1_idx,
            top3_indices=top3_indices,
            top3_probs=top3_probs,
            thresholds=thresholds,
        )
        avg_log_probs.append(avg_log_prob)

        positions.append(
            AlignedPosition(
                target_phoneme=target_phoneme_ipa,
                target_index_in_transcript=target_position,
                start_ms=int(round(run_start * frame_duration_ms)),
                end_ms=int(round(run_end * frame_duration_ms)),
                avg_log_prob=avg_log_prob,
                top1_predicted=vocab_inverse_ipa.get(span_top1_idx, f"<vocab:{span_top1_idx}>"),
                top3_alternatives=[
                    AlignedTop3Alternative(
                        phoneme=vocab_inverse_ipa.get(idx, f"<vocab:{idx}>"),
                        prob=prob,
                    )
                    for idx, prob in zip(top3_indices, top3_probs)
                ],
                match_classification=classification,
            )
        )
        target_position += 1

    # Pathological case: forced_align consumed fewer target tokens than expected.
    # torchaudio.functional.forced_align should never produce this when
    # target_length <= input_length, so treat as a warning rather than
    # manufacturing zero-duration positions that violate the start_ms<end_ms
    # invariant the spec requires.
    if target_position < num_targets:
        missing = num_targets - target_position
        warnings.append(
            f"Forced alignment consumed only {target_position} of {num_targets} "
            f"target phonemes; {missing} positions could not be aligned and are "
            "omitted from positions/summary_by_phoneme."
        )

    summary = _summarise_by_phoneme(positions, thresholds)
    quality = _classify_alignment_quality(avg_log_probs, thresholds)

    return AlignmentResult(
        transcript_format=transcript.format,
        alignment_quality=quality,
        positions=positions,
        summary_by_phoneme=summary,
        alignment_warnings=warnings,
    )
