"""Deterministic tests for marker-based language match scoring.

Builds PhonemeInventory by hand — no model load. Profiles are heuristics, so
tests check the SHAPE of the discrimination (English-like inventory matches
English and mismatches Arabic), not exact scores.
"""

from __future__ import annotations

from app.services import language_check, phonemes


def _inv(counts: dict[str, int]) -> phonemes.PhonemeInventory:
    return phonemes.PhonemeInventory(
        counts=dict(counts),
        total_tokens=sum(counts.values()),
        occurrences=[],
    )


# Synthesized inventories meant to look like fluent speech in each language.
# Token counts are ~40-60 to clear MIN_TOKENS_FOR_SCORING with room.
_ENGLISH_LIKE = {
    # Distinctive positive markers
    "θ": 2, "ð": 3,
    "ɹ": 5, "w": 3, "ə": 6,
    # Generic English speech-like phonemes
    "ɪ": 5, "ɛ": 3, "æ": 3, "ʌ": 2,
    "t": 6, "n": 5, "s": 4, "l": 3,
    "k": 3, "d": 3, "p": 2, "m": 3,
    "i": 2, "u": 1,
}

_SPANISH_LIKE = {
    # Distinctive markers
    "ɾ": 4, "ɲ": 2, "x": 2, "β": 2,
    # Generic Spanish speech
    "a": 8, "e": 6, "o": 5, "i": 4, "u": 2,
    "s": 5, "n": 4, "l": 3, "d": 3, "t": 3, "k": 2, "m": 3, "p": 2,
}

_MANDARIN_LIKE = {
    "tɕ": 2, "tɕʰ": 1, "ɕ": 2,
    "ʂ": 1, "ʐ": 1,
    "a": 8, "i": 4, "u": 4, "ə": 3,
    "n": 5, "ŋ": 4, "m": 3, "t": 3, "k": 2, "p": 2, "s": 2,
}

_ARABIC_LIKE = {
    "ħ": 2, "ʕ": 2, "x": 2, "q": 2, "tˤ": 1,
    "a": 10, "i": 4, "u": 3,
    "l": 5, "n": 4, "m": 4, "r": 4, "s": 3, "t": 3, "k": 3, "b": 2,
}


# ─── core verdicts ──────────────────────────────────────────────────────────


def test_english_inventory_matches_english_claim() -> None:
    inv = _inv(_ENGLISH_LIKE)
    res = language_check.score_language_match(inv, "english")
    assert res is not None
    assert res.verdict == "matches", f"expected matches, got {res.verdict} (score={res.score:.2f}, notes={res.notes})"
    assert res.score is not None and res.score >= 0.75


def test_arabic_inventory_matches_arabic_claim() -> None:
    inv = _inv(_ARABIC_LIKE)
    res = language_check.score_language_match(inv, "arabic")
    assert res is not None
    assert res.verdict == "matches", f"expected matches, got {res.verdict} (score={res.score:.2f}, notes={res.notes})"


def test_mandarin_inventory_matches_mandarin_claim() -> None:
    inv = _inv(_MANDARIN_LIKE)
    res = language_check.score_language_match(inv, "mandarin")
    assert res is not None
    assert res.verdict == "matches", f"expected matches, got {res.verdict} (score={res.score:.2f}, notes={res.notes})"


def test_english_inventory_mismatches_arabic_claim() -> None:
    """Speaker claims Arabic L1 but inventory looks English — should flag."""
    inv = _inv(_ENGLISH_LIKE)
    res = language_check.score_language_match(inv, "arabic")
    assert res is not None
    assert res.verdict == "mismatch", f"expected mismatch, got {res.verdict} (score={res.score:.2f}, notes={res.notes})"


def test_english_inventory_mismatches_mandarin_claim() -> None:
    inv = _inv(_ENGLISH_LIKE)
    res = language_check.score_language_match(inv, "mandarin")
    assert res is not None
    assert res.verdict == "mismatch", f"expected mismatch, got {res.verdict} (score={res.score:.2f}, notes={res.notes})"


def test_arabic_inventory_mismatches_english_claim() -> None:
    """The reverse — distinctive Arabic markers (pharyngeals, emphatic, /q/) contradict English."""
    inv = _inv(_ARABIC_LIKE)
    res = language_check.score_language_match(inv, "english")
    assert res is not None
    assert res.verdict == "mismatch", f"expected mismatch, got {res.verdict} (score={res.score:.2f}, notes={res.notes})"


def test_mandarin_inventory_mismatches_english_claim() -> None:
    inv = _inv(_MANDARIN_LIKE)
    res = language_check.score_language_match(inv, "english")
    assert res is not None
    assert res.verdict == "mismatch", f"expected mismatch, got {res.verdict} (score={res.score:.2f}, notes={res.notes})"


# ─── guard rails ────────────────────────────────────────────────────────────


def test_short_clip_returns_insufficient() -> None:
    inv = _inv({"a": 3, "t": 2, "n": 2})  # 7 tokens, below MIN_TOKENS_FOR_SCORING
    res = language_check.score_language_match(inv, "english")
    assert res is not None
    assert res.verdict == "insufficient_signal"
    assert res.score is None


def test_unknown_language_returns_unknown_verdict() -> None:
    inv = _inv(_ENGLISH_LIKE)
    res = language_check.score_language_match(inv, "klingon")
    assert res is not None
    assert res.verdict == "unknown_language"
    assert res.score is None


def test_empty_claimed_language_returns_none() -> None:
    inv = _inv(_ENGLISH_LIKE)
    assert language_check.score_language_match(inv, None) is None
    assert language_check.score_language_match(inv, "") is None


# ─── alias normalization ────────────────────────────────────────────────────


def test_language_lookup_fallbacks_to_first_or_last_word() -> None:
    """Free-form user input should resolve to canonical profiles via word fallback."""
    inv = _inv(_ENGLISH_LIKE)
    for raw in ("Mandarin Chinese", "Chinese Mandarin", "Mandarin (Beijing)"):
        res = language_check.score_language_match(inv, raw)
        assert res is not None and res.claimed_language == "mandarin", f"failed for {raw!r}"

    inv_pt = _inv({"a": 30, "n": 5, "s": 5, "ʁ": 3, "ɲ": 2, "ʎ": 1, "ẽ": 2})
    for raw in ("Brazilian Portuguese", "Portuguese (Brazil)", "European Portuguese"):
        res = language_check.score_language_match(inv_pt, raw)
        assert res is not None and res.claimed_language == "portuguese", f"failed for {raw!r}"

    inv_hi = _inv({"a": 20, "pʰ": 2, "ʈ": 3})
    res = language_check.score_language_match(inv_hi, "Hindi (India)")
    assert res is not None and res.claimed_language == "hindi"


def test_language_aliases_normalize() -> None:
    inv = _inv(_ENGLISH_LIKE)
    for alias in ("english", "English", " EN ", "en-us", "en_GB"):
        res = language_check.score_language_match(inv, alias)
        assert res is not None
        assert res.claimed_language == "english"

    inv_es = _inv(_SPANISH_LIKE)
    for alias in ("spanish", "es", "es-MX", "castilian"):
        res = language_check.score_language_match(inv_es, alias)
        assert res is not None
        assert res.claimed_language == "spanish"


# ─── result detail shape ────────────────────────────────────────────────────


def test_result_includes_positive_and_negative_breakdown() -> None:
    inv = _inv(_ENGLISH_LIKE)
    res = language_check.score_language_match(inv, "english")
    assert res is not None
    assert res.positive_total > 0
    assert res.negative_total > 0
    # positive_results reports every marker; positive_total may be smaller when
    # optional markers are missing (excluded from denominator).
    assert len(res.positive_results) >= res.positive_total
    assert len(res.negative_results) == res.negative_total
    # In a matching clip, most negative checks should be clean.
    assert res.negative_clean >= res.negative_total - 1


def test_hindi_optional_markers_dont_inflate_score_on_english_audio() -> None:
    """Regression: Hindi's voiced_aspirated and retroflex_flap are optional bonus
    markers. When absent (as in English speech), they should be excluded from the
    positive_total — not counted as phantom hits via the old threshold=0 bug."""
    inv = _inv(_ENGLISH_LIKE)
    res = language_check.score_language_match(inv, "hindi")
    assert res is not None
    assert res.verdict == "mismatch", f"expected mismatch, got {res.verdict} (score={res.score:.2f}, notes={res.notes})"
    assert res.positive_hits == 0
    # Hindi profile has 4 markers total; 2 are optional and missing → excluded.
    # Effective positive_total should be 2 (the required aspirated/retroflex stops).
    assert res.positive_total == 2


# Real-world phoneme counts from Shristi's English test clip (12.4s, 96 tokens).
# These are reverse-engineered from observed verdicts: the model emits English
# medial flap-r as /ɾ/, has heavy /l/, has /ð/ but few /θ/, etc. Used to lock
# in that thin profiles don't false-positive on actual English speech.
_REAL_ENGLISH_CLIP = {
    "θ": 0, "ð": 3, "ɹ": 2, "w": 2, "ə": 4,
    "ɾ": 2,        # English medial flap-r (city, butter)
    "l": 6,        # heavy English /l/
    "t": 8, "n": 6, "s": 5, "k": 3, "d": 3, "p": 2, "m": 3, "b": 2,
    "i": 4, "ɪ": 4, "ɛ": 3, "æ": 3, "ʌ": 2, "ɑ": 2, "u": 1,
    "f": 2, "v": 1,
}


def test_english_clip_does_not_false_positive_japanese() -> None:
    """Wav2Vec2 emits /ɾ/ for English medial flaps; threshold=4 keeps Japanese discriminative."""
    inv = _inv(_REAL_ENGLISH_CLIP)
    res = language_check.score_language_match(inv, "japanese")
    assert res is not None
    assert res.verdict == "mismatch", f"expected mismatch, got {res.verdict} (score={res.score:.2f}, notes={res.notes})"


def test_english_clip_does_not_false_positive_italian() -> None:
    inv = _inv(_REAL_ENGLISH_CLIP)
    res = language_check.score_language_match(inv, "italian")
    assert res is not None
    assert res.verdict == "mismatch", f"expected mismatch, got {res.verdict} (score={res.score:.2f}, notes={res.notes})"


def test_english_clip_does_not_false_positive_korean() -> None:
    """Korean's broad /ɾ l/ marker was removed because English /l/ is too frequent at any threshold."""
    inv = _inv(_REAL_ENGLISH_CLIP)
    res = language_check.score_language_match(inv, "korean")
    assert res is not None
    assert res.verdict == "mismatch", f"expected mismatch, got {res.verdict} (score={res.score:.2f}, notes={res.notes})"


def test_notes_omit_optional_missing_markers() -> None:
    """Cosmetic: optional markers that miss shouldn't appear in the 'expected but missing' note,
    so the visible count matches the denominator."""
    inv = _inv(_ENGLISH_LIKE)
    res = language_check.score_language_match(inv, "hindi")
    assert res is not None
    joined = " ".join(res.notes)
    # Hindi's two optional markers should NOT appear in the notes when missing.
    assert "voiced_aspirated" not in joined
    assert "retroflex_flap" not in joined


def test_hindi_optional_marker_counted_when_present() -> None:
    """If an optional marker is actually produced, it counts as a hit (and is included in the denominator)."""
    # Real-Hindi-like inventory: required markers hit + one optional (retroflex_flap) hit
    inv = _inv({
        "pʰ": 2, "tʰ": 2, "kʰ": 1,    # aspirated stops
        "ʈ": 3, "ɖ": 2,                # retroflex stops
        "ɽ": 1,                         # optional retroflex flap, PRESENT
        # voiced_aspirated NOT present — should be excluded
        "a": 8, "i": 4, "u": 3, "n": 4, "m": 3, "l": 2, "s": 3, "k": 2,
    })
    res = language_check.score_language_match(inv, "hindi")
    assert res is not None
    assert res.verdict == "matches", f"expected matches, got {res.verdict} (score={res.score:.2f}, notes={res.notes})"
    # 3 effective positives: 2 required + 1 optional-that-hit. Missing optional excluded.
    assert res.positive_total == 3
    assert res.positive_hits == 3


def test_mismatch_notes_are_human_readable() -> None:
    inv = _inv(_ENGLISH_LIKE)
    res = language_check.score_language_match(inv, "arabic")
    assert res is not None
    # The English-like inventory will violate Arabic's negatives (it has English /ɹ/ etc.)
    # AND miss Arabic's positives (no /ħ/, /ʕ/, /q/, /tˤ/).
    joined = " ".join(res.notes)
    assert "missing" in joined or "unexpected" in joined


# ─── supported_languages helper ─────────────────────────────────────────────


def test_supported_languages_lists_expected_keys() -> None:
    langs = set(language_check.supported_languages())
    # Spot-check that the headline languages are configured.
    assert {"english", "spanish", "french", "german", "mandarin",
            "hindi", "arabic", "japanese"} <= langs
