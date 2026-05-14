"""Deterministic tests for stretch-phoneme scoring.

These build a `PhonemeInventory` by hand and verify the scoring without
loading the Wav2Vec2 model.
"""

from __future__ import annotations

from app.services import phonemes


def _inv(counts: dict[str, int]) -> phonemes.PhonemeInventory:
    """Build a bare PhonemeInventory from a counts dict — no occurrences needed for scoring."""
    return phonemes.PhonemeInventory(
        counts=dict(counts),
        total_tokens=sum(counts.values()),
        occurrences=[],
    )


# ─── score_stretch — Arabic ──────────────────────────────────────────────────


def test_arabic_perfect_production_all_hit() -> None:
    """A speaker who produced every probe phoneme at canonical count gets all hits."""
    inv = _inv({
        "x": 2,    # Kh × 2
        "ħ": 2,    # ḥ × 2
        "ʕ": 1,    # ʿ × 1
        "tˤ": 2,   # ṭ × 2
        "r": 1,    # r × 1
        # Other speech tokens
        "a": 8, "i": 4, "l": 2,
    })
    score = phonemes.score_stretch(inv, "arabic")
    assert score is not None
    assert score.expected_language == "arabic"
    statuses = {p.label: p.status for p in score.probes}
    assert statuses == {
        "voiceless_velar_fricative": "hit",
        "voiceless_pharyngeal_fricative": "hit",
        "voiced_pharyngeal_fricative": "hit",
        "emphatic_t": "hit",
        "alveolar_trill": "hit",
    }


def test_arabic_english_speaker_substitutes_get_approximate() -> None:
    """English speaker maps ħ→h, ʕ→ʔ, tˤ→t, r→ɹ; everything lands as approximate."""
    inv = _inv({
        "h": 4,    # for both /x/ (approximate target k/h) and /ħ/ (approximate h)
        "ʔ": 1,    # for /ʕ/
        "t": 2,    # for /tˤ/
        "ɹ": 1,    # for /r/
        "a": 8,
    })
    score = phonemes.score_stretch(inv, "arabic")
    assert score is not None
    statuses = {p.label: p.status for p in score.probes}
    # /x/'s approximate set includes "h", so it lands as approximate not missed
    assert statuses["voiceless_velar_fricative"] == "approximate"
    assert statuses["voiceless_pharyngeal_fricative"] == "approximate"
    assert statuses["voiced_pharyngeal_fricative"] == "approximate"
    assert statuses["emphatic_t"] == "approximate"
    assert statuses["alveolar_trill"] == "approximate"


def test_arabic_silent_clip_all_missed() -> None:
    """Empty inventory → every probe missed."""
    inv = _inv({})
    score = phonemes.score_stretch(inv, "arabic")
    assert score is not None
    for p in score.probes:
        assert p.status == "missed"
        assert p.count == 0
        assert p.approximate_count == 0


def test_arabic_partial_count_yields_approximate() -> None:
    """If the speaker produces /ħ/ once but expected is 2 → approximate, not hit."""
    inv = _inv({"ħ": 1, "x": 2, "ʕ": 1, "tˤ": 2, "r": 1})
    score = phonemes.score_stretch(inv, "arabic")
    assert score is not None
    by_label = {p.label: p for p in score.probes}
    h_result = by_label["voiceless_pharyngeal_fricative"]
    assert h_result.status == "approximate"
    assert h_result.count == 1
    assert h_result.expected_count == 2


# ─── score_stretch — Mandarin ────────────────────────────────────────────────


def test_mandarin_aspirated_palatal_present_is_hit() -> None:
    inv = _inv({"tɕʰ": 1, "m": 6, "a": 8})
    score = phonemes.score_stretch(inv, "mandarin")
    assert score is not None
    assert len(score.probes) == 1
    probe = score.probes[0]
    assert probe.label == "aspirated_palatal_affricate"
    assert probe.status == "hit"


def test_mandarin_unaspirated_substitute_is_approximate() -> None:
    """English speaker says /tʃ/ instead of /tɕʰ/ — that's in the approximate set."""
    inv = _inv({"tʃ": 1, "m": 6, "a": 8})
    score = phonemes.score_stretch(inv, "mandarin")
    assert score is not None
    assert score.probes[0].status == "approximate"
    assert score.probes[0].approximate_count == 1
    assert score.probes[0].count == 0


def test_mandarin_no_palatal_at_all_is_missed() -> None:
    inv = _inv({"m": 6, "a": 8, "n": 1})
    score = phonemes.score_stretch(inv, "mandarin")
    assert score is not None
    assert score.probes[0].status == "missed"


# ─── language alias normalization ───────────────────────────────────────────


def test_language_aliases_are_accepted() -> None:
    inv = _inv({"x": 2, "ħ": 2, "ʕ": 1, "tˤ": 2, "r": 1})
    for alias in ("arabic", "Arabic", " AR ", "msa", "MSA"):
        score = phonemes.score_stretch(inv, alias)
        assert score is not None, f"alias {alias!r} should be accepted"
        assert score.expected_language == "arabic"

    for alias in ("mandarin", "Chinese", "zh", "zh-CN", "cmn"):
        score = phonemes.score_stretch(_inv({"tɕʰ": 1}), alias)
        assert score is not None, f"alias {alias!r} should be accepted"
        assert score.expected_language == "mandarin"


def test_unknown_language_returns_none() -> None:
    inv = _inv({"x": 2, "ħ": 2})
    assert phonemes.score_stretch(inv, "french") is None
    assert phonemes.score_stretch(inv, "") is None
    assert phonemes.score_stretch(inv, None) is None
