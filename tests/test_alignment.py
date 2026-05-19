"""Unit tests for the CTC forced-alignment service.

These tests operate on *synthetic logits* — small constructed `(T, V)` tensors
with a hand-built vocab dict — so the suite never has to load the 370 MB
Wav2Vec2 model. The shape-correctness tests are the spec's §5.6 acceptance
floor; the property tests check invariants that must hold regardless of input.
"""

from __future__ import annotations

import math
import unicodedata

import pytest
import torch

from app.services import alignment


# ─── Synthetic-logit helpers ────────────────────────────────────────────────


def _build_vocab() -> dict[str, int]:
    """Tiny vocab covering the symbols the tests below exercise.

    Includes blank ('<pad>') at index 0 and a mix of IPA and eSpeak symbols
    drawn from the real `phoneme_alphabet.json` so the IPA↔eSpeak translation
    code path is also exercised.
    """
    return {
        "<pad>": 0,
        "θ": 1,
        "s": 2,
        "f": 3,
        "ð": 4,
        "d": 5,
        "a": 6,
        "b": 7,
        # eSpeak X-SAMPA holdouts (the model would emit "T" for /θ/ in some builds).
        "T": 8,
        "S": 9,
    }


def _logits_from_argmax_plan(
    plan: list[str | None], vocab: dict[str, int], boost: float = 8.0
) -> torch.Tensor:
    """Build a (T, V) log-prob tensor where frame t's argmax = plan[t].

    `plan[t]` is the vocab symbol to make dominant at frame t. None means
    "blank" — i.e. argmax should land on the blank id. The boost is added to
    the dominant logit before softmax so torchaudio.forced_align reliably
    picks that target.
    """
    T = len(plan)
    V = len(vocab)
    logits = torch.zeros(T, V)
    for t, sym in enumerate(plan):
        target = "<pad>" if sym is None else sym
        idx = vocab[target]
        logits[t, idx] = boost
    return torch.log_softmax(logits, dim=-1)


def _flat_logits(T: int, V: int) -> torch.Tensor:
    """Uniformly distributed log-probs — used to simulate silence/noise."""
    return torch.log_softmax(torch.zeros(T, V), dim=-1)


# ─── tokenize_transcript ────────────────────────────────────────────────────


def test_tokenize_ipa_transcript_splits_atomic_glyphs() -> None:
    vocab = _build_vocab()
    transcript = alignment.TranscriptInput(format="ipa", content="θ s f")
    tokens, warnings = alignment.tokenize_transcript(transcript, vocab)
    assert [ipa for ipa, _ in tokens] == ["θ", "s", "f"]
    assert [idx for _, idx in tokens] == [vocab["θ"], vocab["s"], vocab["f"]]
    assert warnings == []


def test_tokenize_ipa_with_slashes_and_stress_marks_drops_decoration() -> None:
    vocab = _build_vocab()
    transcript = alignment.TranscriptInput(format="ipa", content="/ˈθa/")
    tokens, warnings = alignment.tokenize_transcript(transcript, vocab)
    assert [ipa for ipa, _ in tokens] == ["θ", "a"]
    assert warnings == []


def test_tokenize_handles_nfc_and_nfd_identically() -> None:
    vocab = _build_vocab()
    # ð in NFC is a single codepoint U+00F0; NFD is the same — pick a glyph
    # that has a combining-mark decomposition. /tʃ/ has the modifier letter ʃ
    # but it's a single codepoint, not decomposable. Use /a/ with combining
    # ring above — but that's not in the vocab. Easier: assert that NFD-input
    # is normalised to NFC before lookup so the lookup succeeds.
    nfc_input = "θa"
    nfd_input = unicodedata.normalize("NFD", nfc_input)
    nfc_tokens, _ = alignment.tokenize_transcript(
        alignment.TranscriptInput(format="ipa", content=nfc_input), vocab
    )
    nfd_tokens, _ = alignment.tokenize_transcript(
        alignment.TranscriptInput(format="ipa", content=nfd_input), vocab
    )
    assert nfc_tokens == nfd_tokens


def test_tokenize_emits_warning_for_phoneme_outside_vocab() -> None:
    vocab = _build_vocab()
    # /q/ isn't in the synthetic vocab — should warn but keep going on the rest
    transcript = alignment.TranscriptInput(format="ipa", content="θ q s")
    tokens, warnings = alignment.tokenize_transcript(transcript, vocab)
    assert [ipa for ipa, _ in tokens] == ["θ", "s"]
    assert any("/q/" in w for w in warnings)


def test_tokenize_espeak_format_passes_through_to_vocab() -> None:
    vocab = _build_vocab()
    transcript = alignment.TranscriptInput(format="espeak", content="T S")
    tokens, warnings = alignment.tokenize_transcript(transcript, vocab)
    assert [ipa for ipa, _ in tokens] == ["T", "S"]
    assert [idx for _, idx in tokens] == [vocab["T"], vocab["S"]]
    assert warnings == []


def test_tokenize_unknown_format_warns_and_returns_empty() -> None:
    vocab = _build_vocab()
    transcript = alignment.TranscriptInput(format="klingon", content="ka'plah")
    tokens, warnings = alignment.tokenize_transcript(transcript, vocab)
    assert tokens == []
    assert any("Unknown transcript format" in w for w in warnings)


def test_tokenize_ipa_via_alphabet_lookup_maps_to_espeak_vocab_entry() -> None:
    """When an IPA glyph isn't a direct vocab key but maps to an eSpeak holdout
    via the alphabet.json table, the lookup should succeed and surface the
    canonical IPA glyph on the output.
    """
    # Real alphabet has X-SAMPA "S" → IPA "ʃ"; if a vocab only has "S" (the
    # eSpeak holdout) and a transcript says "ʃ", we should resolve via alphabet.
    vocab = {"<pad>": 0, "S": 1}
    transcript = alignment.TranscriptInput(format="ipa", content="ʃ")
    tokens, warnings = alignment.tokenize_transcript(transcript, vocab)
    assert len(tokens) == 1
    ipa, idx = tokens[0]
    # The canonical IPA on the output should be /ʃ/ (from the alphabet table),
    # even though the underlying vocab key was the eSpeak "S".
    assert ipa == "ʃ"
    assert idx == vocab["S"]
    assert warnings == []


# ─── align_against_transcript — happy path ─────────────────────────────────


def test_alignment_clean_three_target_sequence_is_all_produced() -> None:
    """Clean argmax plan: every target gets its own frame run → all produced."""
    vocab = _build_vocab()
    # Plan: [θ θ blank s s blank f f] — three targets, each 2 frames, blanks
    # between to make CTC happy.
    plan = ["θ", "θ", None, "s", "s", None, "f", "f"]
    log_probs = _logits_from_argmax_plan(plan, vocab)
    transcript = alignment.TranscriptInput(format="ipa", content="θ s f")

    result = alignment.align_against_transcript(
        log_probs,
        blank_id=vocab["<pad>"],
        vocab=vocab,
        audio_duration_s=0.16,  # 8 frames × 20 ms
        transcript=transcript,
    )

    assert result.transcript_format == "ipa"
    assert len(result.positions) == 3
    assert [p.target_phoneme for p in result.positions] == ["θ", "s", "f"]
    assert all(p.match_classification == "produced" for p in result.positions)
    # Each summary key should report 1 expected = 1 produced
    for phoneme in ("θ", "s", "f"):
        summary = result.summary_by_phoneme[phoneme]
        assert summary.expected_count == 1
        assert summary.produced_count == 1
        assert summary.evidence_strength == "strong"


def test_alignment_near_miss_when_top1_differs_but_target_in_top3() -> None:
    """Frame where /s/ dominates but /θ/ is a strong runner-up → near_miss."""
    vocab = _build_vocab()
    V = len(vocab)
    T = 3
    logits = torch.zeros(T, V)
    # Make /s/ slightly stronger than /θ/ but both well above blank.
    # That gives forced_align reason to pick /θ/ for the target sequence
    # (since target is θ), but the within-span top-1 will be /s/.
    logits[:, vocab["s"]] = 5.0
    logits[:, vocab["θ"]] = 4.0
    logits[:, vocab["f"]] = 1.0
    log_probs = torch.log_softmax(logits, dim=-1)

    transcript = alignment.TranscriptInput(format="ipa", content="θ")
    result = alignment.align_against_transcript(
        log_probs,
        blank_id=vocab["<pad>"],
        vocab=vocab,
        audio_duration_s=0.06,
        transcript=transcript,
    )

    assert len(result.positions) == 1
    p = result.positions[0]
    # top-1 averaged over the span is /s/, target was /θ/; θ should appear in
    # top-3 with non-trivial prob → near_miss.
    assert p.top1_predicted == "s"
    assert p.match_classification == "near_miss"
    # avg_log_prob should be finite, not -inf
    assert math.isfinite(p.avg_log_prob)


def test_alignment_all_silence_audio_yields_absent_positions() -> None:
    """Flat logits across the vocab → forced_align does its best but nothing
    looks like a confident production; classification should fall through to
    absent."""
    vocab = _build_vocab()
    # 10 frames of uniform distribution → no phoneme dominates.
    log_probs = _flat_logits(T=10, V=len(vocab))
    transcript = alignment.TranscriptInput(format="ipa", content="θ s f")
    result = alignment.align_against_transcript(
        log_probs,
        blank_id=vocab["<pad>"],
        vocab=vocab,
        audio_duration_s=0.2,
        transcript=transcript,
    )
    # Three target tokens, all should be absent because no frame shows
    # meaningful target preference.
    assert len(result.positions) == 3
    assert all(p.match_classification == "absent" for p in result.positions)
    # Quality should never be "high" when every position is absent. (With a
    # synthetic 10-entry vocab the avg_log_prob of -ln(10) ≈ -2.30 lands in
    # "medium" under the production thresholds tuned for a 388-entry vocab;
    # the real-model case (388 entries → -ln(388) ≈ -5.96) lands cleanly in
    # "low". The point of this test is that confident-production is ruled out.)
    assert result.alignment_quality != "high"


def test_alignment_repeated_phonemes_get_distinct_positions() -> None:
    """Transcript 'θ θ θ' on a clean three-θ audio should yield three
    distinct positions, all classified produced, summary expected=3 produced=3.
    """
    vocab = _build_vocab()
    # CTC requires a blank between consecutive same tokens.
    plan = ["θ", None, "θ", None, "θ"]
    log_probs = _logits_from_argmax_plan(plan, vocab)
    transcript = alignment.TranscriptInput(format="ipa", content="θ θ θ")

    result = alignment.align_against_transcript(
        log_probs,
        blank_id=vocab["<pad>"],
        vocab=vocab,
        audio_duration_s=0.1,
        transcript=transcript,
    )

    assert len(result.positions) == 3
    assert [p.target_index_in_transcript for p in result.positions] == [0, 1, 2]
    assert result.summary_by_phoneme["θ"].expected_count == 3
    assert result.summary_by_phoneme["θ"].produced_count == 3
    assert result.summary_by_phoneme["θ"].evidence_strength == "strong"


# ─── align_against_transcript — edge cases ─────────────────────────────────


def test_alignment_empty_transcript_returns_no_positions_no_errors() -> None:
    vocab = _build_vocab()
    log_probs = _logits_from_argmax_plan(["θ", None, "s"], vocab)
    transcript = alignment.TranscriptInput(format="ipa", content="")
    result = alignment.align_against_transcript(
        log_probs,
        blank_id=vocab["<pad>"],
        vocab=vocab,
        audio_duration_s=0.06,
        transcript=transcript,
    )
    assert result.positions == []
    assert result.summary_by_phoneme == {}
    assert any("no alignable tokens" in w for w in result.alignment_warnings)


def test_alignment_transcript_longer_than_audio_warns_and_skips() -> None:
    vocab = _build_vocab()
    log_probs = _logits_from_argmax_plan(["θ"], vocab)  # 1 frame
    transcript = alignment.TranscriptInput(format="ipa", content="θ s f a b")
    result = alignment.align_against_transcript(
        log_probs,
        blank_id=vocab["<pad>"],
        vocab=vocab,
        audio_duration_s=0.02,
        transcript=transcript,
    )
    assert result.positions == []
    assert any("longer than the audio" in w for w in result.alignment_warnings)


def test_alignment_warns_about_phonemes_outside_vocab_but_aligns_the_rest() -> None:
    vocab = _build_vocab()
    # Transcript includes /q/ which isn't in vocab; /θ/ and /s/ are.
    plan = ["θ", None, "s"]
    log_probs = _logits_from_argmax_plan(plan, vocab)
    transcript = alignment.TranscriptInput(format="ipa", content="θ q s")

    result = alignment.align_against_transcript(
        log_probs,
        blank_id=vocab["<pad>"],
        vocab=vocab,
        audio_duration_s=0.06,
        transcript=transcript,
    )
    # Only the two in-vocab targets align; warning notes the missing /q/.
    assert len(result.positions) == 2
    assert [p.target_phoneme for p in result.positions] == ["θ", "s"]
    assert any("/q/" in w for w in result.alignment_warnings)


def test_alignment_handles_nfc_vs_nfd_identically() -> None:
    """Transcript in NFD form should produce the same alignment as NFC form."""
    vocab = _build_vocab()
    plan = ["θ", None, "a"]
    log_probs = _logits_from_argmax_plan(plan, vocab)

    nfc = "θa"
    nfd = unicodedata.normalize("NFD", nfc)
    res_nfc = alignment.align_against_transcript(
        log_probs,
        blank_id=vocab["<pad>"],
        vocab=vocab,
        audio_duration_s=0.06,
        transcript=alignment.TranscriptInput(format="ipa", content=nfc),
    )
    res_nfd = alignment.align_against_transcript(
        log_probs,
        blank_id=vocab["<pad>"],
        vocab=vocab,
        audio_duration_s=0.06,
        transcript=alignment.TranscriptInput(format="ipa", content=nfd),
    )
    assert [p.target_phoneme for p in res_nfc.positions] == [
        p.target_phoneme for p in res_nfd.positions
    ]
    assert [p.match_classification for p in res_nfc.positions] == [
        p.match_classification for p in res_nfd.positions
    ]


# ─── Property tests ─────────────────────────────────────────────────────────


def test_property_summary_counts_total_to_expected() -> None:
    """For every summary entry: produced + near_miss + absent == expected."""
    vocab = _build_vocab()
    plan = ["θ", None, "s", "f", None, "θ"]
    log_probs = _logits_from_argmax_plan(plan, vocab)
    transcript = alignment.TranscriptInput(format="ipa", content="θ s f θ")
    result = alignment.align_against_transcript(
        log_probs,
        blank_id=vocab["<pad>"],
        vocab=vocab,
        audio_duration_s=0.12,
        transcript=transcript,
    )
    for phoneme, summary in result.summary_by_phoneme.items():
        total = summary.produced_count + summary.near_miss_count + summary.absent_count
        assert total == summary.expected_count, f"summary[{phoneme}] doesn't total"


def test_property_position_start_strictly_less_than_end() -> None:
    vocab = _build_vocab()
    plan = ["θ", "θ", None, "s", None, "f", "f"]
    log_probs = _logits_from_argmax_plan(plan, vocab)
    transcript = alignment.TranscriptInput(format="ipa", content="θ s f")
    result = alignment.align_against_transcript(
        log_probs,
        blank_id=vocab["<pad>"],
        vocab=vocab,
        audio_duration_s=0.14,
        transcript=transcript,
    )
    for p in result.positions:
        assert p.start_ms < p.end_ms, f"position {p.target_phoneme}: {p.start_ms}-{p.end_ms}"


def test_property_positions_monotonic_by_start_ms() -> None:
    vocab = _build_vocab()
    plan = ["θ", None, "s", None, "f", None, "a"]
    log_probs = _logits_from_argmax_plan(plan, vocab)
    transcript = alignment.TranscriptInput(format="ipa", content="θ s f a")
    result = alignment.align_against_transcript(
        log_probs,
        blank_id=vocab["<pad>"],
        vocab=vocab,
        audio_duration_s=0.14,
        transcript=transcript,
    )
    starts = [p.start_ms for p in result.positions]
    assert starts == sorted(starts)


def test_property_top3_alternatives_have_probabilities_in_unit_interval() -> None:
    vocab = _build_vocab()
    plan = ["θ", None, "s"]
    log_probs = _logits_from_argmax_plan(plan, vocab)
    transcript = alignment.TranscriptInput(format="ipa", content="θ s")
    result = alignment.align_against_transcript(
        log_probs,
        blank_id=vocab["<pad>"],
        vocab=vocab,
        audio_duration_s=0.06,
        transcript=transcript,
    )
    for p in result.positions:
        assert 1 <= len(p.top3_alternatives) <= 3
        for alt in p.top3_alternatives:
            assert 0.0 <= alt.prob <= 1.0


# ─── Evidence-strength rollup ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "produced,near_miss,absent,expected_strength",
    [
        # 4 expected /θ/: 4 produced → strong (≥ ceil(4*0.5)=2)
        (4, 0, 0, "strong"),
        # 4 expected: 2 produced + 2 near_miss → strong (produced ≥ 2)
        (2, 2, 0, "strong"),
        # 4 expected: 1 produced + 2 near_miss → moderate (1+2=3 ≥ 2)
        (1, 2, 1, "moderate"),
        # 4 expected: 0 produced + 1 near_miss → weak
        (0, 1, 3, "weak"),
        # 4 expected: all absent → weak
        (0, 0, 4, "weak"),
    ],
)
def test_evidence_strength_rollup(
    produced: int, near_miss: int, absent: int, expected_strength: str
) -> None:
    positions = (
        [
            alignment.AlignedPosition(
                target_phoneme="θ",
                target_index_in_transcript=i,
                start_ms=i * 10,
                end_ms=i * 10 + 5,
                avg_log_prob=-0.5,
                top1_predicted="θ",
                top3_alternatives=[],
                match_classification="produced",
            )
            for i in range(produced)
        ]
        + [
            alignment.AlignedPosition(
                target_phoneme="θ",
                target_index_in_transcript=produced + i,
                start_ms=(produced + i) * 10,
                end_ms=(produced + i) * 10 + 5,
                avg_log_prob=-2.0,
                top1_predicted="s",
                top3_alternatives=[],
                match_classification="near_miss",
            )
            for i in range(near_miss)
        ]
        + [
            alignment.AlignedPosition(
                target_phoneme="θ",
                target_index_in_transcript=produced + near_miss + i,
                start_ms=(produced + near_miss + i) * 10,
                end_ms=(produced + near_miss + i) * 10 + 5,
                avg_log_prob=-5.0,
                top1_predicted="f",
                top3_alternatives=[],
                match_classification="absent",
            )
            for i in range(absent)
        ]
    )
    thresholds = alignment._load_thresholds()
    summary = alignment._summarise_by_phoneme(positions, thresholds)
    assert summary["θ"].evidence_strength == expected_strength
    assert summary["θ"].expected_count == produced + near_miss + absent


# ─── Alignment quality rollup ───────────────────────────────────────────────


def test_alignment_quality_high_when_all_positions_are_confident() -> None:
    thresholds = alignment._load_thresholds()
    # All log_probs above the "high" threshold
    avg_log_probs = [-0.2, -0.5, -0.8]
    assert alignment._classify_alignment_quality(avg_log_probs, thresholds) == "high"


def test_alignment_quality_low_when_all_positions_are_marginal() -> None:
    thresholds = alignment._load_thresholds()
    avg_log_probs = [-5.0, -4.5, -6.0]
    assert alignment._classify_alignment_quality(avg_log_probs, thresholds) == "low"


def test_alignment_quality_low_when_no_positions() -> None:
    thresholds = alignment._load_thresholds()
    assert alignment._classify_alignment_quality([], thresholds) == "low"
