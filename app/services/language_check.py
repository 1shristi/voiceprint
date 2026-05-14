"""Marker-based language match for clips with a stated speaker language.

Each language profile is a hand-curated list of "smoking gun" phonemes:
  - `positive` markers are phonemes you'd expect to see in fluent speech of
    that language (count >= threshold = "hit").
  - `negative` markers are phonemes whose presence contradicts the claim
    (count > threshold = "violation"). Threshold > 0 absorbs the 1-token
    noise floor where Wav2Vec2-Phoneme occasionally hallucinates a phoneme
    not actually present.

This is intentionally a coarse "did the speaker speak the claimed language at
all" detector, not an accent/fluency scorer. Closely-related languages
(Spanish/Italian, Mandarin/Cantonese, Hindi/Urdu) won't be reliably
discriminable. The goal is to catch the obvious mismatch ("user said L1 is
Hindi but spoke English") before the analysis pipeline wastes Gemini cycles.

Profiles are heuristics based on contrastive phonology, NOT empirically
calibrated against Wav2Vec2-Phoneme's actual emission distribution. Variant
lists are generous to absorb label-shape drift. Tune after observing real
emissions on a calibration set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.phonemes import PhonemeInventory


# ─── Data shapes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PhonemeMarker:
    """A phoneme-or-group with a count threshold.

    For POSITIVE markers: `threshold` is the minimum count to register as a hit
    (floor of 1 enforced — count must be > 0). If `optional` is True, a missing
    marker is EXCLUDED from the positive total instead of counted as a miss —
    used for "bonus" markers that are strong evidence FOR the language when
    present but whose absence doesn't argue against it (e.g. Hindi's voiced
    aspirates /bʱ dʱ ɡʱ/, which are distinctive but low-frequency).

    For NEGATIVE markers: `threshold` is the maximum count tolerated before
    flagging a violation (i.e. up to and including `threshold` is fine).
    `optional` is ignored for negatives.
    """
    label: str
    variants: tuple[str, ...]
    threshold: int = 1
    optional: bool = False


@dataclass(frozen=True)
class LanguageProfile:
    language: str               # canonical key
    aliases: tuple[str, ...]
    positive: tuple[PhonemeMarker, ...]
    negative: tuple[PhonemeMarker, ...]


@dataclass
class MarkerResult:
    label: str
    variants: tuple[str, ...]
    count: int
    threshold: int
    passed: bool  # positive: count >= threshold; negative: count <= threshold


@dataclass
class LanguageMatchResult:
    claimed_language: str             # canonical key
    verdict: str                      # see VERDICTS below
    score: float | None               # 0..1, None if verdict is insufficient/unknown
    positive_hits: int
    positive_total: int
    negative_clean: int
    negative_total: int
    positive_results: list[MarkerResult] = field(default_factory=list)
    negative_results: list[MarkerResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


VERDICT_MATCHES = "matches"
VERDICT_UNCERTAIN = "uncertain"
VERDICT_MISMATCH = "mismatch"
VERDICT_INSUFFICIENT = "insufficient_signal"
VERDICT_UNKNOWN_LANGUAGE = "unknown_language"


# Below this token count we don't have enough data to score reliably.
MIN_TOKENS_FOR_SCORING = 20


# Score → verdict thresholds. Conservative: a clip needs to clear ≥0.75 to be
# called a match (avoiding false positives for "you didn't speak X"), and must
# fall under 0.5 for a hard mismatch. The uncertain band is wide on purpose.
SCORE_MATCHES_AT = 0.75
SCORE_MISMATCH_BELOW = 0.5

# Positive markers (presence of expected phonemes) are stronger evidence than
# clean negatives (absence of contradicting phonemes), so weight them 70/30.
# Without this, a clip with no expected markers but few contradicting markers
# (e.g. Mandarin speech claimed as English — no /θ ð ɹ/ but also none of
# English's negative markers, which are mostly tuned for Romance/Semitic
# contrasts) scores in the "uncertain" band when it's a clear mismatch.
POSITIVE_WEIGHT = 0.7
NEGATIVE_WEIGHT = 0.3


# ─── Language profiles ───────────────────────────────────────────────────────
#
# Variants per marker are intentionally generous: Wav2Vec2-Phoneme emits eSpeak
# IPA labels with inconsistent diacritic placement. When in doubt, list both
# the bare IPA and any plausible affricate/length/aspiration variant.

PROFILES: tuple[LanguageProfile, ...] = (
    LanguageProfile(
        language="english",
        aliases=("en", "en-us", "en-gb", "en_us", "en_gb"),
        positive=(
            PhonemeMarker("voiceless_dental_fricative", ("θ",), threshold=1),
            PhonemeMarker("voiced_dental_fricative", ("ð",), threshold=1),
            PhonemeMarker("alveolar_approximant", ("ɹ", "ɻ"), threshold=2),
            PhonemeMarker("schwa", ("ə",), threshold=3),
            PhonemeMarker("labio_velar_approximant", ("w",), threshold=1),
        ),
        negative=(
            PhonemeMarker("uvular_r", ("ʁ", "χ"), threshold=1),
            PhonemeMarker("voiceless_velar_fricative", ("x",), threshold=2),
            PhonemeMarker("pharyngeals", ("ħ", "ʕ"), threshold=0),
            PhonemeMarker("retroflex_stops", ("ʈ", "ɖ", "ɳ"), threshold=1),
            PhonemeMarker("palatal_nasal", ("ɲ",), threshold=1),
            PhonemeMarker("voiceless_uvular_stop", ("q",), threshold=0),
            PhonemeMarker("emphatics", ("tˤ", "dˤ", "sˤ", "ðˤ"), threshold=0),
            PhonemeMarker("palatal_lateral", ("ʎ",), threshold=1),
            PhonemeMarker("front_rounded_vowels", ("y", "ø", "œ", "yː", "øː"), threshold=1),
        ),
    ),
    LanguageProfile(
        language="spanish",
        aliases=("es", "es-es", "es-mx", "es_es", "es_mx", "castilian"),
        positive=(
            PhonemeMarker("alveolar_tap", ("ɾ",), threshold=2),
            PhonemeMarker("palatal_nasal", ("ɲ",), threshold=1),
            PhonemeMarker("voiceless_velar_fricative", ("x",), threshold=1),
            PhonemeMarker("voiced_bilabial_approximant", ("β",), threshold=1),
        ),
        negative=(
            PhonemeMarker("dental_fricatives", ("ð",), threshold=2),  # Castilian has /θ/; LA Spanish doesn't have /ð/
            PhonemeMarker("english_r", ("ɹ", "ɻ"), threshold=2),
            PhonemeMarker("uvular_r", ("ʁ", "χ"), threshold=1),
            PhonemeMarker("pharyngeals", ("ħ", "ʕ"), threshold=0),
            PhonemeMarker("retroflex_stops", ("ʈ", "ɖ"), threshold=1),
            PhonemeMarker("front_rounded_vowels", ("y", "ø", "œ"), threshold=1),
            PhonemeMarker("aspirated_stops_high", ("pʰ", "tʰ", "kʰ"), threshold=2),
        ),
    ),
    LanguageProfile(
        language="french",
        aliases=("fr", "fr-fr", "fr-ca", "fr_fr", "fr_ca"),
        positive=(
            PhonemeMarker("uvular_r", ("ʁ", "χ"), threshold=2),
            PhonemeMarker("close_front_rounded", ("y", "yː"), threshold=1),
            PhonemeMarker("mid_front_rounded", ("ø", "œ", "øː", "œ̃"), threshold=1),
            PhonemeMarker("nasal_vowels", ("ɑ̃", "ɛ̃", "ɔ̃", "œ̃", "ã", "ẽ", "õ"), threshold=1),
        ),
        negative=(
            PhonemeMarker("english_r", ("ɹ", "ɻ"), threshold=2),
            PhonemeMarker("dental_fricatives", ("θ", "ð"), threshold=1),
            PhonemeMarker("pharyngeals", ("ħ", "ʕ"), threshold=0),
            PhonemeMarker("retroflex_stops", ("ʈ", "ɖ"), threshold=1),
            PhonemeMarker("aspirated_stops_high", ("pʰ", "tʰ", "kʰ"), threshold=2),
        ),
    ),
    LanguageProfile(
        language="german",
        aliases=("de", "de-de", "de_de"),
        positive=(
            PhonemeMarker("uvular_r", ("ʁ", "χ"), threshold=2),
            PhonemeMarker("voiceless_palatal_fricative", ("ç",), threshold=1),
            PhonemeMarker("voiceless_velar_fricative", ("x",), threshold=1),
            PhonemeMarker("front_rounded_vowels", ("y", "ø", "œ", "yː", "øː"), threshold=1),
        ),
        negative=(
            PhonemeMarker("english_r", ("ɹ", "ɻ"), threshold=2),
            PhonemeMarker("dental_fricatives", ("θ", "ð"), threshold=1),
            PhonemeMarker("pharyngeals", ("ħ", "ʕ"), threshold=0),
            PhonemeMarker("retroflex_stops", ("ʈ", "ɖ"), threshold=1),
            PhonemeMarker("palatal_nasal", ("ɲ",), threshold=2),
        ),
    ),
    LanguageProfile(
        language="mandarin",
        aliases=("zh", "zh-cn", "zh_cn", "cmn", "chinese", "mandarin_chinese"),
        positive=(
            PhonemeMarker("alveolo_palatal_affricates", ("tɕ", "tɕʰ", "t͡ɕ", "t͡ɕʰ"), threshold=1),
            PhonemeMarker("alveolo_palatal_fricative", ("ɕ",), threshold=1),
            PhonemeMarker("retroflex_fricatives", ("ʂ", "ʐ"), threshold=1),
            PhonemeMarker("retroflex_affricates", ("tʂ", "tʂʰ", "t͡ʂ", "t͡ʂʰ"), threshold=1),
        ),
        negative=(
            PhonemeMarker("dental_fricatives", ("θ", "ð"), threshold=1),
            PhonemeMarker("english_r", ("ɹ", "ɻ"), threshold=2),
            PhonemeMarker("voiced_stops", ("b", "d", "ɡ"), threshold=4),  # Mandarin stops are unaspirated voiceless, not voiced
            PhonemeMarker("pharyngeals", ("ħ", "ʕ"), threshold=0),
            PhonemeMarker("uvular_r", ("ʁ", "χ"), threshold=1),
        ),
    ),
    LanguageProfile(
        language="hindi",
        aliases=("hi", "hi-in", "hi_in"),
        positive=(
            PhonemeMarker("aspirated_stops", ("pʰ", "tʰ", "kʰ"), threshold=1),
            PhonemeMarker("retroflex_stops", ("ʈ", "ɖ", "ɳ"), threshold=1),
            # Optional: distinctive but low-frequency; absence isn't evidence against Hindi.
            PhonemeMarker("voiced_aspirated", ("bʱ", "dʱ", "ɡʱ", "ɖʱ"), threshold=1, optional=True),
            PhonemeMarker("retroflex_flap", ("ɽ",), threshold=1, optional=True),
        ),
        negative=(
            PhonemeMarker("dental_fricatives", ("θ", "ð"), threshold=1),
            PhonemeMarker("english_r", ("ɹ", "ɻ"), threshold=2),
            PhonemeMarker("uvular_r", ("ʁ", "χ"), threshold=1),
            PhonemeMarker("pharyngeals", ("ħ", "ʕ"), threshold=0),
            PhonemeMarker("front_rounded_vowels", ("y", "ø", "œ"), threshold=1),
        ),
    ),
    LanguageProfile(
        language="arabic",
        aliases=("ar", "msa", "arabic_msa"),
        positive=(
            PhonemeMarker("pharyngeals", ("ħ", "ʕ"), threshold=1),
            PhonemeMarker("voiceless_velar_fricative", ("x",), threshold=1),
            PhonemeMarker("voiceless_uvular_stop", ("q",), threshold=1),
            PhonemeMarker("emphatics", ("tˤ", "dˤ", "sˤ", "ðˤ"), threshold=1),
        ),
        negative=(
            PhonemeMarker("dental_fricatives", ("θ", "ð"), threshold=2),  # MSA actually has θ/ð in some words, so weaker negative
            PhonemeMarker("english_r", ("ɹ", "ɻ"), threshold=2),
            PhonemeMarker("retroflex_stops", ("ʈ", "ɖ"), threshold=1),
            PhonemeMarker("palatal_nasal", ("ɲ",), threshold=1),
            PhonemeMarker("front_rounded_vowels", ("y", "ø", "œ"), threshold=1),
            PhonemeMarker("aspirated_stops_high", ("pʰ", "tʰ", "kʰ"), threshold=2),
        ),
    ),
    LanguageProfile(
        language="japanese",
        aliases=("ja", "ja-jp", "ja_jp"),
        positive=(
            # /ɾ/ is the r-row consonant in Japanese — every ra/ri/ru/re/ro
            # syllable. The Wav2Vec2 model also emits /ɾ/ for American English
            # medial flaps ("city", "butter"), so threshold=1 false-positives
            # on English. A 12s Japanese clip should have many /ɾ/ (r-row
            # syllables are very frequent), so threshold=4 separates cleanly.
            PhonemeMarker("alveolar_tap", ("ɾ",), threshold=4),
            PhonemeMarker("alveolo_palatal_fricative", ("ɕ",), threshold=1),
            # Optional: ç appears in palatalized /h/ before /i/, not in every clip.
            PhonemeMarker("voiceless_palatal_fricative", ("ç",), threshold=1, optional=True),
        ),
        negative=(
            PhonemeMarker("alveolar_lateral", ("l",), threshold=2),
            PhonemeMarker("english_r", ("ɹ", "ɻ"), threshold=2),
            PhonemeMarker("dental_fricatives", ("θ", "ð"), threshold=1),
            PhonemeMarker("uvular_r", ("ʁ", "χ"), threshold=1),
            PhonemeMarker("pharyngeals", ("ħ", "ʕ"), threshold=0),
            PhonemeMarker("aspirated_stops_high", ("pʰ", "tʰ", "kʰ"), threshold=2),
            PhonemeMarker("retroflex_stops", ("ʈ", "ɖ"), threshold=1),
            PhonemeMarker("v_fricative", ("v",), threshold=2),
        ),
    ),
    LanguageProfile(
        language="portuguese",
        aliases=("pt", "pt-br", "pt-pt", "pt_br", "pt_pt"),
        positive=(
            PhonemeMarker("uvular_r", ("ʁ", "χ", "ʀ"), threshold=1),
            PhonemeMarker("palatal_nasal", ("ɲ",), threshold=1),
            PhonemeMarker("palatal_lateral", ("ʎ",), threshold=1),
            PhonemeMarker("nasal_vowels", ("ɐ̃", "ẽ", "ĩ", "õ", "ũ"), threshold=1),
        ),
        negative=(
            PhonemeMarker("english_r", ("ɹ", "ɻ"), threshold=2),
            PhonemeMarker("dental_fricatives", ("θ", "ð"), threshold=1),
            PhonemeMarker("pharyngeals", ("ħ", "ʕ"), threshold=0),
            PhonemeMarker("retroflex_stops", ("ʈ", "ɖ"), threshold=1),
            PhonemeMarker("front_rounded_vowels", ("y", "ø", "œ"), threshold=1),
        ),
    ),
    LanguageProfile(
        language="italian",
        aliases=("it", "it-it", "it_it"),
        positive=(
            # Optional: ʎ is distinctive but only appears in "gli"-type clusters.
            PhonemeMarker("palatal_lateral", ("ʎ",), threshold=1, optional=True),
            PhonemeMarker("palatal_nasal", ("ɲ",), threshold=1),
            PhonemeMarker("alveolar_affricates", ("ts", "dz", "t͡s", "d͡z"), threshold=1),
            # Italian /ɾ/ and /r/ (trill) are frequent in any clip with multi-r
            # words. The model emits /ɾ/ for English medial flaps too, so
            # threshold=4 keeps the marker discriminative without false-positive
            # on English flap-r counts of 1-3.
            PhonemeMarker("alveolar_tap", ("ɾ", "r"), threshold=4),
        ),
        negative=(
            PhonemeMarker("english_r", ("ɹ", "ɻ"), threshold=2),
            PhonemeMarker("dental_fricatives", ("θ", "ð"), threshold=1),
            PhonemeMarker("uvular_r", ("ʁ", "χ"), threshold=1),
            PhonemeMarker("pharyngeals", ("ħ", "ʕ"), threshold=0),
            PhonemeMarker("retroflex_stops", ("ʈ", "ɖ"), threshold=1),
            PhonemeMarker("front_rounded_vowels", ("y", "ø", "œ"), threshold=1),
        ),
    ),
    LanguageProfile(
        language="russian",
        aliases=("ru", "ru-ru", "ru_ru"),
        positive=(
            PhonemeMarker("voiceless_velar_fricative", ("x",), threshold=1),
            PhonemeMarker("retroflex_fricatives", ("ʂ", "ʐ"), threshold=1),
            PhonemeMarker("palatalized_consonants", ("tʲ", "dʲ", "nʲ", "sʲ", "lʲ", "rʲ"), threshold=1),
        ),
        negative=(
            PhonemeMarker("english_r", ("ɹ", "ɻ"), threshold=2),
            PhonemeMarker("dental_fricatives", ("θ", "ð"), threshold=1),
            PhonemeMarker("pharyngeals", ("ħ", "ʕ"), threshold=0),
            PhonemeMarker("uvular_r", ("ʁ", "χ"), threshold=1),
            PhonemeMarker("front_rounded_vowels", ("y", "ø", "œ"), threshold=1),
            PhonemeMarker("retroflex_stops", ("ʈ", "ɖ"), threshold=1),
        ),
    ),
    LanguageProfile(
        language="korean",
        aliases=("ko", "ko-kr", "ko_kr"),
        positive=(
            PhonemeMarker("aspirated_stops", ("pʰ", "tʰ", "kʰ"), threshold=1),
            # Optional: ɕ only appears before /i/ /y/; not in every clip.
            PhonemeMarker("alveolo_palatal_fricative", ("ɕ",), threshold=1, optional=True),
            # NOTE: alveolar_tap_or_lateral (/ɾ l/) was removed — English /l/ is
            # so frequent that no threshold reliably separates Korean from
            # English-claimed-as-Korean. Korean is now a thin profile (1
            # required + 1 optional); short clips may land UNCERTAIN. If false
            # negatives on real Korean become a problem, swap to Whisper-based
            # LID rather than reintroducing this marker.
        ),
        negative=(
            PhonemeMarker("english_r", ("ɹ", "ɻ"), threshold=2),
            PhonemeMarker("dental_fricatives", ("θ", "ð"), threshold=1),
            PhonemeMarker("v_fricative", ("v",), threshold=2),
            PhonemeMarker("f_fricative", ("f",), threshold=2),
            PhonemeMarker("uvular_r", ("ʁ", "χ"), threshold=1),
            PhonemeMarker("pharyngeals", ("ħ", "ʕ"), threshold=0),
            PhonemeMarker("retroflex_stops", ("ʈ", "ɖ"), threshold=1),
        ),
    ),
)


_PROFILE_INDEX: dict[str, LanguageProfile] = {}
for _p in PROFILES:
    _PROFILE_INDEX[_p.language] = _p
    for _alias in _p.aliases:
        _PROFILE_INDEX[_alias] = _p


def _resolve_profile(claimed_language: str | None) -> LanguageProfile | None:
    """Resolve a free-form language string to a configured profile.

    Tries the input verbatim (lowercased), then the first word, then the last
    word, then the input with parenthetical content stripped. Handles common
    user-typed forms like "Mandarin Chinese", "Brazilian Portuguese", and
    "Hindi (India)" without enumerating every alias.
    """
    if not claimed_language:
        return None
    key = claimed_language.strip().lower()
    if profile := _PROFILE_INDEX.get(key):
        return profile

    # Strip parenthetical: "hindi (india)" → "hindi"
    if "(" in key:
        stripped = key.split("(", 1)[0].strip()
        if profile := _PROFILE_INDEX.get(stripped):
            return profile

    parts = key.replace("-", " ").replace("_", " ").split()
    if not parts:
        return None
    # First word: "mandarin chinese" → "mandarin"
    if profile := _PROFILE_INDEX.get(parts[0]):
        return profile
    # Last word: "brazilian portuguese" → "portuguese"
    if profile := _PROFILE_INDEX.get(parts[-1]):
        return profile
    return None


def supported_languages() -> list[str]:
    """Canonical names of all configured language profiles."""
    return [p.language for p in PROFILES]


# ─── Scoring ─────────────────────────────────────────────────────────────────


def _check_positive(counts: dict[str, int], m: PhonemeMarker) -> MarkerResult:
    # Positives need at least one occurrence to be evidence — threshold=0 was a
    # historical footgun that made any marker trivially "pass" with count=0.
    threshold = max(1, m.threshold)
    count = sum(counts.get(v, 0) for v in m.variants)
    return MarkerResult(
        label=m.label,
        variants=m.variants,
        count=count,
        threshold=threshold,
        passed=count >= threshold,
    )


def _check_negative(counts: dict[str, int], m: PhonemeMarker) -> MarkerResult:
    count = sum(counts.get(v, 0) for v in m.variants)
    # Negative marker "passes" (i.e. clean) when count is at or below threshold.
    return MarkerResult(
        label=m.label,
        variants=m.variants,
        count=count,
        threshold=m.threshold,
        passed=count <= m.threshold,
    )


def score_language_match(
    inventory: "PhonemeInventory",
    claimed_language: str | None,
) -> LanguageMatchResult | None:
    """Score whether the clip's phoneme inventory looks like the claimed language.

    Returns None when `claimed_language` is empty (caller didn't ask). Returns
    a result with verdict='unknown_language' when the language isn't in our
    profile set, or 'insufficient_signal' when the clip has too few tokens to
    score meaningfully.
    """
    if not claimed_language:
        return None

    profile = _resolve_profile(claimed_language)
    if profile is None:
        return LanguageMatchResult(
            claimed_language=claimed_language.strip().lower(),
            verdict=VERDICT_UNKNOWN_LANGUAGE,
            score=None,
            positive_hits=0,
            positive_total=0,
            negative_clean=0,
            negative_total=0,
            notes=[f"no language profile configured for {claimed_language!r}"],
        )

    total = inventory.total_tokens
    if total < MIN_TOKENS_FOR_SCORING:
        return LanguageMatchResult(
            claimed_language=profile.language,
            verdict=VERDICT_INSUFFICIENT,
            score=None,
            positive_hits=0,
            positive_total=len(profile.positive),
            negative_clean=0,
            negative_total=len(profile.negative),
            notes=[f"clip has {total} phoneme tokens; need ≥{MIN_TOKENS_FOR_SCORING} to score"],
        )

    counts = inventory.counts
    positive_results = [_check_positive(counts, m) for m in profile.positive]
    negative_results = [_check_negative(counts, m) for m in profile.negative]

    # Optional positives that miss are excluded from the denominator (their
    # absence is not evidence against the language). Optional positives that
    # hit count as a regular hit — bonus evidence FOR the language.
    positive_hits = 0
    positive_denominator = 0
    for marker, result in zip(profile.positive, positive_results, strict=True):
        if result.passed:
            positive_hits += 1
            positive_denominator += 1
        elif not marker.optional:
            positive_denominator += 1

    negative_clean = sum(1 for r in negative_results if r.passed)
    positive_score = positive_hits / positive_denominator if positive_denominator else 1.0
    negative_score = negative_clean / len(negative_results) if negative_results else 1.0
    score = POSITIVE_WEIGHT * positive_score + NEGATIVE_WEIGHT * negative_score

    if score >= SCORE_MATCHES_AT:
        verdict = VERDICT_MATCHES
    elif score < SCORE_MISMATCH_BELOW:
        verdict = VERDICT_MISMATCH
    else:
        verdict = VERDICT_UNCERTAIN

    notes: list[str] = []
    # Only list REQUIRED missing positives in the notes — optional+missed are
    # by design excluded from the score, so listing them as "expected but
    # missing" misrepresents the math (the visible count wouldn't match the
    # denominator). Bonus markers that hit DO appear in positive_results for
    # the caller to inspect.
    missed_positives = [
        m.label
        for m, r in zip(profile.positive, positive_results, strict=True)
        if not r.passed and not m.optional
    ]
    violated_negatives = [r.label for r in negative_results if not r.passed]
    if missed_positives:
        notes.append("expected but missing: " + ", ".join(missed_positives[:5]))
    if violated_negatives:
        notes.append("present but unexpected: " + ", ".join(violated_negatives[:5]))

    return LanguageMatchResult(
        claimed_language=profile.language,
        verdict=verdict,
        score=score,
        positive_hits=positive_hits,
        positive_total=positive_denominator,
        negative_clean=negative_clean,
        negative_total=len(profile.negative),
        positive_results=positive_results,
        negative_results=negative_results,
        notes=notes,
    )
