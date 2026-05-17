"""Notation-alignment audit for Wav2Vec2-Phoneme outputs.

Phase 0 of loop3a-voiceprint-spec. Establishes ground truth for which symbols
the model (`facebook/wav2vec2-lv-60-espeak-cv-ft`) can emit and how each maps
to canonical IPA. Every subsequent phase of Loop 3 depends on this mapping
being correct.

This script has three subcommands:

  audit       Regenerate app/data/phoneme_alphabet.json from the model's
              tokenizer vocabulary. The vocabulary is a strict superset of
              anything the model can ever emit, so vocab-derived enumeration
              is more reproducible than empirical fixture-based audit and is
              guaranteed to cover every symbol the production pipeline can
              see.

  catalogue   Regenerate app/data/notation_drift_catalogue.md by parsing the
              committed L1_PHONEMES snapshot and comparing each phoneme to
              the symbols the model would emit. Categorises mismatches as
              notation-only vs phonological.

  check       Run audit + catalogue against in-memory copies and exit
              non-zero if either differs from the committed files. Used in
              CI as a documentation-update reminder (non-blocking).

  fixtures    Optional empirical pass: load each audio file in a given
              directory through the existing phoneme detector and record
              which symbols were observed. Output goes to
              app/data/_audit_observations.json. Used to spot symbols that
              are in the vocab but never emitted on the kinds of audio we
              actually process, or — more importantly — to flag symbols the
              model emits but that the alphabet hasn't been mapped for.

Usage:

    python scripts/notation_audit.py audit
    python scripts/notation_audit.py catalogue
    python scripts/notation_audit.py check
    python scripts/notation_audit.py fixtures path/to/audio_dir
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

_MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
_ALPHABET_VERSION = "1.0.0"
_AUDITED_AT = "2026-05-16"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "app" / "data"
_ALPHABET_PATH = _DATA_DIR / "phoneme_alphabet.json"
_CATALOGUE_PATH = _DATA_DIR / "notation_drift_catalogue.md"
_SNAPSHOT_PATH = _DATA_DIR / "_l1_phonemes_snapshot.txt"
_OBSERVATIONS_PATH = _DATA_DIR / "_audit_observations.json"

# Tokens emitted by the tokenizer that aren't phonemes. Mirrors the set in
# app/services/phonemes.py so the audit and the runtime agree on what to drop.
_NON_PHONEME_TOKENS = {"<pad>", "<s>", "</s>", "<unk>", "|", "[PAD]", "[UNK]", " ", ""}


# ---------------------------------------------------------------------------
# eSpeak / X-SAMPA holdouts.
#
# The lv-60-espeak-cv-ft tokenizer is largely IPA-native, but a handful of
# entries are X-SAMPA-style holdouts from eSpeak's text output. They have to
# be hand-translated to IPA; this is the only place in Phase 0 where mapping
# is non-trivial.
#
# Source: eSpeak phoneme inventory documentation
#         https://espeak.sourceforge.net/phonemes.html
# Notes:  d^ / t^ / s^ are eSpeak's caret-marked retroflex letters; the
#         vocabulary also contains the bare IPA retroflexes (ɖ, ʈ, ʂ) so
#         either form may appear. ts. / s. / r. / i. are eSpeak's
#         period-marked Mandarin retroflex/syllabic variants. Aspirated
#         clusters of the form Xh (ph, th, kh, tsh, tɕh, ts.h) are eSpeak's
#         "consonant + h" rendering and are equivalent to IPA Xʰ.
# ---------------------------------------------------------------------------
_XSAMPA_TO_IPA: dict[str, str] = {
    "N": "ŋ",
    "S": "ʃ",
    "X": "x",
    "tS": "tʃ",
    "dZ": "dʒ",
    "d[": "d̪",
    "t[": "t̪",
    "d^": "ɖ",
    "t^": "ʈ",
    "s^": "ʂ",
    "t^ː": "ʈː",
    "sx": "sx",  # consonant cluster (Dutch <sch>), not a single phoneme
    "ph": "pʰ",
    "th": "tʰ",
    "kh": "kʰ",
    "tsh": "tsʰ",
    "tɕh": "tɕʰ",
    "ts.h": "tʂʰ",
    "ts.": "tʂ",
    "s.": "ʂ",
    "r.": "ɻ",
    "i.": "ɨ",
    "i.ː": "ɨː",
    "i.1": "ɨ˥",
    "i.2": "ɨ˧˥",
    "i.4": "ɨ˥˩",
    "i.5": "ɨ",
    "i.ɜ": "ɨ˨˩˦",
    "u.": "ɯ",
    "u.ː": "ɯː",
    "a.": "ɑ",
    "a.ː": "ɑː",
    'u"': "y",  # eSpeak <u"> = front rounded /y/; alternative for the bare 'y' entry
    "??": "ʔ",  # eSpeak placeholder; best guess is glottal stop, marked uncertain
    # Bare tone digit (Mandarin tone 1, high-level)
    "1": "˥",
    # Diacritic-only modifier
    "ʲ": "ʲ",
}

# Symbols whose mapping is genuinely uncertain and should be flagged for
# review. Keys must be members of _XSAMPA_TO_IPA so the alphabet entry can
# advertise its caveat. Phase 0 is allowed to "report it; do not invent
# behaviour" (§2 of the spec), so each uncertain mapping is recorded
# explicitly rather than hidden.
_UNCERTAIN_MAPPINGS: set[str] = {
    "??",   # glottal stop is the most likely intent but unconfirmed
    "1",    # bare tone digit; Mandarin tone 1 marker is the most likely intent
    'u"',   # eSpeak's <u"> convention; assumed = /y/
    "sx",   # cluster not a single phoneme
    "d^",   # bare caret form; co-occurs with explicit ɖ in vocab
    "t^",   # bare caret form; co-occurs with explicit ʈ in vocab
    "s^",   # bare caret form; co-occurs with explicit ʂ in vocab
    "t^ː",  # bare caret form long; same as ʈː
    "ʲ",    # standalone palatalisation diacritic
}

# Combining-diacritic descriptions (IPA + a few eSpeak conventions).
_DIACRITICS: dict[str, str] = {
    "ʰ": "aspirated",
    "ʲ": "palatalised",
    "ː": "long",
    "ˤ": "pharyngealised",
    "ʷ": "labialised",
    "ʼ": "ejective",
    "̪": "dental",
    "̃": "nasalised",
    "̩": "syllabic",
    "̞": "lowered",
    "̝": "raised",
    "̥": "voiceless",
    "ᵝ": "compressed lips",
    "̊": "voiceless",
}

# Base-IPA description map. Compositions (diphthongs, affricates, length)
# are built on top of these in `describe()`.
_BASE_DESCRIPTIONS: dict[str, str] = {
    "p": "Voiceless bilabial stop",
    "b": "Voiced bilabial stop",
    "t": "Voiceless alveolar stop",
    "d": "Voiced alveolar stop",
    "ʈ": "Voiceless retroflex stop",
    "ɖ": "Voiced retroflex stop",
    "c": "Voiceless palatal stop",
    "ɟ": "Voiced palatal stop",
    "k": "Voiceless velar stop",
    "ɡ": "Voiced velar stop",
    "q": "Voiceless uvular stop",
    "ʔ": "Glottal stop",
    "m": "Bilabial nasal",
    "n": "Alveolar nasal",
    "ɳ": "Retroflex nasal",
    "ɲ": "Palatal nasal",
    "ŋ": "Velar nasal",
    "ɴ": "Uvular nasal",
    "f": "Voiceless labiodental fricative",
    "v": "Voiced labiodental fricative",
    "θ": "Voiceless dental fricative",
    "ð": "Voiced dental fricative",
    "s": "Voiceless alveolar fricative",
    "z": "Voiced alveolar fricative",
    "ʃ": "Voiceless postalveolar fricative",
    "ʒ": "Voiced postalveolar fricative",
    "ʂ": "Voiceless retroflex fricative",
    "ʐ": "Voiced retroflex fricative",
    "ɕ": "Voiceless alveolo-palatal fricative",
    "ʑ": "Voiced alveolo-palatal fricative",
    "ç": "Voiceless palatal fricative",
    "ʝ": "Voiced palatal fricative",
    "x": "Voiceless velar fricative",
    "ɣ": "Voiced velar fricative",
    "χ": "Voiceless uvular fricative",
    "ʁ": "Voiced uvular fricative",
    "ħ": "Voiceless pharyngeal fricative",
    "ʕ": "Voiced pharyngeal fricative",
    "h": "Voiceless glottal fricative",
    "ɸ": "Voiceless bilabial fricative",
    "β": "Voiced bilabial fricative",
    "ɬ": "Voiceless alveolar lateral fricative",
    "ʋ": "Labiodental approximant",
    "ɹ": "Alveolar approximant",
    "ɻ": "Retroflex approximant",
    "j": "Palatal approximant",
    "w": "Labio-velar approximant",
    "l": "Alveolar lateral",
    "ɫ": "Velarised alveolar lateral",
    "ɭ": "Retroflex lateral",
    "ʎ": "Palatal lateral",
    "ɾ": "Alveolar tap",
    "ɽ": "Retroflex flap",
    "r": "Alveolar trill",
    "i": "Close front unrounded vowel",
    "y": "Close front rounded vowel",
    "ɨ": "Close central unrounded vowel",
    "ʉ": "Close central rounded vowel",
    "ɯ": "Close back unrounded vowel",
    "u": "Close back rounded vowel",
    "ɪ": "Near-close near-front unrounded vowel",
    "ʊ": "Near-close near-back rounded vowel",
    "e": "Close-mid front unrounded vowel",
    "ø": "Close-mid front rounded vowel",
    "ɵ": "Close-mid central rounded vowel",
    "ɘ": "Close-mid central unrounded vowel",
    "o": "Close-mid back rounded vowel",
    "ə": "Mid central vowel (schwa)",
    "ɚ": "Rhotacised schwa",
    "ɛ": "Open-mid front unrounded vowel",
    "œ": "Open-mid front rounded vowel",
    "ɜ": "Open-mid central unrounded vowel",
    "ɞ": "Open-mid central rounded vowel",
    "ʌ": "Open-mid back unrounded vowel",
    "ɔ": "Open-mid back rounded vowel",
    "æ": "Near-open front unrounded vowel",
    "ɐ": "Near-open central vowel",
    "a": "Open front unrounded vowel",
    "ɑ": "Open back unrounded vowel",
    "ɒ": "Open back rounded vowel",
    "ä": "Open central unrounded vowel (eSpeak rendering)",
    "ᵻ": "Near-close central unrounded vowel (Wells)",
    "˥": "High level tone",
}

# Mandarin tone-digit interpretation (eSpeak appends a digit after a vowel to
# mark tone). We don't try to translate digit chains into full IPA tone
# contours — just label them for readers.
_TONE_DIGITS: dict[str, str] = {
    "1": "tone 1 (high level)",
    "2": "tone 2 (rising)",
    "3": "tone 3 (low/dipping)",
    "4": "tone 4 (falling)",
    "5": "tone 5 (neutral)",
    "ɜ": "tone 3 (eSpeak ɜ marker)",
}


def _looks_like_xsampa(symbol: str) -> bool:
    """eSpeak holdouts are the keys of the X-SAMPA table or contain ASCII []^ markers."""
    if symbol in _XSAMPA_TO_IPA:
        return True
    return any(ch in symbol for ch in "[]^")


def _strip_tone_digits(symbol: str) -> tuple[str, list[str]]:
    """Return (symbol_without_trailing_tone_markers, tone_descriptions)."""
    notes: list[str] = []
    body = symbol
    while body and body[-1] in _TONE_DIGITS:
        notes.append(_TONE_DIGITS[body[-1]])
        body = body[:-1]
    return body, list(reversed(notes))


def _diacritic_notes(symbol: str) -> list[str]:
    notes: list[str] = []
    for ch in symbol:
        if ch in _DIACRITICS:
            notes.append(_DIACRITICS[ch])
    # de-dup while preserving order
    seen = set()
    out: list[str] = []
    for n in notes:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _to_ipa(symbol: str) -> str:
    """Map a raw vocab symbol to its canonical IPA form."""
    if symbol in _XSAMPA_TO_IPA:
        return _XSAMPA_TO_IPA[symbol]
    # Apply substring substitutions for compound X-SAMPA tokens, longest-first.
    out = symbol
    for src in sorted(_XSAMPA_TO_IPA, key=len, reverse=True):
        if src in out:
            out = out.replace(src, _XSAMPA_TO_IPA[src])
    # Normalise stray ASCII colon to IPA length marker.
    if ":" in out:
        out = out.replace(":", "ː")
    return out


_AFFRICATES = {"tʃ", "dʒ", "tɕ", "dʑ", "tʂ", "dʐ", "ts", "dz", "pf"}

# Phonetic notation equivalences the model emits as one spelling but is
# commonly written elsewhere as another. The catalogue and any consumer
# normalising on `ipa_alternates` should treat these as the same phoneme.
#
# Notable case: Mandarin /ʈʂ/ (the "zh-" retroflex affricate) is normatively
# written with retroflex /ʈ/ but eSpeak (and therefore this model) emits the
# alveolar-t variant /tʂ/. They denote the same phoneme. linguamatch's
# L1_PHONEMES uses `/ʈʂ/`; without this equivalence the catalogue would mark
# it as a phonological gap, which is misleading.
_KNOWN_EQUIVALENCES: dict[str, list[str]] = {
    "tʂ": ["ʈʂ"],
    "dʐ": ["ɖʐ"],
    "tʂʰ": ["ʈʂʰ"],
}


def _ipa_alternates(ipa: str) -> list[str]:
    out: list[str] = [f"/{ipa}/"]
    if ipa in _AFFRICATES:
        tied = ipa[0] + "͡" + ipa[1:]
        out.append(tied)
        out.append(f"/{tied}/")
    # Affricate-prefixed sequences (e.g. tʃʰ, dʑʲ) also benefit from a tie-bar
    # variant on the affricate prefix.
    elif len(ipa) >= 2 and ipa[:2] in _AFFRICATES:
        tied = ipa[0] + "͡" + ipa[1:]
        out.append(tied)

    for alt in _KNOWN_EQUIVALENCES.get(ipa, []):
        out.append(alt)
        out.append(f"/{alt}/")

    # de-dup
    seen: set[str] = set()
    deduped: list[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def _describe(symbol: str, ipa: str) -> str:
    body, tone_notes = _strip_tone_digits(ipa)
    diacritics = _diacritic_notes(body)
    # Strip diacritics to find the base sequence.
    stripped = "".join(ch for ch in body if ch not in _DIACRITICS and ch != "͡")
    # NFC normalise so combining marks recompose where possible (gives e.g. ã).
    stripped = unicodedata.normalize("NFC", stripped)

    # Compound sequences: diphthongs and affricates.
    if ipa in _AFFRICATES:
        base_desc = "Affricate"
    elif len(stripped) >= 2 and all(ch in _BASE_DESCRIPTIONS for ch in stripped) and \
            all(_BASE_DESCRIPTIONS[ch].endswith("vowel") for ch in stripped):
        base_desc = "Diphthong/triphthong (" + " + ".join(stripped) + ")"
    elif stripped and stripped[0] in _BASE_DESCRIPTIONS:
        base_desc = _BASE_DESCRIPTIONS[stripped[0]]
        if len(stripped) > 1:
            extras = [ch for ch in stripped[1:] if ch in _BASE_DESCRIPTIONS]
            if extras:
                base_desc += " + " + " + ".join(extras)
    elif symbol == "??":
        base_desc = "Glottal stop (assumed; eSpeak placeholder, unconfirmed)"
    elif symbol == "1":
        base_desc = "Bare Mandarin tone-1 marker"
    elif symbol == "ʲ":
        base_desc = "Palatalisation diacritic"
    else:
        base_desc = f"Uncategorised symbol ({ipa!r})"

    parts = [base_desc]
    if diacritics:
        parts.append("(" + ", ".join(diacritics) + ")")
    if tone_notes:
        parts.append("(" + ", ".join(tone_notes) + ")")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Vocabulary loading.
#
# The script must work in three environments:
#   1. Local dev: transformers is installed and the HF cache may already have
#      the tokenizer.
#   2. CI: transformers is installed but the cache is empty. AutoTokenizer
#      will fetch tokenizer files (a few KB) but not the model weights.
#   3. Offline / no transformers: fall back to fetching vocab.json directly
#      from huggingface.co or reading the cached file from disk.
# ---------------------------------------------------------------------------


def _load_vocab() -> dict[str, int]:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
        return dict(tok.get_vocab())
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(
            f"transformers AutoTokenizer load failed ({e!r}); falling back to vocab.json\n"
        )

    # Direct fetch fallback.
    url = f"https://huggingface.co/{_MODEL_NAME}/resolve/main/vocab.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"urllib fetch of vocab.json failed: {e!r}\n")

    # Last-ditch: local cache (developer machines).
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    matches = sorted(cache_root.glob(f"models--{_MODEL_NAME.replace('/', '--')}/snapshots/*/vocab.json"))
    if matches:
        return json.loads(matches[-1].read_text(encoding="utf-8"))

    raise RuntimeError(
        "Could not load tokenizer vocab from transformers, network, or local cache."
    )


# ---------------------------------------------------------------------------
# Alphabet generation.
# ---------------------------------------------------------------------------


def build_alphabet(vocab: dict[str, int]) -> dict[str, Any]:
    """Return the phoneme_alphabet.json payload as a dict, deterministically ordered."""
    symbols = [s for s in vocab if s not in _NON_PHONEME_TOKENS]
    symbols.sort()

    alphabet_entries: list[dict[str, Any]] = []
    for sym in symbols:
        ipa = _to_ipa(sym)
        entry: dict[str, Any] = {
            "espeak_symbol": sym,
            "ipa": ipa,
            "ipa_alternates": _ipa_alternates(ipa),
            "description": _describe(sym, ipa),
        }
        if sym in _UNCERTAIN_MAPPINGS:
            entry["uncertain"] = True
            entry["note"] = (
                "Mapping is the most likely interpretation but is not empirically "
                "confirmed against the model's emissions. Flagged for human review."
            )
        if _looks_like_xsampa(sym) and sym != ipa:
            entry["espeak_holdout"] = True
        alphabet_entries.append(entry)

    return {
        "_source": (
            "Vocabulary-derived audit of "
            f"{_MODEL_NAME}. Every entry in the model's tokenizer vocabulary is "
            "enumerated; X-SAMPA holdouts are translated to IPA against the eSpeak "
            "phoneme inventory documentation."
        ),
        "_audited_at": _AUDITED_AT,
        "_audit_script": "scripts/notation_audit.py audit",
        "_provenance_status": "empirical_with_caveats",
        "_caveats": [
            "Vocabulary enumeration is a strict superset of what the model emits; "
            "symbols never observed on real audio are still listed so consumers "
            "have complete coverage.",
            "Empirical fixture-based validation has not yet been run on a multilingual "
            "corpus; the `fixtures` subcommand exists for that follow-up.",
            "eSpeak->IPA mapping for X-SAMPA holdouts (N, S, X, tS, dZ, d[, t[, "
            "d^, t^, s^, sx, ph, th, kh, ts.h, etc.) follows the eSpeak phoneme "
            "inventory documentation; ambiguous mappings are flagged with the "
            "'uncertain' field.",
            "Tone digits (1, 2, 4, 5) appended to vowels and the ɜ marker represent "
            "Mandarin tone categories; descriptions name the tone class but the IPA "
            "field uses Chao tone-letter approximations only for the bare tone-1 case.",
        ],
        "version": _ALPHABET_VERSION,
        "model": _MODEL_NAME,
        "alphabet": alphabet_entries,
    }


def write_alphabet(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Drift catalogue.
# ---------------------------------------------------------------------------


_L1_LINE_RE = re.compile(r"^\s*([a-z_]+)\s*:\s*\[(.*)\]\s*,?\s*(?://.*)?$")
_PHONEME_TOKEN_RE = re.compile(r'"(/[^"]+/)"|"([^"]+)"')


def parse_l1_phonemes(snapshot_path: Path) -> dict[str, list[str]]:
    """Parse the committed L1_PHONEMES snapshot.

    TypeScript object literals allow duplicate keys; the later occurrence wins
    at runtime. We honour the same semantics: later occurrences overwrite
    earlier ones.
    """
    result: dict[str, list[str]] = {}
    in_block = False
    for line in snapshot_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("const L1_PHONEMES"):
            in_block = True
            continue
        if not in_block:
            continue
        if stripped == "};":
            break
        m = _L1_LINE_RE.match(line)
        if not m:
            continue
        key = m.group(1)
        body = m.group(2).strip()
        tokens: list[str] = []
        for tm in _PHONEME_TOKEN_RE.finditer(body):
            tok = tm.group(1) or tm.group(2)
            tokens.append(tok)
        result[key] = tokens  # later occurrence wins
    return result


def _strip_slashes(token: str) -> str:
    return token.strip("/")


def _model_emits_for(ipa: str, alphabet: dict[str, Any]) -> list[str]:
    """Return the list of vocab symbols whose IPA matches the requested phoneme."""
    out: list[str] = []
    for entry in alphabet["alphabet"]:
        if entry["ipa"] == ipa or ipa in entry.get("ipa_alternates", []):
            out.append(entry["espeak_symbol"])
        elif f"/{entry['ipa']}/" == f"/{ipa}/":
            out.append(entry["espeak_symbol"])
    return out


def build_drift_catalogue(l1: dict[str, list[str]], alphabet: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Notation drift catalogue\n")
    lines.append(
        "Comparison of every phoneme in linguamatch's `L1_PHONEMES` table against "
        "the symbols `facebook/wav2vec2-lv-60-espeak-cv-ft` emits.\n"
    )
    lines.append(f"- Snapshot: `app/data/_l1_phonemes_snapshot.txt`\n")
    lines.append(f"- Alphabet: `app/data/phoneme_alphabet.json` (version {_ALPHABET_VERSION})\n")
    lines.append(f"- Generated: {_AUDITED_AT} by `scripts/notation_audit.py catalogue`\n")
    lines.append("\n## Methodology\n")
    lines.append(
        "For each L1 phoneme token (e.g. `/θ/`, `/pʰ/`), we strip the surrounding "
        "slashes and look up matching vocabulary entries in `phoneme_alphabet.json`. "
        "Each match is reported alongside its eSpeak vocab symbol so consumers can "
        "see what the model would actually emit. Mismatches are classified:\n\n"
    )
    lines.append(
        "- **notation-only** — the phoneme is in the model's vocabulary but under a "
        "different surface form (e.g. linguamatch wants `/tʃ/`, model emits `tS`).\n"
    )
    lines.append(
        "- **phonological** — no vocabulary entry covers this phoneme; the model "
        "cannot emit it and downstream consumers should not expect to see it.\n"
    )
    lines.append(
        "- **direct** — the phoneme matches a vocab entry exactly under its IPA "
        "surface form, no translation needed.\n"
    )

    # Build an IPA -> [vocab symbol] reverse index for quick lookup.
    ipa_to_symbols: dict[str, list[str]] = {}
    for entry in alphabet["alphabet"]:
        ipa_to_symbols.setdefault(entry["ipa"], []).append(entry["espeak_symbol"])
        for alt in entry.get("ipa_alternates", []):
            ipa_to_symbols.setdefault(_strip_slashes(alt), []).append(entry["espeak_symbol"])
    # de-dup
    for k, v in ipa_to_symbols.items():
        ipa_to_symbols[k] = sorted(set(v))

    summary_direct = 0
    summary_notation = 0
    summary_phonological = 0

    per_language: list[tuple[str, list[dict[str, Any]]]] = []
    for lang in sorted(l1):
        rows: list[dict[str, Any]] = []
        for tok in l1[lang]:
            ipa = _strip_slashes(tok)
            matches = sorted(set(ipa_to_symbols.get(ipa, [])))
            if not matches:
                category = "phonological"
                summary_phonological += 1
            elif ipa in matches:
                category = "direct"
                summary_direct += 1
            else:
                category = "notation-only"
                summary_notation += 1
            rows.append({
                "token": tok,
                "ipa": ipa,
                "matches": matches,
                "category": category,
            })
        per_language.append((lang, rows))

    lines.append("\n## Summary\n\n")
    lines.append(f"- direct matches:        {summary_direct}\n")
    lines.append(f"- notation-only drift:   {summary_notation}\n")
    lines.append(f"- phonological gaps:     {summary_phonological}\n")
    total = summary_direct + summary_notation + summary_phonological
    lines.append(f"- total tokens audited:  {total}\n")
    lines.append("\n")

    # Highlight phonological gaps explicitly — these are the cases Loop 3b
    # might need to handle with a translation layer or a "model can't see this"
    # caveat in the UI.
    phon_only: dict[str, list[str]] = {}
    notation_only: dict[str, list[str]] = {}
    for lang, rows in per_language:
        for row in rows:
            if row["category"] == "phonological":
                phon_only.setdefault(row["token"], []).append(lang)
            elif row["category"] == "notation-only":
                notation_only.setdefault(row["token"], []).append(lang)

    if phon_only:
        lines.append("## Phonological gaps (model cannot emit)\n\n")
        lines.append(
            "These phonemes appear in `L1_PHONEMES` but no vocabulary entry covers "
            "them. Free decoding cannot produce them under any notation; Phase 1/2 "
            "consumers should treat them as structurally invisible.\n\n"
        )
        for tok in sorted(phon_only):
            langs = ", ".join(sorted(phon_only[tok]))
            lines.append(f"- `{tok}`: {langs}\n")
        lines.append("\n")

    if notation_only:
        lines.append("## Notation-only drift (translation layer recommended)\n\n")
        lines.append(
            "These phonemes are in the model's vocabulary but under a different "
            "surface form than `L1_PHONEMES` uses. Loop 3b Phase 1 should normalise "
            "via `phoneme_alphabet.json#ipa_alternates`.\n\n"
        )
        for tok in sorted(notation_only):
            langs = ", ".join(sorted(notation_only[tok]))
            ipa = _strip_slashes(tok)
            emitted = sorted(set(ipa_to_symbols.get(ipa, [])))
            lines.append(f"- `{tok}` (emitted as `{', '.join(emitted)}`): {langs}\n")
        lines.append("\n")

    lines.append("## Per-language detail\n\n")
    for lang, rows in per_language:
        lines.append(f"### {lang}\n\n")
        if not rows:
            lines.append("- (no L1 phonemes listed)\n\n")
            continue
        lines.append("| token | IPA | model vocab symbol(s) | category |\n")
        lines.append("|---|---|---|---|\n")
        for row in rows:
            emitted = ", ".join(f"`{m}`" for m in row["matches"]) if row["matches"] else "—"
            lines.append(
                f"| `{row['token']}` | `{row['ipa']}` | {emitted} | {row['category']} |\n"
            )
        lines.append("\n")

    return "".join(lines)


def write_catalogue(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures subcommand (optional empirical pass).
# ---------------------------------------------------------------------------


def run_fixtures(fixtures_dir: Path) -> dict[str, Any]:
    import parselmouth  # noqa: PLC0415

    from app.services import phonemes as phonemes_svc  # noqa: PLC0415

    audio_extensions = {".wav", ".flac", ".ogg", ".mp3", ".webm", ".m4a"}
    files = sorted(p for p in fixtures_dir.rglob("*") if p.suffix.lower() in audio_extensions)
    if not files:
        raise SystemExit(f"No audio files found under {fixtures_dir}")

    observations: dict[str, dict[str, Any]] = {}
    for path in files:
        try:
            sound = parselmouth.Sound(str(path))
            inv = phonemes_svc.extract_phonemes(sound)
            observations[str(path.relative_to(fixtures_dir))] = {
                "total_tokens": inv.total_tokens,
                "unique_symbols": sorted(inv.counts.keys()),
            }
        except Exception as e:  # noqa: BLE001
            observations[str(path.relative_to(fixtures_dir))] = {
                "error": str(e),
            }

    return {
        "_audited_at": _AUDITED_AT,
        "_source_dir": str(fixtures_dir),
        "files": observations,
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _cmd_audit(_args: argparse.Namespace) -> int:
    vocab = _load_vocab()
    payload = build_alphabet(vocab)
    write_alphabet(payload, _ALPHABET_PATH)
    sys.stdout.write(
        f"Wrote {_ALPHABET_PATH.relative_to(_REPO_ROOT)} "
        f"({len(payload['alphabet'])} symbols)\n"
    )
    return 0


def _cmd_catalogue(_args: argparse.Namespace) -> int:
    if not _ALPHABET_PATH.exists():
        sys.stderr.write(
            f"{_ALPHABET_PATH} missing. Run `python scripts/notation_audit.py audit` first.\n"
        )
        return 2
    alphabet = json.loads(_ALPHABET_PATH.read_text(encoding="utf-8"))
    if not _SNAPSHOT_PATH.exists():
        sys.stderr.write(f"{_SNAPSHOT_PATH} missing.\n")
        return 2
    l1 = parse_l1_phonemes(_SNAPSHOT_PATH)
    if not l1:
        sys.stderr.write("Parsed 0 L1 entries from snapshot.\n")
        return 2
    text = build_drift_catalogue(l1, alphabet)
    write_catalogue(text, _CATALOGUE_PATH)
    sys.stdout.write(
        f"Wrote {_CATALOGUE_PATH.relative_to(_REPO_ROOT)} "
        f"({len(l1)} languages)\n"
    )
    return 0


def _cmd_check(_args: argparse.Namespace) -> int:
    vocab = _load_vocab()
    expected_alphabet = build_alphabet(vocab)
    current_alphabet_text = (
        _ALPHABET_PATH.read_text(encoding="utf-8") if _ALPHABET_PATH.exists() else ""
    )
    expected_alphabet_text = json.dumps(expected_alphabet, ensure_ascii=False, indent=2) + "\n"
    alphabet_ok = current_alphabet_text == expected_alphabet_text

    catalogue_ok = True
    if _ALPHABET_PATH.exists() and _SNAPSHOT_PATH.exists():
        alphabet = json.loads(current_alphabet_text or expected_alphabet_text)
        l1 = parse_l1_phonemes(_SNAPSHOT_PATH)
        expected_catalogue = build_drift_catalogue(l1, alphabet)
        current_catalogue = (
            _CATALOGUE_PATH.read_text(encoding="utf-8") if _CATALOGUE_PATH.exists() else ""
        )
        catalogue_ok = expected_catalogue == current_catalogue

    if alphabet_ok and catalogue_ok:
        sys.stdout.write("audit OK: phoneme_alphabet.json and notation_drift_catalogue.md are current\n")
        return 0

    if not alphabet_ok:
        sys.stderr.write(
            "phoneme_alphabet.json is stale. Re-run: "
            "python scripts/notation_audit.py audit\n"
        )
    if not catalogue_ok:
        sys.stderr.write(
            "notation_drift_catalogue.md is stale. Re-run: "
            "python scripts/notation_audit.py catalogue\n"
        )
    return 1


def _cmd_fixtures(args: argparse.Namespace) -> int:
    fixtures_dir = Path(args.fixtures_dir).resolve()
    if not fixtures_dir.exists():
        sys.stderr.write(f"{fixtures_dir} does not exist.\n")
        return 2
    payload = run_fixtures(fixtures_dir)
    _OBSERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OBSERVATIONS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sys.stdout.write(
        f"Wrote {_OBSERVATIONS_PATH.relative_to(_REPO_ROOT)} "
        f"({len(payload['files'])} files)\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit", help="Regenerate phoneme_alphabet.json").set_defaults(
        func=_cmd_audit
    )
    subparsers.add_parser("catalogue", help="Regenerate notation_drift_catalogue.md").set_defaults(
        func=_cmd_catalogue
    )
    subparsers.add_parser("check", help="Check that committed files match a fresh audit").set_defaults(
        func=_cmd_check
    )
    fixtures_p = subparsers.add_parser(
        "fixtures",
        help="Run the phoneme detector over an audio directory and record observations",
    )
    fixtures_p.add_argument("fixtures_dir", help="Directory containing audio fixtures")
    fixtures_p.set_defaults(func=_cmd_fixtures)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
