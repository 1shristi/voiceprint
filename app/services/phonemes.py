"""Phoneme inventory detection using a pretrained Wav2Vec2 phoneme classifier.

Uses facebook/wav2vec2-lv-60-espeak-cv-ft, which outputs eSpeak-style phoneme
labels covering ~200 phonemes across many languages.

The model is loaded lazily on first request (~370 MB download, then cached on disk).
"""

from __future__ import annotations

import os
import sys
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parselmouth


def _autoconfigure_espeak() -> None:
    """If PHONEMIZER_ESPEAK_LIBRARY isn't set, try the standard locations."""
    if "PHONEMIZER_ESPEAK_LIBRARY" in os.environ:
        return
    if sys.platform == "darwin":
        candidates = [
            Path("/opt/homebrew/lib/libespeak-ng.dylib"),  # Apple Silicon Homebrew
            Path("/usr/local/lib/libespeak-ng.dylib"),     # Intel Homebrew
        ]
    else:
        candidates = [
            Path("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"),
            Path("/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1"),
            Path("/usr/lib/libespeak-ng.so.1"),
        ]
    for c in candidates:
        if c.exists():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(c)
            return


_autoconfigure_espeak()


_MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
_TARGET_SR = 16_000

# Markers we drop from the inventory — these aren't phonemes
_NON_PHONEME_TOKENS = {"<pad>", "<s>", "</s>", "<unk>", "|", "[PAD]", "[UNK]", " ", ""}

# Lazy globals — loaded on first call
_lock = threading.Lock()
_model = None
_processor = None
_model_load_error: str | None = None


@dataclass
class PhonemeOccurrence:
    """A single occurrence of a phoneme with timing in the audio."""
    phoneme: str
    start_s: float
    end_s: float


@dataclass
class PhoneProbe:
    """A target phoneme the speaker was asked to produce in the stretch clip.

    `accept` and `approximate` are tuples of eSpeak/IPA token forms the Wav2Vec2
    model might emit — multiple variants are listed because the model's exact
    token shape (combining marks, length, modifier letters) varies per phoneme.
    """
    label: str             # snake_case identifier for downstream UI / prompt
    ipa: str               # canonical IPA glyph for display
    expected_count: int    # canonical occurrences in the stretch sentence
    accept: tuple[str, ...]
    approximate: tuple[str, ...]


@dataclass
class StretchProbeResult:
    label: str
    ipa: str
    expected_count: int
    count: int             # accepted-token total
    approximate_count: int
    status: str            # "hit" | "approximate" | "missed"


@dataclass
class StretchScore:
    expected_language: str
    probes: list[StretchProbeResult]


@dataclass
class AlignmentArtifacts:
    """Frame-level model output reusable by forced-alignment.

    Captured during phoneme extraction so callers that want both free-decoded
    counts and CTC alignment against a known transcript don't pay the model
    forward pass twice. `log_probs` is shape (T, V); `vocab` maps tokenizer
    symbols → vocab index; `blank_id` is the CTC blank token's vocab index.
    `audio_duration_s` lets consumers convert frame indices back to ms.
    """

    log_probs: object  # torch.Tensor (T, V) — typed object to avoid torch import here
    blank_id: int
    vocab: dict[str, int]
    audio_duration_s: float


@dataclass
class PhonemeInventory:
    """Per-phoneme count plus chronological occurrences of each emitted token."""
    counts: dict[str, int]
    total_tokens: int
    occurrences: list[PhonemeOccurrence]
    stretch_score: StretchScore | None = None
    # Populated only when `extract_phonemes(..., include_alignment_artifacts=True)`.
    # Held here (rather than returned as a tuple) so the existing call sites stay
    # source-compatible and alignment-aware callers can opt in.
    alignment_artifacts: AlignmentArtifacts | None = None

    def status(self, phoneme: str, *, present_threshold: int = 2) -> str:
        """Classify a single phoneme as 'present' / 'approximate' / 'absent'."""
        c = self.counts.get(phoneme, 0)
        if c >= present_threshold:
            return "present"
        if c == 1:
            return "approximate"
        return "absent"


# Sentences chosen by linguamatch:
#   mandarin: "Māma qí mǎ, mǎ màn, māma mà mǎ." — tone-driven; only /tɕʰ/ is a
#     segmental probe English speakers don't have. Tones are not captured by the
#     eSpeak-CTC phoneme model, so this set is intentionally thin.
#   arabic:  "Khayṭ ḥarīr ʿalā ḥā'iṭ Khalīl." — rich in pharyngeals and emphatics.
#
# Token variants reflect that the model emits IPA-ish eSpeak labels with
# inconsistent diacritic placement; lists are intentionally generous and will
# be tightened once we observe real emissions on user audio.
PROBE_SETS: dict[str, list[PhoneProbe]] = {
    "mandarin": [
        PhoneProbe(
            label="aspirated_palatal_affricate",
            ipa="tɕʰ",
            expected_count=1,
            accept=("tɕʰ", "t͡ɕʰ", "tʃʰ", "t͡ʃʰ"),
            approximate=("tɕ", "t͡ɕ", "tʃ", "t͡ʃ", "tsʰ", "ts"),
        ),
    ],
    "arabic": [
        PhoneProbe(
            label="voiceless_velar_fricative",
            ipa="x",
            expected_count=2,
            accept=("x",),
            approximate=("k", "kʰ", "h"),
        ),
        PhoneProbe(
            label="voiceless_pharyngeal_fricative",
            ipa="ħ",
            expected_count=2,
            accept=("ħ",),
            approximate=("h",),
        ),
        PhoneProbe(
            label="voiced_pharyngeal_fricative",
            ipa="ʕ",
            expected_count=1,
            accept=("ʕ",),
            approximate=("ʔ", "ʌ"),
        ),
        PhoneProbe(
            label="emphatic_t",
            ipa="tˤ",
            expected_count=2,
            accept=("tˤ", "t̪ˤ"),
            approximate=("t", "tʰ", "t̪"),
        ),
        PhoneProbe(
            label="alveolar_trill",
            ipa="r",
            expected_count=1,
            accept=("r", "rː"),
            approximate=("ɾ", "ɹ"),
        ),
    ],
}


# Map flexible language inputs to canonical PROBE_SETS keys.
_LANGUAGE_ALIASES: dict[str, str] = {
    "mandarin": "mandarin",
    "chinese": "mandarin",
    "zh": "mandarin",
    "zh-cn": "mandarin",
    "zh_cn": "mandarin",
    "cmn": "mandarin",
    "arabic": "arabic",
    "ar": "arabic",
    "msa": "arabic",
    "arabic_msa": "arabic",
}


def _canonical_language(expected_language: str | None) -> str | None:
    if not expected_language:
        return None
    return _LANGUAGE_ALIASES.get(expected_language.strip().lower())


def score_stretch(inventory: PhonemeInventory, expected_language: str | None) -> StretchScore | None:
    """Score a stretch clip's phoneme inventory against per-language probe sets.

    Returns None if `expected_language` is missing or unrecognized. Probes are
    scored independently: a probe is a "hit" if accepted-token count meets the
    expected count, "approximate" if there's at least one accepted or
    approximate token but fewer than expected, and "missed" otherwise.
    """
    key = _canonical_language(expected_language)
    if key is None or key not in PROBE_SETS:
        return None

    results: list[StretchProbeResult] = []
    for probe in PROBE_SETS[key]:
        count = sum(inventory.counts.get(tok, 0) for tok in probe.accept)
        approximate_count = sum(inventory.counts.get(tok, 0) for tok in probe.approximate)
        if count >= probe.expected_count:
            status = "hit"
        elif count >= 1 or approximate_count >= 1:
            status = "approximate"
        else:
            status = "missed"
        results.append(
            StretchProbeResult(
                label=probe.label,
                ipa=probe.ipa,
                expected_count=probe.expected_count,
                count=count,
                approximate_count=approximate_count,
                status=status,
            )
        )

    return StretchScore(expected_language=key, probes=results)


def _ensure_model_loaded() -> None:
    global _model, _processor, _model_load_error

    if _model is not None and _processor is not None:
        return
    if _model_load_error is not None:
        raise RuntimeError(f"phoneme model previously failed to load: {_model_load_error}")

    with _lock:
        # Re-check after acquiring lock
        if _model is not None and _processor is not None:
            return
        try:
            # Imports kept inside the function so cold imports of this module stay cheap.
            from transformers import AutoModelForCTC, AutoProcessor

            _processor = AutoProcessor.from_pretrained(_MODEL_NAME)
            _model = AutoModelForCTC.from_pretrained(_MODEL_NAME)
            _model.eval()
        except Exception as e:
            _model_load_error = str(e)
            raise


def _resample_if_needed(sound: parselmouth.Sound, target_sr: int = _TARGET_SR) -> np.ndarray:
    """Return mono float32 numpy array at the target sample rate."""
    if int(sound.sampling_frequency) != target_sr:
        sound = sound.resample(target_sr)
    samples = np.asarray(sound.values[0], dtype=np.float32)
    # Normalize to [-1, 1] if the signal is louder
    peak = np.max(np.abs(samples))
    if peak > 1.0:
        samples = samples / peak
    return samples


def _decode_with_timing(
    pred_ids,  # 1-D tensor (T,)
    processor,
    *,
    audio_duration_s: float,
    blank_id: int,
) -> list[PhonemeOccurrence]:
    """Collapse CTC output into phoneme occurrences with start/end times."""
    ids = pred_ids.tolist()
    num_frames = len(ids)
    if num_frames == 0 or audio_duration_s <= 0:
        return []
    frame_duration_s = audio_duration_s / num_frames

    occurrences: list[PhonemeOccurrence] = []
    prev_id = -1
    run_start_frame = 0

    def flush(token_id: int, start_frame: int, end_frame: int) -> None:
        if token_id == blank_id or token_id < 0:
            return
        token = processor.tokenizer.convert_ids_to_tokens(int(token_id))
        if not token or token in _NON_PHONEME_TOKENS:
            return
        occurrences.append(
            PhonemeOccurrence(
                phoneme=token,
                start_s=start_frame * frame_duration_s,
                end_s=end_frame * frame_duration_s,
            )
        )

    for i, token_id in enumerate(ids):
        if token_id != prev_id:
            if prev_id != -1:
                flush(prev_id, run_start_frame, i)
            run_start_frame = i
            prev_id = token_id

    flush(prev_id, run_start_frame, num_frames)
    return occurrences


def extract_phonemes(
    sound: parselmouth.Sound,
    *,
    expected_language: str | None = None,
    include_alignment_artifacts: bool = False,
) -> PhonemeInventory:
    """Run the audio through Wav2Vec2-Phoneme and tally phoneme counts + timed occurrences.

    When `expected_language` is provided and matches a known stretch language
    (mandarin, arabic), the returned inventory also carries a `stretch_score`
    scoring user-produced tokens against the per-language probe set.

    When `include_alignment_artifacts=True`, the returned inventory also carries
    `alignment_artifacts` — log-softmaxed frame-level model output plus tokenizer
    metadata — for callers (Loop 3a Phase 1 forced alignment) that want to reuse
    the same forward pass for CTC alignment against a known transcript.
    """
    if os.getenv("VOICEPRINT_DISABLE_PHONEMES") == "1":
        return PhonemeInventory(counts={}, total_tokens=0, occurrences=[])

    _ensure_model_loaded()
    assert _model is not None and _processor is not None  # for type checker

    import torch

    samples = _resample_if_needed(sound, _TARGET_SR)
    audio_duration_s = float(len(samples) / _TARGET_SR)
    inputs = _processor(samples, sampling_rate=_TARGET_SR, return_tensors="pt")

    with torch.no_grad():
        logits = _model(**inputs).logits

    pred_ids = torch.argmax(logits, dim=-1)[0]  # (T,)

    blank_id = _processor.tokenizer.pad_token_id
    if blank_id is None:
        blank_id = 0

    occurrences = _decode_with_timing(
        pred_ids,
        _processor,
        audio_duration_s=audio_duration_s,
        blank_id=int(blank_id),
    )

    counts = Counter(occ.phoneme for occ in occurrences)
    inv = PhonemeInventory(
        counts=dict(counts),
        total_tokens=len(occurrences),
        occurrences=occurrences,
    )
    if expected_language:
        inv.stretch_score = score_stretch(inv, expected_language)
    if include_alignment_artifacts:
        # log_softmax across the vocab axis once here; forced_align expects
        # log-probabilities, not raw logits. Strip the batch dim so the
        # artifacts surface as (T, V) — single-clip is the only shape Loop 3a
        # callers ever produce.
        log_probs = torch.log_softmax(logits[0], dim=-1)
        inv.alignment_artifacts = AlignmentArtifacts(
            log_probs=log_probs,
            blank_id=int(blank_id),
            vocab=dict(_processor.tokenizer.get_vocab()),
            audio_duration_s=audio_duration_s,
        )
    return inv
