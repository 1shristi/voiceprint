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
class PhonemeInventory:
    """Per-phoneme count plus chronological occurrences of each emitted token."""
    counts: dict[str, int]
    total_tokens: int
    occurrences: list[PhonemeOccurrence]

    def status(self, phoneme: str, *, present_threshold: int = 2) -> str:
        """Classify a single phoneme as 'present' / 'approximate' / 'absent'."""
        c = self.counts.get(phoneme, 0)
        if c >= present_threshold:
            return "present"
        if c == 1:
            return "approximate"
        return "absent"


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


def extract_phonemes(sound: parselmouth.Sound) -> PhonemeInventory:
    """Run the audio through Wav2Vec2-Phoneme and tally phoneme counts + timed occurrences."""
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
    return PhonemeInventory(
        counts=dict(counts),
        total_tokens=len(occurrences),
        occurrences=occurrences,
    )
