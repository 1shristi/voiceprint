import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.services import extractor


_ALPHABET_PATH = Path(__file__).resolve().parent.parent / "data" / "phoneme_alphabet.json"


def _read_alphabet_version() -> str:
    """Read the phoneme_alphabet.json `version` field at import time.

    Phase 0 of loop3a: every /analyze response advertises which IPA mapping it
    was generated against so downstream consumers (linguamatch) can validate
    their consumption logic matches voiceprint's current alphabet. We surface
    the version directly from the JSON so the field stays in sync as the
    alphabet evolves.
    """
    return json.loads(_ALPHABET_PATH.read_text(encoding="utf-8"))["version"]


PHONEME_ALPHABET_VERSION = _read_alphabet_version()

router = APIRouter()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("API_KEY", "").strip()
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")


class TranscriptInputModel(BaseModel):
    """Loop 3a Phase 1: optional known transcript the audio should be aligned against.

    The two `expected_language` fields in the request (top-level for stretch-phoneme
    probe scoring, this one nested for orthography→IPA conversion) are deliberately
    independent — see the spec §5.2 warning. Do not collapse them.
    """

    format: str = Field(
        ...,
        description="Transcript notation: 'ipa', 'espeak', or 'orthography_with_dictionary'.",
    )
    content: str = Field(..., description="Transcript text in the declared format.")
    expected_language: str | None = Field(
        default=None,
        description=(
            "Locale hint for orthography→IPA conversion (e.g. 'en', 'zh', 'ar'). "
            "Ignored for format='ipa' and format='espeak'. Distinct from the "
            "top-level expected_language, which weights stretch-phoneme probe scoring."
        ),
    )


class AnalyzeRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded audio (webm or wav).")
    format: str = Field(default="webm", description="Audio container format: 'webm' or 'wav'.")
    # Optional hint for what language the speaker was asked to produce on this clip.
    # Drives per-language stretch-phoneme scoring (Mandarin tones/retroflexes,
    # Arabic pharyngeals/emphatics) — see app/services/phonemes.py::PROBE_SETS.
    expected_language: str | None = Field(
        default=None,
        description="Expected language of the clip (e.g. 'mandarin', 'arabic'). Optional hint for stretch-phoneme scoring.",
    )
    # Optional hint for what language the speaker CLAIMS to be speaking fluently on
    # this clip (e.g. their stated L1, or English for the diagnostic). Drives a
    # marker-based language-match check that flags obvious wrong-language clips.
    # See app/services/language_check.py.
    claimed_language: str | None = Field(
        default=None,
        description="Language the speaker is claimed to be speaking fluently on this clip. Triggers a language-match check against curated marker phonemes.",
    )
    transcript: TranscriptInputModel | None = Field(
        default=None,
        description=(
            "Optional known transcript of the clip. When provided, voiceprint runs "
            "CTC forced alignment over the same Wav2Vec2 logits and returns "
            "per-position evidence in the `aligned_phonemes` response block."
        ),
    )


class F0Block(BaseModel):
    mean_hz: float | None = None
    min_hz: float | None = None
    max_hz: float | None = None
    std_hz: float | None = None
    std_semitones: float | None = None
    range_semitones_p10_p90: float | None = None
    voiced_fraction: float = 0.0


class FormantsBlock(BaseModel):
    f1_mean_hz: float | None = None
    f2_mean_hz: float | None = None
    f3_mean_hz: float | None = None


class PhonemesBlock(BaseModel):
    counts: dict[str, int] = Field(default_factory=dict)
    total_tokens: int = 0


class StretchProbeBlock(BaseModel):
    label: str
    ipa: str
    expected_count: int
    count: int
    approximate_count: int
    status: str  # "hit" | "approximate" | "missed"


class StretchPhonemesBlock(BaseModel):
    expected_language: str
    probes: list[StretchProbeBlock] = Field(default_factory=list)


class LanguageMatchBlock(BaseModel):
    claimed_language: str
    # One of: "matches", "uncertain", "mismatch", "insufficient_signal", "unknown_language"
    verdict: str
    score: float | None = None
    positive_hits: int = 0
    positive_total: int = 0
    negative_clean: int = 0
    negative_total: int = 0
    notes: list[str] = Field(default_factory=list)


class AlignedTop3Block(BaseModel):
    phoneme: str
    prob: float


class AlignedPositionBlock(BaseModel):
    target_phoneme: str
    target_index_in_transcript: int
    start_ms: int
    end_ms: int
    avg_log_prob: float
    top1_predicted: str
    top3_alternatives: list[AlignedTop3Block] = Field(default_factory=list)
    match_classification: str  # "produced" | "near_miss" | "absent"


class AlignedSummaryBlock(BaseModel):
    expected_count: int
    produced_count: int
    near_miss_count: int
    absent_count: int
    evidence_strength: str  # "strong" | "moderate" | "weak"


class AlignedPhonemesBlock(BaseModel):
    transcript_format: str
    alignment_quality: str  # "high" | "medium" | "low"
    positions: list[AlignedPositionBlock] = Field(default_factory=list)
    summary_by_phoneme: dict[str, AlignedSummaryBlock] = Field(default_factory=dict)
    alignment_warnings: list[str] = Field(default_factory=list)


class VotMeasurementBlock(BaseModel):
    phoneme: str
    time_s: float
    vot_ms: float
    aspiration_class: str


class VotBlock(BaseModel):
    aspirated_voiceless_mean_ms: float | None = None
    plain_voiceless_mean_ms: float | None = None
    voiced_mean_ms: float | None = None
    measurements: list[VotMeasurementBlock] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    duration_s: float
    f0: F0Block
    formants: FormantsBlock
    syllable_rate_hz: float | None = None
    vot: VotBlock = Field(default_factory=VotBlock)
    phonemes: PhonemesBlock = Field(default_factory=PhonemesBlock)
    stretch_phonemes: StretchPhonemesBlock | None = None
    language_match: LanguageMatchBlock | None = None
    # Loop 3a Phase 1: per-position evidence when the caller supplied a transcript.
    # Absent (null) when no transcript was provided — legacy behaviour unchanged.
    aligned_phonemes: AlignedPhonemesBlock | None = None
    # Loop 3a Phase 0: the IPA-mapping version this response was generated against.
    # See app/data/phoneme_alphabet.json. Consumers can use this to validate that
    # their phoneme consumption logic matches voiceprint's current alphabet.
    phoneme_alphabet_version: str = PHONEME_ALPHABET_VERSION
    notes: list[str] = Field(default_factory=list)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(require_api_key)],
)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    transcript_input = None
    if req.transcript is not None:
        from app.services.alignment import TranscriptInput

        transcript_input = TranscriptInput(
            format=req.transcript.format,
            content=req.transcript.content,
            expected_language=req.transcript.expected_language,
        )

    try:
        features = extractor.extract_all(
            req.audio_base64,
            req.format,
            expected_language=req.expected_language,
            claimed_language=req.claimed_language,
            transcript=transcript_input,
        )
    except extractor.AudioDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except extractor.FeatureExtractionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e

    return AnalyzeResponse(
        duration_s=features.duration_s,
        f0=F0Block(
            mean_hz=features.f0.mean_hz,
            min_hz=features.f0.min_hz,
            max_hz=features.f0.max_hz,
            std_hz=features.f0.std_hz,
            std_semitones=features.f0.std_semitones,
            range_semitones_p10_p90=features.f0.range_semitones_p10_p90,
            voiced_fraction=features.f0.voiced_fraction,
        ),
        formants=FormantsBlock(
            f1_mean_hz=features.formants.f1_mean_hz,
            f2_mean_hz=features.formants.f2_mean_hz,
            f3_mean_hz=features.formants.f3_mean_hz,
        ),
        syllable_rate_hz=features.syllable_rate_hz,
        vot=VotBlock(
            aspirated_voiceless_mean_ms=features.vot_aspirated_voiceless_mean_ms,
            plain_voiceless_mean_ms=features.vot_plain_voiceless_mean_ms,
            voiced_mean_ms=features.vot_voiced_mean_ms,
            measurements=[
                VotMeasurementBlock(
                    phoneme=m.phoneme,
                    time_s=m.time_s,
                    vot_ms=m.vot_ms,
                    aspiration_class=m.aspiration_class,
                )
                for m in features.vot_measurements
            ],
        ),
        phonemes=PhonemesBlock(
            counts=features.phoneme_counts,
            total_tokens=features.phoneme_total_tokens,
        ),
        stretch_phonemes=(
            StretchPhonemesBlock(
                expected_language=features.stretch_score.expected_language,
                probes=[
                    StretchProbeBlock(
                        label=p.label,
                        ipa=p.ipa,
                        expected_count=p.expected_count,
                        count=p.count,
                        approximate_count=p.approximate_count,
                        status=p.status,
                    )
                    for p in features.stretch_score.probes
                ],
            )
            if features.stretch_score is not None
            else None
        ),
        language_match=(
            LanguageMatchBlock(
                claimed_language=features.language_match.claimed_language,
                verdict=features.language_match.verdict,
                score=features.language_match.score,
                positive_hits=features.language_match.positive_hits,
                positive_total=features.language_match.positive_total,
                negative_clean=features.language_match.negative_clean,
                negative_total=features.language_match.negative_total,
                notes=features.language_match.notes,
            )
            if features.language_match is not None
            else None
        ),
        aligned_phonemes=(
            AlignedPhonemesBlock(
                transcript_format=features.aligned_phonemes.transcript_format,
                alignment_quality=features.aligned_phonemes.alignment_quality,
                positions=[
                    AlignedPositionBlock(
                        target_phoneme=p.target_phoneme,
                        target_index_in_transcript=p.target_index_in_transcript,
                        start_ms=p.start_ms,
                        end_ms=p.end_ms,
                        avg_log_prob=p.avg_log_prob,
                        top1_predicted=p.top1_predicted,
                        top3_alternatives=[
                            AlignedTop3Block(phoneme=a.phoneme, prob=a.prob)
                            for a in p.top3_alternatives
                        ],
                        match_classification=p.match_classification,
                    )
                    for p in features.aligned_phonemes.positions
                ],
                summary_by_phoneme={
                    k: AlignedSummaryBlock(
                        expected_count=v.expected_count,
                        produced_count=v.produced_count,
                        near_miss_count=v.near_miss_count,
                        absent_count=v.absent_count,
                        evidence_strength=v.evidence_strength,
                    )
                    for k, v in features.aligned_phonemes.summary_by_phoneme.items()
                },
                alignment_warnings=list(features.aligned_phonemes.alignment_warnings),
            )
            if features.aligned_phonemes is not None
            else None
        ),
        notes=features.notes,
    )
