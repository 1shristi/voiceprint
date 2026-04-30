import os

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.services import extractor

router = APIRouter()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("API_KEY", "").strip()
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")


class AnalyzeRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded audio (webm or wav).")
    format: str = Field(default="webm", description="Audio container format: 'webm' or 'wav'.")


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
    notes: list[str] = Field(default_factory=list)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(require_api_key)],
)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    try:
        features = extractor.extract_all(req.audio_base64, req.format)
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
        notes=features.notes,
    )
