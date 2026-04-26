import os

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

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


class AnalyzeResponse(BaseModel):
    f0_mean_hz: float | None = None
    f0_min_hz: float | None = None
    f0_max_hz: float | None = None
    f0_std_hz: float | None = None
    syllable_rate_hz: float | None = None
    formants: dict[str, float] | None = None
    vot_ms: float | None = None
    duration_s: float | None = None
    notes: list[str] = Field(default_factory=list)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(require_api_key)],
)
def analyze(_req: AnalyzeRequest) -> AnalyzeResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Feature extraction not yet implemented — comes in phase 2.",
    )
