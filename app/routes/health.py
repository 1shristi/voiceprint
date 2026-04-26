from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/")
def root() -> dict[str, str]:
    return {
        "service": "voiceprint",
        "version": "0.1.0",
        "docs": "/docs",
    }
