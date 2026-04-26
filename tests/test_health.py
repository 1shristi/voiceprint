from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_root() -> None:
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["service"] == "voiceprint"


def test_analyze_not_implemented() -> None:
    client = TestClient(app)
    res = client.post("/analyze", json={"audio_base64": "abc", "format": "webm"})
    assert res.status_code == 501
