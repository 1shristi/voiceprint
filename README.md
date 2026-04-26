# voiceprint

Phonetic feature extraction service. Takes a voice recording, returns measured acoustic features — F0, formants, VOT, syllable rate.

Built to power [linguamatch](https://github.com/1shristi/linguamatch) and similar voice-analysis projects, replacing LLM-based "audio interpretation" with real measurement.

## Stack

- FastAPI (Python 3.11+)
- [parselmouth](https://parselmouth.readthedocs.io/) — Praat bindings (no separate Praat install needed; parselmouth ships with it)
- numpy

## Local development

Requires Python 3.11+.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Service runs on `http://localhost:8000`. API docs at `/docs`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/` | Service info |
| POST | `/analyze` | Extract phonetic features from audio (501 until phase 2) |

## Running tests

```sh
pytest
```

## Docker

```sh
docker build -t voiceprint .
docker run -p 8000:8000 voiceprint
```

## Environment

See `.env.example`. All vars optional:

- `ALLOWED_ORIGINS` — comma-separated origins for CORS. Defaults to `*`
- `API_KEY` — shared secret required in `X-API-Key` header. Disabled when empty
