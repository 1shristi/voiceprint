# voiceprint

Phonetic feature extraction service. Takes a voice recording, returns measured acoustic features — F0, formants, VOT, syllable rate.

Built to power [linguamatch](https://github.com/1shristi/linguamatch) and similar voice-analysis projects, replacing LLM-based "audio interpretation" with real measurement.

## Stack

- FastAPI (Python 3.11+)
- [parselmouth](https://parselmouth.readthedocs.io/) — Praat bindings (no separate Praat install needed; parselmouth ships with it)
- numpy

## Local development

Requires Python 3.11+, `ffmpeg` (decoding webm audio), and `espeak-ng` (used by phonemizer for phoneme detection):

```sh
brew install ffmpeg espeak-ng   # macOS
# or on Debian/Ubuntu:
# apt-get install ffmpeg espeak-ng
```

On macOS you may need to point phonemizer at the espeak library:

```sh
export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
```

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
| POST | `/analyze` | Extract phonetic features from audio |

### `/analyze` returns

```json
{
  "duration_s": 1.23,
  "f0": {
    "mean_hz": 198.4,
    "min_hz": 84.0,
    "max_hz": 312.5,
    "std_hz": 42.1,
    "voiced_fraction": 0.71
  },
  "formants": {
    "f1_mean_hz": 530.0,
    "f2_mean_hz": 1620.0,
    "f3_mean_hz": 2480.0
  },
  "syllable_rate_hz": 4.2,
  "vot_ms": null,
  "phonemes": {
    "counts": {"i": 4, "θ": 2, "ɹ": 7, "...": "..."},
    "total_tokens": 38
  },
  "notes": []
}
```

### Disabling phoneme detection

Set `VOICEPRINT_DISABLE_PHONEMES=1` to skip the Wav2Vec2 phoneme model. Useful for fast/CI test runs or when only acoustic features are needed.

## Running tests

Fast tests (no model downloads):

```sh
pytest -m "not slow"
```

Full suite including phoneme model integration (downloads ~370 MB on first run):

```sh
PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib pytest
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
