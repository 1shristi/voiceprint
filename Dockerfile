FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# parselmouth ships compiled Praat in its wheel, so no apt-get praat needed.
# Install build deps only if a wheel is unavailable for the platform.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip
# Install CPU-only torch first to avoid pulling ~1.5 GB of CUDA libraries we
# don't use. Must come before the regular `pip install .` so torchaudio's
# version resolution sees the CPU torch already in place.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchaudio
RUN pip install --no-cache-dir .

# Pre-download the Wav2Vec2-Phoneme model so cold-start requests don't have
# to fetch ~370 MB. Cached under the standard HF hub directory.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from transformers import AutoModelForCTC, AutoProcessor; \
    name='facebook/wav2vec2-lv-60-espeak-cv-ft'; \
    AutoProcessor.from_pretrained(name); \
    AutoModelForCTC.from_pretrained(name)"

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
