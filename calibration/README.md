# Calibration

Empirical cutoffs for voiceprint metrics. Outputs feed into linguamatch's
`analyze-voice` Edge Function (the consumer that classifies speakers from
voiceprint's measurements).

## RAVDESS — ST pitch dynamism

Goal: pick `f0.std_semitones` and `f0.range_semitones_p10_p90` cutoffs that
classify a speaker as flat / moderate / expressive in a way that matches what
a human listener would say.

Source corpus: [RAVDESS](https://zenodo.org/record/1188976) — Ryerson
Audio-Visual Database of Emotional Speech and Song. Speech audio only. 24
actors × 8 emotions × 2 intensities × 2 statements × ~2 reps = ~1,440 clips.
License is CC BY-NC-SA 4.0 — we derive aggregate statistics, we do not
redistribute clips.

Approach: stratify across four emotion-intensity buckets that map cleanly
onto the flat / expressive axis —

| Bucket | Expected category |
|---|---|
| neutral (no intensity variation) | flat |
| calm, normal intensity | flat |
| happy, strong intensity | expressive |
| angry, strong intensity | expressive |

Sample 60 clips per bucket = 240 total. Run each through voiceprint locally,
capture ST metrics, look at where the distributions cluster, pick cutoffs at
the boundary.

## How to run

```sh
# 1. Voiceprint must be runnable locally — see ../README.md for setup.
# 2. Download RAVDESS (one-time):
cd data/
curl -L -o ravdess.zip https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
unzip -q ravdess.zip -d ravdess/
rm ravdess.zip
cd ..

# 3. Start voiceprint locally (in a separate terminal):
cd ..
source .venv/bin/activate
export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
uvicorn app.main:app --port 8765

# 4. Run calibration:
cd calibration
source ../.venv/bin/activate
python run_calibration.py --voiceprint-url http://localhost:8765 --n 60
# Writes results/ravdess_st_<date>.csv incrementally.

# 5. Read the analysis:
cat results/ravdess_st_<date>_report.md
```
