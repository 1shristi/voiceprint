"""
RAVDESS → voiceprint ST calibration.

Walks data/ravdess/, picks a stratified sample across four emotion-intensity
buckets (neutral, calm-normal, happy-strong, angry-strong), POSTs each clip to
a local voiceprint /analyze, and writes one CSV row per clip with the ST
metrics. The CSV is flushed after every row so an interrupted run loses at
most one clip.

Run: python run_calibration.py --voiceprint-url http://localhost:8765 --n 60
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

# RAVDESS filename: NN-NN-NN-NN-NN-NN-NN.wav, dash-separated.
# Positions: 0=modality, 1=channel, 2=emotion, 3=intensity, 4=statement, 5=rep, 6=actor
EMOTION_NAMES = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
INTENSITY_NAMES = {"01": "normal", "02": "strong"}

# Buckets we calibrate against. Each entry: (emotion_code, intensity_code, label)
BUCKETS = [
    ("01", "01", "neutral"),         # neutral (no strong variant)
    ("02", "01", "calm-normal"),     # calm at normal intensity
    ("03", "02", "happy-strong"),    # happy at strong intensity
    ("05", "02", "angry-strong"),    # angry at strong intensity
]


def parse_filename(stem: str) -> dict[str, str] | None:
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    return {
        "modality": parts[0],
        "channel": parts[1],
        "emotion": parts[2],
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor": parts[6],
    }


def select_sample(data_dir: Path, n_per_bucket: int, seed: int) -> list[tuple[Path, dict[str, str], str]]:
    """Stratified sample: n_per_bucket clips per bucket, balanced as much as possible across actors."""
    rng = random.Random(seed)
    by_bucket: dict[str, list[tuple[Path, dict[str, str]]]] = {label: [] for _, _, label in BUCKETS}
    bucket_lookup = {(emo, ints): label for emo, ints, label in BUCKETS}

    for wav in sorted(data_dir.rglob("*.wav")):
        meta = parse_filename(wav.stem)
        if meta is None:
            continue
        if meta["channel"] != "01":  # speech only
            continue
        label = bucket_lookup.get((meta["emotion"], meta["intensity"]))
        if label is None:
            continue
        by_bucket[label].append((wav, meta))

    selected: list[tuple[Path, dict[str, str], str]] = []
    for label, clips in by_bucket.items():
        if not clips:
            print(f"  WARNING: bucket {label} has 0 clips — RAVDESS not extracted yet?", file=sys.stderr)
            continue
        rng.shuffle(clips)
        take = clips[:n_per_bucket]
        if len(take) < n_per_bucket:
            print(f"  bucket {label}: only {len(take)} available (asked for {n_per_bucket})", file=sys.stderr)
        for wav, meta in take:
            selected.append((wav, meta, label))

    rng.shuffle(selected)  # randomise processing order
    return selected


def post_analyze(voiceprint_url: str, audio_b64: str, timeout_s: int = 60) -> dict:
    body = json.dumps({"audio_base64": audio_b64, "format": "wav"}).encode("utf-8")
    req = urllib.request.Request(
        f"{voiceprint_url.rstrip('/')}/analyze",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voiceprint-url", default="http://localhost:8765")
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data" / "ravdess"))
    parser.add_argument("--results-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--n", type=int, default=60, help="clips per bucket")
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists() or not any(data_dir.rglob("*.wav")):
        print(f"ERROR: no RAVDESS .wav files under {data_dir}. See README.md for download.", file=sys.stderr)
        sys.exit(2)

    # Liveness check.
    try:
        with urllib.request.urlopen(f"{args.voiceprint_url}/health", timeout=5) as resp:
            health = json.loads(resp.read())
        print(f"voiceprint ok: {health}")
    except Exception as e:
        print(f"ERROR: cannot reach voiceprint at {args.voiceprint_url}/health: {e}", file=sys.stderr)
        sys.exit(2)

    sample = select_sample(data_dir, args.n, args.seed)
    print(f"selected {len(sample)} clips across {len(BUCKETS)} buckets")

    csv_path = results_dir / f"ravdess_st_{date.today().isoformat()}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "filename", "actor", "gender", "emotion", "intensity",
            "bucket", "duration_s", "voiced_fraction",
            "f0_std_hz", "f0_std_semitones", "f0_range_hz_max_min", "f0_range_semitones_p10_p90",
            "phonemes_total_tokens", "notes",
        ])
        fh.flush()

        n_ok = 0
        n_fail = 0
        t0 = time.time()
        for i, (wav, meta, bucket) in enumerate(sample, start=1):
            try:
                audio_b64 = base64.b64encode(wav.read_bytes()).decode("ascii")
                resp = post_analyze(args.voiceprint_url, audio_b64, timeout_s=args.timeout)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                n_fail += 1
                print(f"  [{i}/{len(sample)}] FAIL {wav.name}: {type(e).__name__}: {e}", file=sys.stderr)
                continue

            f0 = resp.get("f0", {}) or {}
            phonemes = resp.get("phonemes", {}) or {}
            notes = resp.get("notes", []) or []
            f0_max, f0_min = f0.get("max_hz"), f0.get("min_hz")
            f0_range_hz = (f0_max - f0_min) if (f0_max is not None and f0_min is not None) else None

            actor_id = int(meta["actor"])
            gender = "male" if actor_id % 2 == 1 else "female"

            writer.writerow([
                wav.name, actor_id, gender,
                EMOTION_NAMES[meta["emotion"]], INTENSITY_NAMES[meta["intensity"]],
                bucket, resp.get("duration_s"), f0.get("voiced_fraction"),
                f0.get("std_hz"), f0.get("std_semitones"),
                f0_range_hz, f0.get("range_semitones_p10_p90"),
                phonemes.get("total_tokens"), "; ".join(notes),
            ])
            fh.flush()
            n_ok += 1

            if i % 10 == 0 or i == len(sample):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(sample) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(sample)}] ok={n_ok} fail={n_fail}  rate={rate:.2f}/s  eta={eta:.0f}s")

    print(f"\nwrote {csv_path}  ({n_ok} rows ok, {n_fail} failed)")


if __name__ == "__main__":
    main()
