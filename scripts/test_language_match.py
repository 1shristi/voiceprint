#!/usr/bin/env python3
"""Run a real audio clip through voiceprint's /analyze and pretty-print the
language_match verdict. Standalone — no extra installs beyond Python 3.11+.

Usage:
    scripts/test_language_match.py recording.m4a english
    scripts/test_language_match.py recording.webm arabic --expected arabic
    scripts/test_language_match.py recording.wav spanish --url http://localhost:8000
    scripts/test_language_match.py recording.m4a english --raw   # full JSON response
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


VERDICT_MARKERS = {
    "matches": "[MATCH]",
    "mismatch": "[MISMATCH]",
    "uncertain": "[UNCERTAIN]",
    "insufficient_signal": "[INSUFFICIENT]",
    "unknown_language": "[UNKNOWN]",
}
STATUS_MARKERS = {"hit": "[HIT]", "approximate": "[APPROX]", "missed": "[MISS]"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio_file", help="Path to audio file (webm, wav, m4a, mp3, ogg — anything ffmpeg accepts)")
    ap.add_argument("claimed_language", help="Language to check against (e.g. english, spanish, mandarin, arabic)")
    ap.add_argument("--url", default="https://voiceprint.fly.dev", help="Voiceprint base URL (default: %(default)s)")
    ap.add_argument("--expected", help="Optional expected_language for stretch-phoneme scoring (mandarin or arabic)")
    ap.add_argument("--format", help="Override audio format string sent to server (defaults to file extension)")
    ap.add_argument("--raw", action="store_true", help="Print the full /analyze response JSON")
    ap.add_argument("--timeout", type=int, default=180, help="Request timeout in seconds (default: %(default)s)")
    ap.add_argument(
        "--api-key",
        default=os.environ.get("VOICEPRINT_API_KEY"),
        help="Send x-api-key header (defaults to VOICEPRINT_API_KEY env var)",
    )
    args = ap.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"error: file not found: {audio_path}", file=sys.stderr)
        return 2

    audio_bytes = audio_path.read_bytes()
    fmt = (args.format or audio_path.suffix.lstrip(".")).lower() or "webm"

    body: dict = {
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "format": fmt,
        "claimed_language": args.claimed_language,
    }
    if args.expected:
        body["expected_language"] = args.expected

    url = f"{args.url.rstrip('/')}/analyze"
    # In --raw mode, send progress lines to stderr so stdout stays pure JSON
    # and the script can be piped into jq / python -c safely.
    log_target = sys.stderr if args.raw else sys.stdout
    print(f"POST {url}", file=log_target)
    print(f"     audio={audio_path.name} ({len(audio_bytes):,} bytes, format={fmt})", file=log_target)
    print(f"     claimed_language={args.claimed_language!r}"
          + (f", expected_language={args.expected!r}" if args.expected else ""), file=log_target)
    print(file=log_target)

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["x-api-key"] = args.api_key
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:1000]
        print(f"HTTP {e.code} {e.reason}", file=sys.stderr)
        print(err_body, file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"connection error: {e.reason}", file=sys.stderr)
        print("hint: if this is the first call after deploy, the model is downloading "
              "(2-5 min). Try `curl https://voiceprint.fly.dev/health` first, then retry.",
              file=sys.stderr)
        return 1

    if args.raw:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    _print_summary(data, args.claimed_language)
    return 0


def _print_summary(data: dict, claimed: str) -> None:
    duration = data.get("duration_s") or 0.0
    phon = data.get("phonemes") or {}
    f0 = data.get("f0") or {}
    notes = data.get("notes") or []

    print(f"  duration={duration:.1f}s   phoneme_tokens={phon.get('total_tokens', 0)}   "
          f"voiced_fraction={f0.get('voiced_fraction', 0):.2f}")

    lm = data.get("language_match")
    print()
    if not lm:
        print(f"  language_match: <missing> — server didn't return a block. "
              f"Either claimed_language={claimed!r} wasn't sent, or the server hasn't "
              f"been redeployed with the new code.")
    else:
        marker = VERDICT_MARKERS.get(lm["verdict"], "[?]")
        score = lm.get("score")
        score_str = f"score={score:.2f}" if score is not None else "score=n/a"
        print(f"  language_match  {marker}  verdict={lm['verdict']}  {score_str}")
        print(f"                  positives: {lm['positive_hits']}/{lm['positive_total']} markers found")
        print(f"                  negatives: {lm['negative_clean']}/{lm['negative_total']} clean")
        for n in lm.get("notes", []):
            print(f"                  - {n}")

    sp = data.get("stretch_phonemes")
    if sp:
        print()
        print(f"  stretch_phonemes ({sp['expected_language']}):")
        for p in sp.get("probes", []):
            marker = STATUS_MARKERS.get(p["status"], "[?]")
            print(f"      {marker}  /{p['ipa']}/  {p['label']}: count={p['count']}, "
                  f"expected={p['expected_count']}, approximate_seen={p['approximate_count']}")

    if notes:
        print()
        for n in notes:
            print(f"  note: {n}")
    print()


if __name__ == "__main__":
    sys.exit(main())
