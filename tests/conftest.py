"""Shared test config.

By default, phoneme extraction is disabled in tests so we don't download a
370 MB model on every CI run. Tests that exercise phoneme detection should
unset the env var inside the test (and be marked slow / opt-in).
"""

import os
import sys
from pathlib import Path


os.environ.setdefault("VOICEPRINT_DISABLE_PHONEMES", "1")

# Auto-detect espeak-ng on macOS so phonemizer can find it.
if sys.platform == "darwin" and "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
    candidates = [
        Path("/opt/homebrew/lib/libespeak-ng.dylib"),
        Path("/usr/local/lib/libespeak-ng.dylib"),
    ]
    for c in candidates:
        if c.exists():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(c)
            break
