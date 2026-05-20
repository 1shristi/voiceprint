"""Tests for confusion-matrix smoothing (Loop 3a Phase 2).

Smoothing is pure-function (raw_counts dict → smoothed dict + evidence dict);
none of these tests need the Wav2Vec2 model. The fixture-flavoured "speaker
clip with high /s/ and /f/ but zero /θ/" test in §6.4 lives at the bottom and
double-checks the canonical example from the spec.
"""

from __future__ import annotations

import math

import pytest

from app.services import smoothing


# ─── Math correctness ──────────────────────────────────────────────────────


def test_smoothing_canonical_theta_example_from_spec() -> None:
    """Spec §6.3 worked example: raw {s: 18, f: 12, d: 24}, weights 0.4 / 0.3
    on /s/ and /f/ for /θ/ → smoothed_count[θ] = 0.4×18 + 0.3×12 = 10.8.
    """
    result = smoothing.compute_smoothed_counts({"s": 18, "f": 12, "d": 24})
    assert math.isclose(result.smoothed_counts["θ"], 10.8, abs_tol=1e-9)
    assert math.isclose(result.smoothed_counts["ð"], 0.4 * 24, abs_tol=1e-9)


def test_smoothing_contributes_only_when_near_miss_actually_appeared() -> None:
    # Only /s/ fires; /f/ count is zero → only one contribution row, value
    # equal to weight × raw_s.
    result = smoothing.compute_smoothed_counts({"s": 5})
    evidence = result.confusion_evidence["θ"]
    assert len(evidence.evidence_from) == 1
    assert evidence.evidence_from[0].from_symbol == "s"
    assert math.isclose(evidence.evidence_from[0].contribution, 0.4 * 5)
    assert math.isclose(evidence.smoothed_count, 0.4 * 5)


def test_smoothing_adds_direct_count_to_smoothed_contributions() -> None:
    # Raw /θ/=2, raw /s/=10 → smoothed = 2 + 0.4×10 = 6.0
    result = smoothing.compute_smoothed_counts({"θ": 2, "s": 10})
    assert math.isclose(result.smoothed_counts["θ"], 6.0)
    assert result.confusion_evidence["θ"].raw_count == 2


def test_smoothing_accumulates_across_multiple_near_misses() -> None:
    # All three /θ/ near-misses fire (s, f, t).
    # smoothed = 0.4×10 + 0.3×8 + 0.15×4 = 4.0 + 2.4 + 0.6 = 7.0
    result = smoothing.compute_smoothed_counts({"s": 10, "f": 8, "t": 4})
    assert math.isclose(result.smoothed_counts["θ"], 7.0, abs_tol=1e-9)
    assert len(result.confusion_evidence["θ"].evidence_from) == 3


def test_smoothing_does_not_mutate_input() -> None:
    raw = {"s": 18, "f": 12, "d": 24, "θ": 0}
    original = dict(raw)
    smoothing.compute_smoothed_counts(raw)
    assert raw == original, "compute_smoothed_counts mutated its input dict"


def test_raw_counts_passthrough_for_non_target_symbols() -> None:
    # /s/, /f/, /d/ are not themselves confusion targets — their smoothed
    # value should equal their raw value.
    result = smoothing.compute_smoothed_counts({"s": 18, "f": 12, "d": 24})
    assert result.smoothed_counts["s"] == 18.0
    assert result.smoothed_counts["f"] == 12.0
    assert result.smoothed_counts["d"] == 24.0


def test_target_absent_from_matrix_returns_raw_value_unchanged() -> None:
    # /m/ is not a confusion target — smoothed[m] should equal raw[m] with
    # no evidence entry.
    result = smoothing.compute_smoothed_counts({"m": 7, "a": 3})
    assert result.smoothed_counts["m"] == 7.0
    assert "m" not in result.confusion_evidence


def test_empty_raw_counts_returns_empty_smoothed_and_evidence() -> None:
    result = smoothing.compute_smoothed_counts({})
    assert result.smoothed_counts == {}
    assert result.confusion_evidence == {}


def test_zero_value_targets_are_omitted_from_smoothed_counts() -> None:
    """If a target has raw_count=0 AND no near-miss contributors fire, the
    smoothed value would be 0; we omit it to keep the response lean."""
    # Only /m/ in input — no /s/, /f/, /t/ to contribute to /θ/, no /d/, /v/,
    # /z/ to contribute to /ð/, etc.
    result = smoothing.compute_smoothed_counts({"m": 5})
    assert "θ" not in result.smoothed_counts
    assert "ð" not in result.smoothed_counts
    # /m/ itself is passed through as raw=5
    assert result.smoothed_counts["m"] == 5.0


# ─── eSpeak / IPA normalisation ────────────────────────────────────────────


def test_xsampa_holdouts_normalise_to_ipa_before_smoothing() -> None:
    """Wav2Vec2 vocab has both `S` (X-SAMPA) and `ʃ` (direct IPA) for the
    postalveolar fricative. Smoothing collapses both onto `ʃ` so a /tʃ/
    contribution still picks up the evidence regardless of which form the
    model emitted on this clip."""
    # Confusion matrix has /tʃ/ → contribution from /ʃ/ (weight 0.3).
    # Emit only the X-SAMPA holdout "S" — the smoother should normalise to
    # ʃ and apply the weight.
    result = smoothing.compute_smoothed_counts({"S": 10})
    tʃ_evidence = result.confusion_evidence["tʃ"]
    # ʃ should be the from_symbol, raw_count 10 from the normalised "S".
    contributors = {c.from_symbol: c for c in tʃ_evidence.evidence_from}
    assert "ʃ" in contributors
    assert contributors["ʃ"].raw_count == 10
    assert math.isclose(contributors["ʃ"].contribution, 0.3 * 10)


def test_mixed_xsampa_and_ipa_emissions_sum_under_ipa_key() -> None:
    # Some frames emit "S", others emit "ʃ" — both should count.
    result = smoothing.compute_smoothed_counts({"S": 4, "ʃ": 6})
    tʃ_evidence = result.confusion_evidence["tʃ"]
    contributors = {c.from_symbol: c for c in tʃ_evidence.evidence_from}
    assert contributors["ʃ"].raw_count == 10  # 4 + 6
    assert math.isclose(contributors["ʃ"].contribution, 0.3 * 10)


# ─── Interpretation strings ────────────────────────────────────────────────


def test_interpretation_for_zero_raw_with_evidence_says_misclassifies() -> None:
    result = smoothing.compute_smoothed_counts({"s": 18, "f": 12})
    interp = result.confusion_evidence["θ"].interpretation
    assert "Direct detection: 0" in interp
    assert "may produce" in interp.lower()
    assert "/θ/" in interp


def test_interpretation_for_direct_hit_plus_contributors_combines_them() -> None:
    result = smoothing.compute_smoothed_counts({"θ": 3, "s": 5})
    interp = result.confusion_evidence["θ"].interpretation
    assert "Direct detection: 3" in interp


def test_interpretation_for_direct_only_no_contributors() -> None:
    result = smoothing.compute_smoothed_counts({"θ": 4})
    interp = result.confusion_evidence["θ"].interpretation
    assert "Direct detection: 4" in interp
    assert "No additional smoothed evidence" in interp


# ─── §6.4 fixture: high /s/ + /f/ but zero /θ/ → non-zero smoothed_count[θ] ──


def test_spec_fixture_high_s_and_f_zero_theta_yields_nonzero_smoothed_theta() -> None:
    """Spec §6.4 acceptance fixture: a clip with high /s/ and /f/ counts but
    zero direct /θ/ must produce a non-zero smoothed_count for /θ/.
    """
    raw_counts = {
        "s": 18,
        "f": 12,
        "θ": 0,  # explicit zero
        "a": 30,
        "e": 22,
    }
    result = smoothing.compute_smoothed_counts(raw_counts)
    # Smoothed θ should be > 0 because /s/ and /f/ fired.
    assert "θ" in result.smoothed_counts
    assert result.smoothed_counts["θ"] > 0
    # Should also match the canonical-math value (0.4×18 + 0.3×12 = 10.8).
    assert math.isclose(result.smoothed_counts["θ"], 10.8, abs_tol=1e-9)
    # Evidence trail should attribute the smoothed value to /s/ and /f/.
    contributors = {
        c.from_symbol: c
        for c in result.confusion_evidence["θ"].evidence_from
    }
    assert set(contributors.keys()) == {"s", "f"}
    assert contributors["s"].raw_count == 18
    assert contributors["f"].raw_count == 12
    assert math.isclose(contributors["s"].contribution, 0.4 * 18)
    assert math.isclose(contributors["f"].contribution, 0.3 * 12)


# ─── Confusion matrix sanity ───────────────────────────────────────────────


def test_confusion_matrix_target_count_in_spec_range() -> None:
    """Spec §6.1 asks for ~30–50 confusion targets. Failing this is a
    regression signal that someone trimmed the matrix below the floor."""
    matrix = smoothing._load_confusion_matrix()
    n_targets = len(matrix["confusions"])
    assert 30 <= n_targets <= 50, f"expected 30–50 targets, got {n_targets}"


def test_confusion_matrix_weights_in_unit_interval() -> None:
    """Defensive: any weight above 1.0 would let smoothed_count exceed the
    sum of contributing raw counts; weights below 0 would subtract evidence."""
    matrix = smoothing._load_confusion_matrix()
    for target, data in matrix["confusions"].items():
        for nm in data["near_misses"]:
            w = nm["weight"]
            assert 0 < w <= 1.0, (
                f"matrix[{target}] near-miss /{nm['from_symbol']}/ has "
                f"weight={w} outside (0, 1.0]"
            )


def test_confusion_matrix_every_entry_has_source() -> None:
    """Every near-miss must cite a source per the spec's provenance discipline."""
    matrix = smoothing._load_confusion_matrix()
    for target, data in matrix["confusions"].items():
        for nm in data["near_misses"]:
            assert nm.get("source"), (
                f"matrix[{target}] near-miss /{nm['from_symbol']}/ missing source"
            )


# ─── Route integration ─────────────────────────────────────────────────────


def test_route_response_includes_smoothed_counts_and_confusion_evidence() -> None:
    """End-to-end check: /analyze response carries the new fields. Uses the
    synthetic-sine audio fixture and stubs the phoneme model so the route
    actually populates smoothed_counts/confusion_evidence.
    """
    import base64
    import tempfile
    from pathlib import Path

    import numpy as np
    import parselmouth
    from fastapi.testclient import TestClient

    from app.main import app
    from app.services import phonemes

    SAMPLE_RATE = 16_000
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    samples = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    sound = parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    try:
        sound.save(str(path), "WAV")
        audio_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    finally:
        path.unlink(missing_ok=True)

    # Stub the phoneme model to return a known counts dict.
    def fake_extract(s, *, expected_language=None, include_alignment_artifacts=False):
        return phonemes.PhonemeInventory(
            counts={"s": 18, "f": 12, "d": 24, "a": 30},
            total_tokens=84,
            occurrences=[],
        )

    import pytest as _pytest

    monkey = _pytest.MonkeyPatch()
    monkey.setattr(phonemes, "extract_phonemes", fake_extract)
    monkey.delenv("VOICEPRINT_DISABLE_PHONEMES", raising=False)
    try:
        client = TestClient(app)
        res = client.post(
            "/analyze",
            json={"audio_base64": audio_b64, "format": "wav"},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        # Raw counts unchanged from prior behaviour.
        assert body["phonemes"]["counts"] == {"s": 18, "f": 12, "d": 24, "a": 30}
        # Smoothed view emits /θ/ and /ð/ as non-zero due to confusion evidence.
        assert body["phonemes"]["smoothed_counts"].get("θ") is not None
        assert body["phonemes"]["smoothed_counts"]["θ"] > 0
        # Evidence dict carries the audit trail.
        theta_ev = body["phonemes"]["confusion_evidence"]["θ"]
        assert theta_ev["raw_count"] == 0
        assert theta_ev["smoothed_count"] > 0
        assert len(theta_ev["evidence_from"]) >= 1
    finally:
        monkey.undo()


def test_route_response_smoothed_fields_empty_when_phoneme_model_disabled() -> None:
    """With VOICEPRINT_DISABLE_PHONEMES=1 (the default test environment), raw
    counts come back empty; smoothing has nothing to operate on; both new
    fields should default to {} rather than crash."""
    import base64
    import tempfile
    from pathlib import Path

    import numpy as np
    import parselmouth
    from fastapi.testclient import TestClient

    from app.main import app

    SAMPLE_RATE = 16_000
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    samples = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    sound = parselmouth.Sound(samples, sampling_frequency=SAMPLE_RATE)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    try:
        sound.save(str(path), "WAV")
        audio_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    finally:
        path.unlink(missing_ok=True)

    client = TestClient(app)
    res = client.post(
        "/analyze",
        json={"audio_base64": audio_b64, "format": "wav"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["phonemes"]["counts"] == {}
    assert body["phonemes"]["smoothed_counts"] == {}
    assert body["phonemes"]["confusion_evidence"] == {}
