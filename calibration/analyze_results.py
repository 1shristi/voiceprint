"""
Read a calibration CSV and emit a markdown report:
  - per-bucket summary stats for f0_std_semitones and f0_range_semitones_p10_p90
  - suggested ST cutoffs for the flat / moderate / expressive split
  - comparison vs current Hz cutoffs in linguamatch
  - notes on cross-gender behaviour (sanity-checks the anatomy-invariance claim)

Run: python analyze_results.py --csv results/ravdess_st_<date>.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

# Mapping of buckets to expected category labels. Used to interpret the
# distribution: "flat-leaning" buckets should sit lower in ST, "expressive-
# leaning" buckets should sit higher.
EXPECTED = {
    "neutral": "flat-leaning",
    "calm-normal": "flat-leaning",
    "happy-strong": "expressive-leaning",
    "angry-strong": "expressive-leaning",
}

# Current Hz cutoffs in linguamatch's analyze-voice/index.ts (section 3, pitch_variation).
HZ_CUT_LOW = 20.0   # < HZ_CUT_LOW → "flat"
HZ_CUT_HIGH = 50.0  # ≤ HZ_CUT_HIGH → "moderate"; > HZ_CUT_HIGH → "expressive"


def _f(x: str) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def percentile(values: list[float], p: float) -> float:
    """p in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "n/a"
    return f"{x:.2f}"


def summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def classify_hz(std_hz: float | None) -> str:
    if std_hz is None:
        return "n/a"
    if std_hz < HZ_CUT_LOW:
        return "flat"
    if std_hz <= HZ_CUT_HIGH:
        return "moderate"
    return "expressive"


def classify_st(std_st: float | None, low: float, high: float) -> str:
    if std_st is None:
        return "n/a"
    if std_st < low:
        return "flat"
    if std_st <= high:
        return "moderate"
    return "expressive"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    rows: list[dict] = []
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")

    # Group by bucket.
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r)

    # Aggregate by group: flat-leaning vs expressive-leaning.
    flat_std_st: list[float] = []
    expr_std_st: list[float] = []
    flat_range_st: list[float] = []
    expr_range_st: list[float] = []
    for bucket, brows in by_bucket.items():
        for r in brows:
            sst = _f(r["f0_std_semitones"])
            rst = _f(r["f0_range_semitones_p10_p90"])
            if EXPECTED.get(bucket) == "flat-leaning":
                if sst is not None: flat_std_st.append(sst)
                if rst is not None: flat_range_st.append(rst)
            elif EXPECTED.get(bucket) == "expressive-leaning":
                if sst is not None: expr_std_st.append(sst)
                if rst is not None: expr_range_st.append(rst)

    flat_summary = summarise(flat_std_st)
    expr_summary = summarise(expr_std_st)

    # Cutoff proposal: use group means as boundaries.
    #   flat:       std_st < mean(flat-leaning)
    #   moderate:   mean(flat-leaning) ≤ std_st ≤ mean(expressive-leaning)
    #   expressive: std_st > mean(expressive-leaning)
    # Why means rather than P75/P25: P75/P25 can collapse to a single cutoff when the
    # two groups overlap (which is expected here — RAVDESS labels emotion intent, not
    # produced prosody, so an actor reading "calm" can still produce moderate variation).
    # Means give a stable three-way split that maps "below the typical flat speaker" /
    # "between the two typicals" / "above the typical expressive speaker."
    low_cut = flat_summary.get("mean")
    high_cut = expr_summary.get("mean")
    overlap_p75 = flat_summary.get("p75")
    overlap_p25 = expr_summary.get("p25")
    if overlap_p75 is not None and overlap_p25 is not None and overlap_p75 > overlap_p25:
        cutoff_note = (
            f"Distributions overlap at the tails (flat P75 = {overlap_p75:.2f}, "
            f"expressive P25 = {overlap_p25:.2f}). Using group means as cutoffs gives a "
            f"stable 3-way split. Borderline speakers near the cutoffs will straddle "
            f"categories — that's inherent to the prosody-from-emotion data, not a bug."
        )
    else:
        cutoff_note = (
            f"Distributions separate cleanly (flat P75 = {fmt(overlap_p75)}, "
            f"expressive P25 = {fmt(overlap_p25)}). Group-mean cutoffs proposed below."
        )

    # Round cutoffs to one decimal for cleaner code constants.
    if low_cut is not None: low_cut = round(low_cut, 1)
    if high_cut is not None: high_cut = round(high_cut, 1)

    # Reclassify each row under proposed ST cutoffs and compare to Hz classification.
    reclassified: list[tuple[str, str, str, str]] = []  # (bucket, hz_class, st_class, agree)
    n_changed = 0
    for r in rows:
        std_hz = _f(r["f0_std_hz"])
        std_st = _f(r["f0_std_semitones"])
        hz_class = classify_hz(std_hz)
        st_class = classify_st(std_st, low_cut or 0, high_cut or 0)
        if hz_class != st_class and hz_class != "n/a" and st_class != "n/a":
            n_changed += 1
        reclassified.append((r["bucket"], hz_class, st_class, "yes" if hz_class == st_class else "no"))

    # Per-bucket Hz vs ST classification breakdown.
    bucket_breakdown: dict[str, dict] = {}
    for bucket, brows in by_bucket.items():
        hz_counts: dict[str, int] = defaultdict(int)
        st_counts: dict[str, int] = defaultdict(int)
        for r in brows:
            hz_counts[classify_hz(_f(r["f0_std_hz"]))] += 1
            st_counts[classify_st(_f(r["f0_std_semitones"]), low_cut or 0, high_cut or 0)] += 1
        bucket_breakdown[bucket] = {"hz": dict(hz_counts), "st": dict(st_counts), "n": len(brows)}

    # Gender-anatomy sanity check: flat-leaning + expressive-leaning, separated by gender.
    by_gender_std: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        sst = _f(r["f0_std_semitones"])
        if sst is None:
            continue
        group = EXPECTED.get(r["bucket"])
        if group:
            by_gender_std[(group, r["gender"])].append(sst)

    # Build the markdown report.
    out = csv_path.with_suffix("").as_posix() + "_report.md"
    with open(out, "w") as f:
        w = f.write

        w(f"# RAVDESS ST calibration — {csv_path.stem}\n\n")
        w(f"**Sample:** {len(rows)} clips across {len(by_bucket)} buckets.\n\n")

        w("## Per-bucket distributions (`f0.std_semitones`)\n\n")
        w("| Bucket | n | mean | median | P10 | P25 | P75 | P90 | min | max |\n")
        w("|---|---|---|---|---|---|---|---|---|---|\n")
        for bucket in ["neutral", "calm-normal", "happy-strong", "angry-strong"]:
            if bucket not in by_bucket:
                continue
            vals = [v for v in (_f(r["f0_std_semitones"]) for r in by_bucket[bucket]) if v is not None]
            s = summarise(vals)
            w(f"| {bucket} | {s['n']} | {fmt(s.get('mean'))} | {fmt(s.get('median'))} | "
              f"{fmt(s.get('p10'))} | {fmt(s.get('p25'))} | {fmt(s.get('p75'))} | "
              f"{fmt(s.get('p90'))} | {fmt(s.get('min'))} | {fmt(s.get('max'))} |\n")

        w("\n## Per-bucket distributions (`f0.range_semitones_p10_p90`)\n\n")
        w("| Bucket | n | mean | median | P25 | P75 |\n")
        w("|---|---|---|---|---|---|\n")
        for bucket in ["neutral", "calm-normal", "happy-strong", "angry-strong"]:
            if bucket not in by_bucket:
                continue
            vals = [v for v in (_f(r["f0_range_semitones_p10_p90"]) for r in by_bucket[bucket]) if v is not None]
            s = summarise(vals)
            w(f"| {bucket} | {s['n']} | {fmt(s.get('mean'))} | {fmt(s.get('median'))} | "
              f"{fmt(s.get('p25'))} | {fmt(s.get('p75'))} |\n")

        w("\n## Cross-gender sanity check (anatomy-invariance)\n\n")
        w("ST is supposed to remove the male/female base-frequency bias in pitch dynamism. "
          "If anatomy-invariance holds, flat-leaning male and flat-leaning female speakers "
          "should land in roughly the same `std_semitones` range, and similarly for expressive.\n\n")
        w("| Group | Gender | n | mean std_st | median |\n")
        w("|---|---|---|---|---|\n")
        for group in ["flat-leaning", "expressive-leaning"]:
            for gender in ["male", "female"]:
                vals = by_gender_std.get((group, gender), [])
                s = summarise(vals)
                w(f"| {group} | {gender} | {s['n']} | {fmt(s.get('mean'))} | {fmt(s.get('median'))} |\n")

        w("\n## Proposed ST cutoffs\n\n")
        w(f"- **flat:** `std_semitones < {fmt(low_cut)}`\n")
        w(f"- **moderate:** `{fmt(low_cut)} ≤ std_semitones ≤ {fmt(high_cut)}`\n")
        w(f"- **expressive:** `std_semitones > {fmt(high_cut)}`\n\n")
        w(f"_{cutoff_note}_\n\n")

        w("Derivation: cutoffs sit at the group means. A speaker classified \"flat\" is "
          "more flat than the typical flat-leaning speaker (RAVDESS calm/neutral); a "
          "speaker classified \"expressive\" is more expressive than the typical "
          "expressive-leaning speaker (RAVDESS happy-strong/angry-strong). Anything in "
          "between is moderate. ~50% of each labelled group falls into the corresponding "
          "category — the rest falls into moderate, which is appropriate because RAVDESS "
          "emotion labels measure actor intent, not produced prosody on every clip.\n\n")

        w("## Comparison: Hz vs ST classification\n\n")
        w(f"Under current Hz cutoffs (<{HZ_CUT_LOW} flat, ≤{HZ_CUT_HIGH} moderate, >{HZ_CUT_HIGH} expressive) "
          f"vs proposed ST cutoffs, {n_changed}/{len(rows)} clips would change category.\n\n")
        w("Per-bucket breakdown of Hz vs ST classifications:\n\n")
        w("| Bucket | n | Hz: flat / moderate / expressive | ST: flat / moderate / expressive |\n")
        w("|---|---|---|---|\n")
        for bucket in ["neutral", "calm-normal", "happy-strong", "angry-strong"]:
            if bucket not in bucket_breakdown:
                continue
            b = bucket_breakdown[bucket]
            hz = b["hz"]
            st = b["st"]
            hz_str = f"{hz.get('flat', 0)} / {hz.get('moderate', 0)} / {hz.get('expressive', 0)}"
            st_str = f"{st.get('flat', 0)} / {st.get('moderate', 0)} / {st.get('expressive', 0)}"
            w(f"| {bucket} | {b['n']} | {hz_str} | {st_str} |\n")

        w("\n## Recommendation\n\n")
        flat_st_count = sum(b["st"].get("flat", 0) for k, b in bucket_breakdown.items() if EXPECTED.get(k) == "flat-leaning")
        flat_st_total = sum(b["n"] for k, b in bucket_breakdown.items() if EXPECTED.get(k) == "flat-leaning")
        expr_st_count = sum(b["st"].get("expressive", 0) for k, b in bucket_breakdown.items() if EXPECTED.get(k) == "expressive-leaning")
        expr_st_total = sum(b["n"] for k, b in bucket_breakdown.items() if EXPECTED.get(k) == "expressive-leaning")

        flat_st_pct = (flat_st_count / flat_st_total * 100) if flat_st_total else 0
        expr_st_pct = (expr_st_count / expr_st_total * 100) if expr_st_total else 0

        w(f"- ST classifies **{flat_st_pct:.0f}%** of flat-leaning clips (neutral + calm-normal) as flat.\n")
        w(f"- ST classifies **{expr_st_pct:.0f}%** of expressive-leaning clips (happy-strong + angry-strong) as expressive.\n\n")

        # Cross-tail check: a clip should rarely land in the OPPOSITE category.
        flat_at_expr = sum(b["st"].get("expressive", 0) for k, b in bucket_breakdown.items() if EXPECTED.get(k) == "flat-leaning")
        expr_at_flat = sum(b["st"].get("flat", 0) for k, b in bucket_breakdown.items() if EXPECTED.get(k) == "expressive-leaning")
        flat_cross_pct = (flat_at_expr / flat_st_total * 100) if flat_st_total else 0
        expr_cross_pct = (expr_at_flat / expr_st_total * 100) if expr_st_total else 0
        w(f"- Cross-tail noise: **{flat_cross_pct:.0f}%** of flat-leaning clips classified expressive, "
          f"**{expr_cross_pct:.0f}%** of expressive-leaning clips classified flat.\n\n")

        if flat_st_pct >= 40 and expr_st_pct >= 40 and flat_cross_pct < 25 and expr_cross_pct < 25:
            verdict = (
                "**Cutoffs look defensible.** ~half of each labelled group lands in the expected "
                "category, and cross-tail misclassification is low (<25%). The middle category "
                "(moderate) catches the rest, which is the right behaviour for RAVDESS — emotion "
                "labels measure actor intent, not produced prosody on every clip. Suggest swapping "
                "section 3 of `validateAgainstFeatures` to use these ST cutoffs."
            )
        else:
            verdict = (
                "**Cutoffs are weak.** Either the in-target rate is below 40% or cross-tail noise "
                "is above 25%. Don't ship without investigating: likely needs a larger sample, "
                "different bucket choices, or a non-categorical output (raw ST values)."
            )
        w(verdict + "\n")

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
