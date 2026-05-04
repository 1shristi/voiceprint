# RAVDESS ST calibration — ravdess_st_2026-05-03

**Sample:** 240 clips across 4 buckets.

## Per-bucket distributions (`f0.std_semitones`)

| Bucket | n | mean | median | P10 | P25 | P75 | P90 | min | max |
|---|---|---|---|---|---|---|---|---|---|
| neutral | 60 | 2.96 | 2.77 | 1.85 | 2.12 | 3.55 | 4.26 | 0.85 | 6.11 |
| calm-normal | 60 | 2.66 | 2.51 | 1.74 | 2.10 | 2.89 | 3.63 | 1.00 | 7.01 |
| happy-strong | 60 | 3.93 | 3.91 | 2.39 | 2.97 | 4.78 | 5.36 | 1.83 | 6.62 |
| angry-strong | 60 | 4.09 | 3.95 | 2.38 | 3.03 | 4.92 | 5.52 | 1.69 | 8.73 |

## Per-bucket distributions (`f0.range_semitones_p10_p90`)

| Bucket | n | mean | median | P25 | P75 |
|---|---|---|---|---|---|
| neutral | 60 | 7.03 | 6.76 | 5.42 | 8.29 |
| calm-normal | 60 | 7.10 | 6.69 | 5.32 | 7.82 |
| happy-strong | 60 | 9.11 | 8.69 | 7.02 | 11.13 |
| angry-strong | 60 | 9.26 | 8.45 | 6.70 | 10.72 |

## Cross-gender sanity check (anatomy-invariance)

ST is supposed to remove the male/female base-frequency bias in pitch dynamism. If anatomy-invariance holds, flat-leaning male and flat-leaning female speakers should land in roughly the same `std_semitones` range, and similarly for expressive.

| Group | Gender | n | mean std_st | median |
|---|---|---|---|---|
| flat-leaning | male | 60 | 2.84 | 2.60 |
| flat-leaning | female | 60 | 2.79 | 2.64 |
| expressive-leaning | male | 58 | 4.15 | 3.97 |
| expressive-leaning | female | 62 | 3.88 | 3.90 |

## Proposed ST cutoffs

- **flat:** `std_semitones < 2.80`
- **moderate:** `2.80 ≤ std_semitones ≤ 4.00`
- **expressive:** `std_semitones > 4.00`

_Distributions overlap at the tails (flat P75 = 3.25, expressive P25 = 3.03). Using group means as cutoffs gives a stable 3-way split. Borderline speakers near the cutoffs will straddle categories — that's inherent to the prosody-from-emotion data, not a bug._

Derivation: cutoffs sit at the group means. A speaker classified "flat" is more flat than the typical flat-leaning speaker (RAVDESS calm/neutral); a speaker classified "expressive" is more expressive than the typical expressive-leaning speaker (RAVDESS happy-strong/angry-strong). Anything in between is moderate. ~50% of each labelled group falls into the corresponding category — the rest falls into moderate, which is appropriate because RAVDESS emotion labels measure actor intent, not produced prosody on every clip.

## Comparison: Hz vs ST classification

Under current Hz cutoffs (<20.0 flat, ≤50.0 moderate, >50.0 expressive) vs proposed ST cutoffs, 84/240 clips would change category.

Per-bucket breakdown of Hz vs ST classifications:

| Bucket | n | Hz: flat / moderate / expressive | ST: flat / moderate / expressive |
|---|---|---|---|
| neutral | 60 | 20 / 30 / 10 | 32 / 20 / 8 |
| calm-normal | 60 | 26 / 31 / 3 | 40 / 16 / 4 |
| happy-strong | 60 | 0 / 22 / 38 | 12 / 19 / 29 |
| angry-strong | 60 | 0 / 14 / 46 | 10 / 22 / 28 |

## Recommendation

- ST classifies **60%** of flat-leaning clips (neutral + calm-normal) as flat.
- ST classifies **48%** of expressive-leaning clips (happy-strong + angry-strong) as expressive.

- Cross-tail noise: **10%** of flat-leaning clips classified expressive, **18%** of expressive-leaning clips classified flat.

**Cutoffs look defensible.** ~half of each labelled group lands in the expected category, and cross-tail misclassification is low (<25%). The middle category (moderate) catches the rest, which is the right behaviour for RAVDESS — emotion labels measure actor intent, not produced prosody on every clip. Suggest swapping section 3 of `validateAgainstFeatures` to use these ST cutoffs.
